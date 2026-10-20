"""Covariance-aware centroid estimation for repeated ground-target detections.

Points use (N, D), covariances use (N, D, D). In the mission pipeline D=2
and coordinates are north/east metres. Validation lives in :mod:`check`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2, norm

from . import check


@dataclass(frozen=True, eq=False)
class Data:
    """Detection positions and positive-definite covariance matrices.

    Inputs are copied and read-only, keeping cached inverses consistent.
    A single point/covariance is accepted; empty batches retain their dimensions.
    """

    points: ArrayLike
    covariances: ArrayLike
    covs_inv: NDArray[np.float64] = field(init=False, repr=False)
    scale: NDArray[np.float64] = field(init=False, repr=False)
    scale_inv: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float64, copy=True, ndmin=2)
        covariance = np.array(self.covariances, dtype=np.float64, copy=True)
        if covariance.ndim == 2:
            covariance = covariance[None, :, :]
        check.clustering_data(points, covariance)
        arrays = {
            "points": points,
            "covariances": covariance,
            "covs_inv": np.linalg.inv(covariance),
            "scale": np.linalg.cholesky(covariance),
        }
        arrays["scale_inv"] = np.linalg.inv(arrays["scale"])
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def covs(self) -> NDArray[np.float64]:
        """Covariance alias retained for the decision pipeline."""
        return self.covariances

    @property
    def n_samples(self) -> int:
        return self.points.shape[0]

    @property
    def n_dimensions(self) -> int:
        return self.points.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> Data:
        return Data(self.points[index], self.covariances[index])

    def append(self, data: Data) -> Data:
        check.clustering_dimensions(data.n_dimensions, self.n_dimensions)
        return Data(
            np.concatenate([self.points, data.points]),
            np.concatenate([self.covariances, data.covariances]),
        )


def _distances(data: Data, centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """Squared Mahalanobis distances, using each detection's uncertainty."""
    differences = data.points[:, None, :] - centers[None, :, :]
    return np.einsum("nci,nij,ncj->nc", differences, data.covs_inv, differences)


def _weighted_centers(data: Data, neighbors: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Solve each neighborhood's inverse-covariance weighted least squares."""
    precision = np.einsum("nij,nc->cij", data.covs_inv, neighbors)
    position = np.einsum("nij,nj,nc->ci", data.covs_inv, data.points, neighbors)
    return np.linalg.solve(precision, position[..., None])[..., 0]


class CentroidAnalyzer:
    """Refine detection seeds and retain sufficiently dense, consistent centers.

    ``horizon`` sets a chi-square Mahalanobis radius. ``filter`` sets the
    minimum neighborhood population relative to the densest neighborhood.
    Outputs are ``centroids`` (C,D) and ``probs`` (C,), as in the decision API.
    """

    def __init__(self, horizon: float, filter: float) -> None:
        check.centroid_options(horizon, filter)
        self.horizon = float(horizon)
        self.filter = float(filter)
        self.n_dimensions = 0
        self.centroids = np.empty((0, 0), dtype=np.float64)
        self.probs = np.empty(0, dtype=np.float64)

    @property
    def radius(self) -> float:
        return float(chi2.ppf(self.horizon, df=self.n_dimensions))

    def __call__(
        self,
        data: Data,
        batch_size: int,
        passages: int,
        feedback: ArrayLike | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        check.positive_integer(batch_size, name="batch_size")
        check.positive_integer(passages, name="passages")
        self.n_dimensions = data.n_dimensions
        centers = np.empty((0, data.n_dimensions), dtype=np.float64)
        # The default honors np.random.seed() used by existing mission tests.
        random = np.random if rng is None else rng
        if feedback is not None:
            feedback = np.asarray(feedback, dtype=np.float64)
            check.centroid_feedback(feedback, data.n_dimensions)
            count = min(len(feedback) * 2 // 3, batch_size)
            centers = feedback[random.choice(len(feedback), count, replace=False)]
        count = min(batch_size - len(centers), data.n_samples)
        seeds = data.points[random.choice(data.n_samples, count, replace=False)]
        centers = np.concatenate([centers, seeds])
        self.probs = np.empty(0, dtype=np.float64)
        if data.n_samples == 0:
            self.centroids = np.empty((0, data.n_dimensions), dtype=np.float64)
            return

        for _ in range(passages):
            neighbors = _distances(data, centers) <= self.radius
            populations = neighbors.sum(axis=0)
            keep = (populations > 0) & (populations >= self.filter * populations.max())
            neighbors = neighbors[:, keep]
            if not np.any(keep):
                self.centroids = np.empty((0, data.n_dimensions), dtype=np.float64)
                return
            centers = _weighted_centers(data, neighbors)

        # Score the final weighted centers against the last selected neighborhoods.
        total_distance = (neighbors * _distances(data, centers)).sum(axis=0)
        sqrt_population = np.sqrt(neighbors.sum(axis=0))
        scores = (
            total_distance / sqrt_population
            - chi2.mean(data.n_dimensions) * sqrt_population
        ) / np.sqrt(chi2.var(data.n_dimensions))
        probabilities = norm.sf(scores)
        keep = probabilities >= probabilities.max() * 0.2
        self.centroids = centers[keep]
        self.probs = probabilities[keep]


def plot(
    data: Data,
    selected: ArrayLike,
    centroids: ArrayLike,
    covs_p: float = 0.80,
) -> None:
    """Plot detections and their covariance ellipses; load matplotlib on demand."""
    check.clustering_plot(data.n_dimensions, covs_p)
    selected = np.asarray(selected, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    check.centroid_feedback(selected, 2)
    check.centroid_feedback(centroids, 2)
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots()
    confidence_scale = chi2.ppf(covs_p, df=2)
    for point, covariance in zip(data.points, data.covs):
        values, axes = np.linalg.eigh(covariance)
        angle = np.rad2deg(np.arctan2(axes[1, 1], axes[0, 1]))
        ax.add_patch(
            Ellipse(
                point,
                width=2 * np.sqrt(values[1] * confidence_scale),
                height=2 * np.sqrt(values[0] * confidence_scale),
                angle=angle,
                color="gray",
                alpha=0.15,
            )
        )
    ax.scatter(*data.points.T, s=3, label="Detections")
    ax.scatter(*centroids.T, marker="x", color="red", label="Candidates")
    ax.scatter(*selected.T, marker="x", color="green", label="Selected")
    ax.set_aspect("equal")
    ax.legend()
    fig.show()
