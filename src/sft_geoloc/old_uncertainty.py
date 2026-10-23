"""Monte Carlo geolocation with clipped Gaussian telemetry errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from . import check
from .geolocator import Geolocator


@dataclass(frozen=True)
class GeolocationEstimate:
    """Mission-map means (N, 2) and covariances (N, 2, 2), in metres/metres².

    All samples must hit the surface; otherwise mean/covariance are NaN.
    """

    mean_ne_m: NDArray[np.float64]
    covariance_ne_m2: NDArray[np.float64]
    valid: NDArray[np.bool_]
    valid_samples: NDArray[np.int64]


def estimate_from_samples(
    geolocator: Geolocator,
    pixels_px: ArrayLike,
    observer_latlon_deg: ArrayLike,
    altitude_m: ArrayLike,
    rpy_deg: ArrayLike,
) -> GeolocationEstimate:
    """Estimate N targets from S>=2 poses, using sample covariance (S-1).

    Shapes: pixels (N, 2), observers (S, 2), altitude (S,), attitude (S, 3).
    Missed or below-surface samples invalidate a target's entire distribution.
    """
    pixels = np.asarray(pixels_px, dtype=np.float64)
    positions = np.asarray(observer_latlon_deg, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    rpy = np.asarray(rpy_deg, dtype=np.float64)
    check.pixels_shape(pixels)
    check.altitude_samples(altitude)
    count = len(altitude)
    check.sample_shapes(positions, rpy, count)
    check.finite_samples(pixels, positions, altitude, rpy)
    points, hit = geolocator.pixels_to_ne(
        pixels[:, None, :], positions, np.maximum(altitude, 0.0), rpy
    )
    hit &= altitude >= 0
    valid = np.all(hit, axis=1)
    mean = np.full((len(pixels), 2), np.nan)
    covariance = np.full((len(pixels), 2, 2), np.nan)
    mean[valid] = points[valid].mean(axis=1)
    deviations = points[valid] - mean[valid, None, :]
    covariance[valid] = np.einsum("nsi,nsj->nij", deviations, deviations) / (count - 1)
    return GeolocationEstimate(mean, covariance, valid, hit.sum(axis=1, dtype=np.int64))


@dataclass(frozen=True)
class UncertaintyModel:
    """Confidence radii in metres/degrees, converted to Gaussian scales.

    Position and roll/pitch clip radially; altitude and yaw clip per axis.
    """

    samples: int
    confidence_percent: float
    position_error_m: float
    position_max_error_m: float
    altitude_error_m: float
    altitude_max_error_m: float
    roll_pitch_error_deg: float
    roll_pitch_max_error_deg: float
    yaw_error_deg: float
    yaw_max_error_deg: float

    def __post_init__(self) -> None:
        check.sample_count(self.samples)
        check.confidence_percent(self.confidence_percent)
        for name, value in vars(self).items():
            check.finite_nonnegative(value, name=name)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> UncertaintyModel:
        """Read the existing ``decision_system.geoloc`` subsection."""
        return cls(
            samples=config["monte_carlo"],
            confidence_percent=config["p_value"],
            position_error_m=config["ne_error"],
            position_max_error_m=config["ne_maxerror"],
            altitude_error_m=config["altitude_error"],
            altitude_max_error_m=config["altitude_maxerror"],
            roll_pitch_error_deg=config["rp_error"],
            roll_pitch_max_error_deg=config["rp_maxerror"],
            yaw_error_deg=config["yaw_error"],
            yaw_max_error_deg=config["yaw_maxerror"],
        )

    def _noise(
        self, rng: np.random.Generator, error: float, maximum: float, dimensions: int
    ) -> NDArray[np.float64]:
        scale = error / np.sqrt(chi2.ppf(self.confidence_percent / 100.0, dimensions))
        values = rng.normal(0.0, scale, (self.samples, dimensions))
        if dimensions == 1:
            return np.clip(values[:, 0], -maximum, maximum)
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        factors = np.ones_like(norms)
        np.divide(maximum, norms, out=factors, where=norms > maximum)
        return values * factors

    def estimate(
        self,
        geolocator: Geolocator,
        pixels_px: ArrayLike,
        observer_latlon_deg: ArrayLike,
        altitude_m: float,
        rpy_deg: ArrayLike,
        *,
        rng: np.random.Generator | None = None,
    ) -> GeolocationEstimate:
        """Perturb one exposure's telemetry and estimate all its target positions."""
        check.estimate_pose(observer_latlon_deg, rpy_deg)
        check.estimate_altitude(altitude_m)
        rng = np.random.default_rng() if rng is None else rng
        position_noise = self._noise(
            rng, self.position_error_m, self.position_max_error_m, 2
        )
        altitude_noise = self._noise(
            rng, self.altitude_error_m, self.altitude_max_error_m, 1
        )
        rp_noise = self._noise(
            rng, self.roll_pitch_error_deg, self.roll_pitch_max_error_deg, 2
        )
        yaw_noise = self._noise(rng, self.yaw_error_deg, self.yaw_max_error_deg, 1)
        observer_ne = geolocator.mission_map.gps_to_ne(observer_latlon_deg)
        positions = geolocator.mission_map.ne_to_gps(observer_ne + position_noise)
        rpy_samples = np.broadcast_to(
            np.asarray(rpy_deg, dtype=np.float64), (self.samples, 3)
        ).copy()
        rpy_samples[:, :2] += rp_noise
        rpy_samples[:, 2] += yaw_noise
        return estimate_from_samples(
            geolocator, pixels_px, positions, altitude_m + altitude_noise, rpy_samples
        )
