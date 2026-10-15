from sft_math.typing import Float, Array
import numpy.typing as np_typing
import numpy as np
from sft_math.class_templates import ClassBase


class Data(ClassBase):
    """
    Abstract base class for data handling in clustering algorithms.
    """

    points: Array
    covs_inv: Array
    covs: Array
    scale: Array
    scale_inv: Array

    __constants__ = [
        (lambda x: x, "n_dimensions", ""),
        (lambda x: x, "n_samples", ""),
    ]

    @property
    def n_dimensions(self) -> int:
        return self.points.shape[1]

    @property
    def n_samples(self) -> int:
        return self.points.shape[0]

    def __init__(
        self,
        points: np_typing.ArrayLike,
        covariances: np_typing.ArrayLike,
    ):
        # save points
        np_points = np.array(np.array(points, dtype=Float), copy=False, ndmin=2)
        assert (
            np_points.ndim == 2
        ), f"Points must be a 2D array, but got {np_points.ndim}D"
        n_samples, n_dimensions = np_points.shape
        assert n_dimensions > 0, "Points must have at least one dimension"

        # save covariances
        np_covariances = np.array(
            np.array(covariances, dtype=Float), copy=False, ndmin=3
        )
        # controllo che ogni matrice di covarianza sia quadrata, simmetrica e definita positiva
        assert (
            np_covariances.ndim == 3
        ), f"Covariances must be a 3D array, but got {np_covariances.ndim}D"
        assert np_covariances.shape[0] == n_samples, (
            f"Covariances must have the same number of samples as points, "
            f"but got {np_covariances.shape[0]} samples and {n_samples} points"
        )
        assert np_covariances.shape[1] == n_dimensions, (
            f"Covariances must have the same number of dimensions as points, "
            f"but got {np_covariances.shape[1]} dimensions and {n_dimensions} points"
        )
        assert (
            np_covariances.shape[2] == n_dimensions
        ), f"Covariances must be square matrices, but got {np_covariances.shape[1]}x{np_covariances.shape[2]} matrices"
        # check if covariances are symmetric
        assert np.allclose(
            np_covariances, np_covariances.transpose(0, 2, 1)
        ), "Covariances must be symmetric matrices, but got non-symmetric matrices"
        # check if covariances are positive definite
        assert np.all(
            np.min(np.real(np.linalg.eigvals(np_covariances)), axis=1) > 0
        ), "Covariances must be positive definite matrices, but got non-positive definite matrices"

        self.points = np_points
        self.covs = np_covariances

        # altri parametri
        self.covs_inv = np.linalg.inv(np_covariances)
        self.scale = np.linalg.cholesky(np_covariances)
        self.scale_inv = np.linalg.inv(self.scale)

    def __getitem__(self, index) -> "Data":
        data = self.__class__.__new__(self.__class__)
        data.points = self.points[index]
        data.covs = self.covs[index]
        data.covs_inv = self.covs_inv[index]
        data.scale = self.scale[index]
        data.scale_inv = self.scale_inv[index]
        return data

    def __len__(self) -> int:
        """
        Return the number of items in the data.

        :return: The number of items in the data.
        """
        return self.n_samples

    def append(self, data: "Data") -> "Data":
        """Append data to the current instance.

        Args:
            data (Data): The data to append.

        Returns:
            Data: A new instance of Data with the appended data.
        """
        new_data = self.__class__.__new__(self.__class__)
        new_data.points = np.concatenate((self.points, data.points), axis=0)
        new_data.covs = np.concatenate((self.covs, data.covs), axis=0)
        new_data.covs_inv = np.linalg.inv(new_data.covs)
        new_data.scale = np.linalg.cholesky(new_data.covs)
        new_data.scale_inv = np.linalg.inv(new_data.scale)
        return new_data
