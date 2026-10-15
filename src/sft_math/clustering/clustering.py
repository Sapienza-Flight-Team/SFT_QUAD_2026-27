from sft_math.class_templates import ClassBase
from sft_math.typing import Array, Float
import numpy as np
import numpy.typing as np_typing

from matplotlib.patches import Ellipse


class Clustering(ClassBase):
    """
    Base class for clustering algorithms.
    This class provides a template for clustering algorithms with common methods and properties.
    """

    standard_max_iter: int = 100
    standard_tolerance: float = 1e-6
    standard_momentum: float = 0.5
    weighting_max_iter: int = 100
    distance_w: float = 0.5
    entropy_w: float = 1.0

    class Data(ClassBase):

        weights: Array  # shape (n_simulations, n_samples, )
        points: Array  # shape (n_simulations, n_samples, n_dimensions)
        covariances: (
            Array  # shape (n_simulations, n_samples, n_dimensions, n_dimensions)
        )
        opinions: Array  # shape (n_simulations, n_samples, n_opinions)
        inv_cov: Array  # shape (n_simulations, n_samples, n_dimensions, n_dimensions)

        __constants__ = [
            (lambda x: x, "n_samples", ""),
            (lambda x: x, "n_dimensions", ""),
            (lambda x: x, "n_opinions", ""),
        ]

        @property
        def n_samples(self) -> int:
            return self.points.shape[0]

        @property
        def n_dimensions(self) -> int:
            return self.points.shape[1]

        @property
        def n_opinions(self) -> int:
            return self.opinions.shape[1]

        def __init__(
            self,
            points: np_typing.ArrayLike,
            weights: np_typing.ArrayLike,
            covariances: np_typing.ArrayLike,
            opinions: np_typing.ArrayLike,
        ):
            # save points
            np_points = np.array(np.array(points, dtype=Float), copy=False, ndmin=2)
            assert (
                np_points.ndim == 2
            ), f"Points must be a 2D array, but got {np_points.ndim}D"
            n_samples, n_dimensions = np_points.shape
            assert n_samples > 0, "Points must have at least one sample"
            assert n_dimensions > 0, "Points must have at least one dimension"

            # save weights
            np_weights = np.array(np.array(weights, dtype=Float), copy=False, ndmin=1)
            assert (
                np_weights.ndim == 1
            ), f"Weights must be a 1D array, but got {np_weights.ndim}D"
            assert np_weights.shape[0] == n_samples, (
                f"Weights must have the same number of samples as points, "
                f"but got {np_weights.shape[0]} samples and {n_samples} points"
            )

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

            # save opinions
            np_opinions = np.array(np.array(opinions, dtype=Float), copy=False, ndmin=2)
            assert (
                np_opinions.ndim == 2
            ), f"Opinions must be a 2D array, but got {np_opinions.ndim}D"
            assert np_opinions.shape[0] == n_samples, (
                f"Opinions must have the same number of samples as points, "
                f"but got {np_opinions.shape[0]} samples and {n_samples} points"
            )
            assert np.all(np_opinions >= 0), "Opinions must be non-negative"
            assert np.all(
                np.sum(np_opinions, axis=1) <= 1
            ), "Opinions must sum to 1 for each sample"
            n_opinions = np_opinions.shape[1]
            assert n_opinions > 0, "Opinions must have at least one opinion"

            self.weights = np_weights
            self.points = np_points
            self.covariances = np_covariances
            self.opinions = np_opinions
            self.inv_cov = np.linalg.inv(np_covariances)

        def __getitem__(self, index) -> "Clustering.Data":
            """Get a subset of the data."""
            return Clustering.Data(
                self.points[index, :],
                self.weights[index],
                self.covariances[index, :, :],
                self.opinions[index, :],
            )

        def __setitem__(self, index, data: "Clustering.Data") -> None:
            """Set a subset of the data."""
            assert (
                data.n_dimensions == self.n_dimensions
            ), f"Dimension mismatch: {data.n_dimensions} != {self.n_dimensions}"
            self.points[index, :] = data.points
            self.weights[index] = data.weights
            self.covariances[index, :, :] = data.covariances
            self.opinions[index, :] = data.opinions

        def __len__(self) -> int:
            """Get the number of samples."""
            return self.n_samples

        def __iter__(self):
            """Iterate over the samples."""
            for i in range(self.n_samples):
                yield self[i]

    __constants__ = [
        # dal modello
        (lambda x: x, "n_clusters", ""),
        (lambda x: x, "fuzziness", ""),
        (lambda x: x, "radius", ""),
        (lambda x: x, "distance_w", ""),
        (lambda x: x, "entropy_w", ""),
        # dal fit
        (lambda x: int(x * 10000) / 100, "presence", "%"),
        (lambda x: x, "n_dimensions", ""),
        (lambda x: x, "n_opinions", ""),
    ]

    n_clusters: int
    fuzziness: float
    radius: float

    n_dimensions: int
    n_opinions: int
    ctrs: Array  # shape (n_clusters, n_dimensions)
    conf_matrix: Array  # shape (n_clusters, n_opinions)

    def __init__(
        self,
        n_clusters: int,
        fuzziness: float,
        radius: float,
    ):
        super().__init__()

        assert n_clusters > 0, "Number of clusters must be greater than 0"
        assert fuzziness >= 1, "Fuzziness must be greater than or equal to 1"

        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.radius = radius

    def __compute_MahalanobisDistances(
        self,
        data: Data,
    ) -> Array:
        """Compute the squared Mahalanobis distances from the data points to the cluster centroids.

        Args:
            data (Data): The data object containing points and their covariance matrices.

        Returns:
            Array: An array of squared Mahalanobis distances, shape (n_samples, n_clusters).
        """
        diff = (
            data.points[:, np.newaxis, :] - self.ctrs[np.newaxis, :, :]
        )  # (n_samples, n_clusters, n_dimensions)
        return np.einsum(
            "sci, sij, scj -> sc",
            diff,
            data.inv_cov,
            diff,
            optimize=True,
            dtype=Float,
        )

    def __compute_CrossEntropy(
        self,
        data: Data,
    ) -> Array:
        """Compute the difference on opinion of each datum respect to all clusters. For example:
        if we have 3 opinions and 2 clusters with confusion matrix:
            0.6 0.3 0.1
            0.2 0.4 0.4
        and the datum has opinion 0.5 0.3 0.2
        respect the cluster '0' we have compatibility:
            - (0.5 * log(0.6) + 0.3 * log(0.3) + 0.2 * log(0.1)) =~ 0.467787998
        respect the cluster '1' we have compatibility:
            - (0.5 * log(0.2) + 0.3 * log(0.4) + 0.2 * log(0.4)) =~ 0.548455007

        Args:
            data (Data): The data object containing opinions and their covariance matrices.

        Returns:
            Array: An array of coherence values, shape (n_samples, n_clusters).
        """
        return -np.einsum(
            "so, co -> sc",
            data.opinions,
            np.log(self.conf_matrix[: self.ctrs.shape[0], :]),
            optimize=True,
            dtype=Float,
        )

    def __get_score(
        self,
        data: Data,
    ) -> Array:
        """Compute the score for each point with respect to the clusters.
        This score is a weighted sum of the squared Mahalanobis distances and the cross-entropy
        between the opinions of the points and the cluster centroids.

        Args:
            data (Data): The data object containing points, weights, and opinions.

        Returns:
            Array: An array of scores, shape (n_samples, n_clusters).
        """
        m_dist_sq = self.__compute_MahalanobisDistances(data)
        cross_entropy = self.__compute_CrossEntropy(data)
        return self.distance_w * m_dist_sq + self.entropy_w * cross_entropy

    def __init_centroids(
        self,
        data: Data,
    ) -> Array:
        """
        Initialize centroids for the clustering algorithm.
        This method uses a KMeans++ like algorithm to initialize the centroids based on the data points.

        Args:
            data (Data): The data object containing points and weights.

        Returns:
            Array: An array of scores computed for the initial centroids.
                with shape (n_samples, n_clusters).
        """
        self.ctrs = np.empty((0, self.n_dimensions), dtype=Float)
        probabilities: Array = data.weights / np.sum(data.weights)
        for _ in range(self.n_clusters):
            # inizializzo casualmente il primo centroide usando un campionamento casuale
            ids = np.random.choice(data.n_samples, size=1, p=probabilities)[0]
            new_ctrs = data.points[ids]
            self.ctrs = np.concatenate((self.ctrs, new_ctrs[np.newaxis, :]), axis=0)
            # calcolo le probabilità usando lo score
            scores = self.__get_score(data)
            scores = np.min(scores, axis=1)
            probabilities = scores / np.sum(scores)
        return scores

    def __init_confusion_matrix(
        self,
        data: Data,
    ):
        """Initialize the confusion matrix for the clustering algorithm.

        Args:
            data (Data): The data object containing opinions and weights.

        Returns:
            Array: An array of confusion matrix, shape (n_clusters, n_opinions).
        """
        # inizializzo la confusion matrix con valori casuali
        self.conf_matrix = np.random.rand(self.n_clusters, data.n_opinions)
        self.conf_matrix = self.conf_matrix / np.sum(
            self.conf_matrix, axis=1, keepdims=True
        )

    def __compute_mships(
        self,
        scores: Array,
    ) -> Array:
        """Compute the membership matrix for the clustering algorithm.

        Args:
            scores (Array): The scores computed for each point with respect to the clusters.
                with shape (n_samples, n_clusters).

        Returns:
            Array: An array of membership values, shape (n_samples, n_clusters).
        """
        # static parameters
        values = scores - self.radius

        # individuo i dati per i quali esiste almeno un rate negativo
        mask = np.any(
            values < 0, axis=1
        )  # (n_samples,) with masked_n_samples True components
        if np.any(mask):
            # per tali dati vige la classificazione booleana
            masked_min_values = np.min(values[mask], axis=1)  # (masked_n_samples,)
            masked_mship = np.isclose(
                values[mask], masked_min_values[:, None]
            )  # (masked_n_samples, n_clusters)
            # se un dato ha più True allora ne scelgo uno casuale con probabilità descritta da weights
            masked_counter = np.sum(masked_mship, axis=1)  # (masked_n_samples,)
            # genero numeri casuali tra 0 e counter per ogni dato della maschera
            masked_random_numbers = np.random.randint(
                0, masked_counter
            )  # (masked_n_samples,)
            # seleziono così le membership
            true_indices = np.stack(
                np.where(masked_mship), axis=1
            )  # shape (masked_n_samples, 2)
            selected_indices = (
                np.cumsum(masked_counter) - masked_random_numbers - 1
            )  # shape (masked_n_samples,)
            true_indices = true_indices[selected_indices]
            masked_mship = np.zeros_like(masked_mship, dtype=bool)
            masked_mship[true_indices[:, 0], true_indices[:, 1]] = True
            # arrivati a questo punto masked_mship ha un'attribuzione booleana solo per i dati che hanno almeno un rate negativo
            # per questi dati quindi il rate diverrà 0 solo per le membership booleane true
            values[mask] = (~masked_mship).astype(Float)

        # Se un dato ha dei rate nulli allora comandando quei rate nulli, quindi gli altri rate si possono ignorare
        # mi chiedo per quali dati i values possono essere nulli
        mask = np.any(np.isclose(values, 0), axis=1)
        if np.any(mask):
            # per ogni dato dove i values possono essere nulli impongo 0 agli altri values e 1 a tali values nulli
            values[mask] = np.isclose(values[mask], 0).astype(Float)
        if np.any(~mask):
            # per tutti gli altri dati uso la formula del CMeans
            values[~mask] = values[~mask] ** (-1 / (self.fuzziness - 1))

        # arrivati a questo punto i dati che avevano almeno uno 0 ora hanno rate nulli eccetto laddove c'era lo 0 (ora c'è 1)
        # mentre gli altri dati (privi di 0 e valori negativi) hanno valori positivi

        # normalizzo
        values = np.array(values / np.sum(values, axis=1, keepdims=True))
        return values**self.fuzziness

    def __update_ctrs(
        self,
        data: Data,
        mships_weighted: Array,
    ) -> None:
        """Compute the centroids of the clusters based on the data and their memberships.

        Args:
            data (Data): The data object containing points, weights, and opinions.
        """
        sum_inv_cov = np.einsum(
            "sc, sij -> cij", mships_weighted, data.inv_cov, optimize=True, dtype=Float
        )
        sum_vecs = np.einsum(
            "sc, sij, sj -> ci",
            mships_weighted,
            data.inv_cov,
            data.points,
            optimize=True,
            dtype=Float,
        )
        # calcolo le inverse dei centroidi
        sum_cov = np.linalg.inv(sum_inv_cov)
        # calcolo i centroidi
        self.ctrs = np.einsum(
            "cij, cj -> ci", sum_cov, sum_vecs, optimize=True, dtype=Float
        )

    def __update_conf_matrix(
        self,
        data: Data,
        mships: Array,
    ):
        L = np.einsum("sc, so, s -> co", mships, data.opinions, data.weights)
        self.conf_matrix = L / np.sum(L, axis=-1, keepdims=True)

    def __get_energy(
        self,
        scores: Array,
        mships: Array,
        weights: Array,
    ) -> float:
        """Compute the energy of the clustering.

        Args:
            scores (Array): shape (n_samples, n_clusters)
            mships (Array): shape (n_samples, n_clusters)
            weights (Array): shape (n_samples,)

        Returns:
            float: The energy of the clustering, computed as the average of the scores weighted by the memberships.
        """
        return np.einsum(
            "sc, sc, s -> ",
            scores - self.radius,
            mships,
            weights,
            optimize=True,
            dtype=Float,
        ) / np.sum(weights)

    def __plot(
        self,
        data: Data,
        mships: Array,
        covs_p: float = 0.95,
    ) -> None:
        """Plot the clustering results."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i in range(data.n_samples):
            # calcolo l'ellisse
            vals, vecs = np.linalg.eig(data.inv_cov[i])
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ell = Ellipse(
                data.points[i],
                width=vals[0] * -np.log(1 - covs_p) * 2,
                height=vals[1] * -np.log(1 - covs_p) * 2,
                angle=angle,
                color="gray",
                alpha=0.05,
            )
            ax.add_patch(ell)
        # stabilisco dei colori per ogni simulazione
        ax.scatter(
            self.ctrs[:, 0],
            self.ctrs[:, 1],
            color="red",
            marker="x",
            s=5,
            label="Centroids",
        )
        ax.set_title(f"Clustering Results")
        ax.legend()
        fig.show()

    def set_centroids(
        self,
        centroids: np_typing.ArrayLike,
    ):
        self.ctrs = np.array(centroids, dtype=Float)
        assert (
            self.ctrs.ndim == 2
        ), f"Centroids must be a 2D array, but got {self.ctrs.ndim}D"
        assert (
            self.ctrs.shape[0] == self.n_clusters
        ), f"Centroids must have {self.n_clusters} rows, but got {self.ctrs.shape[0]} rows"

    def set_confusion_matrix(
        self,
        confusion_matrix: np_typing.ArrayLike,
    ):
        self.conf_matrix = np.array(confusion_matrix, dtype=Float)
        assert (
            self.conf_matrix.ndim == 2
        ), f"Confusion matrix must be a 2D array, but got {self.conf_matrix.ndim}D"
        assert (
            self.conf_matrix.shape[0] == self.n_clusters
        ), f"Confusion matrix must have {self.n_clusters} rows, but got {self.conf_matrix.shape[0]} rows"
        assert np.all(
            np.isclose(np.sum(self.conf_matrix, axis=1), 1)
        ), "Confusion matrix must have rows summing to 1"
        assert np.all(self.conf_matrix > 0), "Confusion matrix must be positive"

    def __fit_standard(
        self,
        data: Data,
        batch_size: int,
    ):
        curr_energy = np.inf
        # entro nel loop
        for i in range(self.standard_max_iter):
            # seleziono i dati che potrei usare in questo loop
            sampled_data = data[
                np.random.choice(
                    data.n_samples, size=min(batch_size, data.n_samples), replace=False
                )
            ]
            # calcolo le membership
            scores = self.__get_score(sampled_data)
            mships = self.__compute_mships(scores)
            # test dell'energia
            energy = self.__get_energy(scores, mships, sampled_data.weights)
            if curr_energy - energy < self.standard_tolerance:
                break
            curr_energy = energy
            # ricalcolo i centroidi
            self.__update_ctrs(sampled_data, mships)
            self.__update_conf_matrix(sampled_data, mships)
            self.__plot(sampled_data, mships)

    def outliers_fit(
        self,
        data: Data,
    ):
        # inizializzo i parametri del fit
        self.n_dimensions = data.n_dimensions
        self.n_opinions = data.n_opinions
        if hasattr(self, "conf_matrix") is False:
            self.__init_confusion_matrix(data)
        else:
            assert (
                self.conf_matrix.shape[1] == data.n_opinions
            ), f"Confusion matrix must have {data.n_opinions} columns, but got {self.conf_matrix.shape[1]} columns"

        # calcolo i centroidi se non noti usando KMeans++ like algorithm
        if hasattr(self, "ctrs") is False:
            self.__init_centroids(data)
        else:
            self.ctrs = np.array(centroids, dtype=Float)
            assert (
                self.ctrs.ndim == 2
            ), f"Centroids must be a 2D array, but got {self.ctrs.ndim}D"
            assert (
                self.ctrs.shape[0] == self.n_clusters
            ), f"Centroids must have {self.n_clusters} rows, but got {self.ctrs.shape[0]} rows"
            assert (
                self.ctrs.shape[1] == data.n_dimensions
            ), f"Centroids must have {data.n_dimensions} columns, but got {self.ctrs.shape[1]} columns"
        # entro nel loop
        for i in range(self.weighting_max_iter):
            # per prima cosa eseguo un clustering fuzzy completo
            # Questo clustering mi darà delle membership fuzzy che potrò usare per il sampling
            # Avvio
            self.__fit_standard(data, batch_size=5000)

    """def fit(
        self,
        data: Data,
        confusion_matrix: np_typing.ArrayLike,
        max_lines: int,  # massimo numero di linee supportate
        gen_lines: int,  # numero di linee aperte da un singolo ente
        noise: float,  # variazione delle linee aperte
        initial_centroids: np_typing.ArrayLike | None = None,
    ) -> None:
        self.n_dimensions = data.n_dimensions
        if initial_centroids is None:
            # genero "max_lines" linee
            init_ctrs = np.random.randn(max_lines, self.n_clusters, self.n_dimensions)

    """


if __name__ == "__main__":
    # creo i centroidi
    n_centroids = 5
    n_centroids_test = 5
    n_opinions = 6
    # confusion matrix
    M = np.random.rand(n_centroids, n_opinions)
    M = np.array(M / np.sum(M, axis=1, keepdims=True))  # normalizzo le righe
    centroids = np.random.rand(n_centroids, 2)
    n_samples = 10_000
    # creo le covarianze
    S = np.random.rand(n_samples, 2, 2) * 1.2 + 0.8
    S[:, 0, 1] = (
        np.random.rand(n_samples) * 0.2 - 0.1
    )  # aggiungo un po' di correlazione
    S[:, 1, 0] = 0.0
    S /= 3
    covariances = np.einsum("sij, skj -> sik", S, S) / 5  # prodotto esterno
    weights = np.random.rand(n_samples) * 1.5 + 0.5
    # creo i punti
    # imposto la membership
    mships_idx = np.random.randint(0, n_centroids, size=n_samples)
    # genero le opinioni
    opinions = np.clip(
        M[mships_idx] + np.random.randn(n_samples, n_opinions) * 0.01, 0, 1
    )
    opinions = opinions / (
        1.001 * np.sum(opinions, axis=1, keepdims=True)
    )  # normalizzo le opinioni
    # genero gli errori
    particular_errors = np.random.randn(n_samples, 2)
    particular_errors = np.einsum("sij, sj -> si", S, particular_errors)
    points = centroids[mships_idx] + particular_errors

    data = Clustering.Data(
        points=points,
        weights=weights,
        covariances=covariances,
        opinions=opinions,
    )

    clustering = Clustering(
        n_clusters=n_centroids_test,
        fuzziness=2.0,
        radius=0,
    )

    clustering.outliers_fit(
        data=data,
    )


# i dati hanno shape (Q, S,)
# il clusterize ha shape (P, C,)

# il clustering lavora separatamente producendo P,Q centroidi
