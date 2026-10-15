from sft_math.class_templates import ClassBase
import numpy as np
from numpy import typing as np_typing
from sft_math.typing import Float, Array, Int, Index, Bool, Bools
from typing import Tuple
from math import sqrt, log
import warnings
import itertools

# importo l'inizializzatore per il kmeans KMeans++
from scipy.stats import norm
from scipy.stats import chi2

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


class PClustering(ClassBase):

    nested_tol: float = 1e-3  # tolerance for convergence of energy in fuzzyPClustering
    nested_max_iter: int = 100  # maximum number of iterations for convergence
    limit_permutations: int = (
        7  # maximum number of centroids for a brute force control in __check_permutations
    )
    n_attempts: int = (
        500  # number of attempts for random sampling in __check_permutations
    )
    verbose: bool = True

    conf_M: Array
    n_clusters: int
    p_value: float
    fuzziness: float
    threshold: float

    __constants__ = [
        (lambda x: x, "n_clusters", ""),
        (lambda x: int(x * 1000000) / 10000, "p_value", "%"),
        (lambda x: x, "fuzziness", ""),
        (lambda x: int(x * 1000000) / 10000, "threshold", "%"),
    ]

    @staticmethod
    def plotPClustering(
        points: Array,
        ctrs: Array,
        classes: Index,
        covs_inv: Array,
        mship: Array,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        covs_p: float = 0.99,
    ) -> Tuple[plt.Figure, plt.Axes]:

        n_clusters = ctrs.shape[0]

        fig, ax = plt.subplots()
        # la finestra anche dovrà avere le giuste proporzioni
        h = y_max - y_min
        w = x_max - x_min
        area = h * w
        h *= sqrt(100 / area)
        w *= sqrt(100 / area)
        fig.set_size_inches(int(w), int(h))
        ax.set_xlim(1.5 * x_min, 1.5 * x_max)
        ax.set_ylim(1.5 * y_min, 1.5 * y_max)
        # per ogni dato mostro un'ellisse che rappresenta la matrice di covarianza
        for i in range(points.shape[0]):
            if mship[i].sum() > 0:
                # calcolo l'ellisse
                vals, vecs = np.linalg.eig(covs_inv[i])
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                ell = Ellipse(
                    points[i],
                    width=vals[0] * -log(1 - covs_p) * 2,
                    height=vals[1] * -log(1 - covs_p) * 2,
                    angle=angle,
                    color="gray",
                    alpha=0.05,
                )
                ax.add_patch(ell)
        paths = []
        for c in range(n_clusters):
            # aggiungo gli scatter
            indices = classes == c
            paths.append(
                ax.scatter(points[indices, 0], points[indices, 1], s=1, alpha=0.4)
            )
        for c in range(n_clusters):
            ax.scatter(
                ctrs[c, 0],
                ctrs[c, 1],
                s=100,
                marker="x",
                c=paths[c].get_facecolor(),
                alpha=1,
                label=f"Cluster {c}",
            )
        # aggiungo la legenda
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")
        # costruisco segmenti che collegano i dati ai loro centroidi
        for i in range(points.shape[0]):
            value = np.mean(mship[i, :])
            for c in range(mship.shape[1]):
                if mship[i, c] > value:
                    ax.plot(
                        [points[i, 0], ctrs[c, 0]],
                        [points[i, 1], ctrs[c, 1]],
                        color="black",
                        alpha=0.05,
                    )
        ax.set_title("Cluster Visualization with Centroids")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def __compute_centroids(
        points: Array,
        covs_inv: Array,
        mship: Array,
    ) -> Array:
        """Compute the centroids of the clusters given the current membership.

        Args:
            points (Array): Data points for the current iteration.
                with shape (n_samples, n_dimensions).
            covs_inv (Array): Inverse covariance matrices for the clusters.
                with shape (n_samples, n_dimensions, n_dimensions) and definitly positive.
            mship (Array): Membership matrix indicating the cluster assignment.
                with shape (n_samples, n_clusters) and positive values.

        Returns:
            Array: Centroids of the clusters with shape (n_clusters, n_dimensions).
        """

        # NOTE: non ci aspettiamo che il sistema non sia risolvibile poiché:
        # la somma di matrici definite positive è definita positiva
        # e l'inversa di una matrice definita positiva rimane definita positiva
        # l'unico caso non desiderato è l'assenza di dati nei cluster
        # in tal caso il claster contiene i dati usando mship opposta
        # calcolo i centroidi relativi alle membership correnti

        mask = np.any(mship > 0, axis=0)
        if not np.all(mask):
            pass
        mship = mship.copy()
        mship[:, ~mask] = np.ones_like(mship[:, ~mask], dtype=Float)

        sum_C_inv = np.array(
            np.einsum("sij, sc -> cij", covs_inv, mship), dtype=Float
        )  # (n_clusters, n_dimensions, n_dimensions)
        sum_vecs = np.array(
            np.einsum("sij, sc, sj -> ci", covs_inv, mship, points), dtype=Float
        )  # (n_clusters, n_dimensions)
        try:
            centroids = np.array(np.linalg.solve(sum_C_inv, sum_vecs), dtype=Float)
        except np.linalg.LinAlgError:
            centroids = np.zeros((mship.shape[1], points.shape[1]), dtype=Float)
        return centroids

    @staticmethod
    def __dist_Mahalanobis(
        points: Array,
        covs_inv: Array,
        ctrs: Array,
    ) -> Array:
        """Compute the Mahalanobis distance between points and centroids.
        Args:
            points (Array): Data points to fit.
                with shape (n_samples, n_dimensions).
            covs_inv (Array): Inverse covariance matrices for the clusters.
                with shape (n_samples, n_dimensions, n_dimensions).
            ctrs (Array): Centroids of the clusters.
                with shape (n_clusters, n_dimensions).

        Returns:
            Array: Mahalanobis distance between points and centroids.
                with shape (n_samples, n_clusters).
        """
        diff = np.array(
            points[:, None, :] - ctrs[None, :, :], dtype=Float
        )  # (n_samples, n_clusters, n_dimensions)
        dist = np.einsum(
            "sci, sij, scj -> sc", diff, covs_inv, diff
        )  # (n_samples, n_clusters)
        return dist

    @staticmethod
    def __dist_PMahalanobis(
        points: Array,
        covs_inv: Array,
        ctrs: Array,
        confusion_matrix: Array,
        mship_reference: Index,
    ) -> Array:
        """Compute the rates for the clusters given the current centroids.

        Args:
            points (Array): Data points for the current iteration.
                with shape (n_samples, n_dimensions).
            covs_inv (Array): Inverse covariance matrices for the clusters.
                with shape (n_samples, n_dimensions, n_dimensions).
            ctrs (Array): Centroids of the clusters.
                with shape (n_clusters, n_dimensions).
            confusion_matrix (Array): Confusion matrix for evaluation.
                with shape (n_clusters, n_classes) is a stochastic matrix
            mship_reference (Index): Reference membership matrix for comparison.
                with shape (n_samples,) with values in [0, n_clusters-1].

        Returns:
            Array: Rates for each cluster.
                with shape (n_samples, n_clusters).
        """
        dist = PClustering.__dist_Mahalanobis(points, covs_inv, ctrs)
        # M[r,c] è la probabilità che un dato sia assegnato alla classe c nonostante realmente appartenga alla classe r
        # dist è la distanza tra il dato e un centroide
        rates = (
            0.5 * dist[..., None] - np.log(confusion_matrix)[None, ...]
        )  # (n_samples, n_clusters, n_classes)
        # rates[s, r, c] è quindi il rate del dato (s) assegnato alla classe (c) rispetto il centroide di classe (r)
        # si prende per vera l'opinione del dato e si cerca il centroide più ragionevole

        # scambio le ultime due colonne
        rates = np.moveaxis(rates, -1, -2)  # (n_samples, n_classes, n_clusters)

        # mship_reference è un vettore con un indice per ogni dato, che indica la classe a cui è stato assegnato (la c insomma)
        # voglio avere quindi: risultato[s, r] = rate[s, mship[s], r]
        # quindi il rate dell'associazione dato (s) e class (r)
        return rates[np.arange(mship_reference.shape[0]), mship_reference].reshape(
            points.shape[0], ctrs.shape[0]
        )  # (n_samples, n_clusters)

    @staticmethod
    def __compute_Membership(
        points: Array,
        covs_inv: Array,
        ctrs: Array,
    ) -> Index:
        """Compute the boolean membership of the clusters given the current centroids.

        Args:
            points (Array): Data points for the current iteration.
                with shape (n_samples, n_dimensions).
            covs_inv (Array): Inverse covariance matrices for the clusters.
                with shape (n_samples, n_dimensions, n_dimensions).
            ctrs (Array): Centroids of the clusters.
                with shape (n_clusters, n_dimensions).

        Returns:
            Index: Membership of the data points to the clusters.
                with shape (n_samples,).
        """
        distances = PClustering.__dist_Mahalanobis(
            points, covs_inv, ctrs
        )  # (n_samples, n_clusters)
        return np.array(np.argmin(distances, axis=1), dtype=Int)  # (n_samples,)

    @staticmethod
    def __compute_fuzzyPMembership(
        points: Array,
        covs_inv: Array,
        ctrs: Array,
        confusion_matrix: Array,
        mship_reference: Index,
        fuzziness: float,
        threshold: float,
    ) -> Tuple[Array, Array]:
        """Compute the membership of the clusters given the current centroids.

        Args:
            points (Array): Data points for the current iteration.
                with shape (n_samples, n_dimensions).
            covs_inv (Array): Inverse covariance matrices for the clusters.
                with shape (n_clusters, n_dimensions, n_dimensions).
            ctrs (Array): Centroids of the clusters.
                with shape (n_clusters, n_dimensions).
            confusion_matrix (Array): Confusion matrix for evaluation.
                with shape (n_clusters, n_clusters).
            mship_reference (Index): Reference membership matrix for comparison.
                with shape (n_samples,).
            fuzziness (float): Fuzziness parameter for the membership calculation.
                The higher the value, the more fuzzy the membership calculation.
            threshold (float): Relaxation parameter for the algorithm.
                This value expresses a statistical radius for a boolean behaviour.

        Returns:
            tuple: A tuple containing
                - mship (Array): Membership matrix for each cluster.
                    with shape (n_samples, n_clusters).
                - rates (Array): Rates for each cluster (considering the threshold).
                    with shape (n_samples, n_clusters).
        """
        rates = PClustering.__dist_PMahalanobis(
            points, covs_inv, ctrs, confusion_matrix, mship_reference
        )  # (n_samples, n_clusters)
        rates = rates - threshold
        values = rates.copy()

        # individuo i dati per i quali esiste almeno un rate negativo
        mask = np.any(
            values < 0, axis=1
        )  # (n_samples,) with masked_n_samples True components
        if np.any(mask):
            # per tali dati vige la classificazione booleana
            masked_min_values = np.min(
                values[mask], axis=1
            )  # (masked_n_samples, n_clusters)
            masked_mship = np.isclose(
                values[mask], masked_min_values[:, None]
            )  # (masked_n_samples, n_clusters)
            # se un dato ha più True allora ne scelgo uno casuale
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
            masked_mship = np.zeros_like(masked_mship, dtype=Bool)
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
            values[~mask] = values[~mask] ** (-1 / (fuzziness - 1))

        # arrivati a questo punto i dati che avevano almeno uno 0 ora hanno rate nulli eccetto laddove c'era lo 0 (ora c'è 1)
        # mentre gli altri dati (privi di 0 e valori negativi) hanno valori positivi

        # normalizzo
        values = values / np.sum(values, axis=1, keepdims=True)
        return values**fuzziness, rates

    @staticmethod
    def __energy_fuzzyPClustering(
        mship: Array,
        rates: Array,
    ) -> float:
        """Compute the energy of the fuzzy PClustering model.
        This function computes the energy of the fuzzy PClustering model:
        E = sum_{i: sample, j: centroid} rates[i,j] * mship[i,j]

        Args:
            mship (Array): Membership matrix indicating the cluster assignment.
                with shape (n_samples, n_clusters).
            rates (Array): Rates for each cluster.
                with shape (n_samples, n_clusters).

        Returns:
            float: Energy of the fuzzy PClustering model.
        """
        return np.sum(mship * rates).item()

    @staticmethod
    def __check_permutations(
        points: Array,
        covs_inv: Array,
        ctrs: Array,
        confusion_matrix: Array,
        mship_reference: Index,
        fuzziness: float,
        threshold: float,
    ) -> Tuple[Tuple[int, ...], float, Array, Array, Array]:
        """Check all permutations of centroids, in this way the method is consistence

        Args:
            points (Array): Data points with shape (n_samples, n_dimensions)
            covs_inv (Array): Covariances of data points with shape (n_samples, n_dimensions, n_dimensions)
            ctrs (Array): Centroids with shape (n_clusters, n_dimensions)
            confusion_matrix (Array): Confusion matrix with shape (n_clusters, n_classes)
            mship_reference (Index): Attributions observed with shape (n_samples,)
            fuzziness (float): fuzziness of the method
            threshold (float): threshold for boolean behaviour

        Returns:
            tuple: A tuple with values
                - indices (Tuple[int,...]): indices of centroids with best energy
                - energy (float): best energy reached
                - centroids (Array): best centroids reached
                - mship (Array): best membership reached
                - rates (Array): best rates reached
        """
        n_clusters = confusion_matrix.shape[0]

        permutations: list[tuple[int, ...]]
        if n_clusters >= PClustering.limit_permutations:
            # le permutazioni sono sampling casuali di indici
            tmp: Index = np.array(
                np.random.rand(PClustering.n_attempts, n_clusters).argsort(axis=1),
                dtype=Int,
            )
            permutations = [tuple(row) for row in tmp]
        else:
            permutations = list(itertools.permutations(range(n_clusters)))

        # calcolo l'energia della permutazione 0
        ctrs_sorted = ctrs[permutations[0], :]
        # calcolo l'energia
        mship, rates = PClustering.__compute_fuzzyPMembership(
            points,
            covs_inv,
            ctrs_sorted,
            confusion_matrix,
            mship_reference,
            fuzziness,
            threshold,
        )
        best_energy = PClustering.__energy_fuzzyPClustering(mship, rates)
        best_indices = permutations[0]
        best_centroids: Array = ctrs_sorted
        best_mship: Array = mship.copy()
        best_rates: Array = rates.copy()
        for indices in permutations[1:]:
            ctrs_sorted = ctrs[indices, :]
            mship, rates = PClustering.__compute_fuzzyPMembership(
                points,
                covs_inv,
                ctrs_sorted,
                confusion_matrix,
                mship_reference,
                fuzziness,
                threshold,
            )
            energy = PClustering.__energy_fuzzyPClustering(mship, rates)
            if energy < best_energy:
                best_indices = indices
                best_energy = energy
                best_mship = mship.copy()
                best_rates = rates.copy()
                best_centroids = ctrs_sorted.copy()
        return best_indices, best_energy, best_centroids, best_mship, best_rates

    @staticmethod
    def __fit_fuzzyPClustering(
        points: Array,
        covs_inv: Array,
        ctrs: Array,
        confusion_matrix: Array,
        mship_reference: Index,
        fuzziness: float,
        threshold: float,
    ) -> Tuple[Array, Array, float]:
        """Fit the PClustering model to the data using the Mahalanobis distance.

        Args:
            points (Array): Data to fit.
                with shape (n_samples, n_dimensions).
            covs_inv (Array): Inverse covariance matrices for the clusters.
                with shape (n_clusters, n_dimensions, n_dimensions).
            ctrs (Array): Centroids of the clusters.
                with shape (n_clusters, n_dimensions).
            confusion_matrix (Array): Confusion matrix for evaluation.
                with shape (n_clusters, n_clusters).
            mship_reference (Index): Reference membership matrix for comparison.
                with shape (n_samples,).
            fuzziness (float): Fuzziness parameter for the membership calculation.
                The higher the value, the more fuzzy the membership calculation.
            threshold (float): Relaxation parameter for the algorithm.
                This value expresses a statistical radius for a boolean behaviour.

        Returns:
            tuple: A tuple containing
                - centroids (Array): Centroids of each cluster with shape (n_clusters, n_dimensions).
                - memberships (Array): Membership matrix indicating the cluster assignment for each data point.
                    with shape (n_samples, n_clusters).
                - energy (float): Energy of the fuzzy PClustering model.
        """
        # controllo iniziale delle permutazioni
        _, init_energy, ctrs, mship, rates = PClustering.__check_permutations(
            points,
            covs_inv,
            ctrs,
            confusion_matrix,
            mship_reference,
            fuzziness,
            threshold,
        )

        if PClustering.verbose:
            print(f"    Nested Clustering >> initial energy: {init_energy}")

        for step in range(PClustering.nested_max_iter):

            # minimizzo l'energia usando i centroidi
            ctrs = PClustering.__compute_centroids(
                points,
                covs_inv,
                mship,
            )

            # minimizzo l'energia usando le membership
            mship, rates = PClustering.__compute_fuzzyPMembership(
                points,
                covs_inv,
                ctrs,
                confusion_matrix,
                mship_reference,
                fuzziness,
                threshold,
            )

            # poiché ha un lower bound allora l'energia è destinata a convergere dall'alto
            energy = PClustering.__energy_fuzzyPClustering(mship, rates)

            if PClustering.verbose:
                print(f"        loop {step} with computed energy: {energy}")

            if init_energy - energy < PClustering.nested_tol:
                init_energy = energy
                break

            init_energy = energy
        else:
            warnings.warn(
                f"Maximum number of iterations reached ({PClustering.nested_max_iter})"
            )
        return ctrs, mship, init_energy

    @staticmethod
    def __cumulative_PTest(
        points: Array,
        covs_inv: Array,
        ctrs: Array,
        mship: Bools,
    ) -> Tuple[Array, Index, Array]:
        """This method returns the p-values for each cluster, in particular:

        - the second output is the index of the clusters sorted by distance from the data points.

        - the first output is the cumulative p-values for each cluster sorted by index:
            pvals[i,j] = p-value obtained from data: indices[0:i] for the j-th cluster

        Args:
            points (Array): Data points to fit.
                with shape (n_samples, n_dimensions).
            covs_inv (Array): Inverse covariance matrices for the clusters.
                with shape (n_clusters, n_dimensions, n_dimensions).
            ctrs (Array): Centroids of the clusters.
                with shape (n_clusters, n_dimensions)

        Returns:
            tuple: A tuple with
                - pvalues (Array): P-values for each cluster with shape (n_samples, n_clusters).
                - indices (Index): Indices of the clusters with shape (n_samples, n_clusters).
                - distances (Array): Distances between data and clusters (n_samples, n_clusters)
        """

        distances = PClustering.__dist_Mahalanobis(
            points,
            covs_inv,
            ctrs,
        )

        indices = np.array(
            np.argsort(distances, axis=0), dtype=Int
        )  # (n_samples,n_clusters)
        dist_sorted = np.take_along_axis(
            distances, indices, axis=0
        )  # (n_samples, n_clusters)
        mship_sorted = np.take_along_axis(
            mship,
            indices,
            axis=0,
        )
        cumulative_sum = np.cumsum(dist_sorted, axis=0)  # (n_samples,)
        cumulative_cnt = np.cumsum(mship_sorted, axis=0)  # (n_samples,)
        averages = (
            cumulative_sum / cumulative_cnt
        )  # (n_samples,)  NOTE: è normale che si divida per 0, si tratta di dati vicini ma non assegnati
        pvals = np.array(
            norm.cdf(
                (averages - points.shape[1])
                / sqrt(2 * points.shape[1])
                * np.sqrt(cumulative_cnt)
            ),
            dtype=Float,
        )  # (n_samples,)
        pvals[cumulative_cnt == 0] = 0
        return pvals, indices, dist_sorted

    @staticmethod
    def __filter_PTest(
        points: Array,
        covs_inv: Array,
        ctrs: Array,
        pval: float,
        step_confidence: float,
        mask_reference: Bools,
    ) -> Bools:
        """Filter the clusters based on the P-test.

        Args:
            points (Array): Data points to fit.
                with shape (n_samples, n_dimensions).
            covs_inv (Array): Inverse covariance matrices for the clusters.
                with shape (n_samples, n_dimensions, n_dimensions).
            ctrs (Array): Centroids of the clusters.
                with shape (n_clusters, n_dimensions).
            pval (float): P-value threshold for filtering.
            step_confidence (float): Confidence at the end of the filtering
            mask_reference (Bools): Reference membership matrix for comparison.
                with shape (n_samples,)

        Description:
            The PTest filter cleans the worst data from the model.
            If the filter removes a data then it has a probability of 1-pval
            to having done a mistake.
            moment is the maximum total error of the filter.
            It is the probability of having done a mistake in the filtering.

        Returns:
            Bools: Filtered membership matrix.
        """

        N_max = int(log(step_confidence) / log(1 - pval))

        # computo memberships booleane per avere così dei cluster esclusivi
        mship_index = PClustering.__compute_Membership(points, covs_inv, ctrs)
        mship_boolean: Bools = np.zeros(
            (mship_index.shape[0], ctrs.shape[0]), dtype=Bool
        )
        mship_boolean[np.arange(mship_boolean.shape[0]), mship_index] = True

        pvals, indices, distances = PClustering.__cumulative_PTest(
            points,
            covs_inv,
            ctrs,
            mship_boolean,
        )  # (n_samples, n_clusters), (n_samples,), (n_samples, n_clusters)

        good = (
            pvals <= 1 - pval
        )  # maschera dei dati migliori secondo l'ordine indicato da indices
        mship_sorted = np.take_along_axis(
            mship_boolean,
            indices,
            axis=0,
        )
        good_ship = np.logical_and(good, mship_sorted)  # (n_samples, n_clusters)
        # classificazione dei dati:
        # un dato inlier sia in good_ship che in mask_reference :
        #     essendo confermato da entrambi rimane inlier
        # un dato inlier in good_ship ma non in mask_reference diventa outlier
        #     essendo una novità rimane inlier
        # un dato outlier in good_ship ma non in mask_reference diventa outlier
        #     ne posso eliminare al più N_max
        # un dato outlier in good_ship e in mask_reference diventa outlier
        #     possono rimanere outlier

        # realizzo la maschera 1D dedotta da good_ship nell'ordine canonico dei dati
        # good_ship[i,j] ci dice che il dato indices[i,j] è considerato inlier
        good_ship_restore = np.zeros(good_ship.shape, dtype=Bool)
        np.put_along_axis(
            good_ship_restore,
            indices,
            good_ship,
            axis=0,
        )
        good_restore = np.any(good_ship_restore, axis=1)  # (n_samples,)

        # ora ho una maschera 1D che mi dice quali dati sono buoni
        # confronto good_restore con mask_reference
        loses_data = np.logical_and(mask_reference, ~good_restore)
        if np.any(loses_data):
            tmp = loses_data[loses_data]
            # rendo false tmp.shape[0]-N_max dati casuali
            # quindi calcolo la probabilità di scelta come maggiore per dati con distanza minore
            distances_restore = np.zeros(distances.shape, dtype=Float)
            np.put_along_axis(
                distances_restore,
                indices,
                distances,
                axis=0,
            )
            distances_restore = np.min(distances_restore, axis=1)  # (n_samples,)
            loses_distances = distances_restore[loses_data]

            if PClustering.verbose:
                print(
                    f"    PTest >> {min(N_max, tmp.shape[0])} inlier data points became outliers"
                )
                print(
                    f"    PTest >> {np.sum(np.logical_and(~mask_reference, good_restore))} outlier data points became inliers"
                )

            probabilities = loses_distances / np.sum(loses_distances, keepdims=True)
            tmp = ~tmp
            tmp[
                np.random.choice(
                    tmp.shape[0],
                    size=min(N_max, tmp.shape[0]),
                    replace=False,
                    p=probabilities,
                )
            ] = True
            good_restore[loses_data] = ~tmp

        # NOTE: non mi aspetto di avere solo outliers
        return good_restore

    class Data(ClassBase):
        points: Array
        covs: Array
        mship: Bools
        mship_idx: Index
        n_clusters: int
        n_samples: int
        n_dimensions: int
        covs_inv: Array

        __constants__ = [
            (lambda x: x, "n_clusters", ""),
            (lambda x: x, "n_samples", ""),
            (lambda x: x, "n_dimensions", ""),
        ]

        def __init__(
            self,
            points: np_typing.ArrayLike,
            covs: np_typing.ArrayLike,
            mship: np_typing.ArrayLike,
            n_clusters: int,
        ):
            np_points: Array = np.array(points, dtype=Float)
            np_covs: Array = np.array(covs, dtype=Float)
            np_mship: Index = np.array(mship, dtype=Int)

            # check the shapes
            assert np_points.ndim == 2, f"data must be 2D, but got {np_points.ndim}"
            assert (
                np_points.shape[0] == np_covs.shape[0]
            ), f"data and covs must have the same number of samples, but got {np_points.shape[0]} and {np_covs.shape[0]}"
            n_samples = np_points.shape[0]
            n_dimensions = np_points.shape[1]
            assert (
                np_covs.shape[1] == np_covs.shape[2]
            ), f"covs must be square, but got {np_covs.shape}"
            assert (
                np_covs.shape[1] == n_dimensions
            ), f"covs must have the same number of dimensions as data, but got {np_covs.shape[1]} and {n_dimensions}"
            assert np_mship.ndim == 1, f"mship must be 1D, but got {np_mship.ndim}"
            assert (
                np_mship.shape[0] == n_samples
            ), f"mship must have the same number of samples as data, but got {np_mship.shape[0]} and {n_samples}"
            assert np.all(
                np_mship >= 0
            ), f"mship must be non-negative, but got {np_mship}"
            assert np.all(
                np_mship < n_clusters
            ), f"mship must be less than {n_clusters}"

            # check the covariance matrices
            assert np.allclose(
                np_covs, np.moveaxis(np_covs, -1, -2)
            ), "covariance must be symmetric"
            assert np.all(
                np.linalg.eigvals(np_covs) > 0
            ), "covariance must be positive definite"

            # convert np_mship to boolean
            boolean_np_mship = np.zeros((n_samples, n_clusters), dtype=Bool)
            boolean_np_mship[np.arange(n_samples), np_mship] = True

            self.points = np_points
            self.covs = np_covs
            self.mship = boolean_np_mship
            self.mship_idx = np_mship
            self.n_clusters = n_clusters
            self.n_samples = n_samples
            self.n_dimensions = n_dimensions
            self.covs_inv = np.linalg.inv(np_covs)

        def append(
            self,
            points: np_typing.ArrayLike,
            covs: np_typing.ArrayLike,
            mship: np_typing.ArrayLike,
            n_clusters: int,
        ) -> "PClustering.Data":
            """Append another Data object to this one.

            Args:
                other (PClustering.Data): Another Data object to append.

            Returns:
                PClustering.Data: A new Data object with the combined data.
            """
            np_points: Array = np.array(points, dtype=Float)
            np_covs: Array = np.array(covs, dtype=Float)
            np_mship: Index = np.array(mship, dtype=Int)
            points = np.concatenate((self.points, np_points), axis=0)
            covs = np.concatenate((self.covs, np_covs), axis=0)
            mship = np.concatenate((self.mship_idx, np_mship), axis=0)
            return PClustering.Data(points, covs, mship, self.n_clusters)

    def __init__(
        self,
        conf_M: np_typing.ArrayLike,
        p_value: float,
        fuzziness: float,
        threshold: float,
    ):
        """Constructor for the PClustering class.

        Args:
            conf_M (np_typing.ArrayLike): Confusion matrix for evaluation.
                M[i, j] is the probability that a the class i is assigned to the class j.
            p_value (float): Confidence of the localisation.
                If p_value = 5% then the algorithm mistakes when rejecting the null hypothesis with a probability of 5%.
                The null hypotesis is that the centroid extimation is correct.
                pval < 0.5
            fuzziness (float): Fuzziness parameter for the clustering algorithm.
                The optimization model predicts solutions with more entropy when fuzziness is greater than 1.
                fuzziness = 1 means that the algorithm is a k-means, so this option is invalid.
            threshold (float): Relaxation parameter for the algorithm.
                This value expresses a statistical radius for a boolean behaviour.
        """
        super().__init__()

        np_conf_M = np.array(conf_M, dtype=Float)

        # check inputs
        assert (
            np_conf_M.ndim == 2
        ), f"confusion matrix must be 2D, but got {np_conf_M.ndim}"
        assert (
            np_conf_M.shape[0] == np_conf_M.shape[1]
        ), f"confusion matrix must be square, but got {np_conf_M.shape}"
        assert np.all(
            np_conf_M >= 0
        ), f"confusion matrix must be non-negative, but got {np_conf_M}"
        assert np.all(
            np.isclose(np.sum(np_conf_M, axis=1), 1)
        ), f"confusion matrix must be normalized, but got {np_conf_M}"
        assert (
            p_value > 0 and p_value < 1
        ), f"p_value must be in (0, 1), but got {p_value}"
        assert p_value < 0.5, f"p_value must be lesser than 0.5, but got {p_value}"
        assert fuzziness > 1, f"fuzziness must be greater than 1, but got {fuzziness}"
        assert threshold >= 0, f"threshold must be greater than 0, but got {threshold}"
        assert (
            threshold >= 0.5
        ), f"threshold must be greater then 0.5, but fot {threshold}"

        self.n_clusters = np_conf_M.shape[0]
        self.conf_M = np_conf_M
        self.p_value = p_value
        self.fuzziness = fuzziness
        self.threshold = threshold

    @staticmethod
    def __init_centroids(
        points: Array,
        covs_inv: Array,
        n_clusters: int,
        mship_ref: Index,
        conf_matrix: Array,
    ) -> Array:
        """Initialize the centroids for the PClustering algorithm.

        Args:
            points (Array): Data points to fit.
                with shape (n_samples, n_dimensions).
            covs_inv (Array): Inverse covariance matrices for the clusters.
                with shape (n_samples, n_dimensions, n_dimensions).
            n_clusters (int): Number of clusters to fit.
            mship_ref (Index): Reference membership matrix for comparison.
                with shape (n_samples,).
            conf_matrix (Array): Confusion matrix for evaluation.
                with shape (n_clusters, n_clusters) is a stochastic matrix

        Returns:
            Array: Centroids of the clusters with shape (n_clusters, n_dimensions).
        """

        n_clusters = conf_matrix.shape[0]
        ctrs: Array = np.empty((0, points.shape[1]), dtype=Float)
        # il primo centroide è di tipo 0, la probabilità scelta dipende dall'opinione
        # conf_matrix[0, :] è la probabilità che un dato di tipo 0 venga classificato in un certo modo
        # così il primo rate sarà dato da conf_matrix[0, mship_ref]
        base_rate = conf_matrix[0, mship_ref]  # (n_samples,)
        base_rate /= np.sum(base_rate, axis=0, keepdims=True)  # (n_samples,)
        ctrs = np.append(
            ctrs,
            points[
                np.array(
                    np.random.choice(
                        np.arange(points.shape[0]),
                        size=1,
                        p=base_rate.flatten(),
                        replace=False,
                    )
                ),
                :,
            ],
            axis=0,
        )

        for i in range(1, n_clusters):
            # considero solo i dati di tipo i e solo i centroidi usati
            rates = PClustering.__dist_PMahalanobis(
                points,
                covs_inv,
                ctrs,
                conf_matrix[:i, :]
                / np.sum(
                    conf_matrix[:i, :], axis=1, keepdims=True
                ),  # le classi vanno considerate tutte
                mship_ref,
            )
            # calcolo dai rates la probabilità di scegliere un dato
            rates = np.min(rates, axis=1)  # (n_samples,)
            rates = np.array(
                rates / np.sum(rates, axis=0, keepdims=True), dtype=Float
            )  # (n_samples,)
            # scelgo il prossimo centroide con probabilità descritta dai rates
            ctrs = np.append(
                ctrs,
                points[
                    np.random.choice(
                        np.arange(points.shape[0]),
                        size=1,
                        replace=False,
                        p=rates.flatten(),
                    ),
                    :,
                ],
                axis=0,
            )

        return ctrs

    def fit(
        self,
        data: Data,
        max_iter: int,
        step_confidence: float,
    ) -> Tuple[Array, Array, Bools]:
        """Fit the PClustering model to the data.

        Args:
            data (Data): Data to fit.
            max_iter (int): Maximum number of iterations for the algorithm.
            step_confidence (float): Confidence level for the steps in the algorithm.

        Returns:
            tuple: A tuple with:
                - Centroids (Array): centroids of each cluster.
                - Membership (Array) : a matrix indicating the cluster assignment for each data point.
                - Mask (Array): Boolean mask indicating the data points that are not outliers.
        """

        if PClustering.verbose:
            print("PClustering representation:")
            print(f"{repr(self)}")
            print("Data representation:")
            print(f"{repr(data)}")
            print("Step confidence: {step_confidence}")

        assert max_iter > 0, f"max_iter must be greater then 0, but got {max_iter}"
        assert (
            data.n_clusters == self.n_clusters
        ), f"{data.n_clusters=} must be equal to {self.n_clusters}"

        # compute geometric threshold using data.n_dimensions
        threshold = np.array(
            chi2.ppf(self.threshold, df=data.n_dimensions), dtype=Float
        ).item()

        assert (
            threshold < data.n_dimensions
        ), f"threshold must be less than {data.n_dimensions}, but got {threshold}"

        if PClustering.verbose:
            print(f"Stochastic threshold: {threshold}")

        # Initialize centroids with KMeans++
        mask: Bools = np.ones(data.points.shape[0], dtype=Bool)
        ctrs = PClustering.__init_centroids(
            data.points[mask],
            data.covs_inv[mask],
            data.n_clusters,
            data.mship_idx[mask],
            self.conf_M,
        )
        mship_total: Array = np.zeros(
            (data.points.shape[0], data.n_clusters), dtype=Float
        )

        for step in range(max_iter):
            if PClustering.verbose:
                print(f"Iteration {step + 1}/{max_iter}")
            # calcolo le membership
            ctrs, mship, energy = PClustering.__fit_fuzzyPClustering(
                data.points[mask],
                data.covs_inv[mask],
                ctrs,
                self.conf_M,
                data.mship_idx[mask],
                self.fuzziness,
                threshold,
            )
            mship_total[mask] = mship
            if PClustering.verbose:
                print(f"Nested Clustering computes centroids with {energy=}")
                print(
                    f"Fuzzy quantity of inliers: {np.sum(np.max(mship, axis=1))} over {np.sum(mask)}"
                )

            # rilevo i dati considerabili outlier
            mask_new = PClustering.__filter_PTest(
                data.points,
                data.covs_inv,
                ctrs,
                self.p_value,
                step_confidence,
                mask,
            )
            mship_total[~mask_new] = 0.0
            if PClustering.verbose:
                print(f"Boolean quantity of inliers: {np.sum(mask_new)}")

            if np.all(mask == mask_new):
                break

            if step != max_iter - 1:
                mask = mask_new
                mship = data.mship.astype(Float)
                ctrs = PClustering.__init_centroids(
                    data.points[mask],
                    data.covs_inv[mask],
                    data.n_clusters,
                    data.mship_idx[mask],
                    self.conf_M,
                )
                mship_total = np.zeros(
                    (data.points.shape[0], data.n_clusters), dtype=Float
                )

        else:
            warnings.warn(f"Maximum number of iterations reached ({max_iter})")

        return ctrs, mship_total, mask


"""if __name__ == "__main__":
    # parametri di simulazione
    # grandezza del campo
    x_min, x_max = -90, 90
    y_min, y_max = -10, 10
    min_dist = 15
    # grandezza delle stime
    covs_size = 1
    # numero di classi
    n_clusters = 4
    fakes = 15
    # certezza della matrice di confusione
    M_confidence = 2.0
    # numero di dati
    n_samples = int(n_clusters * 30)
    fakes_n_samples = int(fakes * 15)
    noise = int((n_samples + fakes_n_samples) / 10)
    # valore che rilassa la matrice di confusione
    conf_M_relax_value = 0.05

    # matrice di confusione (M[r,c] indica la probabilità che un campione di tipo r sia classificato come c)
    M = (
        np.random.rand(n_clusters, n_clusters) * M_confidence
    )  # il coefficiente indica la convinzione della matrice
    M[np.arange(n_clusters), np.arange(n_clusters)] = np.sum(M, axis=1)
    M = np.exp(M) / np.sum(np.exp(M), axis=1, keepdims=True)
    print("matrice di confusione\n", (M * 100).astype(int) / 100)

    # quantità di fake localizzati
    fake_M = (
        np.random.rand(n_clusters, n_clusters) * M_confidence
    )  # il coefficiente indica la convinzione della matrice
    fake_M[np.arange(n_clusters), np.arange(n_clusters)] = np.sum(fake_M, axis=1)
    fake_M = np.exp(fake_M) / np.sum(np.exp(fake_M), axis=1, keepdims=True)

    # posiziono i target a terra
    targets = np.random.rand(n_clusters, 2) * np.array(
        [x_max - x_min, y_max - y_min]
    ) + [x_min, y_min]
    t_distance_check = targets[:, None, :] - targets[None, :, :]
    t_distance_check = np.linalg.norm(t_distance_check, axis=-1)
    t_distance_check[np.arange(n_clusters), np.arange(n_clusters)] = np.inf
    while np.any(t_distance_check < min_dist):
        targets = np.random.rand(n_clusters, 2) * np.array(
            [x_max - x_min, y_max - y_min]
        ) + [x_min, y_min]
        t_distance_check = targets[:, None, :] - targets[None, :, :]
        t_distance_check = np.linalg.norm(t_distance_check, axis=-1)
        t_distance_check[np.arange(n_clusters), np.arange(n_clusters)] = np.inf

    # posiziono i fake a terra
    fake_targets = np.random.rand(fakes, 2) * np.array(
        [x_max - x_min, y_max - y_min]
    ) + [x_min, y_min]

    # genero una membership random
    membership = np.random.randint(0, n_clusters, size=n_samples)
    # genero una membership random per i fake
    if fakes > 0:
        fake_classes = np.random.randint(0, n_clusters, size=fakes_n_samples)
        fake_membership = np.random.randint(0, fakes, size=fakes_n_samples)
    else:
        fake_classes = np.empty(fakes_n_samples, dtype=int)
        fake_membership = np.empty(fakes_n_samples, dtype=int)

    # genero le matrici di covarianza per ogni dato
    sigma = np.random.rand(n_samples, 2) + covs_size
    alpha = np.random.rand(n_samples) * 2 * np.pi
    U = np.moveaxis(
        np.array(
            [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]],
            dtype=Float,
        ),
        -1,
        0,
    )
    S = U * np.sqrt(sigma)[:, None, :]
    covs = np.einsum("sik, sjk -> sij", S, S)
    # genero i points
    base = np.random.randn(n_samples, 2)
    points = np.einsum("sij, sj -> si", S, base)
    points += targets[membership]

    # genero le matrici di covarianza per ogni fake
    sigma = np.random.rand(fakes_n_samples, 2) + covs_size
    alpha = np.random.rand(fakes_n_samples) * 2 * np.pi
    U = np.moveaxis(
        np.array(
            [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]],
            dtype=Float,
        ),
        -1,
        0,
    )
    S_fakes = U * np.sqrt(sigma)[:, None, :]
    covs_fakes = np.einsum("sik, sjk -> sij", S_fakes, S_fakes)
    # genero i points
    base_fakes = np.random.randn(fakes_n_samples, 2)
    points_fakes = np.einsum("sij, sj -> si", S_fakes, base_fakes)
    points_fakes += fake_targets[fake_membership]

    # genero le matrici di covarianza per ogni noise
    sigma = np.random.rand(noise, 2) + covs_size
    alpha = np.random.rand(noise) * 2 * np.pi
    U = np.moveaxis(
        np.array(
            [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]],
            dtype=Float,
        ),
        -1,
        0,
    )
    S_noise = U * np.sqrt(sigma)[:, None, :]
    covs_noise = np.einsum("sik, sjk -> sij", S_noise, S_noise)
    # genero i points
    points_noise = np.random.rand(noise, 2) * np.array(
        [x_max - x_min, y_max - y_min]
    ) + [x_min, y_min]

    classes = np.empty((n_samples, n_clusters), dtype=int)

    # genero una matrice di attribuzione propria dell'esperimento peggiorando la matrice di confusione
    M_secrets = M + conf_M_relax_value
    M_secrets = M_secrets / np.sum(M_secrets, axis=1, keepdims=True)
    for c in range(n_clusters):
        # genero interi tra 0 e n_clusters-1 con distribuzione M_secrets[c,:]
        classes[membership == c] = np.random.multinomial(
            1, M_secrets[c, :], size=np.count_nonzero(membership == c)
        )
    classes = np.argmax(classes, axis=1)

    classes_of_fakes = np.random.randint(0, n_clusters, size=fakes)
    classes_fake = np.empty((fakes_n_samples, n_clusters))
    for c in range(fakes):
        # genero interi tra 0 e n_clusters-1 con distribuzione M_secrets[c,:]
        classes_fake[fake_classes == c] = np.random.multinomial(
            1, fake_M[classes_of_fakes[c], :], size=np.count_nonzero(fake_classes == c)
        )
    classes_fake = np.argmax(classes_fake, axis=1)

    classes_noise = np.random.randint(0, n_clusters, size=noise)

    # unisco tutti i dati
    points_total = np.concatenate([points, points_fakes, points_noise], axis=0)
    covs_total = np.concatenate([covs, covs_fakes, covs_noise], axis=0)
    classes_total = np.concatenate([classes, classes_fake, classes_noise], axis=0)

    data = PClustering.Data(points_total, covs_total, classes_total, n_clusters)

    # show points
    true_memberships = np.concatenate(
        [
            membership,
            np.zeros(fakes_n_samples, dtype=Int) - 1,
            np.zeros(noise, dtype=Int) - 1,
        ],
        axis=0,
    )
    membership_boolean = np.zeros((points_total.shape[0], n_clusters), dtype=Float)
    membership_boolean[np.arange(points.shape[0]), membership] = 1

    fig, ax = PClustering.plotPClustering(
        points_total,
        targets,
        classes_total,
        data.covs_inv,
        membership_boolean,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    # save the confusion matrix
    print("centroidi soluzione")
    print(targets)
    np.save("conf_M.npy", M)
    # save the data
    np.save("points.npy", data.points)
    np.save("covs.npy", data.covs)
    np.save("classes.npy", data.mship_idx)

    # per ridurre il numero di iterazioni (e quindi aumentare la velocità):
    # 1) ridurre il p_value, questo ha come effetto cluster finali più grandi ma anche più dati rimovibili per step
    # 2) ridurre il threshold, questo la compatibilità semantica tra il clustering annidato e il PTest
    # 3) ridurre la step_confidence, questo aumenta il numero massimo di dati eliminati in uno step
    # per ridurre le allucinazioni (quindi cluster fake visti come veri o persino cluster persi):
    # 1) aumentare il p_value, questo permette più iniziativa nel PTest in quanto i dati eliminati sono scelti tra molti di più
    # 2) aumentare la fuzziness, questo permette di fondere meglio fake cluster isolati (che non hanno un raggio sufficientemente ampio) e noise
    # 3) aumentare il threshold, potrebbe capitare che un threshold troppo basso non permetta di distinguere cluster veri dal noise
    # 4) aumentare la step_confidence, questo permette di ridurre il tasso di errore in ogni step
    # NOTE: una fuzziness troppo alta rischia di rendere instabile il clustering annidato oltre a ridurre l'iniziativa.
    ctrs, mship_boolean, mask = PClustering(
        M,
        p_value=0.005,  # probabilità di errore nell'espulsione di un dato
        fuzziness=8.0,  # non è un problema avere dati distanti assenti
        threshold=0.7,  # soglia stocastica per comportamenti booleani, tanto è più alta tanto più sono comuni dati con membership booleana
    ).fit(
        data,
        max_iter=1000,
        step_confidence=0.95,  # probabilità complessiva di corretta rimozione dei dati
    )

    print(f"outliers (%): {np.count_nonzero(~mask) / mask.shape[0] * 100}")

    fig, ax = PClustering.plotPClustering(
        points_total[mask],
        ctrs,
        classes_total[mask],
        data.covs_inv[mask],
        mship_boolean.astype(Float)[mask],
        x_min,
        x_max,
        y_min,
        y_max,
    )
    plt.show()
"""
