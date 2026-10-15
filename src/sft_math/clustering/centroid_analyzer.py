# procedura del decision system:
# 1. Per prima cosa si pescano i centroidi tra i dati fino al massimo descritto da un batch
#    questo fa sì che vengano considerati quanti più punti possibili e che, in caso di limiti,
#    vengano considerati con più probabilità i punti non isolati
# 2. Poi ogni dato si pone come "vicino" di un centroide in base ad un threshold.
#    Si nota che ogni centroide ha almeno un dato vicino (essendo lo stesso un dato)
# 3. controllo che il punto valutato abbia un numero sufficiente di vicini:
#    questo criterio è dinamico, ossia fa sempre passare gli outlier di punti per rappresentanza
#    almeno un centroide dovrà passare il primo filtro e sarà il più rappresentato
# 4. faccio sì che il punto che supera il primo filtro sia rettificato come media dei punti che lo circondano
#    il punto proposto dovrebbe già superare il test di centralità per costruzione
#    ma non è detto che superi il test di varianza, pertanto si esegue un test ulteriore
#    se il punto produce un valore troppo raro verrà escluso


from sft_math.clustering.data import Data as Data
from sft_math.class_templates import ClassBase
from sft_math.typing import Array, Float
from scipy.stats import chi2, norm
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


def plot(
    data: Data,
    selected: Array,
    centroids: Array,
    covs_p: float = 0.80,
) -> None:
    """Plot the clustering results."""

    fig, ax = plt.subplots()
    for i in range(data.n_samples):
        # calcolo l'ellisse
        S = np.linalg.cholesky(data.covs[i])
        vals, vecs = np.linalg.eig(S)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ell = Ellipse(
            data.points[i],
            width=vals[0] * chi2.ppf(covs_p, df=data.n_dimensions) * 2,
            height=vals[1] * chi2.ppf(covs_p, df=data.n_dimensions) * 2,
            angle=angle,
            color="gray",
            alpha=0.05,
        )
        ax.add_patch(ell)
    # stabilisco dei colori per ogni simulazione
    ax.scatter(
        data.points[:, 0],
        data.points[:, 1],
        c="blue",
        s=1,
        alpha=0.5,
        label="Data Points",
    )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color="red",
        marker="x",
        s=8,
        label="Centroids",
    )
    ax.scatter(
        selected[:, 0],
        selected[:, 1],
        color="green",
        marker="x",
        s=10,
        label="Selected Points",
    )
    ax.set_title(f"Clustering Results")
    ax.legend()
    fig.show()


class CentroidAnalyzer(ClassBase):

    horizon: float  # it is a p value, so a value between 1 (accepts all points) and 0 (rejects all points)
    filter: float  # it is a filter by quantity of points, if it is 0 it accepts the centroid

    n_dimensions: (
        int  # number of dimensions of the data points, it is set during the call
    )
    centroids: Array  # centroids of the clusters, it is set during the call
    probs: Array  # probabilities of the centroids, it is set during the call

    __constants__ = [
        (lambda x: int(x * 10000) / 100, "horizon", "%"),
        (lambda x: x, "filter", ""),
        # parameters that are not set during initialization
        (lambda x: x, "n_dimensions", ""),
        (lambda x: getattr(x, "tolist"), "centroids", ""),
    ]

    def __init__(
        self,
        horizon: float,
        filter: float,
    ):
        super().__init__()
        assert 0 <= horizon <= 1, "horizon must be a value between 0 and 1"
        assert 0 <= filter < 1, "filter must be a value between 0 and 1"

        self.horizon = horizon
        self.filter = filter

    @property
    def radius(self) -> float:
        return chi2.ppf(self.horizon, df=self.n_dimensions).item()

    def __call__(
        self,
        data: Data,
        batch_size: int,
        passages: int,
        feedback: Array | None = None,
    ):
        self.n_dimensions = data.n_dimensions

        if feedback is None:
            self.centroids = np.zeros((0, self.n_dimensions), dtype=Float)
        else:
            self.centroids = np.array(feedback, dtype=Float).reshape(
                -1, self.n_dimensions
            )
        self.centroids = self.centroids[
            np.random.choice(
                len(self.centroids),
                size=min(int(len(self.centroids) * 2 / 3), batch_size),
                replace=False,
            )
        ]  # 2/3 fa sì che ci sia sempre un po' di iniziativa

        # seleziono i centroidi come un batch da esaminare
        batch_size = min(
            max(batch_size, len(self.centroids)) - len(self.centroids), data.n_samples
        )

        self.centroids = np.concatenate(
            (
                self.centroids,
                data.points[
                    np.random.choice(data.n_samples, size=batch_size, replace=False)
                ],
            ),
            dtype=Float,
            axis=0,
        )

        # ogni centroide si posizione al centro di una regione sufficientemente ricca di dati
        for _ in range(passages):
            # differenze euclidee tra punti e centroidi
            diff = (
                data.points[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
            )  # shape (n_samples, n_centroids, n_dimensions)
            # distanze di mahalanobis
            distances = np.einsum("sci, sij, scj -> sc", diff, data.covs_inv, diff)

            # creo la maschera degli orizzonti
            horizon_mask = distances <= self.radius  # shape (n_samples, n_centroids)
            # affinché la teoria funzioni, devo assicurarmi che ci siano almeno self.filter punti
            counter = np.sum(horizon_mask, axis=0)  # shape (n_centroids,)
            mask = counter >= self.filter * np.max(counter)  # shape (n_centroids,)
            self.centroids = self.centroids[
                mask
            ]  # filtro i centroidi che non hanno abbastanza punti
            distances = distances[:, mask]  # filtro le distanze
            horizon_mask = horizon_mask[:, mask]  # filtro la maschera degli orizzonti
            # calcolo i centroidi proposti da questi punti (rispetto della condizione di centralità)
            sum_cov_inv = np.einsum("sij, sc -> cij", data.covs_inv, horizon_mask)
            sum_vec = np.einsum(
                "sij, si, sc -> cj", data.covs_inv, data.points, horizon_mask
            )
            self.centroids = np.array(
                np.linalg.solve(sum_cov_inv, sum_vec), dtype=Float
            )  # (n_centroids, n_dimensions)

        # eseguo il test chi2 centralizzato
        diff = (
            data.points[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        )  # shape (n_samples, n_centroids, n_dimensions)
        distances = np.einsum("sci, sij, scj -> sc", diff, data.covs_inv, diff)

        total = np.sum(horizon_mask * distances, axis=0)  # shape (n_centroids,)
        sqrt_counter = np.sqrt(np.sum(horizon_mask, axis=0))  # shape (n_centroids,)
        evals = (
            total / sqrt_counter - chi2.mean(self.n_dimensions) * sqrt_counter
        ) / np.sqrt(
            chi2.var(df=self.n_dimensions)
        )  # shape (n_centroids,)
        # le probabilità dei centroidi sono le confidenze dei punti
        self.probs = np.array(norm.sf(evals), dtype=Float)  # shape (n_centroids,)
        # filtro i centroidi in base alla probabilità
        mask = self.probs >= (self.probs.max() * 0.2)
        self.centroids = self.centroids[mask]
        self.probs = self.probs[mask]
