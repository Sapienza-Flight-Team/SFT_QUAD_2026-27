from sft_math.clustering import KMeans, CMeans
from sft_math.clustering.class_templates import DataBase
from sft_math.typing import Array
import numpy as np
import matplotlib.pyplot as plt


def data_plot(
    data: DataBase,
    ctrs: Array | None = None,
):
    if isinstance(data, KMeans.Data):
        kdata = KMeans.Data.from_base(data)
        plt.clf()
        plt.scatter(kdata.points[:, 0], kdata.points[:, 1], c="blue", s=2)
        if ctrs is not None:
            plt.scatter(
                ctrs[:, 0], ctrs[:, 1], c="red", marker="x", s=5, label="Centroids"
            )
        plt.title("Data Points with Centroids")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.show()


def feedback_batch(
    data: DataBase,
    model: KMeans,
    helper: CMeans,
    mode: str,
    **kwargs,
):
    # in questa modalità ogni dato ha una certa probabilità di essere scelto
    model.fit(
        data,
        max_iter=int(kwargs.get("max_iter", 1000)),
        tol=float(kwargs.get("tol", 1e-4)),
        ctrs=kwargs.get("ctrs", None),
    )
    data_plot(data, model.ctrs)

    for i in range(10):
        helper.fit(
            data,
            max_iter=int(kwargs.get("max_iter", 1000)),
            tol=float(kwargs.get("tol", 1e-4)),
            ctrs=model.ctrs,
        )
        p = helper.data_existence
        mask = np.random.rand(getattr(data, "n_samples", 0)) < p
        curr_data = data[mask]
        model.fit(
            curr_data,
            max_iter=int(kwargs.get("max_iter", 1000)),
            tol=float(kwargs.get("tol", 1e-4)),
            ctrs=helper.ctrs,
        )
        data_plot(curr_data, model.ctrs)


n_centroids = 4
n_data = 10000
centroids = np.random.rand(n_centroids, 2)
error = np.random.randn(n_data, 2)
scale = 0.07
indices = np.random.choice(centroids.shape[0], n_data, replace=True)
points = centroids[indices] + error * scale
data = KMeans.Data(points)
Kmodel = KMeans(n_clusters=n_centroids)
Cmodel = CMeans(n_clusters=n_centroids, fuzziness=2.0, threshold=0.0)
feedback_batch(
    data=data, model=Kmodel, helper=Cmodel, mode="random", max_iter=1000, tol=1e-4
)
