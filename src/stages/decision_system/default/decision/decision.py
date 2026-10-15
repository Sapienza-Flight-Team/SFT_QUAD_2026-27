from stages.decision_system.default.class_templates import DecisionTemplate
from typing import Dict, Optional, Any
from logging import Logger
import numpy as np
from numpy import typing as np_typing
from sft_math.clustering import CentroidAnalyzer
from sft_math.clustering import Data
from sft_math.typing import Array

from sft_math.clustering.centroid_analyzer import plot


class Decision(DecisionTemplate):

    percentile: float
    n_clusters: int
    data: Data
    analyzer: CentroidAnalyzer
    feedback: Array | None

    def __init__(self, logger: Logger, configs: Dict[str, Optional[Any]]) -> None:
        self.logger = logger
        self.configs = configs

        if "horizon" not in configs:
            raise Decision.Failure("horizon is required in configs")
        if "filter" not in configs:
            raise Decision.Failure("filter is required in configs")
        if not type(configs["horizon"]) is float:
            raise Decision.Failure(
                f"horizon must be a float, but got {type(configs['horizon'])}"
            )
        if not type(configs["filter"]) is float:
            raise Decision.Failure(
                f"filter must be a float, but got {type(configs['filter'])}"
            )

        # make the analyzer
        self.analyzer = CentroidAnalyzer(
            horizon=configs["horizon"],  # stabilisce l'orizzone
            filter=configs[
                "filter"
            ],  # stabilisce il taglio, prende il cluster più denso e ne prende una percentuale di esso come minimo
        )

        self.data = Data(
            points=np.zeros((0, 2), dtype=np.float64),
            covariances=np.zeros((0, 2, 2), dtype=np.float64),
        )

        self.feedback = None

    def insert(
        self,
        points: np_typing.ArrayLike,
        covariances: np_typing.ArrayLike,
    ) -> None:
        """
        Insert points and covariances into the data for clustering.
        """
        self.data = self.data.append(
            Data(
                points=np.array(points, dtype=np.float64),
                covariances=np.array(covariances, dtype=np.float64),
            )
        )

    def subset(self, mask: np_typing.ArrayLike) -> "Decision":
        """
        Return a new Decision sharing this object's configuration but holding
        only the subset of accumulated data selected by `mask`.

        Used by the instance splitter to treat spatially separated detections
        of the same class as independent instances before clustering. The
        returned Decision starts with no clustering feedback.
        """
        sub = Decision(self.logger, self.configs)
        sub.data = self.data[mask]
        return sub

    def fit(
        self,
        n_clusters: int,
    ) -> np_typing.NDArray[np.float64]:

        if self.data.points.size == 0:
            raise Decision.Failure(
                "No data points available for clustering. Please insert data before fitting."
            )

        self.analyzer(self.data, batch_size=600, passages=4, feedback=self.feedback)

        if self.analyzer.centroids.size == 0:
            self.logger.warning("No centroids found, returning empty array.")
            return np.zeros((0, 2), dtype=np.float64)
        else:
            self.feedback = self.analyzer.centroids
            if n_clusters <= 0:
                return np.zeros((0, 2), dtype=np.float64)
            m = 2  # ? più è alto questo valore meno è rumoroso (puntando maggiormento verso il massimo)
            glob_probs = (self.analyzer.probs**m) / np.sum(self.analyzer.probs**m)
            # Sort centroids by probability descending and select up to n_clusters unique centroids
            sorted_indices = np.argsort(glob_probs)[::-1]
            top_indices = sorted_indices[:n_clusters]
            selected_centroids = self.analyzer.centroids[top_indices]
            # plot(self.data, selected_centroids.reshape(-1, 2), self.analyzer.centroids)
            return selected_centroids
