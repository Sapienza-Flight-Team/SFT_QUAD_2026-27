from multiprocessing.queues import Queue
from typing import Any, List
from common.config import SFTConfig
from core.logger import SFTLogger
from common.multi_event import MultiEvent
from stages.decision_system.default import DecisionSystem
import pandas as pd
import numpy as np
from sft_math.geo.polar import GPS


class Mock(DecisionSystem):

    metadata: List[pd.DataFrame]
    index: int = 0
    metadata_headers: List[str] = [
        "lat",
        "lon",
        "h",
        "psi",
        "phi",
        "theta",
        "photo",
        "x",
        "y",
        "classification",
    ]

    def __init__(
        self,
        configs: SFTConfig,
        logger: SFTLogger,
        in_mission: MultiEvent,
        to_update: MultiEvent | None,
        in_q: Queue | None,
        out_q: List[Queue] | None,
        idx: int,
    ) -> None:
        super().__init__(configs, logger, in_mission, to_update, in_q, out_q, idx)
        merged_metadata = pd.read_csv(
            self.configs.config_dict["decision_system"]["mock"]["metadata_csv"]
        )
        # trasformo i metadata in una lista di metadata per foto
        if "photo" not in merged_metadata:
            raise self.Failure("photo must be a column of the metadata")
        self.metadata = []
        n_photo = max(merged_metadata["photo"])
        for i in range(n_photo):
            if i in merged_metadata["photo"].iloc:
                self.metadata.append(
                    merged_metadata.iloc[
                        np.where(merged_metadata["photo"] == i)[0]
                    ].reset_index(drop=True)
                )

        self.index = 0

    def get(self) -> None:
        """get data from the previous stage, could be messages between queues or files
        saved on disk, can be different approach from put
        """
        _: Any = self.receive_from_queue()

        if self.index == -1:
            return

        metadata: pd.DataFrame
        if self.index < len(self.metadata):
            metadata = self.metadata[self.index]
            self.index += 1
        else:
            self.index = -1
            return

        if not all(
            [col in metadata.columns.to_list() for col in self.metadata_headers]
        ):
            self.logger.error(
                f"metadata headers are: {self.metadata_headers}, but metadata columns are: {metadata.columns}"
            )
            raise DecisionSystem.Failure(
                f"metadata headers are not the same as serie columns, but they are: {metadata.columns}"
            )

        # inserisco il filename nella dupla come ultima componente
        self.outs = (np.empty((), dtype=np.float64), metadata["photo"].iloc[0])
        self.points, self.covariances, _ = self.geoloc(
            np.array(
                [
                    metadata["x"],
                    metadata["y"],
                ],
                dtype=np.float64,
            ).T,
            GPS(
                metadata.iloc[0]["lat"],
                metadata.iloc[0]["lon"],
            ),
            metadata.iloc[0]["h"],
            metadata.iloc[0]["phi"],
            metadata.iloc[0]["theta"],
            metadata.iloc[0]["psi"],
        )
        self.classes = self.class_handler(
            np.array(metadata["classification"], dtype=np.int64)
        )
