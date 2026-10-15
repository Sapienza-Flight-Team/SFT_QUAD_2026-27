from multiprocessing.synchronize import Event as EventClass
from multiprocessing.queues import Queue as QueueClass
from common.config import SFTConfig
from common.multi_event import MultiEvent
from common.pipeline_stage import PipelineStage
import multiprocessing as mp


class StageNetturbino(PipelineStage):
    def __init__(
        self,
        configs: SFTConfig,
        in_mission: MultiEvent | None,
        in_q: QueueClass | None,
        updated: MultiEvent,
        idx: int,
    ) -> None:
        """This is the main constructor for the stage. You should retrieve all
        the necessary configs from the `configs` object, and initialize any
        necessary state for the stage. The `in_q` and `out_q` are the input and
        output queues respectively. You should be consuming the necessary data
        from `in_q` and producing your results to `out_q`.

        Args:
            configs (SFTConfig): The configuration object
            in_q (Queue): The queue with your input data

        Raises:
            NotImplementedError: This is an abstract method, and should be implemented.
            Otherwise, you will get this error.
        """

        super().__init__(
            configs=configs,
            in_mission=in_mission,
            in_q=in_q,
            out_q=None,
            updated=updated,
            idx=idx,
        )
        self.stopped: EventClass = mp.Event()
        self.logger = configs.logger
        self.in_q = in_q

    def mission(self) -> bool:
        assert (
            self.in_q is not None
        ), "PER QUALE MOTIVO DOVRESTI USARE UN NETTURBINO SENZA UNA CODA?"
        while not self.stopped.is_set():
            if self.in_q.qsize() or not self.in_q.empty():
                data = self.in_q.get()
                self.logger.info(data)
                del data

        return True

    def stop(self) -> None:
        self.stopped.set()
