from abc import ABCMeta, abstractmethod
from logging import getLogger
from multiprocessing import Event, Process
from typing import Any

from common.new_structs import DATA_TYPE_REGISTRY
from core.central_registry import CentralRegistry
from core.logger import SFTLogger


class PipelineStage(Process, metaclass=ABCMeta):
    """Stage lifecycle and leases for the subscriptions registered by Pipeline."""

    _stage_name: str

    def __init__(
        self,
        configs: dict[str, Any],
        logger: SFTLogger,
        registry: CentralRegistry,
        stage_name: str,
    ) -> None:
        """Store this stage's flat settings and the registry supplied by Pipeline.

        Starting the stage attaches its registered leases to the registry;
        payloads stay in shared memory and metadata stays with the owner.
        """
        self.configs = configs
        self.logger = logger if logger is not None else getLogger("sft_cv.stage")
        self._stage_name = stage_name
        self.registry = registry
        self._registry_client = None
        self.stopped = Event()
        self.stopped.clear()
        super().__init__()

    def start(self):
        """Attach to the registry before starting this independent process."""
        owner = self.registry
        client = owner.for_stage(self._stage_name, self.stopped)
        self.registry = self._registry_client = client
        try:
            super().start()
        finally:
            self.registry = owner
            client.close()  # The child owns the copied endpoints.
            self._registry_client = None

    @property
    def current_task(self) -> str:
        return self._current_task

    @current_task.setter
    def current_task(self, task: str) -> None:
        self._current_task = task

    def lease_slot_for_reading(self, datatype):
        """Wait for the latest primitive/group; keep returned views in the lease.

        Accept a datatype class, a set of classes, or a registered stream name.
        The returned mapping contains read-only views keyed by primitive class.
        """
        if isinstance(datatype, str):
            datatype = DATA_TYPE_REGISTRY[datatype]["data_type"]
        return self.registry.lease_slot_for_reading(datatype, self._stage_name)

    def lease_slot_for_writing(self, datatype):
        """Lease one primitive by class or stream name; fill its ID before exit."""
        if isinstance(datatype, str):
            datatype = DATA_TYPE_REGISTRY[datatype]["data_type"]
        return self.registry.lease_slot_for_writing(datatype)

    def run(self) -> None:
        """Run mission until it returns True; always finish stage cleanup."""
        import os

        try:
            self.update()
            self.logger.info(f"[{self._stage_name}] Process started (pid={os.getpid()})")
            graceful_stop = False
            while not graceful_stop:
                graceful_stop = self.mission()
            self.logger.info(f"[{self._stage_name}] Mission loop exited (graceful={graceful_stop})")
        except (EOFError, BrokenPipeError):
            if not self.stopped.is_set():
                self.logger.exception("[%s] Registry connection failed", self._stage_name)
                raise
        except Exception:
            self.logger.exception("[%s] Stage failed", self._stage_name)
            raise
        finally:
            try:
                self._dump_data()
            finally:
                if self._registry_client is not None:
                    self._registry_client.close()
                self.logger.info(f"[{self._stage_name}] Process shutting down")

    @abstractmethod
    def mission(self) -> bool:
        """Process registry data using with-block leases; return True to exit.

        Write primitives directly into their leased buffers and keep every
        derived array or view inside its lease. Check stopped between leases.
        """
        raise NotImplementedError()

    @abstractmethod
    def stop(self) -> None:
        """Request mission shutdown and perform stage-specific cleanup."""
        raise NotImplementedError()

    @abstractmethod
    def _dump_data(self) -> None:
        """Flush stage-owned results after the mission loop exits."""
        raise NotImplementedError()

    def set_mission(self, mission: str) -> None:
        if self._current_task != mission:
            self._current_task = mission
            self.update()

    def update(self) -> None:
        """Override to refresh stage resources from the supplied self.configs."""
        pass
