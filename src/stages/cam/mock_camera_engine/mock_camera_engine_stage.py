import datetime
import multiprocessing as mp
import time
from typing import Optional, Tuple
#from shapely.geometry import Point, Polygon
#import pypylon.pylon as pylon
import numpy as np
from numpy import _typing
from common.config import SFTConfig
from core.logger import SFTLogger
from common.pipeline_stage import PipelineStage
from core.central_registry import CentralRegistry
from common.new_structs import PhotoExif

# saver
import csv


class CameraEngine(PipelineStage):
    def __init__(
        self,
        configs: SFTConfig,
        logger: SFTLogger,
        registry: CentralRegistry,
        stage_name: str,

    ) -> None:
        """__init__ CameraEngine

        Class that is responsible for the acquisition of pictures from the jetson's camera

        The process continuously captures input from the camera and deposits handles to the images
        it in the output queue; it logs and saves the pictures to disk.

        configs : SFTConfig
            A configuration object, exposes arbitrary configs written in sft_config.yaml for
            use in the program
        out_q : mp.Queue
            A queue that will be used to deposit the handles of the pictures

        """
        super().__init__(
            configs,
            logger,
            registry=registry,
            stage_name=stage_name
        )
        self.stopped = mp.Event()  # flag used to gently stop acquisition
        self.stopped.clear()
        self.is_initialized = mp.Event()
        self.config = configs
        self.logger = logger
        self.logger.warn("CAMERA ENGINE IS ALIVE!!!")

    def mission(self) -> bool:
        """run Get frames from the camera

        Returns
        -------
        None
            Does not return anything, the frames are put in the output queue as a
            TaggedPicture struct
        """

    def stop(self) -> None:
        """stop Stops the camera engine

        Sets the stop flag, the camera engine will stop after the current
        acquisition is finished
        """
        if self.is_initialized.is_set():
            self.logger.info("Setting stop flag on CameraEngine")
            self.stopped.set()
            self.in_mission.clear(self.idx)
            self.to_update.clear(self.idx)

            self.logger.info(f"Captured {self.img_count} images")
            self.logger.info(
                "Quitting camera connection and shutting down gracefully"
            )
            self.cam.Close()
            self.f.flush()
            self.logger.info("Camera connection closed")
            return True
        else:
            self.logger.error(
                "Trying to stop a non-constructed CameraEngine, check if the camera is plugged in."
            )

    

    def update(self) -> None:
        super().update()
        
        return None

    def put(self, metadata: PhotoExif) -> None:
        pass

    def get(self) -> None:
        raise NotImplementedError

    def _dump_data(self) -> None:
        pass
