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
from common.structs import PhotoExif, Waypoint
#from common.structs import TeensyData, unpack
from common.pipeline_stage import PipelineStage
import os
from sft_math.geo.utils import magnorth2truenorth
from core.central_registry import CentralRegistry

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
        self.config_dict = configs.config_dict
        self.yaml_path = self.config_dict["utils"]["yaml_path"]
        self.logger.warn("Camera process initialized!")

    def mission(self) -> bool:
        """run Get frames from the camera

        Returns
        -------
        None
            Does not return anything, the frames are put in the output queue as a
            TaggedPicture struct
        """
        self.cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        self.cam.Open()
        self.setup_cam()
        self.cam.StartGrabbing()
        self.img_count = 0
        while True:
            while (
                self.in_mission.is_set(self.idx)
                and self.cam.IsGrabbing()
                and not self.to_update.is_set(self.idx)
            ):
                try:
                    ret = self.acquire_taggedpic()
                except Exception as e:
                    self.logger.warning(
                        f"Error {type(e).__name__}: {e} encountered while acquiring image"
                    )
                    ret = None
                    continue

                image, metadata = ret
                self.img_count += 1


            if self.to_update.is_set(self.idx):
                self.update()

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

    def setup_cam(
            self,
        ) -> None:

        """setup_cam
        Deferred initialization of the camera, this is done to avoid
        the camera being initialized before the process is spawned,
        resulting in incompatible address spaces.
        """
        if self.is_initialized.is_set():
            self.logger.warning("Camera already initialized")
            return

        start = time.time()
        self.logger.info("Configuring camera parameters")
        self.profile = self.config_dict["camera"]["profile"]
        self.cam.TriggerSelector = "FrameStart"
        self.cam.TriggerSource.Value = "Line1"
        self.cam.TriggerActivation = "RisingEdge"
        self.cam.TriggerMode.Value = "On"
        self.cam.Gamma = self.config_dict["camera"]["gamma"]
        self.cam.BslColorSpace = self.config_dict["camera"]["bsl_color_space"]
        self.cam.BlackLevel = self.config_dict["camera"]["black_level"]
        self.cam.BslLightSourcePreset.Value = self.config_dict["camera"][
            "light_source_preset"
        ]
        self.cam.AutoGainUpperLimit = self.config_dict["camera"]["profile"][
            "max_auto_gain"
        ]
        self.cam.PixelFormat = self.config_dict["camera"]["pixel_format"]
        self.cam.BalanceWhiteAuto = self.config_dict["camera"]["balance_white_auto"]
        self.cam.GainAuto = self.config_dict["camera"]["auto_gain"]
        self.cam.AutoGainLowerLimit = self.config_dict["camera"]["profile"][
            "min_auto_gain"
        ]
        self.cam.ExposureMode = self.config_dict["camera"]["exposure_mode"]
        self.cam.ExposureAuto = self.config_dict["camera"]["exposure_auto"]
        self.cam.AutoExposureTimeLowerLimit = self.config_dict["camera"]["profile"][
            "min_exposure_time"
        ]
        self.cam.AutoExposureTimeUpperLimit = self.config_dict["camera"]["profile"][
            "max_exposure_time"
        ]
        self.cam.AutoTargetBrightness = self.config_dict["camera"]["profile"][
            "auto_target_brightness"
        ]
        
        self.cam.LineSelector.Value = "Line2"
        self.cam.LineMode.Value = "Output"
        self.cam.LineSource.Value = "ExposureActive"
        self.cam.LineInverter.Value = False

        self.cam.MaxNumBuffer.Value = 10
        self.cam.OutputQueueSize.Value = 10

        try:
            self.cam.ChunkModeActive.Value = True
            self.cam.ChunkSelector.Value = "FrameID"
            self.cam.ChunkEnable.Value = True
            self.logger.info(
                "CAMERA CHUNK CONFIGURED: ChunkModeActive=True, "
                "ChunkSelector=FrameID, ChunkEnable=True"
            )
        except Exception as e:
            self.logger.warning(f"Could not configure FrameID chunk: {e}")

        self.logger.info("Camera setup done")

    def acquire_taggedpic(
        self,
    ) -> Optional[Tuple[np.ndarray, PhotoExif]]:
        """start_acquisition Acquire a tagged picture from the camera

        Returns
        -------
        TaggedPicture
            A struct containing a pair (RawImage, Waypoint)
        """
        start = time.time()
        grabResult = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            stop = time.time()
            self.logger.debug(f"Image acquired in {stop - start} seconds")
            img: _typing.NDArray[np.uint8] = grabResult.Array
            stop = time.time() - start
            self.logger.debug(f"to RAM took {stop} seconds")

            timestamp = time.time()
            metadata = PhotoExif(
                filename=os.path.join(
                    self.out_dir, f"{self.img_count}_{self.initial_timestamp}.npy"
                ),
                exposure_time=self.cam.ExposureTime.Value,
                timestamp=timestamp,
                gain=self.cam.Gain.Value,
                black_level=self.cam.BlackLevel.Value,
                bsl_color_space=self.cam.BslColorSpace.Value,
                gamma=self.cam.Gamma.Value,
                light_source_preset=self.cam.BslLightSourcePreset.Value,
                pixel_format=self.cam.PixelFormat.Value,
            )

            stop = time.time() - start
            self.logger.debug(f"Acquisition took {stop} seconds \n")
            return img, metadata
        raise

    def update(self) -> None:
        super().update()
        self.config_dict = self.configs.config_dict
        if not self.is_initialized:
            self.cam.StopGrabbing()
            while True:
                try:
                    self.cam.RetrieveResult()
                except Exception:
                    break
            self.is_initialized.clear()
            self.setup_cam()
            self.ready = True
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.logger.info("Camera updated")
        return None

    def put(self, metadata: PhotoExif) -> None:
        if self.out_q is None:
            self.logger.error("Output queue is None, cannot put metadata")
            return
        self.send_to_queues(metadata)

    def get(self) -> None:
        raise NotImplementedError

    def _dump_data(self) -> None:
        pass
