import datetime
import multiprocessing as mp
import time
from typing import Optional, Tuple, Any, TextIO
from shapely.geometry import Point, Polygon
import pypylon.pylon as pylon
import numpy as np
from numpy import _typing
from common.config import SFTConfig
from core.logger import SFTLogger
from common.structs import PhotoExif, Waypoint
from common.pipeline_stage import PipelineStage
import os
from common.multi_event import MultiEvent
from sft_math.geo.utils import magnorth2truenorth

# saver
import csv


class PixhawkDataSaver(PipelineStage):
    def __init__(
        self,
        configs: SFTConfig,
        logger: SFTLogger,
        in_mission: MultiEvent,
        in_q: Optional[mp.Queue],
        out_q: Optional[list[mp.Queue]],
        idx: int,
        to_update: MultiEvent,
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
            in_mission,
            out_q=out_q,
            in_q=None,
            idx=idx,
            to_update=to_update,
        )
        self.stopped = mp.Event()  # flag used to gently stop acquisition
        self.stopped.clear()
        self.in_mission = in_mission
        self.to_update = to_update
        self.is_initialized = mp.Event()
        self.in_q = in_q
        self.out_q = out_q
        self.config = configs
        self.logger = logger
        self.idx = idx
        self.config_dict = configs.config_dict
        self.yaml_path = self.config_dict["utils"]["yaml_path"]
        self.emulatedEnvironment = self.config_dict["camera"]["emulated"]
        self.out_dir = configs.config_dict["camera"]["img_out_dir"]
        self.csv_path = self.config_dict["camera"]["csv_out_dir"]
        self.mode = self.config_dict["picture_saver"]["saving_mode"]
        self.airdrop_bound = self.config.config_dict["airdrop_bound"]
        self.mapping_bound = self.config.config_dict["mapping_bound"]
        if self.out_q:
            self.camera_missions = tuple(
                range(1, len(self.out_q) + 1)
            )  # mapping and inference
        else:
            self.camera_missions = tuple(range(1, 4))
        # TODO: adjust the dimensions of the polygon based on:
        #       - the vertical field of view of the camera
        #       - the speed of the aircraft
        self.geo_fence = Polygon(
            [
                self.airdrop_bound["bl"],
                self.airdrop_bound["br"],
                self.airdrop_bound["tr"],
                self.airdrop_bound["tl"],
            ]
        )
        self.mapping_geo_fence = Polygon(
            [
                self.mapping_bound["bl"],
                self.mapping_bound["br"],
                self.mapping_bound["tr"],
                self.mapping_bound["tl"],
            ]
        )
        # initial timestamp of image handles
        self.initial_timestamp = (
            datetime.datetime.fromtimestamp(time.time())
            .isoformat(timespec="minutes", sep="-")
            .replace(":", "-")
        )

        ######### SAVER
        # Use simple attribute paths (dot-separated) instead of lambdas so the
        # CameraEngine instance is picklable when processes are started with
        # the 'spawn' method. The values are resolved at save time.
        self.METADATA_COLUMNS = {
            "filename": "filename",
            "timestamp": "timestamp",
            "exposure_time": "exposure_time",
            "gain": "gain",
            "black_level": "black_level",
            "bsl_color_space": "bsl_color_space",
            "gamma": "gamma",
            "light_source_preset": "light_source_preset",
            "pixel_format": "pixel_format",
            "teensy_timestamp": "pix_data.timestamp",
            "lat": "pix_data.lat",
            "lon": "pix_data.lon",
            "h": "pix_data.h",
            "vground": "pix_data.vground",
            "roll": "pix_data.roll",
            "pitch": "pix_data.pitch",
            "yaw": "pix_data.yaw",
            "heading": "pix_data.heading",
            "mission_mode": "pix_data.mission_mode",
            "mission_state": "pix_data.mission_state",
        }
        self.saved_img_count = 0
        self.out_dir = configs.config_dict["camera"]["img_out_dir"]
        self.out_metadata = configs.config_dict["picture_saver"][
            "metadata_db"
        ]  # csv file with metadata
        db_creation_time = time.strftime("%Y%m%d-%H%M%S")
        # append the hash to the file base name
        self.out_metadata = self.out_metadata.replace(
            ".csv", f"_{db_creation_time}.csv"
        )
        self.first_pic_datetime = (
            datetime.datetime.fromtimestamp(time.time())
            .isoformat(timespec="minutes", sep="-")
            .replace(":", "-")
        )

        # Do not open the file here (parent process). The file handle will be
        # opened in the child process inside `setup_cam` to avoid pickling
        # non-serializable objects when using the 'spawn' start method.
        # annotate types for static type checkers
        self.f: Optional[TextIO] = None
        # csv.writer has no stable public type, use Any for the writer
        self.writer: Optional[Any] = None
        self._metadata_initialized = False

    def mission(self) -> bool:

        self.setup_cam()
        self.img_count = 0
        while True:

            try:
                ret = self.acquire_taggedpic()
            except Exception as e:
                self.logger.warning(
                    f"Error {type(e).__name__}: {e} encountered while acquiring image"
                )
                ret = None
                continue

            if not ret:  # to be expected when photoes are taken outside of the geofence
                # self.logger.warning("Tagged pic returned as None")
                continue

            image, metadata = ret

            self.save(image, metadata)
            self.put((metadata))

            self.img_count += 1

            if self.stopped.is_set():
                self.logger.info(f"Captured {self.img_count} images")
                if self.f is not None:
                    try:
                        self.f.flush()
                    except Exception:
                        pass
                self.logger.info("Camera connection closed")
                return True

            if self.to_update.is_set(self.idx):
                self.update()

    def setup_cam(self):
        try:
            if not getattr(self, "_metadata_initialized", False):
                out_dir = os.path.dirname(self.out_metadata)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                # Open CSV for writing metadata
                self.f = open(self.out_metadata, mode="w", newline="", encoding="utf-8")
                self.writer = csv.writer(self.f)
                # write header
                try:
                    self.writer.writerow(list(self.METADATA_COLUMNS.keys()))
                except Exception:
                    pass
                self._metadata_initialized = True
        except Exception:
            # best-effort: don't crash the camera setup if metadata writer can't be created
            self.logger.warning(
                "Could not initialize metadata writer; metadata will be skipped"
            )

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
        else:
            self.logger.error(
                "Trying to stop a non-constructed CameraEngine, check if the camera is plugged in."
            )
        # Close metadata file if open (child process local file handle)
        try:
            if self.f is not None:
                try:
                    self.f.flush()
                except Exception:
                    pass
                try:
                    self.f.close()
                except Exception:
                    pass
        except Exception:
            pass

    def acquire_taggedpic(
        self,
    ) -> Optional[Tuple[np.ndarray, PhotoExif]]:
        """start_acquisition Acquire a tagged picture from the camera

        Returns
        -------
        TaggedPicture
            A struct containing a pair (RawImage, Waypoint)
        """

        if True:
            img: _typing.NDArray[np.uint8] = np.empty((0,), dtype=np.uint8)
            try:
                pix_data = self.receive_from_queue(timeout=5)
            except Exception as e:
                self.logger.warning(f"probable timeout exception: {e}")
                return None

            if pix_data is None:
                return None

            if not pix_data:  # or not self.is_in(pix_data):
                return None

            # convert yaw from magnetic north to true north
            #! CHECK IF MAGNETIC_DECLINATION IS UP TO DATE (it's dependent on the location and the date)
            """pix_data.yaw = magnorth2truenorth(
                pix_data.yaw
            ).item() """  # ? magnorth2truenorth return a numpy array, in this case with only one element

            timestamp = time.time()
            metadata = PhotoExif(
                filename=os.path.join(
                    self.out_dir, f"{self.img_count}_{self.initial_timestamp}.npy"
                ),
                exposure_time=0.0,
                timestamp=timestamp,
                gain=0.0,
                black_level=0,
                bsl_color_space="",
                gamma=0,
                light_source_preset="",
                pixel_format="",
                pix_data=pix_data,
            )

            return img, metadata

    def is_in(self, resp: Any) -> bool:
        """This function evaluates wheather the drone is inside the shooting area,
        a.k.a the airdrop boundary

        Returns:
            bool: drone is in the area (True), or it isn't (False)
        """
        # Get latitude and longitude from communication

        if not resp:
            self.logger.error(f"unable to locate drone position, returning True")
            return True

        is_in_one_of_fences = Point(resp.lat, resp.lon).within(self.geo_fence)
        is_in_one_of_fences = is_in_one_of_fences or Point(resp.lat, resp.lon).within(
            self.mapping_geo_fence
        )
        return is_in_one_of_fences

    def save(self, image: np.ndarray, metadata: PhotoExif) -> None:

        row = []
        for path in self.METADATA_COLUMNS.values():
            # format timestamp as ISO string, others are raw values
            try:
                if path == "timestamp":
                    ts = getattr(metadata, "timestamp")
                    val = datetime.datetime.fromtimestamp(ts).isoformat(
                        timespec="seconds", sep=" "
                    )
                else:
                    val = self._resolve_path(metadata, path)
            except Exception:
                val = ""
            row.append(val)

        if self.writer is not None:
            try:
                self.writer.writerow(row)
                # flush to ensure data is written to disk immediately
                try:
                    if self.f is not None:
                        self.f.flush()
                        import os as _os

                        try:
                            _os.fsync(self.f.fileno())
                        except Exception:
                            # some platforms or file-like objects may not support fsync
                            pass
                except Exception:
                    pass
            except Exception:
                self.logger.warning("Failed to write metadata row")
        else:
            self.logger.warning(
                "Metadata writer not initialized; skipping metadata write"
            )

    def _resolve_path(self, obj: object, path: str):
        """Resolve a dot-separated attribute path from obj (e.g. 'pix_data.lat')."""
        parts = path.split(".")
        val = obj
        for p in parts:
            val = getattr(val, p)
        return val

    def update(self) -> None:
        self.configs = SFTConfig(self.yaml_path, setup_logger=False)
        self.config_dict = self.configs.config_dict
        self.is_initialized.clear()
        self.setup_cam()
        self.to_update.clear(self.idx)
        self.logger.info("Camera updated")
        return None

    def put(self, metadata: PhotoExif) -> None:
        # If there are downstream queues, send the metadata to them.
        # It's valid for a stage to have no output queues (e.g. last stage),
        # so don't treat that as an error — send_to_queues already handles None.
        self.send_to_queues(metadata)

    def get(self) -> None:
        raise NotImplementedError

    def _dump_data(self) -> None:
        pass
