"""MAVLink Communicator stage

Publishes `PixData` instances to the pipeline out queues. Supports a demo
mode (simulated telemetry) and a minimal real-mode using pymavlink if
available.
"""

from __future__ import annotations
import datetime
import time
import usb.core
import usb.util
from typing import Optional
import socket
from pymavlink import mavutil
from sft_math import survey
from sft_math.survey import CameraParams
from common.pipeline_stage import PipelineStage
from common.config import SFTConfig
from core.logger import SFTLogger
from common.mavlink_structs import PixData
from multiprocessing.queues import Queue
from contextlib import contextmanager

class MAVLinkCommunicator(PipelineStage):
    """Simple communicator that publishes PixData objects.

    Behavior:
    - If configs.config_dict.get('mavlink', {}).get('demo', True) is True -> run simulated data
    - Otherwise attempt to use pymavlink (best-effort) to receive real telemetry.
    """

    def __init__(
        self,
        configs: SFTConfig,
        logger: SFTLogger,
        idx: int,
        out_q: Optional[list] = None,
        in_q: Optional[Queue] = None,
        in_command_q: Optional[list] = None,
        out_command_q: Optional[list] = None,
        stage_name: str = "Communicator",
    ):
        super().__init__(
            configs=configs,
            logger=logger,
            in_q=in_q,
            out_q=out_q,
            in_command_q=in_command_q,
            out_command_q=out_command_q,
            stage_name=stage_name,
            idx=idx,
        )

        serial_port = configs.config_dict["mavlink"]["serial_port"]
        self.height_threshold = configs.config_dict["utils"]["height_threshold"]
        self.connection_string = serial_port
        self.timeout = configs.config_dict["mavlink"]["timeout"]
        mav_cfg = configs.config_dict.get("mavlink", {})
        self.rate_hz = int(mav_cfg.get("rate_hz", 10))
        self.mav_start_time: Optional[int] = None
        # runtime state
        self.stopped = False


    def mission(self) -> bool:
        # Minimal real-mode: attempt to import pymavlink and mirror messages into PixData
        try:
            from pymavlink import mavutil
        except Exception:
            self.logger.error("pymavlink not available; cannot run real MAVLink mode")
            return True

        from common.mavlink_structs import grip_2_cmd

        conn = self.configs.config_dict.get("mavlink", {}).get(
            "connection", self.connection_string
        )
        baud = int(self.configs.config_dict.get("mavlink", {}).get("baud", 57600))
        fake_wp_survey = self.configs.config_dict.get("mavlink", {}).get(
            "fake_wp_survey", False
        )

        @contextmanager
        def managed_mavlink_connection(conn, baud):

            pass

            # No hardcoded path: use the configured serial connection string.
            # If the YAML sets mavlink.serial_port, that value will be used here.
            conn = self.configs.config_dict.get("mavlink", {}).get(
                "connection", self.connection_string
            )

            self.logger.info(f"Opening MAVLink connection @ {conn}")

            while True:
                try:
                    mav = mavutil.mavlink_connection(conn, baud=baud)
                    msg = mav.wait_heartbeat(timeout=1)
                    if msg is not None:
                        self.logger.info("MAVLink connection opened successfully.")
                        self.mav_start_time = None
                        yield mav
                        break
                    else:
                        raise Exception("No heartbeat received")
                except Exception as e:
                    self.logger.error(f"Unable to open MAVLink connection: {e}")

            try:
                mav.close()
            except Exception:
                self.logger.error(
                    "By logic this is never supposed to happen, yet here we are."
                )

            self.logger.info("MAVLink connection closed")

        while not self.stopped:
            with managed_mavlink_connection(conn, baud) as mav:
                data = PixData()
                try:
                            
                    self.send_to_command_queues("GLHF")
                    self.logger.info("MAVLink OK")

                    while not self.stopped:
                        msg = mav.recv_match(blocking=True, timeout=5)
                        if msg is None:
                            raise Exception("No message received")
                        mtype = msg.get_type()
                        if mtype is None:
                            raise Exception("No message received")
                        # self.logger.debug(f"Received MAVLink message: {mtype}")
                        # Simple mapping for some message types

                        elif mtype == "ATTITUDE":
                            data.roll = getattr(msg, "roll", data.roll)
                            data.pitch = getattr(msg, "pitch", data.pitch)
                            data.yaw = getattr(msg, "yaw", data.yaw)

                        elif mtype == "CAMERA_IMAGE_CAPTURED":
                            data.camera_trigger_seq = getattr(msg, "image_index", 0)
                            data.lat = getattr(msg, "lat", data.lat * 1e7) / 1e7
                            data.lon = getattr(msg, "lon", data.lon * 1e7) / 1e7
                            data.alt_msl = (
                                getattr(msg, "alt", data.alt_msl * 1000) / 1000.0
                            )
                            data.alt_rel = (
                                getattr(msg, "relative_alt", data.alt_rel * 1000)
                                / 1000.0
                            )
                            self.send_to_queues(data)

                        elif mtype == "HEARTBEAT":
                            data.flight_mode = mavutil.mode_string_v10(msg)
                            data.armed = (
                                msg.base_mode
                                & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                            ) != 0

                        elif mtype == "MISSION_CURRENT":
                            data.mission_current_wp = getattr(
                                msg, "seq", data.mission_current_wp
                            )
                            data.mission_total_wp = getattr(
                                msg, "total", data.mission_total_wp
                            )
                            data.mission_state = getattr(
                                msg, "mission_state", data.mission_state
                            )
                            data.mission_mode = getattr(
                                msg, "mission_mode", data.mission_mode
                            )
                            data.mission_id = getattr(
                                msg, "mission_id", data.mission_id
                            )

                        elif mtype == "MISSION_ITEM_REACHED":
                            data.mission_current_wp = getattr(
                                msg, "seq", data.mission_current_wp
                            )

                        elif mtype == "TIMESYNC":
                            if self.mav_start_time is None:
                                self.mav_start_time = time.time_ns() // 1000
                            data.teensy_timestamp = int(
                                getattr(msg, "ts1", data.teensy_timestamp * 1000)
                                // 1000
                                + self.mav_start_time
                            )  # convert to ms

                        data.message_count += 1
                        data.last_update = time.time()

                except KeyboardInterrupt:
                    self.logger.info("Real MAVLink loop interrupted by user")
                    self.stopped = True
                except Exception as e:
                    self.send_to_command_queues("FF15")
                    self.logger.error(f"Real MAVLink loop error: {e}")
                    self.logger.info("Attempting recovery...")
                    time.sleep(2)
        return True

    def stop(self) -> None:
        self.logger.info("Stopping MAVLinkCommunicator")
        self.stopped = True

    def update(self) -> None:
        super().update()

    def get(self, data):
        # TO-DO ma che cazzo ne so
        raise NotImplementedError

    def put(self, data):
        raise NotImplementedError

    def _dump_data(self) -> None:
        pass

    def print_utc_from_micros(self, micros: int):
        """
        Convert microseconds since Unix epoch to UTC datetime and print.
        """
        # Convert microseconds → seconds (float)
        t_sec = micros / 1_000_000
        # Convert to UTC datetime
        utc_dt = datetime.datetime.utcfromtimestamp(t_sec)
        print("UTC datetime:", utc_dt.isoformat(sep=" "))

    def ff15(self) -> None:
        pass

    def glhf(self) -> None:
        pass

    def survey_stop(self) -> None:
        pass
