#!/usr/bin/env python3
"""
Mock MAVLink telemetry source for pipeline.py.

By default this streams directly to the pipeline MAVLink listener:

    python tools/mock_mavlink.py --replay-csv data/outs/metadata_20260520-223005.csv

Matching sft_config.yaml values:

    mavlink.connection: udpin:0.0.0.0:14030
    mavlink.mock_type: fake
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass

import pandas as pd

os.environ["MAVLINK20"] = "1"

from pymavlink import mavutil

PIPELINE_MAVLINK_HOST = "127.0.0.1"
PIPELINE_MAVLINK_PORT = 14030
CAMERA_TRIGGER_START_DELAY_S = 10.0


@dataclass(frozen=True)
class MockMAVLinkConfig:
    sysid: int
    compid: int
    udp_host: str
    udp_port: int
    replay_csv: str | None
    rate_hz: float
    loop: bool
    camera_trigger_start_delay_s: float = CAMERA_TRIGGER_START_DELAY_S


DEFAULT_GRIPPER = (0, 0)


def parse_gripper_config(value) -> tuple[int, int] | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(float(value[0])), int(float(value[1]))
        except (TypeError, ValueError):
            return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) >= 2:
        return int(float(numbers[0])), int(float(numbers[1]))

    return None


class MockMAVLinkAutopilot:
    """Emits MAVLink telemetry from a CSV replay or synthetic flight pattern."""

    def __init__(
        self,
        mav,
        replay_csv: str | None = None,
        loop_replay: bool = False,
    ):
        self.mav = mav
        self.start_time = time.time()
        self.message_count = 0
        self.loop_replay = loop_replay
        self._last_gripper: tuple[int, int] | None = None
        self._active_gripper: tuple[int, int] | None = None

        # Initial synthetic position (Rome, Italy).
        self.start_lat = 41.9028
        self.start_lon = 12.4964
        self.start_alt = 100.0

        # Flight state.
        self.armed = False
        self.flight_mode = "STABILIZE"
        self.mission_current = 0
        self.mission_total = 5

        # Battery state.
        self.battery_voltage = 16.8
        self.battery_remaining = 100

        if replay_csv:
            self.metadata = pd.read_csv(replay_csv)
            if self.metadata.empty:
                raise ValueError(f"{replay_csv} does not contain any rows")
            self.mission_total = len(self.metadata)
        else:
            self.metadata = None

    def _metadata_row(self):
        if self.metadata is None:
            return None
        if self.message_count >= len(self.metadata):
            if self.loop_replay:
                return self.metadata.iloc[self.message_count % len(self.metadata)]
            return None
        return self.metadata.iloc[self.message_count]

    def _csv_value(self, *columns: str, default=0.0):
        row = self._metadata_row()
        if row is None:
            return default
        for column in columns:
            if column in row and pd.notna(row[column]):
                return row[column]
        return default

    def _current_gripper(self) -> tuple[int, int] | None:
        gripper_id = self._csv_value("gripper_id", default=None)
        if gripper_id is not None:
            try:
                parsed_gripper_id = int(float(gripper_id))
                if parsed_gripper_id == 0:
                    return DEFAULT_GRIPPER
                return parsed_gripper_id, 1
            except (TypeError, ValueError):
                pass

        return parse_gripper_config(self._csv_value("gripper", default=None))

    @staticmethod
    def _is_active_gripper(gripper: tuple[int, int] | None) -> bool:
        return gripper is not None and gripper != DEFAULT_GRIPPER and gripper[1] != 0

    def send_gripper_command(self, gripper: tuple[int, int]):
        self.mav.command_long_send(
            0,
            0,
            mavutil.mavlink.MAV_CMD_DO_GRIPPER,
            0,
            float(gripper[0]),
            float(gripper[1]),
            0,
            0,
            0,
            0,
            0,
        )

    def send_replay_gripper_if_needed(self):
        gripper = self._current_gripper()
        if gripper is None or gripper == self._last_gripper:
            return

        sent_stop: tuple[int, int] | None = None
        if self._is_active_gripper(self._active_gripper):
            sent_stop = (self._active_gripper[0], 0)
            self.send_gripper_command(sent_stop)

        if gripper != sent_stop:
            self.send_gripper_command(gripper)

        self._active_gripper = gripper if self._is_active_gripper(gripper) else None
        self._last_gripper = gripper

    def send_final_survey_stop(self):
        if not self._is_active_gripper(self._active_gripper):
            return

        self.send_gripper_command((self._active_gripper[0], 0))
        self._active_gripper = None
        self._last_gripper = None

    def advance_replay_row(self) -> bool:
        if self.metadata is None:
            self.message_count += 1
            return True

        self.message_count += 1
        if self.message_count < len(self.metadata):
            return True

        self.send_final_survey_stop()
        if not self.loop_replay:
            return False

        self.message_count = 0
        return True

    def get_time_ms(self) -> int:
        """Time since boot in milliseconds."""
        if self.metadata is not None:
            return int((self.get_time_us() // 1000) % (2**32))
        return int((time.time() - self.start_time) * 1000)

    def get_time_us(self) -> int:
        """Time since boot in microseconds."""
        if self.metadata is not None:
            raw_timestamp = int(
                self._csv_value("teensy_timestamp", "timestamp", default=0)
            )
            if raw_timestamp > 10_000_000_000_000:
                return raw_timestamp
            return raw_timestamp * 1000
        return int((time.time() - self.start_time) * 1e6)

    def update_flight_state(self):
        """Update synthetic flight state based on elapsed wall time."""
        t = time.time() - self.start_time

        if t > 5:
            self.armed = True

        modes = ["STABILIZE", "LOITER", "AUTO", "GUIDED"]
        self.flight_mode = modes[int(t / 20) % len(modes)]

        if self.armed and self.flight_mode in ["AUTO", "GUIDED"]:
            self.mission_current = int(t / 15) % (self.mission_total + 1)

        self.battery_voltage = max(14.0, 16.8 - (t / 600) * 2.8)
        self.battery_remaining = max(0, int(100 - (t / 600) * 100))

    def get_position(self) -> tuple[float, float, float, float]:
        """Calculate current position."""
        t = time.time() - self.start_time
        radius = 0.001
        angular_speed = 0.1

        lat = self.start_lat + radius * math.cos(angular_speed * t)
        lon = self.start_lon + radius * math.sin(angular_speed * t)
        alt_rel = self.start_alt + 10 * math.sin(0.2 * t)
        alt_msl = alt_rel + 50

        if self.metadata is not None:
            lat = float(self._csv_value("lat", default=lat))
            lon = float(self._csv_value("lon", default=lon))
            alt_msl = float(self._csv_value("alt_msl", "h", default=alt_msl))
            alt_rel = float(self._csv_value("alt_rel", "h", default=alt_msl))

        return lat, lon, alt_rel, alt_msl

    def get_velocity(self) -> tuple[float, float, float, float]:
        """Calculate current velocity."""
        t = time.time() - self.start_time
        angular_speed = 0.1

        vground = 10 + 2 * math.sin(0.3 * t)
        if self.metadata is not None:
            vground = float(self._csv_value("vground", "vmod", default=vground))
        vx = vground * math.cos(angular_speed * t)
        vy = vground * math.sin(angular_speed * t)
        vz = 2 * math.cos(0.2 * t)

        return vx, vy, vz, vground

    def get_attitude(self) -> tuple[float, float, float]:
        """Calculate current attitude."""
        t = time.time() - self.start_time
        angular_speed = 0.1

        roll = 0.1 * math.sin(0.5 * t)
        pitch = 0.05 * math.sin(0.3 * t)
        yaw = (angular_speed * t) % (2 * math.pi)

        if self.metadata is not None:
            row = self._metadata_row()
            if row is not None and "roll" in row:
                roll = float(row["roll"])
            else:
                roll = math.radians(float(self._csv_value("phi", default=0.0)))
            if row is not None and "pitch" in row:
                pitch = float(row["pitch"])
            else:
                pitch = math.radians(float(self._csv_value("theta", default=0.0)))
            if row is not None and "yaw" in row:
                yaw = float(row["yaw"])
            else:
                yaw = math.radians(float(self._csv_value("psi", default=0.0)))

        return roll, pitch, yaw

    def get_heading(self) -> int:
        """Get heading in degrees (0-359)."""
        _, _, yaw = self.get_attitude()
        return int(math.degrees(yaw)) % 360

    def send_heartbeat(self):
        """Send HEARTBEAT message."""
        base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        if self.armed:
            base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

        custom_mode_map = {
            "STABILIZE": 0,
            "LOITER": 5,
            "AUTO": 3,
            "GUIDED": 4,
        }
        custom_mode = custom_mode_map.get(self.flight_mode, 0)

        self.mav.heartbeat_send(
            type=mavutil.mavlink.MAV_TYPE_QUADROTOR,
            autopilot=mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
            base_mode=base_mode,
            custom_mode=custom_mode,
            system_status=(
                mavutil.mavlink.MAV_STATE_ACTIVE
                if self.armed
                else mavutil.mavlink.MAV_STATE_STANDBY
            ),
        )

    def send_sys_status(self):
        """Send SYS_STATUS message."""
        self.mav.sys_status_send(
            onboard_control_sensors_present=0xFFFFFFFF,
            onboard_control_sensors_enabled=0xFFFFFFFF,
            onboard_control_sensors_health=0xFFFFFFFF,
            load=500,
            voltage_battery=int(self.battery_voltage * 1000),
            current_battery=1500,
            battery_remaining=self.battery_remaining,
            drop_rate_comm=0,
            errors_comm=0,
            errors_count1=0,
            errors_count2=0,
            errors_count3=0,
            errors_count4=0,
        )

    def send_global_position_int(self):
        """Send GLOBAL_POSITION_INT message."""
        lat, lon, alt_rel, alt_msl = self.get_position()
        vx, vy, vz, _ = self.get_velocity()
        heading = self.get_heading()

        self.mav.global_position_int_send(
            time_boot_ms=self.get_time_ms(),
            lat=int(lat * 1e7),
            lon=int(lon * 1e7),
            alt=int(alt_msl * 1000),
            relative_alt=int(alt_rel * 1000),
            vx=int(vx * 100),
            vy=int(vy * 100),
            vz=int(vz * 100),
            hdg=heading * 100,
        )

    def send_attitude(self):
        """Send ATTITUDE message."""
        roll, pitch, yaw = self.get_attitude()

        self.mav.attitude_send(
            time_boot_ms=self.get_time_ms(),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            rollspeed=0.01,
            pitchspeed=0.01,
            yawspeed=0.05,
        )

    def send_timesync(self):
        """Send TIMESYNC message for communicator readiness."""
        self.mav.timesync_send(tc1=0, ts1=self.get_time_us())

    def send_camera_trigger(self):
        """Send CAMERA_TRIGGER so the communicator publishes telemetry."""
        self.mav.camera_trigger_send(
            time_usec=self.get_time_us(), seq=self.message_count
        )
        print(f"CAMERA_TRIGGER seq={self.message_count}")

    def send_vfr_hud(self):
        """Send VFR_HUD message."""
        _, _, _, vground = self.get_velocity()
        _, _, alt_rel, _ = self.get_position()
        heading = self.get_heading()
        _, _, vz = self.get_velocity()[:3]

        self.mav.vfr_hud_send(
            airspeed=vground,
            groundspeed=vground,
            heading=heading,
            throttle=50,
            alt=alt_rel,
            climb=-vz,
        )

    def send_mission_current(self):
        """Send MISSION_CURRENT message."""
        try:
            mission_state = int(float(self._csv_value("mission_state", default=0)))
        except (ValueError, TypeError):
            mission_state = 0
            
        try:
            mission_mode = int(float(self._csv_value("mission_mode", default=0)))
        except (ValueError, TypeError):
            mission_mode = 0

        self.mav.mission_current_send(
            seq=self.mission_current,
            total=self.mission_total,
            mission_state=mission_state,
            mission_mode=mission_mode,
        )

    def send_gps_raw_int(self):
        """Send GPS_RAW_INT message."""
        lat, lon, _, alt_msl = self.get_position()

        self.mav.gps_raw_int_send(
            time_usec=self.get_time_us(),
            fix_type=3,
            lat=int(lat * 1e7),
            lon=int(lon * 1e7),
            alt=int(alt_msl * 1000),
            eph=100,
            epv=100,
            vel=int(self.get_velocity()[3] * 100),
            cog=self.get_heading() * 100,
            satellites_visible=12,
        )

    def send_all_messages(self):
        """Send a complete telemetry update."""
        self.update_flight_state()
        self.send_heartbeat()
        self.send_timesync()
        self.send_global_position_int()
        self.send_attitude()

        if self.message_count % 2 == 0:
            self.send_sys_status()
            self.send_vfr_hud()

        if self.message_count % 3 == 0:
            self.send_mission_current()
            self.send_gps_raw_int()


def create_mavlink(config: MockMAVLinkConfig):
    connection = f"udpout:{config.udp_host}:{config.udp_port}"
    print(f"Mock MAVLink streaming to {connection}")
    return mavutil.mavlink_connection(
        connection,
        source_system=config.sysid,
        source_component=config.compid,
    )


def start_server(config: MockMAVLinkConfig):
    """Start the mock MAVLink telemetry stream."""
    mav = create_mavlink(config).mav

    print(f"Autopilot: SYSID={config.sysid}, COMPID={config.compid}")
    if config.replay_csv:
        print(f"Replay CSV: {config.replay_csv}")
        print(f"Replay loop: {config.loop}")
    else:
        print("Replay CSV: none, using synthetic telemetry")

    autopilot = MockMAVLinkAutopilot(
        mav,
        replay_csv=config.replay_csv,
        loop_replay=config.loop,
    )
    period = 1.0 / max(config.rate_hz, 1e-6)
    first_camera_trigger_time = time.monotonic() + config.camera_trigger_start_delay_s
    print(
        f"Camera triggers start after {config.camera_trigger_start_delay_s:.1f}s "
        "of telemetry warmup"
    )
    if config.replay_csv:
        camera_trigger_time = first_camera_trigger_time

    try:
        while True:

            autopilot.update_flight_state()
            autopilot.send_heartbeat()
            autopilot.send_timesync()
            autopilot.send_global_position_int()
            autopilot.send_attitude()
            autopilot.send_sys_status()
            autopilot.send_vfr_hud()
            autopilot.send_mission_current()
            autopilot.send_gps_raw_int()

            now = time.monotonic()

            if config.replay_csv:
                if now >= camera_trigger_time:
                    current_timestamp = autopilot.get_time_us()
                    autopilot.send_replay_gripper_if_needed()
                    autopilot.send_camera_trigger()
                    has_next_row = autopilot.advance_replay_row()

                    if not has_next_row:
                        print("Replay CSV complete")
                        break

                    next_timestamp = autopilot.get_time_us()
                    delta_time = (next_timestamp - current_timestamp) / 1_000_000
                    if delta_time <= 0:
                        delta_time = period

                    camera_trigger_time += delta_time

                sleep_time = min(
                    period, max(0.0, camera_trigger_time - time.monotonic())
                )
            else:
                if now >= first_camera_trigger_time:
                    autopilot.send_camera_trigger()
                    autopilot.advance_replay_row()
                sleep_time = period

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except (ConnectionResetError, BrokenPipeError):
        print("Client disconnected")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay or synthesize MAVLink telemetry for pipeline.py.",
        epilog=(
            f"Default UDP output matches mavlink.connection=udpin:0.0.0.0:{PIPELINE_MAVLINK_PORT} "
            "in sft_config.yaml."
        ),
    )
    parser.add_argument(
        "--sysid", type=int, default=1, help="Autopilot system ID (default: 1)"
    )
    parser.add_argument(
        "--compid", type=int, default=1, help="Autopilot component ID (default: 1)"
    )
    parser.add_argument(
        "--udp-host",
        type=str,
        default=PIPELINE_MAVLINK_HOST,
        help=f"UDP host to send MAVLink packets to (default: {PIPELINE_MAVLINK_HOST})",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=PIPELINE_MAVLINK_PORT,
        help=f"UDP port to send MAVLink packets to (default: {PIPELINE_MAVLINK_PORT})",
    )
    parser.add_argument(
        "--replay-csv",
        type=str,
        default=None,
        help="CSV metadata file to replay. If omitted, synthetic telemetry is used.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop --replay-csv instead of stopping after the last row.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Telemetry update rate in Hz",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> MockMAVLinkConfig:
    replay_csv = args.replay_csv

    if replay_csv == "":
        replay_csv = None

    return MockMAVLinkConfig(
        sysid=args.sysid,
        compid=args.compid,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        replay_csv=replay_csv,
        rate_hz=args.rate_hz,
        loop=args.loop,
        camera_trigger_start_delay_s=CAMERA_TRIGGER_START_DELAY_S,
    )


def main():
    args = build_parser().parse_args()
    config = config_from_args(args)

    while True:
        try:
            start_server(config)
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Restarting server in 2 seconds...")
            time.sleep(2)


if __name__ == "__main__":
    main()
