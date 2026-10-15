from dataclasses import dataclass, field
from typing import Optional, Tuple
import time
from enum import IntEnum


@dataclass
class LatLon:
    lat: float
    lon: float


@dataclass
class CameraParams:
    sensor_width_mm: float = 0.0
    sensor_height_mm: float = 0.0
    image_width_px: float = 0.0
    image_height_px: float = 0.0
    focal_length_mm: float = 0.0
    frontal_overlap: float = 0.0  # 0-1
    side_overlap: float = 0.0  # 0-1
    altitude_m: float = 0.0
    landscape: bool = True


@dataclass
class Transect:
    """Coppia inizio/fine di una striscia di volo."""

    start: LatLon
    end: LatLon
    altitude_m: float


class MissionState(IntEnum):
    UNKNOWN = 0
    NOT_STARTED = 1
    ACTIVE = 2
    NOT_STANDARD = 3  # state not used normally
    PAUSED = 4
    COMPLETE = 5


class MissionMode(IntEnum):
    UNKNOWN = 0
    IN_MISSION = 1
    SUSPENDED = 2


@dataclass
class PixData:
    """Dati telemetrici da Pixhawk via MAVLink

    See tools/mavlink_test.py for mapping notes and rationale for using a
    Python dataclass instead of a ctypes.Structure.
    """

    # GPS Position
    lat: float = 0.0  # gradi decimali
    lon: float = 0.0  # gradi decimali
    alt_msl: float = 0.0  # altitudine MSL (metri)
    alt_rel: float = 0.0  # altitudine relativa al home (metri)

    # Attitude (degrees). MAVSDK already exposes Euler attitude in degrees.
    roll: float = 0.0  # phi
    pitch: float = 0.0  # theta
    yaw: float = 0.0  # psi

    # Timestamp
    teensy_timestamp: int = 0  # millisecondi boot time
    system_time: int = 0  # Unix epoch microseconds

    # Camera trigger sequence (CAMERA_TRIGGER.seq from PX4).
    # -1 = no trigger seq seen yet (distinguishable from a real seq of 0).
    camera_trigger_seq: int = -1

    # Flight mode e stato
    flight_mode: str = "UNKNOWN"
    armed: bool = False
    system_status: str = "UNKNOWN"
    ekf_ready: int = 0

    # Missione
    mission_current_wp: int = 0
    mission_total_wp: int = 0
    mission_state: MissionState = MissionState.UNKNOWN
    mission_mode: MissionMode = MissionMode.UNKNOWN
    mission_id: int = 0
    gripper: Tuple[int, int] = (0, 0)

    # Batteria
    battery_voltage: Optional[float] = None
    battery_current: Optional[float] = None
    battery_remaining: Optional[int] = None

    # Stats
    last_update: float = field(default_factory=time.time)
    message_count: int = 0

    def to_dict(self) -> dict:
        """Converti in dizionario per logging"""
        return {
            "GPS": f"{self.lat:.6f}, {self.lon:.6f}",
            "Alt": f"MSL:{self.alt_msl:.1f}m, REL:{self.alt_rel:.1f}m",
            "Speed": f"GND:{self.vground:.1f}m/s, N:{self.vx:.1f}, E:{self.vy:.1f}, D:{self.vz:.1f}",
            "Attitude": (
                f"R:{self.roll:.2f}°, "
                f"P:{self.pitch:.2f}°, "
                f"Y:{self.yaw:.2f}°"
            ),
            "Status": f"{self.flight_mode} {'[ARMED]' if self.armed else '[DISARMED]'} ({self.system_status})",
            "Mission": f"{self.mission_current_wp}/{self.mission_total_wp}",
            "Battery": (
                f"{self.battery_voltage:.1f}V {self.battery_remaining}%"
                if self.battery_voltage
                else "N/A"
            ),
            "Messages": self.message_count,
        }