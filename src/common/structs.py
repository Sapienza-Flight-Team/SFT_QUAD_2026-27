"""
This module describes a series of data structures that are used throughout the project.

the dataclass and dataclass_json decorators are used to automatically generate methods
such as __init__, __str__ and JSON serialization/deserialization.

Each dataclass is expected to be data-only, hence to have NO custom-defined methods.
"""

from dataclasses import dataclass
from enum import Enum

from dataclasses_json import dataclass_json

import ctypes

from common.mavlink_structs import PixData


@dataclass_json
@dataclass
class Waypoint:
    """Waypoint
    A waypoint is a GPS coordinate
    """

    lat: float
    lon: float


class SavingMode(Enum):
    """SavingMode `SavingMode` is an enum that describes the saving mode of our picturesaver module

    Parameters
    ----------
    Enum : str
        Enum superclass
    """

    NPY = "NUMPY"
    RAW = "RAW"
    PNG = "PNG"


class WayPoint(ctypes.Structure):  # in degrees
    _fields_ = (
        ("lat", ctypes.c_double),
        ("lon", ctypes.c_double),
    )


class Servo(ctypes.Structure):
    _fields_ = (("n", ctypes.c_uint8),)  # servo number


@dataclass_json
@dataclass
class PhotoExif:
    """PhotoExif
    A dataclass that contains the exif data of a photo taken by the camera.
    It is used to store the filename, timestamp, exposure time, exposure delay and gain of the photo. It also
    stores pix_data, which contains the GPS coordinates and other UAV telemetry associated with the photo.
    The filename is the path to the photo on the filesystem.
    """

    filename: str
    timestamp: float
    exposure_time: float
    gain: float
    black_level: int
    bsl_color_space: str
    gamma: int
    light_source_preset: str
    pixel_format: str
    pix_data: PixData
    shm_name: str | None = None
    shm_shape: tuple | None = None
    shm_dtype: str | None = None
    full_image_shape: tuple | None = None


SIZE_WP = ctypes.sizeof(WayPoint)
SIZE_SERVO = ctypes.sizeof(Servo)

if __name__ == "__main__":
    print(f"Size of WayPoint: {SIZE_WP} bytes")

pipeline_stage_type = {
    1: "Input Generator",
    2: "Aggregator",
    3: "Raw input consumer",
    4: "Processor",
    5: "Output Handler"
}
