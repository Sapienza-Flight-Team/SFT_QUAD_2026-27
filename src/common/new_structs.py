"""Pointer-free ctypes messages with an image layout fixed at pipeline startup.

Image dimensions come from the selected tagged_image producer's configuration
when this module is imported. Restart after changing them: resizing a live
ctypes class would invalidate existing instances and shared-memory allocations.
"""

from __future__ import annotations

import ctypes as ct
from importlib import import_module
from typing import Literal, TypedDict
from math import prod

import yaml
from common.config import MISSION_CONFIG_PATH
from core.stage_catalog import build_stage_catalog, read_profile

TEXT_CAPACITY = 64


def _image_shape() -> tuple[int, ...]:
    with MISSION_CONFIG_PATH.open(encoding="utf-8") as file:
        stages = read_profile(build_stage_catalog(), yaml.safe_load(file))
    producer = next(
        spec
        for spec in stages.values()
        if "tagged_image"
        in yaml.safe_load(spec["contract"].read_text(encoding="utf-8"))["output"]
    )
    config_module = producer["module"].rsplit(".", 1)[0] + "." + producer["config"].stem
    image = {**import_module(config_module).CONFIG, **producer["settings"]}
    width, height = image["resolution"]
    return (
        (height, width)
        if image["channels"] == 1
        else (height, width, image["channels"])
    )


IMAGE_SHAPE = _image_shape()
IMAGE_SIZE = prod(IMAGE_SHAPE)

FIELD_REGISTRY: dict[str, list[tuple[str, type]]] = {
    "PhotoExif": [
        ("frame_counter", ct.c_uint64),
        ("exposure_time", ct.c_double),
        ("gain", ct.c_double),
        ("black_level", ct.c_double),
        ("bsl_color_space", ct.c_char * TEXT_CAPACITY),
        ("gamma", ct.c_double),
        ("light_source_preset", ct.c_char * TEXT_CAPACITY),
    ],
    "PixData": [
        ("lat", ct.c_double),
        ("lon", ct.c_double),
        ("alt_msl", ct.c_double),
        ("alt_rel", ct.c_double),
        ("roll", ct.c_double),
        ("pitch", ct.c_double),
        ("yaw", ct.c_double),
        ("camera_trigger_seq", ct.c_int64),
        ("mission_current_wp", ct.c_uint32),
        ("mission_total_wp", ct.c_uint32),
    ],
}


class PhotoExif(ct.Structure):

    # The SDK appends chunk metadata after pixels when grabbing into this buffer.
    _fields_ = [
        *FIELD_REGISTRY["PhotoExif"],
        ("image", ct.c_uint8 * IMAGE_SIZE),
        ("_camera_chunks", ct.c_uint8 * 4096),
    ]


class PixData(ct.Structure):
    _fields_ = FIELD_REGISTRY["PixData"]

    def __init__(self):
        self.camera_trigger_seq


class AggregatedMetadata(ct.Structure):

    _fields_ = [
        *FIELD_REGISTRY["PhotoExif"],
        *FIELD_REGISTRY["PixData"],
        ("image", ct.c_uint8 * IMAGE_SIZE),
    ]

    def __init__(self):
        self.camera_trigger_seq = -1


# A common registry ID backed by each struct's existing capture field.
PhotoExif.ID = PhotoExif.frame_counter
PixData.ID = PixData.camera_trigger_seq
AggregatedMetadata.ID = AggregatedMetadata.frame_counter


class ConnectionSpec(TypedDict):
    connection_type: str
    wait_for_consumer: bool
    data_type: type[ct.Structure] | frozenset[type[ct.Structure]]


# Keys identify streams, not classes: separate streams may share a schema.
DATA_TYPE_REGISTRY: dict[str, ConnectionSpec] = {
    "main_camera": {
        "connection_type": "shm",
        "wait_for_consumer": False,
        "data_type": frozenset({PhotoExif, PixData}),
    },
    "sensors": {
        "connection_type": "fifo",
        "wait_for_consumer": False,
        "data_type": PixData,
    },
    "tagged_image": {
        "connection_type": "shm",
        "wait_for_consumer": False,
        "data_type": PhotoExif,
    },
    "aggregator_output": {
        "connection_type": "shm",
        "wait_for_consumer": False,
        "data_type": AggregatedMetadata,
    },
    "resized_aggregator_output": {
        "connection_type": "shm",
        "wait_for_consumer": False,
        "data_type": AggregatedMetadata,
    },
}
