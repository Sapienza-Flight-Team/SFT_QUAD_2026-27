from contextlib import contextmanager
from math import prod
import multiprocessing as mp

import numpy as np
import pytest

from common.data_connection import DataConnection
from common.new_structs import (
    AggregatedMetadata,
    MissionMode,
    MissionState,
    PhotoExif,
    PixData,
    TEXT_CAPACITY,
    get_buffer_layout,
)


@contextmanager
def _connection(layout):
    # The pipeline prepares the layout and owns cleanup; the identifier is opaque.
    connection = DataConnection("shm", False, "test_stream", layout=layout)
    assert connection.layout is layout
    connection.register_producer("p1")
    connection.register_consumer("c1")
    connection.register_consumer("c2")
    try:
        yield connection
    finally:
        resource = connection.CONNECTION["p1"]
        resource.close()
        resource.unlink()


def _read_single(connection, consumer="c1"):
    frames = connection.get(consumer)
    assert len(frames) == 1
    assert frames[0].keys() == {"metadata", "image"}
    return frames[0]["metadata"], frames[0]["image"]


def _aggregate_in_child(connection, queue):
    layout = connection.layout
    telemetry = queue.get(timeout=5)
    metadata = AggregatedMetadata(
        photo_exif=PhotoExif(frame_counter=42, exposure_time=125.5),
        pix_data=telemetry,
    )
    source = np.arange(prod(layout.image_shape), dtype=layout.image_dtype).reshape(
        layout.image_shape
    )
    try:
        connection.push("p1", metadata, source)
    finally:
        connection.CONNECTION["p1"].close()


def test_metadata_defaults_and_nested_value_semantics():
    telemetry = PixData()
    assert telemetry.camera_trigger_seq == -1
    assert telemetry.flight_mode == b"UNKNOWN"
    assert telemetry.system_status == b"UNKNOWN"
    assert telemetry.mission_state == MissionState.UNKNOWN
    assert telemetry.mission_mode == MissionMode.UNKNOWN
    assert AggregatedMetadata().pix_data.camera_trigger_seq == -1
    telemetry.camera_trigger_seq = 42
    telemetry.mission_mode = MissionMode.IN_MISSION
    aggregate = AggregatedMetadata(PhotoExif(frame_counter=42), telemetry)
    telemetry.camera_trigger_seq = 43
    assert aggregate.frame_counter == 42
    assert aggregate.camera_trigger_seq == 42
    assert aggregate.mission_mode == MissionMode.IN_MISSION


def test_aggregate_exposes_flat_fields_in_the_same_memory():
    metadata = AggregatedMetadata(frame_counter=42, lat=45.123, flight_mode=b"AUTO")
    assert metadata.photo_exif.frame_counter == 42
    assert metadata.pix_data.lat == 45.123
    assert metadata.pix_data.flight_mode == b"AUTO"
    metadata.lat = 46.0
    assert metadata.pix_data.lat == 46.0
    metadata.photo_exif.frame_counter = 43
    assert metadata.frame_counter == 43
    layout = get_buffer_layout(AggregatedMetadata, (3, 2), 1)
    with _connection(layout) as connection:
        connection.push("p1", metadata, np.zeros((2, 3), dtype=np.uint8))
        view, image = _read_single(connection, "c1")
    assert view.lat == 46.0
    assert view.frame_counter == 43
    assert image.shape == (2, 3)
    assert AggregatedMetadata.lat.offset == (
        AggregatedMetadata.pix_data.offset + PixData.lat.offset
    )


def test_text_fields_are_bounded_inline_bytes():
    photo = PhotoExif(bsl_color_space=b"sRgb")
    assert photo.bsl_color_space == b"sRgb"
    with pytest.raises(ValueError):
        photo.light_source_preset = b"x" * (TEXT_CAPACITY + 1)
    with pytest.raises(TypeError):
        photo.bsl_color_space = "sRgb"


@pytest.mark.parametrize("metadata_type", [PhotoExif, PixData, AggregatedMetadata])
@pytest.mark.parametrize(
    "resolution,channels,shape,dtype",
    [
        ((5, 3), 1, (3, 5), np.uint8),
        ((4, 2), 3, (2, 4, 3), np.uint8),
        ((2, 3), 1, (3, 2), np.uint16),
        ((4, 2), 4, (2, 4, 4), np.uint8),
        ((5, 3), 3, (3, 5, 3), np.float32),
    ],
)
def test_camera_layout_roundtrip_with_strided_source(
    metadata_type,
    resolution,
    channels,
    shape,
    dtype,
):
    metadata = metadata_type()
    if metadata_type is PhotoExif:
        metadata.frame_counter = 2**40
        metadata.gain = 1.25
    elif metadata_type is PixData:
        metadata.lat = 45.123
        metadata.flight_mode = b"AUTO"
    else:
        metadata.frame_counter = 42
        metadata.alt_rel = 100.5
    layout = get_buffer_layout(metadata_type, resolution, channels, dtype)
    assert layout.image_shape == shape
    assert layout.image_dtype == np.dtype(dtype)
    source = np.arange(prod(shape) * 2, dtype=dtype).reshape(*shape, 2)[..., 0]
    with _connection(layout) as connection:
        assert (
            layout.size == layout.image_offset + prod(shape) * np.dtype(dtype).itemsize
        )
        connection.push("p1", metadata, source)
        returned_metadata, image = _read_single(connection, "c1")
    assert image.flags.c_contiguous
    assert image.flags.aligned
    np.testing.assert_array_equal(image, source)
    assert bytes(returned_metadata) == bytes(metadata)
    source.fill(0)
    assert image.flat[-1] != 0


def test_local_copies_survive_overwrite_and_buffer_close():
    layout = get_buffer_layout(AggregatedMetadata, (3, 2), 1)
    with _connection(layout) as connection:
        connection.push(
            "p1",
            AggregatedMetadata(frame_counter=8, camera_trigger_seq=8, lat=45.123),
            np.full((2, 3), 7, dtype=np.uint8),
        )
        first, first_image = _read_single(connection, "c1")
        second, second_image = _read_single(connection, "c2")
        first.pix_data.lat = 1
        first_image.fill(9)
        assert second.lat == 45.123
        assert not np.shares_memory(first_image, second_image)
        connection.push(
            "p1",
            AggregatedMetadata(frame_counter=99),
            np.zeros((2, 3), dtype=np.uint8),
        )
    # No returned object depends on the shared-memory block.
    assert first.frame_counter == second.frame_counter == 8
    assert first.camera_trigger_seq == second.camera_trigger_seq == 8
    assert second.lat == 45.123
    np.testing.assert_array_equal(first_image, np.full((2, 3), 9))
    np.testing.assert_array_equal(second_image, np.full((2, 3), 7))


def test_spawned_aggregator_reads_telemetry_queue_and_writes_shared_frame():
    ctx = mp.get_context()
    layout = get_buffer_layout(AggregatedMetadata, (4, 3), 3)
    with _connection(layout) as connection:
        queue = ctx.Queue()
        process = ctx.Process(target=_aggregate_in_child, args=(connection, queue))
        try:
            process.start()
            queue.put(PixData(lat=45.123, camera_trigger_seq=42, flight_mode=b"AUTO"))
            process.join(timeout=10)
            assert process.exitcode == 0
            metadata, image = _read_single(connection, "c1")
            np.testing.assert_array_equal(
                image, np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            )
            assert metadata.frame_counter == 42
            assert metadata.exposure_time == 125.5
            assert metadata.lat == 45.123
            assert metadata.camera_trigger_seq == 42
            assert metadata.flight_mode == b"AUTO"
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                process.close()
            queue.close()
            queue.join_thread()
