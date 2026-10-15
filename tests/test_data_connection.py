import multiprocessing as mp
from queue import Empty
from unittest.mock import patch

import numpy as np
import pytest

from common.data_connection import DataConnection
from common.new_structs import (
    AggregatedMetadata,
    DATA_TYPE_REGISTRY,
    PhotoExif,
    PixData,
    get_buffer_layout,
)


def _close_resources(connection, *, unlink=False):
    """The test harness owns cleanup, just as the pipeline will in production."""
    for resource in connection.CONNECTION.values():
        resource.close()
        if connection.connection_type == "shm":
            if unlink:
                resource.unlink()
        else:
            resource.join_thread()


def _publish(connection, producer, values, started, finished):
    started.set()
    try:
        for value in values:
            if connection.connection_type == "fifo":
                connection.push(producer, PixData(lat=value, camera_trigger_seq=value))
            else:
                push_frame(connection, value, producer)
    finally:
        _close_resources(connection)
    finished.set()


def _publish_pair(connection, started, finished):
    started.set()
    try:
        connection.push(
            "p1",
            PhotoExif(frame_counter=7),
            np.full((4, 5), 7, dtype=np.uint16),
        )
    finally:
        _close_resources(connection)
    finished.set()


def _hold_write(connection, started, entered, release, finished):
    """Pause after metadata is written but before pixels are updated."""
    copyto = np.copyto

    def slow_copy(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("writer was not released")
        return copyto(*args, **kwargs)

    started.set()
    try:
        with patch("common.data_connection.np.copyto", side_effect=slow_copy):
            push_frame(connection, 1)
    finally:
        _close_resources(connection)
    finished.set()


def _hold_copy(connection, consumer, expected, started, entered, release, finished):
    """Pause inside copying, where get() must protect both metadata and pixels."""
    ndarray = np.ndarray

    def slow_array(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("reader was not released")
        return ndarray(*args, **kwargs)

    started.set()
    try:
        with patch("common.data_connection.np.ndarray", side_effect=slow_array):
            frames = connection.get(consumer)
        assert [int(frame["image"].flat[0]) for frame in frames] == expected
        for frame in frames:
            np.testing.assert_array_equal(
                frame["image"],
                np.full(frame["image"].shape, frame["metadata"].frame_counter),
            )
    finally:
        # The returned objects may remain alive while SHM handles are closed.
        _close_resources(connection)
    finished.set()


@pytest.fixture
def ctx():
    return mp.get_context()


@pytest.fixture
def connections(ctx):
    owned = []

    def make(
        kind="shm",
        producers=("p1",),
        consumers=("c1",),
        wait=False,
        data_type=None,
        resolution=(5, 4),
        channels=1,
        dtype=np.uint16,
    ):
        data_type = data_type or ("tagged_image" if kind == "shm" else "sensors")
        layout = (
            get_buffer_layout(
                DATA_TYPE_REGISTRY[data_type], resolution, channels, dtype
            )
            if kind == "shm"
            else None
        )
        connection = DataConnection(
            kind,
            wait,
            data_type,
            layout=layout,
        )
        owned.append(connection)
        for producer in producers:
            connection.register_producer(producer)
        for consumer in consumers:
            connection.register_consumer(consumer)
        return connection

    yield make
    for connection in reversed(owned):
        _close_resources(connection, unlink=True)


@pytest.fixture
def processes(ctx, connections):
    active = []

    def start(target, *args):
        process = ctx.Process(target=target, args=args)
        process.start()
        active.append(process)
        return process

    yield start
    for process in reversed(active):
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        process.close()


def start_writer(ctx, processes, connection, producer, values):
    started, finished = ctx.Event(), ctx.Event()
    return (
        processes(_publish, connection, producer, values, started, finished),
        started,
        finished,
    )


def finish(worker):
    process, started, finished = worker
    assert started.wait(5)
    assert finished.wait(5)
    process.join(timeout=5)
    assert process.exitcode == 0


def start_reader(ctx, processes, connection, consumer, expected):
    started, entered, release, finished = (ctx.Event() for _ in range(4))
    process = processes(
        _hold_copy, connection, consumer, expected, started, entered, release, finished
    )
    return process, started, entered, release, finished


def push_frame(connection, value, producer="p1"):
    layout = connection.layout
    connection.push(
        producer,
        layout.metadata_type(frame_counter=value),
        np.full(layout.image_shape, value, dtype=layout.image_dtype),
    )


def read_values(connection, consumer="c1"):
    values = []
    while True:
        try:
            items = connection.get(consumer)
        except Empty:
            return values
        values.extend(
            (
                item.camera_trigger_seq
                if connection.connection_type == "fifo"
                else int(item["image"].flat[0])
            )
            for item in items
        )


def read_single(connection, consumer="c1"):
    frames = connection.get(consumer)
    assert len(frames) == 1
    assert frames[0].keys() == {"metadata", "image"}
    return frames[0]["metadata"], frames[0]["image"]


@pytest.mark.parametrize("kind", ["fifo", "shm"])
@pytest.mark.parametrize(
    "order",
    [
        (
            ("producer", "p1"),
            ("producer", "p2"),
            ("consumer", "c1"),
            ("consumer", "c2"),
        ),
        (
            ("consumer", "c1"),
            ("consumer", "c2"),
            ("producer", "p1"),
            ("producer", "p2"),
        ),
        (
            ("consumer", "c1"),
            ("producer", "p1"),
            ("consumer", "c2"),
            ("producer", "p2"),
        ),
    ],
)
def test_registration_orders_and_broadcast(connections, processes, ctx, kind, order):
    connection = connections(kind, producers=(), consumers=())
    for role, name in order:
        getattr(connection, f"register_{role}")(name)
    assert connection.producers == {"p1", "p2"}
    assert connection.consumers == {"c1", "c2"}
    assert read_values(connection) == []
    workers = [
        start_writer(ctx, processes, connection, producer, [value])
        for producer, value in (("p1", 10), ("p2", 20))
    ]
    for worker in workers:
        finish(worker)
    for consumer in connection.consumers:
        assert sorted(read_values(connection, consumer)) == [10, 20]
        assert read_values(connection, consumer) == []


@pytest.mark.parametrize("producer_count,consumer_count", [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("kind", ["fifo", "shm"])
def test_fifo_preserves_all_items_and_shm_keeps_latest(
    connections,
    processes,
    ctx,
    producer_count,
    consumer_count,
    kind,
):
    connection = connections(
        kind,
        producers=tuple(f"p{i}" for i in range(producer_count)),
        consumers=tuple(f"c{i}" for i in range(consumer_count)),
    )
    workers = [
        start_writer(ctx, processes, connection, f"p{i}", [i * 10, i * 10 + 1])
        for i in range(producer_count)
    ]
    for worker in workers:
        finish(worker)
    expected = [
        value
        for i in range(producer_count)
        for value in ([i * 10, i * 10 + 1] if kind == "fifo" else [i * 10 + 1])
    ]
    for consumer in connection.consumers:
        values = read_values(connection, consumer)
        assert sorted(values) == expected
        if kind == "fifo":
            for i in range(producer_count):
                assert [value for value in values if value // 10 == i] == [
                    i * 10,
                    i * 10 + 1,
                ]


@pytest.mark.parametrize(
    "data_type,metadata_type",
    [
        ("tagged_image", PhotoExif),
        ("aggregator_output", AggregatedMetadata),
        ("resized_aggregator_output", AggregatedMetadata),
    ],
)
@pytest.mark.parametrize(
    "resolution,channels,shape,dtype",
    [
        ((4, 3), 3, (3, 4, 3), np.uint8),
        ((9, 7), 1, (7, 9), np.uint16),
    ],
)
def test_pipeline_dimensions_size_all_producer_buffers(
    connections,
    data_type,
    metadata_type,
    resolution,
    channels,
    shape,
    dtype,
):
    connection = connections(
        data_type=data_type,
        producers=("p1", "p2"),
        resolution=resolution,
        channels=channels,
        dtype=dtype,
    )
    layout = connection.layout
    assert layout.metadata_type is metadata_type
    assert layout.image_shape == shape
    assert layout.image_dtype == np.dtype(dtype)
    assert (
        layout.size == layout.image_offset + np.prod(shape) * np.dtype(dtype).itemsize
    )
    for producer in connection.producers:
        assert connection.CONNECTION[producer].size >= layout.size
        push_frame(connection, 17, producer)
    frames = connection.get("c1")
    assert len(frames) == len(connection.producers)
    for frame in frames:
        metadata, image = frame["metadata"], frame["image"]
        assert image.shape == shape
        assert image.dtype == dtype
        assert isinstance(metadata, metadata_type)
        assert metadata.frame_counter == 17


def test_shm_returns_independent_mutable_copies_without_pickle(connections):
    connection = connections(data_type="aggregator_output", consumers=("c1", "c2"))
    source = np.arange(20, dtype=np.uint16).reshape(4, 5)
    metadata = AggregatedMetadata(frame_counter=42, lat=45.123)
    with (
        patch("pickle.dumps", side_effect=AssertionError("unexpected serialization")),
        patch("pickle.loads", side_effect=AssertionError("unexpected deserialization")),
    ):
        connection.push("p1", metadata, source)
        source.fill(999)
        metadata.lat = 0
        first_metadata, first_image = read_single(connection, "c1")
        assert first_metadata.lat == 45.123
        assert first_image.flags.writeable
        assert first_image.flags.owndata
        assert not connection.events["p1"]["c1"].is_set()
        first_metadata.lat = 1
        first_image.fill(88)
        second_metadata, second_image = read_single(connection, "c2")
        assert second_metadata.lat == 45.123
        np.testing.assert_array_equal(second_image, np.arange(20).reshape(4, 5))
        push_frame(connection, 12)
        assert second_metadata.frame_counter == 42
        np.testing.assert_array_equal(second_image, np.arange(20).reshape(4, 5))


def test_fifo_returns_metadata_image_pair(connections, processes, ctx):
    connection = connections("fifo")
    started, finished = ctx.Event(), ctx.Event()
    worker = processes(_publish_pair, connection, started, finished)
    finish((worker, started, finished))
    items = connection.get("c1")
    assert len(items) == 1
    metadata, image = items[0]
    assert metadata.frame_counter == 7
    np.testing.assert_array_equal(image, np.full((4, 5), 7))
    with pytest.raises(Empty):
        connection.get("c1")


def test_waiting_writer_requires_every_consumer_to_copy(connections, processes, ctx):
    connection = connections(consumers=("c1", "c2"), wait=True)
    push_frame(connection, 0)
    writer = start_writer(ctx, processes, connection, "p1", [1])
    assert writer[1].wait(5)
    assert not writer[2].wait(0.1)
    first_metadata, first_image = read_single(connection, "c1")
    assert not connection.events["p1"]["c1"].is_set()
    assert not writer[2].wait(0.1)
    second_metadata, second_image = read_single(connection, "c2")
    finish(writer)
    # Local objects survive the next publication, with no explicit release.
    assert first_metadata.frame_counter == second_metadata.frame_counter == 0
    assert not np.any(first_image)
    assert not np.any(second_image)
    assert read_values(connection, "c1") == [1]
    assert read_values(connection, "c2") == [1]


def test_waiting_writer_preserves_every_publication(connections, processes, ctx):
    connection = connections(wait=True)
    push_frame(connection, 0)
    writer = start_writer(ctx, processes, connection, "p1", range(1, 21))
    for value in range(21):
        assert connection.events["p1"]["c1"].wait(5)
        metadata, image = read_single(connection, "c1")
        assert metadata.frame_counter == value
        assert int(image.flat[0]) == value
    finish(writer)


def test_readers_of_the_same_buffer_take_turns(connections, processes, ctx):
    connection = connections(consumers=("c1", "c2"))
    push_frame(connection, 0)
    first = start_reader(ctx, processes, connection, "c1", [0])
    assert first[2].wait(5)
    second = start_reader(ctx, processes, connection, "c2", [0])
    assert second[1].wait(5)
    assert not second[2].wait(0.1)
    first[3].set()
    finish((first[0], first[1], first[4]))
    assert second[2].wait(5)
    second[3].set()
    finish((second[0], second[1], second[4]))


def test_reader_and_writer_exclude_each_other(connections, processes, ctx):
    connection = connections(consumers=("c1", "c2"))
    push_frame(connection, 0)
    first = start_reader(ctx, processes, connection, "c1", [0])
    assert first[2].wait(5)
    started, entered, release, finished = (ctx.Event() for _ in range(4))
    writer = processes(_hold_write, connection, started, entered, release, finished)
    assert started.wait(5)
    assert not entered.wait(0.1)
    first[3].set()
    finish((first[0], first[1], first[4]))
    assert entered.wait(5)

    # The writer has updated metadata but not pixels: readers must stay out.
    late = start_reader(ctx, processes, connection, "c2", [1])
    assert late[1].wait(5)
    assert not late[2].wait(0.1)
    release.set()
    finish((writer, started, finished))
    assert late[2].wait(5)
    late[3].set()
    finish((late[0], late[1], late[4]))
    assert not connection.events["p1"]["c2"].is_set()


@pytest.mark.parametrize("wait", [False, True])
def test_copying_reader_does_not_block_another_producer(
    connections, processes, ctx, wait
):
    connection = connections(producers=("p1", "p2"), wait=wait)
    push_frame(connection, 0)
    reader = start_reader(ctx, processes, connection, "c1", [0, 10])
    assert reader[2].wait(5)
    finish(start_writer(ctx, processes, connection, "p2", [10]))
    reader[3].set()
    finish((reader[0], reader[1], reader[4]))
    assert read_values(connection) == []


def test_failed_copy_releases_lock_and_leaves_frame_pending(
    connections, processes, ctx
):
    connection = connections()
    push_frame(connection, 0)
    with patch("common.data_connection.np.ndarray", side_effect=MemoryError):
        with pytest.raises(MemoryError):
            connection.get("c1")
    assert connection.events["p1"]["c1"].is_set()
    finish(start_writer(ctx, processes, connection, "p1", [1]))
    assert read_values(connection) == [1]


def test_failed_write_releases_lock_without_publishing(connections, processes, ctx):
    connection = connections(consumers=("c1", "c2"))
    with patch("common.data_connection.np.copyto", side_effect=RuntimeError):
        with pytest.raises(RuntimeError):
            push_frame(connection, 0)
    assert not any(event.is_set() for event in connection.events["p1"].values())
    finish(start_writer(ctx, processes, connection, "p1", [1]))
    assert read_values(connection, "c1") == [1]
    assert read_values(connection, "c2") == [1]


def test_registry_can_be_extended_for_another_logical_name(connections, monkeypatch):
    monkeypatch.setitem(DATA_TYPE_REGISTRY, "another_camera", PhotoExif)
    connection = connections(data_type="another_camera")
    push_frame(connection, 8)
    assert read_values(connection) == [8]


@pytest.mark.parametrize("kind", ["fifo", "shm"])
def test_empty_get_raises_empty(connections, kind):
    connection = connections(kind)
    with pytest.raises(Empty):
        connection.get("c1")


def test_get_copies_all_ready_producers_and_skips_idle_ones(connections):
    connection = connections(producers=("p1", "p2", "idle"), consumers=("c1", "c2"))
    push_frame(connection, 1, "p1")
    push_frame(connection, 2, "p2")
    frames = connection.get("c1")
    assert len(frames) == 2
    assert [frame["metadata"].frame_counter for frame in frames] == [1, 2]
    for frame in frames:
        assert frame.keys() == {"metadata", "image"}
        np.testing.assert_array_equal(
            frame["image"], np.full((4, 5), frame["metadata"].frame_counter)
        )
    assert not connection.events["p1"]["c1"].is_set()
    assert not connection.events["p2"]["c1"].is_set()
    assert connection.events["p1"]["c2"].is_set()
    assert connection.events["p2"]["c2"].is_set()
    with pytest.raises(Empty):
        connection.get("c1")
    push_frame(connection, 3, "p1")
    assert read_values(connection) == [3]
    assert read_values(connection, "c2") == [3, 2]
    # The batch's copies remain intact after subsequent writes.
    assert [int(frame["image"].flat[0]) for frame in frames] == [1, 2]


def test_fifo_batch_limit_and_partial_batch(connections, processes, ctx):
    connection = connections("fifo")
    finish(start_writer(ctx, processes, connection, "p1", range(12)))
    assert [item.camera_trigger_seq for item in connection.get("c1")] == list(range(10))
    assert [item.camera_trigger_seq for item in connection.get("c1")] == [10, 11]
    with pytest.raises(Empty):
        connection.get("c1")
