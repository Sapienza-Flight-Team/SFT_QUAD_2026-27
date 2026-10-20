import ctypes as ct
import struct
from contextlib import ExitStack
from queue import Queue
from threading import Barrier, Event, Thread

import pytest

from core.central_registry import ARRAY_LENGTH, BUFFER_LENGTH, MATCHING_WINDOW, CentralRegistry


class Image(ct.Structure):
    _fields_ = [("ID", ct.c_uint64), ("value", ct.c_uint64)]


class Sensors(ct.Structure):
    _fields_ = [("value", ct.c_uint64), ("ID", ct.c_int64)]


ALL = frozenset({Image, Sensors})


@pytest.fixture
def registry():
    registry = CentralRegistry(matching_window=1)
    registry.register_producer("camera", Image)
    registry.register_producer("controller", Sensors)
    yield registry
    for helper in registry.data_array.values():
        helper.close()


def fill(datatype, view, capture_id):
    record = datatype.from_buffer(view)
    record.value = 123
    record.ID = capture_id


def publish(registry, capture_id, datatype=ALL):
    for primitive in datatype if isinstance(datatype, frozenset) else (datatype,):
        with registry.lease_slot_for_writing(primitive) as view:
            fill(primitive, view, capture_id)


def ids(buffers):
    return {struct.unpack_from("<q", view, primitive.ID.offset)[0]
            for primitive, view in buffers.items()}


def subscribe(registry, consumer="reader", datatype=ALL):
    registry.register_consumer(consumer, datatype)
    required = datatype if isinstance(datatype, frozenset) else frozenset({datatype})
    return registry.events[required][consumer]


def free_counts(registry):
    return {primitive: len(helper.free_slots) for primitive, helper in registry.data_array.items()}


@pytest.mark.parametrize("first,second", [(Image, Sensors), (Sensors, Image)])
def test_group_waits_for_matching_capture(registry, first, second):
    event = subscribe(registry)
    publish(registry, 10, first)
    publish(registry, 9, second)
    assert not event.is_set()
    publish(registry, 10, second)
    assert event.is_set()
    with registry.lease_slot_for_reading(ALL, "reader") as buffers:
        assert ids(buffers) == {10}
    assert not event.is_set()
    assert registry.entries[10][first] != registry.entries[10][second]


def test_independent_readers_and_read_only_shared_views(registry):
    first = subscribe(registry, "first")
    second = subscribe(registry, "second")
    publish(registry, 1)
    with registry.lease_slot_for_reading(ALL, "first") as a:
        assert not first.is_set() and second.is_set()
        with registry.lease_slot_for_reading(ALL, "second") as b:
            assert ids(a) == ids(b) == {1}
            block, _ = registry.data_array[Image].slots[registry.entries[1][Image]]
            assert a[Image].obj is block.buf.obj
            with pytest.raises(TypeError):
                a[Image][0] = 0
        with pytest.raises(ValueError):
            len(b[Image])
    assert not first.is_set() and not second.is_set()


def test_publication_during_read_is_not_lost(registry):
    event = subscribe(registry)
    publish(registry, 1)
    with registry.lease_slot_for_reading(ALL, "reader") as buffers:
        publish(registry, 2)
        assert ids(buffers) == {1}
        assert event.is_set()
    assert event.is_set()
    with registry.lease_slot_for_reading(ALL, "reader") as buffers:
        assert ids(buffers) == {2}
    assert not event.is_set()


@pytest.mark.parametrize("newer", [False, True])
def test_failed_read_can_retry(registry, newer):
    event = subscribe(registry)
    publish(registry, 1)
    with pytest.raises(RuntimeError):
        with registry.lease_slot_for_reading(ALL, "reader"):
            if newer:
                publish(registry, 2)
            raise RuntimeError("processing failed")
    assert event.is_set()
    with registry.lease_slot_for_reading(ALL, "reader") as buffers:
        assert ids(buffers) == {2 if newer else 1}


def test_failed_read_waits_if_the_newest_match_was_evicted(registry):
    event = subscribe(registry)
    publish(registry, 1)
    with pytest.raises(RuntimeError):
        with registry.lease_slot_for_reading(ALL, "reader") as held:
            publish(registry, 2)
            publish(registry, 3, Image)
            assert ids(held) == {1}
            assert not event.is_set()
            raise RuntimeError("processing failed")
    assert not event.is_set()
    publish(registry, 3, Sensors)
    with registry.lease_slot_for_reading(ALL, "reader") as matched:
        assert ids(matched) == {3}


def test_reader_sleeps_until_group_is_ready(registry, monkeypatch):
    event = subscribe(registry)
    waiting = Event()
    wait = event.wait
    output = Queue()

    def observed_wait():
        waiting.set()
        return wait()

    def read():
        with registry.lease_slot_for_reading(ALL, "reader") as buffers:
            output.put(ids(buffers))

    monkeypatch.setattr(event, "wait", observed_wait)
    worker = Thread(target=read, daemon=True)
    worker.start()
    assert waiting.wait(2)
    publish(registry, 3, Image)
    assert output.empty() and not event.is_set()
    publish(registry, 3, Sensors)
    worker.join(2)
    assert not worker.is_alive()
    assert output.get_nowait() == {3}


def test_eviction_skips_readers_and_reuses_released_tail(registry):
    subscribe(registry)
    capacity = registry.array_length
    publish(registry, 0)
    with registry.lease_slot_for_reading(ALL, "reader") as buffers:
        for capture_id in range(1, capacity + 1):
            publish(registry, capture_id)
        assert ids(buffers) == {0}
        assert 0 in registry.entries and 1 not in registry.entries
        assert next(iter(registry.entries)) == 0
    publish(registry, capacity + 1)
    assert 0 not in registry.entries
    assert len(registry.entries) == capacity
    assert set(free_counts(registry).values()) == {1}
    assert list(registry.entries) == list(range(2, capacity + 2))


def test_all_consumers_can_hold_distinct_entries_while_writing_continues(registry):
    consumers = 6
    for index in range(consumers):
        subscribe(registry, str(index))
    with ExitStack() as leases:
        held = []
        for capture_id in range(consumers):
            publish(registry, capture_id)
            held.append(leases.enter_context(registry.lease_slot_for_reading(ALL, str(capture_id))))
        for capture_id in range(consumers, consumers + 100):
            publish(registry, capture_id)
        assert [ids(buffers) for buffers in held] == [{i} for i in range(consumers)]
        assert registry.array_length == consumers + 1
        assert registry.buffer_length == consumers + 2
        assert len(registry.entries) == consumers + 1
        assert set(free_counts(registry).values()) == {1}


def test_latest_is_capture_order_and_survives_eviction(registry):
    event = subscribe(registry)
    second_event = subscribe(registry, "second")
    publish(registry, 5, Image)
    publish(registry, 10)
    with registry.lease_slot_for_reading(ALL, "reader") as buffers:
        assert ids(buffers) == {10}
    publish(registry, 5, Sensors)
    assert not event.is_set()
    for capture_id in range(11, 11 + registry.array_length):
        publish(registry, capture_id, Image)
    assert 10 not in registry.entries and registry.latest[ALL] == 10
    before = free_counts(registry)
    publish(registry, 9, Sensors)
    assert 9 not in registry.entries
    assert free_counts(registry) == before
    assert not second_event.is_set() and not event.is_set()
    publish(registry, 11, Sensors)
    assert second_event.is_set() and event.is_set()


def test_matching_another_primitive_during_read(registry):
    subscribe(registry, datatype=Image)
    event = subscribe(registry, "group")
    publish(registry, 7, Image)
    with registry.lease_slot_for_reading(Image, "reader") as buffers:
        publish(registry, 7, Sensors)
        assert ids(buffers) == {7}
    assert event.is_set()
    with registry.lease_slot_for_reading(ALL, "group") as buffers:
        assert ids(buffers) == {7}


def test_failed_writes_release_buffers(registry):
    publish(registry, 1)
    before = free_counts(registry)
    with pytest.raises(RuntimeError):
        with registry.lease_slot_for_writing(Sensors) as view:
            fill(Sensors, view, 2)
            raise RuntimeError("capture failed")
    assert free_counts(registry) == before
    assert set(registry.entries) == {1}
    with pytest.raises(ValueError):
        len(view)


@pytest.mark.parametrize("datatype", [ALL, set(ALL)])
def test_writes_require_a_primitive(registry, datatype):
    before = free_counts(registry)
    with pytest.raises((KeyError, TypeError)):
        with registry.lease_slot_for_writing(datatype):
            pass
    assert not registry.entries
    assert free_counts(registry) == before


def test_concurrent_producers_and_reader(registry):
    subscribe(registry)
    barrier = Barrier(2)
    errors, received = Queue(), []
    last = 59

    def produce(primitive):
        try:
            for capture_id in range(last + 1):
                publish(registry, capture_id, primitive)
                barrier.wait(5)
        except BaseException as error:
            errors.put(error)
            barrier.abort()

    def consume():
        try:
            while not received or received[-1] != last:
                with registry.lease_slot_for_reading(ALL, "reader") as buffers:
                    capture_ids = ids(buffers)
                    assert len(capture_ids) == 1
                    received.append(capture_ids.pop())
        except BaseException as error:
            errors.put(error)

    workers = [Thread(target=consume, daemon=True)] + [
        Thread(target=produce, args=(primitive,), daemon=True) for primitive in ALL
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    assert errors.empty(), list(errors.queue)
    assert not any(worker.is_alive() for worker in workers)
    assert received == sorted(set(received)) and received[-1] == last
    assert set(free_counts(registry).values()) == {1}


def test_capacity_grows_per_consumer_and_per_new_datatype(registry):
    assert registry.array_length == ARRAY_LENGTH == 1
    assert registry.buffer_length == BUFFER_LENGTH == 2
    subscribe(registry, "first", Image)
    subscribe(registry, "first", ALL)  # One consumer with two subscriptions.
    assert registry.array_length == 2 and registry.buffer_length == 3
    subscribe(registry, "second")
    assert registry.array_length == 3 and registry.buffer_length == 4

    class Extra(ct.Structure):
        _fields_ = [("ID", ct.c_uint64), ("value", ct.c_uint64)]

    registry.register_producer("extra", Extra)
    for helper in registry.data_array.values():
        assert len(helper.slots) == 4
        assert sum(block.size for block in helper.buffers) == 4 * helper.cell_size
    assert ARRAY_LENGTH == 1 and BUFFER_LENGTH == 2


def test_live_growth_preserves_reader_writer_views_and_buffer_names(registry):
    subscribe(registry, "first")
    publish(registry, 0)
    with registry.lease_slot_for_reading(ALL, "first") as held:
        locations = {kind: registry.data_array[kind].slot_location(registry.entries[0][kind])
                     for kind in ALL}
        publish(registry, 1)
        with registry.lease_slot_for_writing(Image) as writing:
            fill(Image, writing, 2)
            address = ct.addressof(ct.c_char.from_buffer(writing))
            subscribe(registry, "second")
            assert ct.addressof(ct.c_char.from_buffer(writing)) == address
            assert ids(held) == {0}
        publish(registry, 2, Sensors)
        for kind, location in locations.items():
            assert registry.data_array[kind].slot_location(registry.entries[0][kind]) == location
        with registry.lease_slot_for_reading(ALL, "second") as newest:
            assert ids(newest) == {2}
        assert ids(held) == {0}


def test_live_producer_registration_can_complete_an_existing_capture(registry):
    class Extra(ct.Structure):
        _fields_ = [("ID", ct.c_uint64), ("value", ct.c_uint64)]

    subscribe(registry, "first")
    publish(registry, 4)
    with registry.lease_slot_for_reading(ALL, "first") as held:
        registry.register_producer("extra", Extra)
        combined = ALL | {Extra}
        event = subscribe(registry, "second", combined)
        assert not event.is_set()
        publish(registry, 4, Extra)
        assert event.is_set() and ids(held) == {4}
        with registry.lease_slot_for_reading(combined, "second") as newest:
            assert ids(newest) == {4}


def test_failed_write_does_not_evict_a_retained_capture(registry):
    event = subscribe(registry)
    publish(registry, 0)
    with registry.lease_slot_for_reading(ALL, "reader") as held:
        publish(registry, 1)
        with pytest.raises(RuntimeError):
            with registry.lease_slot_for_writing(Image) as view:
                assert 1 in registry.entries and event.is_set()
                fill(Image, view, 2)
                raise RuntimeError("capture failed")
        assert ids(held) == {0}
        assert set(free_counts(registry).values()) == {1}
        publish(registry, 3)
        assert event.is_set()


def test_subscriptions_before_producers_and_independent_groups():
    registry = CentralRegistry(matching_window=1)
    try:
        image = subscribe(registry, "reader", Image)
        combined = subscribe(registry, "reader", ALL)
        registry.register_producer("camera", Image)
        registry.register_producer("controller", Sensors)
        assert registry.buffer_length == 3
        assert all(len(helper.buffers) == 1 for helper in registry.data_array.values())
        publish(registry, 7, Image)
        assert image.is_set() and not combined.is_set()
        with registry.lease_slot_for_reading(Image, "reader") as buffers:
            assert ids(buffers) == {7}
            publish(registry, 7, Sensors)
            assert combined.is_set()
        with registry.lease_slot_for_reading(ALL, "reader") as buffers:
            assert ids(buffers) == {7}
        assert not image.is_set() and not combined.is_set()
    finally:
        for helper in registry.data_array.values():
            helper.close()


def test_primitive_reader_can_wait_before_its_producer_exists():
    registry = CentralRegistry()
    event = subscribe(registry, datatype=Image)
    waiting, output = Event(), Queue()
    original_wait = event.wait

    def wait():
        waiting.set()
        return original_wait()

    def read():
        with registry.lease_slot_for_reading(Image, "reader") as buffers:
            output.put(ids(buffers))

    event.wait = wait
    worker = Thread(target=read, daemon=True)
    worker.start()
    try:
        assert waiting.wait(2)
        registry.register_producer("camera", Image)
        publish(registry, 7, Image)
        worker.join(2)
        assert not worker.is_alive() and output.get_nowait() == {7}
    finally:
        for helper in registry.data_array.values():
            helper.close()


def test_group_can_subscribe_to_a_producer_registered_later(registry):
    class Extra(ct.Structure):
        _fields_ = [("value", ct.c_uint16), ("counter", ct.c_uint32)]

    Extra.ID = Extra.counter
    combined = ALL | {Extra}
    event = subscribe(registry, datatype=combined)
    publish(registry, 7)
    assert not event.is_set()
    registry.register_producer("extra", Extra)
    publish(registry, 7, Extra)
    assert event.is_set()
    with registry.lease_slot_for_reading(combined, "reader") as buffers:
        assert struct.unpack_from("@I", buffers[Extra], Extra.ID.offset)[0] == 7
        assert ids({kind: buffers[kind] for kind in ALL}) == {7}


def test_reintroduced_id_does_not_restore_evicted_primitive(registry):
    image = subscribe(registry, datatype=Image)
    publish(registry, 14, Image)
    publish(registry, 15, Sensors)
    publish(registry, 16, Sensors)
    assert 14 not in registry.entries and not image.is_set()
    publish(registry, 14, Sensors)
    late_image = subscribe(registry, "late", Image)
    assert registry.latest[frozenset({Image})] == 14
    assert 14 not in registry.entries
    assert not image.is_set() and not late_image.is_set()


def test_missing_older_id_is_rejected_before_any_complete_packet(registry):
    subscribe(registry)
    for capture_id in (10, 11, 12):
        publish(registry, capture_id, Image)
    before = free_counts(registry)
    publish(registry, 10, Sensors)
    assert list(registry.entries) == [11, 12]
    assert registry.latest_id == 12 and registry.latest[ALL] is None
    assert free_counts(registry) == before


def test_existing_older_id_can_still_receive_missing_fields(registry):
    event = subscribe(registry)
    publish(registry, 10, Image)
    publish(registry, 11, Image)
    publish(registry, 10, Sensors)
    assert registry.latest_id == 11 and event.is_set()
    with registry.lease_slot_for_reading(ALL, "reader") as buffers:
        assert ids(buffers) == {10}


def test_matching_recovers_after_reader_releases_with_skewed_producers(registry):
    subscribe(registry, "dummy")
    event = subscribe(registry, "viewer")
    publish(registry, 0, Sensors)
    publish(registry, 1, Sensors)
    publish(registry, 0, Image)
    with registry.lease_slot_for_reading(ALL, "dummy") as oldest:
        for capture_id in (2, 3, 4):
            publish(registry, capture_id, Sensors)
            publish(registry, capture_id - 1, Image)
        with registry.lease_slot_for_reading(ALL, "viewer"):
            publish(registry, 5, Sensors)
            publish(registry, 4, Image)
            publish(registry, 6, Sensors)
        publish(registry, 5, Image)
        for capture_id in range(7, 11):
            publish(registry, capture_id, Sensors)
            publish(registry, capture_id - 1, Image)
        assert ids(oldest) == {0}
        assert registry.latest[ALL] == 9 and event.is_set()
        with registry.lease_slot_for_reading(ALL, "viewer") as buffers:
            assert ids(buffers) == {9}


@pytest.mark.parametrize("first,second", [(Image, Sensors), (Sensors, Image)])
def test_delayed_matching_uses_its_window_even_with_reader_held_captures(first, second):
    registry = CentralRegistry()
    for consumer in ("held_a", "held_b", "viewer"):
        subscribe(registry, consumer)
    for primitive in ALL:
        registry.register_producer(primitive.__name__, primitive)
    try:
        with ExitStack() as leases:
            held = []
            for capture_id, consumer in enumerate(("held_a", "held_b")):
                publish(registry, capture_id)
                held.append(leases.enter_context(registry.lease_slot_for_reading(ALL, consumer)))
            for capture_id in range(2, 2 + MATCHING_WINDOW):
                publish(registry, capture_id, first)
            for capture_id in range(2, 80):
                publish(registry, capture_id, second)
                with registry.lease_slot_for_reading(ALL, "viewer") as matched:
                    publish(registry, capture_id + MATCHING_WINDOW, first)
                    assert ids(matched) == {capture_id}
                    assert [ids(buffers) for buffers in held] == [{0}, {1}]
            assert registry.latest[ALL] == 79
            assert len(registry.entries) == registry.array_length == 3 + MATCHING_WINDOW
            assert registry.buffer_length == registry.array_length + 1
    finally:
        registry.close()


def test_delay_beyond_retention_discards_missing_id_but_keeps_existing_matches():
    registry = CentralRegistry()
    subscribe(registry)
    for primitive in ALL:
        registry.register_producer(primitive.__name__, primitive)
    try:
        for capture_id in range(registry.array_length + 3):
            publish(registry, capture_id, Image)
        before = list(registry.entries), free_counts(registry)
        publish(registry, 0, Sensors)
        assert (list(registry.entries), free_counts(registry)) == before
        assert registry.latest[ALL] is None
        retained_id = next(iter(registry.entries))
        publish(registry, retained_id, Sensors)
        with registry.lease_slot_for_reading(ALL, "reader") as buffers:
            assert ids(buffers) == {retained_id}
    finally:
        registry.close()


def test_mixed_groups_and_out_of_order_arrivals_match_reference():
    from random import Random
    from tools._registry_reference import CentralRegistry as ReferenceRegistry

    class Extra(ct.Structure):
        _fields_ = [("ID", ct.c_uint64), ("value", ct.c_uint64)]

    primitives = (Image, Sensors, Extra)
    groups = (Image, Sensors, Extra, ALL, frozenset({Image, Extra}), frozenset(primitives))
    registries = (CentralRegistry(), ReferenceRegistry())
    random = Random(73)
    try:
        for registry in registries:
            for consumer, group in enumerate(groups):
                subscribe(registry, consumer, group)
            for primitive in primitives:
                registry.register_producer(primitive.__name__, primitive)
        arrivals = [(capture_id, primitive) for capture_id in range(50) for primitive in primitives]
        # Small overlapping batches exercise matching as well as rejected late captures.
        for start in range(0, len(arrivals), 15):
            batch = arrivals[start:start + 15]
            random.shuffle(batch)
            for capture_id, primitive in batch:
                for registry in registries:
                    publish(registry, capture_id, primitive)
                assert list(registries[0].entries) == list(registries[1].entries)
                for consumer, group in enumerate(groups):
                    required = group if isinstance(group, frozenset) else frozenset({group})
                    assert registries[0].latest[required] == registries[1].latest[required]
                    ready = [registry.events[required][consumer].is_set() for registry in registries]
                    assert ready[0] == ready[1]
                    if ready[0] and random.random() < .3:
                        for registry in registries:
                            with registry.lease_slot_for_reading(group, consumer) as buffers:
                                assert ids(buffers) == {registry.latest[required]}
    finally:
        for registry in registries:
            for helper in registry.data_array.values():
                helper.close()


@pytest.mark.parametrize("primitive", [Image, Sensors])
def test_repeated_id_zero_does_not_exhaust_slots_or_replace_published_data(primitive):
    registry = CentralRegistry()
    registry.register_consumer("reader", primitive)
    registry.register_producer("producer", primitive)
    helper = registry.data_array[primitive]
    try:
        assert len(helper.slots) == 18
        publish(registry, 0, primitive)
        original_slot = registry.entries[0][primitive]
        for _ in range(100):
            with registry.lease_slot_for_writing(primitive) as view:
                record = primitive.from_buffer(view)
                record.ID, record.value = 0, 999
                del record
            assert registry.entries[0][primitive] == original_slot
            assert len(helper.free_slots) == 17
        with registry.lease_slot_for_reading(primitive, "reader") as views:
            assert struct.unpack_from("@Q", views[primitive], primitive.value.offset)[0] == 123
        publish(registry, 1, primitive)
        with registry.lease_slot_for_reading(primitive, "reader") as views:
            assert ids(views) == {1}
    finally:
        registry.close()


def test_duplicate_while_reader_holds_data_preserves_original_and_slot_budget(registry):
    subscribe(registry)
    publish(registry, 0)
    original_slots = dict(registry.entries[0])
    available = free_counts(registry)
    with registry.lease_slot_for_reading(ALL, "reader") as views:
        for _ in range(50):
            publish(registry, 0)
        assert dict(registry.entries[0]) == original_slots
        assert free_counts(registry) == available
        assert ids(views) == {0}
        assert not registry.events[ALL]["reader"].is_set()
