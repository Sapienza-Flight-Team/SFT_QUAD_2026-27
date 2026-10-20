"""Bounded, zero-copy input aggregation by capture ID.

One producer per primitive and one active write lease; each primitive is written
once per nonnegative ID.
Consumers hold one read lease; views stay inside it. Metadata and events stay
with the owner; stages attach to shared memory through registry pipes. Storage
can grow without moving data. _record_data logs inputs before matching or discard.

See docs/central_registry.md for worked mask examples, lease lifetimes, capacity
budgets, and the packed pipe protocol. The caches here keep that work out of the
per-frame path; changing them requires a performance comparison.
"""

from collections import OrderedDict, deque
from contextlib import contextmanager
from multiprocessing import Pipe, shared_memory
from struct import Struct
from threading import Event, Lock, Thread
import ctypes as ct
import logging

ARRAY_LENGTH = 1  # Minimum retained entries.
BUFFER_LENGTH = 2  # Minimum cells per stripe.
MATCHING_WINDOW = 16  # Recent captures retained in addition to reader-held entries.


def _primitives(datatype):
    return frozenset(datatype) if isinstance(datatype, (set, frozenset)) else frozenset({datatype})


class SharedMemoryHelper:
    """Shared payload storage; the registry serializes access to the local slot pool."""

    def __init__(self, datatype, length):
        self.cell_size = ct.sizeof(datatype)
        # Decode only the ID field, even for a struct containing a large image.
        # Native-width unsigned formats also cover nonnegative signed ID fields.
        self.id_offset = datatype.ID.offset
        self.id_reader = Struct({1: "B", 2: "H", 4: "I", 8: "Q"}[datatype.ID.size])
        self.buffers = []
        self.slots = []  # slot -> (shared-memory block, byte offset).
        self.free_slots = deque()
        self.grow(length)

    def grow(self, length):
        """Append storage; existing views and shared-memory names remain valid."""
        start = len(self.slots)
        if length <= start:
            return
        buffer = shared_memory.SharedMemory(create=True, size=(length - start) * self.cell_size)
        self.buffers.append(buffer)
        self.slots.extend((buffer, i * self.cell_size) for i in range(length - start))
        self.free_slots.extend(range(start, length))

    def reserve_slot(self):
        """Take the spare cell guaranteed by the registry's capacity budget."""
        # One active writer per primitive makes this the current transport slot.
        slot = self.writing_slot = self.free_slots.popleft()
        return slot, self.slot_view(slot)

    def release_slot(self, slot):
        self.free_slots.append(slot)

    def slot_view(self, slot):
        buffer, start = self.slots[slot]
        return buffer.buf[start : start + self.cell_size]

    def slot_location(self, slot):
        buffer, start = self.slots[slot]
        return buffer.name, start

    def close(self):
        for buffer in self.buffers:
            buffer.close()
            buffer.unlink()


class Entry(dict):
    """One capture: primitive -> stripe slot, with three small metadata fields.

    readers pins every cell belonging to this capture against eviction.
    mask has one bit set per published primitive; it mirrors the dictionary keys.
    latest_groups contains only groups whose latest ID points to this entry,
    so eviction can clear their events without scanning every tracked group.
    """

    __slots__ = ("readers", "mask", "latest_groups")

    def __init__(self):
        self.readers = self.mask = 0
        self.latest_groups = set()


class CentralRegistry:
    def __init__(self, logger=None, matching_window=MATCHING_WINDOW):
        self.logger = logger if logger is not None else logging.getLogger("sft_cv.registry")
        self.matching_window = matching_window
        self.array_length = max(ARRAY_LENGTH, matching_window)
        self.buffer_length = max(BUFFER_LENGTH, self.array_length + 1)
        self.data_array = dict()  # primitive -> shared-memory stripe.
        self.entries = OrderedDict()  # capture_id -> Entry; oldest arrival first.
        self.latest_id = -1  # Highest received ID, including incomplete captures.
        self._lock = Lock()  # Metadata and slot pools belong to the registry process.

        self.latest = dict()  # group -> highest complete ID; survives eviction.
        self.events = dict()  # group -> {consumer: unread-data Event}.
        self.last_read = dict()  # (consumer, required_datatypes) -> capture_id.
        self.producers = dict()  # primitive datatype -> producer ID.
        self._consumers = set()
        # Bits never change: 1 << n assigns primitive n its own binary position.
        # Group masks OR those bits together; entry masks record present fields.
        self._bits = dict()  # primitive -> presence bit.
        self._masks = dict()  # required_datatypes -> presence mask.
        self._affected = dict()  # primitive -> groups to check on publication.
        self._singletons = dict()  # primitive -> normalized specification.
        self._connections = []

    def _track_group(self, required):
        """Compile presence tests and publication indexes once per group.

        Example: Image=001, Sensors=010 gives {Image, Sensors} the mask 011.
        An entry matches exactly when (entry.mask & 011) == 011. Extra bits are
        allowed. A group may be registered before its producers: its bits are
        assigned now and reused when those producers register.
        """
        if required in self._masks:
            return
        if len(required) == 1:
            self._singletons[next(iter(required))] = required
        mask = 0
        for primitive in required:
            # setdefault keeps an existing bit; a new primitive gets the next
            # unused power of two. OR combines requirements without duplicates.
            mask |= self._bits.setdefault(primitive, 1 << len(self._bits))
            self._affected.setdefault(primitive, []).append(required)
        self._masks[required] = mask
        # Live registration must also discover matches published before it.
        latest = max((capture_id for capture_id, entry in self.entries.items()
                      if entry.mask & mask == mask), default=None)
        self.latest[required] = latest
        if latest is not None:
            self.entries[latest].latest_groups.add(required)

    def _signal(self, required, consumer):
        """Set readiness only for a retained match newer than this subscriber's cursor."""
        capture_id = self.latest[required]
        ready = capture_id in self.entries and capture_id > self.last_read[consumer, required]
        event = self.events[required][consumer]
        if ready != event.is_set():
            event.set() if ready else event.clear()

    def register_consumer(self, consumer, datatype):
        """Register a latest-only subscription for a primitive or set of primitives."""
        required = _primitives(datatype)
        with self._lock:
            self.events.setdefault(required, {})[consumer] = Event()
            self.last_read[consumer, required] = -1
            self._consumers.add(consumer)
            # Count stages, not subscriptions: each consumer holds one read.
            self.array_length = max(ARRAY_LENGTH, len(self._consumers) + self.matching_window)
            self.buffer_length = max(BUFFER_LENGTH, self.array_length + 1)
            for helper in self.data_array.values():
                helper.grow(self.buffer_length)
            self._track_group(required)
            self._signal(required, consumer)
            self.logger.info("[registry] Consumer %s subscribed to %s; entry capacity=%d",
                             consumer, ", ".join(kind.__name__ for kind in required), self.array_length)

    def register_producer(self, producer, datatype):
        """Register one producer for a primitive ctypes datatype."""
        with self._lock:
            helper = self.data_array[datatype] = SharedMemoryHelper(datatype, self.buffer_length)
            self.producers[datatype] = producer
            self._track_group(frozenset({datatype}))
            self._track_group(frozenset(self.data_array))
            helper.bit = self._bits[datatype]
            self.logger.info("[registry] Producer %s registered for %s: %d slots x %d bytes, memory=%s",
                             producer, datatype.__name__, len(helper.slots), helper.cell_size,
                             helper.buffers[0].name)

    def _evict(self):
        """Free the oldest entry with no readers."""
        for capture_id, entry in self.entries.items():
            if entry.readers:
                continue
            del self.entries[capture_id]
            for primitive, slot in entry.items():
                self.data_array[primitive].release_slot(slot)
            # Keep latest[group] as a high-water mark after eviction. Clear the
            # event because its payload is gone; older matches cannot replace it.
            for required in entry.latest_groups:
                for event in self.events.get(required, {}).values():
                    event.clear()
            self.logger.debug("[registry] Evicted capture %d; freed slots=%s", capture_id, entry)
            return
        self.logger.error("[registry] No entry without readers available to reclaim")
        raise BufferError("No entry without readers available to reclaim")

    def _record_data(self, datatype, capture_id, slot, buffer):
        """Temporary recorder; persist this leased buffer before returning."""
        self.logger.info("[registry:data] producer=%s datatype=%s capture_id=%d slot=%d bytes=%d",
                         self.producers[datatype], datatype.__name__, capture_id, slot, len(buffer))

    @contextmanager
    def lease_slot_for_writing(self, datatype):
        """Yield one primitive's writable view; associate it by its ID on success."""
        helper = self.data_array[datatype]
        with self._lock:
            slot, view = helper.reserve_slot()
        published = False
        try:
            yield view

            capture_id = helper.id_reader.unpack_from(view, helper.id_offset)[0]
            self._record_data(datatype, capture_id, slot, view)

            with self._lock:
                self.latest_id = max(self.latest_id, capture_id)
                entry = self.entries.get(capture_id)
                if entry is not None and datatype in entry:
                    # Published cells are immutable, including while readers hold
                    # them. Replacing this reference would also leak the old slot.
                    self.logger.warning(
                        "[registry] Discarding duplicate %s capture=%d; keeping original slot=%d",
                        datatype.__name__, capture_id, entry[datatype],
                    )
                    return
                if entry is None:
                    # Delayed fields may join an existing older capture, but a
                    # missing older ID must not recreate an evicted entry.
                    if capture_id < self.latest_id:
                        self.logger.warning("[registry] Discarding stale %s capture=%d; latest received=%d",
                                            datatype.__name__, capture_id, self.latest_id)
                        return
                    if len(self.entries) == self.array_length:
                        self._evict()
                    entry = self.entries[capture_id] = Entry()
                entry[datatype] = slot
                # OR adds this primitive while preserving all other present bits.
                entry.mask |= helper.bit
                published = True
                self.logger.debug("[registry] Published %s capture=%d slot=%d",
                                  datatype.__name__, capture_id, slot)
                for required in self._affected[datatype]:
                    previous = self.latest[required]
                    mask = self._masks[required]
                    # AND keeps only required bits. Equality means none is missing.
                    if entry.mask & mask == mask and (previous is None or capture_id > previous):
                        old = self.entries.get(previous)
                        if old is not None:
                            old.latest_groups.discard(required)
                        self.latest[required] = capture_id
                        # Maintain the inverse of latest so eviction is selective.
                        entry.latest_groups.add(required)
                        if previous is None:
                            self.logger.info("[registry] First complete %s capture=%d",
                                             ", ".join(kind.__name__ for kind in required), capture_id)
                        # A higher latest ID is unread by every subscriber.
                        for event in self.events.get(required, {}).values():
                            if not event.is_set():
                                event.set()
        finally:
            view.release()
            if not published:
                with self._lock:
                    helper.release_slot(slot)

    @contextmanager
    def lease_slot_for_reading(self, datatype, consumer):
        """Wait for a newer match and yield read-only views valid inside the lease."""
        required = (frozenset(datatype) if isinstance(datatype, (set, frozenset))
                    else self._singletons[datatype])
        subscription = (consumer, required)
        event = self.events[required][consumer]
        while True:
            event.wait()
            with self._lock:
                # Eviction may clear readiness between waking and taking the lock.
                if not event.is_set():
                    continue
                capture_id = self.latest[required]
                entry = self.entries[capture_id]
                buffers = {
                    primitive: self.data_array[primitive].slot_view(entry[primitive]).toreadonly()
                    for primitive in required
                }
                entry.readers += 1
                # The cursor also identifies this active lease for the pipe server.
                # A failed lease restores it, allowing the current match to retry.
                previous = self.last_read[subscription]
                self.last_read[subscription] = capture_id
                event.clear()
                self.logger.debug("[registry] Consumer %s reading capture=%d; readers=%d",
                                  consumer, capture_id, entry.readers)
                break

        completed = False
        try:
            yield buffers
            completed = True
        finally:
            for view in buffers.values():
                view.release()
            with self._lock:
                entry.readers -= 1
                if not completed:
                    self.last_read[subscription] = previous
                    self._signal(required, consumer)
                self.logger.debug("[registry] Consumer %s released capture=%d; readers=%d, completed=%s",
                                  consumer, capture_id, entry.readers, completed)

    def for_stage(self, stage, stopped):
        """Create a process-safe reference for this stage's registered leases."""
        readers, writers = {}, {}
        with self._lock:
            subscriptions = [(group, False) for group, consumers in self.events.items()
                             if stage in consumers]
            subscriptions += [(kind, True) for kind, producer in self.producers.items()
                              if stage == producer]
        for datatype, writing in subscriptions:
            # Send this ordering with the client. Rebuilding a frozenset's order
            # in a spawned process could pair references with the wrong primitive.
            primitives = (datatype,) if writing else tuple(datatype)
            owner, remote = Pipe()
            (writers if writing else readers)[datatype] = remote, primitives
            worker = Thread(target=self._serve,
                            args=(owner, datatype, stage, writing, primitives, stopped), daemon=True)
            self._connections.append((worker, owner, stopped))
            worker.start()
        self.logger.info("[registry] Connected stage %s: %d inputs, %d outputs",
                         stage, len(readers), len(writers))
        return RegistryClient(readers, writers, stopped)

    def _serve(self, pipe, datatype, stage, writing, primitives, stopped):
        """Hold each lease until a stage replies: 1=success, 2=failure, 0=stop."""
        if not writing:
            with self._lock:
                event = _ConsumerEvent(stopped)
                if self.events[datatype][stage].is_set():
                    event.set()
                self.events[datatype][stage] = event
        known_blocks, layout = {}, Struct("<" + "IQ" * len(primitives))
        try:
            while pipe.recv_bytes():
                try:
                    lease = (self.lease_slot_for_writing(datatype) if writing
                             else self.lease_slot_for_reading(datatype, stage))
                    with lease as buffers:
                        if writing:
                            helper = self.data_array[datatype]
                            # The writer's reserved slot gives a portable location
                            # directly, without scanning blocks or inspecting pointers.
                            locations = [helper.slot_location(helper.writing_slot)]
                        else:
                            entry = self.entries[self.last_read[stage, datatype]]
                            locations = [self.data_array[kind].slot_location(entry[kind])
                                         for kind in primitives]
                        _send_references(pipe, locations, known_blocks, layout)
                        reply = pipe.recv_bytes()
                        if reply == b"2":
                            raise _CancelledLease
                        if reply == b"0":
                            break
                except _CancelledLease:
                    pass
        except (EOFError, BrokenPipeError):
            pass
        except Exception:
            self.logger.exception("[registry] Connection failed for %s", stage)
            raise
        finally:
            stopped.set()
            pipe.close()

    def close(self):
        """Close stage connections and release the registry's shared memory."""
        for _, _, stopped in self._connections:
            stopped.set()
        for worker, pipe, _ in self._connections:
            worker.join(3)
            pipe.close()
        for helper in self.data_array.values():
            helper.close()
        self.logger.info("[registry] Closed")


class _CancelledLease(Exception):
    pass


class _ConsumerEvent(Event):
    """Sleep on the normal event, checking shutdown every 100 ms while waiting.

    This timed wait blocks the thread; it does not spin or poll buffer contents.
    Only subscriptions attached to stage processes need the stop check.
    """

    def __init__(self, stopped):
        super().__init__()
        self.stopped = stopped

    def wait(self, timeout=None):
        while not super().wait(0.1):
            if self.stopped.is_set():
                raise EOFError("Stage stopped")
        return True


def _send_references(pipe, locations, known_blocks, layout):
    """Send names once (B), then packed references (F) for this connection.

    B contains NUL-separated names, appended to the client's block table in the
    same order as known_blocks. F contains one little-endian I/Q pair per
    primitive: a 4-byte block-table index and an 8-byte byte offset. Each later
    F message therefore needs 1 + 12 * primitive_count bytes, regardless of
    image size. The multiprocessing pipe adds its own framing around this body.
    """
    new_blocks, references = [], []
    for name, offset in locations:
        if name not in known_blocks:
            known_blocks[name] = len(known_blocks)
            new_blocks.append(name.encode())
        references.extend((known_blocks[name], offset))
    if new_blocks:
        pipe.send_bytes(b"B" + b"\0".join(new_blocks))
    pipe.send_bytes(b"F" + layout.pack(*references))


class RegistryClient:
    """A stage's registered ports; shared memory attaches on first use."""

    def __init__(self, readers, writers, stopped):
        self.readers, self.writers, self.stopped = readers, writers, stopped
        self.blocks, self.layouts = {}, {}  # Per pipe; compiled only in the child.

    def _receive(self, port):
        pipe, primitives = port
        if pipe not in self.blocks:
            self.blocks[pipe] = []
            self.layouts[pipe] = Struct("<" + "IQ" * len(primitives))
        blocks = self.blocks[pipe]
        while True:
            message = pipe.recv_bytes()
            if message[:1] == b"B":
                # New blocks can appear during acquisition as capacity grows.
                blocks.extend(shared_memory.SharedMemory(name=name.decode())
                              for name in message[1:].split(b"\0"))
                continue
            references = self.layouts[pipe].unpack_from(message, 1)
            # These slices map shared storage; unpacking copies only tiny metadata.
            return {kind: blocks[block].buf[offset : offset + ct.sizeof(kind)]
                    for kind, block, offset in zip(primitives, references[::2], references[1::2])}

    def lease_slot_for_writing(self, datatype):
        return self._lease(self.writers[datatype], writing=True)

    def lease_slot_for_reading(self, datatype, consumer):
        return self._lease(self.readers[_primitives(datatype)], writing=False)

    @contextmanager
    def _lease(self, port, writing):
        pipe, primitives = port
        pipe.send_bytes(b"W" if writing else b"R")
        buffers = self._receive(port)
        if not writing:
            buffers = {kind: view.toreadonly() for kind, view in buffers.items()}
        completed = False
        try:
            yield buffers[primitives[0]] if writing else buffers
            completed = True
        finally:
            for view in buffers.values():
                view.release()
            reply = b"1" if completed else b"2"
            if not writing and self.stopped.is_set():
                reply = b"0"
            pipe.send_bytes(reply)

    def close(self):
        for pipe, _ in (*self.readers.values(), *self.writers.values()):
            pipe.close()
        for blocks in self.blocks.values():
            for block in blocks:
                block.close()
