"""Bounded, zero-copy input aggregation by capture ID.

Registration may grow storage during acquisition without moving existing data.
Each consumer holds one read lease; each primitive has one producer and one
write lease at a time. Producers write each primitive once per nonnegative ID.
Buffer views stay inside leases. Metadata belongs to one registry process;
IPC and disk persistence are not implemented here.
"""

from collections import OrderedDict, deque
from contextlib import contextmanager
from multiprocessing import Event, Semaphore, shared_memory
from threading import RLock
import ctypes as ct

from core.central_registry import ARRAY_LENGTH, BUFFER_LENGTH, MATCHING_WINDOW


def _primitives(datatype):
    return frozenset(datatype) if isinstance(datatype, (set, frozenset)) else frozenset({datatype})


class SharedMemoryHelper:
    """Shared payload storage; the registry serializes access to the local slot pool."""

    def __init__(self, datatype, length):
        self.cell_size = ct.sizeof(datatype)
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
        """Take a free slot, or return None if the registry must free one."""
        if not self.free_slots:
            return None
        slot = self.free_slots.popleft()
        return slot, self.slot_view(slot)

    def release_slot(self, slot):
        """Return a slot freed by the registry."""
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


class CentralRegistry:

    def __init__(self, matching_window=MATCHING_WINDOW):
        self.matching_window = matching_window
        self.array_length = max(ARRAY_LENGTH, matching_window)
        self.buffer_length = max(BUFFER_LENGTH, self.array_length + 1)
        self.data_array = dict()  # primitive -> shared-memory stripe.
        self.entries = dict()  # capture_id -> {"slot": logical_slot, primitive: buffer_slot}.
        self.latest_id = -1
        # Oldest first; None marks an unused slot so allocation always uses eviction.
        self.arrival_order = OrderedDict.fromkeys(range(self.array_length))
        self._lock = RLock()  # Metadata and slot pools belong to the registry process.

        # Highest completed capture ID per requested set; never moves backwards.
        # A latest entry may have been evicted, in which case readers wait for newer data.
        self.latest = dict()
        # required_datatypes -> {consumer: Event}, set while a newer match is available.
        self.events = dict()
        self.last_read = dict()  # (consumer, required_datatypes) -> capture_id.
        # One semaphore per logical capture entry, covering all associated buffers.
        self.semaphore = [Semaphore(0) for _ in range(self.array_length)]
        self.producers = dict()  # primitive datatype -> producer ID.

    def register_consumer(self, consumer, datatype):
        """Register a latest-only subscription for a primitive or set of primitives."""
        required = _primitives(datatype)
        with self._lock:
            self.events.setdefault(required, {})[consumer] = Event()
            self.last_read[consumer, required] = -1
            for semaphore in self.semaphore:
                semaphore.release()
            consumers = len({name for name, _ in self.last_read})
            self.array_length = max(ARRAY_LENGTH, consumers + self.matching_window)
            self.buffer_length = max(BUFFER_LENGTH, self.array_length + 1)
            for slot in range(len(self.semaphore), self.array_length):
                self.semaphore.append(Semaphore(len(self.last_read)))
                self.arrival_order[slot] = None
                self.arrival_order.move_to_end(slot, last=False)
            for helper in self.data_array.values():
                helper.grow(self.buffer_length)
            self._update_latest()

    def register_producer(self, producer, datatype):
        """Register one producer for a primitive ctypes datatype."""
        with self._lock:
            self.data_array[datatype] = SharedMemoryHelper(datatype, self.buffer_length)
            self.producers[datatype] = producer
            self._update_latest()

    def _evict(self, datatype=None):
        """Free the oldest entry with no readers, optionally selecting a stripe."""
        reader_count = len(self.last_read)
        for slot_id, previous_id in self.arrival_order.items():
            entry = self.entries.get(previous_id, {})
            if datatype is not None and datatype not in entry:
                continue
            semaphore = self.semaphore[slot_id]
            acquired = 0
            try:
                for _ in range(reader_count):
                    if not semaphore.acquire(block=False):
                        break
                    acquired += 1
                if acquired == reader_count:
                    self.entries.pop(previous_id, None)
                    for primitive, slot in entry.items():
                        if primitive != "slot":
                            self.data_array[primitive].release_slot(slot)
                    self.arrival_order[slot_id] = None
                    self.arrival_order.move_to_end(slot_id, last=False)
                    return slot_id
            finally:
                for _ in range(acquired):
                    semaphore.release()
        raise BufferError("No entry without readers available to reclaim")

    def _new_entry(self, capture_id):
        slot = self._evict()
        self.arrival_order[slot] = capture_id
        self.arrival_order.move_to_end(slot)
        entry = {"slot": slot}
        self.entries[capture_id] = entry
        return entry

    def _update_latest(self):
        """Track the highest completed IDs and signal each consumer's unread match."""
        requirements = set(self.latest) | set(self.events)
        requirements.add(frozenset(self.data_array))
        requirements.update(frozenset({primitive}) for primitive in self.data_array)
        for required in requirements:
            latest = max(
                (capture_id for capture_id, entry in self.entries.items()
                 if required.issubset(entry)),
                default=None,
            )
            previous = self.latest.get(required)
            if previous is not None and (latest is None or previous > latest):
                latest = previous
            self.latest[required] = latest
            for consumer, event in self.events.get(required, {}).items():
                if (latest in self.entries and required.issubset(self.entries[latest])
                        and latest > self.last_read[consumer, required]):
                    event.set()
                else:
                    event.clear()

    @contextmanager
    def lease_slot_for_writing(self, datatype):
        """Yield one primitive's writable view; associate it by its ID on success."""
        helper = self.data_array[datatype]
        with self._lock:
            lease = helper.reserve_slot()
            if lease is None:
                self._evict(datatype)
                self._update_latest()
                lease = helper.reserve_slot()
        slot, view = lease
        published = False
        try:
            yield view

            capture_id = datatype.from_buffer(view).ID
            # TODO: Save the completed buffer before publishing or discarding it.

            with self._lock:
                self.latest_id = max(self.latest_id, capture_id)
                entry = self.entries.get(capture_id)
                if entry is None:
                    if capture_id < self.latest_id:
                        return
                    entry = self._new_entry(capture_id)
                entry[datatype] = slot
                published = True
                self._update_latest()
        finally:
            view.release()
            if not published:
                with self._lock:
                    helper.release_slot(slot)

    @contextmanager
    def lease_slot_for_reading(self, datatype, consumer):
        """Wait for a newer match and yield read-only views valid inside the lease."""
        required = _primitives(datatype)
        subscription = (consumer, required)
        event = self.events[required][consumer]
        while True:
            event.wait()
            with self._lock:
                self._update_latest()
                if not event.is_set():
                    continue
                capture_id = self.latest[required]
                entry = self.entries[capture_id]
                buffers = {
                    primitive: self.data_array[primitive].slot_view(entry[primitive]).toreadonly()
                    for primitive in required
                }
                semaphore = self.semaphore[entry["slot"]]
                semaphore.acquire()
                previous = self.last_read[subscription]
                self.last_read[subscription] = capture_id
                event.clear()
                break

        completed = False
        try:
            yield buffers
            completed = True
        finally:
            for view in buffers.values():
                view.release()
            with self._lock:
                semaphore.release()
                if not completed:
                    self.last_read[subscription] = previous
                self._update_latest()
