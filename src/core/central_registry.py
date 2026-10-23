"""Latest capture groups in shared memory; see docs/central_registry.md.
Register before first use; one writer per primitive and one read per consumer.
Views stay inside leases. Finish stage clients before closing the owner."""

import ctypes as ct
from collections import deque
from contextlib import contextmanager
from functools import cache
from multiprocessing import Pipe, shared_memory
from selectors import EVENT_READ, PollSelector
from struct import Struct
from threading import Condition, Event, Lock, Thread

MATCHING_WINDOW = 16
_COMMAND = Struct("<BI")  # Operation byte and port index; payloads never enter the pipe.


@cache
def _singleton(datatype):
    return frozenset({datatype})


def _primitives(datatype):
    return frozenset(datatype) if isinstance(datatype, (set, frozenset)) else _singleton(datatype)


@cache
def _frame(length):
    return Struct(f"<{length}I")


class Entry(dict):
    __slots__ = ("readers", "mask")

    def __init__(self):
        self.readers = self.mask = 0


class RegistryClient:
    """Shared lease interface; the owner overrides remote slot management."""

    def __init__(self, pipe, ports, names):
        self.pipe, self.names, self.data_array = pipe, names, {}
        self._sizes = {kind: ct.sizeof(kind) for kind in names}
        self.ports = {kind: (primitives, tuple(_COMMAND.pack(op, i) for op in range(3)))
                      for i, (kind, primitives) in enumerate(ports)}

    def lease_slot_for_writing(self, datatype):
        return self._lease(datatype)

    def lease_slot_for_reading(self, datatype, consumer):
        return self._lease(_primitives(datatype), consumer)

    @contextmanager
    def _lease(self, datatype, consumer=None):
        with self._slots(datatype, consumer) as slots:
            views, reading = {}, isinstance(datatype, frozenset)
            try:
                for kind, slot in slots.items():
                    size = self._sizes[kind]
                    views[kind] = self.data_array[kind].buf[slot * size : (slot + 1) * size]
                    if reading:
                        views[kind] = views[kind].toreadonly()
                yield views if reading else views[datatype]
            finally:
                for view in views.values():
                    view.release()  # Release views before returning ownership of their slots.

    @contextmanager
    def _slots(self, datatype, consumer=None):
        primitives, commands = self.ports[datatype]
        self.pipe.send_bytes(commands[0])  # 0=request, 1=commit, 2=cancel.
        slots, completed = _frame(len(primitives)).unpack(self.pipe.recv_bytes()), False
        try:
            for kind in primitives:
                if kind not in self.data_array:
                    self.data_array[kind] = shared_memory.SharedMemory(name=self.names[kind])
            yield dict(zip(primitives, slots))
            completed = True
        finally:
            self.pipe.send_bytes(commands[1 if completed else 2])

    def close(self):
        self.pipe.close()
        for memory in self.data_array.values():
            memory.close()
        self.data_array.clear()


class CentralRegistry(RegistryClient):
    def __init__(self, logger=None, matching_window=MATCHING_WINDOW):
        self.matching_window = max(1, matching_window)
        self.producers, self.latest, self.last_read = {}, {}, {}
        self.data_array, self.free_slots, self.entries = {}, {}, {}
        self.latest_id = -1
        self._lock, self._conditions, self._connections = Lock(), {}, []
        self._prepared = self._closed = False

    def register_producer(self, producer, datatype):
        if self._prepared or self._closed:
            raise RuntimeError("Registration is closed")
        if datatype in self.producers:
            raise ValueError(f"{datatype.__name__} already has a producer")
        self.producers[datatype] = producer

    def register_consumer(self, consumer, datatype):
        if self._prepared or self._closed:
            raise RuntimeError("Registration is closed")
        required = _primitives(datatype)
        self.latest.setdefault(required, None)
        self._conditions.setdefault(required, Condition(self._lock))
        self.last_read[consumer, required] = -1

    def prepare(self):
        if self._closed:
            raise EOFError("Registry closed")
        if self._prepared:
            return
        with self._lock:
            if self._prepared:
                return
            self.array_length = len({consumer for consumer, _ in self.last_read}) + self.matching_window
            self.buffer_length = self.array_length + 1
            self._sizes = {kind: ct.sizeof(kind) for kind in self.producers}
            self._bits = {kind: 1 << i for i, kind in enumerate(self.producers)}
            self._masks = {group: sum(self._bits[kind] for kind in group) for group in self.latest}
            self._affected = {kind: tuple(group for group in self.latest if kind in group) for kind in self.producers}
            self._ids = {kind: (kind.ID.offset, Struct({1: "B", 2: "H", 4: "I", 8: "Q"}[kind.ID.size]).unpack_from)
                         for kind in self.producers}
            for kind, size in self._sizes.items():
                self.data_array[kind] = shared_memory.SharedMemory(create=True, size=size * self.buffer_length)
                self.free_slots[kind] = deque(range(self.buffer_length))
            self._prepared = True

    def _publish(self, datatype, slot, capture_id):  # Caller holds the metadata lock.
        self.latest_id = max(self.latest_id, capture_id)
        entry = self.entries.get(capture_id)
        if entry is None:
            if capture_id < self.latest_id:
                return False
            if len(self.entries) == self.array_length:
                oldest = next(key for key, value in self.entries.items() if not value.readers)
                for kind, old_slot in self.entries.pop(oldest).items():
                    self.free_slots[kind].append(old_slot)
            entry = self.entries[capture_id] = Entry()
        if datatype in entry:
            return False
        entry[datatype] = slot
        entry.mask |= self._bits[datatype]
        for group in self._affected[datatype]:
            previous, mask = self.latest[group], self._masks[group]
            if entry.mask & mask == mask and (previous is None or capture_id > previous):
                self.latest[group] = capture_id
                self._conditions[group].notify_all()  # Only this newly advanced group's readers.
        return True

    @contextmanager
    def _slots(self, datatype, consumer=None, stopped=None):
        self.prepare()
        writing = not isinstance(datatype, frozenset)
        with self._lock:
            if writing:
                slots = {datatype: self.free_slots[datatype].popleft()}
            else:
                subscription = consumer, datatype
                while True:
                    if self._closed or (stopped is not None and stopped.is_set()):
                        raise EOFError("Registry subscription stopped")
                    capture_id = self.latest[datatype]
                    if capture_id in self.entries and capture_id > self.last_read[subscription]:
                        break
                    self._conditions[datatype].wait()  # No timeout or periodic wake-up.
                entry = self.entries[capture_id]
                entry.readers += 1
                slots = {kind: entry[kind] for kind in datatype}
        completed = False
        try:
            yield slots
            if writing:
                offset, decode = self._ids[datatype]
                capture_id = decode(self.data_array[datatype].buf, slots[datatype] * self._sizes[datatype] + offset)[0]
                with self._lock:
                    completed = self._publish(datatype, slots[datatype], capture_id)
            else:
                completed = True
        finally:
            if not writing or not completed:
                with self._lock:
                    if writing:
                        self.free_slots[datatype].append(slots[datatype])
                    else:
                        entry.readers -= 1
                        if completed:
                            self.last_read[subscription] = capture_id

    def for_stage(self, stage, stopped):
        self.prepare()
        groups = [group for consumer, group in self.last_read if consumer == stage]
        specs = groups + [kind for kind, producer in self.producers.items() if producer == stage]
        ports = [(kind, tuple(_primitives(kind))) for kind in specs]
        owner, remote = Pipe()
        wake, signal = Pipe(duplex=False)
        local_stop = Event()  # The watcher mirrors the shared stop event for cheap checks.
        worker = Thread(target=self._serve, args=(owner, wake, ports, stage, stopped, local_stop), daemon=True)
        watcher = Thread(target=self._stop, args=(stopped, groups, signal, local_stop), daemon=True)
        self._connections.append((worker, watcher, stopped))
        watcher.start()
        worker.start()
        return RegistryClient(remote, ports, {kind: memory.name for kind, memory in self.data_array.items()})

    def _stop(self, stopped, groups, signal, local_stop):
        stopped.wait()  # One sleeping watcher per stage; shutdown needs no polling.
        local_stop.set()
        with self._lock:
            for group in groups:
                self._conditions[group].notify_all()
        signal.close()  # EOF wakes an idle broker; no periodic check or stop message.

    def _serve(self, pipe, wake, ports, stage, stopped, local_stop):
        leases, frames = {}, [_frame(len(primitives)) for _, primitives in ports]
        selector = PollSelector()  # Reuse registrations for the lifetime of this broker.
        selector.register(pipe, EVENT_READ)
        selector.register(wake, EVENT_READ)
        try:
            while leases or not local_stop.is_set():
                if not leases and any(key.fileobj is wake for key, _ in selector.select()):
                    break
                operation, index = _COMMAND.unpack(pipe.recv_bytes())
                if operation == 0:
                    datatype, primitives = ports[index]
                    lease = self._slots(datatype, stage, local_stop)
                    slots = lease.__enter__()
                    leases[index] = lease
                    pipe.send_bytes(frames[index].pack(*(slots[kind] for kind in primitives)))
                else:
                    leases.pop(index).__exit__(None if operation == 1 else GeneratorExit, None, None)
        except (EOFError, BrokenPipeError):
            pass
        finally:
            selector.close()
            for lease in leases.values():
                lease.__exit__(GeneratorExit, None, None)
            stopped.set()
            pipe.close()
            wake.close()

    def close(self):
        with self._lock:
            self._closed = True
            for _, _, stopped in self._connections:
                stopped.set()
            for condition in self._conditions.values():
                condition.notify_all()
        for worker, watcher, _ in self._connections:
            worker.join()
            watcher.join()
        for memory in self.data_array.values():
            memory.close()
            memory.unlink()
        self.data_array.clear()
