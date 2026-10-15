"""Isolated registry prototypes and repeatable bookkeeping benchmarks.

Run from the project root: .venv/bin/python tools/benchmark_central_registry.py
No production classes are patched. Payload allocation and consumer processing
are outside the timed region; the large-image case only writes its ID field.
"""

import argparse
import cProfile
import ctypes as ct
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import pstats
import statistics
import struct
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from time import perf_counter_ns, process_time_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _registry_reference import CentralRegistry, _primitives
from core.central_registry import CentralRegistry as OptimizedRegistry


class ReaderCountRegistry(CentralRegistry):
    """One integer per full entry, protected by the owner's existing RLock."""

    def __init__(self):
        super().__init__()
        self.readers = [0] * self.array_length

    def register_consumer(self, consumer, datatype):
        with self._lock:
            super().register_consumer(consumer, datatype)
            self.readers.extend([0] * (self.array_length - len(self.readers)))

    def _evicted(self, capture_id):
        pass

    def _evict(self, datatype=None):
        for slot_id, capture_id in self.arrival_order.items():
            entry = self.entries.get(capture_id, {})
            if self.readers[slot_id] or (datatype is not None and datatype not in entry):
                continue
            self.entries.pop(capture_id, None)
            for primitive, slot in entry.items():
                if primitive != "slot":
                    self.data_array[primitive].release_slot(slot)
            self.arrival_order[slot_id] = None
            self.arrival_order.move_to_end(slot_id, last=False)
            self._evicted(capture_id)
            return slot_id
        raise BufferError("No entry without readers available to reclaim")

    def _before_read(self, required, consumer):
        self._update_latest()

    def _after_read(self, required, consumer):
        self._update_latest()

    def _read_started(self, required, consumer):
        pass

    @contextmanager
    def lease_slot_for_reading(self, datatype, consumer):
        required = _primitives(datatype)
        subscription = (consumer, required)
        event = self.events[required][consumer]
        while True:
            event.wait()
            with self._lock:
                self._before_read(required, consumer)
                if not event.is_set():
                    continue
                capture_id = self.latest[required]
                entry = self.entries[capture_id]
                buffers = {
                    primitive: self.data_array[primitive].slot_view(entry[primitive]).toreadonly()
                    for primitive in required
                }
                slot = entry["slot"]
                self.readers[slot] += 1
                previous = self.last_read[subscription]
                self.last_read[subscription] = capture_id
                event.clear()
                self._read_started(required, consumer)
                break
        completed = False
        try:
            yield buffers
            completed = True
        finally:
            for view in buffers.values():
                view.release()
            with self._lock:
                self.readers[slot] -= 1
                if not completed:
                    self.last_read[subscription] = previous
                self._after_read(required, consumer)


class IncrementalRegistry(ReaderCountRegistry):
    """Same dictionary/OrderedDict pool, updating only affected subscriptions."""

    def __init__(self):
        self.ready = set()
        self.groups_by_primitive = {}
        super().__init__()

    def _update_latest(self):
        # Registration is infrequent; keep the existing full rebuild there.
        super()._update_latest()
        self.ready = {(consumer, required)
                      for required, consumers in self.events.items()
                      for consumer, event in consumers.items() if event.is_set()}
        self.groups_by_primitive = {
            primitive: tuple(required for required in self.latest if primitive in required)
            for primitive in self.data_array
        }

    def _signal(self, required, consumer):
        key = (consumer, required)
        capture_id = self.latest[required]
        available = (capture_id in self.entries
                     and required.issubset(self.entries[capture_id])
                     and capture_id > self.last_read[key])
        if available and key not in self.ready:
            self.events[required][consumer].set()
            self.ready.add(key)
        elif not available and key in self.ready:
            self.events[required][consumer].clear()
            self.ready.remove(key)

    def _evicted(self, capture_id):
        for required, latest in self.latest.items():
            if latest == capture_id:
                for consumer in self.events.get(required, {}):
                    self._signal(required, consumer)

    def _before_read(self, required, consumer):
        pass

    def _after_read(self, required, consumer):
        self._signal(required, consumer)

    def _read_started(self, required, consumer):
        self.ready.discard((consumer, required))

    @contextmanager
    def lease_slot_for_writing(self, datatype):
        helper = self.data_array[datatype]
        with self._lock:
            lease = helper.reserve_slot()
            if lease is None:
                self._evict(datatype)
                lease = helper.reserve_slot()
        slot, view = lease
        published = False
        try:
            yield view
            capture_id = datatype.from_buffer(view).ID
            with self._lock:
                entry = self.entries.get(capture_id)
                if entry is None:
                    latest = self.latest.get(frozenset(self.data_array))
                    if latest is not None and capture_id < latest:
                        return
                    entry = self._new_entry(capture_id)
                entry[datatype] = slot
                published = True
                for required in self.groups_by_primitive[datatype]:
                    previous = self.latest[required]
                    if required.issubset(entry) and (previous is None or capture_id >= previous):
                        self.latest[required] = capture_id
                        for consumer in self.events.get(required, {}):
                            self._signal(required, consumer)
        finally:
            view.release()
            if not published:
                with self._lock:
                    helper.release_slot(slot)


VARIANTS = {"current": CentralRegistry, "reader_counts": ReaderCountRegistry,
            "incremental": IncrementalRegistry, "optimized": OptimizedRegistry}


def record_types(stripes):
    return tuple(type(f"Stripe{i}", (ct.Structure,), {
        "_fields_": [("ID", ct.c_uint64), ("value", ct.c_uint64)]
    }) for i in range(stripes))


def publish(registry, primitives, capture_id):
    for primitive in primitives:
        with registry.lease_slot_for_writing(primitive) as view:
            struct.pack_into("<Q", view, primitive.ID.offset, capture_id)


@contextmanager
def workload(variant, consumers, primitives, mode):
    registry = VARIANTS[variant]()
    required = frozenset(primitives)
    try:
        for consumer in range(consumers):
            registry.register_consumer(consumer, required)
        for primitive in primitives:
            registry.register_producer(primitive.__name__, primitive)
        with ExitStack() as held:
            if mode == "pinned":
                for capture_id in range(consumers - 1):
                    publish(registry, primitives, capture_id)
                    held.enter_context(registry.lease_slot_for_reading(required, capture_id))
            active = range(consumers) if mode == "fast" else (consumers - 1,)

            def step(capture_id):
                publish(registry, primitives, capture_id)
                for consumer in active:
                    with registry.lease_slot_for_reading(required, consumer):
                        pass

            yield registry, step
    finally:
        for helper in registry.data_array.values():
            helper.close()


def sample(variant, consumers, primitives, mode, iterations, profile=False):
    with workload(variant, consumers, primitives, mode) as (registry, step):
        start_id = consumers
        for capture_id in range(start_id, start_id + 100):
            step(capture_id)
        start_id += 100
        profiler = cProfile.Profile() if profile else None
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            if profiler:
                profiler.enable()
            elapsed = []
            cpu_start = process_time_ns()
            wall_start = perf_counter_ns()
            for capture_id in range(start_id, start_id + iterations):
                before = perf_counter_ns()
                step(capture_id)
                elapsed.append(perf_counter_ns() - before)
            wall = perf_counter_ns() - wall_start
            cpu = process_time_ns() - cpu_start
            if profiler:
                profiler.disable()
        finally:
            if gc_enabled:
                gc.enable()
        result = {"variant": variant, "consumers": consumers, "stripes": len(primitives),
                  "mode": mode, "iterations": iterations,
                  "wall_us": wall / iterations / 1000, "cpu_us": cpu / iterations / 1000,
                  "p95_us": sorted(elapsed)[int(.95 * (iterations - 1))] / 1000,
                  "payload_bytes": sum(buffer.size for helper in registry.data_array.values()
                                       for buffer in helper.buffers)}
        if profiler:
            stats = pstats.Stats(profiler)
            result["profile_seconds"] = stats.total_tt
            result["profile"] = [
                {"file": filename, "line": line, "function": name, "calls": calls,
                 "self_seconds": own, "cumulative_seconds": cumulative}
                for (filename, line, name), (_, calls, own, cumulative, _) in
                sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:35]
            ]
        return result


def consume_pipe(connection, cpu, compact=False):
    """Attach once per block and acknowledge each lease without copying payloads."""
    from multiprocessing.shared_memory import SharedMemory
    os.sched_setaffinity(0, {cpu})
    blocks = [] if compact else {}
    references = struct.Struct("<IQIQ")
    started = process_time_ns()
    try:
        while True:
            locations = connection.recv_bytes() if compact else connection.recv()
            if compact and locations[:1] == b"B":
                blocks.extend(SharedMemory(name=name.decode()) for name in locations[1:].split(b"\0"))
                continue
            if locations == (b"S" if compact else "start"):
                started = process_time_ns()
                connection.send_bytes(b"1") if compact else connection.send(None)
            elif locations == (b"X" if compact else None):
                elapsed = process_time_ns() - started
                connection.send_bytes(struct.pack("<Q", elapsed)) if compact else connection.send(elapsed)
                return
            else:
                ids = []
                if compact:
                    values = references.unpack_from(locations, 1)
                    ids = [struct.unpack_from("<Q", blocks[block].buf, offset)[0]
                           for block, offset in zip(values[::2], values[1::2])]
                else:
                    for name, offset in locations:
                        if name not in blocks:
                            blocks[name] = SharedMemory(name=name)
                        ids.append(struct.unpack_from("<Q", blocks[name].buf, offset)[0])
                assert len(set(ids)) == 1
                connection.send_bytes(b"1") if compact else connection.send(None)
    finally:
        for block in blocks if compact else blocks.values():
            block.close()
        connection.close()


def broker_sample(variant, consumers, iterations, owner_cpu, consumer_cpu):
    """Serial pipe round trips with real processes; deliberately no image work."""
    context = mp.get_context("spawn")
    registry = VARIANTS[variant]()
    primitives = record_types(2)
    required = frozenset(primitives)
    processes, connections = [], []
    compact = variant == "optimized"
    known_blocks = [{} for _ in range(consumers)]
    references = struct.Struct("<IQIQ")
    try:
        for consumer in range(consumers):
            registry.register_consumer(consumer, required)
        for primitive in primitives:
            registry.register_producer(primitive.__name__, primitive)
        for consumer in range(consumers):
            owner, reader = context.Pipe()
            process = context.Process(target=consume_pipe, args=(reader, consumer_cpu, compact))
            process.start()
            reader.close()
            processes.append(process)
            connections.append(owner)

        def step(capture_id):
            publish(registry, primitives, capture_id)
            for consumer, connection in enumerate(connections):
                with registry.lease_slot_for_reading(required, consumer):
                    entry = registry.entries[registry.latest[required]]
                    locations = tuple(registry.data_array[primitive].slot_location(entry[primitive])
                                      for primitive in primitives)
                    if compact:
                        new_blocks, values = [], []
                        known = known_blocks[consumer]
                        for name, offset in locations:
                            if name not in known:
                                known[name] = len(known)
                                new_blocks.append(name.encode())
                            values.extend((known[name], offset))
                        if new_blocks:
                            connection.send_bytes(b"B" + b"\0".join(new_blocks))
                        connection.send_bytes(b"F" + references.pack(*values))
                        connection.recv_bytes()
                    else:
                        connection.send(locations)
                        connection.recv()

        for capture_id in range(100):
            step(capture_id)
        for connection in connections:
            if compact:
                connection.send_bytes(b"S")
                connection.recv_bytes()
            else:
                connection.send("start")
                connection.recv()
        cpu_start, wall_start = process_time_ns(), perf_counter_ns()
        for capture_id in range(100, iterations + 100):
            step(capture_id)
        wall, cpu = perf_counter_ns() - wall_start, process_time_ns() - cpu_start
        consumer_cpu_total = 0
        for connection in connections:
            if compact:
                connection.send_bytes(b"X")
                consumer_cpu_total += struct.unpack("<Q", connection.recv_bytes())[0]
            else:
                connection.send(None)
                consumer_cpu_total += connection.recv()
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
        return {"variant": variant, "consumers": consumers, "iterations": iterations,
                "wall_us": wall / iterations / 1000, "owner_cpu_us": cpu / iterations / 1000,
                "consumers_cpu_us": consumer_cpu_total / iterations / 1000,
                "owner_cpu": owner_cpu, "consumer_cpu": consumer_cpu}
    finally:
        for connection in connections:
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        for helper in registry.data_array.values():
            helper.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--consumers", type=int, nargs="+", default=[1, 6, 32, 64])
    parser.add_argument("--stripes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["current", "optimized"])
    parser.add_argument("--output", type=Path, default=Path("/tmp/central_registry_benchmark.json"))
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--broker-only", action="store_true")
    parser.add_argument("--consumer-cpu", type=int, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    if args.broker_only:
        results = [broker_sample(variant, consumers, args.iterations, args.cpu, args.consumer_cpu)
                   for consumers in args.consumers for variant in args.variants
                   for _ in range(args.repeats)]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
        return
    results = []
    for stripes in args.stripes:
        primitives = record_types(stripes)
        for consumers in args.consumers:
            for mode in ("fast", "pinned"):
                for variant in args.variants:
                    runs = [sample(variant, consumers, primitives, mode, args.iterations)
                            for _ in range(args.repeats)]
                    result = runs[0].copy()
                    for key in ("wall_us", "cpu_us", "p95_us"):
                        result[key] = statistics.median(run[key] for run in runs)
                    result["runs"] = runs
                    results.append(result)
                    print(f"p={stripes} k={consumers:2} {mode:6} {variant:13} "
                          f"{result['wall_us']:9.1f} us, CPU {result['cpu_us']:9.1f} us", flush=True)
    from common.new_structs import PhotoExif, PixData
    for variant in args.variants:
        results.append(sample(variant, 6, (PixData, PhotoExif), "fast", args.iterations))
        results[-1]["full_image_layout"] = True
    profiles = [sample(variant, 6, record_types(2), mode, min(args.iterations, 500), profile=True)
                for mode in ("fast", "pinned") for variant in args.variants]
    output = {"python": sys.version, "platform": platform.platform(), "cpu_affinity": [args.cpu],
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
                  ("baseline", Path(__file__).with_name("_registry_reference.py")),
                  ("optimized", Path(__file__).resolve().parents[1] / "src/core/central_registry.py"),
                  ("harness", Path(__file__)))},
              "repeats": args.repeats, "results": results, "profiles": profiles}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
