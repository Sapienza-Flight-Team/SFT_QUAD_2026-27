# Implemented registry optimizations

The registry now uses one capture-ID `OrderedDict`, whole-entry reader counts,
compiled presence masks and incremental latest updates. Producer and consumer
lease signatures are unchanged, including primitive/group subscriptions,
capture-ID ordering, failed-read retry and registration during acquisition.

## What changed

- A publication checks only groups containing the primitive just written. Full
  entry scans occur only when a new group is registered.
- Each group has a cached mask. Primitive specifications, IDs' byte offsets and
  native integer decoders are cached at registration.
- Advancing a group's latest ID makes it unread for every subscriber. Only unset
  events are set; no per-consumer match/last-read rescan is needed on publication.
- Successful read release only decrements the entry's reader count. Failed reads
  restore their previous last-read ID and refresh that subscription.
- `entries` is also the arrival-order table. Entries contain only primitive slot
  indices as dictionary items, with `readers`, `mask` and `latest_groups` in slots.
  The separate logical slots, arrival table and semaphore array are removed.
- Each entry records which groups currently point to it. Eviction clears only
  those groups' set events and preserves their latest-ID watermarks.
- Owner-thread events and a plain owner-thread lock replace multiprocessing
  events and the reentrant lock. Payload storage still uses shared memory.
- The demo registers its known consumers before allocating stripes, producing
  one shared-memory block per stripe at startup. Live growth still appends blocks
  without moving existing payloads or invalidating held views.
- The demo announces new blocks once per consumer connection, then sends a
  25-byte frame message: a tag and two block-ID/64-bit-offset pairs. Acknowledgements
  use one byte. Shared-memory names and datatype classes are absent from normal
  frame messages. Newly allocated blocks can be announced during acquisition.
- Acquisition, consumer processing, ID decoding and the future persistence hook
  remain outside the metadata lock. Slot-pool changes and publication stay under it.

The existing owner-process/Event design is retained. Group conditions and directly
shared metadata are alternative architectures, rather than additional layers to
combine with this one. Native code and a second reclamation index remain conditional
on future profiling; the registry preserves exact oldest-unread eviction with a
simple scan. No polling, extra validation framework or new delivery modes were added.

## Before and after

Measured on the same AMD Ryzen 7 7800X3D with CPython 3.12.12. The original registry
is frozen in `tools/_registry_reference.py`, used only by tests and the benchmark.
The benchmark's historical variant name `current` refers to that original code;
`optimized` refers to the production class. Source hashes are included in the
[raw results](central_registry_optimized_benchmark.json).

Two stripes, median elapsed microseconds per complete packet, three runs of 400
packets each, owner on CPU 0:

| Consumers | Pattern | Before | After |
|---:|:---|---:|---:|
| 6 | Every consumer reads each packet | 142.0 | 17.4 |
| 6 | Five consumers hold distinct older packets | 53.8 | 8.6 |
| 32 | Every consumer reads each packet | 2,116.9 | 71.4 |
| 32 | All but one hold older packets | 223.5 | 7.7 |
| 64 | Every consumer reads each packet | 8,644.5 | 144.0 |
| 64 | All but one hold older packets | 573.3 | 9.9 |

For six fast consumers, measured CPU time projects to about 0.85% versus 0.10%
of one core at 60 packets/second. This is a bookkeeping projection, not full-pipeline
CPU utilization. Four stripes with six fast consumers measured 277.4 versus
23.1 microseconds. With the actual 16,130,400-byte `PhotoExif` layout and `PixData`,
changing only IDs measured 131.7 versus 17.2 microseconds. Six consumers still
allocate seven cells per stripe: 112,913,304 payload bytes for these two types.

Registration, allocation, pixel filling, actual consumer processing, encoding and
disk I/O are excluded. There are 100 warm-up packets and garbage collection is
disabled only during timed samples. CPU frequency and scheduling noise affect
absolute measurements; these numbers are not latency guarantees.

The separate broker test uses real spawned processes, immediate acknowledgements
and serial round trips. The owner runs on CPU 0 and consumers share CPU 2. Before
uses the original pickled name/offset messages; after uses the compact demo protocol.
Medians of three runs of 1,000 packets:

| Consumers | Before elapsed µs | After elapsed µs | Before owner CPU µs | After owner CPU µs |
|---:|---:|---:|---:|---:|
| 1 | 48.9 | 14.8 | 38.9 | 10.8 |
| 6 | 227.9 | 69.9 | 171.6 | 42.5 |

These measurements include improvements in both registry bookkeeping and message
format. They do not isolate the benefit of binary messages alone or represent
concurrent full-resolution camera throughput. Raw
[broker results](central_registry_optimized_broker_benchmark.json) are preserved.

The optimized six-consumer fast profile has six event sets and six event clears
per packet, and no full latest rebuild on its timed path. Remaining costs are
mainly leases, notifications, view creation and ordinary Python execution.

## Validation and compatibility

24 registry/demo tests pass. They cover matching and capture-ID ordering,
read-only shared views, publication during reads, failed reads/writes, multiple
readers, minimum capacity, live consumer/producer registration and views remaining
valid across growth. Added tests also exercise:

- Primitive readers waiting before their producer exists.
- Different subscriptions belonging to the same consumer.
- Groups containing a producer registered later, including a 32-bit ID alias at
  a nonzero byte offset.
- Reintroduced partial IDs not reviving an evicted primitive's event.
- Deterministic mixed/out-of-order inputs compared with the original registry.
- Actual demo producer/consumer protocol and block attachment during live growth,
  with a consumer holding an older packet until acknowledgement.

The last test is headless; the GUI preview was not exercised. Real camera acquisition
and end-to-end encoding were not part of this change. A broader attempted struct
check cannot collect `tests/test_new_structs.py` because it imports `MissionMode`,
which is missing from the unchanged `common.new_structs` module.

`entries[capture_id][primitive]`, `latest`, `events`, `last_read` and buffer locations
remain usable as before. The old `arrival_order`, semaphore array and `entry["slot"]`
are removed. Registry events are local to its owner process; remote consumers use
the IPC transport to receive references and acknowledge releases. ID fields are
native integer fields of 1, 2, 4 or 8 bytes; producers still guarantee nonnegative
IDs and no repeated write of a primitive for the same capture.

## Reproduction

Run from the project root:

```sh
PYTHONPATH=src .venv/bin/pytest -q tests/test_central_registry.py tests/test_central_registry_demo.py
.venv/bin/python tools/benchmark_central_registry.py --iterations 400 --repeats 3 --output /tmp/registry-optimized.json
.venv/bin/python tools/benchmark_central_registry.py --broker-only --consumers 1 6 --iterations 1000 --repeats 3 --output /tmp/registry-broker-optimized.json
```

`--variants current reader_counts incremental` repeats the earlier prototype
comparison. `--consumers`, `--stripes`, `--cpu` and `--consumer-cpu` control the
workload and CPU placement. See the [earlier analysis](central_registry_performance.md)
for the original profiles and variable-frame-rate encoding experiment.
