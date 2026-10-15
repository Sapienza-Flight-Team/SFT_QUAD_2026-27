# Central registry performance

This report describes the original prototype study. The production implementation
has since been optimized; see [the implemented changes and fresh measurements](central_registry_optimized_performance.md).

The shared-memory pool, capture-ID dictionary and `OrderedDict` are a reasonable
fit for this registry. The current implementation spends substantially more time
than necessary rebuilding latest matches and touching consumer events. Keeping
those data structures but updating metadata incrementally was 4.7 times faster
with six consumers reading every packet, and 29 times faster with 64 consumers.
This does not establish a universally fastest implementation.

During this original study, production `src/core/central_registry.py` was left unchanged. The experimental
implementations live only in [the benchmark](../tools/benchmark_central_registry.py).

## Measurements

Measured on an AMD Ryzen 7 7800X3D with CPython 3.12.12, Linux, registry revision
`2b59634f5f203785d124061e37689e357575ed16`. The owner was pinned to CPU 0. Results are medians of three runs,
normally 1,000 packets per run; the four-stripe/64-consumer cases used 300.
Allocation and registration are outside timing, with 100 warm-up packets.
Garbage collection is disabled only during the timed interval. The loop includes
normal Python context-manager, timer and small metadata allocation costs.

One packet means writing each primitive once, followed by one read lease per
active consumer. All consumers subscribe to the set of all registered primitives.
The two modes are:

- **Fast:** every consumer reads each new complete packet and immediately releases it.
- **Pinned:** all but one consumer hold distinct older packets for the entire run;
  one consumer reads each new packet. This exercises eviction past busy slots and
  approximates the demo with slow background consumers and an active viewer.

All three variants use the same shared-memory helper, capacity, payload views,
owner `RLock`, capture-ID matching and latest-only semantics:

- **Current:** the production registry.
- **Reader counts:** an integer count per full logical slot replaces semaphore
  probing during eviction and semaphore acquisition/release during reads.
- **Incremental:** reader counts plus checking only groups affected by a newly
  published primitive. Reads do not rescan entries, and events are set/cleared
  when availability changes. Full rebuilding is retained for registration.

The prototypes retain the base class's unused semaphores during setup so that
registration/storage remains comparable. A production counter implementation
could remove those semaphores entirely. Counts are safe here because metadata
already belongs to one process and every change happens under its lock; plain
Python counters would not replace synchronization for independently mutated
metadata in several processes.

### Two stripes: elapsed microseconds per packet

| Consumers | Mode | Current | Reader counts | Incremental |
|---:|:---|---:|---:|---:|
| 1 | Fast | 25.4 | 23.0 | 9.0 |
| 6 | Fast | 123.5 | 125.1 | 26.3 |
| 6 | Pinned | 49.0 | 45.2 | 10.8 |
| 32 | Fast | 1,853.8 | 1,896.3 | 118.9 |
| 32 | Pinned | 206.8 | 153.0 | 16.6 |
| 64 | Fast | 6,800.2 | 6,858.4 | 234.7 |
| 64 | Pinned | 481.5 | 285.8 | 24.8 |

CPU time is close to elapsed time in these single-threaded measurements. At
60 complete packets/second, measured bookkeeping CPU time projects to:

| Consumers | Mode | Current: percent of one core | Incremental: percent of one core |
|---:|:---|---:|---:|
| 6 | Fast | 0.74% | 0.16% |
| 6 | Pinned | 0.29% | 0.064% |
| 32 | Fast | 11.10% | 0.71% |
| 64 | Fast | 40.56% | 1.40% |

These are bookkeeping projections, not measured CPU utilization for a complete
60 Hz camera pipeline. They omit acquisition, consumer computation, encoding,
disk writes, IPC and waiting-thread scheduling.

Increasing to four stripes with six fast consumers gave 239.4 microseconds for
the current registry and 34.1 for incremental updates. Four stripes and 64 fast
consumers gave 10,960.7 and 260.6 respectively. More tracked groups add work;
these measurements do not exercise every possible combination of primitives.

Using the actual `PixData` and `PhotoExif` layouts (5320 x 3032 grayscale pixels,
16,130,400-byte image struct) with six fast consumers gave 118.5 microseconds
current and 25.8 incremental. Only ID fields are written in this measurement.
The near-identical small/large-layout timings agree with the source and existing
shared-view tests: leasing/publishing does not copy image bytes. This test does
not exercise image-memory bandwidth, cache interference or camera DMA.

### Process communication measured separately

A separate test spawns consumers, sends each shared-memory name/offset over a
pipe, reads matching IDs directly from attached memory, and acknowledges while
the owner holds the read lease. Attachments are cached and warmed up. The owner
uses CPU 0; consumer processes share CPU 2. No payload goes through the pipe.

| Consumers | Variant | Elapsed µs/packet | Owner CPU µs/packet | Sum of consumer CPU µs/packet |
|---:|:---|---:|---:|---:|
| 1 | Current | 35.3 | 28.6 | 5.4 |
| 1 | Incremental | 18.5 | 14.8 | 5.1 |
| 6 | Current | 212.2 | 162.1 | 45.3 |
| 6 | Incremental | 99.2 | 61.4 | 43.2 |

Round trips are serial, with immediate acknowledgements. This measures a simple
broker path with real processes; it is not a throughput measurement of the GUI
demo or concurrent camera acquisition. It shows that IPC/process overhead becomes
a larger fraction of cost after bookkeeping is reduced.

## What the profile shows

A separate `cProfile` run of the current six-consumer fast workload attributes
approximately **81% of instrumented time** to `_update_latest`, including the
functions it calls. Each complete packet invokes it 15 times and triggers
42 `Event.set()` calls and 54 `Event.clear()` calls. Incremental updates reduce
this to six sets and six clears, with no full rebuild on the steady-state path.

The percentage is from instrumented execution; profiler overhead changes
absolute timings. The tables above were measured without profiling.

For each update, the current implementation reconstructs the requirement set,
scans every retained entry for each group, and revisits every consumer event.
Doing that before and after each consumer read produces roughly quadratic growth
in consumer count for these workloads. Setting an already-set multiprocessing
event also acquires its condition lock and calls notification machinery.

Eviction acquires and releases up to the total subscription count of semaphore
permits for each candidate entry. If many older entries are held by readers,
this can add another quadratic number of semaphore operations. Direct reader
counts remove that part, explaining their larger improvement in pinned workloads.
They make little difference in the fast workload because latest/event rebuilding
dominates it. The 1–2% differences between current and counts in fast cases should
be treated as measurement variation, not an established regression.

## Alternatives and recommendation

The smallest useful change is to keep the existing storage/matching structure
and update only metadata that changed:

1. On publication, test the changed capture against groups that include the
   primitive just written. Advance a group's latest ID only if this match is
   complete and its capture ID is newer.
2. Update events for affected subscriptions on publication, eviction and read
   completion/retry. Avoid repeated sets/clears when the state is unchanged.
3. Use one reader count per full slot under the existing owner lock. Eviction
   checks whether that count is zero instead of draining semaphore permits.

The benchmark prototypes preserve out-of-order matching, capture-ID ordering,
reader protection, failed-read retry and live registration. The production
registry and **both prototypes each passed all 18 existing registry tests**.
This is useful evidence, not a claim of exhaustive validation for every timing
or subscription pattern.

A plain circular array could reduce bookkeeping for an ordered single-stream
producer, but this registry still needs capture-ID matching, out-of-order arrivals,
live growth and protection for held slots. A ring alone does not solve those
requirements. Replacing `OrderedDict` is not the first optimization indicated by
this profile: the entire current eviction operation accounts for only about 1.9%
of the six-consumer fast instrumented run. The faster measured alternative keeps
`OrderedDict` and dictionaries.

If acquisition were already ordered and consumers did not aggregate streams,
separate latest-value buffers per stripe would be simpler. That would change
the matched-packet semantics required here. Native atomic shared metadata could
be investigated if the eventual broker becomes a bottleneck, but this benchmark
does not establish a need for that complexity.

At the current six-consumer/60 Hz scale, bookkeeping is inexpensive in absolute
terms. Incremental updates are worthwhile for simpler work per packet and better
scaling, rather than because the measured demo-sized registry consumes a large
fraction of the CPU.

## Variable frame rate and video consumers

**Variable frame rate works.** A local FFmpeg/libx264 test encoded five H.264
frames into MP4 at presentation times 0, 17, 67, 100 and 350 milliseconds.
`ffprobe` reported exactly those timestamps, with successive intervals of 17,
50, 33 and 250 milliseconds. The probe output is saved in
[central_registry_vfr_probe.json](central_registry_vfr_probe.json).
This checks timestamp preservation, not full-resolution encoding throughput.

FFmpeg supports passing through timestamps with `-fps_mode passthrough` or `vfr`;
`vfr` additionally avoids duplicate timestamps. Constant-rate mode may duplicate
or drop frames. A suitable encoder time base must represent the desired timing.
See [FFmpeg's frame-rate mode and time-base documentation](https://ffmpeg.org/ffmpeg.html#Advanced-options).

The registry currently stores capture IDs but **no capture timestamp** in either
input struct. For an encoder, include acquisition time in the shared metadata
and derive presentation timestamps from it, using one clock and epoch. For
example, with a 1/1000-second time base:

```python
pts = round((capture_time_ns - first_capture_time_ns) / 1_000_000)
```

Use sufficient precision for the actual acquisition rate. Capture IDs can supply
timing only if their relationship to a fixed capture cadence is known. Encoder
processing/completion time is unsuitable: it includes scheduling and pipeline
delays. [PyAV's official example](https://github.com/PyAV-Org/PyAV/blob/main/examples/numpy/generate_video_with_pts.py)
demonstrates encoding frames with explicit PTS and variable durations.

The ordinary [`cv2.VideoWriter(fps, ...).write(frame)` interface](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html)
specifies a stream rate and accepts an image without a per-call timestamp. It
does not automatically preserve irregular acquisition timing because calls arrive
at different wall-clock intervals. An explicit-timestamp FFmpeg/PyAV path is a
better fit for this consumer.

A latest-only encoder can skip captures if it is slower than acquisition. VFR
preserves elapsed time between the frames it receives; it cannot restore skipped
frames. If the desired recording must contain every captured frame, the encoder
must keep up or a separate recording/backlog path must retain those frames. The
registry's proposed input persistence remains a TODO, so it does not currently
provide such a path. Also, zero-copy registry leases do not establish zero-copy
camera acquisition or video encoding: pixel conversion and encoder-owned buffering
must be considered separately. The camera was not available for that test.

## Reproduction

Run from the project root with the existing virtual environment:

```sh
.venv/bin/python tools/benchmark_central_registry.py --variants current reader_counts incremental --iterations 1000 --repeats 3 --output /tmp/registry-results.json
.venv/bin/python tools/benchmark_central_registry.py --broker-only --variants current incremental --consumers 1 6 --iterations 1000 --repeats 3 --output /tmp/registry-broker-results.json
PYTHONPATH=src .venv/bin/pytest -q tests/test_central_registry.py
```

The script accepts `--consumers`, `--stripes`, `--cpu` and `--consumer-cpu` to
change the workload and placement. Pick CPUs available on the machine.

Validate each experimental registry against the same existing suite:

```sh
PYTHONPATH=src:tools .venv/bin/python - <<'PY'
import pytest
from benchmark_central_registry import ReaderCountRegistry, IncrementalRegistry

class UseVariant:
    def __init__(self, variant):
        self.variant = variant

    def pytest_collection_modifyitems(self, items):
        for item in items:
            item.module.CentralRegistry = self.variant

for variant in (ReaderCountRegistry, IncrementalRegistry):
    result = pytest.main(['tests/test_central_registry.py', '-q'], plugins=[UseVariant(variant)])
    if result:
        raise SystemExit(result)
PY
```

The variable-rate check uses small synthetic frames:

```sh
ffmpeg -hide_banner -loglevel error -f lavfi -i 'testsrc2=size=64x64:rate=1000' -frames:v 5 -vf "setpts='if(eq(N,0),0,if(eq(N,1),17,if(eq(N,2),67,if(eq(N,3),100,350))))'" -c:v libx264 -preset ultrafast -bf 0 -fps_mode vfr -enc_time_base 1:1000 -y /tmp/registry_vfr.mp4
ffprobe -v error -select_streams v:0 -show_entries frame=pts_time,duration_time -of json /tmp/registry_vfr.mp4
```

Raw results: [bookkeeping and profiles](central_registry_benchmark.json),
[process round trips](central_registry_broker_benchmark.json). The machine uses
FFmpeg n9.0.2. Timing will vary with CPU frequency, OS activity and hardware.
