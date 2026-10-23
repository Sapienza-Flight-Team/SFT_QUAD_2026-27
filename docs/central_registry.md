# Central registry

This guide describes [central_registry.py](../src/core/central_registry.py):
a latest-data registry with shared-memory buffers, exact capture matching,
and blocking reads. It first explains the design, then every function.

## Contents

- [Overview](#overview)
- [State and storage](#state-and-storage)
- [A complete local example](#a-complete-local-example)
- [Module helpers and Entry](#module-helpers-and-entry)
- [RegistryClient functions](#registryclient-functions)
- [CentralRegistry functions](#centralregistry-functions)
- [Operating rules](#operating-rules)

## Overview

The registry connects producers and consumers without sending payloads through
Python queues. Each primitive datatype has its own shared-memory allocation.
Producers borrow writable cells; consumers borrow read-only views of cells
that have already been published.

A **primitive** is a fixed-size `ctypes.Structure` class.
A **group** is a set of primitive classes required together.
A **capture entry** maps a capture ID to the slots containing its primitives.
A **subscription** is one consumer's request for one group.

For example, a consumer requesting `{Image, Sensors}` can read capture 42 only
after both primitives with ID 42 have been published. Matching compares IDs,
not timestamps. A consumer requesting just `Image` can proceed sooner.

Reads return the **latest complete group** available for that subscription.
If captures 42, 43, and 44 complete between reads, the consumer can receive 44
directly. The registry does not preserve a delivery queue for each consumer.

The `last_read` cursor records the last successful read. It prevents the next
read from immediately returning the same capture and lets the consumer sleep
until something newer is ready. It does not require reading every capture.

The write lifecycle is:

```text
reserve a free slot → fill its bytes and ID → publish → match subscribed groups
```

The read lifecycle is:

```text
wait for a newer complete group → pin its entry → read views → unpin and advance
```

A lease is a context manager defining how long those views remain valid.
While any reader holds an entry, its cells cannot be reclaimed.
Payload processing happens outside the metadata lock.

There are three classes:

| Class | Responsibility |
| --- | --- |
| `Entry` | Store one capture's primitive slots, reader count, and presence mask. |
| `RegistryClient` | Present the lease API and request slots over a pipe. |
| `CentralRegistry` | Own memory, match groups, manage leases, and serve clients. |

The owner inherits the client's view-building code but overrides slot
management. Local callers and remote stage clients therefore use the same API.

## State and storage

All matching metadata lives in the owner process. Clients receive shared-memory
names and slot numbers; payload bytes stay in shared memory.

| Owner field | Meaning |
| --- | --- |
| `producers[kind]` | Producer identity registered for this primitive. |
| `entries[id][kind]` | Slot containing that primitive for this capture. |
| `latest_id` | Highest ID presented for publication, including incomplete captures. |
| `latest[group]` | Highest ID that completed the group; initially `None`. |
| `last_read[consumer, group]` | Last successful read ID; initially `-1`. |
| `data_array[kind]` | Shared-memory allocation for one primitive type. |
| `free_slots[kind]` | Deque of cells available to writers. |
| `_conditions[group]` | Sleeping readers interested in that group. |
| `_connections` | Broker threads, stop watchers, and stage stop events. |

A slot is a physical cell index, not a capture ID. Different primitives from
the same capture can occupy different slot indices. For a primitive of size
`size`, slot `s` occupies bytes `s * size : (s + 1) * size`.

The owner uses one plain `Lock` for metadata. Each group has a `Condition`
sharing that lock, so publishing can wake only the groups that advanced.
A condition wait releases the lock while sleeping and reacquires it on waking.

Preparation also caches primitive sizes, ID decoders, matching bitmasks, and
the groups affected by each primitive. These avoid rebuilding layouts and
scanning unrelated subscriptions during normal publication.
Each broker also reuses a readiness selector and checks a local stop flag;
the shared process event is handled by its sleeping watcher.

## A complete local example

Run with `src` on `PYTHONPATH`. The producer writes through ctypes; the reader
uses `struct.unpack_from`, which accepts the read-only views.

<!-- runnable: local-example -->
```python
import ctypes as ct
import struct

from core.central_registry import CentralRegistry


class Image(ct.Structure):
    _fields_ = [("ID", ct.c_uint64), ("pixels", ct.c_uint8 * 4)]


class Sensors(ct.Structure):
    _fields_ = [("altitude", ct.c_double), ("ID", ct.c_uint64)]


registry = CentralRegistry()
registry.register_producer("camera", Image)
registry.register_producer("sensors", Sensors)
registry.register_consumer("viewer", {Image, Sensors})

try:
    with registry.lease_slot_for_writing(Image) as buffer:
        image = Image.from_buffer(buffer)
        try:
            image.ID = 42
            image.pixels[:] = [10, 20, 30, 40]
        finally:
            del image

    with registry.lease_slot_for_writing(Sensors) as buffer:
        sensors = Sensors.from_buffer(buffer)
        try:
            sensors.ID = 42
            sensors.altitude = 100.0
        finally:
            del sensors

    with registry.lease_slot_for_reading({Image, Sensors}, "viewer") as buffers:
        capture_id = struct.unpack_from("@Q", buffers[Image], Image.ID.offset)[0]
        altitude = struct.unpack_from(
            "@d", buffers[Sensors], Sensors.altitude.offset
        )[0]
        print(capture_id, altitude)  # 42 100.0
finally:
    registry.close()
```

Deleting the ctypes objects before leaving each write lease releases their
exported buffer references. Any derived NumPy arrays, slices, or memoryviews
must also finish within the lease. Reading again without a new matching
capture would block; raising inside a read lease leaves its cursor unchanged.

## Module helpers and Entry

### `_singleton(datatype)`

Returns `frozenset({datatype})`. The `@cache` decorator reuses that immutable
group for repeated single-primitive requests, avoiding repeated allocation.

### `_primitives(datatype)`

Normalizes a primitive or group into a `frozenset`. Sets and frozensets become
immutable groups; any other input is treated as one primitive via `_singleton`.
The resulting group is the dictionary key used for subscriptions and matching.

### `_frame(length)`

Returns a cached `Struct("<{length}I")` for slot replies: `length` little-endian,
unsigned 32-bit integers. Caching avoids recreating the binary layout for
every lease. A group containing three primitives needs a 12-byte slot reply.

### `Entry.__init__()`

Initializes `readers = 0` and `mask = 0` on an otherwise empty dictionary.
Dictionary items later map primitive classes to slot indices.
The two attributes use `__slots__`, avoiding a separate instance attribute dict.

`readers` pins the entire capture while leases are active. `mask` records which
primitives are present, allowing a bitwise completeness check.

## RegistryClient functions

### `RegistryClient.__init__(pipe, ports, names)`

Stores the pipe, shared-memory names, and an initially empty attachment map.
It caches `ctypes.sizeof` for each primitive so buffer slicing needs no layout
calculation during a lease.

Each port describes a read group or a writable primitive, plus its ordered
primitive tuple. The constructor indexes these ports and prebuilds their three
command headers using `_COMMAND = Struct("<BI")`:

| Operation | Value | Effect |
| --- | --- | --- |
| Request | 0 | Obtain slots for this port. |
| Commit | 1 | Complete the lease successfully. |
| Cancel | 2 | Release the lease without committing it. |

Each header is five bytes: one operation byte and a four-byte port index.
The cached primitive tuple fixes the order used to encode and decode slots.

### `RegistryClient.lease_slot_for_writing(datatype)`

Returns `_lease(datatype)`: a context manager yielding one writable memoryview.
The argument is a registered primitive class. Successful exit publishes its
contents; an exception cancels publication and returns the cell.

### `RegistryClient.lease_slot_for_reading(datatype, consumer)`

Normalizes the primitive or group and returns its lease context manager.
The context yields `{primitive: read_only_memoryview}`, even for one primitive.
On the owner, `consumer` selects the subscription cursor. A remote client's
identity is already bound to its stage connection by `for_stage`.

### `RegistryClient._lease(datatype, consumer=None)`

This is the common view-building layer for both local and remote leases:

1. Enter `_slots` to obtain `{primitive: slot_index}`.
2. Slice each shared-memory allocation using its cached primitive size.
3. Convert read slices with `toreadonly()`.
4. Yield a mapping for reads, or the single primitive's view for writes.
5. Release all generated views in `finally`, before exiting `_slots`.

That cleanup order matters: slots become reusable only after the views have
been released. Exceptions reach the underlying slot context, which cancels
the lease. The owner inherits this method unchanged.

### `RegistryClient._slots(datatype, consumer=None)`

This is the remote implementation of slot management. It sends the cached
request header, blocks for a slot reply, and decodes it using `_frame`.
It attaches to each required shared-memory allocation on first use and reuses
that attachment on later requests.

The method yields a primitive-to-slot dictionary. Normal exit sends commit;
an exception, including attachment failure after the reply, sends cancel.

Commands use `send_bytes` and `recv_bytes`: payloads and per-request Python
objects are not pickled. There is no completion acknowledgement after commit.
The pipe preserves command order, but leaving a remote write lease does not
prove that the owner has already updated its dictionaries at that instant.

### `RegistryClient.close()`

Closes the connection, closes every attached shared-memory handle, and clears
the attachment map. It does not unlink the owner's allocations.
Close after finishing leases and releasing derived buffer references.
Disconnecting also lets the broker cancel any outstanding leases it owns.

## CentralRegistry functions

### `CentralRegistry.__init__(logger=None, matching_window=MATCHING_WINDOW)`

Creates empty metadata, the shared lock, and lifecycle flags. No shared memory
or threads are allocated yet. `logger` is accepted but unused.
`matching_window` defaults to 16 and is clamped to at least one.

The owner does not call the client constructor: it manages slots directly
instead of holding a remote pipe. `prepare` later supplies its size cache.

### `CentralRegistry.register_producer(producer, datatype)`

Associates one primitive class with its producer identity.
A second producer registration for the same datatype raises `ValueError`.
Registration after preparation or shutdown raises `RuntimeError`.

Producer identities are used by `for_stage` to select writable ports.
The local write API assumes callers use the appropriate producer.

### `CentralRegistry.register_consumer(consumer, datatype)`

Normalizes the requested group, creates its latest-ID entry and condition if
needed, and initializes `last_read[consumer, group]` to `-1`.

Consumers of the same group share matching state and a condition, while their
read cursors remain separate. Re-registering a subscription before preparation
resets its cursor. Registration after preparation or shutdown is rejected.

### `CentralRegistry.prepare()`

Freezes registration and allocates storage. It runs explicitly or automatically
on the first lease or `for_stage` call. Repeated calls return once prepared;
calls after close raise `EOFError`. A second check inside the lock prevents
concurrent callers from allocating twice.

For `k` distinct consumers and matching window `w`, it sets:

```text
array_length  = k + w       # Maximum number of retained capture entries.
buffer_length = k + w + 1   # Cells allocated for each primitive.
```

At most `k` captures can be pinned under the operating rules. Since `w >= 1`,
a full entry table still has an unpinned entry available for eviction.
The extra cell lets a producer reserve space before its capture ID is known.

Preparation builds these caches:

| Cache | Purpose |
| --- | --- |
| `_sizes[kind]` | Byte length of one primitive. |
| `_bits[kind]` | A unique power-of-two presence bit. |
| `_masks[group]` | Combined bits required by a group. |
| `_affected[kind]` | Only the groups containing this primitive. |
| `_ids[kind]` | ID field offset and a bound integer decoder. |

It then creates one shared-memory block per primitive and a deque containing
all its slot indices. Only after allocation does it mark itself prepared.

### `CentralRegistry._publish(datatype, slot, capture_id)`

Runs with the metadata lock held and returns whether the slot was retained.
First it advances the global highest-seen ID and looks up the capture entry.

If the entry is absent and the ID is below the global highest ID, publication
is rejected. Existing older entries can still receive missing primitives.
When a new entry is needed and capacity is full, the oldest unpinned entry
is removed and all its primitive slots are returned to their free deques.

A second publication of the same primitive for the same capture is rejected,
preserving the original bytes. The caller reclaims rejected write slots.

For an accepted primitive, the method records its slot and sets its presence
bit. It checks only `_affected[datatype]`, using:

```python
complete = entry.mask & required_mask == required_mask
```

For example, bits `001` and `100` satisfy a group mask of `101`.
A complete capture advances `latest[group]` only if its ID exceeds the previous
value. Only an advancing group's condition receives `notify_all()`.

Eviction does not reset `latest[group]`: that value is historical, and the
read path separately verifies that its capture still exists.

### `CentralRegistry._slots(datatype, consumer=None, stopped=None)`

This overrides the client's pipe implementation with actual ownership logic.
It calls `prepare` and distinguishes a read group from a writable primitive.

For a write, it removes one free slot while holding the lock, then yields it
with the lock released. After successful filling, it decodes the ID directly
from shared memory using the cached offset and decoder, outside the lock.
It then locks only to publish. Failed or rejected writes return their slot;
an accepted write leaves the slot owned by its capture entry.

For a read, it loops under the lock until the group's latest capture exists
and is newer than that subscription's cursor. It checks owner shutdown and
the optional local stage stop event before waiting, raising `EOFError` if stopped.
Otherwise it sleeps on that group's condition, with no timeout or polling.

Once eligible, the entry's reader count is incremented and its slots are
yielded with the lock released. On exit the count is decremented.
Only successful exit advances `last_read`; failure leaves the cursor unchanged,
so a retry can receive that capture again if it remains available.

The loop rechecks its predicate after every wake. Another reader's activity,
shutdown, or a wake without an eligible capture therefore cannot return stale
or reclaimed cells. Successful writes avoid an extra cleanup lock acquisition.

### `CentralRegistry.for_stage(stage, stopped)`

Prepares the owner and selects the stage's subscribed groups and produced
primitives. These become the connection's indexed read and write ports.

It creates a duplex command pipe, a separate one-way wake pipe, and two daemon
threads: `_serve` handles requests and `_stop` waits for the stage's event.
It records these threads for shutdown and returns a `RegistryClient` containing
the remote endpoint, port definitions, and shared-memory names.

It also creates a local `threading.Event`. The watcher mirrors shared shutdown
into this event, so broker and read-loop checks avoid multiprocessing locks.
The shared event remains the signal used by stage processes and owner shutdown.

The client can be passed to a spawned stage process. Only the owner has the
matching dictionaries and broker threads. The stop event must be shared with
that process, and unused copies of pipe endpoints should be closed so that
disconnects can produce EOF.

### `CentralRegistry._stop(stopped, groups, signal, local_stop)`

Blocks in `stopped.wait()` until shutdown is requested.
It sets the local stop event before acquiring the metadata lock and notifying
the stage's subscribed groups, waking brokers blocked inside read requests.

Finally it closes the wake pipe's sending endpoint. EOF makes the receiving
endpoint ready, waking an idle broker blocked in `selector.select()`.
The watcher uses neither periodic event checks nor timed waits.

### `CentralRegistry._serve(pipe, wake, ports, stage, stopped, local_stop)`

This is one connection's broker loop. It caches reply layouts and holds active
slot contexts in `leases`, indexed by port. One `PollSelector` registers the
command and wake pipes once and is reused until the broker exits. With no
active lease, `select()` blocks for a command or wake-pipe EOF, without a timer.
The loop and read requests check `local_stop`; only the watcher waits on the
shared event. Selector setup and multiprocessing stop checks stay off this path.

For request operation 0, it enters the owner's slot context and replies with
the ordered slot indices. That context remains open across pipe messages.
For commit operation 1, it exits the context normally. For cancel operation 2,
it passes `GeneratorExit` into the context to skip success processing while
still running cleanup. This does not terminate the broker thread.

A stage can hold an input lease while acquiring a different output port.
However, concurrent requests from multiple callers on one client are not
supported: replies have no request identifier or demultiplexing layer.

After stop is set, active leases are allowed to finish or disconnect before
the loop exits. With leases held, the broker waits for commands using
`recv_bytes`; it does not revoke cells still in use.

EOF or a broken pipe ends service. The `finally` block cancels remaining
leases, sets the shared stop event to release the watcher, and closes both
endpoints. It also closes the selector, including on cancellation or disconnect.

### `CentralRegistry.close()`

Marks the owner closed under the lock, sets all recorded stage stop events,
and notifies every group condition. It then releases the lock before joining
broker and watcher threads, allowing their cleanup to acquire it.

Finally it closes and unlinks all owned shared-memory blocks and clears their
map. Entry dictionaries and cursor values remain historical metadata.
There is no join timeout: a connected client holding a lease must finish or
disconnect before shutdown can complete.

## Operating rules

- Register all producers and consumers before preparation or first use.
- Use one producer and at most one outstanding write lease per primitive.
- Hold at most one read lease per consumer, across all its subscriptions.
- Use each remote client sequentially; nested input/output ports are allowed.
- Give every subscribed primitive a producer and use nonempty groups.
- Use fixed-size, pointer-free ctypes layouts with an integer `ID` field or alias.
- ID widths must be 1, 2, 4, or 8 bytes; decoding uses native unsigned integers.
- Use nonnegative IDs from one shared increasing capture sequence. Do not reset
  counters while the registry is live.
- Keep all views and objects referencing their bytes inside the lease.
- Finish local leases and close/join stage clients before closing the owner.

These are caller contracts rather than a comprehensive validation layer.
For example, an unsupported ID width or a subscribed type without a producer
fails during preparation, and overlapping writes can exhaust free cells.
The compact implementation does not provide transactional preparation rollback.

Idle consumers sleep on conditions; idle brokers sleep on pipe readiness;
stop watchers sleep on events. Normal publication checks only affected groups,
and binary commands carry slot metadata rather than payloads. These are the
main mechanisms that keep CPU use low while retaining the small implementation.
