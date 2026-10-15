# Registry demo stages

Run from the project root:

```sh
PYTHONPATH=src .venv/bin/python tools/central_registry_demo.py
```

Each stage follows `src/stages/<role>/default/`: a stage class, a Python config
exposing `CONFIG`, and a YAML input/output contract.
All four classes inherit directly from `PipelineStage`, accept its normal
`configs, logger, registry, stage_name` constructor arguments and use its `run()`
loop. Their `update()` methods load their Python defaults and per-role overrides.

`mission_config.yaml` selects stage implementations and supplies their settings.
`read_profile()` flattens each list of options into a dictionary; `Pipeline`
passes that dictionary to the corresponding stage as `configs`. Stages merge
it with their own `CONFIG` defaults. Logger settings live in `utility_config.yaml`.
Both launchers use these settings. At `INFO`, logs show registrations, shared-memory
allocation, each stage's first activity and shutdown totals. Set `logger.log_level`
to `DEBUG` to see published captures, reader leases and evictions.
Stale captures are logged at `WARNING`; exhausted entries at `ERROR`.

Each completed input also emits an `INFO` record prefixed `[registry:data]`,
with producer, datatype, capture ID, slot and byte size. `CentralRegistry._record_data()`
is the temporary recorder: replace its body with CSV or SQLite persistence later.
It runs before matching or stale-ID discard, outside the metadata lock, while the
buffer is still leased. It currently records metadata rather than saving payloads;
the future writer must finish using the buffer before returning.

| Role | Config | Options |
| --- | --- | --- |
| Image generator | `image_generator_config.py` | Resolution, channels, rate, first capture ID, moving bar and exposure metadata |
| Sensor generator | `sensor_generator_config.py` | Rate, first capture ID, coordinates, altitudes and waypoints |
| Viewer | `viewer_config.py` | Display enable, preview resolution, window name and frame limit |
| Dummy consumer | `dummy_consumer_config.py` | Process count and random lease hold delay range |

Keep the generators' rates and first IDs equal for this demo. They increment
their own counters and run on independent schedules. Matching uses capture IDs,
so image and sensor arrival times can differ without synchronizing the generators.
Image dimensions and channels belong to the producer: its `CONFIG` provides
defaults, overridden under that stage in `mission_config.yaml`. The selected
stage whose contract outputs `tagged_image` determines the `PhotoExif` layout
used by producers, consumers and shared-memory allocation. Restart after
changing the layout.

The generators run in separate processes. The launcher constructs one
`CentralRegistry`, registers inputs and outputs, and passes that registry to each
stage, exactly like `Pipeline`. `PipelineStage` obtains a process-safe reference
when started; transport and shared-memory attachment are internal to the registry.
Storage, matching and process transport all live in `core/central_registry.py`.
The [registry design guide](central_registry.md) explains its bit masks, caches,
lease lifetimes and packed message format with worked examples.
Shared-memory names cross each pipe once per block; later leases send packed
block indexes and byte offsets. Payloads are never sent over the pipes.
Producers fill shared memory
directly, and the registry associates their IDs after release. Readers hold a
complete capture until they release their views. Only the display preview copies
pixels.
Each primitive has its own slot pool: image and sensor data for one capture ID
can occupy different physical slot numbers. The registry matches through
`entries[capture_id][datatype]`. A missing capture ID below the highest received
ID is discarded; an older entry still present can receive its missing fields.

`MATCHING_WINDOW = 16` in `core/central_registry.py` budgets recent captures
for delayed matching, separately from entries pinned by readers. With `k`
consumers, the registry retains up to `k + 16` entries and allocates `k + 17`
cells per stripe. The extra cell lets a producer write before its ID is known
without first evicting a possible match. A 16-capture window covers a lead of
15 captures (250 ms at 60 Hz); longer delays can exhaust retention, evict old
incomplete captures and produce stale-ID warnings. Reader-held entries remain
protected. The window can also be set with `CentralRegistry(matching_window=...)`.

Both consumers subscribe to `main_camera`, the `PixData` + `PhotoExif` group.
Press Q or Esc to stop. For a headless run, set the viewer's `display` to `False`
and `max_frames` to a positive value in `mission_config.yaml`.

To run the configured pipeline:

```sh
PYTHONPATH=src .venv/bin/python -m core.pipeline
```

`Pipeline.build()` constructs and registers stages. `Pipeline.run()` starts them
and waits for them to exit. `PipelineStage.start()` gets a process-safe registry
reference; the registry handles leases and shared-memory access. Stages run
independently with their own stop flags, so exiting the viewer leaves the other
stages running. Ctrl+C stops the configured pipeline and releases its buffers.
