# Metadata saver

`sft_cv.common.metadata_saver.MetadataSaver` persists images and their associated
`PhotoExif.pix_data`. The aggregator must supply a dedicated FIFO of already
matched `(numpy.ndarray, PhotoExif)` pairs. The saver saves every pair in order;
it does no trigger matching, geofencing, or mission filtering.

The aggregator and orchestrator are currently stubs, so the stage still needs
to be connected when those components are implemented. The intended connection
is:

```python
import multiprocessing as mp

from sft_cv.common.metadata_saver import MetadataSaver

# Assemble in the application's guarded main entry point, alongside other stages.
saver_input = mp.Queue(maxsize=10)
saver = MetadataSaver(configs, logger, in_q=saver_input)
saver.start()

# In the aggregator, after attaching the matching PixData to photo_exif:
saver_input.put((image, photo_exif))

# The aggregator sends this after its last frame, using the same FIFO producer.
saver_input.put(None)
saver.join()
```

The queue must preserve every matched frame until consumed. A buffer that only
retains the latest frame cannot guarantee saving every image. A bounded queue
applies backpressure if the disk falls behind.

For the existing shared-memory transport, enqueue `photo_exif` alone with its
`shm_name`, `shm_shape`, and `shm_dtype` populated, and pass the garbage collector's
`ack_q` to the saver. Register the saver as a consumer of each block. The saver
closes its view and sends `ACK` after each write attempt, including failures;
the garbage collector owns unlinking. The producer must not overwrite or release
the buffer before consumption finishes. The tuple form does not manage shared
memory ownership.

With the existing configuration:

```yaml
camera:
  img_out_dir: ./data/outs
picture_saver:
  saving_mode: NPY
  metadata_db: ./data/outs/metadata.csv
```

one run produces:

```text
data/outs/20260923-143000/
  0_2026-09-23-14-30-00.123.npy
  1_2026-09-23-14-30-00.123.npy
  metadata_20260923-143000.csv
```

The CSV contains exactly the 21 columns in `MetadataSaver.METADATA_COLUMNS`, in
that order. Each filename points to the saved `.npy` image. Timestamps are Unix
seconds in `PhotoExif`, formatted as local `YYYY-MM-DD HH:MM:SS` in the CSV, as in
the camera engines. Mission enums are written as their numeric values. Missing
attributes are blank, including `vground` when absent from the current `PixData`.

Images retain their original NumPy shape, dtype, and pixels. `NPY` and `NUMPY`
select this format; other formats are unsupported. If `metadata_db` has a
different parent directory or basename, those are preserved and the same run
timestamp is inserted into its path and filename.

File handles open inside the consuming process. Each CSV row is flushed and
synced after its image is written. Write failures propagate and the mission
closes the CSV. An existing run CSV is never truncated: a timestamp collision
raises `FileExistsError`. For normal shutdown, use the FIFO sentinel shown above;
`stop()` also drains queued frames once the producer has stopped.

For synchronous use, call `saver.save(image, photo_exif)` and close the saver in
a `finally` block. `save()` returns a copy of the metadata with the saved filename;
the input metadata remains unchanged for other consumers.
