# DataConnection

The contract-based pipeline in `src/core/pipeline.py` connects complete,
pointer-free `ctypes.Structure` messages. Image pixels live in each message's
`image` field; no separate image argument or BufferLayout is required.

The project-root `mission_config.yaml` fixes the image layout:

```yaml
image:
  resolution: [5320, 3032]  # [width, height]
  channels: 1              # uint8; Bayer mosaics have one channel
```

`common.new_structs` reads this file relative to its own location when imported,
independently of the working directory. `PhotoExif()` and
`AggregatedMetadata()` need no dimension arguments. Their image fields are
`ctypes.c_uint8 * (width * height * channels)`. Fields remain fixed for the
process lifetime; restart the application after editing image settings.

`DATA_TYPE_REGISTRY` maps each contract stream name to the keyword arguments
for `DataConnection`:

| Stream | Structure | Transport |
| --- | --- | --- |
| `sensors` | `PixData` | FIFO |
| `tagged_image` | `PhotoExif` | Shared memory |
| `aggregator_output` | `AggregatedMetadata` | Shared memory |
| `resized_aggregator_output` | `AggregatedMetadata` | Shared memory |

The two aggregated streams have separate connections even though their schemas
match. Both currently have the same configured image capacity; the registry does
not define a separate smaller layout for resized images.

```python
from common.data_connection import DataConnection
from common.new_structs import DATA_TYPE_REGISTRY, PhotoExif

connection = DataConnection(**DATA_TYPE_REGISTRY["tagged_image"])
connection.register_producer("camera")
connection.register_consumer("aggregator")
try:
    photo = PhotoExif(frame_counter=42)
    photo.image[0] = 255
    connection.push_onto_connection("camera", photo)
    for received in connection.get_from_connection("aggregator"):
        assert received.frame_counter == 42
        assert received.image[0] == 255
finally:
    connection.close(unlink=True)
```

Shared memory allocates one `sizeof(data_type)` buffer and lock per producer,
with an event per consumer. Reads copy the entire message under the lock and
clear that consumer's event. Returned structures remain valid after later
writes or buffer cleanup. With `wait_for_consumer=True`, the next write waits
until all consumers have copied the previous message.

FIFO allocates one queue per producer/consumer pair, broadcasting each message
to every consumer. A read returns at most one ready message from each producer;
shared-memory reads likewise return at most one message per producer.
Both transports raise `queue.Empty` when nothing is ready.

The pipeline validates contracts and stage constructors, registers every endpoint,
then instantiates and starts the stages. Stage connection dictionaries use stream
names. `send_to_connections(data, stream="name")` chooses one output; omitting
the name publishes to all declared outputs with the matching structure.
`receive_from_connection("name")` chooses one input; passing a structure class
is supported only when that class identifies exactly one input.

After stopping and joining children, the pipeline calls
`connection.close(unlink=True)` to release shared memory and queues. Closing
FIFO connections during shutdown may discard unread messages.

The default profile is still part of an unfinished stage migration:
`px4_commands` has no registered schema, the mock camera and aggregator are
placeholders, and the communicator still has its old queue constructor.
Those stages must implement the connection interface before this profile can run.
