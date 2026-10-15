import ctypes as ct
from multiprocessing import Event, Lock, Queue, shared_memory
from queue import Empty
from time import sleep
from typing import Literal


class DataConnection:
    """
    Every data type flowing inside the pipeline is a complete ctypes.Structure
    transferred through an object of this class, under the assumption that throughout the entire life of the pipeline all stages will always work with the same data_type(s)
    and that all stages which work on the same data type want the output of the same stages producing them
    """

    def __init__(
        self,
        connection_type: Literal["fifo", "shm"],
        wait_for_consumer: bool,
        data_type: type[ct.Structure],
    ):
        self.connection_type = connection_type
        self.wait_for_consumer = wait_for_consumer
        self.data_type = data_type # concrete class of the objects flowing in the connection
        self.producers = set() # set of STAGE NAMES
        self.consumers = set() # set of STAGE NAMES

        '''
        if connection is fifo:
            then self.CONNECTION = dict{consumer: mp.Queue()}
        else
            self.CONNECTION = dict{producer: shm}
        '''
        self.CONNECTION = {}

        self.events = {} # dict{producer: dict{consumer: mp.Event()}} serves mainly for the consumer to know there is new data on a producer's connection, but also for the producer to know if the consumers have consumed the latest data (producers * consumers number of events)
        self.locks = {} # dict{producer: mp.Lock()} we don't want to read while writing to shm and vice-versa, one per shared memory buffer

    def register_producer(self, producer):
        if producer in self.producers:
            raise Exception("Producer already present!")
        
        if self.connection_type == "shm":
            self.CONNECTION[producer] = shared_memory.SharedMemory(
                create=True, size=ct.sizeof(self.data_type)
            )
            self.locks[producer] = Lock()

            # if we already have consumers on the DataConnection, we need to create new events for the new producer for all the existing consumers
            self.events[producer] = {
                consumer: Event() for consumer in self.consumers
            }
        else:
            self.CONNECTION[producer] = {
                consumer: Queue() for consumer in self.consumers
            }
        self.producers.add(producer)

    def register_consumer(self, consumer):

        if consumer in self.consumers:
            raise Exception("Consumer already present!")
        if self.connection_type == "shm":
            for producer in self.producers:
                self.events[producer][consumer] = Event()
        else:
            for producer in self.producers:
                self.CONNECTION[producer][consumer] = Queue()

        self.consumers.add(consumer)


    def register(self, direction, stage):
        if direction == "input":
            self.register_producer(stage)
        if direction == "output":
            self.register_consumer(stage)

    def push_onto_connection(self, producer, data: ct.Structure):

        # check if the type of data pushed onto the link matches the type of data
        # stored in self.data_type, especially important so that we don't misuse the preallocated space
        
        if type(data) is not self.data_type:
            raise TypeError(f"Expected {self.data_type.__name__}, got {type(data).__name__}")
        if self.connection_type == "fifo":
            for consumer in self.consumers:
                self.CONNECTION[producer][consumer].put
            return
        
        if producer is None:
            raise Exception("where are you pushing your data? no producer shm specified")
        events = self.events[producer]

        # in practice we never need the following, but what *if* we wanted to wait for consumers to read from shm before overwriting it?
        if self.wait_for_consumer:
            while any(event.is_set() for event in events.values()):
                sleep(0.001)

        with self.locks[producer]:
            buffer = self.CONNECTION[producer].buf
            buffer[: ct.sizeof(self.data_type)] = memoryview(data).cast("B")
            for event in events.values():
                event.set()

    def get_from_connection(self, consumer) -> list[ct.Structure]:

        # returns at most (n producers on the DataConnection) outputs

        output = []
        if self.connection_type == "fifo":
            for producer in self.producers:
                try:
                    output.append(self.CONNECTION[producer][consumer].get_nowait())
                except Empty:
                    pass
        else:
            for producer, events in self.events.items():
                if events[consumer].is_set():
                    with self.locks[producer]:
                        buffer = self.CONNECTION[producer].buf
                        output.append(self.data_type.from_buffer_copy(buffer))
                        
                        events[consumer].clear()
        if not output:
            raise Empty
        return output

    def close(self) -> None:
        if self.connection_type == "shm":
            for producer in self.producers:
                self.CONNECTION[producer].close()
                self.CONNECTION[producer].unlink()
