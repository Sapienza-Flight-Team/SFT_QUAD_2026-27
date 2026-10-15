import datetime
import multiprocessing as mp
import time
from typing import Optional, Tuple, Any, TextIO
from shapely.geometry import Point, Polygon

import socket
import struct
import zstandard as zstd
import threading
import queue
from multiprocessing import shared_memory

import numpy as np
from numpy import _typing
from common.config import SFTConfig
from core.logger import SFTLogger
from common.structs import PhotoExif
from common.pipeline_stage import PipelineStage
import os
from sft_math.geo.utils import magnorth2truenorth
import cv2
import csv


class UnrealCameraEngine(PipelineStage):
    def __init__(
        self,
        configs: SFTConfig,
        logger: SFTLogger,
        in_q: Optional[mp.Queue],
        out_q: Optional[list[mp.Queue]],
        idx: int,
        in_command_q: Optional[list] = None,
        out_command_q: Optional[list] = None,
        ack_q: Optional[mp.Queue] = None,
        stage_name: str = "camera",
    ) -> None:
        super().__init__(
            configs=configs,
            logger=logger,
            out_q=out_q,
            in_q=in_q,
            in_command_q=in_command_q,
            out_command_q=out_command_q,
            ack_q=ack_q,
            stage_name=stage_name,
            idx=idx,
        )
        self.ready = False
        self.camera_was_interrupted = True
        self.to_flush_queue = True

        self.stopped = mp.Event()
        self.stopped.clear()

        self.payload_queue: Optional["queue.Queue[Any]"] = None
        self.tcp_stop_event: Optional[threading.Event] = None
        self.tcp_thread: Optional[threading.Thread] = None

        self.config_dict = configs.config_dict
        self.yaml_path = self.config_dict["utils"]["yaml_path"]
        self.unreal_host_ip = configs.config_dict["mavlink"]["unreal_host_ip"]
        self.unreal_host_port = configs.config_dict["mavlink"][
            "unreal_video_stream_port"
        ]
        self.unreal_connected = False
        self.out_dir = configs.config_dict["camera"]["img_out_dir"]
        self.mode = self.config_dict["picture_saver"]["saving_mode"]
        self.airdrop_bound = self.config_dict["airdrop_bound"]
        if self.out_q:
            self.camera_missions = tuple(range(1, len(self.out_q) + 1))
        else:
            self.camera_missions = tuple(range(1, 4))

        self.geo_fence = Polygon(
            [
                self.airdrop_bound["bl"],
                self.airdrop_bound["br"],
                self.airdrop_bound["tr"],
                self.airdrop_bound["tl"],
            ]
        )
        self.initial_timestamp = (
            datetime.datetime.fromtimestamp(time.time())
            .isoformat(timespec="milliseconds", sep="-")
            .replace(":", "-")
        )

        self.METADATA_COLUMNS = {
            "filename": "filename",
            "timestamp": "timestamp",
            "exposure_time": "exposure_time",
            "gain": "gain",
            "black_level": "black_level",
            "bsl_color_space": "bsl_color_space",
            "gamma": "gamma",
            "light_source_preset": "light_source_preset",
            "pixel_format": "pixel_format",
            "teensy_timestamp": "pix_data.teensy_timestamp",
            "lat": "pix_data.lat",
            "lon": "pix_data.lon",
            "alt_msl": "pix_data.alt_msl",
            "alt_rel": "pix_data.alt_rel",
            "vground": "pix_data.vground",
            "roll": "pix_data.roll",
            "pitch": "pix_data.pitch",
            "yaw": "pix_data.yaw",
            "heading": "pix_data.heading",
            "mission_mode": "pix_data.mission_mode",
            "mission_state": "pix_data.mission_state",
            "gripper": "pix_data.gripper",
        }
        self.saved_img_count = 0
        db_creation_time = time.strftime("%Y%m%d-%H%M%S")
        self.out_dir = (
            configs.config_dict["camera"]["img_out_dir"] + f"/{db_creation_time}"
        )
        self.out_metadata = configs.config_dict["picture_saver"][
            "metadata_db"
        ]  # csv file with metadata
        # append the hash to the file base name
        self.out_metadata = self.out_metadata.replace(
            "metadata.csv", f"{db_creation_time}/metadata_{db_creation_time}.csv"
        )

        self.first_pic_datetime = (
            datetime.datetime.fromtimestamp(time.time())
            .isoformat(timespec="milliseconds", sep="-")
            .replace(":", "-")
        )

        # Do not open the file here (parent process). The file handle will be
        # opened in the child process inside `setup_cam` to avoid pickling
        # non-serializable objects when using the 'spawn' start method.
        # annotate types for static type checkers
        self.f: Optional[TextIO] = None
        # csv.writer has no stable public type, use Any for the writer
        self.writer: Optional[Any] = None
        self._metadata_initialized = False
        self._should_save_image = True

    def _recvall(self, sock: socket.socket, count: int) -> Optional[bytearray]:
        """Helper to ensure we receive EXACT bytes, without permanently freezing."""
        buf = bytearray()
        while count > 0:
            # If the pipeline shuts down, exit cleanly immediately
            if self.tcp_stop_event is not None and self.tcp_stop_event.is_set():
                return None

            try:
                newbuf = sock.recv(count)
                if not newbuf:
                    return None
                buf.extend(newbuf)
                count -= len(newbuf)
            except socket.timeout:
                # This prevents the freeze! It just loops around and checks the stop_event again.
                continue
            except BlockingIOError:
                continue
            except Exception as e:
                self.logger.warning(f"Network stream dropped: {e}")
                return None
        return buf

    def _tcp_worker(self):
        import os

        try:
            os.sched_setaffinity(0, {1})
            self.logger.info("TCP Worker pinned to CPU Core 1")
        except AttributeError:
            pass

            # --- THE WAKE-UP PING ---
        # Trick Unreal Engine into starting its TCP server by firing a dummy picture command
        try:
            wakeup_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # NOTE: Make sure to use the exact UDP port your Blueprint is listening to (e.g., 3001 or 3005)
            wakeup_sock.sendto(b"SNAP\0", (self.unreal_host_ip, 3001))
            wakeup_sock.close()
            self.logger.info("[UDP] Fired Wake-Up Ping to open Unreal Engine port")
            time.sleep(0.5)  # Give Unreal half a second to boot the server
        except Exception as e:
            self.logger.warning(f"Wakeup ping failed: {e}")

        client_socket = None

        # --- CONNECTION LOOP ---
        while not self.tcp_stop_event.is_set():
            self.logger.info(
                f"[TCP] Connecting to Unreal Server at {self.unreal_host_ip}:{self.unreal_host_port}..."
            )

            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16777216)

            # Short timeout so it doesn't hang if Unreal is closed
            client_socket.settimeout(2.0)

            try:
                client_socket.connect((self.unreal_host_ip, int(self.unreal_host_port)))

                # IMPORTANT: DO NOT SET TIMEOUT TO NONE.
                # Keep a short timeout (0.5s) active during the stream so _recvall never freezes.
                client_socket.settimeout(0.5)

                self.unreal_connected = True
                self.logger.info("[TCP] Connected! Receiving stream...")
                break

            except Exception as e:
                self.logger.warning(f"Failed to connect: {e}")
                client_socket.close()
                time.sleep(1)

        try:
            while not self.tcp_stop_event.is_set() and self.unreal_connected:
                t_net_start = time.perf_counter()

                # Because of our updated _recvall, this will now safely time out every 0.5s
                # without breaking the loop, ensuring the thread stays highly responsive.
                size_data = self._recvall(client_socket, 4)
                if not size_data:
                    continue  # Keep looping and waiting

                payload_size = struct.unpack("<I", size_data)[0]
                compressed_payload = self._recvall(client_socket, payload_size)

                if compressed_payload:
                    network_end_time = time.perf_counter()
                    network_duration = (network_end_time - t_net_start) * 1000

                    if self.payload_queue.full():
                        try:
                            self.payload_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.payload_queue.put(
                        (compressed_payload, network_end_time, network_duration)
                    )

        except Exception as e:
            self.logger.error(f"TCP Worker Error: {e}")
        finally:
            if client_socket:
                client_socket.close()
            self.unreal_connected = False
            self.logger.info("TCP Worker shutting down.")

    def setup_metadata_writer(self):

        try:
            if not getattr(self, "_metadata_initialized", False):
                out_dir = os.path.dirname(self.out_metadata)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                # Open CSV for writing metadata
                self.f = open(self.out_metadata, mode="w", newline="", encoding="utf-8")
                self.writer = csv.writer(self.f)
                # write header
                try:
                    self.writer.writerow(list(self.METADATA_COLUMNS.keys()))
                except Exception:
                    pass
                self._metadata_initialized = True
        except Exception:
            # best-effort: don't crash the camera setup if metadata writer can't be created
            self.logger.warning(
                "Could not initialize metadata writer; metadata will be skipped"
            )

    def mission(self) -> bool:

        self.payload_queue = queue.Queue(maxsize=5)
        self.tcp_stop_event = threading.Event()
        self.zstd_decompressor = zstd.ZstdDecompressor()

        self.setup_metadata_writer()

        def wait_for_ready():
            self.check_inbound_command(block=True, timeout=100)

        while not self.stopped.is_set():

            wait_for_ready()
            if self.stopped.is_set():
                break
            self.img_count = 0

            self.tcp_stop_event.clear()
            tcp_thread = threading.Thread(target=self._tcp_worker, daemon=True)
            tcp_thread.start()
            self.tcp_thread = tcp_thread

            while self.ready and not self.stopped.is_set():
                self.check_inbound_command()
                if self.ready == False:
                    break
                try:
                    ret = self.acquire_taggedpic()
                except Exception as e:
                    self.logger.warning(
                        f"Error {type(e).__name__}: {e} encountered while acquiring image"
                    )
                    self.camera_was_interrupted = True
                    ret = None
                    continue

                if not ret:
                    continue

                image, metadata, shm = ret

                if self._should_save_image:
                    self.save(image, metadata)
                self.put(metadata)
                shm.close()

                self.img_count += 1

            # Mission loop cleanup
            self.tcp_stop_event.set()
            if self.tcp_thread and self.tcp_thread.is_alive():
                self.tcp_thread.join(timeout=2.0)

            if self.stopped.is_set():
                self.logger.info(f"Captured {self.img_count} images")
                self.logger.info(
                    "Quitting camera connection and shutting down gracefully"
                )
                if self.f is not None:
                    try:
                        self.f.flush()
                    except Exception:
                        pass
                self.logger.info("Camera connection closed")
                return True

        return self.stopped.is_set()

    def stop(self) -> None:
        self.logger.info("Setting stop flag on UnrealCameraEngine")
        self.stopped.set()
        if self.tcp_stop_event is not None:
            self.tcp_stop_event.set()  # Ensure network thread shuts down

        try:
            if self.f is not None:
                try:
                    self.f.flush()
                except Exception:
                    pass
                try:
                    self.f.close()
                except Exception:
                    pass
        except Exception:
            pass

    def setup_cam(self) -> None:
        if getattr(self, "_metadata_initialized", False):
            self.logger.warning("Metadata writer already initialized")
            return
        start = time.time()
        try:
            if not getattr(self, "_metadata_initialized", False):
                if self.out_dir:
                    os.makedirs(self.out_dir, exist_ok=True)
                out_dir = os.path.dirname(self.out_metadata)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                self.f = open(self.out_metadata, mode="w", newline="", encoding="utf-8")
                self.writer = csv.writer(self.f)
                try:
                    self.writer.writerow(list(self.METADATA_COLUMNS.keys()))
                except Exception:
                    pass
                self._metadata_initialized = True
        except Exception:
            self.logger.warning(
                "Could not initialize metadata writer; metadata will be skipped"
            )

        stop = time.time() - start
        self.logger.debug(f"Metadata writer setup took {stop} seconds")
        return

    def acquire_taggedpic(
        self,
    ) -> Optional[Tuple[np.ndarray, PhotoExif, shared_memory.SharedMemory]]:
        t_wait_start = time.perf_counter()

        if self.to_flush_queue and self.in_q is not None:
            from queue import Empty

            while True:
                try:
                    self.in_q.get_nowait()
                except Empty:
                    break
            self.logger.info("Queue flushed")
            self.to_flush_queue = False

        try:
            if self.payload_queue is None:
                return None

            try:
                compressed_payload, network_end_time, network_duration = (
                    self.payload_queue.get(timeout=0.5)
                )
            except queue.Empty:
                return None

            t_wait_end = time.perf_counter()
            idle_wait_latency = (t_wait_end - t_wait_start) * 1000
            queue_latency = (t_wait_end - network_end_time) * 1000

            frame_shape = (3032, 5320)
            frame_size = 3032 * 5320

            shm, ref_count, ref_lock = self.allocate_shm_with_refcount(frame_size)

            with self.zstd_decompressor.stream_reader(compressed_payload) as reader:
                view = shm.buf
                assert view is not None
                offset = 0
                while offset < frame_size:
                    bytes_read = reader.readinto(view[offset:])  # type: ignore[index]
                    if not bytes_read:
                        break
                    offset += bytes_read

            t_end_decomp = time.perf_counter()
            decomp_latency = (t_end_decomp - t_wait_end) * 1000

            bayer_rggb: np.ndarray = np.ndarray(
                frame_shape, dtype=np.uint8, buffer=shm.buf
            )

            t_end_numpy = time.perf_counter()
            numpy_latency = (t_end_numpy - t_end_decomp) * 1000

            try:
                pix_data = self.receive_from_queue(timeout=100)
            except Exception as e:
                self.logger.warning(f"probable timeout exception: {e}")
                shm.close()  # Clean up memory if we abort
                return None

            try:
                if pix_data is None or not pix_data:
                    shm.close()
                    return None
            except Exception as e:
                self.logger.error(f"GG, go next, {e}")
                self.ready = False
                shm.close()
                return None

            t_end_mavlink = time.perf_counter()
            mavlink_latency = (t_end_mavlink - t_end_numpy) * 1000

            total_local_latency = (
                queue_latency + decomp_latency + numpy_latency + mavlink_latency
            )
            payload_mb = len(compressed_payload) / (1024 * 1024)

            self.logger.debug(f"Frame Processed | {payload_mb:.2f} MB")
            self.logger.debug(f"  [Network Thread]")
            self.logger.debug(f"  ├─ Frame Download: {network_duration:6.2f} ms")
            self.logger.debug(f"  [Main Thread]")
            self.logger.debug(f"  ├─ Idle Waiting: {idle_wait_latency:6.2f} ms")
            self.logger.debug(f"  ├─ Queue Stale:  {queue_latency:6.2f} ms")
            self.logger.debug(f"  ├─ Zstd Decmp:   {decomp_latency:6.2f} ms")
            self.logger.debug(f"  ├─ NumPy Init:   {numpy_latency:6.2f} ms")
            self.logger.debug(f"  ├─ MAVLink Sync: {mavlink_latency:6.2f} ms")
            self.logger.debug(f"  └─ Total CPU:    {total_local_latency:6.2f} ms\n")

            timestamp = time.time_ns()

            metadata = PhotoExif(
                filename=os.path.join(
                    self.out_dir, f"{self.img_count}_{self.initial_timestamp}.npy"
                ),
                exposure_time=0.0,
                timestamp=timestamp,
                gain=0.0,
                black_level=0,
                bsl_color_space="",
                gamma=0,
                light_source_preset="",
                pixel_format="BayerRG8",
                pix_data=pix_data,
                shm_name=shm.name,
                shm_shape=frame_shape,
                shm_dtype="uint8",
                ref_count=ref_count,
                ref_lock=ref_lock,
            )

            return bayer_rggb, metadata, shm

        except Exception as e:
            self.logger.warning(f"Error during acquisition: {e}")
            return None

    def is_in(self, resp: Any) -> bool:
        if not resp:
            self.logger.error(f"unable to locate drone position, returning True")
            return True
        is_in_one_of_fences = Point(resp.lat, resp.lon).within(self.geo_fence)
        return is_in_one_of_fences

    def save(self, image: np.ndarray, metadata: PhotoExif) -> None:

        np.save(metadata.filename, image, allow_pickle=False)
        row = []
        for path in self.METADATA_COLUMNS.values():
            # format timestamp as ISO string, others are raw values
            try:
                if path == "timestamp":
                    ts = getattr(metadata, "timestamp")
                    val = datetime.datetime.fromtimestamp(ts).isoformat(
                        timespec="seconds", sep=" "
                    )
                else:
                    val = self._resolve_path(metadata, path)
            except Exception:
                val = ""
            row.append(val)

        if self.writer is not None:
            try:
                self.writer.writerow(row)
                # flush to ensure data is written to disk immediately
                try:
                    if self.f is not None:
                        self.f.flush()
                        import os as _os

                        try:
                            _os.fsync(self.f.fileno())
                        except Exception:
                            # some platforms or file-like objects may not support fsync
                            pass
                except Exception:
                    pass
            except Exception:
                self.logger.warning("Failed to write metadata row")
        else:
            self.logger.warning(
                "Metadata writer not initialized; skipping metadata write"
            )

    def _resolve_path(self, obj: object, path: str):
        parts = path.split(".")
        val = obj
        for p in parts:
            val = getattr(val, p)
        return val

    def put(self, metadata: PhotoExif) -> None:
        self.send_to_queues(metadata)

    def get(self) -> None:
        raise NotImplementedError

    def _dump_data(self) -> None:
        pass

    def ff15(self) -> None:
        self.ready = False

    def glhf(self) -> None:
        self.logger.info("All clear to start CameraEngine!")
        self.ready = True

    def bullseye_stop(self) -> None:
        pass

    def perimeter_stop(self) -> None:
        pass

    def numbers_stop(self) -> None:
        pass

    def error(self) -> None:
        pass
