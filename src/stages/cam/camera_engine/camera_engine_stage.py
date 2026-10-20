"""Acquire into leased shared memory and optionally clock fake sensors."""

from time import monotonic

from pypylon import genicam, pylon

from common.new_structs import PhotoExif
from common.pipeline_stage import PipelineStage
from .camera_buffer import LeasedBuffer
from .camera_setup import reset_frame_counter, set_enum, setup_camera


class CameraEngine(PipelineStage):
    def __init__(self, configs, logger, registry, stage_name):
        super().__init__(configs, logger, registry, stage_name)
        self.cam = self.capture_connection = None
        self.capture_period = 0
        self.last_frame_id = None

    def setup_cam(self):
        if not (
            self.configs["chunk_mode_active"]
            and self.configs["chunk_enable"]
            and self.configs["chunk_selector"] == "CounterValue"
        ):
            raise ValueError("Camera capture requires an enabled CounterValue chunk")
        self.cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        self.cam.Open()
        setup_camera(self.cam, self.configs)
        self.next_frame_at = monotonic()

    def reset_frame_counter(self):
        """Reset without rebooting; requires paused triggers and a fresh session."""
        reset_frame_counter(self.cam, self.configs["chunk_counter_selector"])
        self.last_frame_id = None

    def _capture_id(self, result):
        """Read the selected counter's snapshot belonging to this exact image."""
        nodemap = result.GetChunkDataNodeMap()
        set_enum(
            nodemap, "ChunkCounterSelector", self.configs["chunk_counter_selector"]
        )
        node = nodemap.GetNode("ChunkCounterValue")
        if node is None or not genicam.IsReadable(node):
            raise RuntimeError("Grab result has no readable ChunkCounterValue")
        capture_id = node.GetValue()
        if capture_id < 0 or (
            self.last_frame_id is not None and capture_id <= self.last_frame_id
        ):
            raise RuntimeError(
                f"Camera counter did not advance: {self.last_frame_id} -> {capture_id}. "
                "Check counter setup; resetting IDs requires a fresh registry session."
            )
        if self.last_frame_id is not None and capture_id != self.last_frame_id + 1:
            self.logger.warning(
                "Camera counter gap: %d -> %d", self.last_frame_id, capture_id
            )
        return capture_id

    def _read_camera_value(self, name, default):
        node = self.cam.GetNodeMap().GetNode(name)
        return (
            node.GetValue()
            if node is not None and genicam.IsReadable(node)
            else default
        )

    def mission(self):
        if self.stopped.is_set():
            return True
        if self.cam is None:
            self.setup_cam()
        software_trigger = (
            self.cam.TriggerMode.Value == "On"
            and self.cam.TriggerSource.Value == "Software"
        )
        if software_trigger:
            if self.stopped.wait(max(0, self.next_frame_at - monotonic())):
                return True
            self.next_frame_at = monotonic() + self.capture_period
        try:
            with self.lease_slot_for_writing("tagged_image") as buffer:
                photo = PhotoExif.from_buffer(buffer)
                try:
                    self.cam.SetBufferFactory(LeasedBuffer(buffer))
                    self.cam.StartGrabbingMax(1, pylon.GrabStrategy_OneByOne)
                    if software_trigger:
                        self.cam.WaitForFrameTriggerReady(
                            1000, pylon.TimeoutHandling_ThrowException
                        )
                        self.cam.ExecuteSoftwareTrigger()
                    result = self.cam.RetrieveResult(500, pylon.TimeoutHandling_Return)
                    if result is None:
                        raise TimeoutError("Camera grab timed out")
                    with result:
                        if not result.GrabSucceeded():
                            raise TimeoutError(result.ErrorDescription)
                        capture_id = self._capture_id(result)
                        photo.ID = capture_id
                        photo.exposure_time, photo.gain = (
                            self.cam.ExposureTime.Value,
                            self.cam.Gain.Value,
                        )
                        photo.gamma, photo.black_level = (
                            self.cam.Gamma.Value,
                            self.cam.BlackLevel.Value,
                        )
                        photo.bsl_color_space = self._read_camera_value(
                            "BslColorSpace", ""
                        ).encode()
                        photo.light_source_preset = self._read_camera_value(
                            "BslLightSourcePreset", ""
                        ).encode()
                finally:
                    self.cam.StopGrabbing()
                    self.cam.SetBufferFactory(None)
                    del photo
        except TimeoutError as error:
            self.logger.warning("%s", error)
            return self.stopped.is_set()
        self.last_frame_id = capture_id
        if self.capture_connection is not None:
            self.capture_connection.send(capture_id)
        return self.stopped.is_set()

    def stop(self):
        self.stopped.set()

    def _dump_data(self):
        if self.cam is not None:
            self.cam.Close()
        if self.capture_connection is not None:
            self.capture_connection.close()
