import ctypes as ct

from pypylon import pylon

from common.new_structs import PhotoExif


class LeasedBuffer(pylon.BufferFactory):
    """Let pylon write pixels and chunks directly into one registry lease."""

    def __init__(self, buffer):
        self.view = buffer[PhotoExif.image.offset :]
        self.allocated = False

    def AllocateBuffer(self, size):
        if self.allocated or size > len(self.view):
            raise ValueError("Camera payload does not fit the single leased buffer")
        self.allocated = True
        return ct.addressof(ct.c_uint8.from_buffer(self.view))

    def FreeBuffer(self, buffer, context):
        pass  # The registry owns this memory.

    def OnReleased(self):
        self.view.release()
