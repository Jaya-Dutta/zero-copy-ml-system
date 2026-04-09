import struct
import time
from multiprocessing import shared_memory
from filelock import FileLock
from .config import SHM_NAME, SHM_SIZE, LOCK_FILE

STATE_IDLE = 0
STATE_DATA_READY = 1
STATE_RESULT_READY = 2

class IPCManager:
    def __init__(self, create=False):
        self.create = create
        try:
            self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=create, size=SHM_SIZE)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=False, size=SHM_SIZE)
        if self.create:
            self._reset_state()
            
    def _reset_state(self):
        self.shm.buf[0] = STATE_IDLE
        self.shm.buf[1:5] = struct.pack("I", 0)

    def write_data(self, data_bytes: bytes):
        """API calls this to write data directly to shared memory, zero-copy for worker to read."""
        size = len(data_bytes)
        if size > SHM_SIZE - 5:
            raise MemoryError(f"Payload size {size} exceeds shared memory capacity {SHM_SIZE - 5} bytes.")
            
        try:
            self.shm.buf[1:5] = struct.pack("I", size)
            self.shm.buf[5:5+size] = data_bytes
            self.shm.buf[0] = STATE_DATA_READY
        except Exception as e:
            raise RuntimeError(f"Shared memory write failed: {e}")

    def wait_for_result(self):
        """API busy-waits for worker to finish (Ultra-low latency Spinlock)"""
        while self.shm.buf[0] != STATE_RESULT_READY:
            pass # Pure tight loop to avoid nanosecond scale Windows scheduler penalty
            
        size = struct.unpack("I", self.shm.buf[1:5])[0]
        # Copy out the payload since we'll immediately mark memory as Idle
        result_bytes = bytes(self.shm.buf[5:5+size])
        
        self.shm.buf[0] = STATE_IDLE
        return result_bytes

    def wait_for_data(self):
        """Worker busy-waits for API data."""
        while self.shm.buf[0] != STATE_DATA_READY:
            pass # Pure tight loop to avoid thread suspension latency
            
        size = struct.unpack("I", self.shm.buf[1:5])[0]
        # Returns a MEMORYVIEW directly reflecting shared memory for TRUE ZERO-COPY PyArrow reading
        return self.shm.buf[5:5+size]

    def write_result(self, result_bytes: bytes):
        """Worker serializes results natively to Arrow in shared memory."""
        size = len(result_bytes)
        if size > SHM_SIZE - 5:
            raise MemoryError("Result size too large for ShM buffer limit")

        self.shm.buf[1:5] = struct.pack("I", size)
        self.shm.buf[5:5+size] = result_bytes
        self.shm.buf[0] = STATE_RESULT_READY

    def cleanup(self):
        self.shm.close()
        if self.create:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
