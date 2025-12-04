import threading
from contextlib import contextmanager

from typing import Generic, TypeVar
T = TypeVar('T')

class RWResource(Generic[T]):
    def __init__(self, initial: T):
        self._data = initial

        # Reader/writer state
        self._reader_count = 0
        self._writer_waiting = 0
        self._writing = False

        # Internal locks/conditions
        self._lock = threading.Lock()
        self._ok_to_read = threading.Condition(self._lock)
        self._ok_to_write = threading.Condition(self._lock)

    # -------------------------
    #        READ API
    # -------------------------
    @contextmanager
    def read(self):
        with self._lock:
            # Writer-priority rule: if a writer is waiting or writing, block readers
            while self._writer_waiting > 0 or self._writing:
                self._ok_to_read.wait()

            self._reader_count += 1

        try:
            yield self._data
        finally:
            with self._lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    # Last reader wakes writers if they are waiting
                    self._ok_to_write.notify()
    

    # -------------------------
    #        WRITE API
    # -------------------------
    @contextmanager
    def write(self):
        with self._lock:
            self._writer_waiting += 1

            while self._reader_count > 0 or self._writing:
                self._ok_to_write.wait()

            self._writer_waiting -= 1
            self._writing = True

        # Expose a proxy so user can modify the resource cleanly
        class WriterProxy:
            def __init__(self, value):
                self.value = value

        proxy = WriterProxy(self._data)

        try:
            yield proxy
        finally:
            with self._lock:
                # Commit the updated value
                self._data = proxy.value
                self._writing = False

                # Writers are first in priority  
                self._ok_to_write.notify()

                # If no writers waiting, allow readers
                if self._writer_waiting == 0:
                    self._ok_to_read.notify_all()
