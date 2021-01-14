import multiprocessing
import threading


class BaseProcessor(threading.Thread):
    def __init__(self, receiver):
        super().__init__()
        self.receiver, self.sender = multiprocessing.Pipe(duplex=False)
        self._src_receiver = receiver

    def run(self) -> None:
        raise Exception("run function is not implemented yet")
