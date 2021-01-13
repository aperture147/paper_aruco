import queue
import threading


class BaseProcessor(threading.Thread):
    def __init__(self, src_queue):
        super().__init__()
        self.queue = queue.PriorityQueue()
        self.src_queue = src_queue

    def run(self) -> None:
        raise Exception("run function is not implemented yet")
