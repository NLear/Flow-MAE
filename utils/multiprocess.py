import queue
from concurrent.futures import ProcessPoolExecutor


class BoundProcessPoolExecutor(ProcessPoolExecutor):
    """对ThreadPoolExecutor 进行重写，给队列设置边界 qsize
    """

    def __init__(self, qsize: int = 16, *args, **kwargs):
        super(BoundProcessPoolExecutor, self).__init__(*args, **kwargs)
        self.qsize = qsize
        self._work_queue = queue.Queue(maxsize=qsize)
