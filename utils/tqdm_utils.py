import contextlib
import sys
import time

from tqdm import trange, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class DummyFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


if __name__ == '__main__':

    def blabla():
        print("Foo blabla")


    # tqdm call to sys.stdout must be done BEFORE stdout redirection
    # and you need to specify sys.stdout, not sys.stderr (default)
    for i in tqdm(range(3), file=sys.stdout):
        with nostdout():
            blabla()
            time.sleep(.5)
            if i == 1:
                tqdm.write("tqdm write")
    print('Done!')

    with logging_redirect_tqdm([logger]):
        for i in trange(10, file=sys.stdout):
            with nostdout():
                time.sleep(0.4)
                if i == 4:
                    logger.warning("console logging redirected to `tqdm.write()`")
                    print("test")
