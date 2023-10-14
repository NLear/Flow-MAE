# coding=utf-8
import functools
import time
from typing import Callable, Any

import math

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    t_int = int(t)
    h, m, s, ss = t_int // 3600, (t_int // 60) % 60, t_int % 60, math.modf(t)[0]
    return f"{h}:{m:02d}:{s:02d}:{ss:.4f}" if h != 0 else f"{m:02d}:{s:02d}:{ss:.4f}"


def display_run_time(func: Callable[..., Any]) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Callable:
        t1 = time.perf_counter()
        logger.info("start test")
        res = func(*args, **kwargs)
        t2 = time.perf_counter()
        logger.info("end test")
        logger.info(f"time elapsed: {format_time(t2 - t1)} s")
        return res

    return wrapper


def catch_exception(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Callable:
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            # pprint(e)
            # traceback.print_exc()
            # logger.error(traceback.format_exc())
        else:
            return res

    return wrapper
