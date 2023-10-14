# coding=utf-8
""" Logging utilities."""
import logging
import logging.config
import threading
from typing import Optional, List

LOGGER_CONFIG = {
    'version': 1,
    'incremental': False,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s >> %(message)s'
        },
        'detailed': {
            'format': '[%(asctime)s-%(levelname)s-%(filename)s(%(lineno)d)] >> %(message)s',
            'datefmt': '%Y/%m/%d %H:%M',
            'style': '%'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'level': 'DEBUG',
            'formatter': 'detailed'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logging.log',
            'level': 'DEBUG',
            'formatter': 'detailed'
        },
    },
    'loggers': {
        '': {  # folder logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        '__main__': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
    }
}

_lock = threading.Lock()

_is_initialized: bool = False
_root_level: Optional[int] = logging.INFO
_root_handlers: Optional[List[logging.Formatter]] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.
    """

    global _is_initialized, _root_level, _root_handlers

    with _lock:
        if not _is_initialized:
            logging.config.dictConfig(LOGGER_CONFIG)
            _root_level = logging.root.level
            _root_handlers = logging.root.handlers
            _is_initialized = True

        logger = logging.getLogger(name)
        logger.propagate = False
        if not logger.hasHandlers():
            logger.setLevel(_root_level)
            for hdl in _root_handlers:
                logger.addHandler(hdl)
        return logger


if __name__ == '__main__':
    logger = get_logger("test")
    logger.info("console logging redirected to `tqdm.write()`")
