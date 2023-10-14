# coding=utf-8
"""This is csv files container classes. """
import re
from pathlib import Path
from typing import Dict

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class GlobFiles:
    """
    It recursively finds all files in a directory that match a given pattern and
    are larger than a given threshold
    """
    def __init__(self, root: str, file_pattern: str, threshold: int = 0):
        self.root: Path = Path(root).resolve()
        self.file_pattern: str = file_pattern
        self.threshold = threshold
        self.files: Dict[str, Dict[str, str]] = {}

        assert self.root.is_dir(), f"root dir {self.root} not exist"
        self.__find_files_recursive(root=self.root)

    def __find_files_recursive(self, root: Path):
        for entry in root.iterdir():
            if entry.is_dir():
                self.__find_files_recursive(entry)
        files = filter(
            lambda x: x.stat().st_size > self.threshold,
            root.glob(self.file_pattern)
        )
        files = {re.sub(r"[^.\w]+", "_", f.stem): [f.as_posix()] for f in files}
        split = root.relative_to(self.root).as_posix()
        if files:
            self.files[split] = files
