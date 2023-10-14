from pathlib import PosixPath
from typing import Any, Dict, List, Union, Type

JSON = Union[Dict[str, Any], List[Any], int, str, float, bool, Type[None]]
TypeJSON = Union[Dict[str, 'JSON'], List['JSON'], Type[None]]
PathLike = Union[str, PosixPath]
