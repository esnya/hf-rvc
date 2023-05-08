from os import PathLike
from pathlib import Path
from typing import Union


def remove_extension(file_path: Union[str, PathLike]) -> str:
    p = Path(file_path)
    return str(p.parent / p.stem)
