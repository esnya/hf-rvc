from os import PathLike
from pathlib import Path


def remove_extension(file_path: str | PathLike) -> str:
    p = Path(file_path)
    return str(p.parent / p.stem)
