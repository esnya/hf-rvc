from os import PathLike
from typing import Any

import torch

from ..models.vits.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsidConfig,
)
from ..utils.path import remove_extension


def load_vits_checkpoint(vits_path: str | PathLike) -> dict[str, Any]:
    return torch.load(vits_path, map_location="cpu", weights_only=True)


def extract_vits_config(
    vits_checkpoint: dict[str, Any],
) -> SynthesizerTrnMs256NSFsidConfig:
    return SynthesizerTrnMs256NSFsidConfig(*vits_checkpoint["config"])


def extract_vits_state(vits_checkpoint: dict[str, Any]) -> dict[str, Any]:
    def _fix_key(key: str) -> str:
        return key.replace(".gamma", ".weight").replace(".beta", ".bias")

    return {_fix_key(key): value for key, value in vits_checkpoint["weight"].items()}


def convert_vits(
    vits_checkpoint: str | PathLike | dict[str, Any],
    save_directory: str | PathLike | None = None,
    safe_serialization=True,
) -> SynthesizerTrnMs256NSFsid:
    """Convert VITS model.

    Args:
        vits_checkpoint: Path to the original VITS checkpoint or a dictionary containing the checkpoint data.
        save_directory: Directory to save the converted VITS model (optional).
        safe_serialization: Set to False to disable safe serialization (default: True).
    """

    if not save_directory and not isinstance(vits_checkpoint, dict):
        save_directory = remove_extension(vits_checkpoint)

    if not isinstance(vits_checkpoint, dict):
        vits_checkpoint = load_vits_checkpoint(vits_checkpoint)

    vits_config = extract_vits_config(vits_checkpoint)
    vits = SynthesizerTrnMs256NSFsid(vits_config)

    vits.load_state_dict(
        extract_vits_state(vits_checkpoint),
        strict=False,
    )

    if save_directory:
        vits.save_pretrained(save_directory, safe_serialization=safe_serialization)

    return vits


if __name__ == "__main__":
    from argh import ArghParser

    parser = ArghParser()
    parser.set_default_command(convert_vits)
    parser.dispatch()
