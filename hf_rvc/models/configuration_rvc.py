from dataclasses import dataclass
from typing import Any

from transformers import HubertConfig, PretrainedConfig

from .vits.models import SynthesizerTrnMs256NSFsidConfig


@dataclass
class RVCConfig(PretrainedConfig):
    def __init__(
        self,
        hubert=HubertConfig(vocab_size=256),
        vits: SynthesizerTrnMs256NSFsidConfig = SynthesizerTrnMs256NSFsidConfig(),
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hubert = hubert
        self.vits = vits

    def to_dict(self) -> dict[str, Any]:
        dict = super().to_dict()
        dict["hubert"] = self.hubert.to_dict()
        dict["vits"] = self.vits.__dict__
        return dict
