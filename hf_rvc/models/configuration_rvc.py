from dataclasses import dataclass
from typing import Any, Dict

from transformers import HubertConfig, PretrainedConfig

from .vits.models import SynthesizerTrnMs256NSFsidConfig


@dataclass
class RVCConfig(PretrainedConfig):
    model_type = "rvc"
    is_composition = True

    def __init__(
        self,
        hubert=None,
        vits=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hubert = (
            hubert if isinstance(hubert, HubertConfig)
            else HubertConfig(**(hubert or {"vocab_size": 256}))
        )
        self.vits = (
            vits if isinstance(vits, SynthesizerTrnMs256NSFsidConfig)
            else SynthesizerTrnMs256NSFsidConfig(**(vits or {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        dict = super().to_dict()
        dict["hubert"] = self.hubert.to_dict()
        dict["vits"] = self.vits.to_dict()
        return dict
