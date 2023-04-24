import torch
from transformers import HubertConfig, HubertForCTC, PreTrainedModel

from .configuration_rvc import RVCConfig
from .vits.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsidConfig


class RVCModel(PreTrainedModel):
    config_class = HubertConfig
    is_composition = True

    def __init__(
        self,
        config: RVCConfig,
    ):
        super().__init__(config)
        self.hubert = HubertForCTC(
            config.hubert
            if isinstance(config.hubert, HubertConfig)
            else HubertConfig(**config.hubert)
        )
        self.add_module("hubert", self.hubert)
        self.vits = SynthesizerTrnMs256NSFsid(
            config.vits
            if isinstance(config.vits, SynthesizerTrnMs256NSFsidConfig)
            else SynthesizerTrnMs256NSFsidConfig(**config.vits)
        )
        self.add_module("vits", self.vits)

        self.sid = torch.tensor([0], dtype=torch.long)

        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        f0_coarse: torch.Tensor,
        f0: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.hubert(input_values).logits.repeat_interleave(2, dim=1)
        phone_lengths = logits.shape[-2]
        output, *_ = self.vits.infer(
            logits,
            torch.tensor([phone_lengths]),
            f0_coarse[:, :phone_lengths],
            f0[:, :phone_lengths],
            self.sid,
        )

        return output

    def to(self, *args, **kwargs) -> "RVCModel":
        model = super().to(*args, **kwargs)
        model.sid = self.sid.to(*args, **kwargs)
        return model
