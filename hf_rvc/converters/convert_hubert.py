from os import PathLike
from typing import Any

import torch
from transformers import HubertConfig, HubertForCTC

from ..utils.path import remove_extension


def load_fairseq_hubert(model_path: str, unsafe: bool = False):
    from fairseq import checkpoint_utils

    if not unsafe:
        raise ValueError(
            "Safe load failed. Re-running with `unsafe` set to `False`"
            " will likely succeed, but it can result in arbitrary code execution."
            "Do it only if you get the file from a trusted source. "
        )

    [fairseq_hubert], *_ = checkpoint_utils.load_model_ensemble_and_task(
        [model_path], suffix=""
    )
    fairseq_hubert = fairseq_hubert.float()
    fairseq_hubert.eval()

    return fairseq_hubert


def extract_hubert_config(fairseq_hubert) -> HubertConfig:
    assert isinstance(fairseq_hubert.final_proj, torch.nn.Linear)

    hf_hubert_config = HubertConfig(vocab_size=fairseq_hubert.final_proj.out_features)

    return hf_hubert_config


def extract_hubert_state(
    config_or_model: HubertConfig | HubertForCTC, fairseq_hubert
) -> dict[str, Any]:
    fairseq_state_dict = fairseq_hubert.state_dict()
    assert isinstance(fairseq_state_dict, dict)

    if isinstance(config_or_model, HubertForCTC):
        hf_hubert_model = config_or_model
    else:
        hf_hubert_model = HubertForCTC(config_or_model)
    hf_state_dict = hf_hubert_model.state_dict()

    MAPPING = {
        r"post_extract_proj": r"feature_projection.projection",
        r"encoder.pos_conv.0": r"encoder.pos_conv_embed.conv",
        r"encoder\.layers\.([0-9]+)\.self_attn.k_proj": r"encoder.layers.\1.attention.k_proj",
        r"encoder\.layers\.([0-9]+)\.self_attn.v_proj": r"encoder.layers.\1.attention.v_proj",
        r"encoder\.layers\.([0-9]+)\.self_attn.q_proj": r"encoder.layers.\1.attention.q_proj",
        r"encoder\.layers\.([0-9]+)\.self_attn.out_proj": r"encoder.layers.\1.attention.out_proj",
        r"encoder\.layers\.([0-9]+)\.self_attn_layer_norm": r"encoder.layers.\1.layer_norm",
        r"encoder\.layers\.([0-9]+)\.fc1": r"encoder.layers.\1.feed_forward.intermediate_dense",
        r"encoder\.layers\.([0-9]+)\.fc2": r"encoder.layers.\1.feed_forward.output_dense",
        r"encoder\.layers\.([0-9]+)\.final_layer_norm": r"encoder.layers.\1.final_layer_norm",
        r"encoder.layer_norm": r"encoder.layer_norm",
        r"w2v_model.layer_norm": r"feature_projection.layer_norm",
        r"w2v_encoder.proj": r"lm_head",
        r"mask_emb": r"masked_spec_embed",
        r"final_proj\.": r"lm_head.",
        r"layer_norm\.": r"feature_projection.layer_norm.",
        r"feature_extractor\.conv_layers\.([0-9]+)\.0\.": r"feature_extractor.conv_layers.\1.conv.",
        r"feature_extractor\.conv_layers\.0\.2\.": r"feature_extractor.conv_layers.0.layer_norm.",
    }

    required_keys = set(hf_state_dict.keys())

    def _convert_key(key):
        import re

        if key in hf_state_dict:
            return key
        if f"hubert.{key}" in hf_state_dict:
            return f"hubert.{key}"

        for pattern, repl in MAPPING.items():
            replaced = re.sub(pattern, repl, key)
            if replaced in hf_state_dict:
                return replaced
            if f"hubert.{replaced}" in hf_state_dict:
                return f"hubert.{replaced}"

        print("ERROR: Failed to map key", key)
        return key

    remove_keys = set(["label_embs_concat"])

    converted_dict = {
        _convert_key(key): value
        for key, value in fairseq_state_dict.items()
        if key not in remove_keys
    }
    for key in converted_dict.keys():
        if key in required_keys:
            required_keys.remove(key)
    for key in required_keys:
        print("ERROR: Failed to load key", key)

    return converted_dict


def convert_hubert(
    fairseq_hubert: Any | str | PathLike = "./models/hubert_base.pt",
    save_directory: str | PathLike | None = None,
    safe_serialization=True,
    unsafe: bool = False,
) -> HubertForCTC:
    """Convert Hubert model.

    Args:
        fairseq_hubert: Path to the original fairseq Hubert model or an instance of the HubertModel class.
        save_directory: Directory to save the converted Hubert model (optional).
        safe_serialization: Set to False to disable safe serialization (default: True).
        unsafe: Set to True to load untrusted models (default: False).
    """

    from fairseq.models.hubert.hubert import HubertModel

    if not isinstance(fairseq_hubert, HubertModel):
        if save_directory is None:
            save_directory = remove_extension(str(fairseq_hubert))

        fairseq_hubert = load_fairseq_hubert(str(fairseq_hubert), unsafe=unsafe)
    assert isinstance(fairseq_hubert, HubertModel)
    assert isinstance(fairseq_hubert.final_proj, torch.nn.Linear)

    hf_hubert_config = extract_hubert_config(fairseq_hubert)
    hf_hubert = HubertForCTC(hf_hubert_config)

    hf_hubert.load_state_dict(extract_hubert_state(hf_hubert, fairseq_hubert))

    if save_directory:
        hf_hubert.save_pretrained(save_directory, safe_serialization=safe_serialization)

    return hf_hubert


if __name__ == "__main__":
    from argh import ArghParser

    parser = ArghParser()
    parser.set_default_command(convert_hubert)
    parser.dispatch()
