import logging
from functools import lru_cache
from time import time
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import torch
from pyparsing import Opt

from ..models.feature_extraction_rvc import F0_EXTRACTORS, RVCFeatureExtractor
from ..models.modeling_rvc import RVCModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(
    model_path: str, device: str, fp16: bool, better_tranformer: bool
) -> Tuple[RVCFeatureExtractor, RVCModel, Dict[str, Dict[str, Any]]]:
    logger.info("Loading model from %s", model_path)
    model = RVCModel.from_pretrained(model_path)
    assert isinstance(model, RVCModel)
    model = model.to(device, dtype=torch.float16 if fp16 else torch.float32)
    if better_tranformer:
        model = cast(RVCModel, model.to_bettertransformer())
    logger.info(
        "Model loaded: %s(%s, device=%s, dtype=%s, training=%s)",
        model.__class__.__name__,
        model.name_or_path,
        model.device,
        model.dtype,
        model.training,
    )

    logger.info("Loading feature extractor from %s", model_path)
    feature_extractor = RVCFeatureExtractor.from_pretrained(model_path)
    logger.info(
        "Feature extractor loaded: %s(sampling_rate=%s)",
        feature_extractor.__class__.__name__,
        feature_extractor.sampling_rate,
    )

    return (
        feature_extractor,
        model.eval(),
        dict(
            model=dict(
                class_name=model.__class__.__name__,
                name_or_path=model.name_or_path,
                device=str(model.device),
                dtype=str(model.dtype),
                training=model.training,
            ),
            feature_extractor=dict(
                class_name=feature_extractor.__class__.__name__,
                sampling_rate=feature_extractor.sampling_rate,
            ),
        ),
    )


Audio = Tuple[int, np.ndarray]


@torch.inference_mode()
def _vc_offline(
    model: Optional[RVCModel],
    feature_extractor: Optional[RVCFeatureExtractor],
    f0_method: Optional[str],
    f0_up_key: Optional[float],
    inputs: Optional[Audio],
) -> Optional[Audio]:
    import librosa

    if model is None or feature_extractor is None or inputs is None:
        return None

    input_sr, input_audio = inputs

    input_audio = input_audio.astype(np.float32).mean(-1) / 32768
    input_audio = librosa.resample(input_audio, orig_sr=input_sr, target_sr=16000)

    if f0_method and feature_extractor.f0_method != f0_method:
        feature_extractor.set_f0_method(f0_method)

    if f0_up_key is None:
        f0_up_key = 0

    model_inputs = feature_extractor(
        input_audio,
        sampling_rate=16000,
        f0_up_key=f0_up_key,
        return_tensors="pt",
    )

    model_output = (
        model(**model_inputs.to(model.device, model.dtype))
        .cpu()
        .float()
        .flatten()
        .numpy()
    )

    model_output = model_output * 32768
    model_output = model_output.astype(np.int16)

    return (48000, model_output.reshape((-1, 1)))


@torch.inference_mode()
def _vc(
    model: Optional[RVCModel],
    feature_extractor: Optional[RVCFeatureExtractor],
    f0_method: Optional[str],
    f0_up_key: Optional[float],
    latency: Optional[float],
    padding_seconds: Optional[float],
    min_volume: Optional[float],
    inputs: Optional[Audio],
    buffer: Optional[np.ndarray],
    output: Optional[Audio],
) -> Tuple[Optional[Audio], Optional[np.ndarray]]:
    import librosa

    start_time = time()

    if model is None or feature_extractor is None or inputs is None:
        return output, buffer

    input_sr, input_audio = inputs

    input_audio = input_audio.astype(np.float32).mean(-1) / 32768
    input_audio = librosa.resample(input_audio, orig_sr=input_sr, target_sr=16000)

    if buffer is None:
        buffer = input_audio
    else:
        buffer = np.concatenate([buffer, input_audio])

    if not latency:
        latency = 0.5

    if not padding_seconds:
        padding_seconds = 0
    padding_frames = int(padding_seconds * 16000)
    buffering_frames = int(latency * 16000) + padding_frames

    buffer = buffer[-buffering_frames:]

    logger.info("Buffered %ss", buffer.size / 16000)

    if buffer.size < buffering_frames or min_volume and buffer.max() < min_volume:
        return output, buffer

    if padding_frames > 0:
        padded = np.pad(buffer, (0, padding_frames), "reflect")
    else:
        padded = buffer

    logger.info("Padded %ss", padded.size / 16000)

    if f0_method and feature_extractor.f0_method != f0_method:
        feature_extractor.set_f0_method(f0_method)

    if f0_up_key is None:
        f0_up_key = 0

    model_inputs = feature_extractor(
        padded,
        sampling_rate=16000,
        f0_up_key=f0_up_key,
        return_tensors="pt",
    )

    output_padding_frames = int(padding_seconds * 48000)
    output_frames = int(latency * 48000)

    model_output = (
        model(**model_inputs.to(model.device, model.dtype))
        .cpu()
        .float()
        .flatten()
        .numpy()
    )

    logger.info("Raw Output %ss", model_output.size / 48000)

    model_output = model_output[:-output_padding_frames][-output_frames:]
    model_output = model_output * 32768
    model_output = model_output.astype(np.int16)

    logger.info("Unpadded Output %ss", model_output.size / 48000)

    elapsed = time() - start_time
    logger.info("Process Time: %ss (x%s)", elapsed, latency / elapsed)

    return (48000, model_output.reshape((-1, 1))), buffer[-padding_frames:]


def gradio_vc(server_name: str = "localhost", server_port: int = 7860, **kwargs):
    import gradio as gr

    with gr.Blocks() as app:
        with gr.Column():
            model_info = gr.Json(label="Loaded Model Info")
            model = gr.State()
            feature_extractor = gr.State()

            path_to_model = gr.Textbox(label="Path to Model")
            device = gr.Dropdown(
                label="Device",
                choices=["cpu", "cuda"]
                + [f"cuda:{n}" for n in range(torch.cuda.device_count())],
                value="cuda" if torch.cuda.is_available() else "cpu",
            )
            fp16 = gr.Checkbox(
                label="FP16",
                value=torch.cuda.is_available(),
            )

            better_transformer = gr.Checkbox(
                label="Better Transformer",
                value=False,
            )

            load_button = gr.Button(value="Load Model", variant="primary")

            load_button.click(
                _load_model,
                inputs=[path_to_model, device, fp16, better_transformer],
                outputs=[feature_extractor, model, model_info],
            )

        with gr.Column():
            f0_method = gr.Dropdown(
                label="F0 Method", choices=list(F0_EXTRACTORS.keys()), value="pm"
            )
            f0_up_key = gr.Number(label="F0 Upkey", value=0)

        with gr.Tab("Streaming"):
            latency = gr.Number(label="Latency", value=0.5)
            padding_seconds = gr.Number(label="Padding Seconds", value=0.1)
            min_volume = gr.Number(label="Min Volume", value=0.01)

            audio_streaming_input = gr.Audio(
                label="Audio Input",
                source="microphone",
                type="numpy",
                streaming=True,
            )

            audio_buffer = gr.State()

        with gr.Tab("Record"):
            audio_recorded_input = gr.Audio(
                label="Audio Input",
                source="microphone",
                type="numpy",
                streaming=False,
            )

        with gr.Tab("File"):
            audio_file_input = gr.Audio(
                label="Audio Input",
                source="upload",
                type="numpy",
                streaming=False,
            )

        with gr.Column():
            audio_output = gr.Audio(
                label="Audio Output", interactive=False, autoplay=True
            )

        audio_recorded_input.change(
            _vc_offline,
            [
                model,
                feature_extractor,
                f0_method,
                f0_up_key,
                audio_recorded_input,
            ],
            [audio_output],
        )

        audio_file_input.change(
            _vc_offline,
            [
                model,
                feature_extractor,
                f0_method,
                f0_up_key,
                audio_file_input,
            ],
            [audio_output],
        )

        audio_streaming_input.change(
            _vc,
            [
                model,
                feature_extractor,
                f0_method,
                f0_up_key,
                latency,
                padding_seconds,
                min_volume,
                audio_streaming_input,
                audio_buffer,
                audio_output,
            ],
            [audio_output, audio_buffer],
        )

    app.launch(server_name=server_name, server_port=server_port, **kwargs)
