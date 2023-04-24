import math
import multiprocessing
import time
from os import PathLike
from typing import ContextManager, Literal

import numpy as np
import pyaudio
import torch

from ..models import RVCFeatureExtractor, RVCModel


class PerfCounter(ContextManager):
    def __init__(self, name: str | None = None):
        self.name = name

    def __enter__(self) -> "PerfCounter":
        self._start_time = time.time()
        return super().__enter__()

    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self._start_time
        if self.name:
            print(f"{self.name}:\t{self.elapsed:0.3f}s")


def get_volumes(audio_input: np.ndarray, window_size: int) -> np.ndarray:
    output_length = len(audio_input) // window_size
    x = audio_input[: output_length * window_size]
    x = np.reshape(x, (output_length, window_size))
    x = np.max(x, axis=1)
    return x


def output_process_target(
    queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,  # type: ignore
    sampling_rate: int,
    output_device_index: int | None = None,
) -> None:
    pa = pyaudio.PyAudio()

    output_stream = pa.open(
        rate=sampling_rate,
        channels=1,
        format=pyaudio.paFloat32,
        output_device_index=output_device_index,
        output=True,
    )

    try:
        while output_stream.is_active() and not stop_event.is_set():
            # with PerfCounter("output"):
            output = queue.get()
            output_stream.write(output)
    finally:
        stop_event.set()
        output_stream.close()
        pa.terminate()


def unroll_mean(x, y, window):
    return np.repeat(x, window).reshape(-1, window).mean(axis=1)[: y.shape[0] * window]


def realtime_vc(
    model: str | PathLike | RVCModel,
    feature_extractor: str | PathLike | RVCFeatureExtractor | None = None,
    buffering_seconds: float = 0.5,
    auto_latency: bool = False,
    min_buffering_seconds: float = 0.1,
    max_buffering_seconds: float = 1.0,
    padding_seconds: float = 0.1,
    f0_up_key: float = 0,
    f0_method: Literal["pm", "harvest"] = "pm",
    input_device_index: int | None = None,
    output_device_index: int | None = None,
    output_sampling_rate: int = 48000,
    volume: float = 1.0,
    device: str | torch.device = "cpu",
    fp16: bool = False,
) -> None:
    """
    Real-time voice conversion.

    Args:
        model (str | PathLike | RVCModel): Path to the model or the model itself.
        feature_extractor (str | PathLike | RVCFeatureExtractor | None, optional):
            Path to the feature extractor or the feature extractor itself.
            Defaults to None.
        buffering_seconds (float, optional): Buffering seconds. Defaults to 0.1.
        auto_latency (bool, optional): Automatically adjust latency. Defaults to False.
        min_buffering_seconds (float | None, optional):
            Minimum buffering seconds. Defaults to 0.1. used with auto_latency.
        max_buffering_seconds (float | None, optional):
            Maximum buffering seconds. Defaults to 1.0. used with auto_latency.
        padding_seconds (float, optional): Padding seconds. Defaults to 0.1.
        f0_up_key (float, optional): F0 up key. Defaults to 0.
        f0_method (Literal["pm", "harvest"], optional): F0 method. Defaults to "pm".
        input_device_index (int | None, optional): Input device index. Defaults to None.
        output_device_index (int | None, optional): Output device index. Defaults to None.
        volume (float, optional): Volume. Defaults to 1.0.
        device (str | torch.device, optional): Device. Defaults to "cpu". "cuda" is recommended.
        fp16 (bool, optional): Use fp16. Defaults to False.
    """

    if not isinstance(feature_extractor, RVCFeatureExtractor):
        if feature_extractor is None:
            if isinstance(model, RVCModel):
                raise ValueError(
                    "If model is an RVCModel, feature_extractor must be specified."
                )
            feature_extractor = model
        feature_extractor = RVCFeatureExtractor.from_pretrained(feature_extractor)
    if not isinstance(model, RVCModel):
        model = RVCModel.from_pretrained(model)  # type: ignore
    assert isinstance(feature_extractor, RVCFeatureExtractor)
    assert isinstance(model, RVCModel)

    model = model.to(device)
    if fp16:
        model = model.half()

    feature_extractor.set_f0_method(f0_method)

    input_sampling_rate = feature_extractor.sampling_rate
    f0_window = feature_extractor.window

    f0_seconds = f0_window / input_sampling_rate
    f0_sampling_rate = input_sampling_rate / f0_window
    if min_buffering_seconds is None:
        min_buffering_seconds = buffering_seconds
    min_buffering_frames = int(f0_sampling_rate * min_buffering_seconds) * f0_window
    max_buffering_frames = (
        int(f0_sampling_rate * max_buffering_seconds) * f0_window
        if max_buffering_seconds
        else min_buffering_frames
    )

    buffering_frames = min_buffering_frames

    input_padding_frames = (
        int(input_sampling_rate * padding_seconds / f0_window) * f0_window
    )

    pa = pyaudio.PyAudio()
    input_stream = pa.open(
        rate=input_sampling_rate,
        channels=1,
        format=pyaudio.paFloat32,
        input=True,
        input_device_index=int(input_device_index)
        if input_device_index is not None
        else None,
        frames_per_buffer=max_buffering_frames,
    )

    output_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    output_process = multiprocessing.Process(
        target=output_process_target,
        args=(
            output_queue,
            stop_event,
            output_sampling_rate,
            output_device_index,
        ),
    )
    output_process.start()

    input_audio_buffer = np.zeros((input_padding_frames,), dtype=np.float32)

    try:
        with torch.no_grad():
            while input_stream.is_active() and not stop_event.is_set():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                input_audio = np.frombuffer(
                    input_stream.read(buffering_frames),
                    dtype=np.float32,
                )

                with PerfCounter() as pc:
                    input_volumes = get_volumes(input_audio, f0_window)
                    input_audio_buffer = np.concatenate(
                        (input_audio_buffer, input_audio)
                    )[-(input_padding_frames + buffering_frames) :]
                    input_frames = input_audio.size

                    features = feature_extractor(
                        np.pad(
                            input_audio_buffer, (0, input_padding_frames), "reflect"
                        ),
                        sampling_rate=16000,
                        f0_up_key=f0_up_key,
                        return_tensors="pt",
                    )
                    if fp16:
                        features = features.to(torch.float16)
                    features = features.to(device)

                    output_frames = (
                        input_frames * output_sampling_rate // input_sampling_rate
                    )
                    output_padding_frames = (
                        input_padding_frames
                        * output_sampling_rate
                        // input_sampling_rate
                    )
                    output = (
                        model(**features)
                        .cpu()
                        .flatten()
                        .numpy()[
                            -int(output_frames + output_padding_frames) : -int(
                                output_padding_frames
                            )
                        ]
                        * np.repeat(input_volumes, output_frames // input_volumes.size)
                        * volume
                    )

                    output_queue.put_nowait(output.tobytes())

                if auto_latency:
                    buffering_seconds = buffering_frames / input_sampling_rate
                    if pc.elapsed >= buffering_seconds:
                        buffering_frames = min(
                            math.ceil(
                                (pc.elapsed + f0_window / input_sampling_rate)
                                * input_sampling_rate
                                / f0_window
                                + 1
                            )
                            * f0_window,
                            max_buffering_frames,
                        )
                    elif pc.elapsed + 2 * f0_seconds < buffering_seconds:
                        buffering_frames = max(
                            buffering_frames - f0_window,
                            min_buffering_frames,
                        )
                    print(
                        f"latency: {buffering_frames / input_sampling_rate * 1000:4.0f}ms",
                    )
    finally:
        stop_event.set()
        input_stream.close()
        pa.terminate()
        output_process.join()


if __name__ == "__main__":
    from argh import ArghParser

    parser = ArghParser()
    parser.set_default_command(realtime_vc)
    parser.dispatch()
