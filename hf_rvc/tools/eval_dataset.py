import numpy as np
import pyaudio
import torch
from datasets import Audio, Dataset, load_dataset
from librosa.core import resample
from scipy.io.wavfile import write

from ..models import RVCFeatureExtractor, RVCModel


def eval_dataset(
    model_name: str,
    dataset_path: str = "mozilla-foundation/common_voice_12_0",
    dataset_name: str = "ja",
    dataset_split: str = "validation",
    shuffle: bool = True,
    shuffle_seed: int | None = None,
    num_data: int = 10,
    all_data: bool = False,
    output_device_index: int | None = None,
    output_file_path: str | None = None,
    f0_method: str = "pm",
    auto_m2f: bool = False,
    auto_f2m: bool = False,
    f0_up_key: float = 0.0,
    skip_original: bool = False,
    pad_frames: int = 8000,
):
    """Evaluate dataset.

    Args:
        model_name: Model name to be used for evaluation.
        dataset_path: Path to the dataset (default: "mozilla-foundation/common_voice_12_0").
        dataset_name: Dataset name (default: "ja").
        dataset_split: Dataset split, e.g. "train", "validation", or "test" (default: "validation").
        shuffle: Set to True to shuffle the dataset (default: True).
        shuffle_seed: Seed for dataset shuffle (default: None).
        num_data: Number of data samples to evaluate (default: 10).
        all_data: Set to True to evaluate all data samples (default: False).
        output_device_index: Index of the output device for audio playback (default: None).
        output_file_path: Path to save the output file (default: None).
        f0_method: F0 extraction method, "pm", or "harvest" (default: "pm").
        auto_m2f: Set to True to automatically convert male to female (default: False).
        auto_f2m: Set to True to automatically convert female to male (default: False).
        f0_up_key: F0 key up value (default: 0.0).
        skip_original: Set to True to skip the original audio (default: False).
        pad_frames: Number of frames to pad before concatenation (default: 8000).
    """

    def _get_auto_key(source_gender: str) -> float:
        if auto_m2f and source_gender == "male":
            return 12
        if auto_f2m and source_gender == "female":
            return -12
        return 0

    dataset = load_dataset(dataset_path, dataset_name, split=dataset_split)
    assert isinstance(dataset, Dataset)

    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)

    if not all_data:
        dataset = dataset.select(range(num_data))

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    pa = pyaudio.PyAudio()
    output_stream = pa.open(
        rate=48000,
        channels=1,
        format=pyaudio.paFloat32,
        output=True,
        output_device_index=None
        if output_device_index is None
        else int(output_device_index),
    )

    feature_extractor = RVCFeatureExtractor.from_pretrained(model_name)
    assert isinstance(feature_extractor, RVCFeatureExtractor)
    feature_extractor.set_f0_method(f0_method)

    model = RVCModel.from_pretrained(model_name)
    assert isinstance(model, RVCModel)

    with torch.no_grad():
        for n, data in enumerate(dataset, start=1):
            assert isinstance(data, dict)

            key = f0_up_key + _get_auto_key(data["gender"])
            print(data["sentence"], f"({data['gender'] or 'unknown'}/{key:+0.1f})")

            audio = data["audio"]

            input_features = feature_extractor(
                np.pad(audio["array"], (pad_frames, pad_frames), mode="reflect"),
                sampling_rate=audio["sampling_rate"],
                f0_up_key=key,
                return_tensors="pt",
            )

            output_pad_frames = pad_frames * audio["sampling_rate"] // 48000
            output = (
                model(**input_features)
                .numpy()
                .flatten()[output_pad_frames:-output_pad_frames]
            )

            resampled_input = resample(
                audio["array"], orig_sr=16000, target_sr=48000
            ).astype(np.float32)

            if not skip_original:
                output_stream.write(resampled_input.tobytes())
            output_stream.write(output.tobytes())

            if output_file_path is not None:
                write(f"{output_file_path}_{n:04}_orig.wav", 48000, resampled_input)
                write(f"{output_file_path}_{n:04}_conv.wav", 48000, output)

    pa.terminate()


if __name__ == "__main__":
    from argh import ArghParser

    parser = ArghParser()
    parser.set_default_command(eval_dataset)
    parser.dispatch()
