from typing import Any, Union

import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils.generic import TensorType


def extract_f0_pm(
    audio: np.ndarray,
    sampling_rate: int,
    time_step: float,
    f0_min: float,
    f0_max: float,
) -> np.ndarray:
    import parselmouth

    sound = parselmouth.Sound(audio, sampling_rate)
    res = sound.to_pitch_ac(
        time_step=time_step,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    )
    return res.selected_array["frequency"]


def extract_f0_harvest(
    audio: np.ndarray,
    sampling_rate: int,
    time_step: float,
    f0_min: float,
    f0_max: float,
) -> np.ndarray:
    import pyworld
    import scipy.signal as signal

    f0, t = pyworld.harvest(  # type: ignore
        audio.astype(np.double),
        fs=sampling_rate,
        f0_ceil=f0_max,
        f0_floor=f0_min,
        frame_period=10,
    )
    f0 = pyworld.stonemask(audio.astype(np.double), f0, t, sampling_rate)  # type: ignore
    f0 = signal.medfilt(f0, 3)
    return f0


def extract_f0_yin(
    audio: np.ndarray,
    sampling_rate: int,
    time_step: float,
    f0_min: float,
    f0_max: float,
):
    import librosa

    f0 = librosa.yin(
        y=audio,
        sr=sampling_rate,
        fmin=f0_min,
        fmax=f0_max,
        hop_length=int(time_step * sampling_rate),
    )
    f0[f0 >= f0_max] = 0
    return f0


def extract_f0_pyin(
    audio: np.ndarray,
    sampling_rate: int,
    time_step: float,
    f0_min: float,
    f0_max: float,
):
    import librosa

    f0, *_ = librosa.pyin(
        y=audio,
        sr=sampling_rate,
        fmin=f0_min,
        fmax=f0_max,
        hop_length=int(time_step * sampling_rate),
        fill_na=0,
    )
    return f0


F0_EXTRACTORS = {
    "pm": extract_f0_pm,
    "harvest": extract_f0_harvest,
    "yin": extract_f0_yin,
    "pyin": extract_f0_pyin,
}


class RVCFeatureExtractor(Wav2Vec2FeatureExtractor):
    model_input_names = ["input_values", "f0_coarse", "f0"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=False,
        do_normalize=True,
        f0_method="pm",
        window=160,
        f0_min=50,
        f0_max=1100,
        *args,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            do_normalize=do_normalize,
            *args,
            **kwargs,
        )

        self.window = window
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.time_step = window / sampling_rate
        self.f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        self.f0_method = f0_method
        self._f0_extractor = F0_EXTRACTORS.get(f0_method)

    def __call__(
        self,
        audio: np.ndarray,
        sampling_rate: Union[int, None] = None,
        f0_up_key: float = 0,
        return_tensors: Union[str, TensorType, None] = None,
    ) -> BatchFeature:
        input_values = Wav2Vec2FeatureExtractor.__call__(
            self, audio, sampling_rate=sampling_rate
        ).input_values

        f0_coarse, f0 = self._extract_f0_features(audio, f0_up_key=f0_up_key)

        features = BatchFeature(
            {
                "input_values": input_values,
                "f0_coarse": f0_coarse,
                "f0": f0,
            }
        )

        if return_tensors is not None:
            features = features.convert_to_tensors(return_tensors)

        return features

    def set_f0_method(self, f0_method: str):
        """
        Set the f0 extraction method.

        The available methods are:
            pm: Praat's Pitch, A reliable and accurate F0 estimation method, suitable for handling noise but with slower processing speed.
            harvest: A high-accuracy F0 estimation method with relatively fast processing speed, suitable for handling a wide range of F0 frequencies.
            yin: A well-balanced F0 estimation method in terms of processing speed and accuracy, offering some noise tolerance.
            pyin: PYIN: A probabilistic YIN-based F0 estimation method with good accuracy and noise tolerance, but slower processing speed.
        """
        self.f0_method = f0_method
        self._f0_extractor = F0_EXTRACTORS.get(f0_method)

    def _extract_f0_features(
        self, audio: np.ndarray, f0_up_key: float = 0, p_len: Union[int, None] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if p_len is None:
            p_len = audio.shape[-1] // self.window

        if self._f0_extractor is None:
            raise ValueError(f"Invalid f0_method: {self.f0_method}")
        f0 = self._f0_extractor(
            audio, self.sampling_rate, self.time_step, self.f0_min, self.f0_max
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")  # type: ignore

        f0 = f0.astype(np.float64)

        f0 *= pow(2, f0_up_key / 12)

        f0bak = f0.copy().astype(np.float32)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)

        if f0bak.ndim == 1:
            f0bak = f0bak.reshape((1, -1))

        if f0_coarse.ndim == 1:
            f0_coarse = f0_coarse.reshape((1, -1))

        return f0_coarse, f0bak

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this feature extractor instance.
        """
        output = super().to_dict()

        output.pop("_f0_extractor", None)

        return output
