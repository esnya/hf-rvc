import unittest

import numpy as np
from datasets import Audio, Dataset, load_dataset

from hf_rvc.models.feature_extraction_rvc import (
    extract_f0_harvest,
    extract_f0_pm,
    extract_f0_pyin,
    extract_f0_yin,
)


class TestFeatureExtractionRVC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampling_rate = 16000
        cls.window = 160
        cls.time_step = cls.window / cls.sampling_rate
        cls.f0_min = 50
        cls.f0_max = 1100
        cls.test_window = 40

        dataset = load_dataset("common_voice", "ja", split="validation[:10%]")
        assert isinstance(dataset, Dataset)
        cls.dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=cls.sampling_rate)
        )

        cls.audio = dataset[0]["audio"]["array"]
        cls.audio = cls.audio[cls.audio.size // 4 : -cls.audio.size // 4]
        cls.audio = cls.audio[: (cls.audio.size // cls.window) * cls.window]
        cls.f0 = extract_f0_harvest(
            cls.audio, cls.sampling_rate, cls.time_step, cls.f0_min, cls.f0_max
        )

    def assertF0Extracted(self, func):
        for data in self.dataset.select(range(4)):
            assert isinstance(data, dict)
            audio = data["audio"]
            # print(data["sentence"], audio["array"].size, sep=" \t")

            ref = extract_f0_harvest(
                audio["array"],
                audio["sampling_rate"],
                self.time_step,
                self.f0_min,
                self.f0_max,
            )
            ref[ref == 0] = np.nan
            ref_median = np.nanmedian(ref).item()

            f0 = func(
                audio["array"],
                audio["sampling_rate"],
                self.time_step,
                self.f0_min,
                self.f0_max,
            )
            f0[f0 == 0] = np.nan
            # print(
            #     audio["array"].size / f0.size,
            #     # np.nanmean(f0),
            #     np.nanmedian(f0),
            #     np.nanmin(f0),
            #     # np.nanmax(f0),
            #     sep=" \t",
            # )
            self.assertAlmostEqual(
                audio["array"].size / f0.size, self.window, delta=2.5
            )
            self.assertAlmostEqual(np.nanmedian(f0).item(), ref_median, delta=50)

    def test_extract_f0_pm(self):
        self.assertF0Extracted(extract_f0_pm)

    def test_extract_f0_harvest(self):
        self.assertF0Extracted(extract_f0_harvest)

    def test_extract_f0_yin(self):
        self.assertF0Extracted(extract_f0_yin)

    def test_extract_f0_pyin(self):
        self.assertF0Extracted(extract_f0_pyin)


if __name__ == "__main__":
    unittest.main()
