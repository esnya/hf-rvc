"""Optional UI contracts with local media and no model or server startup."""
from importlib import import_module, metadata, util
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import numpy as np
import torch
from transformers import BatchFeature

if util.find_spec("gradio") is not None:
    import gradio as gr
else:
    gr = None


@unittest.skipUnless(gr is not None, "Install the local recipe and .[gradio] UI extra")
class TestGradioCompatibility(unittest.TestCase):
    def test_maintenance_distribution_and_image_codecs(self):
        from gradio.processing_utils import encode_pil_to_bytes
        from PIL import Image, ImageSequence

        self.assertEqual(metadata.version("gradio"), "6.16.0+esnya.pillow13.1")
        self.assertEqual(metadata.version("pillow"), "13.0.0")
        for mode, color in [("RGB", (10, 20, 30)), ("RGBA", (10, 20, 30, 40)), ("L", 90), ("I;16", 65000)]:
            image = Image.new(mode, (6, 4), color)
            image.info["fixture"] = "metadata survives"
            with self.subTest(mode=mode), Image.open(BytesIO(encode_pil_to_bytes(image, "PNG"))) as decoded:
                decoded.load()
                self.assertEqual(decoded.tobytes(), image.tobytes())
                self.assertEqual(decoded.info["fixture"], image.info["fixture"])
        values = [-2147483648, -1, 0, 1, 255, 256, 65535, 65536, 70000, 2147483647]
        integer = Image.new("I", (len(values), 1))
        integer.putdata(values)
        integer.info["fixture"] = "integer metadata"
        with Image.open(BytesIO(encode_pil_to_bytes(integer, "PNG"))) as decoded:
            self.assertEqual(list(decoded.getdata()), [min(65535, max(0, value)) for value in values])
            self.assertEqual(decoded.info["fixture"], "integer metadata")
        for codec in ["jpeg", "tiff"]:
            with self.subTest(codec=codec), Image.open(BytesIO(encode_pil_to_bytes(Image.new("RGB", (6, 4), "red"), codec))) as decoded:
                decoded.load()
                self.assertEqual(decoded.size, (6, 4))
        stream = BytesIO()
        Image.new("RGB", (6, 4), "red").save(stream, "GIF", save_all=True, append_images=[Image.new("RGB", (6, 4), "blue")])
        stream.seek(0)
        with Image.open(stream) as image:
            encoded = encode_pil_to_bytes(image, "gif")
        with Image.open(BytesIO(encoded)) as decoded:
            self.assertEqual([frame.convert("RGB").getpixel((0, 0)) for frame in ImageSequence.Iterator(decoded)], [(255, 0, 0), (0, 0, 255)])

    def test_actual_component_tree_without_launch_or_models(self):
        ui = import_module("hf_rvc.tools.gradio_vc")
        captured = []
        def capture_launch(blocks, **kwargs):
            captured.append(blocks)
        with patch.object(gr.Blocks, "launch", capture_launch), \
             patch.object(ui.RVCModel, "from_pretrained") as model_loader, \
             patch.object(ui.RVCFeatureExtractor, "from_pretrained") as feature_loader, \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda, "device_count", return_value=0):
            ui.gradio_vc()
        model_loader.assert_not_called()
        feature_loader.assert_not_called()
        self.assertEqual(len(captured), 1)
        audio = [component for component in captured[0].config["components"] if component["type"] == "audio"]
        self.assertEqual(len(audio), 4)
        self.assertEqual(sum(component["props"].get("streaming", False) for component in audio), 1)
        self.assertEqual(sum(component["props"].get("sources") == ["microphone"] for component in audio), 2)

    def test_actual_wav_component_input_and_output(self):
        from gradio.data_classes import FileData
        values = np.stack([np.arange(160, dtype=np.int16), -np.arange(160, dtype=np.int16)], axis=1)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(values.tobytes())
            component = gr.Audio(sources=["upload"], type="numpy", format="wav")
            component.GRADIO_CACHE = folder
            rate, data = component.preprocess(FileData(path=str(path)))
            self.assertEqual(rate, 8000)
            np.testing.assert_array_equal(data, values)
            result = component.postprocess((rate, data))
            with wave.open(result.path, "rb") as encoded:
                self.assertEqual(encoded.getframerate(), rate)
                self.assertEqual(encoded.getnchannels(), 2)
                self.assertEqual(encoded.readframes(encoded.getnframes()), values.tobytes())

    def test_actual_audio_callbacks_with_generated_tensor_output(self):
        ui = import_module("hf_rvc.tools.gradio_vc")
        class Features:
            f0_method = "pm"
            def set_f0_method(self, value):
                self.f0_method = value
            def __call__(self, audio, **kwargs):
                self.last_kwargs = kwargs
                self.last_audio = audio
                return BatchFeature({"input_values": torch.from_numpy(audio.copy())[None, :]})
        class GeneratedOutput:
            device = torch.device("cpu")
            dtype = torch.float32
            def __call__(self, input_values):
                return torch.full((1, input_values.shape[-1] * 3), 0.25)
        features = Features()
        model = GeneratedOutput()
        values = np.full((800, 2), 4096, dtype=np.int16)
        rate, converted = ui._vc_offline(model, features, "harvest", 3, (8000, values))
        self.assertEqual(rate, 48000)
        self.assertEqual(converted.shape, (4800, 1))
        self.assertEqual(converted.dtype, np.int16)
        self.assertTrue(np.all(converted == 8192))
        self.assertEqual(features.f0_method, "harvest")
        self.assertEqual(features.last_kwargs["sampling_rate"], 16000)
        self.assertEqual(features.last_kwargs["f0_up_key"], 3)
        output, buffer = ui._vc(model, features, "pm", 0, 0.01, 0.01, 0, (8000, values), None, None)
        self.assertEqual(output[0], 48000)
        self.assertEqual(output[1].shape, (480, 1))
        self.assertEqual(buffer.shape, (160,))
        self.assertEqual(ui._vc_offline(None, None, None, None, None), None)


if __name__ == "__main__":
    unittest.main()
