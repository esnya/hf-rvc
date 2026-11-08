import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import HubertConfig

from hf_rvc import RVCConfig, RVCModel, SynthesizerTrnMs256NSFsidConfig
from hf_rvc.converters.convert_vits import load_vits_checkpoint


class TestLocalModel(unittest.TestCase):
    @staticmethod
    def tiny_config():
        return RVCConfig(
            hubert=HubertConfig(
                vocab_size=256,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                conv_dim=(8, 8, 8),
                conv_stride=(5, 2, 2),
                conv_kernel=(10, 3, 3),
                num_conv_pos_embeddings=16,
                num_conv_pos_embedding_groups=2,
                mask_time_prob=0,
            ),
            vits=SynthesizerTrnMs256NSFsidConfig(
                spec_channels=9,
                segment_size=4,
                inter_channels=8,
                hidden_channels=8,
                filter_channels=16,
                n_heads=2,
                n_layers=1,
                resblock_kernel_sizes=[3],
                resblock_dilation_sizes=[[1, 3, 5]],
                upsample_rates=[2, 2],
                upsample_initial_channel=16,
                upsample_kernel_sizes=[4, 4],
                spk_embed_dim=1,
                gin_channels=8,
                sr=16000,
            ),
        )

    def test_cpu_forward(self):
        model = RVCModel(self.tiny_config()).eval()
        with torch.inference_mode():
            output = model(
                torch.zeros(1, 1600),
                torch.ones(1, 160, dtype=torch.long),
                torch.zeros(1, 160),
            )
        self.assertEqual(output.shape[:2], (1, 1))
        self.assertGreater(output.shape[-1], 0)
        self.assertTrue(torch.isfinite(output).all())

    def test_local_safetensors_roundtrip(self):
        model = RVCModel(self.tiny_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            self.assertTrue((Path(directory) / "model.safetensors").is_file())
            restored, loading = RVCModel.from_pretrained(
                directory, local_files_only=True, output_loading_info=True
            )
        self.assertIsInstance(restored.config, RVCConfig)
        self.assertIsInstance(restored.config.hubert, HubertConfig)
        self.assertEqual(restored.hubert.config.hidden_size, 16)
        self.assertEqual(loading["missing_keys"], [])
        self.assertEqual(loading["unexpected_keys"], [])
        torch.testing.assert_close(restored.state_dict(), model.state_dict())

    def test_vits_safetensors_roundtrip(self):
        from hf_rvc import SynthesizerTrnMs256NSFsid

        model = SynthesizerTrnMs256NSFsid(self.tiny_config().vits)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored, loading = SynthesizerTrnMs256NSFsid.from_pretrained(
                directory, local_files_only=True, output_loading_info=True
            )
        self.assertEqual(loading["missing_keys"], [])
        self.assertEqual(loading["unexpected_keys"], [])
        torch.testing.assert_close(restored.state_dict(), model.state_dict())

    def test_legacy_hubert_weight_norm_checkpoint(self):
        model = RVCModel(self.tiny_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            path = Path(directory) / "model.safetensors"
            legacy = {}
            for key, value in load_file(path).items():
                if key.startswith("hubert."):
                    key = key.replace(
                        ".parametrizations.weight.original0", ".weight_g"
                    ).replace(".parametrizations.weight.original1", ".weight_v")
                legacy[key] = value
            save_file(legacy, path, metadata={"format": "pt"})
            restored, loading = RVCModel.from_pretrained(
                directory, local_files_only=True, output_loading_info=True
            )
        self.assertEqual(loading["missing_keys"], [])
        self.assertEqual(loading["unexpected_keys"], [])
        torch.testing.assert_close(restored.state_dict(), model.state_dict())

    def test_local_weights_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save({"weight": {"test": torch.ones(2)}}, path)
            checkpoint = load_vits_checkpoint(path)
        torch.testing.assert_close(checkpoint["weight"]["test"], torch.ones(2))


if __name__ == "__main__":
    unittest.main()
