"""Main entry point for the hf_rvc package."""

from argh import ArghParser

from .converters import convert_hubert, convert_rvc, convert_vits
from .tools import eval_dataset, gradio_vc, list_audio_devices, realtime_vc


def main() -> None:
    """Main entry point function."""
    parser = ArghParser()
    parser.add_commands(
        [
            convert_hubert,
            convert_vits,
            convert_rvc,
            realtime_vc,
            gradio_vc,
            eval_dataset,
            list_audio_devices,
        ]
    )
    parser.dispatch()


if __name__ == "__main__":
    main()
