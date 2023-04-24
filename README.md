HF RVC
----

HF RVC is a package for Retrieval-based-Voice-Conversion (RVC) implementation using HuggingFace's [transformers](https://github.com/huggingface/transformers), along with the capability to convert from original unsafe models. The library is easy to use and provides an efficient way to perform voice conversion tasks.

## Original Implementation

The original implementation of the Retrieval-based-Voice-Conversion (RVC) can be found at the following GitHub repository: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.


## Installation

This library supports Python 3.10 only. If you would like to use this library with an older version of Python, you may need to modify the code to ensure compatibility. In such cases, we welcome pull requests for any necessary changes.

To install the library, you can use the following command:

```
pip install git+https://github.com/esnya/hf-rvc.git#egg=hf-rvc
```


## Basic Usage

Here's an example of how to use the package as a library:

```python
from hf_rvc import RVCFeatureExtractor, RVCModel

feature_extractor = RVCFeatureExtractor.from_pretrained(model_name)
# feature_extractor.set_f0_method("harvest")

model = RVCModel.from_pretrained(model_name)

input_features = feature_extractor(
    audio["array"],
    sampling_rate=audio["sampling_rate"],
    f0_up_key=key,
    return_tensors="pt",
)
output = model(**input_features).numpy()
```


## Command Line Tools

HF RVC also provides several command line tools. You can check the available commands with --help:

```bash
python -m hf_rvc --help
```

For more detailed usage of each tool, you can check the --help option of the specific command.

### Model Conversion

Since the models are not yet available on HuggingFace, you may need to convert the original `.pth` model files or the original `hubert_base.pt` model to be compatible with HF RVC. This can be done using the `convert-***` command line tools. The conversion not only adapts the models to the different frameworks but also, by default, ensures compatibility with the [safetensor](https://github.com/huggingface/safetensors) format for **secure** model sharing.

When using the command line tools to load the `hubert_base.pt` model, you must provide the `--unsafe` option to confirm that you have obtained the model from a trusted source.

Example:

```bash
python -m hf_rvc convert-rvc --hubert-path <path_to_hubert_base.pt> --unsafe <path_to_original_vits_model.pth>
```

### Real-time Voice Conversion

A real-time voice conversion tool that allows you to perform voice conversion on-the-fly. To use this feature, simply run the following command:

```
python -m hf_rvc realtime-vc path/to/converted_or_hugging_face/model
```

This tool accepts various arguments, including the model, feature extractor, buffering settings, F0 method, input and output device indices, and more. It enables users to adjust the conversion parameters in, offering greater control and customization.

```
python -m hf_rvc realtime-vc --help
```

For a detailed explanation of each argument, please run the `--help` command as shown above.

## Attribution

The code in `hf_rvc/models/vits` is based on the original implementation published under the MIT License Copyright (c) 2023 liujing04. The code has been slightly modified by esnya.

## Contributing

We welcome contributions to the package! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request on the GitHub repository.


## License

HF RVC is licensed under the [MIT License](./LICENSE).
