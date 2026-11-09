# Dependency compatibility and security

The November 2026 environment uses Python 3.11, Transformers 5.10.1 and
Torch 2.13.0 from PyPI. Package metadata pins the same Transformers version as
Pipfile and Pipfile.lock. The Linux PyPI Torch 2.13.0 wheels select CUDA 13 dependencies. The obsolete
CUDA 11.8 source and its dependency records are removed rather than requesting
a release absent from that index. CPU API/checkpoint/UI verification does not
validate CUDA 13 driver support or GPU inference.

CI reads the candidate Torch version from the lock and installs its matching
official CPU wheel. It applies the remaining locked runtime versions as
constraints, installs the actual package and UI recipe, checks installed
Torch/Transformers versions against the lock, and runs pip check. This prevents
a dependency-only PR from passing tests against older setup.cfg or hard-coded
Torch selections. Constraints do not constitute a full CUDA Pipenv sync.

The existing datasets 5.0.1 requires fsspec <=2026.6.0. Preserve that compatible
record rather than the incompatible 2026.7.0 suggested by dependency automation.
evaluate requires at least 0.4.6 so a resolver cannot silently fall back to
obsolete evaluate/datasets releases while keeping another installed dependency.

Transformers 5 returns empty loading-info sets where 4 returned empty lists.
The round-trip tests assert that missing/unexpected keys are empty, independent
of collection type, then compare every saved tensor. They retain normal RVC,
VITS and legacy HuBERT weight-normalization checkpoint coverage; checkpoint
formats and model architecture are unchanged. The generated tiny CPU forward
and local safetensors tests fetch no model weights or datasets.

The UI continues to use gradio==6.16.0+esnya.pillow13.1 with native Pillow 13.
The [reviewable recipe](../dependency_patches/gradio/README.md) checks the
official upstream wheel URL, size, SHA-256, prepared metadata and patch contexts,
preserves license notices and rebuilds RECORD. It preserves integer-PNG clipping
explicitly and declares Pillow <14. The local version identifies this as a
maintenance distribution, not an official Gradio release. Metadata preparation
is offline; wheel builds fetch only the fixed public artifact or use the
ESNYA_COMPAT_WHEELHOUSE cache, without executing upstream build code.

CPU checks cover the real UI tree with launch disabled, generated WAV input and
output, conversion callbacks, PNG metadata/clipping, JPEG, TIFF and GIF. They do
not start a service. Common Voice integration, CUDA inference, microphone/voice
quality and optional fairseq conversion remain outside these checks. The
fairseq converter retains its trusted-legacy-checkpoint/explicit --unsafe
requirement. No model or dataset licence is accepted by the offline workflow.

This compatibility validation is not a fresh universal vulnerability scan.
The earlier advisory applicability analysis should be reassessed before adding
new model families, remote custom code or different checkpoint-loading paths.
