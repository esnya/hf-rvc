# Dependency compatibility and security

The October 2026 update keeps the HuBERT/RVC implementation on Transformers
4.48.3. The old deserialization advisories were patched in 4.36.0 and 4.48.0:
[GHSA-3863-2447-669p](https://github.com/advisories/GHSA-3863-2447-669p),
[GHSA-qxrp-vhvm-j765](https://github.com/advisories/GHSA-qxrp-vhvm-j765),
[GHSA-wrfc-pvp9-mr9g](https://github.com/advisories/GHSA-wrfc-pvp9-mr9g), and
[GHSA-hxxf-235m-72v3](https://github.com/advisories/GHSA-hxxf-235m-72v3).
4.48.3 is the patch release within that minor series; this is a bounded
compatibility update, not a claim that all Transformers advisories are cleared.
The dependency environment and package metadata now require Python 3.11,
retiring the old Python 3.8/3.9/3.10 test targets.

The project directly constructs `HubertForCTC` and its own `RVCModel`, and saves
model weights and feature extractors. It does not load LightGlue, load a remote
generation implementation, or save tokenizer/processor chat templates. The
4.48.3 wheel also validates the attention implementation against built-in names
and has no remote-kernel loader. The newer broad version alerts
[GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72),
[GHSA-fgcw-684q-jj6r](https://github.com/advisories/GHSA-fgcw-684q-jj6r), and
[GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)
remain visible in version-only scanners. Their reported execution paths are
not used by these tools. Reassess this decision before adding model families,
custom code loading, or tokenizer/processor saving.

`torch.load(weights_only=True)` is used by the VITS converter. Its safety
bypass was fixed in PyTorch 2.6.0
([GHSA-53q9-r3pm-6pq6](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)).
The package now requires at least that version. Pipenv pins 2.7.1+cu118 because
2.13.0 is not published on the configured CUDA 11.8 index. The optional fairseq
converter still requires trusted legacy checkpoints and the explicit `--unsafe`
flag; it is outside the offline smoke test.

The upgrade also preserves VITS `weight_g`/`weight_v` checkpoint names while
allowing Transformers to migrate HuBERT weight normalization names. Without
that distinction the newer generic loader silently leaves VITS parameters
uninitialized. The offline tests compare every tensor after local RVC and VITS
safetensors round trips, including the correct composite RVC configuration.

Gradio is held to the 6.16 patch series, compatible with the Hub version required
by Transformers 4.48.3. It fixes the cookie injection and FileExplorer traversal
advisories. **Pillow remains unresolved:** Gradio 6.16.0 and even the current
6.30.0 require `pillow<13.0`, while the current critical Pillow fixes require
13.0.0 (which also requires Python 3.11+). This update does not override that
upstream constraint and must not be described as a clean security scan.

The default CI tests use locally constructed tiny models, generated tensor
checkpoints, and local safetensors round trips. They require no model weights,
Hub access, or dataset license acceptance. The existing Common Voice feature
extraction suite remains an explicit integration test and is not invoked by
the offline workflow. A passing CPU test does not validate CUDA, microphone
devices, fairseq conversion, or live voice quality.
