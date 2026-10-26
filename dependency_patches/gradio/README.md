# Gradio Pillow 13 compatibility maintenance build

This local-version distribution is based on the exact official Gradio 6.16.0
wheel identified by patch.json. It retains the original runtime/frontend files
and license notices except the declared PNG conversion and dependency metadata
patch. It preserves legacy integer-PNG clipping explicitly under Pillow 13 and
widens its declared Pillow upper bound to 14. This is not an official release.

The stdlib-only PEP 517 backend validates upstream URL, size, digest and exact
patch contexts, then rebuilds RECORD. Metadata preparation is offline; wheel
building fetches only the fixed public wheel, or uses ESNYA_COMPAT_WHEELHOUSE.
It never imports or executes upstream Python/build code. No private project
manifest, source, credential or installed-package inventory is uploaded.
The original public Gradio runtime version stays 6.16.0 for frontend protocol
compatibility; distribution metadata identifies the maintenance version.
