# T-302 provenance and modifications

Derived from [WeSpeaker `simclr_wrapper.py`](https://github.com/wenet-e2e/wespeaker/blob/9fecd6cb4f47475d01761d87c826298dff4ef18c/wespeaker/ssl/models/simclr_wrapper.py),
commit `9fecd6cb4f47475d01761d87c826298dff4ef18c`, copyright 2023 Zhengyang Chen.
Apache-2.0; full upstream LICENSE is included. No root NOTICE file was present.

Exact borrowed regions: lines 36–40 (Linear/ReLU/Linear MLP) and 54–85
(InfoNCE labels, normalization, diagonal removal, positive-first assembly).
`projection.py` is a modified excerpt, not the entire encoder wrapper.

Changes by T-302: detached encoder; configurable rectangular dimensions defaulting
to 192→64→32; unit output; exactly two aligned views; shape/batch/temperature
validation; bool labels allocated on the input device; standalone logits function.
No pretrained weights, audio, training runner, external services or live integration.
Caller selects a device and moves the head/tensors; the module never downloads models.

Usage from repository root: `.venv/bin/python eval/borrowed/T-302/smoke.py`.
Tests load the path using importlib because `T-302` is not a Python identifier.
This is a repository-local research artifact, not included in the hearing wheel.
