# Vendored Ultralytics source

Any code copied or adapted from Ultralytics' YOLO26 source goes here, nowhere else in this repo.

Rules for anything added to this directory:

1. Keep the original Ultralytics copyright notice at the top of every file.
2. Add an AGPL-3.0 header (this whole repo is AGPL-3.0, but be explicit in files that are directly derived from Ultralytics' code, not just adapted).
3. Note in a comment which upstream file/commit/version it was copied from, for traceability.
4. Nothing outside `yolo26_rgb/models/` should import from this directory directly, go through `models/backbone.py` or `models/heads.py`, not straight into `_vendor`.

## Current contents

- `blocks.py`: backbone/neck building blocks (`Conv`, `C3k2`/`C3k`/`C2f`, `SPPF`, `C2PSA`, `Bottleneck`, `PSABlock`/`Attention`, `Concat`), vendored from `ultralytics/nn/modules/conv.py` and `ultralytics/nn/modules/block.py`. Only the classes `yolo26-rgb.yaml` actually references were kept.
- `depth_head.py`: the original `Depth` head, vendored from `ultralytics/nn/modules/head.py`, unmodified. Kept as reference, `../heads.py`'s `RGBHead` reuses its multi-scale fusion structure but is original code, not a copy, see that file's docstring.
- `yolo26-depth.yaml`: the original Ultralytics depth config, copied unmodified (already carries its own Ultralytics AGPL-3.0 header). Not used by this repo's model anymore, kept for reference/traceability.
- `yolo26-rgb.yaml`: this repo's actual config, adapted from `yolo26-depth.yaml`, identical backbone/neck, only the final head line points at `RGBHead` instead of `Depth`. Carries its own header noting the adaptation.

`../backbone.py` (the YAML-to-`nn.Module` builder) and `../heads.py` (`RGBHead` + the `Yolo26RGB` wrapper) are **not** in this directory, they're original code that assembles/adapts the vendored pieces above, not copies of Ultralytics source. `backbone.py`'s docstring explains which specific logic it traces from Ultralytics' `parse_model()`, since it deliberately mirrors that algorithm's channel-scaling behavior even though the implementation itself is a from-scratch rewrite.
