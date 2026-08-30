# yolo26-rgb

YOLO26's dense depth-estimation head, repurposed to output 3-channel RGB instead of 1-channel depth, for image restoration tasks (denoising, dehazing, deraining).

## Quickstart

```python
from yolo26_rgb import YOLO26RGB

model = YOLO26RGB("yolo26n-depth.pt")  # downloads + loads the depth-pretrained backbone automatically
```

That's it — `YOLO26RGB(...)` parses the scale (`n`/`s`/`m`/`l`/`x`) out of the checkpoint name, downloads it from Ultralytics' own release assets if it isn't already cached locally, extracts and loads the pretrained backbone/neck, and hands back a ready-to-use model (a plain `nn.Module`, call it directly: `model(image_tensor)`). Same convention as `ultralytics.YOLO("yolo26n-depth.pt")`. Prefer training from scratch instead? `YOLO26RGB("n", pretrained=False)`.

`model(image_tensor)` takes `(B, 3, H, W)` in `[0, 1]` and returns a bare `(B, 3, H, W)` tensor — no dict, no metadata, any input size (internally padded to a multiple of 32 and cropped back). Nothing else to unpack, so it drops straight into an existing restoration pipeline. One thing to know: the output itself is **not** clamped to `[0, 1]` (`RGBHead` predicts a residual correction added to the input, NAFNet/Restormer-style, rather than regressing a bounded image directly) — `.clamp(0, 1)` before saving/displaying as an image.

## Why this exists

Detection and segmentation backbones are usually evaluated for cross-task transfer within vision (does a good detector also segment well), but rarely against dense pixel-regression restoration tasks like deraining. YOLO26 already ships a depth-estimation head, itself a dense, full-resolution regression task, which makes it a more interesting test case than a plain classification backbone. Classification-pretrained backbones (e.g. ResNet-UNet) are known to underperform purpose-built restoration architectures on rain removal despite far more parameters, a plausible reason being the classification-tuned stem's aggressive early downsampling losing fine rain-streak detail before any residual block runs. Swapping YOLO26's depth head for an RGB head is architecturally straightforward, since the decoder already upsamples to full input resolution, the open question is whether that decoder's inductive bias, tuned for real-time detection speed and then adapted once for depth, transfers to rain removal any better.

n/s/m/l/x variants, each trained and evaluated against the same mixed-domain recipe and a 10-test-set protocol.

## Install

```bash
pip install -e .
```

Only pulls in `torch` — this release is the model itself, nothing else. Training and evaluation for the checkpoints in this repo used an external package, [ClearView](https://github.com/dronefreak/clearview) (data loading, mixed-domain training recipes, the 10-test-set eval protocol), but ClearView is not a dependency of this package and its training/eval scripts aren't included here.

`YOLO26RGB(...)` covers the common case; for lower-level control (downloading a checkpoint without loading it yet, loading into a model you already constructed) see the `yolo26_rgb.models.pretrained` module docstring.

## Structure

```text
yolo26_rgb/
└── models/
    ├── __init__.py       # get_model() registry
    ├── backbone.py        # original: YAML -> nn.Module builder + the layer-graph runner
    ├── heads.py            # original: RGBHead + the Yolo26RGB wrapper
    ├── pretrained.py       # download/extract/load a YOLO26 depth-pretrained backbone
    └── _vendor/            # Ultralytics YOLO26 source this depends on, AGPL-3.0, original copyright retained
        ├── blocks.py        # backbone/neck building blocks (Conv, C3k2, SPPF, C2PSA, Concat, ...)
        ├── depth_head.py     # the original 1-channel Depth head, RGBHead's structural reference
        ├── yolo26-depth.yaml # the original Ultralytics depth config, kept for reference
        └── yolo26-rgb.yaml   # this repo's actual config (same backbone/neck, RGBHead instead of Depth)
```

Just the model — this release doesn't include the training/evaluation scripts or mixed-domain configs used to produce the released checkpoints (see [Install](#install)).

`get_model("yolo26_rgb_{n,s,m,l,x}")` builds every scale variant; each runs a forward+backward pass and returns a bare `(B, 3, H, W)` tensor at full input resolution (not a downsampled dict like the original depth head), and round-trips through `state_dict()` in a plain `{"model_state_dict": ...}` checkpoint format.

## License: AGPL-3.0

AGPL-3.0, see [LICENSE](LICENSE), inherited from depending on AGPL-licensed Ultralytics source, see the "why AGPL-3.0" section above.

Ultralytics' YOLO26 source is licensed AGPL-3.0 (or requires an Enterprise license for closed use, see [their licensing page](https://www.ultralytics.com/license)), which this repo inherits as a result of depending on it.

Any code copied or adapted from Ultralytics' YOLO26 source lives under [`yolo26_rgb/models/_vendor/`](yolo26_rgb/models/_vendor/), each file keeps its original Ultralytics copyright notice plus an AGPL-3.0 header. This is **not affiliated with or endorsed by Ultralytics**, it repurposes their published architecture for a different task, no claim of ownership over the original YOLO26 design.

## Citation

```bibtex
@software{saksena2026yolo26rgb,
  author = {Saksena, Saumya Kumaar},
  title = {yolo26-rgb: YOLO26's depth head repurposed for RGB image restoration},
  year = {2026},
  url = {https://github.com/dronefreak/yolo26-rgb}
}
```

Built on Ultralytics YOLO26, see [Ultralytics' licensing terms](https://www.ultralytics.com/license) for the underlying architecture. If citing the architecture this repo is built on:

```bibtex
@article{jocher2026yolo26,
  title = {Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
  author = {Jocher, Glenn and Qiu, Jing and Liu, Mengyu and Lyu, Shuai and Akyon, Fatih Cagatay and Kalfaoglu, Muhammet Esat},
  year = {2026},
  eprint = {2606.03748},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2606.03748},
  url = {https://arxiv.org/abs/2606.03748}
}

@software{ultralytics_yolo,
  author = {Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
  title = {Ultralytics YOLO},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```
