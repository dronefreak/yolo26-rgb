# yolo26-rgb

YOLO26's dense depth-estimation head, repurposed to output 3-channel RGB instead of 1-channel depth, for image restoration tasks (denoising, dehazing, deraining). Built on top of [ClearView](https://github.com/dronefreak/clearview) for data loading, mixed-domain training recipes, evaluation metrics, and eval protocol, so results here are directly comparable to ClearView's own model zoo.

**Status: scaffolding only. No model implementation yet, this is project structure, not a working release.**

## Why this exists

Detection and segmentation backbones are usually evaluated for cross-task transfer within vision (does a good detector also segment well), but rarely against dense pixel-regression restoration tasks like deraining. YOLO26 already ships a depth-estimation head, itself a dense, full-resolution regression task, which makes it a more interesting test case than a plain classification backbone. ClearView's own [benchmark results](https://github.com/dronefreak/clearview/blob/main/.github/README.md) already show that ResNet18/34/50-UNet, all classification backbones, underperform purpose-built restoration architectures despite far more parameters, a plausible reason being the classification-tuned stem's aggressive early downsampling losing fine rain-streak detail before any residual block runs. Swapping YOLO26's depth head for an RGB head is architecturally straightforward, since the decoder already upsamples to full input resolution, the open question is whether that decoder's inductive bias, tuned for real-time detection speed and then adapted once for depth, transfers to rain removal any better.

n/s/m/l/x variants, each trained and evaluated against the exact same mixed-domain recipe and 10-test-set protocol as the rest of the ClearView zoo.

## Why this is a separate repo, and why AGPL-3.0

Ultralytics' YOLO26 source is licensed AGPL-3.0 (or requires an Enterprise license for closed use, see [their licensing page](https://www.ultralytics.com/license)). Rather than vendor their code into [ClearView](https://github.com/dronefreak/clearview) itself (Apache-2.0) and risk pulling the whole project under copyleft, this repurposed variant lives in its own repo, under AGPL-3.0, and depends on ClearView as an ordinary pip dependency (the safe direction, permissive code can be freely used by a copyleft project, not the other way around). ClearView itself has no dependency on this repo and never imports from it.

Any code copied or adapted from Ultralytics' YOLO26 source lives under [`yolo26_rgb/models/_vendor/`](yolo26_rgb/models/_vendor/), each file keeps its original Ultralytics copyright notice plus an AGPL-3.0 header. This is **not affiliated with or endorsed by Ultralytics**, it repurposes their published architecture for a different task, no claim of ownership over the original YOLO26 design.

## Install

```bash
pip install -e .
```

Pulls in [ClearView](https://github.com/dronefreak/clearview) from GitHub as a dependency, along with torch/torchvision.

## Structure

```
yolo26_rgb/
├── models/
│   ├── _vendor/        # Ultralytics YOLO26 source this depends on, AGPL-3.0, original copyright retained
│   └── heads.py         # the rewritten 3-channel RGB output head (TODO)
├── scripts/
│   ├── train.py         # training entry point, reuses clearview.data / clearview.utils.metrics (TODO)
│   └── evaluate.py       # evaluation entry point, same 10-test-set protocol as ClearView (TODO)
configs/                 # training mix configs, likely symlinked or copied from clearview/configs/mix/
```

## License

AGPL-3.0, see [LICENSE](LICENSE). This is a deliberate departure from ClearView's Apache-2.0 license, forced by depending on AGPL-licensed source, not a project-wide policy change, see the "why AGPL-3.0" section above.

## Citation

```bibtex
@software{saksena2026yolo26rgb,
  author = {Saksena, Saumya Kumaar},
  title = {yolo26-rgb: YOLO26's depth head repurposed for RGB image restoration},
  year = {2026},
  url = {https://github.com/dronefreak/yolo26-rgb}
}
```

Built on:
- Ultralytics YOLO26, see [Ultralytics' licensing terms](https://www.ultralytics.com/license) for the underlying architecture.
- [ClearView](https://github.com/dronefreak/clearview), Saksena, 2025, Apache-2.0.
