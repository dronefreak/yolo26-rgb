"""The RGB output head, and the Yolo26RGB wrapper model. Original code.

Vendored Ultralytics building blocks live under `_vendor/` (`blocks.py` for
the backbone/neck modules, `depth_head.py` for the original `Depth` head
this is adapted from), not here, see `_vendor/README.md` for the rules.

`RGBHead` reuses `Depth`'s multi-scale fusion (project each of P3/P4/P5 to
`c_mid` channels, then progressively upsample-and-fuse down to P3
resolution), that part isn't depth-specific, it's just feature fusion. What
changes is everything after that:

- No `exp()` / log-affine calibration (depth-specific, values there must
  stay positive and log-scaled, RGB pixels don't).
- Two extra skip connections (from backbone P1/2 and P2/4, stride 2 and 4)
  feed into the tail at the matching resolution, projected to `c_head`
  channels and added in, the same proj+add pattern `Depth`'s own fusion
  already uses. Without these the tail would have to reconstruct all fine
  detail from an 8x-downsampled representation with no path back to
  anything higher-resolution, the same failure mode this project's own
  hypothesis blames ResNet's stem for. ClearView's ResNetUNet baseline
  carries skips down to stride 2 (its pre-maxpool `e0`), this matches that,
  not a design invented for its own sake.
- The tail predicts a residual, added back to the (padded) input in
  `Yolo26RGB.forward`, rather than regressing the output image directly.
  Matches NAFNet/Restormer's convention (see `clearview.models.nafnet`):
  no final `Sigmoid`, no in-model clamp, output isn't hard-bounded to
  [0, 1], relies on the pixel loss to pull it there during training, same
  as those baselines. Always returns a bare tensor, no train/eval dict
  split like `Depth` has, ClearView's loss functions expect a tensor.
- RGBHead's own conv blocks (`ConvLN` below) use LayerNorm instead of the
  vendored `Conv`'s BatchNorm: restoration training tends to run small
  batches on large crops, and BN's batch-dependent running stats get noisy
  there in a way LayerNorm (normalizes per-sample, over channels) doesn't,
  see NAFNet/Restormer's own choice of the same. Deliberately scoped to
  RGBHead only, not the vendored backbone/neck `Conv` (still BatchNorm) -
  keeps the backbone/neck loadable from a YOLO26 pretrained checkpoint (see
  `yolo26_rgb.models.pretrained`), and, more generally, keeps this repo's
  base architecture on BatchNorm, the choice that stays deployment-friendly
  (BatchNorm folds into the preceding conv at ONNX/TensorRT export time,
  LayerNorm structurally can't) and compatible with the whole YOLO26
  pretrained zoo (detect/segment/pose/OBB/classify, not just depth), not
  just this one task. A LayerNorm-everywhere variant was prototyped
  (2026-08-18) and removed (2026-08-19) to keep the base package simple -
  see ROADMAP.md's open questions if reconsidering it.
"""

import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vendor.blocks import autopad
from .pretrained import get_pretrained_state_dict, load_pretrained_backbone

_SCALES = ("n", "s", "m", "l", "x")
_WEIGHTS_RE = re.compile(r"yolo26([nsmlx])(?:-depth)?(?:\.pt)?$")
# Released Hub repos are named ".../yolo26-rgb-<scale>" (e.g. dronefreak/yolo26-rgb-s),
# which _WEIGHTS_RE deliberately does not match (there is no "yolo26<scale>" substring).
_REPO_SCALE_RE = re.compile(r"yolo26-rgb-([nsmlx])(?![a-z])")


def _parse_scale(weights: str) -> str:
    """Extract a scale letter from a weights shorthand, e.g. "yolo26n-depth.pt" -> "n".

    Accepts a bare scale letter too ("n"), and a path with any directory prefix.
    """
    weights = str(weights)
    if weights in _SCALES:
        return weights
    m = _WEIGHTS_RE.search(weights)
    if not m:
        raise ValueError(
            f"Could not parse a scale ({'/'.join(_SCALES)}) from weights={weights!r}. "
            "Expected a bare scale letter or a filename like 'yolo26n-depth.pt'."
        )
    return m.group(1)


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors.

    Normalizes each spatial location over the channel dimension, independently per
    sample, unlike BatchNorm2d which normalizes over the batch. Same formulation as
    NAFNet/Restormer's LayerNorm2d, standard in restoration literature, original
    implementation (not vendored, no Ultralytics/ClearView code copied).
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class ConvLN(nn.Module):
    """Conv2d + LayerNorm2d + SiLU, RGBHead's own conv block, see module docstring.

    Mirrors the vendored `Conv`'s call signature so it's a drop-in swap, LayerNorm
    instead of BatchNorm is the only difference.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=True)
        self.norm = LayerNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class RGBHead(nn.Module):
    """YOLO26-RGB output head: multi-scale fusion + full-resolution RGB restoration.

    Attributes:
        nl (int): Number of pyramid levels fused (P3/P4/P5).

    Examples:
        >>> head = RGBHead(ch=(256, 512, 1024), skip_ch=(32, 16))
        >>> x = [
        ...     torch.randn(1, 256, 80, 80),    # P3
        ...     torch.randn(1, 512, 40, 40),    # P4
        ...     torch.randn(1, 1024, 20, 20),   # P5
        ...     torch.randn(1, 32, 160, 160),   # P2/4 skip
        ...     torch.randn(1, 16, 320, 320),   # P1/2 skip
        ... ]
        >>> out = head(x)  # (1, 3, 640, 640), full input resolution, unbounded (residual)
    """

    def __init__(self, c_mid: int = 256, ch: tuple = (), skip_ch: tuple = ()):
        """Initialize RGBHead.

        Args:
            c_mid (int): Number of intermediate channels for the fusion decoder.
            ch (tuple): Input channel sizes from backbone/neck feature maps (P3, P4, P5).
            skip_ch (tuple): Input channel sizes for the tail's skip connections, in the
                order they're consumed (coarser first): (P2/4, P1/2). Empty tuple disables
                skip connections entirely (standalone use with only `ch` set, unlike the
                docstring example above, which sets both `ch` and `skip_ch`).
        """
        super().__init__()
        self.nl = len(ch)  # number of fused pyramid levels

        # Project each pyramid level to c_mid channels, same as RGBHead's depth-head ancestor.
        self.proj = nn.ModuleList(ConvLN(c, c_mid, k=1) for c in ch)
        self.refine = nn.ModuleList(
            nn.Sequential(ConvLN(c_mid, c_mid, k=3), ConvLN(c_mid, c_mid, k=3))
            for _ in ch[:-1]
        )

        c_head = (
            c_mid // 4
        )  # 64 for the default c_mid=256, matches every scale variant since c_mid isn't width-scaled
        self.head_proj = ConvLN(c_mid, c_mid // 2, k=3)  # P3 resolution (input/8)
        self.head_up = nn.ConvTranspose2d(
            c_mid // 2, c_mid // 2, kernel_size=2, stride=2, bias=True
        )  # -> input/4
        self.head_reduce = ConvLN(c_mid // 2, c_head, k=3)

        # Skip projections, matched 1:1 with skip_ch, consumed coarser-first (P2/4 then P1/2)
        # to line up with the tail's two upsample stages below.
        self.skip_proj = nn.ModuleList(ConvLN(c, c_head, k=1) for c in skip_ch)

        self.tail_refine1 = nn.Sequential(
            nn.Conv2d(c_head, c_head, 3, padding=1), nn.SiLU()
        )  # input/4, post P2 skip
        self.tail_up1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )  # -> input/2
        self.tail_refine2 = nn.Sequential(
            nn.Conv2d(c_head, c_head, 3, padding=1), nn.SiLU()
        )  # input/2, post P1 skip
        self.tail_up2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )  # -> input
        self.out_conv = nn.Conv2d(
            c_head, 3, 1
        )  # raw residual, no activation, see module docstring

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale features and predict a full-resolution RGB residual.

        Args:
            x: Feature tensors [P3, P4, P5, P2, P1] from the backbone/neck (the last two
                only present when `skip_ch` was non-empty at construction time).

        Returns:
            (B, 3, H, W) tensor, H/W match the model's input size. Unbounded, meant to be
            added to the input image by the caller (`Yolo26RGB.forward`), not a standalone
            output image.
        """
        feats = [self.proj[i](x[i]) for i in range(self.nl)]
        skip_feats = [proj(x[self.nl + i]) for i, proj in enumerate(self.skip_proj)]

        out = feats[-1]
        for i in range(self.nl - 2, -1, -1):
            # Depth uses align_corners=True here because it's baked into the
            # released depth weights (see _vendor/depth_head.py). That only
            # matters if proj/refine get loaded from that checkpoint, which
            # nothing does today, so this matches the tail's convention
            # instead. Revisit if pretrained loading lands on reusing these.
            out = F.interpolate(
                out, scale_factor=2, mode="bilinear", align_corners=False
            )
            out = out + feats[i]
            out = self.refine[i](out)

        out = self.head_proj(out)
        out = self.head_up(out)  # input/4
        out = self.head_reduce(out)
        if len(skip_feats) > 0:
            out = out + skip_feats[0]  # P2/4 skip
        out = self.tail_refine1(out)

        out = self.tail_up1(out)  # input/2
        if len(skip_feats) > 1:
            out = out + skip_feats[1]  # P1/2 skip
        out = self.tail_refine2(out)

        out = self.tail_up2(out)  # input
        return self.out_conv(out)


class Yolo26RGB(nn.Module):
    """YOLO26-RGB: YOLO26's dense prediction pathway repurposed for RGB image restoration.

    Wraps `backbone.YoloBackboneNeck` (built from `_vendor/yolo26-rgb.yaml`,
    the same architecture as `yolo26-depth.yaml` with the final head swapped
    from `Depth` to `RGBHead`) so the whole thing runs and checkpoints as
    one model. Forward pass takes (B, 3, H, W) in [0, 1] and returns
    (B, 3, H, W): `RGBHead` predicts a residual correction which is added
    back to the (padded) input here, NAFNet/Restormer-style, rather than
    regressing the output image directly, `state_dict()` saves directly as
    {"model_state_dict": ...} so ClearView's checkpoint-stripping /
    model-card pipeline works unmodified.

    Examples:
        >>> model = Yolo26RGB(scale="n")  # random init
        >>> x = torch.rand(1, 3, 256, 256)
        >>> out = model(x)  # (1, 3, 256, 256)
        >>> x2 = torch.rand(1, 3, 257, 300)  # not a multiple of `stride`
        >>> out2 = model(x2)  # (1, 3, 257, 300), padded internally then cropped back

        >>> # or, one-line depth-pretrained backbone (downloads + loads automatically):
        >>> model = Yolo26RGB("yolo26n-depth.pt")

        >>> # or, a trained deraining model from the Hugging Face Hub (returned in eval mode):
        >>> model = Yolo26RGB.from_pretrained("dronefreak/yolo26-rgb-s")
    """

    # 5 stride-2 convs in the backbone (P1 through P5, see yolo26-rgb.yaml),
    # constant across every scale variant, scale only changes channel widths/
    # depths, not this. The neck's Concat layers need each pyramid level's
    # spatial size to divide out exactly, so inputs get padded up to a
    # multiple of this before the forward pass and cropped back after,
    # otherwise non-multiple-of-32 inputs raise a shape-mismatch error deep
    # inside the neck.
    stride = 32

    def __init__(
        self,
        weights: str | None = None,
        scale: str = "n",
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Yolo26RGB.

        Args:
            weights: shorthand for loading a depth-pretrained backbone, e.g.
                "yolo26n-depth.pt" (or just "n") - parses the scale out and, if
                `pretrained` is True, downloads/extracts/loads that checkpoint's
                backbone/neck automatically (see `yolo26_rgb.models.pretrained`),
                mirroring `ultralytics.YOLO("yolo26n-depth.pt")`'s own convention.
                Overrides `scale` when given.
            scale: one of "n", "s", "m", "l", "x". Ignored if `weights` is given.
            pretrained: when `weights` is given, whether to actually download/load
                the depth-pretrained backbone (default True), vs. only using
                `weights` to pick the scale and otherwise starting from random
                initialization (False). Has no effect if `weights` is None.
        """
        super().__init__()
        from pathlib import Path

        from .backbone import YoloBackboneNeck

        if weights is not None:
            scale = _parse_scale(weights)

        cfg_path = Path(__file__).parent / "_vendor" / "yolo26-rgb.yaml"
        self.net = YoloBackboneNeck(cfg_path, scale=scale, ch=3, **kwargs)

        if weights is not None and pretrained:
            state_dict_path = get_pretrained_state_dict(scale)
            load_pretrained_backbone(self, state_dict_path)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        filename: str | None = None,
        scale: str | None = None,
        map_location: Any = "cpu",
        **hf_kwargs: Any,
    ) -> "Yolo26RGB":
        """Build a `Yolo26RGB` with a released deraining checkpoint from the Hugging Face Hub.

        Downloads `yolo26{scale}-rgb.pt` from `repo_id` (via `huggingface_hub`, cached in the
        usual HF cache), constructs the matching-scale model with a random-init head, and loads
        the full checkpoint into it. Unlike the `weights=` constructor arg - which only loads
        YOLO26's *depth* backbone into an otherwise random model - this restores a fully trained
        restoration model.

        Args:
            repo_id: e.g. "dronefreak/yolo26-rgb-s". The scale is parsed from the trailing
                "-n" / "-s" unless `scale` is passed explicitly.
            filename: checkpoint file within the repo. Defaults to "yolo26{scale}-rgb.pt".
            scale: "n" or "s" (any of `_SCALES` is accepted). Parsed from `repo_id` when omitted.
            map_location: forwarded to `torch.load`; also the device the returned model is moved
                to. Defaults to CPU - call `.to(...)` / `.cuda()` afterwards as needed.
            **hf_kwargs: forwarded to `huggingface_hub.hf_hub_download` (`revision`, `token`,
                `cache_dir`, `local_files_only`, ...).

        Returns:
            A `Yolo26RGB` in `eval()` mode with the released weights loaded.
        """
        if scale is None:
            m = _REPO_SCALE_RE.search(repo_id)
            if not m:
                raise ValueError(
                    f"Could not parse a scale from repo_id={repo_id!r}; pass scale=... "
                    "explicitly (expected an id like 'dronefreak/yolo26-rgb-s')."
                )
            scale = m.group(1)
        if scale not in _SCALES:
            raise ValueError(f"Unknown scale {scale!r}, expected one of {_SCALES}")
        if filename is None:
            filename = f"yolo26{scale}-rgb.pt"

        try:
            from huggingface_hub import hf_hub_download
        except (
            ImportError
        ) as e:  # pragma: no cover - dependency is declared in pyproject
            raise ImportError(
                "YOLO26RGB.from_pretrained needs `huggingface_hub` (`pip install huggingface_hub`, "
                "or reinstall this package). To load a local checkpoint without it, use "
                'YOLO26RGB(scale=...) then model.load_state_dict(torch.load(path)["model_state_dict"]).'
            ) from e

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, **hf_kwargs)

        model = cls(scale=scale, pretrained=False)
        # Our own checkpoint: a plain {"model_state_dict": tensors} dict (see class docstring),
        # or a bare state_dict. weights_only=False matches yolo26_rgb.scripts.evaluate.
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        model.load_state_dict(state_dict)
        if isinstance(map_location, (str, torch.device)):
            model.to(map_location)
        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (-h) % self.stride
        pad_w = (-w) % self.stride
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        out = (
            self.net(x) + x
        )  # residual: net predicts a correction, add back to the (padded) input
        return out[..., :h, :w]
