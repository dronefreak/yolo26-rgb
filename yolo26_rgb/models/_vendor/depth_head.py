# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Vendored from ultralytics/nn/modules/head.py (the Depth class). YOLO26's
# monocular depth-estimation head: takes 3 multi-scale backbone/neck
# feature maps (P3/P4/P5) and produces a dense, full-resolution depth map
# via progressive upsampling and fusion. Kept here unmodified, as a
# reference and a base to adapt for RGB output, see ../heads.py for the
# actual RGB variant.
#
# Not affiliated with or endorsed by Ultralytics.

import torch
import torch.nn as nn
from torch.nn import functional as F

from .blocks import Conv


class Depth(nn.Module):
    """YOLO Depth head for monocular depth estimation.

    A dense prediction head that takes multi-scale backbone features and produces a single-channel depth map via
    progressive upsampling and fusion.

    Attributes:
        nl (int): Number of pyramid levels.
        cal_a (torch.Tensor): Log-affine calibration scale buffer, identity 1.0 by default.
        cal_b (torch.Tensor): Log-affine calibration offset buffer, identity 0.0 by default.

    Examples:
        >>> depth = Depth(ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> out = depth(x)  # training: {"depth": (1, 1, 160, 160)} at P2 resolution (input/4)
    """

    export = False  # export mode

    def __init__(self, c_mid: int = 256, ch: tuple = ()):
        """Initialize Depth head.

        Args:
            c_mid (int): Number of intermediate channels for the fusion decoder.
            ch (tuple): Input channel sizes from backbone feature maps (P3, P4, P5).
        """
        super().__init__()
        self.nl = len(ch)  # number of detection layers (pyramid levels)

        # Project each pyramid level to c_mid channels
        self.proj = nn.ModuleList(Conv(c, c_mid, k=1) for c in ch)

        # Refinement blocks after each of the nl-1 fusion steps (the coarsest level is not refined)
        self.refine = nn.ModuleList(
            nn.Sequential(Conv(c_mid, c_mid, k=3), Conv(c_mid, c_mid, k=3))
            for _ in ch[:-1]
        )

        self.head = nn.Sequential(
            Conv(c_mid, c_mid // 2, k=3),
            nn.ConvTranspose2d(
                c_mid // 2, c_mid // 2, kernel_size=2, stride=2, bias=True
            ),
            Conv(c_mid // 2, c_mid // 4, k=3),
            nn.Conv2d(c_mid // 4, 1, kernel_size=1),
        )
        # Initialize to ~1.2 m so early exp() outputs stay well-conditioned. Unmodified
        # reference code (see module docstring), not on any path this repo actually runs --
        # ignores below are for nn.Module's generic __getattr__ stub (Tensor | Module),
        # not a real type issue, this works fine at runtime.
        self.head[-1].bias.data.fill_(0.182)  # type: ignore[operator]

        # Scale-only log-affine calibration d' = exp(a·log d + b); identity by default.
        self.register_buffer("cal_a", torch.ones(1))
        self.register_buffer("cal_b", torch.zeros(1))

    def forward(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor] | torch.Tensor:
        """Fuse multi-scale features and predict depth.

        Args:
            x: List of feature tensors [P3, P4, P5] from the backbone/neck.

        Returns:
            Training: dict {"depth": (B, 1, H/4, W/4)}, the raw head output the loss supervises.
            Eval: (B, 1, H/4, W/4) with calibration applied; the predictor/validator resize to image/GT size.
            Export (self.export=True): (B, 1, H, W), upsampled 4x to the input size. Output is unbounded.
        """
        # Project all levels to same channel dim
        feats = [self.proj[i](x[i]) for i in range(self.nl)]

        out = feats[-1]
        for i in range(self.nl - 2, -1, -1):
            # align_corners=True is baked into the released depth weights. Constant scale (consecutive pyramid
            # levels) keeps the upsample static for dynamic-shape CoreML export; output size is identical.
            out = F.interpolate(
                out, scale_factor=2, mode="bilinear", align_corners=True
            )
            out = out + feats[i]
            out = self.refine[i](out)

        out = self.head(out)  # (B, 1, H/4, W/4)
        depth = torch.exp(out.clamp(-4.0, 5.0))

        if self.training:
            return {"depth": depth}

        depth = depth.pow(self.cal_a) * self.cal_b.exp()  # type: ignore[arg-type,operator]
        if self.export:
            depth = F.interpolate(
                depth, scale_factor=4.0, mode="bilinear", align_corners=False
            )
        return depth
