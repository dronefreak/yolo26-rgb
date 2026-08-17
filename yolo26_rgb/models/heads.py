"""The rewritten RGB output head, and the Yolo26RGB wrapper model.

TODO: this is where the actual work happens.

Plan, per the conversation that spawned this repo:
- Take YOLO26's depth-estimation decoder/neck (already does dense,
  full-resolution regression, unlike the detection/segmentation heads),
  vendor whatever source is needed under `_vendor/`, keep original
  Ultralytics copyright + AGPL-3.0 header on those files.
- Replace the final 1-channel depth output conv with a 3-channel RGB one.
- Wrap it so it matches ClearView's checkpoint convention: forward pass
  takes (B, 3, H, W) in [0, 1], returns (B, 3, H, W) in [0, 1], and
  `state_dict()` can be saved directly as {"model_state_dict": ...} so the
  existing checkpoint-stripping / model-card pipeline works unmodified.
- One class per variant (n/s/m/l/x), or one class with a `variant` kwarg,
  whichever matches how the vendored source is structured upstream.

Nothing implemented yet.
"""

import torch.nn as nn


class Yolo26RGB(nn.Module):
    """Placeholder. See module docstring for the plan."""

    def __init__(self, variant: str = "n", **kwargs):
        super().__init__()
        raise NotImplementedError(
            "yolo26-rgb model not implemented yet, this is scaffolding only"
        )
