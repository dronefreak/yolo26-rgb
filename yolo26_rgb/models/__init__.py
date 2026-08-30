"""Model registry for yolo26-rgb.

Mirrors ClearView's `get_model(name, **kwargs)` pattern (see
`clearview.models.get_model`) so tooling built around that convention
(checkpoint loading, the model-card generation pattern) transfers over,
even though this package is never imported by ClearView itself.
"""

from typing import Any

from .heads import Yolo26RGB

MODEL_REGISTRY: dict = {
    "yolo26_rgb_n": lambda **kwargs: Yolo26RGB(scale="n", **kwargs),
    "yolo26_rgb_s": lambda **kwargs: Yolo26RGB(scale="s", **kwargs),
    "yolo26_rgb_m": lambda **kwargs: Yolo26RGB(scale="m", **kwargs),
    "yolo26_rgb_l": lambda **kwargs: Yolo26RGB(scale="l", **kwargs),
    "yolo26_rgb_x": lambda **kwargs: Yolo26RGB(scale="x", **kwargs),
}


def get_model(name: str, **kwargs: Any) -> Yolo26RGB:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
