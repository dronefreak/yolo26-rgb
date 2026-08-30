"""Minimal YAML -> nn.Module builder for yolo26-depth-style configs. Original code.

Trimmed from Ultralytics' `parse_model()` (originally in `nn/tasks.py`,
which dispatches across 30+ module types covering every YOLO task the
package supports). `yolo26-rgb.yaml` only ever uses 7: Conv, C3k2, SPPF,
C2PSA, Concat, nn.Upsample, and a dense head (`RGBHead`, or the original
vendored `Depth` this repo's config no longer uses but keeps around for
reference, see `_vendor/depth_head.py`). This file only knows about those.

The channel-scaling and repeat-count logic below is copied faithfully from
the relevant branches of the original `parse_model`, not reinvented, this
is the part most likely to silently produce a wrong-shaped model if
guessed instead of traced from source. The building blocks it assembles
(Conv/C3k2/SPPF/C2PSA/Concat) are vendored Ultralytics code, see
`_vendor/blocks.py`; `RGBHead` is original code, see `heads.py`, this file
is original too, just structured around the vendored modules' interface.
"""

from __future__ import annotations

import ast
import contextlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml as pyyaml

from ._vendor.blocks import C2PSA, Concat, Conv, SPPF, C3k2, make_divisible
from .heads import RGBHead

# Module name (as it appears in the YAML) -> class. Extend this when a new
# module type shows up in a config, don't add classes here that no config
# actually uses.
MODULE_MAP: dict[str, type] = {
    "Conv": Conv,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "C2PSA": C2PSA,
    "Concat": Concat,
    "RGBHead": RGBHead,
}

# Modules whose "repeats" count is passed in as a constructor arg (they
# handle the repetition internally) rather than being wrapped n times in
# an nn.Sequential by this parser. Matches Ultralytics' `repeat_modules`
# for the subset relevant here.
REPEAT_ARG_MODULES = frozenset({C3k2, C2PSA})


def load_yaml_config(path: str | Path) -> dict:
    """Load a yolo26-depth-style YAML config into a plain dict."""
    with open(path) as f:
        return pyyaml.safe_load(f)


def parse_model_lite(
    d: dict, ch: int, scale: str, verbose: bool = False
) -> tuple[nn.Sequential, list[int]]:
    """Build the backbone+neck+head from a parsed YAML dict.

    Args:
        d: parsed YAML (must have "backbone", "head", "scales" keys).
        ch: number of input channels (3 for RGB).
        scale: one of "n", "s", "m", "l", "x", selects (depth, width, max_channels)
            from d["scales"].
        verbose: print each constructed layer.

    Returns:
        (assembled nn.Sequential, sorted list of layer indices to cache
        outputs from for later skip-connection lookups)
    """
    depth, width, max_channels = d["scales"][scale]

    layers: list[nn.Module] = []
    save: list[int] = []
    ch_list = [ch]  # running per-layer output channel count, index 0 = input

    for i, (f, n, m_name, args) in enumerate(d["backbone"] + d["head"]):
        args = list(args)  # don't mutate the parsed yaml in place
        for j, a in enumerate(args):
            # YAML has no bareword null distinct from the string "None", a
            # bareword `None` in the config parses as the *string* "None",
            # not Python's None. Ultralytics' loader fixes this the same
            # way: run any string arg through literal_eval so "None" ->
            # None, "True" -> True, "1024" -> 1024, etc. Non-literal
            # strings (e.g. "nearest") just fail the eval and stay strings.
            if isinstance(a, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    args[j] = ast.literal_eval(a)

        if m_name.startswith("nn."):
            m = getattr(nn, m_name[3:])
        else:
            if m_name not in MODULE_MAP:
                raise ValueError(
                    f"Unknown module '{m_name}' in config, not in MODULE_MAP. "
                    "This parser only knows the module types yolo26-depth.yaml "
                    "actually uses, add it to MODULE_MAP in backbone.py if a new "
                    "config needs it (and add the class to _vendor/blocks.py or "
                    "heads.py)."
                )
            m = MODULE_MAP[m_name]

        n_scaled = max(round(n * depth), 1) if n > 1 else n

        if m in (Conv, C3k2, SPPF, C2PSA):
            c1 = ch_list[f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in REPEAT_ARG_MODULES:
                args.insert(2, n_scaled)  # repeats become a constructor arg
                n_scaled = 1
            if m is C3k2 and scale in "mlx":
                # Force C3k=True for M/L/X sizes, matches upstream exactly,
                # not just whatever the yaml literal says.
                args[3] = True
        elif m is Concat:
            c2 = sum(ch_list[x] for x in f)
        elif m is RGBHead:
            # First 3 `from` entries are the P3/P4/P5 fusion levels (same as
            # Depth), any further entries are finer-resolution skip
            # connections consumed by the tail, coarser first.
            args = [args[0], [ch_list[x] for x in f[:3]], [ch_list[x] for x in f[3:]]]
            # RGBHead has no output channel count in the graph sense, nothing
            # can legally follow it (it's always the terminal layer, and its
            # forward() returns an image tensor, not a feature map another
            # layer's `from` index could reference). 0 is an obviously-invalid
            # placeholder (real channel counts are never 0), set explicitly
            # rather than silently reusing whatever c2 was left over from the
            # previous iteration, which happened to be harmless only because
            # nothing ever reads this entry. Kept as `int` (not `int | None`)
            # so ch_list's element type stays plain `int` for every other
            # branch that legitimately reads it (Concat's channel sum, etc).
            c2 = 0
        else:
            # nn.Upsample and anything else with unchanged channel count
            c2 = ch_list[f] if isinstance(f, int) else ch_list[f[-1]]

        module = (
            nn.Sequential(*(m(*args) for _ in range(n_scaled)))
            if n_scaled > 1
            else m(*args)
        )
        # nn.Module.type() is a real method (dtype conversion); shadowing it with a plain
        # attribute here is deliberate and matches Ultralytics' own parse_model() exactly,
        # works fine at runtime (Python allows overwriting a bound method with an instance
        # attribute). No `type: ignore` needed here under this repo's pinned mypy (v1.8.0,
        # see .pre-commit-config.yaml) -- add one back only if a mypy upgrade starts
        # flagging this assignment again.
        module.i, module.f, module.type = i, f, m_name

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(module)

        if i == 0:
            ch_list = []
        ch_list.append(c2)

        if verbose:
            n_params = sum(p.numel() for p in module.parameters())
            print(
                f"{i:>3} {str(f):>12} {n_scaled:>3} {n_params:>10,}  {m_name:<12} {args}"
            )

    return nn.Sequential(*layers), sorted(set(save))


class YoloBackboneNeck(nn.Module):
    """Runs the assembled layer list, handling the skip-connection graph.

    Trimmed from `BaseModel._predict_once`, that method is already small
    and clean, kept close to verbatim. Everything else on `BaseModel`
    (fuse/info/load/loss/_profile_one_layer/_predict_augment, all the
    task-specific subclasses) isn't needed, we use ClearView's own
    training loop and loss instead.
    """

    def __init__(
        self, cfg_path: str | Path, scale: str, ch: int = 3, verbose: bool = False
    ):
        super().__init__()
        self.yaml = load_yaml_config(cfg_path)
        self.model, self.save = parse_model_lite(
            self.yaml, ch=ch, scale=scale, verbose=verbose
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `out` is deliberately loosely typed: mid-graph it's a Tensor for most layers but a
        # list[Tensor] going into Concat (multi-input `f`), unlike the public (x) -> Tensor
        # contract this method actually honors end to end.
        y: list[torch.Tensor | None] = []
        out: Any = x
        for m in self.model:
            mf: Any = (
                m.f
            )  # dynamically set in parse_model_lite (int | list[int]), not a
            # real nn.Module attribute -- Any sidesteps nn.Module.__getattr__'s generic
            # Tensor | Module stub, which doesn't know that.
            if mf != -1:
                out = (
                    y[mf]
                    if isinstance(mf, int)
                    else [out if j == -1 else y[j] for j in mf]
                )
            out = m(out)
            y.append(out if m.i in self.save else None)
        return out
