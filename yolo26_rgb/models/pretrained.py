"""Downloading and loading YOLO26 depth-pretrained backbone/neck weights into Yolo26RGB.

Original code. Three stages, kept as separate functions since they have different
dependencies and failure modes:

1. `download_depth_checkpoint`: fetch the raw, full Ultralytics checkpoint (a pickled
   `ultralytics.nn.tasks.DepthModel`) from Ultralytics' own GitHub release assets —
   the same file `ultralytics.YOLO("yolo26n-depth.pt")` auto-downloads. Stdlib-only
   (`urllib`), skips the download if the file already exists locally, matching that
   same download-if-missing behavior rather than re-downloading every run. These
   checkpoints aren't committed to this repo (`*.pt` is gitignored, and they're
   Ultralytics' own release assets, not this repo's to redistribute), so a fresh
   clone needs this step before anything else here works.
2. `extract_state_dict`: unpickle that raw checkpoint and save a plain,
   dependency-free `state_dict` .pt file, the format `load_pretrained_backbone` below
   actually consumes. Does **not** need `ultralytics` installed: the raw checkpoint is a
   pickled `ultralytics.nn.tasks.DepthModel`, but all we actually want out of it is
   tensor values, never any of `ultralytics`' own code running. `_StubUnpickler` below
   substitutes a generic `nn.Module` subclass for every class outside the standard
   library/torch/numpy, so the object graph reconstructs correctly (real `torch.nn.Module`
   methods — `__setstate__`, `state_dict()` — work directly off the restored `__dict__`,
   independent of which class the original object claimed to be) without ever importing
   `ultralytics`. Verified byte-for-byte identical against a real `ultralytics`-based
   extraction before relying on this.
3. `load_pretrained_backbone`: the part that actually touches a `Yolo26RGB` instance,
   documented in full below. `get_pretrained_state_dict` chains 1+2 for the common
   case of "just get me a loadable state_dict for this scale."

Checkpoint keys look like ``model.<i>.<submodule path>...`` (Ultralytics' `DepthModel.model`
is the same `nn.Sequential` this repo's `YoloBackboneNeck.model` mirrors, indexed the same
way since `yolo26-rgb.yaml`'s backbone/neck is unmodified from `yolo26-depth.yaml`, see
`_vendor/README.md`). `Yolo26RGB`'s corresponding keys are ``net.model.<i>....``, one prefix
deeper for the `Yolo26RGB` wrapper. Only layers before the head (the last entry in
`net.model`, index ``len(net.model) - 1``) are loaded — the head is structurally different
between `Depth` and `RGBHead` (see `heads.py`) and always starts randomly initialized.

`Yolo26RGB(weights="yolo26n-depth.pt")` (see `heads.py`) covers the common case by calling
`get_pretrained_state_dict` + `load_pretrained_backbone` internally. For lower-level control —
downloading a checkpoint without loading it yet, or loading into a model you already
constructed — use these directly:

    from yolo26_rgb.models.pretrained import get_pretrained_state_dict, load_pretrained_backbone
    from yolo26_rgb.models.heads import Yolo26RGB

    state_dict_path = get_pretrained_state_dict("n")  # downloads to ./pretrained/ if missing,
                                                        # skips if already there
    model = Yolo26RGB(scale="n")
    load_pretrained_backbone(model, state_dict_path)
"""

import pickle
import sys
import types
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    # Only for type-checking: heads.py already imports from this module at runtime, a real
    # top-level import here would be circular. Yolo26RGB isn't needed at runtime, only as a
    # more precise type than plain nn.Module for load_pretrained_backbone's `model` param
    # (it needs `.net.model`, which nn.Module's generic __getattr__ stub can't confirm exists).
    from .heads import Yolo26RGB

# Matches the URLs Ultralytics' own `YOLO("yolo26n-depth.pt")` auto-download uses
# (observed directly from a real download, not guessed):
# https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-depth.pt
_ASSETS_RELEASE = "v8.4.0"
_SCALES = ("n", "s", "m", "l", "x")


def _checkpoint_url(scale: str) -> str:
    if scale not in _SCALES:
        raise ValueError(f"Unknown scale {scale!r}, expected one of {_SCALES}")
    return f"https://github.com/ultralytics/assets/releases/download/{_ASSETS_RELEASE}/yolo26{scale}-depth.pt"


def download_depth_checkpoint(
    scale: str, dest_dir: str | Path = ".", verbose: bool = True
) -> Path:
    """Download yolo26{scale}-depth.pt from Ultralytics' GitHub release assets.

    Skips the download and returns the existing path if the file is already present
    (checked by filename only, no checksum), matching `YOLO("yolo26n-depth.pt")`'s own
    download-if-missing behavior — safe to call unconditionally before every use.

    Args:
        scale: one of "n", "s", "m", "l", "x".
        dest_dir: directory to save/look for the checkpoint in.
        verbose: print progress.

    Returns:
        Path to the local `.pt` file — the raw Ultralytics checkpoint, not a plain
        state_dict, see `extract_state_dict`.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"yolo26{scale}-depth.pt"

    if dest_path.exists():
        if verbose:
            print(f"{dest_path} already present, skipping download.", file=sys.stderr)
        return dest_path

    url = _checkpoint_url(scale)
    if verbose:
        print(f"Downloading {url} -> {dest_path}", file=sys.stderr)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if not verbose or total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        print(
            f"\r  {done / 1e6:.1f}/{total_size / 1e6:.1f} MB ({100 * done / total_size:.0f}%)",
            end="",
            flush=True,
            file=sys.stderr,
        )

    # Download to a temp name and rename on success, so a failed/interrupted download
    # never leaves a partial file sitting at the real path looking like a valid cache hit.
    tmp_path = dest_path.with_suffix(".pt.partial")
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
        tmp_path.rename(dest_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if verbose:
            print(file=sys.stderr)

    return dest_path


# `builtins`/`__builtin__` classes actually needed to unpickle a checkpoint's object graph
# (containers/types pickle itself may reference while reconstructing objects), deliberately
# not the full builtins namespace: that would also let a REDUCE opcode invoke real, dangerous
# callables (eval/exec/compile/__import__/open/...) straight from an untrusted checkpoint.
# Extend only if a legitimate checkpoint's extraction actually raises for a missing name here.
_SAFE_BUILTINS = frozenset(
    {
        "object",
        "set",
        "frozenset",
        "dict",
        "list",
        "tuple",
        "bytearray",
        "complex",
        "slice",
        "range",
        "type",
    }
)


class _StubUnpickler(pickle.Unpickler):
    """Substitutes a generic `nn.Module` subclass for any class outside the stdlib/torch/numpy.

    Lets a raw Ultralytics checkpoint unpickle without `ultralytics` installed. See module
    docstring for why this is safe here: we only ever read tensor values off the result via
    `state_dict()`, a real (imported) `torch.nn.Module` method that works directly off the
    object's restored `__dict__`, not anything specific to the original
    `ultralytics.nn.tasks.DepthModel` class. `builtins`/`__builtin__` are allowed through only
    for `_SAFE_BUILTINS`, not wholesale, so this doesn't just trade one code-execution surface
    (arbitrary `ultralytics` classes) for another (arbitrary real builtins reachable via a
    pickle REDUCE opcode, e.g. `eval`).
    """

    _cache: dict[tuple[str, str], type] = {}

    def find_class(self, module: str, name: str) -> Any:
        top = module.split(".")[0]
        if top in ("torch", "numpy", "collections"):
            return super().find_class(module, name)
        if top in ("builtins", "__builtin__"):
            if name not in _SAFE_BUILTINS:
                raise pickle.UnpicklingError(
                    f"Refusing to unpickle {module}.{name}: not in _SAFE_BUILTINS. "
                    "_StubUnpickler exists to avoid executing arbitrary code from an "
                    "untrusted checkpoint; if a legitimate checkpoint genuinely needs this "
                    "name, add it to _SAFE_BUILTINS after confirming it isn't a dangerous "
                    "callable (eval/exec/compile/__import__/open/...)."
                )
            return super().find_class(module, name)
        key = (module, name)
        cls = self._cache.get(key)
        if cls is None:
            cls = type(name, (nn.Module,), {"__module__": module})
            self._cache[key] = cls
        return cls


def _stub_pickle_module() -> types.ModuleType:
    """A minimal fake module exposing `.Unpickler`, for `torch.load(..., pickle_module=...)`."""
    mod = types.ModuleType("_yolo26_rgb_stub_pickle_module")
    mod.Unpickler = _StubUnpickler  # type: ignore[attr-defined]
    mod.load = pickle.load  # type: ignore[attr-defined]
    mod.loads = pickle.loads  # type: ignore[attr-defined]
    return mod


def extract_state_dict(
    checkpoint_path: str | Path, dest_path: str | Path | None = None
) -> Path:
    """Extract a plain, dependency-free state_dict from a raw Ultralytics depth checkpoint.

    Does not need `ultralytics` installed, see module docstring. Skips the extraction and
    returns the existing path if `dest_path` already exists.

    Args:
        checkpoint_path: path to the raw `yolo26{scale}-depth.pt`, e.g. from
            `download_depth_checkpoint`.
        dest_path: where to save the extracted state_dict. Defaults to
            `<checkpoint stem>-state-dict.pt` next to the checkpoint.

    Returns:
        Path to the saved plain state_dict `.pt` file.
    """
    checkpoint_path = Path(checkpoint_path)
    if dest_path is None:
        dest_path = checkpoint_path.with_name(f"{checkpoint_path.stem}-state-dict.pt")
    dest_path = Path(dest_path)

    if dest_path.exists():
        return dest_path

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_stub_pickle_module(),
    )
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = model.float().state_dict()
    torch.save(state_dict, dest_path)
    return dest_path


def get_pretrained_state_dict(
    scale: str, cache_dir: str | Path = "pretrained", verbose: bool = True
) -> Path:
    """Download (if needed) and extract (if needed) a scale's depth-pretrained state_dict.

    Chains `download_depth_checkpoint` + `extract_state_dict`; the returned path is
    directly usable as `load_pretrained_backbone`'s `state_dict_path` argument. Both
    steps individually skip their work if the target file already exists, so this is
    cheap to call unconditionally (e.g. from `Yolo26RGB(weights=...)` or a training
    script's `--pretrained` flag). No extra dependencies needed for either step.
    """
    raw_path = download_depth_checkpoint(scale, dest_dir=cache_dir, verbose=verbose)
    return extract_state_dict(raw_path)


def load_pretrained_backbone(
    model: "Yolo26RGB", state_dict_path: str | Path, verbose: bool = True
) -> dict[str, list[str]]:
    """Load a YOLO26 depth checkpoint's backbone/neck weights into a `Yolo26RGB` instance.

    Args:
        model: a `Yolo26RGB` instance (needs `.net.model`, an `nn.Sequential` of layers
            matching the yaml, head last).
        state_dict_path: path to a plain state_dict `.pt` file extracted from a released
            depth checkpoint, see module docstring.
        verbose: print a match/skip summary.

    Returns:
        dict with "matched"/"missing"/"unexpected" key lists, for verification, e.g. by a
        test asserting "matched" covers every backbone/neck tensor and nothing else.
    """
    ckpt_sd = torch.load(state_dict_path, map_location="cpu")

    n_layers = len(model.net.model)  # includes the head
    head_idx = n_layers - 1

    # "model.<i>...." -> "net.model.<i>...", dropping the head layer entirely.
    mapped = {}
    for k, v in ckpt_sd.items():
        layer_idx = int(k.split(".")[1])
        if layer_idx >= head_idx:
            continue
        mapped[f"net.{k}"] = v

    result = model.load_state_dict(mapped, strict=False)

    own_backbone_keys = {
        f"net.model.{i}.{suffix}"
        for i in range(head_idx)
        for suffix in model.net.model[i].state_dict().keys()
    }
    matched = own_backbone_keys - set(result.missing_keys)
    missing_backbone = own_backbone_keys - matched

    if verbose:
        print(
            f"Pretrained backbone load: {len(matched)}/{len(own_backbone_keys)} "
            f"backbone/neck tensors matched, head (layer {head_idx}) left random-init, "
            f"{len(result.unexpected_keys)} unexpected keys ignored.",
            file=sys.stderr,
        )
        if missing_backbone:
            print(
                f"  WARNING: backbone/neck tensors that did NOT match: {sorted(missing_backbone)}",
                file=sys.stderr,
            )

    return {
        "matched": sorted(matched),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }
