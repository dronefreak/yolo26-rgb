"""Training entry point for yolo26-rgb.

TODO: not implemented. Intended to mirror `clearview.scripts.train`'s CLI
shape (--mix-config / --val-mix-config / --model / --epochs / --batch-size
etc.) so the same `configs/mix/rain_mixed_synthetic_real.yaml` recipe used
by every ClearView model can be reused here unmodified, for a fair
apples-to-apples comparison.

Reuses, once written:
- `clearview.data.datasets.MixedDataset` for the training mix
- `clearview.utils.metrics.compute_metrics` / `MetricsTracker` for the
  exact same PSNR/SSIM/MAE/MSE/Rain-Removal-Rate/NIQE definitions
  ClearView's own models are scored on
- `yolo26_rgb.models.get_model` for the model registry (see models/__init__.py)
"""


def main() -> None:
    raise NotImplementedError("yolo26-rgb training script not implemented yet")


if __name__ == "__main__":
    main()
