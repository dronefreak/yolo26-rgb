"""Evaluation entry point for yolo26-rgb.

TODO: not implemented. Intended to mirror `clearview.scripts.evaluate`'s
CLI shape and reuse `clearview.utils.metrics.compute_metrics` directly, so
results land in the same `results.json` shape ClearView's own
`runs/<model>/eval/<dataset>/results.json` files use, and slot straight
into the same model-card table generation already built for ClearView's
model zoo.

Should be run against the same 10 test sets ClearView evaluates against
(Rain100L/H, Test100/1200/2800, DDN-Data, SPA-Data, RealRain-1k-H/L,
AllWeather) for the comparison to mean anything.
"""


def main() -> None:
    raise NotImplementedError("yolo26-rgb evaluation script not implemented yet")


if __name__ == "__main__":
    main()
