"""Plot local experiment losses against gradient steps and wall-clock time.

Example: uv run --with matplotlib python -m cs336_basics.plot_metrics runs/baseline runs/lr-test
"""

import argparse
import json
import math
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Run directories containing metrics.jsonl")
    parser.add_argument("--output", type=Path, default=Path("runs/loss_curves.png"))
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for run_dir in args.run_dirs:
        with (run_dir / "metrics.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        for axis, x_key in zip(axes, ("iteration", "elapsed_seconds")):
            for loss_key, style in (("train_loss", "-"), ("val_loss", "--")):
                points = [record for record in records if x_key in record and loss_key in record]
                if not points:
                    continue
                # Nonfinite losses create gaps instead of connecting across divergence.
                losses = [
                    record[loss_key] if record[loss_key] is not None else math.nan
                    for record in points
                ]
                axis.plot(
                    [record[x_key] for record in points], losses, style, marker=".",
                    label=f"{run_dir.name}: {loss_key}",
                )
    for axis, label in zip(axes, ("Gradient steps (completed updates)", "Wall-clock time (seconds)")):
        axis.set_xlabel(label)
        axis.set_ylabel("Cross-entropy loss (nats/token)")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend(fontsize="small")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved loss curves to {args.output}")


if __name__ == "__main__":
    main()
