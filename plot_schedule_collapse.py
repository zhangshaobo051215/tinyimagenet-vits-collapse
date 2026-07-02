import argparse
import csv
from pathlib import Path

import numpy as np


STYLE = {
    "constant": ("black", "-"),
    "linear_up": ("#1f77b4", "-"),
    "linear_down": ("#d62728", "-"),
    "cosine_wave": ("#2ca02c", "-"),
}

LABEL = {
    "constant": "constant",
    "linear_up": "linear up 1->2",
    "linear_down": "linear down 1->0.5",
    "cosine_wave": "cosine wave",
}


def parse_args():
    parser = argparse.ArgumentParser("Plot schedule collapse")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def load_metrics(path):
    by_schedule = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_schedule.setdefault(row["schedule"], []).append(row)
    return by_schedule


def series(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_schedule = load_metrics(args.metrics)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    specs = [
        (axes[0, 0], "matrix_lr", "Learning rate", "optimizer step", "matrix LR"),
        (axes[0, 1], "matrix_rms_ratio", "Norm", "optimizer step", "matrix norm ratio"),
        (axes[1, 0], "lr_over_matrix_rms_ratio", "Learning rate / norm", "optimizer step", "LR/norm ratio"),
        (axes[1, 1], "loss_ema", "Training loss EMA", "optimizer step", "training loss EMA"),
    ]

    for ax, key, title, xlabel, ylabel in specs:
        for schedule in ["constant", "linear_up", "linear_down", "cosine_wave"]:
            if schedule not in by_schedule:
                continue
            rows = by_schedule[schedule]
            color, linestyle = STYLE[schedule]
            ax.plot(
                series(rows, "step"),
                series(rows, key),
                label=LABEL[schedule],
                color=color,
                linestyle=linestyle,
                linewidth=1.9,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Per-tensor Frobenius norm control, Adam, LR x norm schedule ratio")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
