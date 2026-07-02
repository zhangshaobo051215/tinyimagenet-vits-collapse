import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Plot collapse experiment metrics")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def load_metrics(path):
    by_run = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_run.setdefault(row["run_id"], []).append(row)
    return by_run


def series(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def run_meta(run_id):
    norm, rest = run_id.split("_lr", 1)
    lr, wd = rest.split("_wd", 1)
    return norm, float(lr), float(wd)


def label_for(run_id):
    norm, lr, wd = run_meta(run_id)
    norm_label = "LN" if norm == "layernorm" else "RMS"
    return f"{norm_label} lr={lr:g} wd={wd:g}"


def style_for(run_id):
    norm, lr, wd = run_meta(run_id)
    color_map = {
        ("layernorm", 0.001): "#1f77b4",
        ("layernorm", 0.003): "#ff7f0e",
        ("rmsnorm", 0.001): "#2ca02c",
        ("rmsnorm", 0.003): "#d62728",
    }
    linestyle = "-" if wd == 0 else "--"
    return color_map.get((norm, lr), "#333333"), linestyle


def save_all_runs_plot(by_run, x_key, y_key, output_path, xlabel, ylabel, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for run_id in sorted(by_run):
        rows = by_run[run_id]
        color, linestyle = style_for(run_id)
        ax.plot(
            series(rows, x_key),
            series(rows, y_key),
            label=label_for(run_id),
            linewidth=1.8,
            color=color,
            linestyle=linestyle,
            alpha=0.9,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_same_lr_pair_grid(by_run, output_path):
    import matplotlib.pyplot as plt

    pairs = []
    keys = {}
    for run_id in sorted(by_run):
        norm, lr, wd = run_meta(run_id)
        keys[(norm, lr, wd)] = run_id
    for norm in ["layernorm", "rmsnorm"]:
        for lr in sorted({k[1] for k in keys if k[0] == norm}):
            a = keys.get((norm, lr, 0.0))
            candidates = [k for k in keys if k[0] == norm and k[1] == lr and k[2] != 0.0]
            if a and candidates:
                b = keys[candidates[0]]
                pairs.append((a, b))

    fig, axes = plt.subplots(len(pairs), 2, figsize=(11.5, 3.0 * len(pairs)), squeeze=False)
    for row_idx, (run_a, run_b) in enumerate(pairs):
        rows_a = by_run[run_a]
        rows_b = by_run[run_b]
        norm, lr, _ = run_meta(run_a)
        title_prefix = ("LayerNorm" if norm == "layernorm" else "RMSNorm") + f" lr={lr:g}: wd 0 vs wd {run_meta(run_b)[2]:g}"

        for ax, x_key, xlabel in [
            (axes[row_idx][0], "step", "step"),
            (axes[row_idx][1], "clock_lr_over_rms2", "sum lr / weight_rms^2"),
        ]:
            for run_id, rows in [(run_a, rows_a), (run_b, rows_b)]:
                color, linestyle = style_for(run_id)
                ax.plot(
                    series(rows, x_key),
                    series(rows, "loss_ema"),
                    label=label_for(run_id),
                    linewidth=1.9,
                    color=color,
                    linestyle=linestyle,
                )
            ax.set_title(f"{title_prefix} vs {xlabel}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("loss EMA")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_same_lr_pair_zoom(by_run, output_dir):
    import matplotlib.pyplot as plt

    keys = {}
    for run_id in sorted(by_run):
        norm, lr, wd = run_meta(run_id)
        keys[(norm, lr, wd)] = run_id

    saved = []
    for norm in ["layernorm", "rmsnorm"]:
        for lr in sorted({k[1] for k in keys if k[0] == norm}):
            run_a = keys.get((norm, lr, 0.0))
            candidates = [k for k in keys if k[0] == norm and k[1] == lr and k[2] != 0.0]
            if not run_a or not candidates:
                continue
            run_b = keys[candidates[0]]
            rows_a = by_run[run_a]
            rows_b = by_run[run_b]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for ax, x_key, xlabel in [
                (axes[0], "step", "step"),
                (axes[1], "clock_lr_over_rms2", "sum lr / weight_rms^2"),
            ]:
                for run_id, rows in [(run_a, rows_a), (run_b, rows_b)]:
                    color, linestyle = style_for(run_id)
                    ax.plot(
                        series(rows, x_key),
                        series(rows, "loss_ema"),
                        label=label_for(run_id),
                        linewidth=2.0,
                        color=color,
                        linestyle=linestyle,
                    )
                ax.set_xlabel(xlabel)
                ax.set_ylabel("loss EMA")
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8)
            title_norm = "LayerNorm" if norm == "layernorm" else "RMSNorm"
            fig.suptitle(f"{title_norm} lr={lr:g}: weight decay pair")
            fig.tight_layout()
            lr_tag = f"{lr:g}".replace(".", "p")
            filename = f"pair_{norm}_lr{lr_tag}_wd_overlay.png"
            out_path = output_dir / filename
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            saved.append(out_path)
    return saved


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    by_run = load_metrics(args.metrics)
    save_all_runs_plot(
        by_run,
        "step",
        "loss_ema",
        output_dir / "all_loss_vs_step.png",
        "step",
        "loss EMA",
        "All Runs: Loss vs Step",
    )
    save_all_runs_plot(
        by_run,
        "clock_lr_over_rms2",
        "loss_ema",
        output_dir / "all_loss_vs_lr_over_rms2_clock.png",
        "sum lr / weight_rms^2",
        "loss EMA",
        "All Runs: Loss vs Effective Clock",
    )
    save_all_runs_plot(
        by_run,
        "clock_update_angle",
        "loss_ema",
        output_dir / "all_loss_vs_update_angle_clock.png",
        "sum actual update angle",
        "loss EMA",
        "All Runs: Loss vs Actual Update-Angle Clock",
    )
    save_all_runs_plot(
        by_run,
        "step",
        "logit_rms",
        output_dir / "all_logit_rms_vs_step.png",
        "step",
        "logit RMS",
        "All Runs: Logit RMS vs Step",
    )
    save_same_lr_pair_grid(by_run, output_dir / "same_lr_wd_pairs_step_vs_clock.png")
    save_same_lr_pair_zoom(by_run, output_dir)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
