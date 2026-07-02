import argparse
import csv
import itertools
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from timm.layers import RmsNorm
from timm.models import create_model
from timm.utils import accuracy
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

from dataset import ImageNetDataset


def parse_args():
    parser = argparse.ArgumentParser("Tiny-ImageNet ViT collapse experiment")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--norm-layers", nargs="+", default=["layernorm"], choices=["layernorm", "rmsnorm"])
    parser.add_argument("--rmsnorm-eps", type=float, default=1e-6)
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3])
    parser.add_argument("--wds", nargs="+", type=float, default=[0.0, 0.05])
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smooth", type=float, default=0.0)
    parser.add_argument("--randaug-magnitude", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--include-head-in-clock", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="Write matplotlib plots after the run")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_norm_layer(norm_layer, eps):
    if norm_layer == "rmsnorm":
        return partial(RmsNorm, eps=eps)
    return None


def load_train_dataset(data_root, img_size, max_samples, seed, randaug_magnitude):
    data_root = Path(data_root)
    with open(data_root / "train_dataset.pkl", "rb") as f:
        train_data, train_labels = pickle.load(f)

    if max_samples is not None and max_samples < len(train_data):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_data), generator=generator)[:max_samples]
        train_data = train_data[indices]
        train_labels = train_labels[indices]

    transform_steps = [
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
    ]
    if randaug_magnitude > 0:
        transform_steps.append(transforms.RandAugment(num_ops=2, magnitude=randaug_magnitude))

    return ImageNetDataset(
        train_data,
        train_labels.type(torch.LongTensor),
        transform=transforms.Compose(transform_steps),
        normalize=transforms.Compose([
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ]),
    )


def make_loader(dataset, batch_size, num_workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        drop_last=False,
    )


def create_vit_model(model_name, img_size, norm_layer, rmsnorm_eps):
    kwargs = {
        "pretrained": False,
        "num_classes": 200,
        "img_size": img_size,
    }
    norm = build_norm_layer(norm_layer, rmsnorm_eps)
    if norm is not None:
        kwargs["norm_layer"] = norm
    return create_model(model_name, **kwargs)


def logits_from_output(output):
    if isinstance(output, (tuple, list)):
        return sum(output) / len(output)
    return output


def clock_parameters(model, include_head):
    for name, param in model.named_parameters():
        if not param.requires_grad or param.ndim < 2:
            continue
        if not include_head and (name.startswith("head") or name.startswith("head_dist")):
            continue
        yield name, param


def param_norm_stats(params):
    norm_sq = torch.zeros((), device=params[0][1].device)
    numel = 0
    with torch.no_grad():
        for _, param in params:
            tensor = param.detach().float()
            norm_sq += torch.sum(tensor * tensor)
            numel += tensor.numel()
    norm = torch.sqrt(norm_sq).item()
    rms = math.sqrt(norm_sq.item() / max(numel, 1))
    return norm, rms, numel


def snapshot_params(params):
    return [param.detach().float().clone() for _, param in params]


def update_angle_stats(before, params, eps=1e-12):
    before_norm_sq = torch.zeros((), device=params[0][1].device)
    after_norm_sq = torch.zeros((), device=params[0][1].device)
    dot = torch.zeros((), device=params[0][1].device)
    delta_sq = torch.zeros((), device=params[0][1].device)
    with torch.no_grad():
        for before_tensor, (_, param) in zip(before, params):
            after = param.detach().float()
            before_norm_sq += torch.sum(before_tensor * before_tensor)
            after_norm_sq += torch.sum(after * after)
            dot += torch.sum(before_tensor * after)
            delta = after - before_tensor
            delta_sq += torch.sum(delta * delta)

    before_norm = torch.sqrt(before_norm_sq + eps)
    after_norm = torch.sqrt(after_norm_sq + eps)
    delta_norm = torch.sqrt(delta_sq + eps)
    update_rel_norm = (delta_norm / (before_norm + eps)).item()

    denom = before_norm * after_norm + eps
    cos = torch.clamp(dot / denom, -1.0, 1.0)
    angle = torch.acos(cos).item()
    sin_angle = torch.sqrt(torch.clamp(1.0 - cos * cos, min=0.0)).item()
    return update_rel_norm, angle, sin_angle


def make_optimizer(args, model, lr, wd):
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=args.momentum, nesterov=True)


def make_scheduler(args, optimizer, total_steps):
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    return None


def run_one(args, dataset, base_states, norm_layer, lr, wd, device, metrics_writer, metrics_file):
    run_id = f"{norm_layer}_lr{lr:g}_wd{wd:g}"
    set_seed(args.seed)
    model = create_vit_model(args.model, args.img_size, norm_layer, args.rmsnorm_eps)
    model.load_state_dict(base_states[norm_layer])
    model = model.to(device)
    model.train()

    loader = make_loader(dataset, args.batch_size, args.num_workers, args.seed)
    total_steps = len(loader) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    optimizer = make_optimizer(args, model, lr, wd)
    scheduler = make_scheduler(args, optimizer, total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    amp_enabled = args.amp and device.type == "cuda"

    clock_params = list(clock_parameters(model, args.include_head_in_clock))
    if not clock_params:
        raise ValueError("No matrix-like trainable parameters were found for the clock metrics.")

    loss_ema = None
    cumulative_lr_over_rms2 = 0.0
    cumulative_update_angle = 0.0
    cumulative_update_rel_norm = 0.0
    global_step = 0
    start_time = time.time()

    iterator = tqdm(total=total_steps, desc=run_id)
    for epoch in range(args.epochs):
        for batch_idx, (x, y) in enumerate(loader):
            if args.max_steps is not None and global_step >= args.max_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            lr_now = optimizer.param_groups[0]["lr"]
            weight_norm, weight_rms, clock_numel = param_norm_stats(clock_params)
            lr_over_weight_rms = lr_now / max(weight_rms, 1e-12)
            lr_over_weight_rms2 = lr_now / max(weight_rms * weight_rms, 1e-12)
            cumulative_lr_over_rms2 += lr_over_weight_rms2
            before = snapshot_params(clock_params)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = logits_from_output(model(x))
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            update_rel_norm, update_angle, update_sin = update_angle_stats(before, clock_params)
            cumulative_update_angle += update_angle
            cumulative_update_rel_norm += update_rel_norm

            with torch.no_grad():
                loss_value = loss.item()
                if loss_ema is None:
                    loss_ema = loss_value
                else:
                    loss_ema = loss_ema * 0.98 + loss_value * 0.02
                acc1, acc5 = accuracy(logits, y, topk=(1, 5))
                logit_rms = torch.sqrt(torch.mean(logits.detach().float() ** 2)).item()

            global_step += 1
            if global_step % args.log_every == 0 or global_step == total_steps:
                metrics_writer.writerow({
                    "run_id": run_id,
                    "norm_layer": norm_layer,
                    "model": args.model,
                    "epoch": epoch,
                    "step": global_step,
                    "lr": lr_now,
                    "weight_decay": wd,
                    "loss": loss_value,
                    "loss_ema": loss_ema,
                    "acc1": acc1.item(),
                    "acc5": acc5.item(),
                    "logit_rms": logit_rms,
                    "weight_norm": weight_norm,
                    "weight_rms": weight_rms,
                    "clock_numel": clock_numel,
                    "lr_over_weight_rms": lr_over_weight_rms,
                    "lr_over_weight_rms2": lr_over_weight_rms2,
                    "clock_lr_over_rms2": cumulative_lr_over_rms2,
                    "update_rel_norm": update_rel_norm,
                    "update_angle": update_angle,
                    "update_sin": update_sin,
                    "clock_update_angle": cumulative_update_angle,
                    "clock_update_rel_norm": cumulative_update_rel_norm,
                    "elapsed_sec": time.time() - start_time,
                })
                metrics_file.flush()

            iterator.set_postfix(loss=f"{loss_ema:.3f}", clock=f"{cumulative_lr_over_rms2:.2e}")
            iterator.update(1)

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    iterator.close()
    return run_id


def load_metrics(metrics_csv):
    by_run = {}
    with open(metrics_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_run.setdefault(row["run_id"], []).append(row)
    return by_run


def numeric_series(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def pairwise_alignment(metrics_csv, output_dir):
    by_run = load_metrics(metrics_csv)
    summaries = []
    for run_a, run_b in itertools.combinations(sorted(by_run), 2):
        rows_a, rows_b = by_run[run_a], by_run[run_b]
        n = min(len(rows_a), len(rows_b))
        if n == 0:
            continue

        loss_a = numeric_series(rows_a[:n], "loss_ema")
        loss_b = numeric_series(rows_b[:n], "loss_ema")
        step_mae = float(np.mean(np.abs(loss_a - loss_b)))

        clock_mae = np.nan
        update_clock_mae = np.nan
        for x_key, target in [
            ("clock_lr_over_rms2", "clock_mae"),
            ("clock_update_angle", "update_clock_mae"),
        ]:
            x_a = numeric_series(rows_a, x_key)
            x_b = numeric_series(rows_b, x_key)
            y_a = numeric_series(rows_a, "loss_ema")
            y_b = numeric_series(rows_b, "loss_ema")
            lo = max(float(np.min(x_a)), float(np.min(x_b)))
            hi = min(float(np.max(x_a)), float(np.max(x_b)))
            if hi > lo:
                grid = np.linspace(lo, hi, 100)
                aligned = float(np.mean(np.abs(np.interp(grid, x_a, y_a) - np.interp(grid, x_b, y_b))))
                if target == "clock_mae":
                    clock_mae = aligned
                else:
                    update_clock_mae = aligned

        summaries.append({
            "run_a": run_a,
            "run_b": run_b,
            "step_loss_ema_mae": step_mae,
            "clock_lr_over_rms2_loss_ema_mae": clock_mae,
            "clock_update_angle_loss_ema_mae": update_clock_mae,
            "n_step_points": n,
        })

    out_path = Path(output_dir) / "pairwise_alignment.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = [
            "run_a",
            "run_b",
            "step_loss_ema_mae",
            "clock_lr_over_rms2_loss_ema_mae",
            "clock_update_angle_loss_ema_mae",
            "n_step_points",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def maybe_plot(metrics_csv, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_error": repr(exc)}

    by_run = load_metrics(metrics_csv)
    plot_specs = [
        ("step", "loss_ema", "loss_vs_step.png", "step", "loss EMA"),
        ("clock_lr_over_rms2", "loss_ema", "loss_vs_lr_over_rms2_clock.png", "sum lr / weight_rms^2", "loss EMA"),
        ("clock_update_angle", "loss_ema", "loss_vs_update_angle_clock.png", "sum update angle", "loss EMA"),
        ("step", "logit_rms", "logit_rms_vs_step.png", "step", "logit RMS"),
    ]

    saved = []
    for x_key, y_key, filename, xlabel, ylabel in plot_specs:
        plt.figure(figsize=(8, 5))
        for run_id, rows in sorted(by_run.items()):
            x = numeric_series(rows, x_key)
            y = numeric_series(rows, y_key)
            plt.plot(x, y, label=run_id, linewidth=1.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_path = Path(output_dir) / filename
        plt.savefig(out_path, dpi=160)
        plt.close()
        saved.append(str(out_path))

    return {"plots": saved}


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_train_dataset(
        args.data_root,
        args.img_size,
        args.max_train_samples,
        args.seed,
        args.randaug_magnitude,
    )

    base_states = {}
    for norm_layer in args.norm_layers:
        set_seed(args.seed)
        base_model = create_vit_model(args.model, args.img_size, norm_layer, args.rmsnorm_eps)
        base_states[norm_layer] = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        del base_model

    metrics_csv = output_dir / "metrics.csv"
    fieldnames = [
        "run_id",
        "norm_layer",
        "model",
        "epoch",
        "step",
        "lr",
        "weight_decay",
        "loss",
        "loss_ema",
        "acc1",
        "acc5",
        "logit_rms",
        "weight_norm",
        "weight_rms",
        "clock_numel",
        "lr_over_weight_rms",
        "lr_over_weight_rms2",
        "clock_lr_over_rms2",
        "update_rel_norm",
        "update_angle",
        "update_sin",
        "clock_update_angle",
        "clock_update_rel_norm",
        "elapsed_sec",
    ]

    run_ids = []
    with open(metrics_csv, "w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        for norm_layer, lr, wd in itertools.product(args.norm_layers, args.lrs, args.wds):
            run_id = run_one(args, dataset, base_states, norm_layer, lr, wd, device, writer, metrics_file)
            run_ids.append(run_id)

    summaries = pairwise_alignment(metrics_csv, output_dir)
    plot_status = maybe_plot(metrics_csv, output_dir) if args.plot else {"plots": [], "skipped": True}
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "device": str(device),
            "num_samples": len(dataset),
            "runs": run_ids,
            "pairwise_alignment": summaries,
            "plot_status": plot_status,
        }, f, indent=2)

    print(f"Wrote metrics to {metrics_csv}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
