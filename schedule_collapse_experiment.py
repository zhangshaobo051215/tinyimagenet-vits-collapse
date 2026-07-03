import argparse
import csv
import json
import math
import pickle
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timm.layers import RmsNorm
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

from dataset import ImageNetDataset


SCHEDULES = ("constant", "linear_up", "linear_down", "cosine_wave")


def parse_args():
    parser = argparse.ArgumentParser("Schedule-ratio collapse experiment with per-tensor norm control")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--schedules", nargs="+", default=list(SCHEDULES), choices=SCHEDULES)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--gamma-lr-mult", type=float, default=1.0)
    parser.add_argument("--head-lr-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=8192)
    parser.add_argument("--steps-per-run", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--control-start-step", type=int, default=5000)
    parser.add_argument("--decay-frac", type=float, default=0.1)
    parser.add_argument("--floor-lr-mult", type=float, default=0.1)
    parser.add_argument("--include-head-in-norm-control", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rmsnorm-eps", type=float, default=1e-6)
    parser.add_argument("--cooldown-frac", type=float, default=0.25)
    parser.add_argument("--ema-beta", type=float, default=0.98)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def schedule_ratio(name, progress):
    progress = min(max(progress, 0.0), 1.0)
    if name == "constant":
        return 1.0
    if name == "linear_up":
        return 1.0 + progress
    if name == "linear_down":
        return 1.0 - 0.5 * progress
    if name == "cosine_wave":
        return 1.0 + 0.5 * math.sin(4.0 * math.pi * progress)
    raise ValueError(f"Unknown schedule {name}")


def base_lr_multiplier(step, total_steps, warmup_steps, decay_frac, floor_lr_mult):
    iter_num = step - 1
    if warmup_steps > 0 and iter_num < warmup_steps:
        return (iter_num + 1) / warmup_steps

    decay_steps = max(1, int(round(total_steps * decay_frac)))
    decay_start = max(warmup_steps, total_steps - decay_steps)
    if iter_num < decay_start:
        return 1.0

    progress = min(1.0, (iter_num - decay_start + 1) / max(1, total_steps - decay_start))
    return floor_lr_mult + (1.0 - progress) * (1.0 - floor_lr_mult)


def load_train_dataset(data_root, img_size, max_samples, seed):
    data_root = Path(data_root)
    with open(data_root / "train_dataset.pkl", "rb") as f:
        train_data, train_labels = pickle.load(f)

    if max_samples is not None and max_samples < len(train_data):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_data), generator=generator)[:max_samples]
        train_data = train_data[indices]
        train_labels = train_labels[indices]

    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
    ])
    normalize = transforms.Compose([
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])
    return ImageNetDataset(
        train_data,
        train_labels.type(torch.LongTensor),
        transform=transform,
        normalize=normalize,
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


def create_rmsnorm_vit(model_name, img_size, rmsnorm_eps):
    return create_model(
        model_name,
        pretrained=False,
        num_classes=200,
        img_size=img_size,
        norm_layer=partial(RmsNorm, eps=rmsnorm_eps),
    )


def is_classifier_head(name):
    return name.startswith(("head", "head_dist", "fc", "classifier"))


def split_parameters(model, include_head_in_norm_control):
    matrix_params = []
    vector_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_classifier_head(name) and not include_head_in_norm_control:
            head_params.append(param)
        elif param.ndim >= 2:
            matrix_params.append(param)
        else:
            vector_params.append(param)
    return matrix_params, vector_params, head_params


def global_rms(params):
    if not params:
        return 0.0
    norm_sq = torch.zeros((), device=params[0].device)
    numel = 0
    with torch.no_grad():
        for param in params:
            value = param.detach().float()
            norm_sq += torch.sum(value * value)
            numel += value.numel()
    return math.sqrt(norm_sq.item() / max(numel, 1))


def capture_fro_norms(params):
    reference_norms = []
    with torch.no_grad():
        for param in params:
            reference_norms.append(torch.linalg.vector_norm(param.detach().float()).item())
    return reference_norms


def apply_per_tensor_fro_control(params, reference_norms, target_ratio):
    projected = 0
    skipped_zero = 0
    with torch.no_grad():
        for param, reference_norm in zip(params, reference_norms):
            current_norm = torch.linalg.vector_norm(param.detach().float()).item()
            if current_norm == 0.0:
                skipped_zero += 1
                continue
            target_norm = target_ratio * reference_norm
            param.mul_(target_norm / current_norm)
            projected += 1
    return global_rms(params)


def logits_from_output(output):
    if isinstance(output, (tuple, list)):
        return sum(output) / len(output)
    return output


def set_optimizer_lrs(optimizer, scheduled_lr, base_lr, head_lr_mult):
    for param_group in optimizer.param_groups:
        if param_group.get("lr_role") == "head":
            param_group["lr"] = base_lr * head_lr_mult
        else:
            param_group["lr"] = scheduled_lr


def run_one(args, dataset, base_state, schedule_name, device, writer, metrics_file):
    set_seed(args.seed)
    model = create_rmsnorm_vit(args.model, args.img_size, args.rmsnorm_eps)
    model.load_state_dict(base_state)
    model = model.to(device)
    model.train()

    matrix_params, vector_params, head_params = split_parameters(model, args.include_head_in_norm_control)
    initial_matrix_rms = global_rms(matrix_params)
    reference_fro_norms = None
    reference_matrix_rms = initial_matrix_rms

    if args.weight_decay != 0.0:
        raise ValueError("Per-tensor norm control replaces weight decay; set --weight-decay 0.0")

    optim_groups = [
        {
            "params": matrix_params,
            "lr": args.base_lr,
            "weight_decay": 0.0,
            "optimizer_name": "norm_control_adam",
            "lr_role": "scheduled",
        },
        {
            "params": vector_params,
            "lr": args.base_lr,
            "weight_decay": 0.0,
            "optimizer_name": "adam",
            "lr_role": "scheduled",
        },
    ]
    if head_params:
        optim_groups.append({
            "params": head_params,
            "lr": args.base_lr * args.head_lr_mult,
            "weight_decay": 0.0,
            "optimizer_name": "adam_head",
            "lr_role": "head",
        })
    optimizer = optim.Adam(optim_groups)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    amp_enabled = args.amp and device.type == "cuda"

    loader = make_loader(dataset, args.batch_size, args.num_workers, args.seed)
    total_steps = args.steps_per_run or len(loader)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    loss_ema = None
    start_time = time.time()
    iterator = tqdm(
        total=total_steps,
        desc=schedule_name,
        disable=args.no_progress,
        mininterval=5.0,
    )

    step = 0
    loader_iter = iter(loader)
    while step < total_steps:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        step += 1
        iter_num = step - 1
        active_control_steps = max(1, total_steps - args.control_start_step)
        if iter_num < args.control_start_step:
            control_active = False
            control_progress = 0.0
            ratio = 1.0
        else:
            control_active = True
            control_progress = min(1.0, (iter_num - args.control_start_step + 1) / active_control_steps)
            ratio = schedule_ratio(schedule_name, control_progress)
            if reference_fro_norms is None:
                reference_fro_norms = capture_fro_norms(matrix_params)
                reference_matrix_rms = global_rms(matrix_params)
        progress = step / total_steps
        base_lr_mult = base_lr_multiplier(
            step,
            total_steps,
            args.warmup_steps,
            args.decay_frac,
            args.floor_lr_mult,
        )
        base_lr = args.base_lr * base_lr_mult
        lr = base_lr * ratio
        head_lr = base_lr * args.head_lr_mult
        set_optimizer_lrs(optimizer, lr, base_lr, args.head_lr_mult)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = logits_from_output(model(x))
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if control_active:
            controlled_rms = apply_per_tensor_fro_control(matrix_params, reference_fro_norms, ratio)
        else:
            controlled_rms = global_rms(matrix_params)
        target_rms = reference_matrix_rms * ratio
        lr_over_norm = lr / max(controlled_rms, 1e-12)
        vector_rms = global_rms(vector_params)
        head_rms = global_rms(head_params)

        with torch.no_grad():
            loss_value = loss.item()
            if loss_ema is None:
                loss_ema = loss_value
            else:
                loss_ema = args.ema_beta * loss_ema + (1.0 - args.ema_beta) * loss_value
            logit_rms = torch.sqrt(torch.mean(logits.detach().float() ** 2)).item()

        writer.writerow({
            "schedule": schedule_name,
            "step": step,
            "progress": progress,
            "control_progress": control_progress,
            "control_active": int(control_active),
            "schedule_ratio": ratio,
            "cooldown": base_lr_mult,
            "matrix_lr": lr,
            "gamma_lr": lr,
            "head_lr": head_lr,
            "target_matrix_rms": target_rms,
            "matrix_rms": controlled_rms,
            "matrix_rms_ratio": controlled_rms / max(reference_matrix_rms, 1e-12),
            "lr_over_matrix_rms": lr_over_norm,
            "lr_over_matrix_rms_ratio": lr_over_norm / max(args.base_lr / reference_matrix_rms, 1e-12),
            "gamma_rms": vector_rms,
            "head_rms": head_rms,
            "loss": loss_value,
            "loss_ema": loss_ema,
            "logit_rms": logit_rms,
            "elapsed_sec": time.time() - start_time,
        })
        if args.flush_every <= 1 or step % args.flush_every == 0 or step == total_steps:
            metrics_file.flush()
        if args.print_every > 0 and (step % args.print_every == 0 or step == total_steps):
            print(
                f"{schedule_name} step {step}/{total_steps} "
                f"loss_ema={loss_ema:.4f} ratio={ratio:.4f}",
                flush=True,
            )
        if not args.no_progress:
            iterator.set_postfix(loss=f"{loss_ema:.3f}", ratio=f"{ratio:.2f}")
            iterator.update(1)

    iterator.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_train_dataset(args.data_root, args.img_size, args.max_train_samples, args.seed)
    set_seed(args.seed)
    base_model = create_rmsnorm_vit(args.model, args.img_size, args.rmsnorm_eps)
    base_state = {name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()}
    del base_model

    metrics_csv = output_dir / "metrics.csv"
    fieldnames = [
        "schedule",
        "step",
        "progress",
        "control_progress",
        "control_active",
        "schedule_ratio",
        "cooldown",
        "matrix_lr",
        "gamma_lr",
        "head_lr",
        "target_matrix_rms",
        "matrix_rms",
        "matrix_rms_ratio",
        "lr_over_matrix_rms",
        "lr_over_matrix_rms_ratio",
        "gamma_rms",
        "head_rms",
        "loss",
        "loss_ema",
        "logit_rms",
        "elapsed_sec",
    ]
    with open(metrics_csv, "w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        for schedule_name in args.schedules:
            run_one(args, dataset, base_state, schedule_name, device, writer, metrics_file)

    print(f"Wrote metrics to {metrics_csv}")


if __name__ == "__main__":
    main()
