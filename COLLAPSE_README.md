# Tiny-ImageNet ViT Norm-Control Collapse Experiment

This repository is based on:

- Base code: https://github.com/ehuynh1106/TinyImageNet-Transformers
- Paper: https://arxiv.org/abs/2205.10660
- Norm-control reference implementation: https://github.com/zhes2Hen/temp_super_nano_wd

The main added experiment file is:

- `schedule_collapse_experiment.py`

## What Was Changed

The collapse script ports the norm-control idea from `temp_super_nano_wd` to a Tiny-ImageNet ViT setting:

- Matrix parameters are tensors with `param.ndim >= 2`, excluding classifier head parameters by default.
- Vector parameters are tensors with `param.ndim < 2`.
- Classifier head parameters are trained by ordinary Adam by default and are not norm-controlled. Pass `--include-head-in-norm-control` to recover the earlier behavior.
- Optimizer is `torch.optim.Adam`.
- Weight decay is `0.0`.
- Every matrix tensor records its own reference Frobenius norm.
- Default LR schedule now follows a WSD-style base schedule with `--warmup-steps 1000`, `--decay-frac 0.1`, and `--floor-lr-mult 0.1`.
- Default norm-control start is `--control-start-step 5000`, matching the delayed reference-capture style of `temp_super_nano_wd`.
- After each Adam step, each matrix tensor is projected independently:

```text
W <- W * (target_ratio * reference_frobenius_norm / current_frobenius_norm)
```

The learning rate is matched to the norm schedule:

```text
matrix_lr = base_wsd_lr * norm_ratio
head_lr   = base_wsd_lr
```

## Current Code Defaults

The current code defaults are intended to be closer to `temp_super_nano_wd`:

```text
warmup_steps = 1000
control_start_step = 5000
decay_frac = 0.1
floor_lr_mult = 0.1
classifier head excluded from norm control
```

## 20000-Step Run

The archived 20000-step run in `results/` was generated before the latest code update. It used the earlier settings below, especially `control_start_step = 0` and classifier head included in norm control. It is kept for reference only.

```powershell
python schedule_collapse_experiment.py `
  --data-root C:\path\to\tiny-imagenet-200 `
  --output-dir results\schedule_collapse_per_tensor_20000_steps_quiet `
  --model vit_tiny_patch16_224 `
  --img-size 64 `
  --schedules constant linear_up linear_down cosine_wave `
  --max-train-samples 8192 `
  --batch-size 64 `
  --steps-per-run 20000 `
  --base-lr 0.001 `
  --control-start-step 0 `
  --weight-decay 0.0 `
  --amp `
  --no-progress `
  --flush-every 100 `
  --print-every 1000
```

Results are in:

- `results/schedule_collapse_per_tensor_20000_steps_quiet/report.md`
- `results/schedule_collapse_per_tensor_20000_steps_quiet/schedule_collapse_grid.png`
- `results/schedule_collapse_per_tensor_20000_steps_quiet/schedule_collapse_grid_0_15000.png`
- `results/schedule_collapse_per_tensor_20000_steps_quiet/metrics.csv`

## Important Caveat

This is not the original LLM training code from `temp_super_nano_wd`. It is a ViT/Tiny-ImageNet adaptation.

The archived run used:

```text
control_start_step = 0
```

The current code default now follows the LLM configs more closely:

```text
control_start_iter = 5000
```

meaning they first train normally, then capture the reference norms and start norm control later.
