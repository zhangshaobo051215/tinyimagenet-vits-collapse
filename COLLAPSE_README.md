# Tiny-ImageNet ViT Norm-Control Collapse Experiment

This repository is based on:

- Base code: https://github.com/ehuynh1106/TinyImageNet-Transformers
- Paper: https://arxiv.org/abs/2205.10660
- Norm-control reference implementation: https://github.com/zhes2Hen/temp_super_nano_wd

The main added experiment file is:

- `schedule_collapse_experiment.py`

## What Was Changed

The collapse script ports the norm-control idea from `temp_super_nano_wd` to a Tiny-ImageNet ViT setting:

- Matrix parameters are tensors with `param.ndim >= 2`.
- Vector parameters are tensors with `param.ndim < 2`.
- Optimizer is `torch.optim.Adam`.
- Weight decay is `0.0`.
- Every matrix tensor records its own reference Frobenius norm.
- After each Adam step, each matrix tensor is projected independently:

```text
W <- W * (target_ratio * reference_frobenius_norm / current_frobenius_norm)
```

The learning rate is matched to the norm schedule:

```text
lr = base_lr * norm_ratio * cooldown
```

## 20000-Step Run

The run used:

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

One important difference from the LLM configs is that this run used:

```text
control_start_step = 0
```

The LLM configs in `temp_super_nano_wd` often use:

```text
control_start_iter = 5000
```

meaning they first train normally, then capture the reference norms and start norm control later.
