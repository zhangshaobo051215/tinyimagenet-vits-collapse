# Per-Tensor Norm-Control Schedule Collapse Run, 20000 Steps

## Setup

- Model: `vit_tiny_patch16_224` with RMSNorm.
- Dataset: Tiny-ImageNet training subset, `8192` samples.
- Optimizer: `torch.optim.Adam`.
- Weight decay: `0.0`.
- Norm control: per-matrix-tensor Frobenius projection after each Adam step.
- Run length: `20000` optimizer steps per schedule.
- Schedules: `constant`, `linear_up`, `linear_down`, `cosine_wave`.
- Output metrics: `metrics.csv`.

## Plots

- Full run: `schedule_collapse_grid.png`.
- Zoomed pre-cooldown view, 0-15000 steps: `schedule_collapse_grid_0_15000.png`.

## Final Loss EMA

| schedule | final loss EMA | final matrix norm ratio |
|---|---:|---:|
| constant | 0.000743 | 1.000000 |
| linear_up | 0.000082 | 2.000000 |
| linear_down | 0.008013 | 0.500000 |
| cosine_wave | 0.002759 | 1.000000 |

## Loss EMA Difference vs Constant

| schedule | MAE all steps | MAE 0-15000 | max abs all |
|---|---:|---:|---:|
| linear_up | 0.196699 | 0.238964 | 0.907218 |
| linear_down | 0.223425 | 0.249796 | 1.235857 |
| cosine_wave | 0.430971 | 0.483770 | 1.520954 |

## Sanity Check

- Max norm-schedule mismatch is below `3e-7`.
- Max nonzero-cooldown LR/norm mismatch is below `3e-7`.

Conclusion: the mechanical `LR / norm` collapse is correct, but the Tiny-ImageNet ViT training loss curves do not fully collapse. The final losses all approach zero because this 8192-sample subset is effectively memorized by 20000 steps, while mid-training trajectories still differ noticeably, especially for `cosine_wave`.
