# Gym 2D Scripts

Scripts for gym_2d domain experiments and NTU comparison.

## Experiment Options

| Option | Description | Script |
|--------|-------------|--------|
| **A** | Zero-shot (frozen NTU model) | `eval_gym.py` |
| **B** | Fine-tuned (adapt to gym) | `finetune_gym.py` → `eval_gym.py` |

Comparing both shows: **what improves with adaptation** vs **what uncertainty still catches**.

## Scripts

| Script | Description |
|--------|-------------|
| `eval_gym.py` | MC dropout evaluation with proper preprocessing |
| `finetune_gym.py` | Head-only or full fine-tuning on gym data |
| `compare_domains.py` | NTU vs gym comparison (6 plots + safety table) |
| `gym_experiments.slurm` | Full A+B pipeline for Anvil |

## Quick Start (Local)

```bash
# Option A: Zero-shot eval
python scripts/gym_2d/eval_gym.py \
  --gym_data /path/to/gym_2d.pkl \
  --checkpoint outputs/ntu120_xsub_baseline/best.pt \
  --out_file outputs/uncertainty/gym_frozen_mc.npz

# Option B: Fine-tune then eval
python scripts/gym_2d/finetune_gym.py \
  --gym_data /path/to/gym_2d.pkl \
  --checkpoint outputs/ntu120_xsub_baseline/best.pt \
  --out_dir outputs/gym_finetuned \
  --mode head  # or 'full'

python scripts/gym_2d/eval_gym.py \
  --gym_data /path/to/gym_2d.pkl \
  --checkpoint outputs/gym_finetuned/best.pt \
  --out_file outputs/uncertainty/gym_finetuned_mc.npz

# Compare
python scripts/gym_2d/compare_domains.py \
  --ntu_mc outputs/uncertainty/ntu120_xsub_mc.npz \
  --gym_mc outputs/uncertainty/gym_frozen_mc.npz \
  --out_dir outputs/uncertainty/comparison
```

## Anvil (Full Pipeline)
```bash
sbatch scripts/gym_2d/eval_gym.slurm
```

Runs: Option A → Option B (fine-tune) → Both evals → Both comparisons.

## Fine-tuning Modes

| Mode | Description | Trainable Params |
|------|-------------|------------------|
| `--mode head` | Freeze backbone, train classifier only | ~1-5% |
| `--mode full` | Train all layers with low LR | 100% |


