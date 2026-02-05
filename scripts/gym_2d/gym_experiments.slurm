#!/bin/bash
#SBATCH -J gym_experiments
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --account=nairr250442-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

mkdir -p logs

REPO_DIR="$HOME/projects/gymbuddy-backend"
PY="/home/x-akhanal3/.conda/envs/2024.02-py311/poetry_env/bin/python"

cd "$REPO_DIR" || { echo "Repo not found: $REPO_DIR"; exit 1; }

export DATA_ROOT="/anvil/scratch/x-akhanal3/ai-gym-buddy/data/raw/skeleton"
export OUTPUT_DIR="/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs"

# === EDIT THIS PATH ===
GYM_DATA="/anvil/scratch/x-akhanal3/ai-gym-buddy/data/raw/gym_2d.pkl"

nvidia-smi
$PY -c "import torch; print('cuda?', torch.cuda.is_available())"

echo "=============================================="
echo "OPTION A: Zero-shot evaluation (frozen NTU model)"
echo "=============================================="
$PY scripts/gym_2d/evaluate_mc_dropout.py \
  --gym_data "$GYM_DATA" \
  --checkpoint "$OUTPUT_DIR/ntu120_xsub_baseline/best.pt" \
  --out_file "$OUTPUT_DIR/uncertainty/gym_frozen_mc.npz" \
  --n_passes 20

echo "=============================================="
echo "OPTION B: Fine-tune on gym (head-only)"
echo "=============================================="
$PY scripts/gym_2d/finetune_gym.py \
  --gym_data "$GYM_DATA" \
  --checkpoint "$OUTPUT_DIR/ntu120_xsub_baseline/best.pt" \
  --out_dir "$OUTPUT_DIR/gym_finetuned_head" \
  --mode head \
  --epochs 20 \
  --lr 1e-4

echo "=============================================="
echo "Evaluate fine-tuned model"
echo "=============================================="
$PY scripts/gym_2d/evaluate_mc_dropout.py \
  --gym_data "$GYM_DATA" \
  --checkpoint "$OUTPUT_DIR/gym_finetuned_head/best.pt" \
  --out_file "$OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz" \
  --n_passes 20

echo "=============================================="
echo "Gating sweep: NTU baseline (for comparison)"
echo "=============================================="
$PY scripts/run_gating_sweep.py \
  --mc_file "$OUTPUT_DIR/uncertainty/ntu120_xsub_mc.npz" \
  --out_csv "$OUTPUT_DIR/uncertainty/ntu_gating/gating.csv" \
  --out_fig "$OUTPUT_DIR/uncertainty/ntu_gating/risk_coverage.png" \
  --out_dir "$OUTPUT_DIR/uncertainty/ntu_gating" \
  --dataset_name ntu_baseline

echo "=============================================="
echo "Gating sweep: Gym frozen model"
echo "=============================================="
$PY scripts/gym_2d/run_gating_sweep.py \
  --mc_file "$OUTPUT_DIR/uncertainty/gym_frozen_mc.npz" \
  --out_dir "$OUTPUT_DIR/uncertainty/gym_frozen_gating" \
  --dataset_name gym_frozen

echo "=============================================="
echo "Gating sweep: fine-tuned model"
echo "=============================================="
$PY scripts/gym_2d/run_gating_sweep.py \
  --mc_file "$OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz" \
  --out_dir "$OUTPUT_DIR/uncertainty/gym_finetuned_gating" \
  --dataset_name gym_finetuned

echo "=============================================="
echo "Domain comparison: NTU vs Gym (frozen)"
echo "=============================================="
$PY scripts/gym_2d/compare_domains.py \
  --ntu_mc "$OUTPUT_DIR/uncertainty/ntu120_xsub_mc.npz" \
  --gym_mc "$OUTPUT_DIR/uncertainty/gym_frozen_mc.npz" \
  --out_dir "$OUTPUT_DIR/uncertainty/comparison_frozen" \
  --gym_label "Gym (Frozen)"

echo "=============================================="
echo "Domain comparison: NTU vs Gym (fine-tuned)"
echo "=============================================="
$PY scripts/gym_2d/compare_domains.py \
  --ntu_mc "$OUTPUT_DIR/uncertainty/ntu120_xsub_mc.npz" \
  --gym_mc "$OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz" \
  --out_dir "$OUTPUT_DIR/uncertainty/comparison_finetuned" \
  --gym_label "Gym (Fine-tuned)"

echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="


