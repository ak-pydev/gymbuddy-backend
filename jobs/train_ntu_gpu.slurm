#!/bin/bash
#SBATCH -J ntu120_xsub
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --account=nairr250442-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G


mkdir -p logs

REPO_DIR="/home/x-akhanal3/gymbuddy-backend"
PY="/home/x-akhanal3/.conda/envs/2024.02-py311/poetry_env/bin/python"

cd "$REPO_DIR" || { echo "Repo not found: $REPO_DIR"; exit 1; }

export DATA_ROOT="/anvil/scratch/x-akhanal3/ai-gym-buddy/data/raw/skeleton"
export OUTPUT_DIR="/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs"

nvidia-smi
$PY -c "import torch; print('cuda?', torch.cuda.is_available()); print('torch', torch.__version__)"

$PY scripts/train_full_ntu.py \
  --split xsub \
  --epochs 50 \
  --lr 3e-4 \
  --dropout 0.1 \

# -------------------------
# Evaluate MC Dropout
# -------------------------
echo "=== Running MC Evaluation ==="
UNC_DIR="${OUTPUT_DIR}/uncertainty"
mkdir -p "${UNC_DIR}"
MC_OUT="${UNC_DIR}/ntu120_xsub_mc.npz"

$PY scripts/eval_mc_dropout.py \
  --checkpoint "${OUTPUT_DIR}/ntu120_xsub_baseline/best.pt" \
  --split xsub_val \
  --n_passes 20 \
  --dropout 0.1 \
  --out_file "${MC_OUT}"

test -f "${MC_OUT}" || { echo "MC output not found: ${MC_OUT}"; exit 1; }


