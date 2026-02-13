#!/bin/bash
set -e

# ==============================================================================
# Local Execution Script for Gym 2D Experiments
# ==============================================================================
# Usage: ./run_local.sh
#
# Prerequisites:
# 1. Activate your python environment (conda/poetry)
# 2. Ensure data is in data/raw/skeleton/
# 3. Ensure trained model is in outputs/ntu120_xsub_baseline/best.pt
# ==============================================================================

# 1. Setup Paths
PROJECT_ROOT=$(pwd)
DATA_ROOT="$PROJECT_ROOT/data/raw/skeleton"
OUTPUT_DIR="$PROJECT_ROOT/project_results"
CHECKPOINT="$PROJECT_ROOT/outputs/ntu120_xsub_baseline/best.pt"
GYM_DATA="$DATA_ROOT/gym_2d.pkl"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/uncertainty"
mkdir -p "$OUTPUT_DIR/gym_finetune"

echo "=== Environment Setup ==="
echo "Project Root: $PROJECT_ROOT"
echo "Data Root:    $DATA_ROOT"
echo "Output Dir:   $OUTPUT_DIR"
echo "Checkpoint:   $CHECKPOINT"
echo "Gym Data:     $GYM_DATA"
echo "========================="

# 2. Validation
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    echo "   Please copy your 'best.pt' to that location or update this script."
    exit 1
fi

if [ ! -f "$GYM_DATA" ]; then
    echo "❌ Error: Gym data not found at $GYM_DATA"
    exit 1
fi

# 3. NTU Evaluation (Generate ntu120_xsub_mc.npz)
echo ""
echo "=== Step 1: NTU MC Evaluation ==="
NTU_MC_FILE="$OUTPUT_DIR/uncertainty/ntu120_xsub_mc.npz"
if [ -f "$NTU_MC_FILE" ]; then
    echo "✅ NTU MC output already exists. Skipping..."
else
    python3 scripts/eval_mc_dropout.py \
      --checkpoint "$CHECKPOINT" \
      --data_path "$DATA_ROOT/ntu120/ntu120_3d.pkl" \
      --split "xsub_val" \
      --n_passes 20 \
      --dropout 0.1 \
      --out_file "$NTU_MC_FILE"
fi

# 4. Gym Zero-Shot Evaluation
echo ""
echo "=== Step 2: Gym Zero-Shot Evaluation ==="
GYM_FROZEN_MC="$OUTPUT_DIR/uncertainty/gym_frozen_mc.npz"
python3 scripts/gym_2d/evaluate_mc_dropout.py \
  --gym_data "$GYM_DATA" \
  --checkpoint "$CHECKPOINT" \
  --out_file "$GYM_FROZEN_MC" \
  --n_passes 20

# 5. Gym Finetuning (Optional - slow on CPU)
echo ""
echo "=== Step 3: Gym Finetuning ==="
# Determine device for training command
python3 scripts/gym_2d/finetune_gym.py \
  --gym_data "$GYM_DATA" \
  --checkpoint "$CHECKPOINT" \
  --out_dir "$OUTPUT_DIR/gym_finetune" \
  --epochs 10 \
  --lr 1e-4

# 6. Gym Finetuned Evaluation
echo ""
echo "=== Step 4: Gym Finetuned Evaluation ==="
GYM_FT_CHECKPOINT="$OUTPUT_DIR/gym_finetune/best.pt"
GYM_FT_MC="$OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz"

if [ -f "$GYM_FT_CHECKPOINT" ]; then
    python3 scripts/gym_2d/evaluate_mc_dropout.py \
      --gym_data "$GYM_DATA" \
      --checkpoint "$GYM_FT_CHECKPOINT" \
      --out_file "$GYM_FT_MC" \
      --n_passes 20
else
    echo "⚠️ Finetuned checkpoint not found. Skipping Step 4."
fi

# 7. Analysis
echo ""
echo "=== Step 5: Analysis & Plots ==="
python3 scripts/gym_2d/compare_domains.py \
  --ntu_mc "$NTU_MC_FILE" \
  --gym_mc "$GYM_FROZEN_MC" \
  --out_dir "$OUTPUT_DIR"

python3 scripts/gym_2d/run_gating_sweep.py \
  --mc_file "$GYM_FROZEN_MC" \
  --out_dir "$OUTPUT_DIR/gating_frozen" \
  --dataset_name "Gym-Frozen"

python3 scripts/gym_2d/run_gating_sweep.py \
  --mc_file "$GYM_FT_MC" \
  --out_dir "$OUTPUT_DIR/gating_finetuned" \
  --dataset_name "Gym-Finetuned"

echo ""
echo "🎉 Pipeline Complete! Results in $OUTPUT_DIR"
