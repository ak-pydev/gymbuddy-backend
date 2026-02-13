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

set -euo pipefail
mkdir -p logs

die () { echo "❌ $1"; exit 1; }

# -------------------------
# PATHS (FIXED)
# -------------------------
REPO_DIR="$HOME/gymbuddy-backend"   # ✅ not $HOME/projects/...
PY="/home/x-akhanal3/.conda/envs/2024.02-py311/poetry_env/bin/python"

cd "$REPO_DIR" || die "Repo not found: $REPO_DIR"
test -x "$PY" || die "Python not executable: $PY"

export DATA_ROOT="/anvil/scratch/x-akhanal3/ai-gym-buddy/data/raw/skeleton"
export OUTPUT_DIR="/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs"

# ✅ correct gym path
GYM_DATA="$DATA_ROOT/gym/gym_2d.pkl"

# -------------------------
# SANITY CHECKS
# -------------------------
test -f "$GYM_DATA" || die "gym_2d.pkl not found: $GYM_DATA"
test -f "$OUTPUT_DIR/ntu120_xsub_baseline/best.pt" || die "NTU checkpoint missing: $OUTPUT_DIR/ntu120_xsub_baseline/best.pt"

# Auto-generate NTU MC output if missing
NTU_MC_FILE="$OUTPUT_DIR/uncertainty/ntu120_xsub_mc.npz"
if [ ! -f "$NTU_MC_FILE" ]; then
    echo "⚠️ NTU MC file missing. Generating it now..."
    mkdir -p "$OUTPUT_DIR/uncertainty"
    
    $PY scripts/eval_mc_dropout.py \
      --checkpoint "$OUTPUT_DIR/ntu120_xsub_baseline/best.pt" \
      --split xsub_val \
      --n_passes 20 \
      --dropout 0.1 \
      --out_file "$NTU_MC_FILE" || die "Failed to generate NTU MC output"
fi

test -f "$NTU_MC_FILE" || die "NTU MC file missing: $NTU_MC_FILE"

# scripts (catch typos early)
test -f scripts/gym_2d/eval_gym.py || die "Missing: scripts/gym_2d/eval_gym.py"
test -f scripts/gym_2d/finetune_gym.py || die "Missing: scripts/gym_2d/finetune_gym.py"
test -f scripts/gym_2d/run_gating_sweep.py || die "Missing: scripts/gym_2d/run_gating_sweep.py"
test -f scripts/gym_2d/compare_domains.py || die "Missing: scripts/gym_2d/compare_domains.py"

# paper polish scripts (optional, but verify if you call them)
test -f scripts/gym_2d/plot_training_curves.py || die "Missing: scripts/gym_2d/plot_training_curves.py"
test -f scripts/gym_2d/benchmark_latency.py || die "Missing: scripts/gym_2d/benchmark_latency.py"
test -f scripts/gym_2d/analyze_per_class.py || die "Missing: scripts/gym_2d/analyze_per_class.py"
test -f scripts/gym_2d/qualitative_analysis.py || die "Missing: scripts/gym_2d/qualitative_analysis.py"
test -f scripts/gym_2d/run_ablation.py || die "Missing: scripts/gym_2d/run_ablation.py"

# output dirs
mkdir -p "$OUTPUT_DIR/uncertainty"
mkdir -p "$OUTPUT_DIR/uncertainty/ntu_gating" \
         "$OUTPUT_DIR/uncertainty/gym_frozen_gating" \
         "$OUTPUT_DIR/uncertainty/gym_finetuned_gating" \
         "$OUTPUT_DIR/uncertainty/comparison_frozen" \
         "$OUTPUT_DIR/uncertainty/comparison_finetuned" \
         "$OUTPUT_DIR/ablation"

# -------------------------
# GPU CHECK
# -------------------------
nvidia-smi
$PY -c "import torch; print('cuda?', torch.cuda.is_available(), '| torch:', torch.__version__)"

# ==========================================================
# OPTION A: Zero-shot evaluation (frozen NTU model)
# ==========================================================
echo "=============================================="
echo "OPTION A: Zero-shot evaluation (frozen NTU model)"
echo "=============================================="
$PY scripts/gym_2d/eval_gym.py \
  --gym_data "$GYM_DATA" \
  --checkpoint "$OUTPUT_DIR/ntu120_xsub_baseline/best.pt" \
  --out_file "$OUTPUT_DIR/uncertainty/gym_frozen_mc.npz" \
  --n_passes 20

test -f "$OUTPUT_DIR/uncertainty/gym_frozen_mc.npz" || die "Missing: $OUTPUT_DIR/uncertainty/gym_frozen_mc.npz"

# ==========================================================
# OPTION B: Fine-tune on gym (weighted + smoothing + patience)
# ==========================================================
echo "=============================================="
echo "OPTION B: Fine-tune on gym (weighted + smoothing + patience)"
echo "=============================================="
CHECKPOINT="$OUTPUT_DIR/ntu120_xsub_baseline/best.pt"
GYM_FT_DIR="$OUTPUT_DIR/gym_finetuned_weighted"
mkdir -p "$GYM_FT_DIR"

$PY scripts/gym_2d/finetune_gym.py \
  --gym_data "$GYM_DATA" \
  --checkpoint "$CHECKPOINT" \
  --out_dir "$GYM_FT_DIR" \
  --weighted_loss \
  --label_smoothing 0.1 \
  --epochs 50 \
  --patience 10

test -f "$GYM_FT_DIR/best.pt" || die "Missing fine-tuned checkpoint: $GYM_FT_DIR/best.pt"

# ==========================================================
# Evaluate fine-tuned model (MC Dropout)
# ==========================================================
echo "=============================================="
echo "Evaluate fine-tuned model"
echo "=============================================="
$PY scripts/gym_2d/eval_gym.py \
  --gym_data "$GYM_DATA" \
  --checkpoint "$GYM_FT_DIR/best.pt" \
  --out_file "$OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz" \
  --n_passes 20

test -f "$OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz" || die "Missing: $OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz"

# ==========================================================
# Gating sweeps (NOTE: requires out_csv/out_fig for your script)
# ==========================================================
echo "=============================================="
echo "Gating sweep: NTU baseline (for comparison)"
echo "=============================================="
$PY scripts/run_gating_sweep.py \
  --mc_file "$OUTPUT_DIR/uncertainty/ntu120_xsub_mc.npz" \
  --out_csv "$OUTPUT_DIR/uncertainty/ntu_gating/gating.csv" \
  --out_fig "$OUTPUT_DIR/uncertainty/ntu_gating/risk_coverage.png"

echo "=============================================="
echo "Gating sweep: Gym frozen model"
echo "=============================================="
$PY scripts/gym_2d/run_gating_sweep.py \
  --mc_file "$OUTPUT_DIR/uncertainty/gym_frozen_mc.npz" \
  --out_csv "$OUTPUT_DIR/uncertainty/gym_frozen_gating/gating.csv" \
  --out_fig "$OUTPUT_DIR/uncertainty/gym_frozen_gating/risk_coverage.png"

echo "=============================================="
echo "Gating sweep: Gym fine-tuned model"
echo "=============================================="
$PY scripts/gym_2d/run_gating_sweep.py \
  --mc_file "$OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz" \
  --out_csv "$OUTPUT_DIR/uncertainty/gym_finetuned_gating/gating.csv" \
  --out_fig "$OUTPUT_DIR/uncertainty/gym_finetuned_gating/risk_coverage.png"

# ==========================================================
# Domain comparisons
# ==========================================================
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

# ==========================================================
# PAPER POLISH
# ==========================================================
echo "=============================================="
echo "PAPER POLISH: Training Curves"
echo "=============================================="
$PY scripts/gym_2d/plot_training_curves.py --outputs_dir "$OUTPUT_DIR"

echo "=============================================="
echo "PAPER POLISH: Latency Benchmark"
echo "=============================================="
$PY scripts/gym_2d/benchmark_latency.py --outputs_dir "$OUTPUT_DIR"

echo "=============================================="
echo "PAPER POLISH: Per-Class Analysis (Gym)"
echo "=============================================="
$PY scripts/gym_2d/analyze_per_class.py \
  --mc_file "$OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz" \
  --out_dir "$OUTPUT_DIR" \
  --dataset_name "Gym_Finetuned"

echo "=============================================="
echo "PAPER POLISH: Qualitative Analysis"
echo "=============================================="
$PY scripts/gym_2d/qualitative_analysis.py \
  --mc_file "$OUTPUT_DIR/uncertainty/gym_finetuned_mc.npz" \
  --gym_data "$GYM_DATA" \
  --out_dir "$OUTPUT_DIR"

echo "=============================================="
echo "PAPER POLISH: Ablation Study"
echo "=============================================="
$PY scripts/gym_2d/run_ablation.py \
  --checkpoint "$GYM_FT_DIR/best.pt" \
  --data_path "$GYM_DATA" \
  --out_dir "$OUTPUT_DIR/ablation" \
  --mc_passes "5,10,15,20"

echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Artifacts saved under: $OUTPUT_DIR"
