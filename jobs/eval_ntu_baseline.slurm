#!/bin/bash
#SBATCH -J ntu_mc_eval
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --account=nairr250442-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
mkdir -p logs

die () { echo "❌ $1"; exit 1; }

# -------------------------
# Canonical locations
# -------------------------
USER_NAME="x-akhanal3"
SCRATCH_ROOT="/anvil/scratch/${USER_NAME}"

REPO_DIR="$HOME/gymbuddy-backend"
PY="/home/x-akhanal3/.conda/envs/2024.02-py311/poetry_env/bin/python"

cd "${REPO_DIR}" || die "Repo dir not found: ${REPO_DIR}"
test -x "${PY}" || die "Python not found at ${PY}"

# -------------------------
# Needed by NTU120Dataset
# -------------------------
export DATA_ROOT="${SCRATCH_ROOT}/ai-gym-buddy/data/raw/skeleton"
test -d "${DATA_ROOT}" || die "DATA_ROOT directory not found: ${DATA_ROOT}"

# -------------------------
# NTU outputs (IN SCRATCH)
# -------------------------
OUT_ROOT="${SCRATCH_ROOT}/ai-gym-buddy/outputs"
NTU_EXP_DIR="${OUT_ROOT}/ntu120_xsub_baseline"

# Try to find best checkpoint
BEST="${NTU_EXP_DIR}/best.pt"
LAST="${NTU_EXP_DIR}/last.pt"
FINAL="${NTU_EXP_DIR}/final.pt"

CKPT="${BEST}"
if [ ! -f "${CKPT}" ]; then
  if [ -f "${LAST}" ]; then CKPT="${LAST}"; 
  elif [ -f "${FINAL}" ]; then CKPT="${FINAL}"; 
  else die "No checkpoint found in ${NTU_EXP_DIR}"; fi
fi

echo "Checkpoint: ${CKPT}"

UNC_DIR="${OUT_ROOT}/uncertainty"
MC_OUT="${UNC_DIR}/ntu120_xsub_mc.npz"
CALIB_DIR="${UNC_DIR}/ntu120_xsub_calibration"
STRESS_DIR="${UNC_DIR}/ntu120_xsub_stress"
GATE_DIR="${UNC_DIR}/ntu120_xsub_gating"

mkdir -p "${UNC_DIR}" "${CALIB_DIR}" "${STRESS_DIR}" "${GATE_DIR}"

# -------------------------
# GPU check
# -------------------------
nvidia-smi
$PY -c "import torch; print('CUDA:', torch.cuda.is_available())"

# -------------------------
# (1) MC Dropout Eval
# -------------------------
echo "=== (1) MC DROPOUT EVAL ==="
$PY scripts/eval_mc_dropout.py \
  --checkpoint "${CKPT}" \
  --split xsub_val \
  --n_passes 20 \
  --dropout 0.1 \
  --out "${MC_OUT}"

test -f "${MC_OUT}" || die "MC output not found: ${MC_OUT}"

# -------------------------
# (2) Calibration plots
# -------------------------
echo "=== (2) CALIBRATION ==="
$PY scripts/make_calibration_plots.py \
  --mc_file "${MC_OUT}" \
  --out_dir "${CALIB_DIR}"

# -------------------------
# (3) Stress tests
# -------------------------
echo "=== (3) STRESS TESTS ==="
$PY scripts/stress_tests.py \
  --checkpoint "${CKPT}" \
  --out_dir "${STRESS_DIR}"

# -------------------------
# (4) Gating sweep
# -------------------------
echo "=== (4) GATING SWEEP ==="
GATE_CSV="${GATE_DIR}/ntu120_xsub_gating.csv"
GATE_FIG="${GATE_DIR}/ntu120_xsub_gating.png"

$PY scripts/run_gating_sweep.py \
  --mc_file "${MC_OUT}" \
  --out_csv "${GATE_CSV}" \
  --out_fig "${GATE_FIG}"

echo "=== DONE ==="
