# Anvil Cluster Training Guide

This guide provides "readable and skimmable" instructions for running training on the Anvil cluster.

## 1. Accessing the Cluster
1. Open a terminal (no SSH needed) via **OnDemand** -> **Clusters** -> **Anvil Shell Access**.
2. You should land on a login node prompt.

## 2. Key Paths & Setup
Run the following to set up your environment variables and directories:

```bash
# Set variables
export USERNAME="x-akhanal3"
export SCRATCH="/anvil/scratch/$USERNAME"
export PROJ="$SCRATCH/ai-gym-buddy"
export DATA_ROOT="$PROJ/data/raw/skeleton"
export OUTPUT_DIR="$PROJ/outputs"

# Create directories
mkdir -p "$PROJ"/{data/raw/skeleton,outputs}
```

## 3. Code Location
Keep code in `$HOME` (small storage) and data in `$SCRATCH` (large storage).

```bash
# Setup project in Home
mkdir -p $HOME/projects
cd $HOME/projects
git clone <YOUR_REPO_URL> gymbuddy-backend
cd gymbuddy-backend
```

## 4. Data Placement
Ensure your data is organized as follows on Anvil:

```
/anvil/scratch/<username>/ai-gym-buddy/data/raw/skeleton/
├── ntu120/
│   └── ntu120_3d.pkl
└── kinetics400/
    ├── k400_2d.pkl
    └── kpfiles/...
```

**Sanity Check:**
```bash
ls -lh "$DATA_ROOT/ntu120"
ls -lh "$DATA_ROOT/kinetics400"
du -sh "$DATA_ROOT"
```

## 5. Environment (Poetry)
**Option A (Recommended):** Use the Poetry-created Python directly.

1. **Find your Python path:**
   ```bash
   which python
   # Example output: /home/x-akhanal3/.conda/envs/2024.02-py311/poetry_env/bin/python
   ```

2. **Store it:**
   ```bash
   export PY="/home/x-akhanal3/.conda/envs/2024.02-py311/poetry_env/bin/python"
   ```

3. **Quick Test:**
   ```bash
   $PY -c "import torch; print(torch.__version__)"
   ```
   *Note: Using `$PY` directly avoids `conda init` issues in SLURM jobs.*

## 6. Local Smoke Test (Login Node)
Run a quick CPU-only test on the login node to verify correctness (do NOT run full training here).

```bash
cd $HOME/projects/gymbuddy-backend
export DATA_ROOT="$DATA_ROOT"
export OUTPUT_DIR="$OUTPUT_DIR"

$PY scripts/train_full_ntu.py --split xsub --epochs 1 --debug
```

## 7. GPU Training via SLURM
**Partition:** `ai` (Required)

1. **Create the SLURM script:**
   ```bash
   cd $HOME/projects/gymbuddy-backend
   mkdir -p jobs logs
   nano jobs/train_ntu120_xsub.slurm
   ```

2. **Paste the following content:**
   ```bash
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
   #SBATCH --time=12:00:00

   mkdir -p logs

   REPO_DIR="$HOME/projects/gymbuddy-backend"
   # UPDATE THIS IF DIFFERENT
   PY="/home/x-akhanal3/.conda/envs/2024.02-py311/poetry_env/bin/python"

   cd "$REPO_DIR" || { echo "Repo not found: $REPO_DIR"; exit 1; }

   export DATA_ROOT="/anvil/scratch/x-akhanal3/ai-gym-buddy/data/raw/skeleton"
   export OUTPUT_DIR="/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs"

   # Confirm GPU exists
   nvidia-smi
   $PY -c "import torch; print('cuda?', torch.cuda.is_available())"

   # Run Training
   $PY scripts/train_full_ntu.py \
     --split xsub \
     --epochs 50 \
     --lr 3e-4 \
     --dropout 0.1 \
     --batch_size 32
   ```

3. **Submit the job:**
   ```bash
   sbatch jobs/train_ntu120_xsub.slurm
   ```

4. **Monitor:**
   ```bash
   squeue -u $USER
   tail -f logs/ntu120_xsub-<JOBID>.out
   ```

## 8. MC Dropout Evaluation (Separate Job)
Create a separate job for evaluation after training completes.

1. **Create Script:**
   ```bash
   nano jobs/eval_mc_dropout.slurm
   ```

2. **Template:**
   ```bash
   #!/bin/bash
   #SBATCH -J ntu120_mc
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

   cd "$REPO_DIR" || exit 1

   export DATA_ROOT="/anvil/scratch/x-akhanal3/ai-gym-buddy/data/raw/skeleton"
   export OUTPUT_DIR="/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs"

   # Run Evaluation
   $PY scripts/eval_mc_dropout.py --n_passes 20
   ```

3. **Submit:**
   ```bash
   sbatch jobs/eval_mc_dropout.slurm
   ```

## 9. Results Location
All outputs are saved to:
`/anvil/scratch/<username>/ai-gym-buddy/outputs/`

**Check size:**
```bash
du -sh "$OUTPUT_DIR"
ls "$OUTPUT_DIR"
```

## 10. Pull Results to Laptop (Optional)
Use `rsync` from your laptop:

```bash
rsync -avh --progress \
  x-akhanal3@anvil.rcac.purdue.edu:/anvil/scratch/x-akhanal3/ai-gym-buddy/outputs/ \
  ./outputs_anvil/
```

## 11. Troubleshooting
- **"job rejected: use --partition=ai"**: Ensure `#SBATCH --partition=ai` is in your script.
- **GPU job uses CPU**: Check `nvidia-smi` output in logs. If `cuda? False`, rebuild PyTorch with CUDA support.
- **"File not found"**: Verify `$DATA_ROOT` path is correct.
