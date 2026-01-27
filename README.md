# AI Gym Buddy

Safety-aware fitness coach acting on human skeleton sequences.

## Quick Start Guide

Follow these steps to set up and run the project.

### 1. Environment Setup

This project uses **Poetry** for dependency management.

```bash
# Install dependencies
poetry install
```

### 2. Data Preparation

Ensure your data is placed in the following structure:

```text
data/raw/skeleton/
├── ntu120/
│   └── ntu120_3d.pkl  <-- NTU120 3D Skeleton Data
└── kinetics400/
    ├── k400_2d.pkl    <-- Kinetics 400 Metadata
    └── kpfiles/       <-- Kinetics 400 Skeleton Files
```

**Note for Kinetics Data:**
The Kinetics skeleton files usually come in a zip file. Use the helper script to extract them:

```bash
# Unzip Kinetics 400 data
chmod +x scripts/unzip_kinetics.sh
./scripts/unzip_kinetics.sh
```

### 3. Running the Code

We have created several scripts to verify the data and train a baseline model.

#### Step 1: Inspect Data
Verify that the NTU120 dataset is readable and check its statistics.

```bash
poetry run python scripts/inspect_dataset.py
```

#### Step 2: Train Baseline Model
Train a simple Temporal Convolutional Network (TCN) on a small subset of the NTU120 dataset to verify the pipeline.

```bash
poetry run python scripts/train_baseline.py
```
*Expected Output:* You should see training loss decrease and validation accuracy reach ~20% (significantly higher than random chance ~0.8%).

#### Step 3: Test Data Loaders
Run specific tests to ensure the data loaders are working correctly (shapes, normalization, etc.).

```bash
# Test NTU120 Loader
poetry run python scripts/test_loader.py

# Test Kinetics Loader
poetry run python scripts/test_kinetics.py
```
#### Step 4: Full Training (NTU120)
Train the full Skeleton Transformer on the official `xsub` split.

```bash
#### Step 4: Full Training (NTU120)
Train the full Skeleton Transformer on the official `xsub` split.

```bash
# Run full training (defaults to 50 epochs)
poetry run python scripts/train_full_ntu.py
```
*Output:* 
- `outputs/ntu120_xsub_baseline/best.pt` (Best Checkpoint)
- `outputs/ntu120_xsub_baseline/train_curve.csv` (Learning Curve)
- `outputs/ntu120_xsub_baseline/metrics.json`

#### Step 5: Evaluate Uncertainty & Robustness
After training, run the evaluation suite to analyze model reliability.

```bash
# 1. MC Dropout Inference (Saves .npz stats)
poetry run python scripts/eval_mc_dropout.py
# Output: outputs/uncertainty/ntu120_xsub_mc.npz

# 2. Generate Figures (Reliability, ECE, Error vs Uncertainty)
poetry run python scripts/make_figures.py
# Output: outputs/figs/*.png, ece.json

# 3. Uncertainty-aware Gating Sweep
poetry run python scripts/run_gating_sweep.py
# Output: outputs/gating/gating_sweep.csv (Coverage, Risk, Unsafe Rate)

# 4. Stress Tests (Joint Dropout, Jitter, xView Shift)
poetry run python scripts/stress_tests.py
# Output: outputs/stress_robustness.csv
```

---

## 📂 Project Structure

- **`src/gymbuddy/`**: Main source code.
  - **`data/loaders/`**: Dataset implementations (`ntu120.py`, `kinetics_skeleton.py`).
  - **`models/`**: Neural network models (`baseline.py`, `transformer.py`).
  - **`uncertainty/`**: Uncertainty estimation modules (`mc_dropout.py`).
- **`scripts/`**: Helper scripts for inspection, training, and testing.
  - `train_full_ntu.py`: Main training script.
  - `eval_mc_dropout.py`: Uncertainty inference.
  - `make_calibration_plots.py`: Calibration plotting.
  - `run_gating_sweep.py`: Feedback gating analysis.
  - `stress_tests.py`: Robustness evaluation.
- **`data/`**: Directory for storing raw and processed datasets.

---

## 📚 Data Sources & Citations

This project makes use of the following datasets:

### NTU RGB+D 120
- **Constraint**: Used for 3D skeleton action recognition.
- **Source**: [OpenMMLab / PySKL](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl)
- **Citation**:
  > Liu, J., Shahroudy, A., Perez, M., Wang, G., Duan, L. Y., & Kot, A. C. (2019). NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence.

### Kinetics 400
- **Constraint**: Used for 2D skeleton pre-training/metadata.
- **Metadata**: [k400_hrnet.pkl](https://download.openmmlab.com/mmaction/pyskl/data/k400/k400_hrnet.pkl)
- **Skeletons**: [OpenXLab - Kinetics400-skeleton](https://openxlab.org.cn/datasets/OpenMMLab/Kinetics400-skeleton)
- **Citation**:
  > Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., ... & Zisserman, A. (2017). The kinetics human action video dataset. arXiv preprint arXiv:1705.06950.