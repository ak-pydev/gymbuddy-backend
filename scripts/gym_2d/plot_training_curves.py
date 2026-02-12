import os
import json

import matplotlib.pyplot as plt
import argparse

def plot_ntu_baseline(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return

    epochs = []
    losses = []
    val_accs = []
    
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(float(row['epoch']))
            losses.append(float(row['train_loss']))
            val_accs.append(float(row['val_acc']))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Val Accuracy', color=color)
    ax2.plot(epochs, val_accs, color=color, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('NTU120 X-Sub Baseline Training')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ntu_baseline_curve.png'))
    plt.close()
    print(f"Saved ntu_baseline_curve.png to {out_dir}")

def plot_gym_finetune(json_path, out_dir):
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, history['train_loss'], color=color, linestyle='-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color=color, linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, history['train_acc'], color=color, linestyle='-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], color=color, linestyle='--', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Gym 2D Finetuning (Head Only)')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gym_finetune_curve.png'))
    plt.close()
    print(f"Saved gym_finetune_curve.png to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, required=True)
    args = parser.parse_args()

    out_dir = os.path.join(args.outputs_dir, 'training_analysis')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Plot NTU Baseline
    ntu_csv = os.path.join(args.outputs_dir, 'ntu120_xsub_baseline', 'train_curve.csv')
    plot_ntu_baseline(ntu_csv, out_dir)

    # 2. Plot Gym Finetune
    gym_json = os.path.join(args.outputs_dir, 'gym_finetuned_head', 'history.json')
    plot_gym_finetune(gym_json, out_dir)
