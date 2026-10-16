#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=quick-tests
#SBATCH --output=logs/quick-tests-%j.out
#SBATCH --error=logs/quick-tests-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

nvidia-smi

python -u train_imagenet.py --model mlp --subset 20000 --epochs 10
python -u train_imagenet.py --model alexnet --subset 20000 --epochs 10
python -u train_imagenet.py --model alexnet_deep --subset 20000 --epochs 10