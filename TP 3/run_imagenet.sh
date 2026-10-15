#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -J alexnet-full
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# Activation de l'environnement virtuel Mamba/Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

echo "Début du job $SLURM_JOB_ID sur le nœud $SLURM_NODELIST"
nvidia-smi

# AlexNet, ImageNet complet, 50 époques
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -u train_imagenet.py