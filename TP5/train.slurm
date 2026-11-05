#!/bin/bash
#SBATCH --job-name=tp5-captioning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=tp5-%j.out
#SBATCH --error=tp5-%j.err
#SBATCH --account=csc_8607
#SBATCH --qos=normal
cd ~/TP5

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

echo "Node: $(hostname)"
echo "Python: $(which python)"

nvidia-smi

python -u train.py
