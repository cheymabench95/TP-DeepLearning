#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -J alexnet-full
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Activation de l'environnement Mamba
source ~/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate deeplearning2

cd ~/TP-DeepLearning/TP3

echo "Début du job $SLURM_JOB_ID sur le nœud $SLURM_NODELIST"
nvidia-smi

python -u train_imagenet.py
