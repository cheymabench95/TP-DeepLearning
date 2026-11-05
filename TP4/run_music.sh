#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J music-exp
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd ~/TP-DeepLearning/TP4

source ~/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate deeplearning2

echo "Job ID : $SLURM_JOB_ID"
echo "Noeud GPU : $SLURM_NODELIST"
echo "Experience : ${EXPERIMENT:?Variable EXPERIMENT manquante}"

nvidia-smi

python -u train_music.py --experiment "$EXPERIMENT"

