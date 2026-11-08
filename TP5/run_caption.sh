#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J captioning
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Dossier du TP5
cd ~/TP-DeepLearning/TP5

# Activation de l'environnement Python
source ~/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate deeplearning2

echo "Job ID : $SLURM_JOB_ID"
echo "Noeud GPU : $SLURM_NODELIST"
echo "Nombre d'epoques : ${EPOCHS:-15}"

# Vérification du GPU
nvidia-smi

# Entraînement du modèle
python -u train.py
