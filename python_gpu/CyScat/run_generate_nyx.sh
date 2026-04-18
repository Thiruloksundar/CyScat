#!/bin/bash
#SBATCH --job-name=nyx_gen
#SBATCH --account=rajnrao0
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/nyx_gen_%j.log
#SBATCH --error=logs/nyx_gen_%j.log

# Usage:
#   sbatch run_generate_nyx.sh 1 5     # IDX 1-5 (first batch)
#   sbatch run_generate_nyx.sh 6 10    # IDX 6-10 (second batch)
#   sbatch run_generate_nyx.sh 1 1     # single IDX (for benchmarking)
#
# Estimated time: ~45 min per IDX with 8 GPUs (1440 cyl, period=183.31)
# 5 IDX per job ≈ ~4 hours

IDX_START=${1:-1}
IDX_END=${2:-5}

mkdir -p /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/CyScat/logs
mkdir -p /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/CyScat/nyx_matfiles

# === Load modules ===
module load python/3.10
module load cuda/12.1

# === Activate environment ===
source ~/cyscat_env/bin/activate

# === Run computation ===
cd /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/CyScat

echo "=== Generate Nyx .mat files ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "IDX range: $IDX_START to $IDX_END"
echo ""

python generate_nyx_matfiles.py --idx $IDX_START $IDX_END --gpus 8 --outdir nyx_matfiles

echo ""
echo "=== Job complete: $(date) ==="
