#!/bin/bash
#SBATCH --job-name=nyx_gen
#SBATCH --account=rajnrao0
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/nyx_gen_%j.log
#SBATCH --error=logs/nyx_gen_%j.log

mkdir -p /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/logs
mkdir -p /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/nyx_matfiles

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
echo ""

# Generate all 10 IDX files with 8 GPUs
# Each file: 1440 cylinders, period=183.31, thickness=1100
# Output goes to nyx_matfiles/ directory
python generate_nyx_matfiles.py --idx 1 10 --gpus 8 --outdir nyx_matfiles

echo ""
echo "=== Job complete: $(date) ==="
