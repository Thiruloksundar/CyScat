#!/bin/bash
#SBATCH --job-name=cyscat
#SBATCH --account=rajnrao0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cyscat_%j.log

# === Load modules ===
module load python/3.10
module load cuda/12.1

# === Activate environment ===
source ~/cyscat_env/bin/activate

# === Run computation ===
cd CyScat/

echo "=== CyScat Multi-GPU Computation ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

# NUM_CYL: number of cylinders (default 5000, override with: sbatch run_job.sh 2500)
NUM_CYL=${1:-5000}
# NUM_GPUS: number of GPUs to use in parallel (default 4, override with: sbatch run_job.sh 5000 2)
NUM_GPUS=${2:-4}

echo "Cylinders: $NUM_CYL | GPUs: $NUM_GPUS"
echo ""

python compute_ncyl.py $NUM_CYL --gpus $NUM_GPUS

echo ""
echo "=== Job complete: $(date) ==="
