#!/bin/bash
#SBATCH --job-name=svd_dist
#SBATCH --account=rajnrao0
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/svd_all.log
#SBATCH --error=logs/svd_all.log

mkdir -p /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/CyScat/logs
mkdir -p /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/CyScat/svd_results

module load python/3.10
module load cuda/12.1
source ~/cyscat_env/bin/activate

cd /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/CyScat

python compute_svd_trial.py 0 --n_cyl 1600 --gpus 8 --trials_per_job 200
