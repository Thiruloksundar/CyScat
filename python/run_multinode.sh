#!/bin/bash
#SBATCH --job-name=cyscat_multinode
#SBATCH --account=rajnrao0
#SBATCH --partition=spgpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/multinode.log
#SBATCH --error=logs/multinode.log

mkdir -p /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/CyScat/logs

module load gcc/10.3.0
module load openmpi/4.1.6-cuda
module load python/3.10
module load cuda/12.1
source ~/cyscat_env/bin/activate

cd /gpfs/accounts/rajnrao_root/rajnrao0/thirulok/CyScat_Python_Multi_GPU/CyScat

srun python compute_ncyl_multi_node.py --n_cyl 500 --gpus 8
