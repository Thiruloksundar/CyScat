#!/bin/bash
# Setup CyScat Python environment on UMich Great Lakes
# Run this once after copying files to the server:
#   bash setup_greatlakes.sh

echo "=== Setting up CyScat Python environment ==="

# Load modules
module load python/3.10
module load cuda/12.1

# Create virtual environment
python -m venv ~/cyscat_env
source ~/cyscat_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy scipy matplotlib

# Install CuPy for GPU acceleration
# cupy-cuda12x matches CUDA 12.x on Great Lakes
pip install cupy-cuda12x

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "
import numpy as np
print(f'NumPy: {np.__version__}')
import scipy
print(f'SciPy: {scipy.__version__}')
try:
    import cupy as cp
    print(f'CuPy: {cp.__version__}')
    dev = cp.cuda.Device()
    print(f'GPU: {dev.name}')
    mem_free, mem_total = dev.mem_info
    print(f'GPU Memory: {mem_total/1e9:.1f} GB total, {mem_free/1e9:.1f} GB free')
except Exception as e:
    print(f'CuPy/GPU not available: {e}')
    print('Will fall back to CPU (NumPy)')
"

echo ""
echo "=== Setup complete ==="
echo "To activate: source ~/cyscat_env/bin/activate"
echo "To run: sbatch run_job.sh"
