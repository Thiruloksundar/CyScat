"""
GPU Backend for CyScat
Uses CuPy (NVIDIA GPU) when available, falls back to NumPy (CPU).

Set environment variable CYSCAT_FORCE_CPU=1 to disable GPU even when CuPy is installed.
"""

import numpy as np
import os

_FORCE_CPU = os.environ.get('CYSCAT_FORCE_CPU', '0') == '1'

try:
    if _FORCE_CPU:
        raise ImportError("GPU disabled by CYSCAT_FORCE_CPU=1")
    import cupy as cp
    import cupyx.scipy.linalg as cpx_linalg
    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    GPU_AVAILABLE = False
    xp = np
    cp = None
    cpx_linalg = None


def get_info():
    """Return backend information string."""
    if GPU_AVAILABLE:
        try:
            dev = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            gpu_name = props['name']
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
            mem_free, mem_total = dev.mem_info
            return f"CuPy GPU: {gpu_name}, {mem_total/1e9:.1f}GB total, {mem_free/1e9:.1f}GB free"
        except Exception as e:
            return f"CuPy GPU (name unavailable: {e})"
    return "NumPy CPU (no GPU)"


def to_gpu(arr):
    """Transfer numpy array to GPU."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return np.asarray(arr)


def to_cpu(arr):
    """Transfer GPU array to CPU numpy."""
    if GPU_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def lu_factor(a):
    """LU factorization on GPU or CPU."""
    if GPU_AVAILABLE:
        return cpx_linalg.lu_factor(cp.asarray(a))
    from scipy.linalg import lu_factor as sp_lu_factor
    return sp_lu_factor(a)


def lu_solve(lu_piv, b):
    """LU solve on GPU or CPU. Accepts vector or matrix RHS."""
    if GPU_AVAILABLE:
        b_gpu = cp.asarray(b)
        if b_gpu.ndim == 2:
            # CuPy lu_solve on this version doesn't handle 2D RHS
            cols = [cpx_linalg.lu_solve(lu_piv, b_gpu[:, i])
                    for i in range(b_gpu.shape[1])]
            return cp.column_stack(cols)
        return cpx_linalg.lu_solve(lu_piv, b_gpu)
    from scipy.linalg import lu_solve as sp_lu_solve
    return sp_lu_solve(lu_piv, b)

def inv(a):
    """Matrix inverse on GPU or CPU."""
    if GPU_AVAILABLE:
        return cp.linalg.inv(cp.asarray(a))
    return np.linalg.inv(a)
