"""
JAX Backend for CyScat

Replaces CuPy GPU backend with JAX for automatic differentiation support.
JAX handles GPU/CPU dispatch automatically via jax.default_backend().

Set environment variable JAX_PLATFORM_NAME=cpu to force CPU.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

# JAX backend info
GPU_AVAILABLE = jax.default_backend() != 'cpu'
xp = jnp


def get_info():
    """Return backend information string."""
    backend = jax.default_backend()
    devices = jax.devices()
    if backend == 'gpu':
        return f"JAX GPU: {devices[0]}"
    return f"JAX CPU ({len(devices)} device(s))"


def to_gpu(arr):
    """Convert to JAX array."""
    return jnp.asarray(arr)


def to_cpu(arr):
    """Convert JAX array to numpy."""
    return np.asarray(arr)


def lu_factor(a):
    """LU factorization using JAX."""
    return jla.lu_factor(jnp.asarray(a))


def lu_solve(lu_piv, b):
    """LU solve using JAX."""
    return jla.lu_solve(lu_piv, jnp.asarray(b))


def inv(a):
    """Matrix inverse using JAX."""
    return jnp.linalg.inv(jnp.asarray(a))
