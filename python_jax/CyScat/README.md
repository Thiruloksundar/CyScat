<div align="center">
  <img src="logo.png" alt="CyScat logo" width="160"/>
  <h1>CyScat — Python / JAX</h1>
  <p><em>Electromagnetic scattering matrix computation for periodic arrays of cylinders</em></p>
</div>

Python + JAX translation of the MATLAB CyScat package originally developed by **Curtis Jin**, **Prof. Raj Rao Nadakuditi**, and **Prof. Eric Michielssen** at the University of Michigan.

Translated to Julia and Python by **Thirulok Sundar**, with the help of [Claude](https://claude.ai).

---

## Overview

CyScat computes the **S-matrix** of a periodic slab of infinite dielectric or PEC cylinders using the
T-matrix multiple scattering method. The S-matrix maps input Floquet modes to output Floquet modes and
encodes all transmission and reflection properties of the structure.

Given the S-matrix you can:
- Compute transmission/reflection for any incident wavefront
- Find **open eigenchannels** (modes that pass through with near-perfect transmission) via SVD
- Differentiate through the entire pipeline with **JAX** (`jax.grad`, `jax.jacobian`, `jax.jvp`)
- Optimize cylinder geometry using automatic differentiation

## Features

- **S-matrix computation** — multiple scattering theory with Modified Shanks Transformation for convergence acceleration
- **Cascade** — combine S-matrices via the Redheffer star product for multi-layer structures
- **Full JAX AD** — exact automatic differentiation w.r.t. positions, wavelength, refractive index, and radius via `jax/` subfolder
- **GPU support** — JAX backend runs on CUDA GPUs with no code changes
- **Wave field visualization** — animated wave field showing normal incidence vs. optimal wavefront
- **Optimization** — gradient-based optimization of refractive index and radius
- **Derivative comparison** — `compare_julia_python_derivatives` notebook cross-checks JAX AD against Julia ForwardDiff

## Project Structure

```
CyScat/
├── Scattering_Code/                Core scattering algorithms (standard path)
│   ├── smatrix.py                    S-matrix generation
│   ├── transall.py                   Translation matrix with Shanks acceleration
│   ├── sall.py                       Mie scattering coefficients (Bessel/Hankel)
│   ├── ky.py                         Floquet y-wavenumber
│   ├── cascadertwo.py                Redheffer star product (two S-matrices)
│   ├── smatrix_cascade.py            Auto-cascade for large arrays
│   ├── modified_epsilon_shanks.py    Modified epsilon Shanks transformation
│   ├── smatrix_parameters.py         Spectral/spatial parameter setup
│   ├── bessel_jax.py                 JAX-compatible Bessel/Hankel functions
│   └── jax/                          JAX-differentiable versions (used by all examples)
│       ├── smatrix.py                  Fully differentiable S-matrix (jax.grad / jax.jvp)
│       ├── transall.py                 Translation matrix (JAX-traced path for λ-AD)
│       ├── sall.py                     Mie coefficients (JAX)
│       ├── ky.py                       Floquet wavenumber (JAX)
│       └── bessel_jax.py               Bessel/Hankel functions (JAX)
│
└── examples/
    ├── generate_s_matrix_1layer_dielectric/
    ├── generate_s_matrix_1layer_pec/
    ├── generate_s_matrix_cascaded_layers_dielectric/
    ├── generate_s_matrix_cascaded_layers_pec/
    ├── multiple_cyl_cascade/
    ├── differentiable_s_matrix_demo/          dS21/dx, dS21/dn via JAX grad
    ├── differentiable_s_matrix_wigner_smith/  Wigner-Smith time-delay matrix
    ├── optimize_refractive_index_and_radius/  Adam optimization via JAX
    ├── generate_wave_demo_dielectric/         Animated wave field (dielectric)
    ├── generate_wave_demo_pec/                Animated wave field (PEC)
    ├── maze_s_matrix_wavefront_optimization/  Open eigenchannel through PEC maze
    └── compare_julia_python_derivatives/      JAX AD vs Julia ForwardDiff
```

## Setup

```bash
pip install -r requirements.txt
```

For GPU support, install the appropriate `jax[cuda]` version — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Usage

### Compute an S-matrix

```python
import os
os.environ["JAX_ENABLE_X64"] = "1"
import jax
jax.config.update("jax_enable_x64", True)

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.jax.smatrix import smatrix
import numpy as np

sp = smatrix_parameters(wavelength, period, phiinc,
                        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, period/120)
S, _ = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength, nmax, thickness, sp, "On")
S21 = S[nm:, :nm]
tau = np.linalg.svd(S21, compute_uv=False)  # singular values = channel transmissions
```

### Differentiate with JAX

```python
import jax

def S21_flat(x_positions):
    clocs_new = clocs.at[:, 0].set(x_positions)
    S, _ = smatrix(clocs_new, cmmaxs, cepmus, crads, period, wavelength, nmax, thickness, sp, "On")
    return S[nm:, :nm].flatten()

# Jacobian of S21 w.r.t. cylinder x-positions
J = jax.jacobian(S21_flat, holomorphic=True)(jnp.array(clocs[:, 0]))

# Derivative of S21 w.r.t. wavelength via jvp
_, dS21_dlam = jax.jvp(lambda lam: smatrix(..., lam, ...), (lam,), (1.0,))
```

### Precomputed T-matrix for n/r sweeps

```python
from Scattering_Code.jax.smatrix import smatrix_precompute, smatrix_from_precomputed

precomp = smatrix_precompute(clocs, cmmaxs, period, wavelength, nmax, thickness, sp, "On")

# Now vary n or r cheaply (no T-matrix recomputation)
grad_n = jax.grad(lambda n: objective(smatrix_from_precomputed(precomp, cepmus_fn(n), crads)))
```

### Cascade layers

```python
from Scattering_Code.cascadertwo import cascadertwo

S_total, d_total = cascadertwo(S1, d1, S2, d2)
```

## Key Algorithms

### Multiple Scattering (T-matrix method)
Each cylinder scatters incident fields into cylindrical harmonics. The self-consistent system
`(I − diag(s) · T) · c = diag(s) · v` is solved via LU decomposition.

The translation matrix is computed via the **Graf addition theorem** for Hankel functions,
summed over all periodic images using the **Modified Epsilon Shanks Transformation**.

### S-matrix Normalization
Normalized so that `|S_ij|²` represents energy flux, making S unitary for lossless structures.

### Cascade (Redheffer Star Product)
`S_total = S_A ⋆ S_B` — builds multi-layer structures from a single pre-computed S-matrix.

### Automatic Differentiation
All examples use `Scattering_Code.jax.smatrix` which is fully differentiable via JAX.
- **Positions (x, y)**: `jax.grad` / `jax.jacobian` through T-matrix and projection matrices
- **Wavelength λ**: `jax.jvp` with spectral parameters rebuilt as JAX-traced quantities
- **Refractive index n / radius r**: `jax.grad` through precomputed T-matrix path (fast)

## Credits

- Original MATLAB implementation: **Curtis Jin**, **Prof. Raj Rao Nadakuditi**, **Prof. Eric Michielssen**, University of Michigan
- Translated to Julia and Python by **Thirulok Sundar**, with the help of [Claude](https://claude.ai)

## License

MIT
