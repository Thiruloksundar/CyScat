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
- Differentiate through the entire pipeline with **JAX** (`jax.grad`, `jax.jacobian`)
- Optimize cylinder geometry using automatic differentiation

## Features

- **S-matrix computation** — multiple scattering theory with Modified Shanks Transformation for convergence acceleration
- **Cascade** — combine S-matrices via the Redheffer star product for multi-layer structures
- **JAX AD** — full automatic differentiation through the scattering pipeline via JAX
- **GPU support** — JAX backend runs on CUDA GPUs with no code changes
- **Wave field visualization** — animated wave field showing normal incidence vs. optimal wavefront
- **Optimization** — gradient-based optimization of refractive index and radius

## Project Structure

```
CyScat/
├── Scattering_Code/                Core scattering algorithms
│   ├── smatrix.py                    S-matrix generation (main entry point)
│   ├── transall.py                   Translation matrix with Shanks acceleration
│   ├── sall.py                       Mie scattering coefficients (Bessel/Hankel)
│   ├── ky.py                         Floquet y-wavenumber
│   ├── cascadertwo.py                Redheffer star product (two S-matrices)
│   ├── smatrix_cascade.py            Auto-cascade for large arrays
│   ├── modified_epsilon_shanks.py    Modified epsilon Shanks transformation
│   ├── smatrix_parameters.py         Spectral/spatial parameter setup
│   ├── bessel_jax.py                 JAX-compatible Bessel/Hankel functions
│   └── gpu_backend.py                JAX/NumPy backend selection
│
├── get_partition.py                S-matrix block extraction
├── matrix_derivatives.py           ∂S21/∂x, ∂S21/∂λ, ∂S21/∂n via JAX
├── check_differentiability.py      Gradient verification vs finite differences
├── optimize_n_r.py                 Adam optimization of n and r
├── wave_field_movie.py             Wave field animation (normal + optimal wavefront)
└── notebooks/                      Jupyter notebooks
```

## Setup

```bash
cd CyScat
pip install -r requirements.txt
```

For GPU support, install the appropriate `jax[cuda]` version for your CUDA environment — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Usage

### Compute an S-matrix

```python
from Scattering_Code.smatrix import smatrix
from Scattering_Code.smatrix_parameters import smatrix_parameters
import numpy as np

S, _ = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength, nmax, thickness, sp, "On")
S11 = S[:nm, :nm]
S21 = S[nm:, :nm]
tau = np.linalg.svd(S21, compute_uv=False)
```

### Cascade layers

```python
from Scattering_Code.cascadertwo import cascadertwo

S_total, d_total = cascadertwo(S1, d1, S2, d2)
```

### Differentiate with JAX

```python
import jax
from matrix_derivatives import compute_dS21_dx

dS21_dx = jax.jacobian(compute_dS21_dx)(x_positions)
```

### Wave field animation

```bash
python wave_field_movie.py --num_cyl 50 --pec
```

### Optimize refractive index

```bash
python optimize_n_r.py
```

## Key Algorithms

### Multiple Scattering (T-matrix method)
Each cylinder scatters incident fields into cylindrical harmonics. The self-consistent system
`(I - diag(s) · T) · c = diag(s) · v` is solved via LU decomposition.

The translation matrix is computed via the **Graf addition theorem** for Hankel functions,
summed over all periodic images using the **Modified Epsilon Shanks Transformation**.

### S-matrix Normalization
Normalized so that `|S_ij|²` represents energy flux, making S unitary for lossless structures.

### Cascade (Redheffer Star Product)
`S_total = S_A ⋆ S_B` — builds multi-layer structures from a single pre-computed S-matrix.

### Automatic Differentiation
JAX traces through the full scattering pipeline. Custom implementations of Bessel/Hankel functions
in `bessel_jax.py` are fully differentiable, enabling `jax.grad` and `jax.jacobian` on any output.

## Credits

- Original MATLAB implementation: **Curtis Jin**, **Prof. Raj Rao Nadakuditi**, **Prof. Eric Michielssen**, University of Michigan
- Translated to Julia and Python by **Thirulok Sundar**, with the help of [Claude](https://claude.ai)

## License

MIT
