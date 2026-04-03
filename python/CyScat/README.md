<div align="center">
  <img src="../../logo.png" alt="CyScat logo" width="160"/>
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
- Scale to thousands of cylinders using **multi-GPU cascade** on HPC clusters

## Features

- **S-matrix computation** — multiple scattering theory with Modified Shanks Transformation for convergence acceleration
- **Cascade** — combine S-matrices via the Redheffer star product for multi-layer structures
- **JAX AD** — full automatic differentiation through the scattering pipeline via JAX
- **Multi-GPU / HPC** — partition large arrays across nodes, compute S-matrices in parallel, cascade results
- **Wave field visualization** — animated wave field showing normal incidence vs. optimal wavefront
- **Optimization** — gradient-based optimization of refractive index and radius

## Project Structure

```
CyScat/
├── Scattering_Code/                          Core scattering algorithms
│   ├── smatrix.py                              S-matrix generation (main entry point)
│   ├── smatrix_cascade.py                      Auto-cascade for large arrays
│   ├── cascadertwo.py                          Redheffer star product (two S-matrices)
│   ├── transall.py                             Translation matrix with Shanks acceleration
│   ├── sall.py                                 Mie scattering coefficients (Bessel/Hankel)
│   ├── vall.py                                 Plane wave → cylinder harmonic expansion
│   ├── scattering_coefficients_all.py          Project scattered field onto Floquet modes
│   ├── smatrix_parameters.py                   Spectral/spatial parameter setup
│   ├── modified_epsilon_shanks.py              Modified epsilon Shanks transformation
│   ├── ky.py                                   Floquet y-wavenumber
│   ├── bessel_jax.py                           JAX-compatible Bessel/Hankel functions
│   └── gpu_backend.py                          JAX/NumPy backend selection
│
├── examples/
│   ├── generate_s_matrix_1layer_dielectric/    Dielectric cylinders: geometry, S21 SVD
│   ├── generate_s_matrix_1layer_pec/           PEC cylinders: geometry, unitarity, S21 SVD
│   ├── generate_s_matrix_cascaded_layers_dielectric/  Dielectric cascade (2, 10, 20 layers)
│   ├── generate_s_matrix_cascaded_layers_pec/         PEC cascade (2, 10, 20 layers)
│   ├── multiple_cyl_cascade/                          Large-array cascade (100+ cylinders)
│   ├── differentiable_s_matrix_demo/                  ∂S21/∂x, ∂S21/∂λ, ∂S21/∂n via FD
│   ├── differentiable_s_matrix_wigner_smith/          Wigner-Smith time-delay via ∂S/∂ω
│   ├── optimize_refractive_index_and_radius/          Adam optimization with JAX gradients
│   ├── generate_wave_demo_dielectric/                 Wave field — dielectric cylinders
│   ├── generate_wave_demo_pec/                        Wave field — PEC cylinders
│   └── maze_s_matrix_wavefront_optimization/          PEC maze eigenchannel routing
│
├── notebooks/                                Jupyter notebooks
│   └── cyscat_demo.ipynb
│
├── run_job.sh                                SLURM job script (Great Lakes)
├── run_multinode.sh                          Multi-node SLURM script
├── setup_greatlakes.sh                       Great Lakes environment setup
└── requirements.txt
```

## Setup

### Local

```bash
pip install -r requirements.txt
```

### Great Lakes HPC

```bash
source setup_greatlakes.sh
```

For GPU support, install the appropriate `jax[cuda]` version — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Usage

### Compute an S-matrix

```python
from Scattering_Code.smatrix import smatrix
from Scattering_Code.smatrix_parameters import smatrix_parameters

S, _ = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength, nmax, thickness, sp, "On")
```

### Cascade layers

```python
from Scattering_Code.cascadertwo import cascadertwo

S_total, d_total = cascadertwo(S1, d1, S2, d2)
```

### Differentiate with JAX

```python
from matrix_derivatives import compute_dS21_dx
import jax

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

### Multi-node HPC (Great Lakes)

```bash
sbatch run_multinode.sh
```

## Key Algorithms

### Multiple Scattering (T-matrix method)
Each cylinder scatters incident fields into cylindrical harmonics. The self-consistent system
`(I - diag(s) · T) · c = diag(s) · v` is solved via LU decomposition.

### Cascade (Redheffer Star Product)
`S_total = S_A ⋆ S_B` — builds multi-layer structures from pre-computed S-matrices.

### Automatic Differentiation
JAX traces through the full scattering pipeline. Custom Bessel/Hankel implementations in
`bessel_jax.py` are fully differentiable.

## Credits

- Original MATLAB implementation: **Curtis Jin**, **Prof. Raj Rao Nadakuditi**, **Prof. Eric Michielssen**, University of Michigan
- Translated to Julia and Python by **Thirulok Sundar**, with the help of [Claude](https://claude.ai)

## License

MIT
