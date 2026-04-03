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
- **Multi-GPU / HPC** — partition large arrays across nodes, compute S-matrices in parallel, cascade results (CuPy backend)
- **Wave field visualization** — animated wave field showing normal incidence vs. optimal wavefront
- **Optimization** — gradient-based optimization of refractive index and radius
- **Wigner-Smith matrix** — time-delay eigenvalues from Q = -iS⁻¹ ∂S/∂ω

## Project Structure

```
CyScat/
├── Scattering_Code/                       Core scattering algorithms
│   ├── smatrix.py                           S-matrix generation (main entry point)
│   ├── smatrix_cascade.py                   Auto-cascade for large arrays (multi-GPU)
│   ├── cascadertwo.py                       Redheffer star product (two S-matrices)
│   ├── transall.py                          Translation matrix with Shanks acceleration
│   ├── sall.py                              Mie scattering coefficients (Bessel/Hankel)
│   ├── vall.py                              Plane wave → cylinder harmonic expansion
│   ├── scattering_coefficients_all.py       Project scattered field onto Floquet modes
│   ├── smatrix_parameters.py                Spectral/spatial parameter setup
│   ├── modified_epsilon_shanks.py           Modified epsilon Shanks transformation
│   ├── ky.py                                Floquet y-wavenumber
│   ├── bessel_jax.py                        JAX-compatible Bessel/Hankel functions
│   └── gpu_backend.py                       JAX/NumPy/CuPy backend selection
│
├── examples/
│   ├── generate_s_matrix_1layer_dielectric/          Dielectric cylinders: geometry, S21 SVD
│   ├── generate_s_matrix_1layer_pec/                 PEC cylinders: geometry, unitarity, S21 SVD
│   ├── generate_s_matrix_cascaded_layers_dielectric/ Dielectric cascade (2, 10, 20 layers)
│   ├── generate_s_matrix_cascaded_layers_pec/        PEC cascade (2, 10, 20 layers)
│   ├── multiple_cyl_cascade/                         Large-array cascade (100+ cylinders)
│   ├── differentiable_s_matrix_demo/                 ∂S21/∂x, ∂S21/∂λ, ∂S21/∂n via JAX
│   ├── differentiable_s_matrix_wigner_smith/         Wigner-Smith time-delay via ∂S/∂ω
│   ├── optimize_refractive_index_and_radius/         Adam optimization with JAX gradients
│   ├── generate_wave_demo_dielectric/                Wave field animation — dielectric cylinders
│   ├── generate_wave_demo_pec/                       Wave field animation — PEC cylinders
│   └── maze_s_matrix_wavefront_optimization/         PEC maze eigenchannel routing
│
├── notebooks/                              Jupyter notebooks
│   └── cyscat_demo.ipynb                     Interactive demo notebook
│
├── compute_ncyl.py                         Single-node multi-GPU computation
├── compute_ncyl_multi_node.py              Multi-node cylinder computation (MPI + CuPy)
├── compute_svd_trial.py                    SVD trial computation for statistics
├── combine_svd.py                          Combine SVD results across trials
├── matrix_derivatives.py                   ∂S21/∂x, ∂S21/∂λ, ∂S21/∂n via JAX
├── optimize_n_r.py                         Adam optimization of n and r
├── wave_field_movie.py                     Wave field animation (normal + optimal wavefront)
├── check_differentiability.py              Gradient verification vs finite differences
├── check_differentiability_jax.py          JAX-specific gradient verification
├── get_partition.py                        S-matrix block extraction (S11/S12/S21/S22)
├── position_generator.py                   Random cylinder placement
└── system_initialization.py                System parameter initialization
```

## Setup

### Local

```bash
pip install -r requirements.txt
```

This installs `numpy`, `scipy`, and `matplotlib`. For JAX differentiation features:

```bash
pip install jax jaxlib
```

### Great Lakes HPC (GPU)

One-time setup — creates a virtual environment with CuPy for GPU acceleration:

```bash
bash setup_greatlakes.sh
```

This loads `python/3.10` and `cuda/12.1`, creates `~/cyscat_env`, and installs `numpy`, `scipy`, `matplotlib`, and `cupy-cuda12x`.

To activate the environment in future sessions:

```bash
source ~/cyscat_env/bin/activate
```

## Examples

Each example is a self-contained Jupyter notebook in `examples/` with markdown cells explaining the physics and code.

### Single-Layer S-Matrix (Dielectric)

Computes the S-matrix for a periodic slab of dielectric cylinders (n=1.3). Plots slab geometry and S21 singular value spectrum.

```bash
jupyter notebook examples/generate_s_matrix_1layer_dielectric/generate_s_matrix_1layer_dielectric.ipynb
```

### Single-Layer S-Matrix (PEC)

PEC cylinders (ε = -1). Verifies S-matrix unitarity (S†S = I) and plots S21 singular values showing strong scattering.

```bash
jupyter notebook examples/generate_s_matrix_1layer_pec/generate_s_matrix_1layer_pec.ipynb
```

### Cascaded Layers (Dielectric)

Cascades identical dielectric S-matrices via the Redheffer star product for 2, 10, and 20 layers. Shows how singular value spectrum evolves with depth.

```bash
jupyter notebook examples/generate_s_matrix_cascaded_layers_dielectric/generate_s_matrix_cascaded_layers_dielectric.ipynb
```

### Cascaded Layers (PEC)

Same cascade for PEC cylinders — stronger scattering produces deeper transmission dips.

```bash
jupyter notebook examples/generate_s_matrix_cascaded_layers_pec/generate_s_matrix_cascaded_layers_pec.ipynb
```

### Large-Array Cascade Demo

Cascades 100+ cylinders by partitioning into smaller sub-slabs, computing each S-matrix, and cascading results.

```bash
jupyter notebook examples/multiple_cyl_cascade/multiple_cyl_cascade.ipynb
```

### Differentiable S-Matrix Demo

Computes ∂S21/∂x, ∂S21/∂λ, and ∂S21/∂n using JAX automatic differentiation, validated against finite differences.

```bash
jupyter notebook examples/differentiable_s_matrix_demo/differentiable_s_matrix_demo.ipynb
```

### Wigner-Smith Time-Delay Matrix

Computes the Wigner-Smith matrix Q = -iS⁻¹ ∂S/∂ω from AD derivatives. Eigenvalues give proper delay times — the fundamental time scales of wave transport.

```bash
jupyter notebook examples/differentiable_s_matrix_wigner_smith/differentiable_s_matrix_wigner_smith.ipynb
```

### Optimize Refractive Index and Radius

Adam optimizer minimizes a cost function over refractive index and cylinder radius using JAX gradients through the full scattering pipeline.

```bash
jupyter notebook examples/optimize_refractive_index_and_radius/optimize_refractive_index_and_radius.ipynb
```

### Wave Field Animation (Dielectric)

Generates an animated wave field comparing normal incidence vs. the optimal transmission wavefront (from S21 SVD) through dielectric cylinders.

```bash
jupyter notebook examples/generate_wave_demo_dielectric/generate_wave_demo_dielectric.ipynb
```

### Wave Field Animation (PEC)

Same wave field animation for PEC cylinders — shows dramatic difference between normal incidence (~8% transmission) and optimal wavefront.

```bash
jupyter notebook examples/generate_wave_demo_pec/generate_wave_demo_pec.ipynb
```

### Maze Wavefront Optimization

Routes an open eigenchannel through a PEC cylinder maze. Finds the input wavefront that maximizes transmission through the maze geometry.

```bash
jupyter notebook examples/maze_s_matrix_wavefront_optimization/maze_s_matrix_wavefront_optimization.ipynb
```

### Interactive Notebook

```bash
jupyter notebook notebooks/cyscat_demo.ipynb
```

## Usage

### Compute an S-matrix

```python
from Scattering_Code.smatrix import smatrix
from Scattering_Code.smatrix_parameters import smatrix_parameters

sp = smatrix_parameters(period, wavelength, nmax, thickness)
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

## Running on Great Lakes HPC

CyScat supports multi-GPU computation on the University of Michigan Great Lakes cluster using CuPy for GPU-accelerated linear algebra. The `gpu_backend.py` module automatically selects CuPy (GPU) or NumPy (CPU) based on availability.

### Single-Node GPU Job

Computes S-matrices for a large cylinder array on one node with 4 GPUs:

```bash
sbatch run_job.sh              # default: 5000 cylinders, 4 GPUs
sbatch run_job.sh 2500         # custom cylinder count
sbatch run_job.sh 2500 2       # custom cylinder count and GPU count
```

- **Partition:** `spgpu` | **GPUs:** 4 (V100) | **Memory:** 48 GB | **Time:** 1 hour
- Runs `compute_ncyl.py` which partitions cylinders into sub-slabs, computes each S-matrix on a separate GPU, and cascades the results.

### Multi-Node GPU Job

Distributes computation across 2 nodes (8 GPUs each) using MPI:

```bash
sbatch run_multinode.sh
```

- **Partition:** `spgpu` | **Nodes:** 2 | **GPUs:** 8 per node | **Memory:** 64 GB | **Time:** 6 hours
- Requires `openmpi/4.1.6-cuda` module
- Runs `compute_ncyl_multi_node.py` via `srun` for MPI-based distribution

### SVD Distribution Statistics

Computes transmission singular value distributions over many random cylinder realizations:

```bash
sbatch svd_array_job.sh
```

- **Cylinders:** 1600 | **Trials:** 200 | **GPUs:** 4 | **Time:** 8 hours
- Runs `compute_svd_trial.py`, results saved to `svd_results/`
- Post-process with `python combine_svd.py` to aggregate statistics

### SLURM Configuration

All jobs use:
- **Account:** `rajnrao0`
- **Modules:** `python/3.10`, `cuda/12.1`
- **Environment:** `~/cyscat_env` (created by `setup_greatlakes.sh`)
- **Logs:** saved to `logs/` directory

## Key Algorithms

### Multiple Scattering (T-matrix method)
Each cylinder scatters incident fields into cylindrical harmonics. The self-consistent system
`(I - diag(s) · T) · c = diag(s) · v` is solved via LU decomposition, where:
- `s` = single-cylinder Mie scattering coefficients (Bessel/Hankel)
- `T` = translation matrix (how each cylinder sees the scattered field of all others)
- `v` = excitation by the incident Floquet mode

The translation matrix `T` is computed via the **Graf addition theorem** for Hankel functions,
summed over all periodic images using the **Modified Epsilon Shanks Transformation** for acceleration.

### S-matrix Normalization
The S-matrix is normalized so that `|S_ij|²` represents energy flux:
`S → diag(√(ky_out/k)) · S · diag(1/√(ky_in/k))`
This makes S unitary for lossless structures: `S†S = I`.

### Cascade (Redheffer Star Product)
Identical layers are cascaded via the star product `S_total = S_A ⋆ S_B`, building multi-layer structures from a single pre-computed S-matrix without re-solving the multiple scattering problem.

### Automatic Differentiation
JAX traces through the full scattering pipeline. Custom Bessel/Hankel implementations in
`bessel_jax.py` are fully differentiable, enabling `jax.grad` and `jax.jacobian` through the
entire S-matrix computation.

### Wigner-Smith Time-Delay Matrix
From ∂S/∂ω (computed via AD), the Wigner-Smith matrix Q = -iS⁻¹ ∂S/∂ω gives the proper delay times as its eigenvalues — the fundamental time scales of wave transport through the scattering region.

### Multi-GPU Cascade
For large cylinder arrays (1000+), the array is partitioned into sub-slabs that fit in GPU memory. Each sub-slab's S-matrix is computed independently (potentially on separate GPUs or nodes), then all sub-slab S-matrices are cascaded via the Redheffer star product to obtain the full-array S-matrix.

## Credits

- Original MATLAB implementation: **Curtis Jin**, **Prof. Raj Rao Nadakuditi**, **Prof. Eric Michielssen**, University of Michigan
- Translated to Julia and Python by **Thirulok Sundar**, with the help of [Claude](https://claude.ai)

## License

MIT
