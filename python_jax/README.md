# CyScat — Python/JAX

Electromagnetic scattering matrix computation for arrays of dielectric cylinders, with JAX automatic differentiation support.

Translated from MATLAB (Curtis Jin, Prof. Raj Rao Nadakuditi, University of Michigan).

## Features

- **S-matrix computation** for periodic cylinder arrays using multiple scattering theory
- **JAX autodiff** — differentiable w.r.t. cylinder positions via `jax.grad`
- **Precomputed T-matrix** — 25x speedup for parameter sweeps (refractive index, radius)
- **Optimization** — Adam optimizer for refractive index and radius
- **Matrix derivatives** — dS21/dx, dS21/dlambda, dS21/dn via finite differences
- **Wave field movies** — animated electromagnetic field visualization
- **Cascade** — combine S-matrices of multiple slabs for large-scale simulations

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
cd CyScat

# Basic S-matrix computation (cascade test)
python main_cascade.py

# Optimization of refractive index and radius
python optimize_n_r.py

# Matrix derivatives of S21
python matrix_derivatives.py

# Wave field animation
python wave_field_movie.py --mode normal
python wave_field_movie.py --mode opt_trans

# JAX gradient check
python check_differentiability_jax.py
```

## Jupyter Notebook

A single notebook demonstrating all capabilities:

```bash
cd CyScat/notebooks
jupyter notebook cyscat_demo.ipynb
```

Covers: S-matrix computation, optimization, matrix derivatives, and wave field visualization.

## Project Structure

```
CyScat/
  Scattering_Code/       Core scattering algorithms
    smatrix.py             S-matrix generation (JAX-differentiable)
    transall.py            Translation matrix with Shanks acceleration
    sall.py                Mie scattering coefficients
    bessel_jax.py          JAX-differentiable Bessel/Hankel functions
    ky.py                  Wave vector y-component
    cascadertwo.py         Two-slab cascade
    smatrix_cascade.py     Auto-cascade for large arrays
    ...
  optimize_n_r.py        Optimize n and r with Adam + FD gradients
  matrix_derivatives.py  dS21/dx, dS21/dlambda, dS21/dn
  wave_field_movie.py    Animated wave field visualization
  main_cascade.py        Cascade accuracy test
  main_absorption.py     Unitarity / energy conservation test
  notebooks/
    cyscat_demo.ipynb    All-in-one demonstration notebook
```

## Key Algorithms

- **Multiple scattering**: T-matrix method with Modified Shanks Transformation for lattice sums
- **Cascade**: Redheffer star product for combining S-matrices
- **Differentiation**: JAX autodiff for positions; central finite differences for scalar parameters (n, r, lambda) using precomputed T-matrix for speed
- **Optimization**: Adam in log-space with best-so-far tracking

## References

- C. Jin, R. R. Nadakuditi — CyScat MATLAB package
- Original MATLAB code: University of Michigan
