# CyScat — Python / JAX

Electromagnetic scattering matrix computation for periodic arrays of cylinders, with **full JAX automatic differentiation** support.

Translated from MATLAB (Curtis Jin, Prof. Raj Rao Nadakuditi, Prof. Eric Michielssen, University of Michigan).

## What this version provides

- Exact AD w.r.t. **positions**, **wavelength**, **refractive index**, and **radius** via `jax.grad` / `jax.jvp`
- All examples use `Scattering_Code.jax.smatrix` — the fully differentiable pipeline
- GPU acceleration via JAX/XLA with no code changes
- Cross-validation notebook comparing JAX AD against Julia ForwardDiff

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU support, install the appropriate `jax[cuda]` — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Examples

Open any notebook under `CyScat/examples/`:

```bash
cd CyScat
jupyter notebook examples/generate_s_matrix_1layer_pec/generate_s_matrix_1layer_pec.ipynb
jupyter notebook examples/differentiable_s_matrix_demo/differentiable_s_matrix_demo.ipynb
jupyter notebook examples/optimize_refractive_index_and_radius/optimize_refractive_index_and_radius.ipynb
jupyter notebook examples/compare_julia_python_derivatives/compare_julia_python_derivatives.ipynb
```

See [CyScat/README.md](CyScat/README.md) for full documentation.
