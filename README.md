<div align="center">
  <img src="logo.png" alt="CyScat logo" width="160"/>
  <h1>CyScat</h1>
  <p><em>Electromagnetic scattering matrix computation for periodic arrays of cylinders</em></p>
</div>

Originally developed in MATLAB by **Curtis Jin**, **Prof. Raj Rao Nadakuditi**, and **Prof. Eric Michielssen** at the University of Michigan.

Translated to Julia and Python by **Thirulok Sundar**, with the help of [Claude](https://claude.ai).

---

## Overview

CyScat computes the **S-matrix** of a periodic slab of infinite dielectric or PEC cylinders using the T-matrix multiple scattering method. The S-matrix maps input Floquet modes to output Floquet modes and encodes all transmission and reflection properties of the structure.

Given the S-matrix you can:
- Compute transmission/reflection for any incident wavefront
- Find **open eigenchannels** (modes that pass through with near-perfect transmission) via SVD
- Differentiate through the entire pipeline for gradient-based optimization
- Visualize the animated wave field inside and outside the slab

## Implementations

| | **Julia** | **Python (GPU)** | **Python (JAX)** |
|---|---|---|---|
| **Directory** | [`julia/`](julia/) | [`python_gpu/`](python_gpu/) | [`python_jax/`](python_jax/) |
| **AD framework** | ForwardDiff.jl | Finite differences | JAX autodiff |
| **GPU support** | — | JAX (XLA) | JAX (XLA) |
| **Differentiable w.r.t.** | x, λ, n, r, ω | x, λ, n, r (FD) | x, λ, n, r (exact AD) |
| **Notebooks** | Pluto | Jupyter | Jupyter |

All three implementations share the same example set and produce numerically consistent S-matrices.

## Examples

All implementations include:

```
examples/
├── generate_s_matrix_1layer_dielectric/       S-matrix, SVD, singular values
├── generate_s_matrix_1layer_pec/              PEC cylinders variant
├── generate_s_matrix_cascaded_layers_dielectric/  Multi-layer cascade
├── generate_s_matrix_cascaded_layers_pec/
├── multiple_cyl_cascade/                      Large array via cascade
├── differentiable_s_matrix_demo/              dS21/dx via JAX / ForwardDiff
├── differentiable_s_matrix_wigner_smith/      Wigner-Smith time-delay matrix
├── optimize_refractive_index_and_radius/      Gradient-based optimization
├── generate_wave_demo_dielectric/             Animated wave field
├── generate_wave_demo_pec/
└── maze_s_matrix_wavefront_optimization/      Open eigenchannel routing
```

Plus, in Python JAX only:
```
examples/
└── compare_julia_python_derivatives/          JAX AD vs Julia ForwardDiff comparison
```

## Quick Start

### Julia

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/generate_s_matrix_1layer_pec/generate_s_matrix_1layer_pec.jl
```

### Python (GPU / finite-difference)

```bash
cd python_gpu
pip install -r requirements.txt
jupyter notebook CyScat/examples/generate_s_matrix_1layer_pec/generate_s_matrix_1layer_pec.ipynb
```

### Python (JAX / full autodiff)

```bash
cd python_jax
pip install -r requirements.txt
jupyter notebook CyScat/examples/generate_s_matrix_1layer_pec/generate_s_matrix_1layer_pec.ipynb
```

See the implementation-specific READMEs for full details:
- [Julia README](julia/README.md)
- [Python GPU README](python_gpu/README.md)
- [Python JAX README](python_jax/README.md)

## Credits

- Original MATLAB implementation: **Curtis Jin**, **Prof. Raj Rao Nadakuditi**, **Prof. Eric Michielssen**, University of Michigan
- Translated to Julia and Python by **Thirulok Sundar**, with the help of [Claude](https://claude.ai)

## License

MIT — see [LICENSE](LICENSE)
