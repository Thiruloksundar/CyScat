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

## Implementations

| | **Julia** | **Python / JAX** |
|---|---|---|
| **Directory** | [`julia/`](julia/) | [`python/`](python/) |
| **AD Framework** | ForwardDiff.jl | JAX |
| **GPU Support** | — | Multi-GPU via JAX |
| **Interactive** | Pluto notebook | Jupyter notebooks |

Both implementations share the same examples structure:

```
examples/
├── generate_s_matrix_1layer_dielectric/
├── generate_s_matrix_1layer_pec/
├── generate_s_matrix_cascaded_layers_dielectric/
├── generate_s_matrix_cascaded_layers_pec/
├── multiple_cyl_cascade/
├── differentiable_s_matrix_demo/
├── differentiable_s_matrix_wigner_smith/
├── optimize_refractive_index_and_radius/
├── generate_wave_demo_dielectric/
├── generate_wave_demo_pec/
└── maze_s_matrix_wavefront_optimization/
```

## Quick Start

### Julia

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/generate_s_matrix_1layer_pec/generate_s_matrix_1layer_pec.jl
```

### Python

```bash
cd python
pip install -r requirements.txt
# Open any example notebook in Jupyter
jupyter notebook CyScat/examples/generate_s_matrix_1layer_pec/generate_s_matrix_1layer_pec.ipynb
```

See the [Julia README](julia/README.md) and [Python README](python/CyScat/README.md) for full details.

## Credits

- Original MATLAB implementation: **Curtis Jin**, **Prof. Raj Rao Nadakuditi**, **Prof. Eric Michielssen**, University of Michigan
- Translated to Julia and Python by **Thirulok Sundar**, with the help of [Claude](https://claude.ai)

## License

MIT — see [LICENSE](LICENSE)
