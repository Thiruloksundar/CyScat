# CyScat.jl

Electromagnetic scattering matrix computation for periodic arrays of cylinders.
Julia translation of the MATLAB CyScat package (Curtis Jin, Prof. Raj Rao Nadakuditi, University of Michigan).

## Overview

CyScat computes the **S-matrix** of a periodic slab of infinite dielectric or PEC cylinders using the
T-matrix multiple scattering method. The S-matrix maps input Floquet modes to output Floquet modes and
encodes all transmission and reflection properties of the structure.

Given the S-matrix you can:
- Compute transmission/reflection for any incident wavefront
- Find **open eigenchannels** (modes that pass through with near-perfect transmission) via SVD
- Optimize cylinder geometry for target scattering properties using automatic differentiation
- Compute the **Wigner-Smith time-delay matrix** via ∂S/∂ω

## Features

- **S-matrix computation** — multiple scattering theory with Modified Shanks Transformation for convergence acceleration
- **Cascade** — combine S-matrices via the Redheffer star product for multi-layer structures
- **ForwardDiff AD** — full forward-mode automatic differentiation through the entire pipeline; computes ∂S/∂x, ∂S/∂λ, ∂S/∂n, ∂S/∂ω
- **Wigner-Smith matrix** — time-delay eigenvalues from Q = -iS⁻¹ ∂S/∂ω
- **Optimization** — Adam optimizer for refractive index and radius using ForwardDiff gradients
- **Wave field visualization** — animated wave field showing normal incidence vs. optimal wavefront
- **S-channel maze** — demonstration of open eigenchannel routing through a PEC cylinder maze

## Project Structure

```
CyScat_Julia/
├── CyScat/                              Core library
│   ├── CyScat.jl                          Main module
│   ├── Scattering_Code/                   Core scattering algorithms
│   │   ├── smatrix.jl                       S-matrix generation (main entry point)
│   │   ├── transall.jl                      Translation matrix with Shanks acceleration
│   │   ├── sall.jl                          Mie scattering coefficients (Bessel/Hankel)
│   │   ├── vall.jl                          Plane wave → cylinder harmonic expansion
│   │   ├── scattering_coefficients_all.jl   Project scattered field onto Floquet modes
│   │   ├── smatrix_parameters.jl            Spectral/spatial parameter setup
│   │   ├── smatrix_cascade.jl               Auto-cascade for large arrays
│   │   ├── cascadertwo.jl                   Redheffer star product (two S-matrices)
│   │   ├── modified_epsilon_shanks.jl       Modified epsilon Shanks transformation
│   │   └── ky.jl                            Floquet y-wavenumber
│   ├── get_partition.jl                   S-matrix block extraction (S11/S12/S21/S22)
│   ├── position_generator.jl              Random cylinder placement
│   ├── visualize_slab.jl                  Slab geometry visualization
│   └── visualize_smatrix_data.jl          S-matrix data visualization
│
├── examples/
│   ├── generate_s_matrix_1layer_dielectric/     Dielectric cylinders: geometry, S21 SVD, wavefield
│   ├── generate_s_matrix_1layer_pec/            PEC cylinders: geometry, S21 SVD, wavefield
│   ├── generate_s_matrix_cascaded_layers_dielectric/  Dielectric cascade (2, 10, 20 layers)
│   ├── generate_s_matrix_cascaded_layers_pec/         PEC cascade (2, 10, 20 layers)
│   ├── multiple_cyl_cascade/                          Large-array cascade demo (500+ cylinders)
│   ├── differentiable_s_matrix_wigner_smith/          Wigner-Smith time-delay via ∂S/∂ω
│   ├── maze_s_matrix_wavefront_optimization/          PEC maze eigenchannel routing
│   ├── differentiable_s_matrix_demo/                  ∂S21/∂x, ∂S21/∂λ, ∂S21/∂n via ForwardDiff + FD
│   ├── optimize_refractive_index_and_radius/          Adam optimization of n and r via ForwardDiff
│   └── generate_wave_demo/                            Wave field animation (normal + optimal wavefront)
│
├── notebooks/
│   └── CyScat_Demo.jl                     Interactive Pluto notebook
│
├── Project.toml
└── Manifest.toml
```

## Setup

```bash
cd CyScat_Julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Examples

### Single-Layer S-Matrix (Dielectric)

Compute the S-matrix for 10 dielectric cylinders (n=1.3). Draws geometry, plots transmission eigenvalues, and generates wave field movies for normal incidence and optimal wavefront.

```bash
julia --project=. examples/generate_s_matrix_1layer_dielectric/generate_s_matrix_1layer_dielectric.jl
```

### Single-Layer S-Matrix (PEC)

Same as above for 10 perfectly conducting (PEC) cylinders. No evanescent modes needed; includes unitarity check.

```bash
julia --project=. examples/generate_s_matrix_1layer_pec/generate_s_matrix_1layer_pec.jl
```

### Cascaded Layers (Dielectric)

Compute a single-layer S-matrix for dielectric cylinders (n=1.3), then cascade it with itself to build 2, 10, and 20 identical layers via the Redheffer star product. Shows how transmission decays with layer count.

```bash
julia --project=. examples/generate_s_matrix_cascaded_layers_dielectric/generate_s_matrix_cascaded_layers_dielectric.jl
```

### Cascaded Layers (PEC)

Same cascade method for perfectly conducting cylinders. Includes unitarity checks at each cascade level (PEC is lossless).

```bash
julia --project=. examples/generate_s_matrix_cascaded_layers_pec/generate_s_matrix_cascaded_layers_pec.jl
```

### Large-Array Cascade Demo

Demonstrates partitioning large cylinder arrays (500+) into groups, computing each group's S-matrix independently, then cascading them together — much faster than solving the full system at once.

```bash
julia --project=. examples/multiple_cyl_cascade/cascade_demo.jl
```

### Wigner-Smith Time-Delay Matrix

Compute ∂S/∂ω via ForwardDiff AD, form the Wigner-Smith matrix Q = -iS⁻¹ ∂S/∂ω, and extract the proper delay times (eigenvalues of Q). Cross-validated against finite differences.

```bash
julia --project=. examples/differentiable_s_matrix_wigner_smith/differentiable_s_matrix_wigner_smith.jl
```

### Maze Wavefront Optimization

PEC S-channel maze demonstrating open eigenchannel routing. Normal incidence scatters off the walls; the SVD-optimal wavefront tunnels through the S-shaped channel.

```bash
julia --project=. examples/maze_s_matrix_wavefront_optimization/maze_s_matrix_wavefront_optimization.jl
```

### Differentiable S-Matrix Demo

Compute matrix-valued derivatives ∂S21/∂x (cylinder position), ∂S21/∂λ (wavelength), and ∂S21/∂n (refractive index) via ForwardDiff. Each derivative is a full complex matrix, cross-validated against central finite differences.

```bash
julia --project=. examples/differentiable_s_matrix_demo/differentiable_s_matrix_demo.jl
```

### Optimize Refractive Index and Radius

Adam optimizer with ForwardDiff gradients to minimize/maximize transmission by tuning refractive index or radius. Includes loss surface visualization and optimization trajectory plots.

```bash
julia --project=. examples/optimize_refractive_index_and_radius/optimize_refractive_index_and_radius.jl
```

### Wave Field Animation

Animated EM wave field for a slab of PEC or dielectric cylinders. Shows normal incidence vs. optimal wavefront (SVD of S11), with cylinder outlines and slab boundaries.

```bash
julia --project=. examples/generate_wave_demo/wave_field_movie.jl --pec --num_cyl 50
```

### Interactive Notebook

```bash
julia --project=. -e 'using Pluto; Pluto.run(notebook="notebooks/CyScat_Demo.jl")'
```

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
The entire pipeline is compatible with `ForwardDiff.jl`. Custom chain rules for `besselj`,
`bessely`, and `hankelh2` propagate Dual numbers through Bessel/Hankel function calls.
One forward pass computes `∂f/∂p` for any scalar parameter `p`.

### Wigner-Smith Time-Delay Matrix
From ∂S/∂ω (computed via AD), the Wigner-Smith matrix Q = -iS⁻¹ ∂S/∂ω gives the proper delay times as its eigenvalues — the fundamental time scales of wave transport through the scattering region.

## References

- C. Jin, R. R. Nadakuditi, "CyScat: Electromagnetic Scattering by Periodic Arrays of Cylinders", University of Michigan
- Original MATLAB implementation: Curtis Jin, Prof. Raj Rao Nadakuditi

## License

MIT — see [LICENSE](LICENSE)
