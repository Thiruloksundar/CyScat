"""
run_julia_derivatives.jl

Reads cylinder positions from shared_params.npz (written by the Python notebook),
computes S21 and matrix-valued derivatives using ForwardDiff, and saves results
to julia_derivatives.npz for comparison in the notebook.

Derivatives computed:
  1. ∂S21 / ∂x_0    — ForwardDiff (Dual on x-position of cylinder 0)
  2. ∂S21 / ∂λ      — ForwardDiff (Dual on wavelength)
  3. ∂S21 / ∂n      — ForwardDiff (Dual on refractive index, via ε = n²)

Run from within the compare_julia_python_derivatives/ folder:
    julia run_julia_derivatives.jl
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, "../../../../julia"))

using NPZ, LinearAlgebra, Printf, SpecialFunctions
import ForwardDiff: Dual, value, partials

# ── ForwardDiff rules for Bessel/Hankel ──────────────────────────────────────

function SpecialFunctions.besselj(nu::Integer, x::Dual{T,V,N}) where {T,V,N}
    xv = value(x)
    j_val  = besselj(nu, xv)
    dj_val = (besselj(nu - 1, xv) - besselj(nu + 1, xv)) / 2
    return Dual{T}(j_val, dj_val * partials(x))
end

function SpecialFunctions.bessely(nu::Integer, x::Dual{T,V,N}) where {T,V,N}
    xv = value(x)
    y_val  = bessely(nu, xv)
    dy_val = (bessely(nu - 1, xv) - bessely(nu + 1, xv)) / 2
    return Dual{T}(y_val, dy_val * partials(x))
end

function SpecialFunctions.hankelh2(nu::Integer, x::Dual{T,V,N}) where {T,V,N}
    xv     = value(x)
    h_val  = hankelh2(nu, xv)
    dh_val = (hankelh2(nu - 1, xv) - hankelh2(nu + 1, xv)) / 2
    re_d   = Dual{T}(real(h_val), real(dh_val) * partials(x))
    im_d   = Dual{T}(imag(h_val), imag(dh_val) * partials(x))
    return Complex(re_d, im_d)
end

Base.isfinite(x::Complex{<:Dual}) = isfinite(value(real(x))) && isfinite(value(imag(x)))
Base.isnan(x::Complex{<:Dual})    = isnan(value(real(x)))    || isnan(value(imag(x)))
Base.isinf(x::Complex{<:Dual})    = isinf(value(real(x)))    || isinf(value(imag(x)))

# ── Load CyScat ───────────────────────────────────────────────────────────────

include(joinpath(@__DIR__, "../../../../julia/CyScat/CyScat.jl"))
using .CyScat

# ── Load shared parameters ────────────────────────────────────────────────────

println("Loading shared_params.npz ...")
p = npzread(joinpath(@__DIR__, "shared_params.npz"))

clocs      = p["clocs"]                          # (N, 2)
wavelength = Float64(p["wavelength"][])
period     = Float64(p["period"][])
phiinc     = Float64(p["phiinc"][])
radius     = Float64(p["radius"][])
n_cyl      = Float64(p["n_cyl"][])
mu         = Float64(p["mu"][])
cmmax      = Int(p["cmmax"][])
nmax       = Int(p["nmax"][])
n_eva      = Int(p["n_eva"][])
thickness  = Float64(p["thickness"][])

num_cyl = size(clocs, 1)
nm      = 2 * nmax + 1
cmmaxs  = fill(cmmax, num_cyl)
crads   = fill(radius, num_cyl)
eps_val = n_cyl^2
cepmus  = repeat([eps_val mu], num_cyl, 1)

@printf("  %d cylinders | λ=%.4f | n=%.4f | r=%.4f | nmax=%d | n_eva=%d\n",
        num_cyl, wavelength, n_cyl, radius, nmax, n_eva)

# ── Baseline S-matrix ─────────────────────────────────────────────────────────

sp = smatrix_parameters(wavelength, period, phiinc,
                        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, period / 120)

println("\nComputing baseline S-matrix ...")
t0 = time()
S, _ = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength,
               nmax, thickness, sp, "On")
@printf("  Done in %.2fs\n", time() - t0)

S21_full = S[nm+1:end, 1:nm]
S21      = n_eva > 0 ? S21_full[n_eva+1:end-n_eva, n_eva+1:end-n_eva] : S21_full
@printf("  S21 shape: %d × %d\n", size(S21)...)

# Helper: extract propagating-mode S21 from a full S-matrix with Dual entries
function extract_S21(S_full, nmax, n_eva)
    nm   = 2 * nmax + 1
    S21f = S_full[nm+1:end, 1:nm]
    return n_eva > 0 ? S21f[n_eva+1:end-n_eva, n_eva+1:end-n_eva] : S21f
end

# Helper: extract partial w.r.t. the single tagged Dual
extract_partial(z::Complex{<:Dual}) =
    complex(partials(real(z), 1), partials(imag(z), 1))

# ── 1. ∂S21 / ∂x_0  (ForwardDiff on x-position of cylinder 0) ───────────────

println("\n1. Computing ∂S21/∂x_0 via ForwardDiff ...")
t0 = time()

x0_dual = Dual{:x}(clocs[1, 1], one(Float64))
T_d     = typeof(x0_dual)
clocs_d = Matrix{T_d}(undef, num_cyl, 2)
for i in 1:num_cyl, c in 1:2
    v = clocs[i, c]
    clocs_d[i, c] = (i == 1 && c == 1) ? x0_dual : Dual{:x}(v, zero(Float64))
end
cepmus_d = Matrix{T_d}(repeat([Dual{:x}(eps_val, 0.0)  Dual{:x}(mu, 0.0)], num_cyl, 1))
crads_d  = fill(Dual{:x}(radius, 0.0), num_cyl)

S_d, _ = smatrix(clocs_d, cmmaxs, cepmus_d, crads_d,
                 period, Dual{:x}(wavelength, 0.0),
                 nmax, thickness, sp, "On")
dS21_dx0 = extract_partial.(extract_S21(S_d, nmax, n_eva))
@printf("  Done in %.2fs  ||∂S21/∂x_0||_F = %.6e\n",
        time() - t0, norm(dS21_dx0))

# ── 2. ∂S21 / ∂λ  (ForwardDiff on wavelength) ────────────────────────────────

println("\n2. Computing ∂S21/∂λ via ForwardDiff ...")
t0 = time()

lam_dual = Dual{:lam}(wavelength, one(Float64))
T_l      = typeof(lam_dual)
sp_d     = smatrix_parameters(lam_dual, period, phiinc,
                               1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, period / 120)
clocs_l  = Matrix{T_l}(clocs)
cepmus_l = Matrix{T_l}(repeat([eps_val mu], num_cyl, 1))
crads_l  = fill(T_l(radius), num_cyl)

S_l, _ = smatrix(clocs_l, cmmaxs, cepmus_l, crads_l,
                 period, lam_dual, nmax, thickness, sp_d, "On")
dS21_dlam = extract_partial.(extract_S21(S_l, nmax, n_eva))
@printf("  Done in %.2fs  ||∂S21/∂λ||_F = %.6e\n",
        time() - t0, norm(dS21_dlam))

# ── 3. ∂S21 / ∂n  (ForwardDiff on refractive index, via ε = n²) ─────────────

println("\n3. Computing ∂S21/∂n via ForwardDiff ...")
t0 = time()

n_dual   = Dual{:n}(n_cyl, one(Float64))
eps_dual = n_dual^2
T_n      = typeof(eps_dual)
cepmus_n = Matrix{T_n}(undef, num_cyl, 2)
for i in 1:num_cyl
    cepmus_n[i, 1] = eps_dual
    cepmus_n[i, 2] = T_n(mu)
end
clocs_n = Matrix{T_n}(clocs)
crads_n = fill(T_n(radius), num_cyl)

S_n, _ = smatrix(clocs_n, cmmaxs, cepmus_n, crads_n,
                 period, wavelength, nmax, thickness, sp, "On")
dS21_dn = extract_partial.(extract_S21(S_n, nmax, n_eva))
@printf("  Done in %.2fs  ||∂S21/∂n||_F = %.6e\n",
        time() - t0, norm(dS21_dn))

# ── Save results ──────────────────────────────────────────────────────────────

outfile = joinpath(@__DIR__, "julia_derivatives.npz")
npzwrite(outfile, Dict(
    "S21_real"       => real.(S21),
    "S21_imag"       => imag.(S21),
    "dS21_dx0_real"  => real.(dS21_dx0),
    "dS21_dx0_imag"  => imag.(dS21_dx0),
    "dS21_dlam_real" => real.(dS21_dlam),
    "dS21_dlam_imag" => imag.(dS21_dlam),
    "dS21_dn_real"   => real.(dS21_dn),
    "dS21_dn_imag"   => imag.(dS21_dn),
))
println("\nSaved: julia_derivatives.npz")
println("Done.")
