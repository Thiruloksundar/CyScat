"""
differentiable_s_matrix_wigner_smith.jl
========================================
Compute the Wigner-Smith time-delay matrix Q = -i S⁻¹ ∂S/∂ω for a slab of
10 dielectric cylinders, using automatic differentiation (ForwardDiff).

The Wigner-Smith matrix is the key object connecting scattering theory to
time-delay physics.  Its eigenvalues are the proper delay times of the
scattering channels.

Steps:
  1. Compute S(ω) for 10 cylinders via the CyScat pipeline
  2. Compute ∂S/∂ω via ForwardDiff (AD through the full pipeline)
  3. Form Q = -i S⁻¹ ∂S/∂ω and extract eigenvalues (proper delay times)
  4. Cross-validate ∂S/∂ω against central finite differences

Here ω = 2πc/λ, so ∂S/∂ω = ∂S/∂λ · ∂λ/∂ω = ∂S/∂λ · (-λ²/(2πc)).
We work in units where c=1, so ω = 2π/λ.

Usage:
    julia differentiable_s_matrix_wigner_smith.jl
    julia differentiable_s_matrix_wigner_smith.jl --num_cyl 10 --seed 42
"""

using LinearAlgebra
using Printf
using Random
using ForwardDiff
using SpecialFunctions
using Plots

# ── ForwardDiff rules for Bessel/Hankel ──────────────────────────────────────
import ForwardDiff: Dual, value, partials

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
    xv = value(x)
    h_val  = hankelh2(nu, xv)
    dh_val = (hankelh2(nu - 1, xv) - hankelh2(nu + 1, xv)) / 2
    real_dual = Dual{T}(real(h_val), real(dh_val) * partials(x))
    imag_dual = Dual{T}(imag(h_val), imag(dh_val) * partials(x))
    return Complex(real_dual, imag_dual)
end

Base.isfinite(x::Complex{<:Dual}) = isfinite(value(real(x))) && isfinite(value(imag(x)))
Base.isnan(x::Complex{<:Dual})    = isnan(value(real(x)))    || isnan(value(imag(x)))
Base.isinf(x::Complex{<:Dual})    = isinf(value(real(x)))    || isinf(value(imag(x)))

# ── Load CyScat ──────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "..", "CyScat", "CyScat.jl"))
using .CyScat

# ── Physical constants ───────────────────────────────────────────────────────
const OUTDIR     = @__DIR__
const WAVELENGTH = 0.93
const PERIOD     = 12.81
const RADIUS     = 0.25
const N_CYL_REF  = 1.3
const MU         = 1.0
const CMMAX      = 5
const PHIINC     = π / 2
const Eva_TOL    = 1e-2

# ── CLI args ─────────────────────────────────────────────────────────────────
function parse_args()
    num_cyl = 10; seed = 42
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--num_cyl" && i < length(ARGS); num_cyl = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--seed" && i < length(ARGS); seed = parse(Int, ARGS[i+1]); i += 2
        else i += 1 end
    end
    return num_cyl, seed
end

# ── Setup ────────────────────────────────────────────────────────────────────
function setup_parameters(num_cyl; seed=42)
    n_prop = floor(Int, PERIOD / WAVELENGTH)
    n_eva  = max(floor(Int,
        PERIOD / (2π) * sqrt((log(Eva_TOL) / (2*RADIUS))^2 + (2π/WAVELENGTH)^2)
    ) - n_prop, 0)
    nmax = n_prop + n_eva

    spacing      = 2.5 * RADIUS
    cyls_per_row = floor(Int, PERIOD / spacing)
    rows_needed  = num_cyl / cyls_per_row + 2
    thickness    = round(max(0.5, rows_needed * spacing * 1.5), digits=1)

    rng = MersenneTwister(seed); margin = RADIUS * 1.5; min_sep = 2.5 * RADIUS
    clocs = zeros(num_cyl, 2)
    for i in 1:num_cyl
        for _ in 1:10000
            x = margin + rand(rng) * (PERIOD - 2*margin)
            y = margin + rand(rng) * (thickness - 2*margin)
            if i == 1 || all(sqrt.((x .- clocs[1:i-1,1]).^2 .+
                                   (y .- clocs[1:i-1,2]).^2) .> min_sep)
                clocs[i,:] = [x, y]; break
            end
        end
    end

    return clocs, Dict{String,Any}(
        "wavelength" => WAVELENGTH, "period" => PERIOD, "radius" => RADIUS,
        "eps" => N_CYL_REF^2, "mu" => MU, "cmmax" => CMMAX,
        "phiinc" => PHIINC, "thickness" => thickness,
        "nmax" => nmax, "NoEvaMode" => n_eva, "NoPropagatingModes" => n_prop,
    )
end

# ── Compute full S-matrix at a given wavelength (can be Dual) ────────────────
function compute_S_at_lambda(clocs, params, lambda)
    num_cyl = size(clocs, 1)
    cmmaxs  = fill(params["cmmax"], num_cyl)
    cepmus  = repeat([params["eps"] params["mu"]], num_cyl, 1)
    crads   = fill(params["radius"], num_cyl)

    sp = smatrix_parameters(lambda, params["period"], params["phiinc"],
                            1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, params["period"] / 120)

    # Promote clocs to match Dual type if needed
    T_lam = typeof(lambda)
    clocs_d = Matrix{T_lam}(clocs)

    S, _ = smatrix(clocs_d, cmmaxs, cepmus, crads,
                   params["period"], lambda,
                   params["nmax"], params["thickness"], sp, "On")
    return S
end

# ── Extract truncated S-matrix (propagating modes only) ──────────────────────
function truncate_S(S, nmax, n_eva)
    nm = 2*nmax + 1
    if n_eva > 0
        r = n_eva+1 : nm-n_eva
        S11 = S[r, r]
        S12 = S[r, nm .+ r]
        S21 = S[nm .+ r, r]
        S22 = S[nm .+ r, nm .+ r]
        return [S11 S12; S21 S22]
    else
        return S
    end
end

# ── AD: compute ∂S/∂λ via ForwardDiff ───────────────────────────────────────
function dS_dlambda_AD(clocs, params)
    lam0 = params["wavelength"]
    lam_dual = Dual{:lam}(lam0, one(Float64))

    S_dual = compute_S_at_lambda(clocs, params, lam_dual)
    nmax = params["nmax"]; n_eva = params["NoEvaMode"]
    S_trunc = truncate_S(S_dual, nmax, n_eva)

    # Extract value and partials
    S_val  = map(z -> complex(value(real(z)), value(imag(z))), S_trunc)
    dS_val = map(z -> complex(partials(real(z), 1), partials(imag(z), 1)), S_trunc)
    return S_val, dS_val
end

# ── FD: central finite difference ∂S/∂λ ─────────────────────────────────────
function dS_dlambda_FD(clocs, params; delta=1e-6)
    lam0 = params["wavelength"]
    nmax = params["nmax"]; n_eva = params["NoEvaMode"]

    S_p = compute_S_at_lambda(clocs, params, lam0 + delta)
    S_m = compute_S_at_lambda(clocs, params, lam0 - delta)

    S_p_t = truncate_S(S_p, nmax, n_eva)
    S_m_t = truncate_S(S_m, nmax, n_eva)

    return (S_p_t - S_m_t) / (2 * delta)
end

# ── Main ─────────────────────────────────────────────────────────────────────
function main()
    num_cyl, seed = parse_args()

    println("=" ^ 65)
    println("  CyScat — Wigner-Smith Time-Delay Matrix via AD")
    println("=" ^ 65)
    @printf("  λ=%.2f  Λ=%.2f  r=%.2f  n=%.2f  N_cyl=%d\n",
            WAVELENGTH, PERIOD, RADIUS, N_CYL_REF, num_cyl)

    clocs, params = setup_parameters(num_cyl; seed=seed)
    @printf("  nmax=%d  thickness=%.1f\n\n", params["nmax"], params["thickness"])

    # ── Step 1: Compute S(λ) ─────────────────────────────────────────────────
    println("  [1] Computing S-matrix at λ = $(WAVELENGTH)...")
    t0 = time()
    S0 = compute_S_at_lambda(clocs, params, WAVELENGTH)
    @printf("  Done in %.2fs\n", time() - t0)

    nmax = params["nmax"]; n_eva = params["NoEvaMode"]
    S_trunc = truncate_S(S0, nmax, n_eva)
    n_prop = 2 * params["NoPropagatingModes"] + 1
    @printf("  Full S: %d×%d  →  Truncated S: %d×%d\n",
            size(S0)..., size(S_trunc)...)

    # Unitarity check
    unit_err = norm(S_trunc' * S_trunc - I)
    @printf("  ‖S'S − I‖ = %.2e  %s\n", unit_err, unit_err < 1e-3 ? "✓" : "⚠")

    # ── Step 2: Compute ∂S/∂λ via AD ────────────────────────────────────────
    println("\n  [2] Computing ∂S/∂λ via ForwardDiff...")
    t0 = time()
    S_val, dS_dlam = dS_dlambda_AD(clocs, params)
    @printf("  Done in %.2fs\n", time() - t0)
    @printf("  max|∂S/∂λ| = %.4e\n", maximum(abs.(dS_dlam)))

    # ── Step 3: Form Wigner-Smith matrix Q = -i S⁻¹ ∂S/∂ω ──────────────────
    println("\n  [3] Computing Wigner-Smith matrix Q = -i S⁻¹ ∂S/∂ω...")

    # Convert ∂S/∂λ to ∂S/∂ω.  Since ω = 2π/λ, ∂λ/∂ω = -λ²/(2π)
    # so ∂S/∂ω = ∂S/∂λ · (-λ²/(2π))
    lam0 = params["wavelength"]
    dS_domega = dS_dlam .* (-lam0^2 / (2π))

    Q = -1im * (S_val \ dS_domega)

    # Eigenvalues of Q are the proper delay times
    delay_times = real.(eigvals(Q))
    sort!(delay_times)

    @printf("  Q is %d×%d\n", size(Q)...)
    @printf("  Delay times (eigenvalues of Q):\n")
    for (i, τ) in enumerate(delay_times)
        @printf("    τ_%d = %+.6f\n", i, τ)
    end
    @printf("  Mean delay = %.6f\n", mean(delay_times))

    # Check Hermiticity of Q (should be Hermitian for lossless structures)
    herm_err = norm(Q - Q') / norm(Q)
    @printf("  ‖Q − Q'‖/‖Q‖ = %.2e  %s\n", herm_err,
            herm_err < 1e-6 ? "(Hermitian ✓)" : "(non-Hermitian ⚠)")

    # ── Step 4: Cross-validate with FD ───────────────────────────────────────
    println("\n  [4] Cross-validating ∂S/∂λ against finite differences...")
    t0 = time()
    dS_fd = dS_dlambda_FD(clocs, params; delta=1e-6)
    @printf("  FD done in %.2fs\n", time() - t0)

    err = norm(dS_dlam - dS_fd) / norm(dS_dlam)
    @printf("  ‖AD − FD‖ / ‖AD‖ = %.4e  %s\n", err,
            err < 1e-4 ? "✓ match" : "⚠ mismatch")

    # Show a few entries
    println("\n  Sample entries of ∂S/∂λ:")
    @printf("  %10s  %28s  %28s  %10s\n", "Index", "AD", "FD", "|diff|")
    println("  " * "-" ^ 80)
    n_show = min(5, size(dS_dlam, 1))
    for i in 1:n_show
        ad_v = dS_dlam[i, i]
        fd_v = dS_fd[i, i]
        @printf("  [%d,%d]      %+.6e%+.6ei    %+.6e%+.6ei    %.2e\n",
                i, i, real(ad_v), imag(ad_v), real(fd_v), imag(fd_v),
                abs(ad_v - fd_v))
    end

    # ── Plots ────────────────────────────────────────────────────────────────
    println("\n  [5] Generating plots...")

    # Delay time spectrum
    p1 = bar(1:length(delay_times), delay_times,
             xlabel="Channel index", ylabel="Delay time τ",
             title="Wigner-Smith delay times ($num_cyl cylinders)",
             legend=false, color=:teal, size=(700, 400))
    hline!(p1, [0.0], color=:gray, lw=1, ls=:dash, label="")
    savefig(p1, joinpath(OUTDIR, "wigner_smith_delay_times.png"))
    println("  Saved → wigner_smith_delay_times.png")

    # |∂S/∂λ| heatmap
    p2 = heatmap(abs.(dS_dlam), color=:viridis,
                 xlabel="Column", ylabel="Row",
                 title="|∂S/∂λ| (AD)", size=(600, 500))
    savefig(p2, joinpath(OUTDIR, "dS_dlambda_magnitude.png"))
    println("  Saved → dS_dlambda_magnitude.png")

    # AD vs FD comparison for diagonal entries
    diag_ad = [abs(dS_dlam[i,i]) for i in 1:size(dS_dlam,1)]
    diag_fd = [abs(dS_fd[i,i])   for i in 1:size(dS_fd,1)]
    p3 = plot(diag_ad, label="AD", marker=:circle, lw=2, color=:blue)
    plot!(p3, diag_fd, label="FD", marker=:diamond, lw=2, ls=:dash, color=:red)
    plot!(p3, xlabel="Diagonal index", ylabel="|∂S[i,i]/∂λ|",
          title="AD vs FD — diagonal of ∂S/∂λ", size=(700, 400))
    savefig(p3, joinpath(OUTDIR, "dS_dlambda_ad_vs_fd.png"))
    println("  Saved → dS_dlambda_ad_vs_fd.png")

    println("\n" * "=" ^ 65)
    println("  SUMMARY")
    println("=" ^ 65)
    @printf("  S-matrix: %d×%d (propagating modes)\n", size(S_trunc)...)
    @printf("  ∂S/∂λ:    computed via ForwardDiff AD, verified against FD\n")
    @printf("  ∂S/∂ω:    = ∂S/∂λ · (-λ²/2π)\n")
    @printf("  Q:        = -i S⁻¹ ∂S/∂ω  (Wigner-Smith time-delay matrix)\n")
    @printf("  Delay times: %d eigenvalues, mean = %.6f\n",
            length(delay_times), mean(delay_times))
    @printf("  AD–FD relative error: %.2e\n", err)
    println("=" ^ 65)
end

main()
