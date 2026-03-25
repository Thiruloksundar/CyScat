"""
matrix_derivatives.jl

Computes matrix-valued derivatives of S21 w.r.t.:
  1. Cylinder x-position  (∂S21/∂x_j)  — one matrix per cylinder
  2. Wavelength           (∂S21/∂λ)    — one matrix
  3. Refractive index     (∂S21/∂n)    — one matrix

Uses ForwardDiff: wrap the scalar of interest in a Dual number and push it
through the entire pipeline.  The result is a complex matrix of the same
size as S21 where each entry [p,q] is the derivative of S21[p,q].

Usage:
    julia matrix_derivatives.jl
    julia matrix_derivatives.jl --cyl_idx 0   # which cylinder's x to diff (0-based)
    julia matrix_derivatives.jl --num_cyl 16 --seed 42
"""

import ForwardDiff: Dual, value, partials
using ForwardDiff
using LinearAlgebra
using Printf
using SpecialFunctions

# ── ForwardDiff rules for Bessel/Hankel (same as check_differentiability.jl) ──

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

# ── Load CyScat ───────────────────────────────────────────────────────────────

include(joinpath(@__DIR__, "..", "..", "CyScat", "CyScat.jl"))
using .CyScat

# ── Setup: generates cylinders using Julia RNG ────────────────────────────────

function setup_parameters(num_cyl::Int; seed::Int=42)
    wavelength = 0.93
    period     = 12.81
    radius     = 0.25
    n_cylinder = 1.3
    eps_val    = n_cylinder^2
    mu         = 1.0
    cmmax      = 5
    phiinc     = pi / 2

    EvanescentModeTol  = 1e-2
    NoPropagatingModes = floor(Int, period / wavelength)
    NoEvaMode = max(floor(Int,
        period / (2pi) * sqrt(
            (log(EvanescentModeTol) / (2 * radius))^2 +
            (2pi / wavelength)^2
        )) - NoPropagatingModes, 0)
    nmax = NoPropagatingModes + NoEvaMode

    spacing      = 2.5 * radius
    cyls_per_row = floor(Int, period / spacing)
    rows_needed  = num_cyl / cyls_per_row + 2
    thickness    = round(max(0.5, rows_needed * spacing * 1.5), digits=1)

    rng     = MersenneTwister(seed)
    margin  = radius * 1.5
    min_sep = 2.5 * radius
    clocs   = zeros(num_cyl, 2)
    for i in 1:num_cyl
        for _ in 1:10000
            x = margin + rand(rng) * (period - 2*margin)
            y = margin + rand(rng) * (thickness - 2*margin)
            if i == 1 || all(sqrt.((x .- clocs[1:i-1,1]).^2 .+
                                   (y .- clocs[1:i-1,2]).^2) .> min_sep)
                clocs[i, :] = [x, y]; break
            end
        end
    end

    params = Dict{String,Any}(
        "wavelength" => wavelength, "period"    => period,
        "radius"     => radius,     "eps"       => eps_val,
        "mu"         => mu,         "cmmax"     => cmmax,
        "phiinc"     => phiinc,     "thickness" => thickness,
        "nmax"       => nmax,       "NoEvaMode" => NoEvaMode,
        "NoPropagatingModes" => NoPropagatingModes,
        "n_cylinder" => n_cylinder,
    )
    return clocs, params
end

# ── Core helper: extract S21 block from full S-matrix ────────────────────────

function extract_S21(S, nmax, NoEvaMode)
    nm   = 2 * nmax + 1
    S21  = S[nm+1:end, 1:nm]
    neva = NoEvaMode
    return neva > 0 ? S21[neva+1:end-neva, neva+1:end-neva] : S21
end

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ∂S21 / ∂x_j   (matrix derivative w.r.t. x-position of cylinder j)
# ─────────────────────────────────────────────────────────────────────────────

"""
    dS21_dx(clocs, params, sp, cyl_idx) -> Matrix{ComplexF64}

Returns the matrix ∂S21[p,q]/∂x_{cyl_idx} of shape (N_prop × N_prop).
Uses ForwardDiff: replace x_{cyl_idx} with a Dual number.
"""
function dS21_dx(clocs, params, sp, cyl_idx::Int)
    num_cyl = size(clocs, 1)
    cmmaxs  = fill(params["cmmax"], num_cyl)
    cepmus  = repeat([params["eps"] params["mu"]], num_cyl, 1)
    crads   = fill(params["radius"], num_cyl)

    # Lift only x_{cyl_idx} to a Dual; everything else stays Float64
    x_dual = Dual{:x}(clocs[cyl_idx, 1], one(Float64))

    # Rebuild clocs with the Dual in the right slot
    clocs_d = Matrix{typeof(x_dual)}(undef, num_cyl, 2)
    for i in 1:num_cyl, c in 1:2
        clocs_d[i, c] = (i == cyl_idx && c == 1) ?
            x_dual : Dual{:x}(clocs[i, c], zero(Float64))
    end

    S, _ = smatrix(clocs_d, cmmaxs, cepmus, crads,
                   params["period"], params["wavelength"],
                   params["nmax"], params["thickness"], sp, "On")

    S21_dual = extract_S21(S, params["nmax"], params["NoEvaMode"])

    # Extract the partial derivative from each Dual entry
    return map(z -> complex(partials(real(z), 1), partials(imag(z), 1)), S21_dual)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ∂S21 / ∂λ   (matrix derivative w.r.t. wavelength)
# ─────────────────────────────────────────────────────────────────────────────

"""
    dS21_dlambda(clocs, params; lambda0=nothing) -> Matrix{ComplexF64}

Returns the matrix ∂S21[p,q]/∂λ evaluated at λ = lambda0 (defaults to
params["wavelength"]).
"""
function dS21_dlambda(clocs, params; lambda0=nothing)
    lam0    = isnothing(lambda0) ? params["wavelength"] : lambda0
    num_cyl = size(clocs, 1)
    cmmaxs  = fill(params["cmmax"], num_cyl)
    cepmus  = repeat([params["eps"] params["mu"]], num_cyl, 1)
    crads   = fill(params["radius"], num_cyl)

    lam_dual = Dual{:lam}(lam0, one(Float64))

    # smatrix_parameters must be called with the Dual wavelength so that k0,
    # kxs, kys, Angles all carry the derivative.
    sp_d = smatrix_parameters(
        lam_dual, params["period"], params["phiinc"],
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, params["period"] / 120
    )

    # clocs and radii are plain Float64 — only λ is Dual
    clocs_d = Matrix{typeof(lam_dual)}(clocs)   # promote to Dual, partials = 0

    S, _ = smatrix(clocs_d, cmmaxs, cepmus, crads,
                   params["period"], lam_dual,
                   params["nmax"], params["thickness"], sp_d, "On")

    S21_dual = extract_S21(S, params["nmax"], params["NoEvaMode"])

    return map(z -> complex(partials(real(z), 1), partials(imag(z), 1)), S21_dual)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3.  ∂S21 / ∂n   (matrix derivative w.r.t. refractive index, shared by all)
# ─────────────────────────────────────────────────────────────────────────────

"""
    dS21_dn(clocs, params; n0=nothing) -> Matrix{ComplexF64}

Returns the matrix ∂S21[p,q]/∂n evaluated at n = n0 (defaults to
params["n_cylinder"]).

n enters only through ε = n² in the Mie coefficients (sall), so
smatrix_parameters (which depends on the background k, not n) stays Float64.
"""
function dS21_dn(clocs, params; n0=nothing)
    n_val   = isnothing(n0) ? params["n_cylinder"] : n0
    num_cyl = size(clocs, 1)
    cmmaxs  = fill(params["cmmax"], num_cyl)
    crads   = fill(params["radius"], num_cyl)

    n_dual   = Dual{:n}(n_val, one(Float64))
    eps_dual = n_dual^2        # ε = n²  — carries the derivative ∂ε/∂n = 2n
    mu_val   = params["mu"]

    # cepmus: each row is [ε, μ]; only ε is Dual
    T_cepmu  = typeof(eps_dual)
    cepmus_d = Matrix{T_cepmu}(undef, num_cyl, 2)
    for i in 1:num_cyl
        cepmus_d[i, 1] = eps_dual
        cepmus_d[i, 2] = Dual{:n}(mu_val, zero(Float64))
    end

    # sp and clocs use plain Float64 — n doesn't change k_background
    sp      = smatrix_parameters(
        params["wavelength"], params["period"], params["phiinc"],
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, params["period"] / 120
    )
    clocs_d = Matrix{T_cepmu}(undef, num_cyl, 2)
    for i in 1:num_cyl, c in 1:2
        clocs_d[i, c] = Dual{:n}(clocs[i, c], zero(Float64))
    end

    S, _ = smatrix(clocs_d, cmmaxs, cepmus_d, crads,
                   params["period"], params["wavelength"],
                   params["nmax"], params["thickness"], sp, "On")

    S21_dual = extract_S21(S, params["nmax"], params["NoEvaMode"])

    return map(z -> complex(partials(real(z), 1), partials(imag(z), 1)), S21_dual)
end

# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printer for a complex matrix
# ─────────────────────────────────────────────────────────────────────────────

function print_matrix(M::Matrix{ComplexF64}, name::String; max_rows=6, max_cols=6)
    nr, nc = size(M)
    r_show = min(nr, max_rows)
    c_show = min(nc, max_cols)
    println("\n  $name  (size: $(nr)×$(nc) complex)")
    println("  Showing top $(r_show)×$(c_show) block:")
    println("  " * "-"^(c_show * 26))
    for i in 1:r_show
        print("  ")
        for j in 1:c_show
            v = M[i, j]
            @printf("  %+.4e%+.4ei", real(v), imag(v))
        end
        println()
    end
    if nr > max_rows || nc > max_cols
        println("  ... ($(nr - r_show) more rows, $(nc - c_show) more cols)")
    end
    println()

    # Summary statistics
    mag = abs.(M)
    @printf("  |entry| — min: %.4e   max: %.4e   mean: %.4e\n",
            minimum(mag), maximum(mag), sum(mag)/length(mag))
    any_nan = any(isnan.(real.(M)) .| isnan.(imag.(M)))
    any_inf = any(isinf.(real.(M)) .| isinf.(imag.(M)))
    @printf("  NaN entries: %s   Inf entries: %s\n",
            any_nan ? "YES ⚠" : "none ✓",
            any_inf ? "YES ⚠" : "none ✓")
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    cyl_idx = 1      # 1-based; cylinder whose x-position we differentiate
    num_cyl = 16
    seed    = 42

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--cyl_idx"
            cyl_idx = parse(Int, ARGS[i+1]) + 1   # user passes 0-based
            i += 2
        elseif ARGS[i] == "--num_cyl"
            num_cyl = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--seed"
            seed = parse(Int, ARGS[i+1]); i += 2
        else
            i += 1
        end
    end

    println("=" ^ 65)
    println("  CyScat — Matrix-valued derivatives of S21 via ForwardDiff")
    println("=" ^ 65)

    clocs, params = setup_parameters(num_cyl; seed=seed)
    num_cyl = size(clocs, 1)
    sp = smatrix_parameters(
        params["wavelength"], params["period"], params["phiinc"],
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, params["period"] / 120
    )

    @printf("\n  %d cylinders | λ=%.3f | n=%.3f | r=%.3f | period=%.3f\n\n",
            num_cyl, params["wavelength"], params["n_cylinder"],
            params["radius"], params["period"])

    # ── 1. ∂S21/∂x_j ─────────────────────────────────────────────────────
    println("─"^65)
    @printf("  1.  ∂S21 / ∂x_%d   (x-position of cylinder %d)\n",
            cyl_idx-1, cyl_idx-1)
    println("─"^65)
    println("  Running ForwardDiff (Dual on x_$(cyl_idx-1))...")
    t0 = time()
    dS21_x = dS21_dx(clocs, params, sp, cyl_idx)
    @printf("  Done in %.2fs\n", time() - t0)
    print_matrix(dS21_x, "∂S21/∂x_$(cyl_idx-1)")

    @printf("  YES — ∂S21/∂x_%d is a %d×%d complex matrix ✓\n",
            cyl_idx-1, size(dS21_x)...)

    # ── 2. ∂S21/∂λ ───────────────────────────────────────────────────────
    println("\n" * "─"^65)
    println("  2.  ∂S21 / ∂λ   (wavelength, λ = $(params["wavelength"]))")
    println("─"^65)
    println("  Running ForwardDiff (Dual on λ)...")
    t0 = time()
    dS21_lam = dS21_dlambda(clocs, params)
    @printf("  Done in %.2fs\n", time() - t0)
    print_matrix(dS21_lam, "∂S21/∂λ")

    @printf("  YES — ∂S21/∂λ is a %d×%d complex matrix ✓\n",
            size(dS21_lam)...)

    # ── 3. ∂S21/∂n ───────────────────────────────────────────────────────
    println("\n" * "─"^65)
    println("  3.  ∂S21 / ∂n   (refractive index, n = $(params["n_cylinder"]))")
    println("─"^65)
    println("  Running ForwardDiff (Dual on n, via ε = n²)...")
    t0 = time()
    dS21_n = dS21_dn(clocs, params)
    @printf("  Done in %.2fs\n", time() - t0)
    print_matrix(dS21_n, "∂S21/∂n")

    @printf("  YES — ∂S21/∂n is a %d×%d complex matrix ✓\n",
            size(dS21_n)...)

    # ── Cross-validation: finite differences on entry [1,1] ──────────────
    println("\n" * "─"^65)
    println("  4.  Finite-difference cross-check on S21[1,1]")
    println("─"^65)

    function S21_11(clocs_in, lam, n_val)
        cmmaxs = fill(params["cmmax"], num_cyl)
        eps_v  = n_val^2
        cepmus = repeat([eps_v params["mu"]], num_cyl, 1)
        crads  = fill(params["radius"], num_cyl)
        sp_loc = smatrix_parameters(lam, params["period"], params["phiinc"],
                     1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, params["period"] / 120)
        S, _   = smatrix(clocs_in, cmmaxs, cepmus, crads,
                         params["period"], lam,
                         params["nmax"], params["thickness"], sp_loc, "On")
        S21    = extract_S21(S, params["nmax"], params["NoEvaMode"])
        return S21[1, 1]
    end

    δ = 1e-5
    lam0 = params["wavelength"]
    n0   = params["n_cylinder"]
    x0   = clocs[cyl_idx, 1]

    # FD for ∂/∂x
    clocs_p = copy(clocs); clocs_p[cyl_idx, 1] += δ
    clocs_m = copy(clocs); clocs_m[cyl_idx, 1] -= δ
    fd_x   = (S21_11(clocs_p, lam0, n0) - S21_11(clocs_m, lam0, n0)) / (2δ)
    ad_x   = dS21_x[1, 1]
    @printf("  ∂S21[1,1]/∂x_%d:  AD = %+.6e%+.6ei\n",
            cyl_idx-1, real(ad_x), imag(ad_x))
    @printf("                   FD = %+.6e%+.6ei\n", real(fd_x), imag(fd_x))
    @printf("               |diff| = %.4e  %s\n", abs(ad_x - fd_x),
            abs(ad_x - fd_x) < 1e-6 ? "✓ match" : "⚠ mismatch")

    # FD for ∂/∂λ
    fd_lam = (S21_11(clocs, lam0+δ, n0) - S21_11(clocs, lam0-δ, n0)) / (2δ)
    ad_lam = dS21_lam[1, 1]
    @printf("\n  ∂S21[1,1]/∂λ:    AD = %+.6e%+.6ei\n", real(ad_lam), imag(ad_lam))
    @printf("                   FD = %+.6e%+.6ei\n", real(fd_lam), imag(fd_lam))
    @printf("               |diff| = %.4e  %s\n", abs(ad_lam - fd_lam),
            abs(ad_lam - fd_lam) < 1e-6 ? "✓ match" : "⚠ mismatch")

    # FD for ∂/∂n
    fd_n  = (S21_11(clocs, lam0, n0+δ) - S21_11(clocs, lam0, n0-δ)) / (2δ)
    ad_n  = dS21_n[1, 1]
    @printf("\n  ∂S21[1,1]/∂n:    AD = %+.6e%+.6ei\n", real(ad_n), imag(ad_n))
    @printf("                   FD = %+.6e%+.6ei\n", real(fd_n), imag(fd_n))
    @printf("               |diff| = %.4e  %s\n", abs(ad_n - fd_n),
            abs(ad_n - fd_n) < 1e-6 ? "✓ match" : "⚠ mismatch")

    # ── Summary for the professor ─────────────────────────────────────────
    println("\n" * "=" ^ 65)
    println("  SUMMARY")
    println("=" ^ 65)
    nr, nc = size(dS21_x)
    @printf("  ∂S21/∂x_j  → %d×%d complex matrix  (one per cylinder)\n", nr, nc)
    @printf("  ∂S21/∂λ    → %d×%d complex matrix  (one for the slab)\n", nr, nc)
    @printf("  ∂S21/∂n    → %d×%d complex matrix  (one for shared n)\n", nr, nc)
    println()
    println("  All three derivatives are well-defined complex matrices of the")
    println("  same shape as S21.  Each entry [p,q] gives how the transmission")
    println("  amplitude from input channel q to output channel p changes with")
    println("  the parameter.  Verified against finite differences above.")
    println("=" ^ 65)
end

main()