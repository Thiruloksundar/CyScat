"""
optimize_n_r.jl
===============
Fix 20 cylinder positions, define f(param) = norm(S21(param) * x)
where x is a fixed random complex vector.

Optimize:
  A) Refractive index n (shared across all cylinders), r fixed
  B) Radius r (shared across all cylinders), n fixed

Improvements over v1:
  - Adam optimizer (momentum + adaptive learning rate)
  - Log-space parameterization (param always stays positive)
  - Best-so-far tracking (reports best value visited, not final value)

Usage:
    julia optimize_n_r.jl
"""

using LinearAlgebra
using Printf
using Random
using ForwardDiff
using SpecialFunctions
using DelimitedFiles
using Plots

import ForwardDiff: Dual, value, partials

# ── ForwardDiff rules for Bessel/Hankel ───────────────────────────────────────

function SpecialFunctions.besselj(nu::Integer, x::Dual{T,V,N}) where {T,V,N}
    xv     = value(x)
    j_val  = besselj(nu, xv)
    dj_val = (besselj(nu-1, xv) - besselj(nu+1, xv)) / 2
    return Dual{T}(j_val, dj_val * partials(x))
end

function SpecialFunctions.bessely(nu::Integer, x::Dual{T,V,N}) where {T,V,N}
    xv     = value(x)
    y_val  = bessely(nu, xv)
    dy_val = (bessely(nu-1, xv) - bessely(nu+1, xv)) / 2
    return Dual{T}(y_val, dy_val * partials(x))
end

function SpecialFunctions.hankelh2(nu::Integer, x::Dual{T,V,N}) where {T,V,N}
    xv     = value(x)
    h_val  = hankelh2(nu, xv)
    dh_val = (hankelh2(nu-1, xv) - hankelh2(nu+1, xv)) / 2
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

# ── Parameters ────────────────────────────────────────────────────────────────

const OUTDIR     = @__DIR__
const WAVELENGTH = 0.93
const PERIOD     = 12.81
const PHIINC     = π / 2
const MU         = 1.0
const CMMAX      = 5
const N_CYL      = 20
const SEED_CLOCS = 42
const SEED_X     = 7

const EVA_TOL = 1e-2
const N_PROP  = floor(Int, PERIOD / WAVELENGTH)
const N_EVA   = max(floor(Int,
    PERIOD / (2π) * sqrt(
        (log(EVA_TOL) / (2 * 0.25))^2 + (2π / WAVELENGTH)^2
    )) - N_PROP, 0)
const NMAX = N_PROP + N_EVA

# ── Generate fixed cylinder positions ────────────────────────────────────────

function make_clocs(num_cyl::Int, radius::Float64=0.25; seed::Int=SEED_CLOCS)
    rng       = MersenneTwister(seed)
    margin    = radius * 1.5
    spacing   = 2.5 * radius
    rows      = num_cyl / floor(Int, PERIOD / spacing) + 2
    thickness = round(max(0.5, rows * spacing * 1.5), digits=1)
    clocs     = zeros(num_cyl, 2)
    for i in 1:num_cyl
        for _ in 1:10000
            x = margin + rand(rng) * (PERIOD - 2*margin)
            y = margin + rand(rng) * (thickness - 2*margin)
            ok = i == 1 || minimum(sqrt.((x .- clocs[1:i-1,1]).^2 .+
                                         (y .- clocs[1:i-1,2]).^2)) > spacing
            if ok
                clocs[i,:] = [x, y]
                break
            end
        end
    end
    return clocs, thickness
end

function make_sp()
    return smatrix_parameters(
        WAVELENGTH, PERIOD, PHIINC,
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, PERIOD / 120
    )
end

# ── Extract truncated S21 block ───────────────────────────────────────────────

function extract_S21T(S, nmax::Int, neva::Int)
    nm  = 2 * nmax + 1
    S21 = S[nm+1:end, 1:nm]
    return neva > 0 ? S21[neva+1:end-neva, neva+1:end-neva] : S21
end

# ── Objective functions ───────────────────────────────────────────────────────

function objective_n(n_val, clocs, r_fixed::Float64, x_vec, sp, thickness)
    num_cyl = size(clocs, 1)
    cmmaxs  = fill(CMMAX, num_cyl)
    eps_val = n_val^2
    T_ep    = typeof(eps_val)
    cepmus  = Matrix{T_ep}(undef, num_cyl, 2)
    for i in 1:num_cyl
        cepmus[i,1] = eps_val
        cepmus[i,2] = convert(T_ep, MU)
    end
    crads   = fill(r_fixed, num_cyl)
    clocs_d = Matrix{T_ep}(undef, num_cyl, 2)
    for i in 1:num_cyl, c in 1:2
        clocs_d[i,c] = convert(T_ep, clocs[i,c])
    end
    S, _  = smatrix(clocs_d, cmmaxs, cepmus, crads,
                    PERIOD, WAVELENGTH, NMAX, thickness, sp, "On")
    S21T  = extract_S21T(S, NMAX, N_EVA)
    return norm(S21T * x_vec)
end

function objective_r(r_val, clocs, n_fixed::Float64, x_vec, sp, thickness)
    num_cyl = size(clocs, 1)
    cmmaxs  = fill(CMMAX, num_cyl)
    eps_val = n_fixed^2
    T_r     = typeof(r_val)
    cepmus  = Matrix{Float64}(undef, num_cyl, 2)
    for i in 1:num_cyl
        cepmus[i,1] = eps_val
        cepmus[i,2] = MU
    end
    crads   = fill(r_val, num_cyl)
    clocs_d = Matrix{T_r}(undef, num_cyl, 2)
    for i in 1:num_cyl, c in 1:2
        clocs_d[i,c] = convert(T_r, clocs[i,c])
    end
    S, _  = smatrix(clocs_d, cmmaxs, cepmus, crads,
                    PERIOD, WAVELENGTH, NMAX, thickness, sp, "On")
    S21T  = extract_S21T(S, NMAX, N_EVA)
    return norm(S21T * x_vec)
end

# ── Adam optimizer ────────────────────────────────────────────────────────────
#
# Three improvements over plain gradient descent:
#
#   1. LOG-SPACE: optimise log(param) so param is always positive and gradient
#      scale is consistent. Chain rule: df/d(log_p) = df/dp * p.
#
#   2. ADAM MOMENTUM: m and v accumulate gradient history, which:
#      - smooths oscillations (was the main problem with r)
#      - adapts step size: large steps when gradient is consistent,
#        small steps when gradient is noisy/oscillating
#
#   3. BEST-SO-FAR: records the best (f, param) ever visited during the run.
#      Adam can overshoot and recover; we want the best point, not the last.
#      Best values are appended at the end of the history arrays so that
#      p_hist[end] and f_hist[end] always give the best result.

function adam_optimize(obj_fn, param0::Float64;
                       maximize::Bool     = false,
                       lr::Float64        = 0.1,
                       n_steps::Int       = 25,
                       p_min::Float64     = 1e-6,
                       p_max::Float64     = Inf,
                       beta1::Float64     = 0.9,
                       beta2::Float64     = 0.999,
                       eps::Float64       = 1e-8,
                       param_name::String = "p")

    sign  = maximize ? -1.0 : 1.0
    log_p = log(param0)
    m, v  = 0.0, 0.0

    p_hist = Float64[param0]
    f0     = obj_fn(param0)
    f_hist = Float64[f0]
    @printf("    step  0: %s=%.4f  f=%.6f\n", param_name, param0, f0)

    best_f = f0
    best_p = param0

    for t in 1:n_steps
        t0        = time()
        p_current = exp(log_p)

        # ForwardDiff.derivative: ONE forward pass with Dual — gets grad
        grad_p  = ForwardDiff.derivative(obj_fn, p_current)
        fval    = obj_fn(p_current)

        # Chain rule to log-space
        grad_log = grad_p * p_current

        # Adam update
        m     = beta1 * m + (1 - beta1) * grad_log
        v     = beta2 * v + (1 - beta2) * grad_log^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        log_p = log_p - sign * lr * m_hat / (sqrt(v_hat) + eps)
        log_p = clamp(log_p, log(p_min), log(p_max))

        push!(p_hist, exp(log_p))
        push!(f_hist, fval)

        # Track best
        if (maximize && fval > best_f) || (!maximize && fval < best_f)
            best_f = fval
            best_p = p_current
        end

        @printf("    step %2d: %s=%.4f  f=%.6f  grad=%+.3e  (%.1fs)\n",
                t, param_name, exp(log_p), fval, grad_log, time()-t0)
    end

    # Append best as final entry — caller uses p_hist[end], f_hist[end]
    push!(p_hist, best_p)
    push!(f_hist, best_f)
    @printf("    ★ Best: %s=%.4f  f=%.6f\n", param_name, best_p, best_f)

    return p_hist, f_hist
end

# ── Dense scan ────────────────────────────────────────────────────────────────

function scan(obj_fn, param_vals; param_name="p")
    fvals = Float64[]
    for (i, p) in enumerate(param_vals)
        t0   = time()
        fval = obj_fn(p)
        push!(fvals, fval)
        @printf("    [%d/%d]  %s=%.3f  f=%.6f  (%.1fs)\n",
                i, length(param_vals), param_name, p, fval, time()-t0)
    end
    return fvals
end

# ── Plotting ──────────────────────────────────────────────────────────────────
# p_hist[end] / f_hist[end] = best-so-far (appended by adam_optimize)
# p_hist[1:end-1] / f_hist[1:end-1] = step-by-step trajectory

function make_figure(param_name, scan_vals, scan_f,
                     traj_min, fmin, traj_max, fmax, outfile)

    best_p_min = traj_min[end];  best_f_min = fmin[end]
    best_p_max = traj_max[end];  best_f_max = fmax[end]
    steps_min  = 0:length(fmin)-2
    steps_max  = 0:length(fmax)-2

    # col 1 — loss surface
    p1 = plot(scan_vals, scan_f,
              color=:steelblue, lw=2, label="scan",
              xlabel=param_name, ylabel="f",
              title="Loss surface", grid=true, gridalpha=0.3)
    vline!([best_p_min], color=:red,   lw=1.5, ls=:dash,
           label="best min=$(round(best_p_min,digits=4))")
    vline!([best_p_max], color=:green, lw=1.5, ls=:dash,
           label="best max=$(round(best_p_max,digits=4))")
    scatter!([best_p_min], [best_f_min], color=:red,   ms=7, label="")
    scatter!([best_p_max], [best_f_max], color=:green, ms=7, label="")

    # col 2 — minimization
    p2 = plot(steps_min, fmin[1:end-1],
              color=:red, lw=1.8, label="Adam",
              xlabel="step", ylabel="f",
              title="Minimize  ★ $(param_name)=$(round(best_p_min,digits=4))",
              grid=true, gridalpha=0.3)
    hline!([minimum(scan_f)], color=:grey, lw=1, ls=:dot,  label="scan min")
    hline!([best_f_min],      color=:red,  lw=1, ls=:dash, label="★ best")
    scatter!([argmin(fmin[1:end-1])-1], [best_f_min],
             color=:red, ms=7, label="")

    # col 3 — maximization
    p3 = plot(steps_max, fmax[1:end-1],
              color=:darkgreen, lw=1.8, label="Adam",
              xlabel="step", ylabel="f",
              title="Maximize  ★ $(param_name)=$(round(best_p_max,digits=4))",
              grid=true, gridalpha=0.3)
    hline!([maximum(scan_f)],  color=:grey,      lw=1, ls=:dot,  label="scan max")
    hline!([best_f_max],       color=:darkgreen, lw=1, ls=:dash, label="★ best")
    scatter!([argmax(fmax[1:end-1])-1], [best_f_max],
             color=:darkgreen, ms=7, label="")

    fig = plot(p1, p2, p3, layout=(1,3), size=(1500,480),
               plot_title="f($(param_name))=‖S21·x‖  [$(N_CYL) cylinders, Adam+ForwardDiff]",
               left_margin=8Plots.mm, bottom_margin=8Plots.mm, right_margin=5Plots.mm)
    savefig(fig, outfile)
    println("  Saved → $outfile")
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 62)
    println("  CyScat — n and r Optimization  (Julia + ForwardDiff + Adam)")
    println("=" ^ 62)

    clocs, thickness = make_clocs(N_CYL)
    sp    = make_sp()
    rng   = MersenneTwister(SEED_X)
    x_raw = complex.(randn(rng, 2*N_PROP+1), randn(rng, 2*N_PROP+1))
    x_vec = x_raw / norm(x_raw)

    @printf("\n  Cylinders  : %d  (seed=%d)\n", N_CYL, SEED_CLOCS)
    @printf("  λ=%.2f  period=%.2f  N_prop=%d  N_eva=%d\n",
            WAVELENGTH, PERIOD, N_PROP, N_EVA)
    @printf("  S21_T size : %d×%d\n", 2*N_PROP+1, 2*N_PROP+1)

    # ── PART 1: n ─────────────────────────────────────────────────────────
    println("\n" * "─"^62)
    println("  PART 1: f(n) = ‖S21(n)·x‖,   r = 0.25 fixed")
    println("─"^62)

    R_FIXED = 0.25
    fn_n    = n_val -> objective_n(n_val, clocs, R_FIXED, x_vec, sp, thickness)

    println("\n  Scanning n ∈ [1.0, 3.0] (10 points) ...")
    n_scan_vals = collect(range(1.0, 3.0, length=10))
    t0 = time()
    f_n_scan    = scan(fn_n, n_scan_vals, param_name="n")
    @printf("  Scan done in %.0fs\n", time()-t0)
    scan_min_n  = n_scan_vals[argmin(f_n_scan)]
    scan_max_n  = n_scan_vals[argmax(f_n_scan)]
    @printf("  Scan min: n=%.3f  f=%.6f\n", scan_min_n, minimum(f_n_scan))
    @printf("  Scan max: n=%.3f  f=%.6f\n", scan_max_n, maximum(f_n_scan))

    println("\n  Adam minimize n ...")
    t0 = time()
    traj_n_min, fmin_n = adam_optimize(fn_n, 1.5;
        maximize=false, lr=0.1, n_steps=25,
        p_min=1.0, p_max=3.5, param_name="n")
    @printf("  Done in %.0fs\n", time()-t0)

    println("\n  Adam maximize n ...")
    t0 = time()
    traj_n_max, fmax_n = adam_optimize(fn_n, 1.5;
        maximize=true, lr=0.1, n_steps=25,
        p_min=1.0, p_max=3.5, param_name="n")
    @printf("  Done in %.0fs\n", time()-t0)

    try
        make_figure("n", n_scan_vals, f_n_scan,
                    traj_n_min, fmin_n, traj_n_max, fmax_n,
                    joinpath(OUTDIR, "optimize_n_results_julia.png"))
    catch e
        println("  Plot error: $e")
    end

    # ── PART 2: r ─────────────────────────────────────────────────────────
    println("\n" * "─"^62)
    println("  PART 2: f(r) = ‖S21(r)·x‖,   n = 1.3 fixed")
    println("─"^62)

    N_FIXED = 1.3
    fn_r    = r_val -> objective_r(r_val, clocs, N_FIXED, x_vec, sp, thickness)

    println("\n  Scanning r ∈ [0.15, 0.44] (8 points) ...")
    r_scan_vals = collect(range(0.15, 0.44, length=8))
    t0 = time()
    f_r_scan    = scan(fn_r, r_scan_vals, param_name="r")
    @printf("  Scan done in %.0fs\n", time()-t0)
    scan_min_r  = r_scan_vals[argmin(f_r_scan)]
    scan_max_r  = r_scan_vals[argmax(f_r_scan)]
    @printf("  Scan min: r=%.3f  f=%.6f\n", scan_min_r, minimum(f_r_scan))
    @printf("  Scan max: r=%.3f  f=%.6f\n", scan_max_r, maximum(f_r_scan))

    println("\n  Adam minimize r ...")
    t0 = time()
    traj_r_min, fmin_r = adam_optimize(fn_r, 0.25;
        maximize=false, lr=0.05, n_steps=25,
        p_min=0.15, p_max=0.44, param_name="r")
    @printf("  Done in %.0fs\n", time()-t0)

    println("\n  Adam maximize r ...")
    t0 = time()
    traj_r_max, fmax_r = adam_optimize(fn_r, 0.25;
        maximize=true, lr=0.05, n_steps=25,
        p_min=0.15, p_max=0.44, param_name="r")
    @printf("  Done in %.0fs\n", time()-t0)

    try
        make_figure("r", r_scan_vals, f_r_scan,
                    traj_r_min, fmin_r, traj_r_max, fmax_r,
                    joinpath(OUTDIR, "optimize_r_results_julia.png"))
    catch e
        println("  Plot error: $e")
    end

    # ── Summary ───────────────────────────────────────────────────────────
    println("\n" * "="^62)
    println("  FINAL SUMMARY")
    println("="^62)
    @printf("\n  n  (r=%.2f fixed):\n", R_FIXED)
    @printf("    Scan min : n=%.3f  f=%.6f\n", scan_min_n, minimum(f_n_scan))
    @printf("    Adam min : n=%.4f  f=%.6f  %s\n", traj_n_min[end], fmin_n[end],
            abs(traj_n_min[end]-scan_min_n)<0.25 ? "✓" : "⚠ local min")
    @printf("    Scan max : n=%.3f  f=%.6f\n", scan_max_n, maximum(f_n_scan))
    @printf("    Adam max : n=%.4f  f=%.6f  %s\n", traj_n_max[end], fmax_n[end],
            abs(traj_n_max[end]-scan_max_n)<0.25 ? "✓" : "⚠ local max")
    @printf("\n  r  (n=%.1f fixed):\n", N_FIXED)
    @printf("    Scan min : r=%.3f  f=%.6f\n", scan_min_r, minimum(f_r_scan))
    @printf("    Adam min : r=%.4f  f=%.6f  %s\n", traj_r_min[end], fmin_r[end],
            abs(traj_r_min[end]-scan_min_r)<0.06 ? "✓" : "⚠ local min")
    @printf("    Scan max : r=%.3f  f=%.6f\n", scan_max_r, maximum(f_r_scan))
    @printf("    Adam max : r=%.4f  f=%.6f  %s\n", traj_r_max[end], fmax_r[end],
            abs(traj_r_max[end]-scan_max_r)<0.06 ? "✓" : "⚠ local max")
    println("\n  Plots: optimize_n_results_julia.png  optimize_r_results_julia.png")
    println("="^62)
end

main()