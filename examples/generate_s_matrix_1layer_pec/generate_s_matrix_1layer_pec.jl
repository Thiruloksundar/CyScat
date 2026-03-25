"""
generate_s_matrix_1layer_pec.jl
================================
Compute the S-matrix for a single layer of 10 perfectly conducting (PEC) cylinders.

  1. Draw the slab geometry (cylinder positions)
  2. Compute the S-matrix and show singular values of S21
  3. Generate a wave field movie with normal incidence
  4. Generate a wave field movie with the optimal (maximum-transmission) wavefront

Usage:
    julia generate_s_matrix_1layer_pec.jl
    julia generate_s_matrix_1layer_pec.jl --num_cyl 20
"""

using LinearAlgebra
using SpecialFunctions
using Printf
using Random
using Plots
using Statistics

# ── Load CyScat ──────────────────────────────────────────────────────────────
const _CYSC = joinpath(@__DIR__, "..", "..", "CyScat")
push!(LOAD_PATH, _CYSC)
push!(LOAD_PATH, joinpath(_CYSC, "Scattering_Code"))

include(joinpath(_CYSC, "Scattering_Code", "ky.jl"))
include(joinpath(_CYSC, "Scattering_Code", "smatrix_parameters.jl"))
include(joinpath(_CYSC, "Scattering_Code", "sall.jl"))
include(joinpath(_CYSC, "Scattering_Code", "vall.jl"))
include(joinpath(_CYSC, "Scattering_Code", "modified_epsilon_shanks.jl"))
include(joinpath(_CYSC, "Scattering_Code", "simulation_time_profile.jl"))
include(joinpath(_CYSC, "Scattering_Code", "transall.jl"))
include(joinpath(_CYSC, "Scattering_Code", "scattering_coefficients_all.jl"))
include(joinpath(_CYSC, "Scattering_Code", "smatrix.jl"))
include(joinpath(_CYSC, "get_partition.jl"))

# ── Physical constants ───────────────────────────────────────────────────────
const OUTDIR     = @__DIR__
const WAVELENGTH = 0.93
const PERIOD     = 12.81
const RADIUS     = 0.25
const MU         = 1.0
const CMMAX      = 5
const PHIINC     = π / 2        # normal incidence
const GRID_RES   = 200
const PR_Y       = 7

# ── CLI args ─────────────────────────────────────────────────────────────────
function parse_args()
    num_cyl = 10; seed = 42; frames = 120; fps = 20
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--num_cyl" && i < length(ARGS); num_cyl = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--seed" && i < length(ARGS); seed = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--frames" && i < length(ARGS); frames = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--fps" && i < length(ARGS); fps = parse(Int, ARGS[i+1]); i += 2
        else i += 1 end
    end
    return num_cyl, seed, frames, fps
end

function compute_nmax()
    n_prop = floor(Int, PERIOD / WAVELENGTH)
    # PEC: no evanescent modes needed
    return n_prop, 0, n_prop
end

function place_cylinders(num_cyl, period, thickness, radius; seed=42)
    rng = MersenneTwister(seed); margin = radius * 1.5; min_sep = 2.5 * radius
    clocs = zeros(num_cyl, 2)
    for i in 1:num_cyl
        for _ in 1:10000
            x = margin + rand(rng) * (period - 2*margin)
            y = margin + rand(rng) * (thickness - 2*margin)
            if i == 1 || all(sqrt.((x .- clocs[1:i-1,1]).^2 .+
                                   (y .- clocs[1:i-1,2]).^2) .> min_sep)
                clocs[i,:] = [x, y]; break
            end
        end
    end
    return clocs
end

function auto_thickness(num_cyl)
    spacing = 2.5 * RADIUS
    cyls_per_row = floor(Int, PERIOD / spacing)
    rows_needed  = num_cyl / cyls_per_row + 2
    return round(max(0.5, rows_needed * spacing * 1.5), digits=1)
end

# ── Plot geometry ────────────────────────────────────────────────────────────
function plot_geometry(clocs, thickness; outfile="pec_geometry")
    num_cyl = size(clocs, 1)
    p = plot(aspect_ratio=:equal, size=(700, 400),
             xlabel="x", ylabel="y",
             title="$num_cyl PEC cylinders  (λ=$(WAVELENGTH), Λ=$(PERIOD))")
    hline!(p, [0.0, thickness], color=:magenta, lw=2, label="")
    vline!(p, [0.0, PERIOD], color=:gray, lw=1, ls=:dash, label="")
    θ = range(0, 2π, length=60)
    for i in 1:num_cyl
        plot!(p, clocs[i,1] .+ RADIUS .* cos.(θ),
                 clocs[i,2] .+ RADIUS .* sin.(θ),
              seriestype=:shape, fillcolor=:gray, fillalpha=0.6,
              linecolor=:black, lw=1, label="")
    end
    xlims!(p, -0.5, PERIOD + 0.5); ylims!(p, -0.5, thickness + 0.5)
    savefig(p, outfile * ".png")
    println("  Saved → $(outfile).png")
end

# ── Plot singular values ─────────────────────────────────────────────────────
function plot_singular_values(S, nmax, num_cyl; outfile="pec_singular_values")
    nm = 2*nmax + 1
    S21 = S[nm+1:end, 1:nm]
    tau = svdvals(S21)
    p = bar(1:length(tau), tau .^2,
            xlabel="Channel index", ylabel="τ²",
            title="Transmission eigenvalues — $num_cyl PEC cylinders",
            legend=false, color=:navy, ylims=(0, 1.05), size=(700, 400))
    hline!(p, [1.0], color=:red, lw=1, ls=:dash, label="")
    savefig(p, outfile * ".png")
    println("  Saved → $(outfile).png")
    @printf("  max(τ²) = %.6f   sum(τ²) = %.4f\n", maximum(tau.^2), sum(tau.^2))
end

# ── Wave field computation ───────────────────────────────────────────────────
function build_fields(S, nmax, thickness, mode,
                      clocs, crads, cmmaxs, cepmus, period, lambda_wave, sp)
    nm    = 2*nmax + 1
    n_eva = 0
    S11 = S[1:nm, 1:nm]; S21 = S[nm+1:end, 1:nm]

    Fsvd = svd(S11)
    if mode == "opt_trans"
        v_in = Fsvd.V[:, end]
    else
        v_in = zeros(ComplexF64, nm)
        v_in[nmax + 1] = 1.0
    end
    Input = copy(v_in)

    percent_trans = sum(abs2.(S21 * Input)) / sum(abs2.(Input)) * 100
    mode_label = mode == "opt_trans" ? "Optimal Wavefront" : "Normal Incidence"
    label = @sprintf("%s — %.1f%% transmitted", mode_label, percent_trans)

    k     = 2π / lambda_wave
    t_mat = transall(clocs, cmmaxs, period, lambda_wave, sp["phiinc"], sp, 1)
    s_vec = sall(cmmaxs, cepmus, crads, lambda_wave)
    z     = I - Diagonal(s_vec) * t_mat
    lu_f  = lu(z)

    v_rhs = zeros(ComplexF64, size(z, 1))
    for nin in -nmax:nmax
        kxex = sp["kxs"][sp["MiddleIndex"] + nin]
        kxex = isa(kxex, AbstractArray) ? kxex[1] : kxex
        v_s  = Diagonal(s_vec) * vall(clocs, cmmaxs, lambda_wave, kxex, 1)
        v_rhs += v_s * Input[nin + nmax + 1]
    end
    c_vector = lu_f \ v_rhs

    Nx     = GRID_RES; Ly = PR_Y * lambda_wave
    y_full = collect(range(-Ly, thickness + Ly, length=GRID_RES * 2))
    x_phys = collect(range(0.0, period, length=Nx))
    m_flo  = collect(-nmax:nmax)
    kxs    = 2π / period .* m_flo
    kys_v  = ky(k, complex.(kxs))
    prop   = 1:nm       # all modes are propagating for PEC

    P1    = Diagonal(1.0 ./ sqrt.(kys_v ./ k))
    Inc_c = P1 * Input; Ref_c = P1 * (S11 * Input); Tra_c = P1 * (S21 * Input)
    num_cyl = size(clocs, 1)
    m_max   = Int((size(c_vector, 1) / num_cyl - 1) / 2)

    FullField = zeros(ComplexF64, Nx, length(y_full))
    for (jy, y) in enumerate(y_full), ix in 1:Nx
        x = x_phys[ix]
        if y < 0.0
            FullField[ix, jy] =
                sum(Inc_c[prop] .* exp.(1im .* (-kxs[prop] .* x .- kys_v[prop] .* y))) +
                sum(Ref_c[prop] .* exp.(1im .* (-kxs[prop] .* x .+ kys_v[prop] .* y)))
        elseif y > thickness
            FullField[ix, jy] =
                sum(Tra_c[prop] .* exp.(1im .* (-kxs[prop] .* x .- kys_v[prop] .* (y - thickness))))
        else
            total_f = sum(Inc_c[prop] .* exp.(1im .* (-kxs[prop] .* x .- kys_v[prop] .* y)))
            for i in 1:num_cyl
                dx, dy = x - clocs[i,1], y - clocs[i,2]
                r = sqrt(dx^2 + dy^2)
                if r <= crads[i]; total_f = 0.0im; break; end
                θ = atan(dy, dx)
                idx0 = (i-1)*(2*m_max+1) + 1
                for m_idx in -m_max:m_max
                    total_f += c_vector[idx0 + m_idx + m_max] * hankelh2(m_idx, k*r) * exp(1im * m_idx * θ)
                end
            end
            FullField[ix, jy] = total_f
        end
    end
    return FullField, x_phys ./ lambda_wave, y_full ./ lambda_wave, thickness, label
end

function make_movie(fields_list, x_grid, y_full, d, clocs, crads, lambda_wave, outfile;
                    n_frames=60, fps=20)
    T_period = 4.0; omega = 2π / T_period; dt = T_period / n_frames
    vmax = maximum([maximum(abs.(real.(f[1]))) for f in fields_list]) * 0.7
    vmax = max(vmax, 1e-10)

    anim = @animate for global_frame in 0:(length(fields_list) * n_frames - 1)
        seg_idx = (global_frame ÷ n_frames) + 1
        t = (global_frame % n_frames) * dt
        Field, label = fields_list[seg_idx]
        F_plot = real.(Field .* exp(1im * omega * t))
        p = heatmap(y_full, x_grid, F_plot, color=:jet, clims=(-vmax, vmax),
                    xlabel="y/λ", ylabel="x/λ", title=label,
                    aspect_ratio=:equal, size=(900, 500))
        vline!(p, [0.0, d/lambda_wave], color=:magenta, lw=2, label="")
        for i in 1:size(clocs, 1)
            r_lam = crads[i] / lambda_wave
            cx, cy = clocs[i,1] / lambda_wave, clocs[i,2] / lambda_wave
            θ = range(0, 2π, length=40)
            plot!(p, cy .+ r_lam .* cos.(θ), cx .+ r_lam .* sin.(θ),
                  color=:black, lw=1, label="")
        end
    end
    try mp4(anim, outfile * ".mp4", fps=fps); println("  Saved → $(outfile).mp4")
    catch; gif(anim, outfile * ".gif", fps=fps); println("  Saved → $(outfile).gif") end
end

# ── Main ─────────────────────────────────────────────────────────────────────
function main()
    num_cyl, seed, frames, fps = parse_args()
    thickness = auto_thickness(num_cyl)
    n_prop, n_eva, nmax = compute_nmax()

    println("=" ^ 65)
    println("  CyScat — Single-Layer PEC Example")
    println("=" ^ 65)
    @printf("  λ=%.2f  Λ=%.2f  r=%.2f  N_cyl=%d  seed=%d\n",
            WAVELENGTH, PERIOD, RADIUS, num_cyl, seed)
    @printf("  nmax=%d  (propagating only, no evanescent)  thickness=%.1f\n",
            nmax, thickness)

    clocs  = place_cylinders(num_cyl, PERIOD, thickness, RADIUS; seed=seed)
    sp     = smatrix_parameters(WAVELENGTH, PERIOD, PHIINC,
                                1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, PERIOD/120)
    cmmaxs = fill(CMMAX, num_cyl)
    cepmus = repeat([-1.0 MU], num_cyl, 1)      # ε = -1 → PEC
    crads  = fill(RADIUS, num_cyl)

    # 1. Geometry
    println("\n  [1] Drawing geometry...")
    plot_geometry(clocs, thickness; outfile=joinpath(OUTDIR, "pec_geometry"))

    # 2. S-matrix + singular values
    println("\n  [2] Computing S-matrix...")
    t0 = time()
    S, _ = smatrix(clocs, cmmaxs, cepmus, crads, PERIOD, WAVELENGTH,
                   nmax, thickness, sp, "On")
    @printf("  S-matrix done in %.1fs\n", time() - t0)

    # Unitarity check (PEC is lossless)
    nm = 2*nmax + 1
    S11 = S[1:nm, 1:nm]; S21 = S[nm+1:end, 1:nm]
    unit_err = norm(S11'*S11 + S21'*S21 - I)
    @printf("  ‖S11'S11 + S21'S21 − I‖ = %.2e  %s\n",
            unit_err, unit_err < 1e-3 ? "✓" : "⚠")

    plot_singular_values(S, nmax, num_cyl; outfile=joinpath(OUTDIR, "pec_singular_values"))

    # 3 & 4. Wave field movies
    fields_list = Tuple{Matrix{ComplexF64}, String}[]
    local x_grid, y_full
    for mode in ["normal", "opt_trans"]
        println("\n  [$(mode == "normal" ? 3 : 4)] Building field: $mode...")
        t0 = time()
        F, xg, yf, _, lbl = build_fields(S, nmax, thickness, mode,
                                          clocs, crads, cmmaxs, cepmus,
                                          PERIOD, WAVELENGTH, sp)
        @printf("  Done in %.1fs  %s\n", time() - t0, lbl)
        push!(fields_list, (F, lbl)); x_grid = xg; y_full = yf
    end

    println("\n  Generating movie...")
    make_movie(fields_list, x_grid, y_full, thickness, clocs, crads,
               WAVELENGTH, joinpath(OUTDIR, "pec_wavefield"); n_frames=frames, fps=fps)

    println("\n" * "=" ^ 65)
    println("  All outputs saved in current directory.")
    println("=" ^ 65)
end

main()
