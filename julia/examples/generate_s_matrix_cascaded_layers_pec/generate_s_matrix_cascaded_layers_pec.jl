"""
generate_s_matrix_cascaded_layers_pec.jl
========================================
Demonstrate the Redheffer star-product cascade method with PEC cylinders.

A single slab of 10 perfectly conducting (PEC) cylinders is computed once,
then cascaded with itself to build multi-layer structures of 2, 10, and 20
identical layers.

For each layer count:
  1. Draw the cascaded geometry
  2. Show singular values of S21
  3. Generate wave field movies (normal + optimal wavefront)
  4. Unitarity check (PEC is lossless)

Usage:
    julia generate_s_matrix_cascaded_layers_pec.jl
    julia generate_s_matrix_cascaded_layers_pec.jl --num_cyl 10 --seed 42
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
include(joinpath(_CYSC, "Scattering_Code", "cascadertwo.jl"))
include(joinpath(_CYSC, "get_partition.jl"))

# ── Physical constants ───────────────────────────────────────────────────────
const OUTDIR     = @__DIR__
const WAVELENGTH = 0.93
const PERIOD     = 12.81
const RADIUS     = 0.25
const MU         = 1.0
const CMMAX      = 5
const PHIINC     = π / 2
const GRID_RES   = 200
const PR_Y       = 7

# ── CLI args ─────────────────────────────────────────────────────────────────
function parse_args()
    num_cyl = 10; seed = 42; frames = 100; fps = 20
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

# ── Cascade: repeat S-matrix n_layers times ──────────────────────────────────
function cascade_n_layers(S_single, d_single, n_layers)
    Scas = copy(S_single)
    dcas = d_single
    for _ in 2:n_layers
        Scas, dcas = cascadertwo(Scas, dcas, S_single, d_single)
    end
    return Scas, dcas
end

# ── Plot cascaded geometry (tiled) ───────────────────────────────────────────
function plot_cascaded_geometry(clocs_single, d_single, n_layers; outfile="cascade_geometry")
    p = plot(aspect_ratio=:equal, size=(900, 350),
             xlabel="x", ylabel="y",
             title="$n_layers cascaded layers ($(size(clocs_single,1)) PEC cyl each)")

    total_d = d_single * n_layers
    θ = range(0, 2π, length=40)
    for layer in 1:n_layers
        y_off = (layer - 1) * d_single
        for i in 1:size(clocs_single, 1)
            plot!(p, clocs_single[i,1] .+ RADIUS .* cos.(θ),
                     clocs_single[i,2] .+ y_off .+ RADIUS .* sin.(θ),
                  seriestype=:shape, fillcolor=:gray, fillalpha=0.6,
                  linecolor=:black, lw=0.8, label="")
        end
        hline!(p, [y_off], color=:magenta, lw=1, ls=:dash, label="")
    end
    hline!(p, [total_d], color=:magenta, lw=2, label="")
    hline!(p, [0.0], color=:magenta, lw=2, label="")
    xlims!(p, -0.5, PERIOD + 0.5)
    ylims!(p, -1.0, total_d + 1.0)
    savefig(p, outfile * ".png")
    println("  Saved → $(outfile).png")
end

# ── Plot singular values ─────────────────────────────────────────────────────
function plot_singular_values(S, nmax, n_layers; outfile="sv")
    nm  = 2*nmax + 1
    S21 = S[nm+1:end, 1:nm]
    tau = svdvals(S21)

    p = bar(1:length(tau), tau .^2,
            xlabel="Channel index", ylabel="τ²",
            title="Transmission eigenvalues — $n_layers layers (PEC)",
            legend=false, color=:navy, ylims=(0, 1.05), size=(700, 400))
    hline!(p, [1.0], color=:red, lw=1, ls=:dash, label="")
    savefig(p, outfile * ".png")
    println("  Saved → $(outfile).png")
    @printf("  max(τ²) = %.6f   sum(τ²) = %.4f\n", maximum(tau.^2), sum(tau.^2))
end

# ── Wave field ───────────────────────────────────────────────────────────────
"""
Build the complex field for a cascaded PEC structure.

For each layer we re-solve the multiple scattering problem using the
single-layer T-matrix and the field incident on that layer (obtained by
cascading S-matrices up to that layer). This gives the correct Hankel-
expansion field that scatters off the PEC cylinders.
"""
function build_fields_cascade(S_cas, S_single, nmax, total_thickness, d_single,
                              n_layers, mode, clocs, crads, cmmaxs, cepmus, sp)
    nm  = 2*nmax + 1
    S11 = S_cas[1:nm, 1:nm]
    S21 = S_cas[nm+1:end, 1:nm]

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

    k     = 2π / WAVELENGTH
    m_flo = collect(-nmax:nmax)
    kxs   = 2π / PERIOD .* m_flo
    kys_v = ky(k, complex.(kxs))

    P1    = Diagonal(1.0 ./ sqrt.(kys_v ./ k))
    Inc_c = P1 * Input
    Ref_c = P1 * (S11 * Input)
    Tra_c = P1 * (S21 * Input)

    # ── Pre-compute single-layer scattering data for interior field ──────
    t_mat = transall(clocs, cmmaxs, PERIOD, WAVELENGTH, sp["phiinc"], sp, 1)
    s_vec = sall(cmmaxs, cepmus, crads, WAVELENGTH)
    z     = I - Diagonal(s_vec) * t_mat
    lu_f  = lu(z)
    num_cyl = size(clocs, 1)
    m_max   = Int((size(z, 1) / num_cyl - 1) / 2)

    # Pre-compute v_s for each input mode
    v_s_modes = Vector{Vector{ComplexF64}}(undef, nm)
    for (idx, nin) in enumerate(-nmax:nmax)
        kxex = sp["kxs"][sp["MiddleIndex"] + nin]
        kxex = isa(kxex, AbstractArray) ? kxex[1] : kxex
        v_s_modes[idx] = Diagonal(s_vec) * vall(clocs, cmmaxs, WAVELENGTH, kxex, 1)
    end

    # Compute the field incident on each layer
    layer_inputs = Vector{Vector{ComplexF64}}(undef, n_layers)
    layer_inputs[1] = Input
    if n_layers > 1
        S_sub = copy(S_single)
        d_sub = d_single
        for L in 2:n_layers
            nm_s = size(S_sub, 1) ÷ 2
            S21_sub = S_sub[nm_s+1:end, 1:nm_s]
            layer_inputs[L] = S21_sub * Input
            if L < n_layers
                S_sub, d_sub = cascadertwo(S_sub, d_sub, S_single, d_single)
            end
        end
    end

    # ── Compute field on grid ────────────────────────────────────────────
    Nx     = GRID_RES
    Ly     = PR_Y * WAVELENGTH
    d      = total_thickness
    y_full = collect(range(-Ly, d + Ly, length=GRID_RES * 2))
    x_phys = collect(range(0.0, PERIOD, length=Nx))

    FullField = zeros(ComplexF64, Nx, length(y_full))
    for (jy, y) in enumerate(y_full), ix in 1:Nx
        x = x_phys[ix]
        if y < 0.0
            FullField[ix, jy] =
                sum(Inc_c .* exp.(1im .* (-kxs .* x .- kys_v .* y))) +
                sum(Ref_c .* exp.(1im .* (-kxs .* x .+ kys_v .* y)))
        elseif y > d
            FullField[ix, jy] =
                sum(Tra_c .* exp.(1im .* (-kxs .* x .- kys_v .* (y - d))))
        else
            # Interior: find which layer this point belongs to
            layer = min(Int(floor(y / d_single)) + 1, n_layers)
            y_off = (layer - 1) * d_single
            y_local = y - y_off

            # Solve for scattering coefficients with this layer's input
            v_rhs = zeros(ComplexF64, size(z, 1))
            L_input = layer_inputs[layer]
            L_inc_c = P1 * L_input
            for (idx, _) in enumerate(-nmax:nmax)
                v_rhs += v_s_modes[idx] * L_input[idx]
            end
            c_vector = lu_f \ v_rhs

            # Incident Floquet field at this y_local
            total_f = sum(L_inc_c .* exp.(1im .* (-kxs .* x .- kys_v .* y_local)))
            # Add scattered field from each cylinder
            for i in 1:num_cyl
                dx = x - clocs[i,1]
                dy = y_local - clocs[i,2]
                r = sqrt(dx^2 + dy^2)
                if r <= crads[i]; total_f = 0.0im; break; end
                θ_cyl = atan(dy, dx)
                idx0 = (i-1)*(2*m_max+1) + 1
                for m_idx in -m_max:m_max
                    total_f += c_vector[idx0 + m_idx + m_max] * hankelh2(m_idx, k*r) * exp(1im * m_idx * θ_cyl)
                end
            end
            FullField[ix, jy] = total_f
        end
    end
    return FullField, x_phys ./ WAVELENGTH, y_full ./ WAVELENGTH, d, label
end

function make_movie(fields_list, x_grid, y_full, d, clocs_single, d_single, n_layers,
                    outfile; n_frames=50, fps=20)
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
        vline!(p, [0.0, d / WAVELENGTH], color=:magenta, lw=2, label="")
        # Draw cylinder outlines for each cascaded layer
        θ = range(0, 2π, length=40)
        for layer in 1:n_layers
            y_off = (layer - 1) * d_single
            for i in 1:size(clocs_single, 1)
                cx = clocs_single[i,1] / WAVELENGTH
                cy = (clocs_single[i,2] + y_off) / WAVELENGTH
                r_lam = RADIUS / WAVELENGTH
                plot!(p, cy .+ r_lam .* cos.(θ), cx .+ r_lam .* sin.(θ),
                      color=:black, lw=1, label="")
            end
        end
    end

    try
        mp4(anim, outfile * ".mp4", fps=fps)
        println("  Saved → $(outfile).mp4")
    catch e
        gif(anim, outfile * ".gif", fps=fps)
        println("  Saved → $(outfile).gif")
    end
end

# ── Main ─────────────────────────────────────────────────────────────────────
function main()
    num_cyl, seed, frames, fps = parse_args()
    n_prop, n_eva, nmax = compute_nmax()
    thickness = auto_thickness(num_cyl)

    println("=" ^ 65)
    println("  CyScat — Cascaded Layers (PEC) Example")
    println("=" ^ 65)
    @printf("  λ=%.2f  Λ=%.2f  r=%.2f  PEC (ε=-1)\n", WAVELENGTH, PERIOD, RADIUS)
    @printf("  Single layer: %d cylinders, thickness=%.1f, nmax=%d (propagating only)\n",
            num_cyl, thickness, nmax)

    # Compute single-layer S-matrix
    clocs  = place_cylinders(num_cyl, PERIOD, thickness, RADIUS; seed=seed)
    sp     = smatrix_parameters(WAVELENGTH, PERIOD, PHIINC,
                                1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, PERIOD/120)
    cmmaxs = fill(CMMAX, num_cyl)
    cepmus = repeat([-1.0 MU], num_cyl, 1)      # ε = -1 → PEC
    crads  = fill(RADIUS, num_cyl)

    println("\n  Computing single-layer S-matrix...")
    t0 = time()
    S_single, _ = smatrix(clocs, cmmaxs, cepmus, crads, PERIOD, WAVELENGTH,
                          nmax, thickness, sp, "On")
    @printf("  Done in %.1fs\n", time() - t0)

    # Unitarity check for single layer (PEC is lossless)
    nm = 2*nmax + 1
    S11_single = S_single[1:nm, 1:nm]; S21_single = S_single[nm+1:end, 1:nm]
    unit_err = norm(S11_single'*S11_single + S21_single'*S21_single - I)
    @printf("  Single-layer unitarity: ‖S11'S11 + S21'S21 − I‖ = %.2e  %s\n",
            unit_err, unit_err < 1e-3 ? "✓" : "⚠")

    # Cascade for 2, 10, 20 layers
    for n_layers in [2, 10, 20]
        println("\n" * "─" ^ 65)
        @printf("  Cascading %d identical layers\n", n_layers)
        println("─" ^ 65)

        t0 = time()
        S_cas, d_cas = cascade_n_layers(S_single, thickness, n_layers)
        @printf("  Cascade done in %.3fs  (total thickness=%.1f)\n", time() - t0, d_cas)

        # Unitarity check for cascaded structure
        S11_c = S_cas[1:nm, 1:nm]; S21_c = S_cas[nm+1:end, 1:nm]
        unit_err_c = norm(S11_c'*S11_c + S21_c'*S21_c - I)
        @printf("  Cascaded unitarity: ‖S11'S11 + S21'S21 − I‖ = %.2e  %s\n",
                unit_err_c, unit_err_c < 1e-3 ? "✓" : "⚠")

        tag = "pec_$(n_layers)layers"

        # 1. Geometry
        plot_cascaded_geometry(clocs, thickness, n_layers;
                               outfile=joinpath(OUTDIR, "$(tag)_geometry"))

        # 2. Singular values
        plot_singular_values(S_cas, nmax, n_layers;
                             outfile=joinpath(OUTDIR, "$(tag)_singular_values"))

        # 3 & 4. Wave field movies
        fields_list = Tuple{Matrix{ComplexF64}, String}[]
        local x_grid, y_full
        for mode in ["normal", "opt_trans"]
            @printf("  Building field: %s...\n", mode)
            F, xg, yf, _, lbl = build_fields_cascade(
                S_cas, S_single, nmax, d_cas, thickness,
                n_layers, mode, clocs, crads, cmmaxs, cepmus, sp)
            push!(fields_list, (F, "$(n_layers) layers — $lbl"))
            x_grid = xg; y_full = yf
        end

        make_movie(fields_list, x_grid, y_full, d_cas,
                   clocs, thickness, n_layers,
                   joinpath(OUTDIR, "$(tag)_wavefield");
                   n_frames=frames, fps=fps)
    end

    println("\n" * "=" ^ 65)
    println("  All outputs saved in $(OUTDIR)")
    println("=" ^ 65)
end

main()
