"""
s_channel_maze.jl
==================
S-shaped PEC cylinder maze — wave field visualization.

Geometry (one unit cell, periodic in x):

  y=L  oooooooooo [EXIT] oooo      <- top face: opening at x ≈ 3D/4
       o         o      o   o
       o  upper  o      o   o
       o  channel ooooooo   o      <- horizontal wall
       o                    o
       ooooooo  lower  o    o      <- horizontal wall
               o channel    o
  y=0  oooooooo [ENTRY] ooooo      <- bottom face: opening at x ≈ D/4

The walls are dense PEC cylinders on a grid; the S-path is left empty.

Normal incidence  → spreads across both walls, low transmission.
Optimal wavefront → SVD of S11 finds the open eigenchannel, tunnels through.

Usage:
    julia --project=. s_channel_maze.jl
    julia --project=. s_channel_maze.jl --grid_res 60 --frames 60
"""

using LinearAlgebra
using SpecialFunctions
using Printf
using Plots

# ── Parse CLI arguments ────────────────────────────────────────────────────────
function parse_args()
    d = Dict{String,Any}(
        "grid_res" => 200,
        "frames"   => 160,
        "fps"      => 20,
        "output"   => "s_channel_maze",
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--grid_res" && i < length(ARGS); d["grid_res"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--frames"   && i < length(ARGS); d["frames"]   = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--fps"      && i < length(ARGS); d["fps"]      = parse(Int, ARGS[i+1]); i += 2
        elseif a in ("-o","--output") && i < length(ARGS); d["output"] = ARGS[i+1]; i += 2
        else i += 1
        end
    end
    return d
end

# ── Physical constants ─────────────────────────────────────────────────────────
const OUTDIR      = @__DIR__
const WAVELENGTH  = 0.93          # λ (arbitrary units)
const PERIOD      = 12.81         # D — periodic cell width
const RADIUS      = 0.25          # cylinder radius  (r ≈ 0.27λ)
const PHIINC      = π / 2         # normal incidence
const PR_Y        = 5             # plot region: extend 5λ outside slab

# ── Maze parameters ────────────────────────────────────────────────────────────
const CMMAX_MAZE  = 3             # azimuthal modes per cylinder (3 sufficient for PEC)
const CYL_SPACING = 0.68          # center-to-center grid spacing (≈ 2.7 r)
const CHANNEL_W   = 3.0           # channel width  (≈ 3.8 λ)
const SLAB_THICK  = 14.0          # slab thickness L

# ── Load scattering-matrix engine (same files as wave_field_movie.jl) ──────────
const SCRIPT_DIR = joinpath(@__DIR__, "..", "..", "CyScat")
push!(LOAD_PATH, SCRIPT_DIR)
push!(LOAD_PATH, joinpath(SCRIPT_DIR, "Scattering_Code"))

include(joinpath(SCRIPT_DIR, "Scattering_Code", "ky.jl"))
include(joinpath(SCRIPT_DIR, "Scattering_Code", "smatrix_parameters.jl"))
include(joinpath(SCRIPT_DIR, "Scattering_Code", "smatrix.jl"))
include(joinpath(SCRIPT_DIR, "get_partition.jl"))

# ── S-channel cylinder placement ──────────────────────────────────────────────
"""
    place_s_channel(period, thickness, channel_w, cyl_spacing, radius)

Fill [0,D]×[0,L] with a grid of PEC cylinders EXCEPT inside the S-shaped path.

S-path (union of three rectangular strips):
  1. Lower vertical  : |x − D/4|   < W/2   AND   y < L/2 + W/2
  2. Upper vertical  : |x − 3D/4|  < W/2   AND   y > L/2 − W/2
  3. Horizontal conn.: |y − L/2|   < W/2   AND   D/4−W/2 < x < 3D/4+W/2

The overlap of 1+3 and 2+3 ensures smooth junctions at the bends.
"""
function place_s_channel(period, thickness, channel_w, cyl_spacing, radius)
    margin  = radius * 1.6
    xs = collect(margin : cyl_spacing : period    - margin)
    ys = collect(margin : cyl_spacing : thickness - margin)

    x_lower = period    / 4        # x-centre of lower vertical channel
    x_upper = 3*period  / 4        # x-centre of upper vertical channel
    y_mid   = thickness / 2        # y-level of horizontal connector
    W       = channel_w

    clocs_list = NTuple{2,Float64}[]
    for x in xs, y in ys
        in_lower = abs(x - x_lower) < W/2  &&  y  < y_mid + W/2
        in_upper = abs(x - x_upper) < W/2  &&  y  > y_mid - W/2
        in_conn  = abs(y - y_mid)   < W/2  &&
                   x > x_lower - W/2       &&  x < x_upper + W/2
        if !(in_lower || in_upper || in_conn)
            push!(clocs_list, (x, y))
        end
    end

    n     = length(clocs_list)
    clocs = zeros(n, 2)
    for (i, (xi, yi)) in enumerate(clocs_list)
        clocs[i, 1] = xi
        clocs[i, 2] = yi
    end
    return clocs
end

# ── Setup: place cylinders + compute S-matrix ─────────────────────────────────
function setup_maze()
    clocs   = place_s_channel(PERIOD, SLAB_THICK, CHANNEL_W, CYL_SPACING, RADIUS)
    num_cyl = size(clocs, 1)
    println("  Placed $num_cyl PEC cylinders in S-channel maze")

    # Propagating modes only (no evanescent needed for PEC walls)
    n_prop = floor(Int, PERIOD / WAVELENGTH)
    nmax   = n_prop
    n_eva  = 0

    # smatrix_parameters: same signature as wave_field_movie.jl, adjust cmmax args
    sp = smatrix_parameters(WAVELENGTH, PERIOD, PHIINC,
                            1e-10, 1e-4, CMMAX_MAZE, 3, 1000, 3, CMMAX_MAZE, 1, PERIOD/100)

    cmmaxs = fill(CMMAX_MAZE, num_cyl)
    cepmus = repeat([-1.0 1.0], num_cyl, 1)   # ε < 0  →  PEC
    crads  = fill(RADIUS, num_cyl)

    @printf("  Computing S-matrix  (nmax=%d, %d cylinders) …\n", nmax, num_cyl)
    t0 = time()
    S, _ = smatrix(clocs, cmmaxs, cepmus, crads, PERIOD, WAVELENGTH,
                   nmax, SLAB_THICK, sp, "On")
    @printf("  S-matrix done in %.1f s\n", time() - t0)

    return clocs, SLAB_THICK, S, n_prop, n_eva, nmax, crads, cmmaxs, cepmus, sp
end

# ── Field computation ──────────────────────────────────────────────────────────
"""
    build_fields(S, nmax, n_eva, thickness, mode,
                 clocs, crads, cmmaxs, cepmus, period, lambda_wave, sp;
                 grid_res=80)

Compute the complex field over the full domain for the given mode:
  "normal"      → normally-incident plane wave  (n=0 Floquet mode)
  "opt_trans"   → minimum-reflection eigenvector of S11 (open eigenchannel)

Regions:
  y < 0          : incident + reflected Floquet waves  (propagating only)
  0 ≤ y ≤ thick  : incident Floquet + sum of H₂ⁿ cylinder expansions
  y > thick      : transmitted Floquet waves            (propagating only)

Bugs fixed vs. original wave_field_movie.jl:
  1. hankelh2 (not hankelh1) — matches the exp(+iωt) time convention
  2. Evanescent modes excluded from reflected AND transmitted sums
"""
function build_fields(S, nmax, n_eva, thickness, mode,
                      clocs, crads, cmmaxs, cepmus, period, lambda_wave, sp;
                      grid_res=80)
    nm  = 2*nmax + 1
    S11 = S[1:nm,    1:nm]
    S21 = S[nm+1:end, 1:nm]

    # 1.  Choose input wavefront
    R_trunc = n_eva > 0 ? S11[n_eva+1:end-n_eva, n_eva+1:end-n_eva] : S11
    Fsvd    = svd(R_trunc)

    if mode == "opt_trans"
        v_in = Fsvd.V[:, end]          # smallest singular value of S11
        mode_label = "Open Eigenchannel"
    elseif mode == "opt_reflect"
        v_in = Fsvd.V[:, 1]
        mode_label = "Closed Eigenchannel"
    else                                # "normal"
        v_in = zeros(ComplexF64, size(R_trunc, 1))
        v_in[Int(floor(size(R_trunc,1)/2) + 1)] = 1.0
        mode_label = "Normal Incidence"
    end

    Input = zeros(ComplexF64, nm)
    if n_eva > 0
        Input[n_eva+1:end-n_eva] = v_in
    else
        Input = copy(v_in)
    end

    # 2.  Transmission percentage
    percent_trans = sum(abs2.(S21 * Input)) / sum(abs2.(Input)) * 100
    label = @sprintf("%s — %.1f%% transmitted", mode_label, percent_trans)

    # 3.  Solve for cylinder scattering coefficients under this input
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

    # 4.  Build spatial grid
    Nx     = grid_res
    Ly     = PR_Y * lambda_wave
    y_full = collect(range(-Ly, thickness + Ly, length = grid_res * 2))
    x_phys = collect(range(0.0, period, length = Nx))

    m_flo = collect(-nmax:nmax)
    kxs   = 2π / period .* m_flo
    kys   = ky(k, complex.(kxs))
    prop  = n_eva+1 : nm-n_eva          # propagating-mode indices

    P1        = Diagonal(1.0 ./ sqrt.(kys ./ k))
    Inc_c     = P1 * Input
    Ref_c     = P1 * (S11 * Input)
    Tra_c     = P1 * (S21 * Input)

    num_cyl  = size(clocs, 1)
    m_max    = Int((size(c_vector, 1) / num_cyl - 1) / 2)

    FullField = zeros(ComplexF64, Nx, length(y_full))

    for (jy, y) in enumerate(y_full), ix in 1:Nx
        x = x_phys[ix]

        if y < 0.0
            # ── Reflection region: propagating modes only ──────────────
            FullField[ix, jy] =
                sum(Inc_c[prop] .* exp.(1im .* (-kxs[prop] .* x .- kys[prop] .* y))) +
                sum(Ref_c[prop] .* exp.(1im .* (-kxs[prop] .* x .+ kys[prop] .* y)))

        elseif y > thickness
            # ── Transmission region: propagating modes only ───────────
            FullField[ix, jy] =
                sum(Tra_c[prop] .* exp.(1im .* (-kxs[prop] .* x .- kys[prop] .* (y - thickness))))

        else
            # ── Interior: incident Floquet + H₂ⁿ cylinder expansions ──
            total_f = sum(Inc_c[prop] .*
                            exp.(1im .* (-kxs[prop] .* x .- kys[prop] .* y)))

            for i in 1:num_cyl
                dx = x - clocs[i,1]
                dy = y - clocs[i,2]
                r  = sqrt(dx^2 + dy^2)
                if r <= crads[i]
                    total_f = 0.0im   # inside cylinder — field is zero
                    break
                end
                θ = atan(dy, dx)
                idx_start = (i-1)*(2*m_max + 1) + 1
                for m_idx in -m_max:m_max
                    # *** hankelh2 matches the exp(+iωt) convention ***
                    total_f += c_vector[idx_start + m_idx + m_max] *
                                hankelh2(m_idx, k*r) * exp(1im * m_idx * θ)
                end
            end
            FullField[ix, jy] = total_f
        end
    end

    return FullField, x_phys ./ lambda_wave, y_full ./ lambda_wave, thickness, label
end

# ── Animation ─────────────────────────────────────────────────────────────────
function make_movie(fields_list, x_grid, y_full, d, clocs, crads, lambda_wave, outfile;
                    n_frames=80, fps=20)
    T_period = 4.0
    omega    = 2π / T_period
    dt       = T_period / n_frames
    n_seg    = length(fields_list)

    vmax = maximum([maximum(abs.(real.(f[1]))) for f in fields_list]) * 0.75
    vmax = max(vmax, 1e-10)

    d_lam  = d / lambda_wave
    W_lam  = CHANNEL_W / lambda_wave

    # Precompute cylinder outlines (in y/λ, x/λ coords matching heatmap axes)
    cyl_data = [(clocs[i,2]/lambda_wave, crads[i]/lambda_wave,
                 clocs[i,1]/lambda_wave) for i in 1:size(clocs,1)]

    total_frames = n_seg * n_frames
    println("  Rendering $total_frames frames …")

    anim = @animate for global_frame in 0:(total_frames-1)
        seg_idx = (global_frame ÷ n_frames) + 1
        t       = (global_frame % n_frames) * dt
        Field, label = fields_list[seg_idx]
        F_plot  = real.(Field .* exp(1im * omega * t))

        p = heatmap(y_full, x_grid, F_plot,
                    color=:jet, clims=(-vmax, vmax),
                    xlabel="y/λ", ylabel="x/λ",
                    title=label,
                    aspect_ratio=:equal, size=(1000, 500),
                    colorbar=true)

        # Slab boundaries
        vline!(p, [0.0, d_lam], color=:white, lw=2, label="")

        # Channel entry / exit markers
        hline!(p, [PERIOD/4/lambda_wave - W_lam/2,
                   PERIOD/4/lambda_wave + W_lam/2],
               color=:cyan, lw=1, ls=:dot, label="")
        hline!(p, [3*PERIOD/4/lambda_wave - W_lam/2,
                   3*PERIOD/4/lambda_wave + W_lam/2],
               color=:cyan, lw=1, ls=:dot, label="")

        # Cylinder outlines (draw only a subset near slab for speed)
        for (cy, r, cx) in cyl_data
            θs = range(0, 2π, length=20)
            plot!(p, cy .+ r .* cos.(θs), cx .+ r .* sin.(θs),
                  color=:black, lw=0.6, label="")
        end
    end

    try
        mp4(anim, outfile * ".mp4"; fps=fps)
        println("  Saved → $(outfile).mp4")
    catch e
        @warn "MP4 failed ($e) — saving GIF instead"
        gif(anim, outfile * ".gif"; fps=fps)
        println("  Saved → $(outfile).gif")
    end
end

# ── Static preview plot ────────────────────────────────────────────────────────
function static_preview(fields_list, x_grid, y_full, d, clocs, crads, lambda_wave, outfile)
    vmax = maximum([maximum(abs.(f[1])) for f in fields_list]) * 0.75
    vmax = max(vmax, 1e-10)
    d_lam = d / lambda_wave

    plots_list = []
    for (F, label) in fields_list
        F_plot = real.(F)
        p = heatmap(y_full, x_grid, F_plot,
                    color=:jet, clims=(-vmax, vmax),
                    xlabel="y/λ", ylabel="x/λ", title=label,
                    aspect_ratio=:equal, size=(700, 400))
        vline!(p, [0.0, d_lam], color=:white, lw=2, label="")
        for i in 1:size(clocs,1)
            r   = crads[i] / lambda_wave
            cy  = clocs[i,2] / lambda_wave
            cx  = clocs[i,1] / lambda_wave
            θs  = range(0, 2π, length=20)
            plot!(p, cy .+ r.*cos.(θs), cx .+ r.*sin.(θs),
                  color=:black, lw=0.5, label="")
        end
        push!(plots_list, p)
    end

    combined = plot(plots_list..., layout=(1, length(plots_list)),
                    size=(700*length(plots_list), 420))
    savefig(combined, outfile * "_preview.png")
    println("  Saved → $(outfile)_preview.png")
end

# ── Main ──────────────────────────────────────────────────────────────────────
function main()
    args = parse_args()
    grid_res = args["grid_res"]
    n_frames = args["frames"]
    fps      = args["fps"]
    outfile  = joinpath(OUTDIR, args["output"])

    println("=" ^ 62)
    println("  CyScat — S-Channel Maze")
    println("=" ^ 62)
    @printf("  λ=%.2f  D=%.2f  r=%.2f  W=%.2f  L=%.2f\n",
            WAVELENGTH, PERIOD, RADIUS, CHANNEL_W, SLAB_THICK)
    @printf("  Channel entry  x ≈ D/4  = %.2f  (±%.2f)\n",
            PERIOD/4, CHANNEL_W/2)
    @printf("  Channel exit   x ≈ 3D/4 = %.2f  (±%.2f)\n",
            3*PERIOD/4, CHANNEL_W/2)

    clocs, thickness, S, n_prop, n_eva, nmax, crads, cmmaxs, cepmus, sp =
        setup_maze()

    @printf("  nmax=%d  (M=%d propagating modes)\n", nmax, 2*nmax+1)

    # Sanity check: how unitary is S?
    nm   = 2*nmax + 1
    S11  = S[1:nm, 1:nm];  S21 = S[nm+1:end, 1:nm]
    unit_err = norm(S11'*S11 + S21'*S21 - I)
    @printf("  Unitarity check ‖S11'S11 + S21'S21 − I‖ = %.2e\n", unit_err)

    fields_list = Tuple{Matrix{ComplexF64}, String}[]
    local x_grid, y_full, thickness_val

    for mode in ["normal", "opt_trans", "opt_reflect"]
        @printf("\n  Building field: mode = %s …\n", mode)
        t0 = time()
        F, xg, yf, tv, lbl = build_fields(
            S, nmax, n_eva, thickness, mode,
            clocs, crads, cmmaxs, cepmus, PERIOD, WAVELENGTH, sp;
            grid_res = grid_res)
        @printf("  Done in %.1f s   %s\n", time()-t0, lbl)
        push!(fields_list, (F, lbl))
        x_grid, y_full, thickness_val = xg, yf, tv
    end

    println("\n  Saving static preview …")
    static_preview(fields_list, x_grid, y_full, thickness_val,
                   clocs, crads, WAVELENGTH, outfile)

    println("\n  Generating animation …")
    make_movie(fields_list, x_grid, y_full, thickness_val,
               clocs, crads, WAVELENGTH, outfile;
               n_frames=n_frames, fps=fps)

    println("=" ^ 62)
end

main()