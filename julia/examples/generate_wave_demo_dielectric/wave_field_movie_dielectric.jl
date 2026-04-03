"""
wave_field_movie.jl
===================
Generates an animation of the EM wave field scattered by a slab of
cylinders, matching the MATLAB CyScat GUI visualization style (Curtis Jin).

The video shows two segments:
  1. Normal incidence — shows how much a plane wave transmits
  2. Optimal wavefront — SVD of S11, maximizes transmission

Usage:
    julia --project=. examples/wave_field_movie.jl
    julia --project=. examples/wave_field_movie.jl --pec --num_cyl 50
    julia --project=. examples/wave_field_movie.jl --mode normal

Outputs: wave_field_movie.mp4 (or .gif if ffmpeg unavailable)
"""

using LinearAlgebra
using SpecialFunctions
using Printf
using Statistics
using Random
using Plots

# ── Parse arguments ────────────────────────────────────────────────────────────
function parse_args()
    args = Dict{String,Any}(
        "mode"    => "both",       # "both" = normal + optimal, or "normal", "opt_trans", "opt_reflect"
        "num_cyl" => 50,
        "fps"     => 20,
        "frames"  => 160,
        "output"  => "wave_field_movie_dielectric",
        "seed"    => 42,
        "pec"     => false,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--mode" && i < length(ARGS)
            args["mode"] = ARGS[i+1]; i += 2
        elseif a == "--num_cyl" && i < length(ARGS)
            args["num_cyl"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--fps" && i < length(ARGS)
            args["fps"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--frames" && i < length(ARGS)
            args["frames"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--seed" && i < length(ARGS)
            args["seed"] = parse(Int, ARGS[i+1]); i += 2
        elseif a in ("-o", "--output") && i < length(ARGS)
            args["output"] = ARGS[i+1]; i += 2
        elseif a == "--pec"
            args["pec"] = true; i += 1
        else
            i += 1
        end
    end
    return args
end

# ── Physical constants ─────────────────────────────────────────────────────────
const OUTDIR     = @__DIR__
const WAVELENGTH = 0.93
const PERIOD     = 12.81
const RADIUS     = 0.25
const N_CYL      = 1.3
const MU         = 1.0
const CMMAX      = 5
const PHIINC     = π / 2
const Eva_TOL    = 1e-2
const GRID_RES   = 200
const PR_Y       = 7         # Plot region: extend 7λ on each side (like MATLAB default ~10)

function compute_nmax()
    n_prop = floor(Int, PERIOD / WAVELENGTH)
    n_eva  = max(floor(Int,
        PERIOD / (2π) * sqrt((log(Eva_TOL) / (2*RADIUS))^2 + (2π/WAVELENGTH)^2)
    ) - n_prop, 0)
    return n_prop, n_eva, n_prop + n_eva
end

# ── Load Julia scattering code ─────────────────────────────────────────────────
const SCRIPT_DIR = joinpath(@__DIR__, "..", "..", "CyScat")

push!(LOAD_PATH, SCRIPT_DIR)
push!(LOAD_PATH, joinpath(SCRIPT_DIR, "Scattering_Code"))

include(joinpath(SCRIPT_DIR, "Scattering_Code", "ky.jl"))
include(joinpath(SCRIPT_DIR, "Scattering_Code", "smatrix_parameters.jl"))
include(joinpath(SCRIPT_DIR, "Scattering_Code", "smatrix.jl"))
include(joinpath(SCRIPT_DIR, "get_partition.jl"))

# ── Cylinder placement ─────────────────────────────────────────────────────────
function place_cylinders(num_cyl, period, thickness, radius; seed=42)
    rng     = MersenneTwister(seed)
    margin  = radius * 1.5
    min_sep = 2.5 * radius
    clocs   = zeros(num_cyl, 2)
    for i in 1:num_cyl
        placed = false
        for _ in 1:10000
            x = margin + rand(rng) * (period - 2*margin)
            y = margin + rand(rng) * (thickness - 2*margin)
            if i == 1 || all(sqrt.((x .- clocs[1:i-1,1]).^2 .+
                                   (y .- clocs[1:i-1,2]).^2) .> min_sep)
                clocs[i,:] = [x, y]
                placed = true
                break
            end
        end
        placed || @warn "Could not place cylinder $i"
    end
    return clocs
end

# ── Setup: place cylinders + compute S-matrix ─────────────────────────────────
function setup(num_cyl; seed=42, pec=false)
    spacing      = 2.5 * RADIUS
    cyls_per_row = floor(Int, PERIOD / spacing)
    rows_needed  = num_cyl / cyls_per_row + 2
    thickness    = round(max(0.5, rows_needed * spacing * 1.5), digits=1)

    clocs  = place_cylinders(num_cyl, PERIOD, thickness, RADIUS; seed=seed)

    n_prop, n_eva_dielectric, _ = compute_nmax()
    n_eva = pec ? 0 : n_eva_dielectric
    nmax  = n_prop + n_eva

    sp     = smatrix_parameters(WAVELENGTH, PERIOD, PHIINC,
                                1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, PERIOD/120)
    cmmaxs = fill(CMMAX, num_cyl)
    eps_v  = pec ? -1.0 : N_CYL^2
    cepmus = repeat([eps_v MU], num_cyl, 1)
    crads  = fill(RADIUS, num_cyl)

    material = pec ? "PEC" : "Dielectric (n=$(N_CYL))"
    println("  Material: $material")
    println("  Computing S-matrix ($num_cyl cylinders, nmax=$nmax) ...")
    t0 = time()
    S, _ = smatrix(clocs, cmmaxs, cepmus, crads, PERIOD, WAVELENGTH,
                   nmax, thickness, sp, "On")
    @printf("  S-matrix done in %.1fs\n", time() - t0)

    return clocs, thickness, S, n_prop, n_eva, nmax, material, crads, cmmaxs, cepmus, sp
end

# ── Build incident wavefront ──────────────────────────────────────────────────
function make_input(S, nmax, n_eva, mode)
    nm   = 2*nmax + 1
    S11  = S[1:nm,    1:nm]
    S21  = S[nm+1:end, 1:nm]

    if mode == "opt_trans"
        # Optimize transmission via SVD of S11 (minimize reflection → maximize transmission)
        # Use V[:,end] of S11 (smallest singular value = least reflected)
        R_trunc = n_eva > 0 ? S11[n_eva+1:end-n_eva, n_eva+1:end-n_eva] : S11
        T_trunc = n_eva > 0 ? S21[n_eva+1:end-n_eva, n_eva+1:end-n_eva] : S21
        F = svd(R_trunc)
        v_opt = F.V[:, end]  # last right singular vector = minimum reflection
        Input = zeros(ComplexF64, nm)
        if n_eva > 0
            Input[n_eva+1:end-n_eva] = v_opt
        else
            Input = v_opt
        end
        # Compute transmission coefficient for this input
        tc = sum(abs2, T_trunc * v_opt)
        label = @sprintf("Optimal Wavefront — %.2f%% transmitted.", tc * 100)
    elseif mode == "opt_reflect"
        R_trunc = n_eva > 0 ? S11[n_eva+1:end-n_eva, n_eva+1:end-n_eva] : S11
        F = svd(R_trunc)
        v_opt = F.V[:, 1]
        Input = zeros(ComplexF64, nm)
        if n_eva > 0
            Input[n_eva+1:end-n_eva] = v_opt
        else
            Input = v_opt
        end
        label = @sprintf("Optimal Reflection (τ=%.4f)", F.S[1])
    else  # normal incidence
        Input = zeros(ComplexF64, nm)
        Input[nmax+1] = 1.0
        # Compute transmission coefficient for normal incidence
        T_trunc = n_eva > 0 ? S21[n_eva+1:end-n_eva, n_eva+1:end-n_eva] : S21
        center = n_eva > 0 ? (nmax - n_eva + 1) : (nmax + 1)
        tc = sum(abs2, T_trunc[:, center])
        label = @sprintf("Normal Incidence — %.2f%% transmitted.", tc * 100)
    end

    return S11, S21, Input, label
end
#=
# ── Build field on unified y-grid ─────────────────────────────────────────────
"""
Compute the field across three regions, stitched into one array.
Axes use x/λ and y/λ units to match MATLAB.
"""
function build_fields(S, nmax, n_eva, thickness, mode, clocs, crads)
    nm  = 2*nmax + 1
    S11, S21, Input, label = make_input(S, nmax, n_eva, mode)

    k   = 2π / WAVELENGTH
    m   = collect(-nmax:nmax)
    kxs = 2π / PERIOD .* m
    kys = ky(k, complex.(kxs))

    nor = sqrt.(kys ./ k)
    P1  = Diagonal(1.0 ./ nor)

    Incident_c = P1 * Input
    Reflect_c  = P1 * (S11 * Input)
    Trans_c    = P1 * (S21 * Input)

    Nx = GRID_RES

    # Use x/λ units
    x_grid = collect(range(0, PERIOD / WAVELENGTH, length=Nx))

    # Propagating-mode subsets for incident field
    if n_eva > 0
        prop = n_eva+1:nm-n_eva
        inc_c  = Incident_c[prop]; kxs_p = kxs[prop]; kys_p = kys[prop]
    else
        inc_c  = Incident_c; kxs_p = kxs; kys_p = kys
    end

    # Three y-regions with λ-scaled coordinates
    Ly       = PR_Y * WAVELENGTH           # extend PR_Y wavelengths on each side
    d        = thickness
    Ny_side  = GRID_RES
    Ny_slab  = max(60, round(Int, GRID_RES * d / (2*Ly)))

    y_ref    = collect(range(-Ly, 0,    length=Ny_side))
    y_slab   = collect(range(0,   d,    length=Ny_slab))
    y_trans  = collect(range(d,   d+Ly, length=Ny_side))

    # Scale x for Floquet phases (physical units, not normalized)
    x_phys = collect(range(0, PERIOD, length=Nx))

    # ── Reflection field (y < 0): incident + backscattered ──
    exp_x_p  = exp.(1im .* x_phys * (-kxs_p)')
    exp_y_i  = exp.(-1im .* kys_p * y_ref')
    IncField_ref = exp_x_p * Diagonal(inc_c) * exp_y_i

    exp_x_a  = exp.(1im .* x_phys * (-kxs)')
    exp_y_b  = exp.(1im .* kys * y_ref')
    BackField = exp_x_a * Diagonal(Reflect_c) * exp_y_b

    RefField = IncField_ref .+ BackField

    # ── Slab interior: propagating modes only ──
    prop_mask = [abs(imag(kys[im])) < 1e-10 for im in 1:nm]
    prop_idx  = findall(prop_mask)

    kxs_slab = kxs[prop_idx]
    kys_slab = real.(kys[prop_idx])
    inc_slab = Incident_c[prop_idx]
    ref_slab = Reflect_c[prop_idx]
    tra_slab = Trans_c[prop_idx]
    exp_x_slab = exp.(1im .* x_phys * (-kxs_slab)')

    SlabField = zeros(ComplexF64, Nx, Ny_slab)

    for jy in 1:Ny_slab
        y  = y_slab[jy]
        α  = d > 0 ? y / d : 0.0

        fwd_amp = inc_slab .* (1 - α) .+ tra_slab .* α
        bwd_amp = ref_slab .* (1 - α)

        fwd_phase = exp.(-1im .* kys_slab .* y)
        bwd_phase = exp.( 1im .* kys_slab .* y)

        field_modes = fwd_amp .* fwd_phase .+ bwd_amp .* bwd_phase
        SlabField[:, jy] = exp_x_slab * field_modes
    end

    # Zero out field inside cylinder radii
    for jy in 1:Ny_slab, ix in 1:Nx
        px = x_phys[ix]
        py = y_slab[jy]
        for ic in 1:size(clocs, 1)
            if (px - clocs[ic,1])^2 + (py - clocs[ic,2])^2 < crads[ic]^2
                SlabField[ix, jy] = 0.0
                break
            end
        end
    end

    # ── Transmission field (y > d) ──
    exp_y_t    = exp.(-1im .* kys * (y_trans .- d)')
    TransField = exp_x_a * Diagonal(Trans_c) * exp_y_t

    # ── Stitch and convert y to y/λ ──
    y_full    = vcat(y_ref, y_slab, y_trans) ./ WAVELENGTH
    FullField = hcat(RefField, SlabField, TransField)

    return FullField, x_grid, y_full, d, label, Input, nmax, n_eva, kxs, k
end
=#
# ... (Keep arguments and physical constants as they were) ...

# ── Corrected build_fields to match MATLAB Plot 22 ──
using SpecialFunctions: hankelh1, hankelh2
using LinearAlgebra
using Printf

function build_fields(S, nmax, n_eva, thickness, mode, clocs, crads, 
                      cmmaxs, cepmus, period, lambda_wave, sp)
    nm = 2 * nmax + 1
    S11 = S[1:nm, 1:nm]
    S21 = S[nm+1:end, 1:nm]
    
    # 1. SVD & Input Selection
    R_trunc = n_eva > 0 ? S11[n_eva+1:end-n_eva, n_eva+1:end-n_eva] : S11
    F = svd(R_trunc)
    
    if mode == "opt_trans"
        v_in = F.V[:, end]
    elseif mode == "opt_reflect"
        v_in = F.V[:, 1]
    else
        v_in = zeros(ComplexF64, size(R_trunc, 1))
        v_in[Int(floor(size(R_trunc,1)/2)+1)] = 1.0
    end

    Input = zeros(ComplexF64, nm)
    if n_eva > 0; Input[n_eva+1:end-n_eva] = v_in; else Input = v_in; end

    # 2. Dynamic Transmission % (propagating modes only)
    S21_trunc = n_eva > 0 ? S21[n_eva+1:end-n_eva, n_eva+1:end-n_eva] : S21
    trans_coeffs = S21_trunc * v_in
    percent_trans = (sum(abs2.(trans_coeffs)) / sum(abs2.(v_in))) * 100
    mode_label = mode == "opt_trans" ? "Optimal Wavefront" :
                 mode == "opt_reflect" ? "Optimal Reflection" : "Normal Incidence"
    label = @sprintf("%s — %.2f%% transmitted.", mode_label, percent_trans)

    # 3. Solve for c_vector (Scattering Coefficients)
    k = 2π / lambda_wave
    t_mat = transall(clocs, cmmaxs, period, lambda_wave, sp["phiinc"], sp, 1)
    s_vec = sall(cmmaxs, cepmus, crads, lambda_wave)
    z = I - Diagonal(s_vec) * t_mat
    
    # Matching your efficiency: solve via LU
    lu_fact = lu(z)
    
    v_total = zeros(ComplexF64, size(z, 1))
    for nin in -nmax:nmax
        kxex = sp["kxs"][sp["MiddleIndex"] + nin]
        v_single = Diagonal(s_vec) * vall(clocs, cmmaxs, lambda_wave, kxex, 1)
        v_total += v_single * Input[nin + nmax + 1]
    end
    c_vector = lu_fact \ v_total

    # 4. Field Computation
    Nx = GRID_RES
    Ly = PR_Y * lambda_wave
    y_full = collect(range(-Ly, thickness + Ly, length=GRID_RES * 2))
    x_phys = collect(range(0, period, length=Nx))
    FullField = zeros(ComplexF64, Nx, length(y_full))

    m_flo = collect(-nmax:nmax)
    kxs = 2π / period .* m_flo
    kys = ky(k, complex.(kxs))
    P1 = Diagonal(1.0 ./ sqrt.(kys ./ k))
    
    Inc_c, Ref_c, Tra_c = P1*Input, P1*(S11*Input), P1*(S21*Input)
    num_cyl, m_max = size(clocs, 1), Int((size(c_vector, 1) / size(clocs, 1) - 1) / 2)

    for (jy, y) in enumerate(y_full), ix in 1:Nx
        x = x_phys[ix]
        if y < 0
            prop = n_eva+1:nm-n_eva
            FullField[ix, jy] = sum(Inc_c[prop] .* exp.(1im .* (-kxs[prop] .* x .- kys[prop] .* y))) + 
                                sum(Ref_c[prop] .* exp.(1im .* (-kxs[prop] .* x .+ kys[prop] .* y)))  # <-- add [prop] here
        elseif y > thickness
            prop = n_eva+1:nm-n_eva
            FullField[ix, jy] = sum(Tra_c[prop] .* exp.(1im .* (-kxs[prop] .* x .- kys[prop] .* (y - thickness))))
        else
            # Interior with physical cylinders
            total_f = sum(Inc_c[n_eva+1:nm-n_eva] .* exp.(1im .* (-kxs[n_eva+1:nm-n_eva] .* x .- kys[n_eva+1:nm-n_eva] .* y)))
            for i in 1:num_cyl
                dx, dy = x - clocs[i,1], y - clocs[i,2]
                r = sqrt(dx^2 + dy^2)
                if r <= crads[i]
                    total_f = 0.0im; break
                end
                θ = atan(dy, dx)
                idx_start = (i-1)*(2*m_max + 1) + 1
                for m_idx in -m_max:m_max
                    total_f += c_vector[idx_start + m_idx + m_max] * hankelh2(m_idx, k*r) * exp(1im * m_idx * θ)
                end
            end
            FullField[ix, jy] = total_f
        end
    end
    return FullField, x_phys ./ lambda_wave, y_full ./ lambda_wave, thickness, label
end

function make_movie(fields_list, x_grid, y_full, d, clocs, crads, lambda_wave, outfile; n_frames=80, fps=20)
    T_period, n_seg = 4.0, length(fields_list)
    omega, dt = 2π / T_period, T_period / n_frames
    
    # Global max for consistent color scaling
    vmax = maximum([maximum(abs.(real.(f[1]))) for f in fields_list]) * 0.7

    anim = @animate for global_frame in 0:(n_seg * n_frames - 1)
        seg_idx = (global_frame ÷ n_frames) + 1
        t = (global_frame % n_frames) * dt
        Field, label = fields_list[seg_idx]
        F_plot = real.(Field .* exp(1im * omega * t))

        # 1. Plot the Wave Field
        p = heatmap(y_full, x_grid, F_plot, color=:jet, clims=(-vmax, vmax),
                    xlabel="y/λ", ylabel="x/λ", title=label,
                    aspect_ratio=:equal, size=(900, 500))
        
        # 2. Plot Slab Boundaries
        vline!(p, [0.0, d/lambda_wave], color=:magenta, lw=2, label="")

        # 3. ADD THIS: Plot Black Outlines for Cylinders
        # We loop through each cylinder and draw a hollow circle
        for i in 1:size(clocs, 1)
            r_lam = crads[i] / lambda_wave
            cx, cy = clocs[i, 1] / lambda_wave, clocs[i, 2] / lambda_wave
            θ = range(0, 2π, length=40)
            
            # Use seriestype=:path for just the outline
            plot!(p, cy .+ r_lam .* cos.(θ), cx .+ r_lam .* sin.(θ), 
                  color=:black, lw=1.2, label="")
        end
    end
    mp4(anim, outfile * ".mp4", fps=fps)
end

# ── Plot: Transmission eigenvalue distribution f(τ) ──────────────────────────
"""
Plot histogram of transmission singular values (like Curtis's f(τ) plot).
"""
function plot_transmission_distribution(S, nmax, n_eva, num_cyl, material;
                                        outfile="transmission_distribution")
    nm  = 2*nmax + 1
    S11 = S[1:nm,    1:nm]
    S21 = S[nm+1:end, 1:nm]

    # Truncate evanescent modes
    if n_eva > 0
        S11T = S11[n_eva+1:end-n_eva, n_eva+1:end-n_eva]
        S21T = S21[n_eva+1:end-n_eva, n_eva+1:end-n_eva]
    else
        S11T = S11
        S21T = S21
    end

    tau = svdvals(S21T)

    p = histogram(tau, bins=range(0, 1.05, length=50), normalize=:pdf,
                  xlabel="τ", ylabel="f(τ)",
                  title="$num_cyl $material Cylinders",
                  legend=false, color=:navy, linecolor=:navy,
                  xlims=(0, 1.05), size=(700, 500))

    savefig(p, outfile * ".png")
    println("  Saved -> $(outfile).png")
    return p
end

# ── Plot: Modal coefficients of optimal wavefront ────────────────────────────
"""
Plot magnitude and phase of the optimal wavefront modal coefficients
(like Curtis's "Modal coefficients of the Optimal Wavefront" plot).
"""
function plot_modal_coefficients(S, nmax, n_eva;
                                 outfile="modal_coefficients")
    nm  = 2*nmax + 1
    S11 = S[1:nm,    1:nm]
    S21 = S[nm+1:end, 1:nm]
    k   = 2π / WAVELENGTH
    m   = collect(-nmax:nmax)
    kxs = 2π / PERIOD .* m
    kys = ky(k, complex.(kxs))
    nor = sqrt.(kys ./ k)
    P1  = Diagonal(1.0 ./ nor)

    # Truncate evanescent modes for SVD
    if n_eva > 0
        S11T = S11[n_eva+1:end-n_eva, n_eva+1:end-n_eva]
        kxs_prop = kxs[n_eva+1:end-n_eva]
    else
        S11T = S11
        kxs_prop = kxs
    end

    # Optimal wavefront: last right singular vector of S11 (min reflection)
    v_opt = svd(S11T).V[:, end]

    # Physical modal amplitudes (with normalization)
    # Define the propagating range once
    prop_range = (n_eva + 1):(nm - n_eva)

    # 1. Initialize the full modal input (size nm)
    Input_full = zeros(ComplexF64, nm)
    
    # 2. Place the optimal wavefront into the propagating slots
    # This works perfectly even if n_eva is 0
    Input_full[prop_range] = v_opt

    # 3. Apply the power-normalization matrix P1
    # P1 should be a Diagonal matrix of size (nm x nm)
    modal_amp = P1 * Input_full

    # Propagating modes only for plotting
    if n_eva > 0
        modal_prop = modal_amp[n_eva+1:end-n_eva]
    else
        modal_prop = modal_amp
    end

    angles = 180 / π .* asin.(kxs_prop ./ k)

    p = plot(layout=(1,1), size=(800, 500))
    plot!(p, angles, abs.(modal_prop),
          color=:blue, lw=2, marker=:circle, markersize=3,
          ylabel="Magnitude", label="Magnitude",
          legend=:topright)

    # Phase on twin y-axis (use twinx)
    p2 = twinx(p)
    plot!(p2, angles, angle.(modal_prop) .* (180/π),
          color=:green, lw=2, ls=:dash, marker=:diamond, markersize=3,
          ylabel="Phase (°)", label="Phase",
          legend=:topleft)

    title!(p, "Modal coefficients of the Optimal Wavefront")
    xlabel!(p, "Angles (°)")

    savefig(p, outfile * ".png")
    println("  Saved -> $(outfile).png")
    return p
end
#=
# ── Render animation (Curtis-style single panel) ─────────────────────────────
function make_movie(fields_list, x_grid, y_full, d, clocs, crads, num_cyl, material;
                    n_frames=80, fps=20, outfile="wave_field_movie")

    T_period = 4.0
    omega    = 2π / T_period
    dt       = T_period / n_frames
    d_lam    = d / WAVELENGTH

    # Precompute cylinder outlines in y/λ, x/λ coordinates
    nθ = range(0, 2π, length=60)
    cyl_data = [(cyl[2] / WAVELENGTH .+ crads[ic] / WAVELENGTH .* cos.(nθ),
                 cyl[1] / WAVELENGTH .+ crads[ic] / WAVELENGTH .* sin.(nθ))
                for (ic, cyl) in enumerate(eachrow(clocs))]

    # Determine global color range across all fields
    vmax = maximum(maximum(abs.(real.(F))) for (F, _) in fields_list)
    vmax = max(vmax, 1e-10)

    # Build frame list: for "both" mode, show each mode for n_frames
    total_frames = length(fields_list) * n_frames

    println("  Rendering $total_frames frames ($(length(fields_list)) segment(s) × $n_frames) ...")
    anim = @animate for global_frame in 0:total_frames-1
        seg_idx = global_frame ÷ n_frames + 1
        frame   = global_frame % n_frames
        t       = frame * dt
        phase   = exp(1im * omega * t)

        Field, label = fields_list[seg_idx]
        F = real.(Field .* phase)

        p = heatmap(y_full, x_grid, F,
                    color=:jet, clims=(-vmax, vmax),
                    xlabel="y/λ", ylabel="x/λ",
                    colorbar=true, aspect_ratio=:none,
                    size=(1000, 600))

        # Slab boundary lines (magenta like MATLAB)
        vline!(p, [0.0],   color=:magenta, lw=2.5, label="")
        vline!(p, [d_lam], color=:magenta, lw=2.5, label="")

        # Draw cylinders
        for (cy, cx) in cyl_data
            plot!(p, cy, cx, seriestype=:shape,
                  fillcolor=:white, fillalpha=0.0,
                  linecolor=:blue, lw=0.8, label="")
        end

        title!(p, "$label")
    end

    try
        mp4(anim, outfile * ".mp4", fps=fps)
        println("  Saved -> $(outfile).mp4")
    catch e
        @warn "MP4 failed ($e), saving GIF..."
        gif(anim, outfile * ".gif", fps=fps)
        println("  Saved -> $(outfile).gif")
    end
end
=#

# ── Main ───────────────────────────────────────────────────────────────────────
function main()
    args = parse_args()
    mode     = args["mode"]
    num_cyl  = args["num_cyl"]
    fps      = args["fps"]
    n_frames = args["frames"]
    outfile  = joinpath(OUTDIR, args["output"])
    seed     = args["seed"]
    pec      = args["pec"]

    println("=" ^ 65)
    println("  CyScat -- Wave Field Movie (Julia)")
    println("=" ^ 65)
    @printf("  Mode: %s  |  Cylinders: %d  |  Seed: %d  |  PEC: %s\n",
            mode, num_cyl, seed, pec)
    @printf("  λ=%.2f  Λ=%.2f  r=%.2f\n", WAVELENGTH, PERIOD, RADIUS)

    clocs, thickness, S, n_prop, n_eva, nmax, material, crads, cmmaxs, cepmus, sp =
        setup(num_cyl; seed=seed, pec=pec)
    @printf("  NMAX=%d  (N_prop=%d  N_eva=%d)\n", nmax, n_prop, n_eva)

    # Build field arrays for each mode
    # Build field arrays for each mode
    # 1. Initialize the list for movie segments
    fields_list = Tuple{Matrix{ComplexF64}, String}[]
    modes_to_run = (mode == "both") ? ["normal", "opt_trans"] : [mode]

    local x_grid, y_full, thickness_val

    for m in modes_to_run
        println("\n  Building field for mode=$m ...")
        
        # Pass the full physics set to build_fields
        F, x_grid, y_full, thickness_val, label = 
            build_fields(S, nmax, n_eva, thickness, m, clocs, crads, 
                         cmmaxs, cepmus, PERIOD, WAVELENGTH, sp)
            
        @printf("  vmax=%.4f  label: %s\n", maximum(abs.(real.(F))), label)
        push!(fields_list, (F, label))
    end

    println("\n  Generating animation...")
    # Add clocs and crads back here:
    make_movie(fields_list, x_grid, y_full, thickness_val, clocs, crads, WAVELENGTH, outfile;
               n_frames=n_frames, fps=fps)
    # Generate standalone plots
    println("\n  Generating standalone plots ...")
    plot_transmission_distribution(S, nmax, n_eva, num_cyl, material;
                                   outfile=outfile * "_tau_dist")
    plot_modal_coefficients(S, nmax, n_eva;
                            outfile=outfile * "_modal_coeffs")

    println("=" ^ 65)
end

main()
