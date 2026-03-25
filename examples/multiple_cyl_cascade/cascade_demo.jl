"""
Cascade Demo for Large Cylinder Computations

This script demonstrates how to use the cascade method to compute
S-matrices for large numbers of cylinders (e.g., 500+) by:
1. Dividing cylinders into smaller groups
2. Computing S-matrix for each group independently
3. Cascading all S-matrices together

This is much more efficient than computing the full S-matrix at once.
"""

using LinearAlgebra
using Printf
using Random

const CYSC = joinpath(@__DIR__, "..", "..", "CyScat")

include(joinpath(CYSC, "Scattering_Code", "ky.jl"))
include(joinpath(CYSC, "Scattering_Code", "smatrix_parameters.jl"))
include(joinpath(CYSC, "Scattering_Code", "sall.jl"))
include(joinpath(CYSC, "Scattering_Code", "vall.jl"))
include(joinpath(CYSC, "Scattering_Code", "modified_epsilon_shanks.jl"))
include(joinpath(CYSC, "Scattering_Code", "simulation_time_profile.jl"))
include(joinpath(CYSC, "Scattering_Code", "transall.jl"))
include(joinpath(CYSC, "Scattering_Code", "scattering_coefficients_all.jl"))
include(joinpath(CYSC, "Scattering_Code", "smatrix.jl"))
include(joinpath(CYSC, "Scattering_Code", "cascadertwo.jl"))
include(joinpath(CYSC, "get_partition.jl"))

"""
Generate random cylinder positions within a slab.
"""
function generate_cylinder_positions(num_cyl, period, thickness, radius; seed=42)
    Random.seed!(seed)
    margin = radius * 2.0
    clocs = zeros(num_cyl, 2)

    for i in 1:num_cyl
        for attempt in 1:1000
            x = margin + rand() * (period - 2*margin)
            y = margin + rand() * (thickness - 2*margin)

            # Check for overlap with existing cylinders
            ok = true
            for j in 1:(i-1)
                dist = sqrt((x - clocs[j,1])^2 + (y - clocs[j,2])^2)
                if dist < 2.5 * radius
                    ok = false
                    break
                end
            end

            if ok
                clocs[i, :] = [x, y]
                break
            end
        end
    end

    return clocs
end

"""
Divide cylinders into groups based on their y-position (along slab thickness).
Returns groups sorted by y-position for proper cascading order.
"""
function divide_into_groups(clocs, num_groups)
    n = size(clocs, 1)

    # Sort by y-position
    sorted_indices = sortperm(clocs[:, 2])
    sorted_clocs = clocs[sorted_indices, :]

    # Divide into groups
    group_size = ceil(Int, n / num_groups)
    groups = []

    for g in 1:num_groups
        start_idx = (g-1) * group_size + 1
        end_idx = min(g * group_size, n)
        if start_idx <= n
            push!(groups, sorted_clocs[start_idx:end_idx, :])
        end
    end

    return groups
end

"""
Compute group thickness based on y-range of cylinders in the group.
"""
function compute_group_thickness(group_clocs, radius, margin=0.0)
    y_min = minimum(group_clocs[:, 2]) - radius - margin
    y_max = maximum(group_clocs[:, 2]) + radius + margin
    return y_max - y_min
end

"""
Compute S-matrix using cascade method for large numbers of cylinders.
"""
function smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, wavelength, nmax,
                          num_groups; verbose=true)
    total_time = 0.0

    # Divide cylinders into groups
    groups = divide_into_groups(clocs, num_groups)
    actual_num_groups = length(groups)

    if verbose
        println("=" ^ 60)
        println("CASCADE S-MATRIX COMPUTATION")
        println("=" ^ 60)
        println("Total cylinders: $(size(clocs, 1))")
        println("Number of groups: $actual_num_groups")
        for (i, g) in enumerate(groups)
            println("  Group $i: $(size(g, 1)) cylinders")
        end
        println()
    end

    # Compute S-matrix for each group
    S_matrices = []
    d_values = []

    for (i, group_clocs) in enumerate(groups)
        n_cyl = size(group_clocs, 1)
        group_cmmaxs = cmmaxs[1:n_cyl]  # Use same cmmax for all
        group_cepmus = cepmus[1:n_cyl, :]
        group_crads = crads[1:n_cyl]

        # Shift y-coordinates to start from 0 for this group
        y_min = minimum(group_clocs[:, 2])
        shifted_clocs = copy(group_clocs)
        shifted_clocs[:, 2] .-= (y_min - crads[1] * 2)  # Add margin

        thickness = compute_group_thickness(group_clocs, crads[1], crads[1])

        if verbose
            @printf("Computing group %d/%d (%d cylinders, thickness=%.2f)...\n",
                    i, actual_num_groups, n_cyl, thickness)
        end

        # Set up parameters
        sp = smatrix_parameters(wavelength, period, 0.1, 1e-8, 0.0001, 10, 10, 100, 5, 5, 1, 0.01)

        # Compute S-matrix for this group
        t_start = time()
        S, STP = smatrix(shifted_clocs, group_cmmaxs, group_cepmus, group_crads,
                         period, wavelength, nmax, thickness, sp, "Off")
        t_elapsed = time() - t_start
        total_time += t_elapsed

        if verbose
            @printf("  → Completed in %.2f seconds\n", t_elapsed)
        end

        push!(S_matrices, S)
        push!(d_values, thickness)
    end

    # Cascade all S-matrices
    if verbose
        println("\nCascading S-matrices...")
    end

    t_start = time()
    Scas = S_matrices[1]
    dcas = d_values[1]

    for i in 2:length(S_matrices)
        Scas, dcas = cascadertwo(Scas, dcas, S_matrices[i], d_values[i])
        if verbose
            @printf("  Cascaded groups 1-%d, total thickness=%.2f\n", i, dcas)
        end
    end
    cascade_time = time() - t_start
    total_time += cascade_time

    if verbose
        println()
        println("=" ^ 60)
        @printf("SUMMARY\n")
        @printf("  S-matrix size: %d × %d\n", size(Scas)...)
        @printf("  Total thickness: %.2f\n", dcas)
        @printf("  Group computation time: %.2f seconds\n", total_time - cascade_time)
        @printf("  Cascade time: %.4f seconds\n", cascade_time)
        @printf("  Total time: %.2f seconds\n", total_time)
        println("=" ^ 60)
    end

    return Scas, dcas, total_time
end

"""
Run cascade demo with different numbers of cylinders and groups.
"""
function run_cascade_demo()
    println("\n" * "=" ^ 70)
    println("CASCADE DEMO FOR LARGE CYLINDER COMPUTATIONS")
    println("=" ^ 70)

    # Parameters
    wavelength = 0.8
    period = 5.0
    radius = 0.1
    epsilon = 4.0
    mu = 1.0
    nmax = 2  # Keep small for demo
    cmmax = 2

    # Test with different numbers of cylinders
    test_configs = [
        (num_cyl=10, num_groups=2),
        (num_cyl=20, num_groups=4),
        # (num_cyl=50, num_groups=5),  # Uncomment for larger tests
        (num_cyl=500, num_groups=10),
    ]

    for config in test_configs
        num_cyl = config.num_cyl
        num_groups = config.num_groups
        thickness = num_cyl * 0.3  # Scale thickness with cylinders

        println("\n" * "-" ^ 70)
        println("Test: $num_cyl cylinders, $num_groups groups")
        println("-" ^ 70)

        # Generate cylinder positions
        clocs = generate_cylinder_positions(num_cyl, period, thickness, radius)

        # Material parameters (same for all cylinders)
        cmmaxs = fill(cmmax, num_cyl)
        cepmus = hcat(fill(epsilon, num_cyl), fill(mu, num_cyl))
        crads = fill(radius, num_cyl)

        # Compute using cascade
        Scas, dcas, cascade_time = smatrix_cascade(
            clocs, cmmaxs, cepmus, crads, period, wavelength, nmax, num_groups
        )

        # Extract transmission and reflection
        S11 = smat_to_s11(Scas)
        S21 = smat_to_s21(Scas)

        center = nmax + 1  # Center mode index
        R = sum(abs.(S11[:, center]).^2)
        T = sum(abs.(S21[:, center]).^2)

        println("\nResults:")
        @printf("  Reflection (R): %.6f\n", R)
        @printf("  Transmission (T): %.6f\n", T)
        @printf("  R + T: %.6f\n", R + T)
    end

    println("\n" * "=" ^ 70)
    println("Cascade demo complete!")
    println("=" ^ 70)
end
# Run the demo
run_cascade_demo()
