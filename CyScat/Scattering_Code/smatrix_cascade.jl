"""
smatrix_cascade - Auto-cascading S-matrix computation

Automatically divides large cylinder configurations into groups
and cascades their S-matrices. Falls back to direct computation
for small configurations.

Usage: Drop-in replacement for smatrix() - same arguments, same output.
"""

using LinearAlgebra
using Printf

# Ensure cascadertwo is available
if !isdefined(Main, :cascadertwo)
    include("cascadertwo.jl")
end

"""
    smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, lambda_wave, nmax, d, sp, interaction;
                    cascade_threshold=30, cylinders_per_group=15)

Compute S-matrix with automatic cascading for large numbers of cylinders.

Same interface as `smatrix()`, with two extra keyword arguments:
- `cascade_threshold`: Number of cylinders above which cascading is used (default: 30)
- `cylinders_per_group`: Target number of cylinders per group (default: 15)

# Returns
- `S`: Scattering matrix (same format as smatrix)
- `STP`: Simulation time profile dictionary
"""
function smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, lambda_wave, nmax, d, sp, interaction;
                          cascade_threshold=150, cylinders_per_group=100)
    no_cylinders = size(clocs, 1)

    # For small problems, use direct computation
    if no_cylinders <= cascade_threshold
        println("Direct computation ($no_cylinders cylinders)")
        return smatrix(clocs, cmmaxs, cepmus, crads, period, lambda_wave, nmax, d, sp, interaction)
    end

    # ===== AUTOMATIC CASCADE =====
    num_groups = max(2, ceil(Int, no_cylinders / cylinders_per_group))

    println("=" ^ 60)
    @printf("AUTO-CASCADE: %d cylinders → %d groups (~%d each)\n",
            no_cylinders, num_groups, ceil(Int, no_cylinders / num_groups))
    println("=" ^ 60)

    total_start = time()

    # Sort cylinders by y-position for proper cascade ordering
    sorted_idx = sortperm(clocs[:, 2])
    sorted_clocs = clocs[sorted_idx, :]
    sorted_cmmaxs = cmmaxs[sorted_idx]
    sorted_cepmus = cepmus[sorted_idx, :]
    sorted_crads = crads[sorted_idx]

    # Divide into groups
    group_size = ceil(Int, no_cylinders / num_groups)

    S_list = []
    d_list = Float64[]

    for g in 1:num_groups
        i_start = (g - 1) * group_size + 1
        i_end = min(g * group_size, no_cylinders)
        if i_start > no_cylinders
            break
        end

        # Extract group data
        g_clocs = sorted_clocs[i_start:i_end, :]
        g_cmmaxs = sorted_cmmaxs[i_start:i_end]
        g_cepmus = sorted_cepmus[i_start:i_end, :]
        g_crads = sorted_crads[i_start:i_end]
        n_in_group = i_end - i_start + 1

        # Compute group thickness from y-range
        y_min = minimum(g_clocs[:, 2])
        y_max = maximum(g_clocs[:, 2])
        margin = maximum(g_crads) * 2.0
        g_thickness = max(y_max - y_min + 2 * margin, 4 * margin)

        # Shift y-coordinates so group starts near y=0
        g_clocs_shifted = copy(g_clocs)
        g_clocs_shifted[:, 2] .-= (y_min - margin)

        @printf("  Group %d/%d: %d cylinders, thickness=%.3f\n",
                g, num_groups, n_in_group, g_thickness)

        # Compute S-matrix for this group (use original sp to preserve MiddleIndex/kxs)
        S_g, _ = smatrix(g_clocs_shifted, g_cmmaxs, g_cepmus, g_crads,
                          period, lambda_wave, nmax, g_thickness, sp, interaction)

        push!(S_list, S_g)
        push!(d_list, g_thickness)
    end

    # Cascade all groups
    println("\nCascading $(length(S_list)) groups...")
    cascade_start = time()

    Scas = S_list[1]
    dcas = d_list[1]
    for i in 2:length(S_list)
        Scas, dcas = cascadertwo(Scas, dcas, S_list[i], d_list[i])
        @printf("  Cascaded 1-%d, thickness=%.3f\n", i, dcas)
    end
    cascade_time = time() - cascade_start
    total_time = time() - total_start

    println()
    println("=" ^ 60)
    @printf("CASCADE COMPLETE\n")
    @printf("  S-matrix size: %d × %d\n", size(Scas)...)
    @printf("  Total thickness: %.3f\n", dcas)
    @printf("  Total time: %.2f seconds\n", total_time)
    println("=" ^ 60)

    # Build STP dict
    STP = Dict(
        "TST" => total_time,
        "cascade_time" => cascade_time,
        "num_groups" => length(S_list),
        "method" => "cascade"
    )

    return Scas, STP
end
