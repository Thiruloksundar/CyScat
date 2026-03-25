"""
PositionGenerator
Coder: Curtis Jin
Date: 2011/MAR/31st Thursday
Contact: jsirius@umich.edu
Description: Generates Position

Julia translation
"""

using Random
using LinearAlgebra
using MAT

# Include distmat
include("Scattering_Code/distmat.jl")

"""
    position_generator(GP, optional_filename=nothing)

Generate cylinder positions based on various methods.

# Arguments
- `GP`: Global Parameters dictionary
- `optional_filename`: Optional filename for loading custom or saved positions

# Returns
- `modified_no_cylinders`: Final number of cylinders after removing overlaps
- `initial_positions`: Initial normalized positions (0-1 range)
- `real_positions`: Real positions in physical coordinates
"""
function position_generator(GP::Dict, optional_filename::Union{String, Nothing}=nothing)
    no_cylinders = GP["NoCylinders"]
    radius = GP["Radius"]
    width = GP["Width"]
    thickness = GP["Thickness"]
    random_set = GP["RandomSet"]
    rand_factor = GP["RandomFactor"]
    min_inter_distance = GP["MinInterDistance"]

    positions = nothing

    # Modulating the Position of the scatterer
    if random_set == "Custom"
        println("Generating Particles(Custom)...")

        if isnothing(optional_filename)
            println("Please specify a file name!")
            show_error_message(
                ["You are in \"Custom\" mode.",
                 "Please specify a file name which contains the \"InitialPositions\" vector."],
                "No file error!", "error"
            )
            return 0, nothing, nothing
        end

        if !isfile(optional_filename)
            println("$optional_filename does not exist!")
            show_error_message(
                ["$optional_filename does not exist!",
                 "Please enter the correct workspace file."],
                "No matching file error!", "error"
            )
            return 0, nothing, nothing
        end

        println("Loading $optional_filename...")
        mat_data = matread(optional_filename)

        if !haskey(mat_data, "InitialPositions")
            show_error_message(
                ["You are in \"Custom\" mode.",
                 "$optional_filename does not contain \"InitialPositions\" structure.",
                 "Please enter a workspace file that contains \"InitialPositions\" structure,",
                 "or Regenerate the S-Matrix."],
                "Wrong workspace file error!", "error"
            )
            return 0, nothing, nothing
        end

        positions = mat_data["InitialPositions"]

    elseif random_set == "Latin"
        println("Generating Particles(Latin Hypercube)...")
        # Latin Hypercube sampling
        positions = zeros(no_cylinders, 2)
        for dim in 1:2
            perm = randperm(no_cylinders)
            for i in 1:no_cylinders
                positions[i, dim] = (perm[i] - rand()) / no_cylinders
            end
        end

    elseif random_set == "Sobol"
        println("Generating Particles(Sobol)...")
        # Simple quasi-random approximation
        positions = zeros(no_cylinders, 2)
        for i in 1:no_cylinders
            positions[i, 1] = mod(i * 0.618033988749895, 1.0)  # Golden ratio
            positions[i, 2] = mod(i * 0.7548776662466927, 1.0)
        end

    elseif random_set == "Halt"
        println("Generating Particles(Halton)...")
        # Halton sequence approximation
        positions = zeros(no_cylinders, 2)
        for i in 1:no_cylinders
            positions[i, 1] = halton_sequence(i, 2)
            positions[i, 2] = halton_sequence(i, 3)
        end

    elseif random_set == "RandPertDet"
        # Third method: Random perturbation of deterministic grid
        sqrt_cylinders = Int(floor(sqrt(no_cylinders)))
        d = width
        a = radius

        X = range(0, 1, length=sqrt_cylinders + 1)[1:end-1]
        C = collect(X) .+ 1 / sqrt_cylinders / 2

        Cx = repeat(C, 1, sqrt_cylinders)
        Cy = repeat(C', sqrt_cylinders, 1)

        buffer = (width - 2*a) / sqrt_cylinders / 2 - a
        coeff = buffer * rand_factor
        coeff = coeff / (width - 2*a)

        Cx_flat = vec(Cx) .+ coeff * 2 .* (rand(sqrt_cylinders^2) .- 0.5)
        Cy_flat = vec(Cy) .+ coeff * 2 .* (rand(sqrt_cylinders^2) .- 0.5)

        positions = hcat(Cx_flat, Cy_flat)
        no_cylinders = sqrt_cylinders^2

    elseif random_set == "Load"
        if isnothing(optional_filename)
            show_error_message(
                ["You are in \"Load\" mode.",
                 "Please specify a file name of the workspace you want to use."],
                "No file error!", "error"
            )
            return 0, nothing, nothing
        end

        if !isfile(optional_filename)
            show_error_message(
                ["$optional_filename does not exist!",
                 "Please enter the correct workspace file."],
                "No matching file error!", "error"
            )
            return 0, nothing, nothing
        end

        mat_data = matread(optional_filename)

        if !haskey(mat_data, "parameters")
            show_error_message(
                ["$optional_filename does not contain \"parameters\" structure."],
                "Wrong workspace file error!", "error"
            )
            return 0, nothing, nothing
        end

        return 1, nothing, nothing

    else
        # Default: random positions
        println("Generating Particles(Random)...")
        positions = rand(no_cylinders, 2)
    end

    println("Initial number of cylinders: $no_cylinders")

    a = radius
    Lx = width
    Ly = thickness

    # Scaling positions
    temp_positions = zeros(size(positions))
    temp_positions[:, 1] = positions[:, 1] .* (Lx - 2*a)
    temp_positions[:, 2] = positions[:, 2] .* (Ly - 2*a)

    # Shifting positions
    temp_positions = temp_positions .+ a

    if no_cylinders > 1
        # Identifying Overlapping Cylinders
        dmat, _ = distmat(temp_positions)
        overlapping_list = zeros(no_cylinders)

        for idx1 in 1:no_cylinders
            for idx2 in (idx1+1):no_cylinders
                if dmat[idx1, idx2] < min_inter_distance
                    overlapping_list[idx1] = 1
                    overlapping_list[idx2] = 1
                end
            end
        end

        # Removing the overlapping parameters
        println("Removing overlapping cylinders...")
        real_positions = Float64[]
        initial_positions_clean = Float64[]

        for idx in 1:length(overlapping_list)
            if overlapping_list[idx] == 0
                append!(real_positions, temp_positions[idx, :])
                append!(initial_positions_clean, positions[idx, :])
            end
        end

        if length(real_positions) > 0
            real_positions = reshape(real_positions, 2, :)'
            initial_positions = reshape(initial_positions_clean, 2, :)'
        else
            real_positions = zeros(0, 2)
            initial_positions = zeros(0, 2)
        end

        modified_no_cylinders = size(real_positions, 1)

        println("Final number of cylinders: $modified_no_cylinders")

        if modified_no_cylinders == 0
            show_error_message(
                ["All the cylinders are overlapping!!",
                 "Change the configuration and try it again."],
                "No cylinder error!", "error"
            )
            return 0, nothing, nothing
        end

        # Sorting the positions in the order of distance from the origin
        r = sqrt.(real_positions[:, 1].^2 .+ real_positions[:, 2].^2)
        sorted_indices = sortperm(r)
        real_positions = real_positions[sorted_indices, :]
        initial_positions = initial_positions[sorted_indices, :]

    else
        # Single cylinder case
        modified_no_cylinders = 1
        real_positions = temp_positions
        initial_positions = positions
    end

    return modified_no_cylinders, initial_positions, real_positions
end


"""
    halton_sequence(index, base)

Generate Halton sequence value.
"""
function halton_sequence(index::Int, base::Int)
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0
        result += f * (i % base)
        i = i ÷ base
        f /= base
    end
    return result
end


"""
    show_error_message(messages, title, msg_type)

Display error message.
"""
function show_error_message(messages::Vector{String}, title::String, msg_type::String)
    println("[$(uppercase(msg_type))] $title:")
    for msg in messages
        println("  $msg")
    end
end
