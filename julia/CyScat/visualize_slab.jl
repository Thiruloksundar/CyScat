"""
VisualizeSlab Code
Modified Date: 2011.05.19.Thursday
Coder: Curtis Jin
Contact: jsirius@umich.edu
Features: Showing visualized results.
          Fixing the display.
          Display the periodic structure

Julia translation
"""

using Plots
using MAT

"""
    visualize_slab(parameters, input_position; ax=nothing)

Visualize the slab structure with cylinders in periodic configuration.

# Arguments
- `parameters`: GP dictionary with slab parameters containing:
  - Radius: cylinder radius
  - Width: slab width
  - Thickness: slab thickness
  - Wavelength: wavelength
  - Period: period of the structure
- `input_position`: Positions of cylinders (Nx2 array) where each row is [x, y]
- `ax`: Plot object (optional, creates new plot if nothing)
"""
function visualize_slab(parameters::Dict, input_position; ax=nothing)
    println("Visualizing the slab...")

    # Extract parameters
    a = parameters["Radius"]
    Lx = parameters["Width"]
    Ly = parameters["Thickness"]
    lambda_wave = parameters["Wavelength"]

    # Try to load settings, use defaults if file doesn't exist
    buffer = 0.1 * lambda_wave  # Default buffer
    draw_buffer_line = 1  # Default to drawing buffer lines

    if isfile("system_settings_workspace.mat")
        try
            mat_data = matread("system_settings_workspace.mat")
            if haskey(mat_data, "AlgorithmSettingParameters")
                algorithm_params = mat_data["AlgorithmSettingParameters"]
                buffer = algorithm_params["Buffer"][1] * lambda_wave
            end
            if haskey(mat_data, "PlotSettingParameters")
                plot_params = mat_data["PlotSettingParameters"]
                draw_buffer_line = Int(plot_params["DrawBufferLine"][1])
            end
        catch
            # Use defaults
        end
    end

    d = Ly + buffer
    period = parameters["Period"]

    # Margins for plot
    margin_xleft = 1.1
    margin_xright = 1.1
    margin_y = 1.1

    # Create new plot if ax not provided
    if isnothing(ax)
        ax = plot(legend=false, aspect_ratio=:equal)
    end

    ## Main Box (center period)
    # Left Boundary
    y = 0:0.01:period
    x = zeros(length(y))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    if draw_buffer_line == 1
        # Right Boundary 1 (buffer line)
        y = 0:0.01:Lx
        x = Ly .* ones(length(y))
        plot!(ax, x, y, linestyle=:dash, color=:cyan, linewidth=1)
    end

    # Right Boundary 2 (main)
    y = 0:0.01:period
    x = d .* ones(length(y))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    if draw_buffer_line == 1
        # Upper Boundary 1 (buffer line)
        x = 0:0.01:Ly
        y = Lx .* ones(length(x))
        plot!(ax, x, y, linestyle=:dash, color=:cyan, linewidth=1)
    end

    # Upper Boundary 2 (main)
    x = 0:0.01:d
    y = period .* ones(length(x))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    # Lower Boundary
    x = 0:0.01:Ly
    y = zeros(length(x))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    ## Upper Box (offset = +period)
    y = period:0.01:2*period
    x = zeros(length(y))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    y = period:0.01:2*period
    x = d .* ones(length(y))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    x = 0:0.01:d
    y = 2*period .* ones(length(x))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    ## Lower Box (offset = -period)
    y = -period:0.01:0
    x = zeros(length(y))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    y = -period:0.01:0
    x = d .* ones(length(y))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    x = 0:0.01:Ly
    y = -period .* ones(length(x))
    plot!(ax, x, y, color=:magenta, linewidth=2)

    ## Drawing Cylinders
    resolution = 100
    theta = range(0, 2π, length=resolution)

    for idx in 1:size(input_position, 1)
        x_circle = a .* cos.(theta)
        y_circle = a .* sin.(theta)

        # Main period (offset = 0)
        plot!(ax, y_circle .+ input_position[idx, 2],
              x_circle .+ input_position[idx, 1],
              color=:blue)

        # Upper period (offset = +period)
        plot!(ax, y_circle .+ input_position[idx, 2],
              x_circle .+ input_position[idx, 1] .+ period,
              color=:blue)

        # Lower period (offset = -period)
        plot!(ax, y_circle .+ input_position[idx, 2],
              x_circle .+ input_position[idx, 1] .- period,
              color=:blue)
    end

    # Set labels and limits
    xlabel!(ax, "y (unit)")
    ylabel!(ax, "x (unit)")
    xlims!(ax, 0, Ly * margin_xright)
    ylims!(ax, -period * (margin_y - 1), period * margin_y)

    return ax
end
