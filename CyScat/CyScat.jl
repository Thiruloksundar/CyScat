"""
CyScat - Cylindrical Scattering Simulator
Main Module

Original MATLAB code by Curtis Jin (jsirius@umich.edu)
University of Michigan

Julia translation
"""
module CyScat

using Reexport

# Include and re-export Scattering_Code module
include("Scattering_Code/Scattering_Code.jl")
@reexport using .Scattering_Code

# Include utility files
include("round2.jl")
include("histn.jl")
include("get_partition.jl")
include("progressbar.jl")

# Export utility functions
export round2, histn
export get_partition, smat_to_s11, smat_to_s12, smat_to_s21, smat_to_s22
export progressbar

end # module
