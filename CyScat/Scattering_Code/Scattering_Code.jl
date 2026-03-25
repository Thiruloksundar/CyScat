"""
Scattering_Code Module
Core scattering matrix computation modules

Original MATLAB code by Curtis Jin (jsirius@umich.edu)
Translated to Julia
"""
module Scattering_Code

using LinearAlgebra
using SpecialFunctions
using ToeplitzMatrices
using ForwardDiff
using Dates

# Export main functions
export smatrix, smatrix_parameters
export sall, vall, transall
export scatteringcoefficientsall
export ky, modified_epsilon_shanks
export simulation_time_profile
export distmat
export efieldall, farefieldall, efieldallperiodic
export coefficients
export cascadertwo, truncator, transper

# Include all source files
include("ky.jl")
include("distmat.jl")
include("modified_epsilon_shanks.jl")
include("simulation_time_profile.jl")
include("sall.jl")
include("vall.jl")
include("transall.jl")
include("scattering_coefficients_all.jl")
include("smatrix_parameters.jl")
include("smatrix.jl")
include("efieldall.jl")
include("farefieldall.jl")
include("efieldallperiodic.jl")
include("coefficients.jl")
include("cascadertwo.jl")
include("truncator.jl")
include("transper.jl")

end # module
