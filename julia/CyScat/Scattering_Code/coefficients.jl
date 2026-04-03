"""
coefficients Code
Coder : Curtis Jin
Date  : 2010/OCT/13th Wednesday
Contact : jsirius@umich.edu
Description : Professor Michelson's version
            : coefficients generating code for all out scattering fields

Julia translation
"""

using LinearAlgebra

include("transall.jl")
include("sall.jl")
include("vall.jl")

"""
    coefficients(clocs, cmmaxs, cepmus, crads, period, lambda_val, phiinc, up_down, sp)

Generate scattering coefficients for all cylinders.

# Arguments
- `clocs`: Cylinder locations (ncyl x 2)
- `cmmaxs`: Maximum mode numbers for each cylinder
- `cepmus`: Relative permittivity/permeability for each cylinder
- `crads`: Cylinder radii
- `period`: Periodic spacing
- `lambda_val`: Wavelength
- `phiinc`: Incident angle
- `up_down`: Direction indicator (1 for up, -1 for down)
- `sp`: S-matrix parameters

# Returns
- `coeff`: Scattering coefficients
"""
function coefficients(clocs, cmmaxs, cepmus, crads, period, lambda_val, phiinc, up_down, sp)
    # Calculate total steps for progress bar (used by transall)
    no_cylinders = length(cmmaxs)
    total_steps = no_cylinders * (no_cylinders + 1) ÷ 2

    t = transall(clocs, cmmaxs, period, lambda_val, phiinc, sp, total_steps)
    s = sall(cmmaxs, cepmus, crads, lambda_val)

    # z = eye(length(s)) - diag(s)*t
    z = I - Diagonal(s) * t

    k = 2π / lambda_val
    kxex = k * cos(phiinc)

    # v = diag(s) * vall(...)
    v = Diagonal(s) * vall(clocs, cmmaxs, lambda_val, kxex, up_down)

    # coeff = z\v (solve linear system)
    coeff = z \ v

    return coeff
end
