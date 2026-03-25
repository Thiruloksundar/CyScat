"""
vall Code
Coder: Curtis Jin
Date: 2010/OCT/13th Wednesday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : planewave to cylinder harmonics converter

Julia translation
"""

include("ky.jl")

"""
    vall(clocs, cmmaxs, lambda_wave, kxex, up_down)

Convert plane wave to cylinder harmonics for all cylinders.

# Arguments
- `clocs`: Cylinder locations (Nx2 array)
- `cmmaxs`: Maximum mode numbers for each cylinder
- `lambda_wave`: Wavelength
- `kxex`: x component of excitation wave vector
- `up_down`: Direction flag (1 for up, -1 for down)

# Returns
- `v`: Cylinder harmonic coefficients
"""
function vall(clocs, cmmaxs, lambda_wave, kxex, up_down)
    no_cylinders = length(cmmaxs)
    tot_no_modes = Int(sum(cmmaxs .* 2 .+ 1))
    CT = Complex{eltype(clocs)}
    v = zeros(CT, tot_no_modes)

    istart = 1  # Julia 1-indexed
    for icyl in 1:no_cylinders
        cmmax = Int(cmmaxs[icyl])
        v[istart:istart + 2*cmmax] = vone(clocs[icyl, :], cmmax, lambda_wave, kxex, up_down)
        istart = istart + 2*cmmax + 1
    end

    return v
end

"""
    vone(cloc, cmmax, lambda_wave, kxex, up_down)

Convert plane wave to cylinder harmonics for one cylinder.

# Arguments
- `cloc`: Cylinder location [x, y]
- `cmmax::Int`: Maximum mode number
- `lambda_wave`: Wavelength
- `kxex`: x component of excitation wave vector
- `up_down`: Direction flag (1 for up, -1 for down)

# Returns
- `v`: Cylinder harmonic coefficients for this cylinder
"""
function vone(cloc, cmmax::Int, lambda_wave, kxex, up_down)
    CT = Complex{eltype(cloc)}
    v = zeros(CT, 2*cmmax + 1)
    offset = cmmax + 1  # Julia 1-indexed

    for cm in -cmmax:cmmax
        v[cm + offset] = vonem(cloc, cm, lambda_wave, kxex, up_down)
    end

    return v
end

"""
    vonem(cloc, cm, lambda_wave, kxex, up_down)

Calculate one cylinder harmonic coefficient.

# Arguments
- `cloc`: Cylinder location [x, y]
- `cm::Int`: Mode number
- `lambda_wave`: Wavelength
- `kxex`: x component of excitation wave vector
- `up_down`: Direction flag (1 for up, -1 for down)

# Returns
- `v`: Cylinder harmonic coefficient
"""
function vonem(cloc, cm::Int, lambda_wave, kxex, up_down)
    k = 2π / lambda_wave

    kyex = ky(k, kxex)
    # Extract scalar if array
    if isa(kyex, AbstractArray)
        kyex = kyex[1]
    end

    # CRITICAL FIX: For evanescent modes where |kxex| > k, arccos returns complex values
    phiinc = acos(Complex(kxex / k))

    if up_down < 0
        kyex = -kyex
        phiinc = -phiinc
    end

    # Calculate the coefficient
    k_vec = [kxex, kyex]
    v = (exp(-1im * dot(cloc, k_vec)) *
         exp(-1im * cm * phiinc) *
         exp(-1im * π/2 * cm))

    return v
end
