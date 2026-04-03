"""
sall Code
Coder: Curtis Jin
Date: 2010/OCT/13th Wednesday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : scattering coefficients generating code

Julia translation
"""

using SpecialFunctions

"""
    sall(cmmaxs, cepmus, crads, lambda_wave)

Generate all scattering coefficients for all cylinders.

# Arguments
- `cmmaxs`: Maximum mode numbers for each cylinder
- `cepmus`: Epsilon and mu values for each cylinder (Nx2 array)
- `crads`: Radii of cylinders
- `lambda_wave`: Wavelength

# Returns
- `s`: Vector of scattering coefficients
"""
function sall(cmmaxs, cepmus, crads, lambda_wave)
    no_cylinders = length(cmmaxs)
    tot_no_modes = Int(sum(cmmaxs .* 2 .+ 1))
    #s = zeros(Complex{promote_type(typeof(real(lambda_wave)), eltype(cepmus))}, tot_no_modes)
    s = zeros(Complex{promote_type(typeof(real(lambda_wave)), eltype(cepmus), eltype(crads))}, tot_no_modes)

    istart = 1  # Julia uses 1-based indexing
    for icyl in 1:no_cylinders
        cmmax = Int(cmmaxs[icyl])
        s[istart:istart + 2*cmmax] = sone(cmmax, cepmus[icyl, :], crads[icyl], lambda_wave)
        istart = istart + 2*cmmax + 1
    end

    return s
end

"""
    sone(cmmax, cepmu, crad, lambda_wave)

Generate scattering coefficients for one cylinder.

# Arguments
- `cmmax::Int`: Maximum mode number
- `cepmu`: Epsilon and mu values [epsilon, mu]
- `crad`: Cylinder radius
- `lambda_wave`: Wavelength

# Returns
- `s`: Scattering coefficients for this cylinder
"""
function sone(cmmax::Int, cepmu, crad, lambda_wave)
    #s = zeros(Complex{promote_type(typeof(real(lambda_wave)), eltype(cepmu))}, 2*cmmax + 1)
    s = zeros(Complex{promote_type(typeof(real(lambda_wave)), eltype(cepmu), eltype(crad))}, 2*cmmax + 1)
    offset = cmmax + 1  # Julia 1-indexed

    for cm in -cmmax:cmmax
        s[cm + offset] = sonem(cm, cepmu, crad, lambda_wave)
    end

    return s
end

"""
    sonem(cm, cepmu, crad, lambda_wave)

Generate scattering coefficient for one mode.

# Arguments
- `cm::Int`: Mode number
- `cepmu`: Epsilon and mu values [epsilon, mu]
- `crad`: Cylinder radius
- `lambda_wave`: Wavelength

# Returns
- `s`: Scattering coefficient for this mode
"""
function sonem(cm::Int, cepmu, crad, lambda_wave)
    k = 2π / lambda_wave

    if cepmu[1] < 0
        # PEC case
        s = -besselj(cm, k*crad) / hankelh2(cm, k*crad)
    else
        # Dielectric case
        ki = k * sqrt(cepmu[1] * cepmu[2])

        a = sqrt(cepmu[1])
        b = sqrt(cepmu[2])

        # Derivative of Bessel J with k
        dbessel_jk = (besselj(cm-1, crad*k) - besselj(cm+1, crad*k)) / 2

        # Derivative of Bessel J with ki
        dbessel_jki = (besselj(cm-1, crad*ki) - besselj(cm+1, crad*ki)) / 2

        # Derivative of Hankel H2 with k
        dhankel_h2k = (hankelh2(cm-1, crad*k) - hankelh2(cm+1, crad*k)) / 2

        numerator = (-b * besselj(cm, crad*ki) * dbessel_jk +
                    a * besselj(cm, crad*k) * dbessel_jki)

        denominator = (b * besselj(cm, crad*ki) * dhankel_h2k -
                      a * dbessel_jki * hankelh2(cm, crad*k))

        s = numerator / denominator
    end

    return s
end
