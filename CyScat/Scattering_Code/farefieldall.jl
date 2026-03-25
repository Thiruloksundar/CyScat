"""
Farfield Calculator
Coder : Curtis Jin
Date  : 2010/OCT/21st Thursday
Contact : jsirius@umich.edu
Description : Professor Michelson's version
            : Far field calculator

Julia translation
"""

"""
    farefieldall(clocs, cmmaxs, lambda_val, coefficients, nphi)

Calculate the far field for all cylinders.

# Arguments
- `clocs`: Cylinder locations (ncyl x 2)
- `cmmaxs`: Maximum mode numbers for each cylinder
- `lambda_val`: Wavelength
- `coefficients`: Scattering coefficients
- `nphi`: Number of angular points

# Returns
- `result`: Far field values with angles (nphi x 2), first column is angles, second is field
"""
function farefieldall(clocs, cmmaxs, lambda_val, coefficients, nphi)
    efieldlist = zeros(ComplexF64, nphi)
    nstart = 1  # Julia 1-indexed

    for icyl in 1:length(cmmaxs)
        cmmax = Int(cmmaxs[icyl])
        num_coeffs = 2 * cmmax + 1
        coeff_slice = coefficients[nstart:nstart + num_coeffs - 1]

        efieldlist .+= farefieldone(
            clocs[icyl, :], cmmax, lambda_val, coeff_slice, nphi
        )
        nstart = nstart + num_coeffs
    end

    # Create angle list and combine with field values
    philist = collect(0:nphi-1) .* 2π / nphi

    # Return as (nphi, 2) array with angles and field values
    result = hcat(philist, efieldlist)

    return result
end


"""
    farefieldone(cloc, cmmax, lambda_val, coefficient, nphi)

Calculate the far field for a single cylinder.
"""
function farefieldone(cloc, cmmax, lambda_val, coefficient, nphi)
    efieldlist = zeros(ComplexF64, nphi)

    for cm in -cmmax:cmmax
        idx = cm + cmmax + 1  # Julia 1-indexed
        efieldlist .+= farefieldonem(
            cloc, cm, lambda_val, coefficient[idx], nphi
        )
    end

    return efieldlist
end


"""
    farefieldonem(cloc, cm, lambda_val, coefficientm, nphi)

Calculate the far field for a single mode of a single cylinder.
"""
function farefieldonem(cloc, cm, lambda_val, coefficientm, nphi)
    dphi = 2π / nphi
    k = 2π / lambda_val

    efieldlist = zeros(ComplexF64, nphi)
    for iphi in 1:nphi
        phi = (iphi - 1) * dphi
        phase1 = exp(1im * k * (cloc[1] * cos(phi) + cloc[2] * sin(phi)))
        phase2 = exp(1im * cm * phi)
        amplitude = sqrt(2 / (π * k))
        phase3 = exp(1im * (cm * π / 2 + π / 4))
        efieldlist[iphi] = phase1 * phase2 * amplitude * phase3 * coefficientm
    end

    return efieldlist
end
