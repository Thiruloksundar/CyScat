"""
efield Calculator
Coder : Curtis Jin
Date  : 2010/OCT/21st Thursday
Contact : jsirius@umich.edu
Description : Professor Michelson's version
            : Near field calculator

Julia translation
"""

using SpecialFunctions
using LinearAlgebra

"""
    efieldall(clocs, crads, cmmaxs, lambda_val, coefficients, far, point_list)

Calculate the electric field at specified points for all cylinders.

# Arguments
- `clocs`: Cylinder locations (ncyl x 2)
- `crads`: Cylinder radii
- `cmmaxs`: Maximum mode numbers for each cylinder
- `lambda_val`: Wavelength
- `coefficients`: Scattering coefficients
- `far`: Far field flag (>0 for far field, <=0 for near field)
- `point_list`: Points at which to calculate field (npoints x 2) or angles for far field

# Returns
- `efieldlist`: Electric field at each point
"""
function efieldall(clocs, crads, cmmaxs, lambda_val, coefficients, far, point_list)
    if far > 0
        # Far field case - point_list contains angles
        efieldlist = zeros(ComplexF64, length(point_list))
    else
        # Near field case - point_list contains (x, y) coordinates
        efieldlist = zeros(ComplexF64, size(point_list, 1))
    end

    ncyl = length(cmmaxs)
    nstart = 1  # Julia 1-indexed

    for icyl in 1:ncyl
        cmmax = Int(cmmaxs[icyl])
        num_coeffs = 2 * cmmax + 1
        coeff_slice = coefficients[nstart:nstart + num_coeffs - 1]

        efieldlist .+= efieldone(
            clocs[icyl, :], crads[icyl], cmmax, lambda_val,
            coeff_slice, far, point_list
        )
        nstart = nstart + num_coeffs
    end

    return efieldlist
end


"""
    efieldone(cloc, crad, cmmax, lambda_val, coefficient, far, point_list)

Calculate the electric field for a single cylinder.
"""
function efieldone(cloc, crad, cmmax, lambda_val, coefficient, far, point_list)
    if far > 0
        efieldlist = zeros(ComplexF64, length(point_list))
    else
        efieldlist = zeros(ComplexF64, size(point_list, 1))
    end

    for cm in -cmmax:cmmax
        idx = cm + cmmax + 1  # Julia 1-indexed
        efieldlist .+= efieldonem(
            cloc, crad, cm, lambda_val, coefficient[idx], far, point_list
        )
    end

    return efieldlist
end


"""
    efieldonem(cloc, crad, cm, lambda_val, coefficientm, far, point_list)

Calculate the electric field for a single mode of a single cylinder.
"""
function efieldonem(cloc, crad, cm, lambda_val, coefficientm, far, point_list)
    k = 2π / lambda_val

    if far > 0
        # Far field calculation
        efieldlist = zeros(ComplexF64, length(point_list))
        for (idx, phi) in enumerate(point_list)
            phase1 = exp(1im * k * (cloc[1] * cos(phi) + cloc[2] * sin(phi)))
            phase2 = exp(1im * cm * phi)
            amplitude = sqrt(2 / (π * k))
            phase3 = exp(1im * (cm * π / 2 + π / 4))
            efieldlist[idx] = phase1 * phase2 * amplitude * phase3 * coefficientm
        end
    else
        # Near field calculation
        efieldlist = zeros(ComplexF64, size(point_list, 1))
        for idx in 1:size(point_list, 1)
            v = point_list[idx, :] .- cloc
            x, y = v[1], v[2]
            r = norm(v)

            if r < crad
                efieldlist[idx] = 0.0 + 0.0im
            else
                temp = hankelh2(cm, k * r) * exp(1im * cm * atan(y, x)) * coefficientm
                efieldlist[idx] = temp
            end
        end
    end

    return efieldlist
end
