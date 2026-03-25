"""
efieldallperiodic Calculator
Coder : Curtis Jin
Date  : 2011/JAN/16th Sunday
Contact : jsirius@umich.edu
Description : Field calculator for Periodic case

Julia translation
"""

using LinearAlgebra

include("ky.jl")
include("modified_epsilon_shanks.jl")

"""
    efieldallperiodic(clocs, crads, cmmaxs, lambda_val, coefficients, point_list, period, phiinc)

Calculate the electric field for periodic structures.

# Arguments
- `clocs`: Cylinder locations (ncyl x 2)
- `crads`: Cylinder radii
- `cmmaxs`: Maximum mode numbers for each cylinder
- `lambda_val`: Wavelength
- `coefficients`: Scattering coefficients
- `point_list`: Points at which to calculate field (npoints x 2)
- `period`: Periodic spacing
- `phiinc`: Incident angle

# Returns
- `efieldlist`: Electric field at each point
"""
function efieldallperiodic(clocs, crads, cmmaxs, lambda_val, coefficients, point_list, period, phiinc)
    efieldlist = zeros(ComplexF64, size(point_list, 1))
    ncyl = length(cmmaxs)
    nstart = 1  # Julia 1-indexed

    for icyl in 1:ncyl
        cmmax = Int(cmmaxs[icyl])
        num_coeffs = 2 * cmmax + 1
        coeff_slice = coefficients[nstart:nstart + num_coeffs - 1]

        efieldlist .+= efieldone_periodic(
            clocs[icyl, :], crads[icyl], cmmax, lambda_val,
            coeff_slice, point_list, period, phiinc
        )
        nstart = nstart + num_coeffs
    end

    return efieldlist
end


"""
    efieldone_periodic(cloc, crad, cmmax, lambda_val, coefficient, point_list, period, phiinc)

Calculate the electric field for a single cylinder in periodic structure.
"""
function efieldone_periodic(cloc, crad, cmmax, lambda_val, coefficient, point_list, period, phiinc)
    efieldlist = zeros(ComplexF64, size(point_list, 1))

    for cm in -cmmax:cmmax
        idx = cm + cmmax + 1  # Julia 1-indexed
        efieldlist .+= efieldonem_periodic(
            cloc, crad, cm, lambda_val, coefficient[idx], point_list, period, phiinc
        )
    end

    return efieldlist
end


"""
    efieldonem_periodic(cloc, crad, cm, lambda_val, coefficientm, point_list, period, phiinc)

Calculate the electric field for a single mode in periodic structure.
Uses spectral method with Shanks transformation for convergence.
"""
function efieldonem_periodic(cloc, crad, cm, lambda_val, coefficientm, point_list, period, phiinc)
    k = 2π / lambda_val
    efieldlist = ComplexF64[]

    # Convergence parameters
    kshanks = 5
    epsseries = 1e-9
    nrepeat = 10
    jmax = 2000

    for idx in 1:size(point_list, 1)
        v = point_list[idx, :] .- cloc
        x, y = v[1], v[2]
        r = norm(v)

        # Initialize Shanks transformation arrays
        ashankseps1 = zeros(ComplexF64, kshanks + 2)
        ashankseps2 = zeros(ComplexF64, kshanks + 2)
        ashankseps1[1] = Inf
        ashankseps2[1] = Inf

        if r < crad
            push!(efieldlist, 0.0 + 0.0im)
        else
            # Initialization
            kx0 = k * cos(phiinc)
            kx = kx0
            ky_val = ky(k, kx)
            if isa(ky_val, AbstractArray)
                ky_val = ky_val[1]
            end
            sign_y = y != 0 ? sign(y) : 1

            # Complex arcsin for evanescent modes
            angle = asin(Complex(kx / k))

            t = (Float64(sign_y)^cm *
                 exp(-1im * kx * x - 1im * ky_val * abs(y) -
                     1im * cm * (sign_y * angle - π)) /
                 ky_val * 2 / period)

            ashankseps1[2] = t
            ts = 1.0 + 0.0im

            # Looping
            j = 1
            irepeat = 0
            while irepeat < nrepeat && j < jmax
                # Spectral case
                kxp = j * 2π / period + kx0
                kxm = -j * 2π / period + kx0

                ky_p = ky(k, kxp)
                ky_m = ky(k, kxm)
                if isa(ky_p, AbstractArray)
                    ky_p = ky_p[1]
                end
                if isa(ky_m, AbstractArray)
                    ky_m = ky_m[1]
                end

                # Complex arcsin for evanescent modes
                angle_p = asin(Complex(kxp / k))
                angle_m = asin(Complex(kxm / k))

                add = (Float64(sign_y)^cm * 2 / period *
                       (exp(-1im * kxp * x - 1im * ky_p * abs(y) -
                            1im * cm * (sign_y * angle_p - π)) / ky_p +
                        exp(-1im * kxm * x - 1im * ky_m * abs(y) -
                            1im * cm * (sign_y * angle_m - π)) / ky_m))

                # Shanks Transformation & Convergence Checker
                told = t
                t = t + add

                if kshanks <= 0
                    rerror = abs(t - told)
                else
                    S = modified_epsilon_shanks(t, ashankseps1, ashankseps2)
                    ashankseps2 = copy(ashankseps1)
                    ashankseps1 = S

                    if j <= 2 * kshanks + 1
                        rerror = 1.0
                    else
                        tsold = ts
                        ts = S[end]
                        rerror = abs(ts - tsold)
                    end
                end

                if rerror < epsseries
                    irepeat = irepeat + 1
                else
                    irepeat = 0
                end

                j = j + 1
            end

            if j == jmax
                println("j has reached jmax!!")
            end

            temp = ts * coefficientm
            push!(efieldlist, temp)
        end
    end

    return efieldlist
end
