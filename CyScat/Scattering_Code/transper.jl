"""
transper - Translation matrix for periodic structures
Coder : Curtis Jin
Contact : jsirius@umich.edu
Description : Translation matrix calculation for periodic case
              using spectral or spatial methods with Shanks transformation

Julia translation
"""

using SpecialFunctions
using LinearAlgebra

include("ky.jl")
include("modified_epsilon_shanks.jl")

"""
    transper(clocob, cmob, clocso, cmso, period, lambda_val, phiinc,
             epsloc, kshanks, epsseries, nrepeat, spectral, spec_cond)

Calculate translation coefficient for periodic structures.

# Arguments
- `clocob`: Observer cylinder location (1x2)
- `cmob`: Observer mode number
- `clocso`: Source cylinder location (1x2)
- `cmso`: Source mode number
- `period`: Periodic spacing (negative for non-periodic)
- `lambda_val`: Wavelength
- `phiinc`: Incident angle
- `epsloc`: Spatial tolerance for close cylinders
- `kshanks`: Shanks transformation order
- `epsseries`: Convergence tolerance
- `nrepeat`: Number of times convergence must be met
- `spectral`: Use spectral method if positive
- `spec_cond`: Y distance threshold for switching to spatial method

# Returns
- `t`: Translation coefficient
"""
function transper(clocob, cmob, clocso, cmso, period, lambda_val, phiinc,
                  epsloc, kshanks, epsseries, nrepeat, spectral, spec_cond)
    k = 2π / lambda_val - 1im * 1e-14
    jmax = 3000
    spectralinternal = spectral

    rv = clocob .- clocso
    r = norm(rv)
    x, y = rv[1], rv[2]

    if period < 0
        # Non-periodic case
        if r < epsloc
            t = 0.0 + 0.0im
        else
            t = trans(clocob, cmob, clocso, cmso, lambda_val)
        end
        return t
    end

    # Periodic case
    # Spectral condition checker
    if abs(y) < spec_cond
        spectralinternal = -1
        println("Calculating Spatial Sum!")
    end

    # Shanks Transformation Initialization
    ashankseps1 = zeros(ComplexF64, kshanks + 2)
    ashankseps2 = zeros(ComplexF64, kshanks + 2)
    if kshanks > 0
        ashankseps1[1] = Inf
        ashankseps2[1] = Inf
    end

    ts = 1.0 + 0.0im

    if spectralinternal < 0
        # Spatial case
        if r < epsloc
            t = 0.0 + 0.0im
        else
            t = trans(clocob, cmob, clocso, cmso, lambda_val)
        end
        if kshanks > 0
            ashankseps1[2] = t
        end
    else
        # Spectral case
        cm = -(cmob - cmso)
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

        if kshanks > 0
            ashankseps1[2] = t
        end
    end

    # Looping
    j = 1
    irepeat = 0
    while irepeat < nrepeat && j < jmax
        # Sequence calculation
        if spectralinternal < 0
            # Non-spectral (spatial) method
            add = (exp(-1im * k * j * period * cos(phiinc)) *
                   trans(clocob, cmob, clocso .+ j .* [period, 0], cmso, lambda_val) +
                   exp(-1im * k * (-j) * period * cos(phiinc)) *
                   trans(clocob, cmob, clocso .- j .* [period, 0], cmso, lambda_val))
        else
            # Spectral method
            cm = -(cmob - cmso)
            kx0 = k * cos(phiinc)
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

            sign_y = y != 0 ? sign(y) : 1

            # Complex arcsin for evanescent modes
            angle_p = asin(Complex(kxp / k))
            angle_m = asin(Complex(kxm / k))

            add = (Float64(sign_y)^cm * 2 / period *
                   (exp(-1im * kxp * x - 1im * ky_p * abs(y) -
                        1im * cm * (sign_y * angle_p - π)) / ky_p +
                    exp(-1im * kxm * x - 1im * ky_m * abs(y) -
                        1im * cm * (sign_y * angle_m - π)) / ky_m))
        end

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
        println("j has reached jmax!! j = $j")
    end

    if kshanks > 0 && period > 0
        t = ts
    end

    return t
end


"""
    trans(clocob, cmob, clocso, cmso, lambda_val)

Calculate single translation coefficient (non-periodic).
"""
function trans(clocob, cmob, clocso, cmso, lambda_val)
    k = 2π / lambda_val
    rv = clocob .- clocso
    x, y = rv[1], rv[2]
    r = norm(rv)
    phip = atan(y, x)

    t = hankelh2(cmob - cmso, k * r) * exp(-1im * (cmob - cmso) * (phip - π))

    return t
end
