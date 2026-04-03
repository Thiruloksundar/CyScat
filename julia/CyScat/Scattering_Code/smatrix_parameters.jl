"""
SMatrixParameters Code
Modified Date: 2011.02.09.Wednesday
Features: Generate a structure that contains parameters for constructing
          a scattering matrix.
Coder: Curtis Jin
Contact: jsirius@umich.edu

Julia translation
"""

include("ky.jl")

"""
    smatrix_parameters(lambda_wave, period, phiinc, epsseries, epsloc,
                       nrepeat_spatial, nrepeat_spectral, jmax,
                       kshanks_spatial, kshanks_spectral, spectral, spectral_cond)

Generate SMatrix parameters structure.

# Arguments
- `lambda_wave`: Wavelength
- `period`: Period of the structure
- `phiinc`: Incident angle
- `epsseries`: Epsilon for series calculation
- `epsloc`: Epsilon for local calculation
- `nrepeat_spatial`: Number of spatial repeats
- `nrepeat_spectral`: Number of spectral repeats
- `jmax`: Maximum j index
- `kshanks_spatial`: Shanks parameter for spatial
- `kshanks_spectral`: Shanks parameter for spectral
- `spectral`: Spectral flag
- `spectral_cond`: Spectral condition parameter

# Returns
- `sp`: Dictionary containing SMatrix parameters
"""
function smatrix_parameters(lambda_wave, period, phiinc, epsseries, epsloc,
                            nrepeat_spatial, nrepeat_spectral, jmax,
                            kshanks_spatial, kshanks_spectral, spectral, spectral_cond)

    k0 = 2π / lambda_wave

    # Generate kxs array
    if phiinc == π / 2
        kxs = 2π * collect(-jmax:jmax) / period
    else
        kxs = 2π * collect(-jmax:jmax) / period .+ k0 * cos(phiinc)
    end

    kxs = reshape(kxs, :, 1)  # Column vector

    # Calculate kys
    kys = ky(k0, kxs)

    middle_index = ceil(Int, length(kxs) / 2)

    two_over_period = 2 / period

    # MATLAB: Angles = asin(kxs/k0);
    # For evanescent modes where |kxs/k0| > 1, arcsin returns complex values
    arg = kxs ./ k0
    angles = asin.(Complex.(arg))  # Force complex to handle evanescent modes

    # MATLAB and Python use different branch cuts for arcsin:
    # For x > 1:  need to conjugate
    # For x < -1: do NOT conjugate
    positive_evanescent_mask = real.(arg) .> 1
    angles[positive_evanescent_mask] .= conj.(angles[positive_evanescent_mask])

    # Generate SMatrixParameter Structure
    sp = Dict{String, Any}(
        "lambda" => lambda_wave,
        "period" => period,
        "phiinc" => phiinc,
        "k0" => k0,
        "kxs" => kxs,
        "kys" => kys,
        "Angles" => angles,
        "TwoOverPeriod" => two_over_period,
        "MiddleIndex" => middle_index,
        "epsseries" => epsseries,
        "epsloc" => epsloc,
        "nrepeatSpatial" => nrepeat_spatial,
        "nrepeatSpectral" => nrepeat_spectral,
        "jmax" => jmax,
        "kshanksSpatial" => kshanks_spatial,
        "kshanksSpectral" => kshanks_spectral,
        "spectral" => spectral,
        "spectralCond" => spectral_cond
    )

    return sp
end
