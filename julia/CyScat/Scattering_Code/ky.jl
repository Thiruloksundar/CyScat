"""
ky Code
Coder: Curtis Jin
Date: 2011/FEB/13th Sunday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : y component of K extractor
             Modified it to be applicable to vectors

Julia translation
"""

"""
    ky(k, kx)

Calculate y component of wave vector.

For evanescent modes (kx > k), ensures imaginary part is negative
so that exp(-i*ky*z) decays for z > 0.

# Arguments
- `k`: Wave number (scalar)
- `kx`: x component of wave vector (scalar or array)

# Returns
- `ky_val`: y component of wave vector
"""
function ky(k, kx)
    # Ensure complex sqrt to handle evanescent modes (k^2 < kx^2)
    ky_val = sqrt.(Complex.(k^2 .- kx.^2))

    # MATLAB: index = find(imag(ky) > 0); ky(index) = conj(ky(index));
    # Handle both scalar and array cases
    if isa(ky_val, Number)
        # Scalar case
        if imag(ky_val) > 0
            ky_val = conj(ky_val)
        end
    else
        # Array case
        mask = imag.(ky_val) .> 0
        ky_val[mask] .= conj.(ky_val[mask])
    end

    return ky_val
end
