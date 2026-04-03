"""
Modified Epsilon Shanks Code
Coder: Curtis Jin
Date: 2010/NOV/21th Sunday
Contact: jsirius@umich.edu
Description: Implementation of recursive version of Shanks Transformation
             It's NOT using determinant method

Julia translation
Fixed: Added safe division checks to prevent inf/NaN propagation
"""

"""
    modified_epsilon_shanks(newentry, a1, a2)

Modified Epsilon Shanks transformation for accelerating convergence.

# Arguments
- `newentry`: New entry to add to the sequence
- `a1`: First auxiliary array
- `a2`: Second auxiliary array

# Returns
- `S`: Transformed sequence
"""
# ForwardDiff AD support: define isfinite/isnan/isinf for Complex{Dual}.
# Guarded so the module still loads without ForwardDiff installed (e.g. Pluto notebooks).
if Base.find_package("ForwardDiff") !== nothing
    @eval begin
        import ForwardDiff
        Base.isfinite(x::Complex{<:ForwardDiff.Dual}) =
            isfinite(ForwardDiff.value(real(x))) && isfinite(ForwardDiff.value(imag(x)))
        Base.isnan(x::Complex{<:ForwardDiff.Dual}) =
            isnan(ForwardDiff.value(real(x))) || isnan(ForwardDiff.value(imag(x)))
        Base.isinf(x::Complex{<:ForwardDiff.Dual}) =
            isinf(ForwardDiff.value(real(x))) || isinf(ForwardDiff.value(imag(x)))
    end
end
function modified_epsilon_shanks(newentry, a1, a2)

    kplus2 = length(a1)

    T = typeof(newentry)

    # Promote arrays to same type as newentry
    if eltype(a1) != T
        a1 = T.(a1)
    end

    if eltype(a2) != T
        a2 = T.(a2)
    end

    S = Vector{T}(undef, kplus2)

    S[1] = T(Inf)
    S[2] = newentry

    for idx in 2:(kplus2 - 1)
        d1 = a2[idx]   - a1[idx]
        d2 = S[idx]    - a1[idx]
        d3 = a2[idx-1] - a1[idx]

        # FIX 2: isfinite on Complex{Dual} uses the override defined at the top
        # of matrix_derivatives.jl (checks value(real) and value(imag)).
        # Safe-denominator pattern: replace near-zero denominators before
        # inverting so the Dual partial never sees a 1/0 derivative.
        eps_s = convert(T, 1e-30)
        safe_d1 = abs(d1) < 1e-30 ? eps_s : d1
        safe_d2 = abs(d2) < 1e-30 ? eps_s : d2
        safe_d3 = abs(d3) < 1e-30 ? eps_s : d3

        f1 = 1 / safe_d1
        f2 = 1 / safe_d2
        f3 = 1 / safe_d3

        if !isfinite(f1) || !isfinite(f2) || !isfinite(f3)
            S[idx+1] = a1[idx]
        else
            denom = f1 + f2 - f3
            safe_denom = abs(denom) < 1e-30 ? eps_s : denom
            candidate = 1 / safe_denom + a1[idx]
            S[idx+1] = isfinite(candidate) ? candidate : a1[idx]
        end
    end

    return S
end