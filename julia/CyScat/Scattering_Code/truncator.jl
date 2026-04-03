"""
Truncator
Coder : Curtis Jin
Date  : 2011/MAY/18th WEDNESDAY
Contact : jsirius@umich.edu
Description : Code for truncating S-matrix to remove evanescent modes

Julia translation
"""

using LinearAlgebra

"""
    truncator(S, nmax, no_eva_mode)

Truncate S-matrix to remove evanescent modes and compute transmission metrics.

# Arguments
- `S`: Full S-matrix
- `nmax`: Maximum mode number
- `no_eva_mode`: Number of evanescent modes to remove from each side

# Returns
- `S_truncated`: Truncated S-matrix (propagating modes only)
- `e`: Eigenvalues (singular values squared) of transmission matrix
- `Gain`: Gain factor (optimal TC / normal TC)
- `OptWF`: Optimal wavefront (right singular vector)
- `DOU`: Degree of unitarity metric
"""
function truncator(S, nmax, no_eva_mode)
    # Extract S-matrix blocks (Julia 1-indexed)
    S11 = S[1:2*nmax+1, 1:2*nmax+1]
    S12 = S[1:2*nmax+1, 2*nmax+2:end]
    S21 = S[2*nmax+2:end, 1:2*nmax+1]
    S22 = S[2*nmax+2:end, 2*nmax+2:end]

    # Truncate to remove evanescent modes
    if no_eva_mode > 0
        trunc_range = (no_eva_mode+1):(2*nmax+1-no_eva_mode)
        S11_truncated = S11[trunc_range, trunc_range]
        S12_truncated = S12[trunc_range, trunc_range]
        S21_truncated = S21[trunc_range, trunc_range]
        S22_truncated = S22[trunc_range, trunc_range]
    else
        S11_truncated = S11
        S12_truncated = S12
        S21_truncated = S21
        S22_truncated = S22
    end

    S_truncated = [S11_truncated S12_truncated; S21_truncated S22_truncated]

    # Calculate degree of unitarity
    test = abs.(S_truncated' * S_truncated)
    DOU = sum(test) / size(test, 1)

    # Get transmission matrix and perform SVD
    divider = size(S_truncated, 1) ÷ 2
    index1 = 1:divider
    index2 = (divider+1):(divider*2)

    T = S_truncated[index2, index1]
    F = svd(T)
    tau = F.S
    V = F.V

    # Eigenvalues (singular values squared)
    e = tau.^2

    # Optimal wavefront
    OptWF = V[:, 1]

    # Calculate gain
    no_propagating_modes = nmax - no_eva_mode
    TC_normal = T[:, no_propagating_modes]
    TC_normal = real(TC_normal' * TC_normal)
    TC_opt = maximum(e)
    Gain = TC_normal != 0 ? TC_opt / TC_normal : Inf

    return S_truncated, e, Gain, OptWF, DOU
end
