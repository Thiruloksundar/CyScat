"""
cascadertwo Code
Coder : Curtis Jin
Date  : 2011/FEB/16th Wednesday
Contact : jsirius@umich.edu
Description : Cascading two systems

Julia translation
"""

using LinearAlgebra

"""
    cascadertwo(S1, d1, S2, d2)

Cascade two S-matrix systems.

# Arguments
- `S1`: S-matrix of first system
- `d1`: Thickness of first system
- `S2`: S-matrix of second system
- `d2`: Thickness of second system

# Returns
- `Scas`: Cascaded S-matrix
- `dcas`: Total thickness
"""
function cascadertwo(S1, d1, S2, d2)
    dcas = d1 + d2

    # Get divider index (half the matrix size)
    divider = size(S1, 1) ÷ 2
    index1 = 1:divider
    index2 = (divider+1):(2*divider)

    I_mat = Matrix{eltype(S1)}(I, divider, divider)

    # Extract sub-matrices from S1
    R1 = S1[index1, index1]           # S11
    T1 = S1[index2, index1]           # S21
    T1tilde = S1[index1, index2]      # S12
    R1tilde = S1[index2, index2]      # S22

    # Extract sub-matrices from S2
    R2 = S2[index1, index1]           # S11
    T2 = S2[index2, index1]           # S21
    T2tilde = S2[index1, index2]      # S12
    R2tilde = S2[index2, index2]      # S22

    # Calculate intermediate matrices
    temp1 = inv(I_mat - R2 * R1tilde)
    temp2 = inv(I_mat - R1tilde * R2)

    # Calculate cascaded S-matrix components
    R = R1 + T1tilde * temp1 * R2 * T1
    T = T2 * temp2 * T1
    Ttilde = T1tilde * temp1 * T2tilde
    Rtilde = R2tilde + T2 * temp2 * R1tilde * T2tilde

    # Assemble cascaded S-matrix
    Scas = [R Ttilde; T Rtilde]

    return Scas, dcas
end
