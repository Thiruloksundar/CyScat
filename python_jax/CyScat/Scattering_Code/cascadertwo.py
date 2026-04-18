"""
cascadertwo Code
Coder : Curtis Jin
Date  : 2011/FEB/16th Wednesday
Contact : jsirius@umich.edu
Description : Cascading two systems

JAX version for automatic differentiation.
"""

import jax.numpy as jnp


def cascadertwo(S1, d1, S2, d2):
    """
    Cascade two S-matrix systems (JAX-differentiable).

    Parameters
    ----------
    S1 : jax array - S-matrix of first system
    d1 : float - Thickness of first system
    S2 : jax array - S-matrix of second system
    d2 : float - Thickness of second system

    Returns
    -------
    Scas : jax array - Cascaded S-matrix
    dcas : float - Total thickness
    """
    S1 = jnp.asarray(S1, dtype=jnp.complex128)
    S2 = jnp.asarray(S2, dtype=jnp.complex128)
    dcas = d1 + d2
    divider = len(S1) // 2

    I = jnp.eye(divider, dtype=jnp.complex128)

    R1 = S1[:divider, :divider]
    T1 = S1[divider:, :divider]
    T1tilde = S1[:divider, divider:]
    R1tilde = S1[divider:, divider:]

    R2 = S2[:divider, :divider]
    T2 = S2[divider:, :divider]
    T2tilde = S2[:divider, divider:]
    R2tilde = S2[divider:, divider:]

    temp1 = jnp.linalg.inv(I - R2 @ R1tilde)
    temp2 = jnp.linalg.inv(I - R1tilde @ R2)

    R = R1 + T1tilde @ temp1 @ R2 @ T1
    T = T2 @ temp2 @ T1
    Ttilde = T1tilde @ temp1 @ T2tilde
    Rtilde = R2tilde + T2 @ temp2 @ R1tilde @ T2tilde

    Scas = jnp.block([[R, Ttilde], [T, Rtilde]])
    return Scas, dcas
