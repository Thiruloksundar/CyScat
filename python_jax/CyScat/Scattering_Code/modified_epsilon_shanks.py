"""
Modified Epsilon Shanks Code
Coder: Curtis Jin
Date: 2010/NOV/21th Sunday
Contact: jsirius@umich.edu
Description: Implementation of recursive version of Shanks Transformation
             It's NOT using determinant method
JAX version for automatic differentiation.
"""

import jax.numpy as jnp


def modified_epsilon_shanks(newentry, a1, a2):
    """
    Modified Epsilon Shanks transformation for accelerating convergence.

    Parameters
    ----------
    newentry : complex - New entry to add to the sequence
    a1 : array - First auxiliary array
    a2 : array - Second auxiliary array

    Returns
    -------
    S : array - Transformed sequence
    """
    kplus2 = len(a1)
    S = jnp.zeros(kplus2, dtype=jnp.complex128)
    S = S.at[0].set(jnp.inf + 0j)
    S = S.at[1].set(newentry)

    for idx in range(1, kplus2 - 1):
        first_factor = 1.0 / (a2[idx] - a1[idx])
        second_factor = 1.0 / (S[idx] - a1[idx])
        third_factor = 1.0 / (a2[idx - 1] - a1[idx])

        all_finite = (jnp.isfinite(first_factor) &
                      jnp.isfinite(second_factor) &
                      jnp.isfinite(third_factor))

        candidate = 1.0 / (first_factor + second_factor - third_factor) + a1[idx]
        val = jnp.where(all_finite, candidate, a1[idx])
        S = S.at[idx + 1].set(val)

    return S
