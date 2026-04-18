"""
ky Code
Coder: Curtis Jin
Date: 2011/FEB/13th Sunday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : y component of K extractor
             Modified it to be applicable to vectors
JAX version for automatic differentiation.

FIX: The original jnp.where(imag > 0, conj(ky), ky) is broken under autodiff.
     JAX evaluates gradients through BOTH branches before masking, so the
     conj() sign-flip on the imaginary gradient leaks through even for entries
     where the condition is False.

     The correct approach: enforce imag(ky) <= 0 analytically by multiplying
     by the sign of the imaginary part directly — no where() needed.
     sign_fix = 1 - 2*(imag(ky_raw) > 0)  gives +1 where imag<=0, -1 where imag>0
     Multiplying ky_raw by sign_fix conjugates only the entries that need it,
     with a gradient that is correct everywhere.
"""

import jax.numpy as jnp


def ky(k, kx):
    """
    Calculate y component of wave vector.

    For propagating modes (|kx| < k):  ky is real and positive.
    For evanescent modes  (|kx| > k):  ky is purely imaginary with imag < 0
                                        (decaying in +y direction).

    Parameters
    ----------
    k  : float or array - Wave number
    kx : float or complex array - x component of wave vector

    Returns
    -------
    ky_val : complex array - y component of wave vector
    """
    # Always compute on the complex branch so evanescent modes give purely
    # imaginary values rather than NaN (works whether kx is real or complex).
    ky_raw = jnp.sqrt(k**2 - kx**2 + 0j)

    # Enforce imag(ky) <= 0 for evanescent decay without jnp.where:
    #   sign_fix = +1 when imag(ky_raw) <= 0  (already correct)
    #   sign_fix = -1 when imag(ky_raw) >  0  (need to conjugate)
    # Multiplying by sign_fix is equivalent to conjugating bad entries but
    # has a well-defined, non-NaN gradient everywhere.
    sign_fix = 1 - 2 * (jnp.imag(ky_raw) > 0).astype(jnp.float64)
    ky_val = jnp.real(ky_raw) + 1j * (jnp.imag(ky_raw) * sign_fix)

    return ky_val