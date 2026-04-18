"""
sall Code
Coder: Curtis Jin
Date: 2010/OCT/13th Wednesday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : scattering coefficients generating code
JAX version for automatic differentiation.
"""

import jax.numpy as jnp
from .bessel_jax import bessel_jv, hankel2


def sall(cmmaxs, cepmus, crads, lambda_wave):
    """
    Generate all scattering coefficients for all cylinders.

    Parameters
    ----------
    cmmaxs : array - Maximum mode numbers for each cylinder
    cepmus : array - Epsilon and mu values (Nx2)
    crads : array - Radii of cylinders
    lambda_wave : float - Wavelength

    Returns
    -------
    s : jax array - Vector of scattering coefficients
    """
    cmmaxs = jnp.asarray(cmmaxs)
    cepmus = jnp.asarray(cepmus)
    crads = jnp.asarray(crads)

    no_cylinders = len(cmmaxs)
    tot_no_modes = int(jnp.sum(cmmaxs * 2 + 1))
    s = jnp.zeros(tot_no_modes, dtype=jnp.complex128)

    istart = 0
    for icyl in range(no_cylinders):
        cmmax = int(cmmaxs[icyl])
        s_one = sone(cmmax, cepmus[icyl, :], crads[icyl], lambda_wave)
        s = s.at[istart:istart + 2 * cmmax + 1].set(s_one)
        istart = istart + 2 * cmmax + 1

    return s


def sone(cmmax, cepmu, crad, lambda_wave):
    """
    Generate scattering coefficients for one cylinder.

    Parameters
    ----------
    cmmax : int - Maximum mode number
    cepmu : array - [epsilon, mu]
    crad : float - Cylinder radius
    lambda_wave : float - Wavelength

    Returns
    -------
    s : jax array - Scattering coefficients
    """
    cms = jnp.arange(-cmmax, cmmax + 1)
    k = 2 * jnp.pi / lambda_wave

    is_pec = cepmu[0] < 0

    def _pec_coeff(cm):
        return -bessel_jv(cm, k * crad) / hankel2(cm, k * crad)

    def _dielectric_coeff(cm):
        ki = k * jnp.sqrt(cepmu[0] * cepmu[1])
        a = jnp.sqrt(cepmu[0])
        b = jnp.sqrt(cepmu[1])

        dbessel_jk = (bessel_jv(cm - 1, crad * k) - bessel_jv(cm + 1, crad * k)) / 2
        dbessel_jki = (bessel_jv(cm - 1, crad * ki) - bessel_jv(cm + 1, crad * ki)) / 2
        dhankel_h2k = (hankel2(cm - 1, crad * k) - hankel2(cm + 1, crad * k)) / 2

        numerator = (-b * bessel_jv(cm, crad * ki) * dbessel_jk +
                     a * bessel_jv(cm, crad * k) * dbessel_jki)
        denominator = (b * bessel_jv(cm, crad * ki) * dhankel_h2k -
                       a * dbessel_jki * hankel2(cm, crad * k))

        return numerator / denominator

    # Compute for each mode
    s_list = []
    for i in range(2 * cmmax + 1):
        cm = int(cms[i])
        if is_pec:
            s_list.append(_pec_coeff(cm))
        else:
            s_list.append(_dielectric_coeff(cm))

    return jnp.array(s_list)
