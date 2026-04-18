"""
coefficients Code
Coder : Curtis Jin
Date  : 2010/OCT/13th Wednesday
Contact : jsirius@umich.edu
Description : Professor Michelson's version
            : coefficients generating code for all out scattering fields

Python translation
"""

import numpy as np
from .transall import transall
from .sall import sall
from .vall import vall


def coefficients(clocs, cmmaxs, cepmus, crads, period, lambda_val, phiinc, up_down, sp):
    """
    Generate scattering coefficients for all cylinders.

    Parameters:
    -----------
    clocs : ndarray
        Cylinder locations (ncyl x 2)
    cmmaxs : ndarray
        Maximum mode numbers for each cylinder
    cepmus : ndarray
        Relative permittivity/permeability for each cylinder
    crads : ndarray
        Cylinder radii
    period : float
        Periodic spacing
    lambda_val : float
        Wavelength
    phiinc : float
        Incident angle
    up_down : int
        Direction indicator (1 for up, -1 for down)
    sp : dict
        S-matrix parameters

    Returns:
    --------
    coeff : ndarray
        Scattering coefficients
    """
    # Calculate total steps for progress bar (used by transall)
    no_cylinders = len(cmmaxs)
    total_steps = no_cylinders * (no_cylinders + 1) // 2

    t = transall(clocs, cmmaxs, period, lambda_val, phiinc, sp, total_steps)
    s = sall(cmmaxs, cepmus, crads, lambda_val)

    # z = eye(length(s)) - diag(s)*t
    z = np.eye(len(s)) - np.diag(s) @ t

    k = 2 * np.pi / lambda_val
    kxex = k * np.cos(phiinc)

    # v = diag(s) * vall(...)
    v = np.diag(s) @ vall(clocs, cmmaxs, lambda_val, kxex, up_down)

    # coeff = z\v (solve linear system)
    coeff = np.linalg.solve(z, v)

    return coeff
