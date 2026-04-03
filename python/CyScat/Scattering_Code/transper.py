"""
transper - Translation matrix for periodic structures
Coder : Curtis Jin
Contact : jsirius@umich.edu
Description : Translation matrix calculation for periodic case
              using spectral or spatial methods with Shanks transformation

Python translation
"""

import numpy as np
from scipy.special import hankel2
from .ky import ky
from .modified_epsilon_shanks import modified_epsilon_shanks


def transper(clocob, cmob, clocso, cmso, period, lambda_val, phiinc,
             epsloc, kshanks, epsseries, nrepeat, spectral, spec_cond):
    """
    Calculate translation coefficient for periodic structures.

    Parameters:
    -----------
    clocob : ndarray
        Observer cylinder location (1x2)
    cmob : int
        Observer mode number
    clocso : ndarray
        Source cylinder location (1x2)
    cmso : int
        Source mode number
    period : float
        Periodic spacing (negative for non-periodic)
    lambda_val : float
        Wavelength
    phiinc : float
        Incident angle
    epsloc : float
        Spatial tolerance for close cylinders
    kshanks : int
        Shanks transformation order
    epsseries : float
        Convergence tolerance
    nrepeat : int
        Number of times convergence must be met
    spectral : int
        Use spectral method if positive
    spec_cond : float
        Y distance threshold for switching to spatial method

    Returns:
    --------
    t : complex
        Translation coefficient
    """
    k = 2 * np.pi / lambda_val - 1j * 1e-14
    jmax = 3000
    spectralinternal = spectral

    rv = clocob - clocso
    r = np.linalg.norm(rv)
    x, y = rv[0], rv[1]

    if period < 0:
        # Non-periodic case
        if r < epsloc:
            t = 0
        else:
            t = trans(clocob, cmob, clocso, cmso, lambda_val)
        return t

    # Periodic case
    # Spectral condition checker
    if np.abs(y) < spec_cond:
        spectralinternal = -1
        print('Calculating Spatial Sum!')

    # Shanks Transformation Initialization
    if kshanks > 0:
        ashankseps1 = np.zeros(kshanks + 2, dtype=complex)
        ashankseps2 = np.zeros(kshanks + 2, dtype=complex)
        ashankseps1[0] = np.inf
        ashankseps2[0] = np.inf

    if spectralinternal < 0:
        # Spatial case
        if r < epsloc:
            t = 0
        else:
            t = trans(clocob, cmob, clocso, cmso, lambda_val)
        ashankseps1[1] = t
        ts = 1
    else:
        # Spectral case
        cm = -(cmob - cmso)
        kx0 = k * np.cos(phiinc)
        kx = kx0

        ky_val = ky(k, kx)
        sign_y = np.sign(y) if y != 0 else 1

        # Complex arcsin for evanescent modes
        angle = np.arcsin(kx / k + 0j)

        t = (sign_y ** cm *
             np.exp(-1j * kx * x - 1j * ky_val * np.abs(y) -
                    1j * cm * (sign_y * angle - np.pi)) /
             ky_val * 2 / period)

        ashankseps1[1] = t
        ts = 1

    # Looping
    j = 1
    irepeat = 0
    while irepeat < nrepeat and j < jmax:
        # Sequence calculation
        if spectralinternal < 0:
            # Non-spectral (spatial) method
            add = (np.exp(-1j * k * j * period * np.cos(phiinc)) *
                   trans(clocob, cmob, clocso + j * np.array([period, 0]), cmso, lambda_val) +
                   np.exp(-1j * k * (-j) * period * np.cos(phiinc)) *
                   trans(clocob, cmob, clocso - j * np.array([period, 0]), cmso, lambda_val))
        else:
            # Spectral method
            kxp = j * 2 * np.pi / period + kx0
            kxm = -j * 2 * np.pi / period + kx0

            ky_p = ky(k, kxp)
            ky_m = ky(k, kxm)

            # Complex arcsin for evanescent modes
            angle_p = np.arcsin(kxp / k + 0j)
            angle_m = np.arcsin(kxm / k + 0j)

            add = (sign_y ** cm * 2 / period *
                   (np.exp(-1j * kxp * x - 1j * ky_p * np.abs(y) -
                           1j * cm * (sign_y * angle_p - np.pi)) / ky_p +
                    np.exp(-1j * kxm * x - 1j * ky_m * np.abs(y) -
                           1j * cm * (sign_y * angle_m - np.pi)) / ky_m))

        # Shanks Transformation & Convergence Checker
        told = t
        t = t + add

        if kshanks <= 0:
            rerror = np.abs(t - told)
        else:
            S = modified_epsilon_shanks(t, ashankseps1, ashankseps2)
            ashankseps2 = ashankseps1.copy()
            ashankseps1 = S

            if j <= 2 * kshanks + 1:
                rerror = 1
            else:
                tsold = ts
                ts = S[-1]
                rerror = np.abs(ts - tsold)

        if rerror < epsseries:
            irepeat = irepeat + 1
        else:
            irepeat = 0

        # Increment
        j = j + 1

    if j == jmax:
        print(f'j has reached jmax!! j = {j}')

    if kshanks > 0 and period > 0:
        t = ts

    return t


def trans(clocob, cmob, clocso, cmso, lambda_val):
    """
    Calculate single translation coefficient (non-periodic).
    """
    k = 2 * np.pi / lambda_val
    rv = clocob - clocso
    x, y = rv[0], rv[1]
    r = np.linalg.norm(rv)
    phip = np.arctan2(y, x)

    t = hankel2(cmob - cmso, k * r) * np.exp(-1j * (cmob - cmso) * (phip - np.pi))

    return t
