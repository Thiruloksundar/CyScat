"""
efieldallperiodic Calculator
Coder : Curtis Jin
Date  : 2011/JAN/16th Sunday
Contact : jsirius@umich.edu
Description : Field calculator for Periodic case

Python translation
"""

import numpy as np
from .ky import ky
from .modified_epsilon_shanks import modified_epsilon_shanks


def efieldallperiodic(clocs, crads, cmmaxs, lambda_val, coefficients, point_list, period, phiinc):
    """
    Calculate the electric field for periodic structures.

    Parameters:
    -----------
    clocs : ndarray
        Cylinder locations (ncyl x 2)
    crads : ndarray
        Cylinder radii
    cmmaxs : ndarray
        Maximum mode numbers for each cylinder
    lambda_val : float
        Wavelength
    coefficients : ndarray
        Scattering coefficients
    point_list : ndarray
        Points at which to calculate field (npoints x 2)
    period : float
        Periodic spacing
    phiinc : float
        Incident angle

    Returns:
    --------
    efieldlist : ndarray
        Electric field at each point
    """
    efieldlist = np.zeros(len(point_list), dtype=complex)
    ncyl = len(cmmaxs)
    nstart = 0  # 0-indexed in Python

    for icyl in range(ncyl):
        cmmax = cmmaxs[icyl]
        num_coeffs = 2 * cmmax + 1
        coeff_slice = coefficients[nstart:nstart + num_coeffs]

        efieldlist = efieldlist + efieldone(
            clocs[icyl, :], crads[icyl], cmmax, lambda_val,
            coeff_slice, point_list, period, phiinc
        )
        nstart = nstart + num_coeffs

    return efieldlist


def efieldone(cloc, crad, cmmax, lambda_val, coefficient, point_list, period, phiinc):
    """
    Calculate the electric field for a single cylinder in periodic structure.
    """
    efieldlist = np.zeros(len(point_list), dtype=complex)

    for cm in range(-cmmax, cmmax + 1):
        idx = cm + cmmax  # coefficient index
        efieldlist = efieldlist + efieldonem(
            cloc, crad, cm, lambda_val, coefficient[idx], point_list, period, phiinc
        )

    return efieldlist


def efieldonem(cloc, crad, cm, lambda_val, coefficientm, point_list, period, phiinc):
    """
    Calculate the electric field for a single mode in periodic structure.
    Uses spectral method with Shanks transformation for convergence.
    """
    k = 2 * np.pi / lambda_val
    efieldlist = []

    # Convergence parameters
    kshanks = 5
    epsseries = 1e-9
    nrepeat = 10
    jmax = 2000

    for idx in range(len(point_list)):
        v = point_list[idx, :] - cloc
        x, y = v[0], v[1]
        r = np.linalg.norm(v)

        # Initialize Shanks transformation arrays
        ashankseps1 = np.zeros(kshanks + 2)
        ashankseps2 = np.zeros(kshanks + 2)
        ashankseps1[0] = np.inf
        ashankseps2[0] = np.inf

        if r < crad:
            efieldlist.append(0)
        else:
            # Initialization
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
                # Spectral case
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
                print('j has reached jmax!!')

            temp = ts * coefficientm
            efieldlist.append(temp)

    return np.array(efieldlist)
