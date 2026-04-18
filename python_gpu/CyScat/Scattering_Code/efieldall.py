"""
efield Calculator
Coder : Curtis Jin
Date  : 2010/OCT/21st Thursday
Contact : jsirius@umich.edu
Description : Professor Michelson's version
            : Near field calculator

Python translation
"""

import numpy as np
from scipy.special import hankel2


def efieldall(clocs, crads, cmmaxs, lambda_val, coefficients, far, point_list):
    """
    Calculate the electric field at specified points for all cylinders.

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
    far : int
        Far field flag (>0 for far field, <=0 for near field)
    point_list : ndarray
        Points at which to calculate field (npoints x 2) or angles for far field

    Returns:
    --------
    efieldlist : ndarray
        Electric field at each point
    """
    if far > 0:
        # Far field case - point_list contains angles
        efieldlist = np.zeros(len(point_list), dtype=complex)
    else:
        # Near field case - point_list contains (x, y) coordinates
        efieldlist = np.zeros(len(point_list), dtype=complex)

    ncyl = len(cmmaxs)
    nstart = 0  # 0-indexed in Python

    for icyl in range(ncyl):
        cmmax = cmmaxs[icyl]
        num_coeffs = 2 * cmmax + 1
        coeff_slice = coefficients[nstart:nstart + num_coeffs]

        efieldlist = efieldlist + efieldone(
            clocs[icyl, :], crads[icyl], cmmax, lambda_val,
            coeff_slice, far, point_list
        )
        nstart = nstart + num_coeffs

    return efieldlist


def efieldone(cloc, crad, cmmax, lambda_val, coefficient, far, point_list):
    """
    Calculate the electric field for a single cylinder.
    """
    if far > 0:
        efieldlist = np.zeros(len(point_list), dtype=complex)
    else:
        efieldlist = np.zeros(len(point_list), dtype=complex)

    for cm in range(-cmmax, cmmax + 1):
        idx = cm + cmmax  # coefficient index
        efieldlist = efieldlist + efieldonem(
            cloc, crad, cm, lambda_val, coefficient[idx], far, point_list
        )

    return efieldlist


def efieldonem(cloc, crad, cm, lambda_val, coefficientm, far, point_list):
    """
    Calculate the electric field for a single mode of a single cylinder.
    """
    k = 2 * np.pi / lambda_val

    if far > 0:
        # Far field calculation
        efieldlist = np.zeros(len(point_list), dtype=complex)
        for idx, phi in enumerate(point_list):
            # exp(1i*k*cloc*[cos(phi);sin(phi)]) * exp(1i*cm*phi) * sqrt(2/(pi*k)) * exp(1i*(cm*pi/2+pi/4)) * coefficientm
            phase1 = np.exp(1j * k * (cloc[0] * np.cos(phi) + cloc[1] * np.sin(phi)))
            phase2 = np.exp(1j * cm * phi)
            amplitude = np.sqrt(2 / (np.pi * k))
            phase3 = np.exp(1j * (cm * np.pi / 2 + np.pi / 4))
            efieldlist[idx] = phase1 * phase2 * amplitude * phase3 * coefficientm
    else:
        # Near field calculation
        efieldlist = np.zeros(len(point_list), dtype=complex)
        for idx in range(len(point_list)):
            v = point_list[idx, :] - cloc
            x, y = v[0], v[1]
            r = np.linalg.norm(v)

            if r < crad:
                efieldlist[idx] = 0
            else:
                # besselh(cm,2,k*r) * exp(1i*cm*atan2(y,x)) * coefficientm
                temp = hankel2(cm, k * r) * np.exp(1j * cm * np.arctan2(y, x)) * coefficientm
                efieldlist[idx] = temp

    return efieldlist
