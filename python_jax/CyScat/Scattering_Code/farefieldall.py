"""
Farfield Calculator
Coder : Curtis Jin
Date  : 2010/OCT/21st Thursday
Contact : jsirius@umich.edu
Description : Professor Michelson's version
            : Far field calculator

Python translation
"""

import numpy as np


def farefieldall(clocs, cmmaxs, lambda_val, coefficients, nphi):
    """
    Calculate the far field for all cylinders.

    Parameters:
    -----------
    clocs : ndarray
        Cylinder locations (ncyl x 2)
    cmmaxs : ndarray
        Maximum mode numbers for each cylinder
    lambda_val : float
        Wavelength
    coefficients : ndarray
        Scattering coefficients
    nphi : int
        Number of angular points

    Returns:
    --------
    efieldlist : ndarray
        Far field values with angles (nphi x 2), first column is angles, second is field
    """
    efieldlist = np.zeros(nphi, dtype=complex)
    nstart = 0  # 0-indexed in Python

    for icyl in range(len(cmmaxs)):
        cmmax = cmmaxs[icyl]
        num_coeffs = 2 * cmmax + 1
        coeff_slice = coefficients[nstart:nstart + num_coeffs]

        efieldlist = efieldlist + farefieldone(
            clocs[icyl, :], cmmax, lambda_val, coeff_slice, nphi
        )
        nstart = nstart + num_coeffs

    # Create angle list and combine with field values
    philist = np.arange(0, nphi) * 2 * np.pi / nphi

    # Return as (nphi, 2) array with angles and field values
    result = np.column_stack([philist, efieldlist])

    return result


def farefieldone(cloc, cmmax, lambda_val, coefficient, nphi):
    """
    Calculate the far field for a single cylinder.
    """
    efieldlist = np.zeros(nphi, dtype=complex)

    for cm in range(-cmmax, cmmax + 1):
        idx = cm + cmmax  # coefficient index
        efieldlist = efieldlist + farefieldonem(
            cloc, cm, lambda_val, coefficient[idx], nphi
        )

    return efieldlist


def farefieldonem(cloc, cm, lambda_val, coefficientm, nphi):
    """
    Calculate the far field for a single mode of a single cylinder.
    """
    dphi = 2 * np.pi / nphi
    k = 2 * np.pi / lambda_val

    efieldlist = np.zeros(nphi, dtype=complex)
    for iphi in range(nphi):
        phi = iphi * dphi
        # exp(1i*k*cloc*[cos(phi);sin(phi)]) * exp(1i*cm*phi) * sqrt(2/(pi*k)) * exp(1i*(cm*pi/2+pi/4)) * coefficientm
        phase1 = np.exp(1j * k * (cloc[0] * np.cos(phi) + cloc[1] * np.sin(phi)))
        phase2 = np.exp(1j * cm * phi)
        amplitude = np.sqrt(2 / (np.pi * k))
        phase3 = np.exp(1j * (cm * np.pi / 2 + np.pi / 4))
        efieldlist[iphi] = phase1 * phase2 * amplitude * phase3 * coefficientm

    return efieldlist
