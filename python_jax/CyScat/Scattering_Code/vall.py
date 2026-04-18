"""
vall Code
Coder: Curtis Jin
Date: 2010/OCT/13th Wednesday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : planewave to cylinder harmonics converter
Translated to Python
"""

import numpy as np
from .ky import ky


def vall(clocs, cmmaxs, lambda_wave, kxex, up_down):
    """
    Convert plane wave to cylinder harmonics for all cylinders
    
    Parameters:
    -----------
    clocs : ndarray
        Cylinder locations (Nx2 array)
    cmmaxs : ndarray
        Maximum mode numbers for each cylinder
    lambda_wave : float
        Wavelength
    kxex : float
        x component of excitation wave vector
    up_down : int
        Direction flag (1 for up, -1 for down)
        
    Returns:
    --------
    v : ndarray
        Cylinder harmonic coefficients
    """
    no_cylinders = len(cmmaxs)
    tot_no_modes = int(np.sum(cmmaxs * 2 + 1))
    v = np.zeros(tot_no_modes, dtype=complex)
    
    istart = 0
    for icyl in range(no_cylinders):
        cmmax = int(cmmaxs[icyl])
        v[istart:istart + 2*cmmax + 1] = vone(clocs[icyl, :], cmmax, lambda_wave, kxex, up_down)
        istart = istart + 2*cmmax + 1
    
    return v


def vone(cloc, cmmax, lambda_wave, kxex, up_down):
    """
    Convert plane wave to cylinder harmonics for one cylinder
    
    Parameters:
    -----------
    cloc : ndarray
        Cylinder location [x, y]
    cmmax : int
        Maximum mode number
    lambda_wave : float
        Wavelength
    kxex : float
        x component of excitation wave vector
    up_down : int
        Direction flag (1 for up, -1 for down)
        
    Returns:
    --------
    v : ndarray
        Cylinder harmonic coefficients for this cylinder
    """
    v = np.zeros(2*cmmax + 1, dtype=complex)
    offset = cmmax
    
    for cm in range(-cmmax, cmmax + 1):
        v[cm + offset] = vonem(cloc, cm, lambda_wave, kxex, up_down)
    
    return v


def vonem(cloc, cm, lambda_wave, kxex, up_down):
    """
    Calculate one cylinder harmonic coefficient
    
    Parameters:
    -----------
    cloc : ndarray
        Cylinder location [x, y]
    cm : int
        Mode number
    lambda_wave : float
        Wavelength
    kxex : float
        x component of excitation wave vector
    up_down : int
        Direction flag (1 for up, -1 for down)
        
    Returns:
    --------
    v : complex
        Cylinder harmonic coefficient
    """
    k = 2 * np.pi / lambda_wave

    kyex = ky(k, kxex)
    # CRITICAL FIX: For evanescent modes where |kxex| > k, arccos returns complex values
    # Must ensure the argument is complex to handle this properly
    phiinc = np.arccos((kxex / k + 0j))
    
    if up_down < 0:
        kyex = -kyex
        phiinc = -phiinc
    
    # Calculate the coefficient
    # Note: @ is matrix multiplication, but for dot product of vectors we can use np.dot
    k_vec = np.array([kxex, kyex])
    v = (np.exp(-1j * np.dot(cloc, k_vec)) * 
         np.exp(-1j * cm * phiinc) * 
         np.exp(-1j * np.pi/2 * cm))
    
    return v