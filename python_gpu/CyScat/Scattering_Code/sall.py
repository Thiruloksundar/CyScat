"""
sall Code
Coder: Curtis Jin
Date: 2010/OCT/13th Wednesday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : scattering coefficients generating code
Translated to Python
"""

import numpy as np
from scipy.special import jv, hankel2


def sall(cmmaxs, cepmus, crads, lambda_wave):
    """
    Generate all scattering coefficients for all cylinders
    
    Parameters:
    -----------
    cmmaxs : ndarray
        Maximum mode numbers for each cylinder
    cepmus : ndarray
        Epsilon and mu values for each cylinder (Nx2 array)
    crads : ndarray
        Radii of cylinders
    lambda_wave : float
        Wavelength
        
    Returns:
    --------
    s : ndarray
        Vector of scattering coefficients
    """
    no_cylinders = len(cmmaxs)
    tot_no_modes = int(np.sum(cmmaxs * 2 + 1))
    s = np.zeros(tot_no_modes, dtype=complex)
    
    istart = 0
    for icyl in range(no_cylinders):
        cmmax = int(cmmaxs[icyl])
        s[istart:istart + 2*cmmax + 1] = sone(cmmax, cepmus[icyl, :], crads[icyl], lambda_wave)
        istart = istart + 2*cmmax + 1
    
    return s


def sone(cmmax, cepmu, crad, lambda_wave):
    """
    Generate scattering coefficients for one cylinder
    
    Parameters:
    -----------
    cmmax : int
        Maximum mode number
    cepmu : ndarray
        Epsilon and mu values [epsilon, mu]
    crad : float
        Cylinder radius
    lambda_wave : float
        Wavelength
        
    Returns:
    --------
    s : ndarray
        Scattering coefficients for this cylinder
    """
    s = np.zeros(2*cmmax + 1, dtype=complex)
    offset = cmmax
    
    for cm in range(-cmmax, cmmax + 1):
        s[cm + offset] = sonem(cm, cepmu, crad, lambda_wave)
    
    return s


def sonem(cm, cepmu, crad, lambda_wave):
    """
    Generate scattering coefficient for one mode
    
    Parameters:
    -----------
    cm : int
        Mode number
    cepmu : ndarray
        Epsilon and mu values [epsilon, mu]
    crad : float
        Cylinder radius
    lambda_wave : float
        Wavelength
        
    Returns:
    --------
    s : complex
        Scattering coefficient for this mode
    """
    k = 2 * np.pi / lambda_wave
    
    if cepmu[0] < 0:
        # PEC case
        s = -jv(cm, k*crad) / hankel2(cm, k*crad)
    else:
        # Dielectric case
        ki = k * np.sqrt(cepmu[0] * cepmu[1])
        
        # Optional absorption (commented out in original)
        # 0.1 - Weak Absorption
        # 1   - Strong Absorption
        # ki = k * (np.sqrt(cepmu[0]*cepmu[1]) - 0.1*1j)
        # ki = k * (-np.sqrt(cepmu[0]*cepmu[1]))
        
        a = np.sqrt(cepmu[0])
        b = np.sqrt(cepmu[1])
        
        # Derivative of Bessel J with k
        dbessel_jk = (jv(cm-1, crad*k) - jv(cm+1, crad*k)) / 2
        
        # Derivative of Bessel J with ki
        dbessel_jki = (jv(cm-1, crad*ki) - jv(cm+1, crad*ki)) / 2
        
        # Derivative of Hankel H2 with k
        dhankel_h2k = (hankel2(cm-1, crad*k) - hankel2(cm+1, crad*k)) / 2
        
        numerator = (-b * jv(cm, crad*ki) * dbessel_jk + 
                    a * jv(cm, crad*k) * dbessel_jki)
        
        denominator = (b * jv(cm, crad*ki) * dhankel_h2k - 
                      a * dbessel_jki * hankel2(cm, crad*k))
        
        s = numerator / denominator
    
    return s