"""
SMatrixParameters Code
Modified Date: 2011.02.09.Wednesday
Features: Generate a structure that contains parameters for constructing
          a scattering matrix.
Coder: Curtis Jin
Contact: jsirius@umich.edu
Translated to Python
"""

import numpy as np


def ky(k0, kxs):
    """Calculate ky from k0 and kxs

    For evanescent modes (kxs > k0), ensures imaginary part is negative
    so that exp(-i*ky*z) decays for z > 0.
    """
    ky_val = np.sqrt(k0**2 - kxs**2 + 0j)
    # For evanescent modes, ensure imag(ky) < 0 for proper decay
    # MATLAB: index = find(imag(ky) > 0); ky(index) = conj(ky(index));
    mask = np.imag(ky_val) > 0
    ky_val[mask] = np.conj(ky_val[mask])
    return ky_val


def smatrix_parameters(lambda_wave, period, phiinc, epsseries, epsloc, 
                       nrepeat_spatial, nrepeat_spectral, jmax, 
                       kshanks_spatial, kshanks_spectral, spectral, spectral_cond):
    """
    Generate SMatrix parameters structure
    
    Parameters:
    -----------
    lambda_wave : float
        Wavelength
    period : float
        Period of the structure
    phiinc : float
        Incident angle
    epsseries : float
        Epsilon for series calculation
    epsloc : float
        Epsilon for local calculation
    nrepeat_spatial : int
        Number of spatial repeats
    nrepeat_spectral : int
        Number of spectral repeats
    jmax : int
        Maximum j index
    kshanks_spatial : int
        Shanks parameter for spatial
    kshanks_spectral : int
        Shanks parameter for spectral
    spectral : int
        Spectral flag
    spectral_cond : float
        Spectral condition parameter
        
    Returns:
    --------
    sp : dict
        Dictionary containing SMatrix parameters
    """
    
    k0 = 2 * np.pi / lambda_wave
    
    # Generate kxs array
    if phiinc == np.pi / 2:
        kxs = 2 * np.pi * np.arange(-jmax, jmax+1) / period
    else:
        kxs = 2 * np.pi * np.arange(-jmax, jmax+1) / period + k0 * np.cos(phiinc)
    
    kxs = kxs.reshape(-1, 1)  # Column vector
    
    # Calculate kys
    kys = ky(k0, kxs)
    
    middle_index = int(np.ceil(len(kxs) / 2)) - 1

    two_over_period = 2 / period
    # MATLAB: Angles = asin(kxs/k0);
    # For evanescent modes where |kxs/k0| > 1, arcsin returns complex values
    # Must ensure the argument is complex to get complex arcsin
    arg = kxs / k0
    angles = np.arcsin(arg + 0j)  # Force complex to handle evanescent modes

    # MATLAB and Python use different branch cuts for arcsin:
    # For x > 1:  Python gives +i*..., MATLAB gives -i*...  → need to conjugate
    # For x < -1: Both Python and MATLAB give +i*...       → do NOT conjugate
    # Solution: only conjugate when kxs/k0 > 1 (positive evanescent modes)
    positive_evanescent_mask = arg > 1
    angles[positive_evanescent_mask] = np.conj(angles[positive_evanescent_mask])
    
    # Generate SMatrixParameter Structure
    sp = {
        'lambda': lambda_wave,
        'period': period,
        'phiinc': phiinc,
        'k0': k0,
        'kxs': kxs,
        'kys': kys,
        'Angles': angles,
        'TwoOverPeriod': two_over_period,
        'MiddleIndex': middle_index,
        'epsseries': epsseries,
        'epsloc': epsloc,
        'nrepeatSpatial': nrepeat_spatial,
        'nrepeatSpectral': nrepeat_spectral,
        'jmax': jmax,
        'kshanksSpatial': kshanks_spatial,
        'kshanksSpectral': kshanks_spectral,
        'spectral': spectral,
        'spectralCond': spectral_cond
    }
    
    return sp