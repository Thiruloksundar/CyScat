"""
ky Code
Coder: Curtis Jin
Date: 2011/FEB/13th Sunday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : y component of K extractor
             Modified it to be applicable to vectors
Translated to Python
"""

import numpy as np


def ky(k, kx):
    """
    Calculate y component of wave vector

    Parameters:
    -----------
    k : float or ndarray
        Wave number
    kx : float or ndarray
        x component of wave vector

    Returns:
    --------
    ky : float or ndarray
        y component of wave vector
    """
    # MATLAB: ky = sqrt(k^2-kx.^2);
    # Ensure complex sqrt to handle evanescent modes (k^2 < kx^2)
    ky_val = np.sqrt(k**2 - kx**2 + 0j)

    # MATLAB: index = find(imag(ky) > 0); ky(index) = conj(ky(index));
    # Handle both scalar and array cases
    is_scalar = np.isscalar(ky_val) or ky_val.shape == ()

    if is_scalar:
        # Scalar case
        if np.imag(ky_val) > 0:
            ky_val = np.conj(ky_val)
    else:
        # Array case
        mask = np.imag(ky_val) > 0
        ky_val[mask] = np.conj(ky_val[mask])

    return ky_val
