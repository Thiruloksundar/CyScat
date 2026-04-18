"""
Truncator
Coder : Curtis Jin
Date  : 2011/MAY/18th WEDNESDAY
Contact : jsirius@umich.edu
Description : Code for truncating S-matrix to remove evanescent modes

Python translation
"""

import numpy as np


def truncator(S, nmax, no_eva_mode):
    """
    Truncate S-matrix to remove evanescent modes and compute transmission metrics.

    Parameters:
    -----------
    S : ndarray
        Full S-matrix
    nmax : int
        Maximum mode number
    no_eva_mode : int
        Number of evanescent modes to remove from each side

    Returns:
    --------
    S_truncated : ndarray
        Truncated S-matrix (propagating modes only)
    e : ndarray
        Eigenvalues (singular values squared) of transmission matrix
    Gain : float
        Gain factor (optimal TC / normal TC)
    OptWF : ndarray
        Optimal wavefront (right singular vector)
    DOU : float
        Degree of unitarity metric
    """
    # Extract S-matrix blocks
    S11 = S[0:2*nmax+1, 0:2*nmax+1]
    S12 = S[0:2*nmax+1, 2*nmax+1:]
    S21 = S[2*nmax+1:, 0:2*nmax+1]
    S22 = S[2*nmax+1:, 2*nmax+1:]

    # Truncate to remove evanescent modes
    S11_truncated = S11[no_eva_mode:-no_eva_mode if no_eva_mode > 0 else None,
                        no_eva_mode:-no_eva_mode if no_eva_mode > 0 else None]
    S12_truncated = S12[no_eva_mode:-no_eva_mode if no_eva_mode > 0 else None,
                        no_eva_mode:-no_eva_mode if no_eva_mode > 0 else None]
    S21_truncated = S21[no_eva_mode:-no_eva_mode if no_eva_mode > 0 else None,
                        no_eva_mode:-no_eva_mode if no_eva_mode > 0 else None]
    S22_truncated = S22[no_eva_mode:-no_eva_mode if no_eva_mode > 0 else None,
                        no_eva_mode:-no_eva_mode if no_eva_mode > 0 else None]

    S_truncated = np.block([[S11_truncated, S12_truncated],
                            [S21_truncated, S22_truncated]])

    # Calculate degree of unitarity
    test = np.abs(S_truncated.conj().T @ S_truncated)
    DOU = np.sum(test) / len(test)

    # Get transmission matrix and perform SVD
    divider = len(S_truncated) // 2
    index1 = slice(0, divider)
    index2 = slice(divider, divider * 2)

    T = S_truncated[index2, index1]
    U, tau, Vh = np.linalg.svd(T)
    V = Vh.conj().T

    # Eigenvalues (singular values squared)
    e = tau ** 2

    # Optimal wavefront
    OptWF = V[:, 0]

    # Calculate gain
    # MATLAB: NoPropagatingModes = nmax - NoEvaMode; TCNormal = T(:,NoPropagatingModes);
    # MATLAB is 1-indexed, so column 1 in MATLAB = column 0 in Python
    no_propagating_modes = nmax - no_eva_mode
    TC_normal = T[:, no_propagating_modes - 1]  # -1 for 0-indexed Python
    TC_normal = (TC_normal.conj().T @ TC_normal).real  # Result is real positive
    TC_opt = np.max(e)
    Gain = TC_opt / TC_normal if TC_normal != 0 else np.inf

    return S_truncated, e, Gain, OptWF, DOU
