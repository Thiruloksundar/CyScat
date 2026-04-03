"""
cascadertwo Code
Coder : Curtis Jin
Date  : 2011/FEB/16th Wednesday
Contact : jsirius@umich.edu
Description : Cascading two systems

GPU-accelerated version: uses GPU for matrix inversions and multiplications.
"""

import numpy as np
from . import gpu_backend as gb


def cascadertwo(S1, d1, S2, d2):
    """
    Cascade two S-matrix systems.

    Parameters
    ----------
    S1 : ndarray - S-matrix of first system
    d1 : float - Thickness of first system
    S2 : ndarray - S-matrix of second system
    d2 : float - Thickness of second system

    Returns
    -------
    Scas : ndarray - Cascaded S-matrix
    dcas : float - Total thickness
    """
    dcas = d1 + d2
    divider = len(S1) // 2

    if gb.GPU_AVAILABLE:
        return _cascade_gpu(S1, S2, divider, dcas)
    else:
        return _cascade_cpu(S1, S2, divider, dcas)


def _cascade_cpu(S1, S2, divider, dcas):
    """Cascade on CPU using numpy."""
    I = np.eye(divider)

    R1 = S1[:divider, :divider]
    T1 = S1[divider:, :divider]
    T1tilde = S1[:divider, divider:]
    R1tilde = S1[divider:, divider:]

    R2 = S2[:divider, :divider]
    T2 = S2[divider:, :divider]
    T2tilde = S2[:divider, divider:]
    R2tilde = S2[divider:, divider:]

    temp1 = np.linalg.inv(I - R2 @ R1tilde)
    temp2 = np.linalg.inv(I - R1tilde @ R2)

    R = R1 + T1tilde @ temp1 @ R2 @ T1
    T = T2 @ temp2 @ T1
    Ttilde = T1tilde @ temp1 @ T2tilde
    Rtilde = R2tilde + T2 @ temp2 @ R1tilde @ T2tilde

    Scas = np.vstack([np.hstack([R, Ttilde]), np.hstack([T, Rtilde])])
    return Scas, dcas


def _cascade_gpu(S1, S2, divider, dcas):
    """Cascade on GPU using CuPy for inversions and multiplications."""
    xp = gb.xp
    S1g = gb.to_gpu(S1)
    S2g = gb.to_gpu(S2)
    I = xp.eye(divider)

    R1 = S1g[:divider, :divider]
    T1 = S1g[divider:, :divider]
    T1tilde = S1g[:divider, divider:]
    R1tilde = S1g[divider:, divider:]

    R2 = S2g[:divider, :divider]
    T2 = S2g[divider:, :divider]
    T2tilde = S2g[:divider, divider:]
    R2tilde = S2g[divider:, divider:]

    temp1 = xp.linalg.inv(I - R2 @ R1tilde)
    temp2 = xp.linalg.inv(I - R1tilde @ R2)

    R = R1 + T1tilde @ temp1 @ R2 @ T1
    T = T2 @ temp2 @ T1
    Ttilde = T1tilde @ temp1 @ T2tilde
    Rtilde = R2tilde + T2 @ temp2 @ R1tilde @ T2tilde

    Scas = gb.to_cpu(xp.vstack([xp.hstack([R, Ttilde]), xp.hstack([T, Rtilde])]))
    return Scas, dcas
