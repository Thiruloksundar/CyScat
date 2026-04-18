"""
IncreaseBuffer - Increase the buffer length of an S-matrix
Coder: Curtis Jin
Date: 2011/MAR/14th Monday
Contact: jsirius@umich.edu
Description: Increase the buffer length by appending free-space propagation phase.
Translated to Python.
"""
import numpy as np
from .ky import ky


def increase_buffer(S, d, dinc, lambda_wave, period, nmax):
    """
    Increase the buffer length of an S-matrix by adding free-space propagation.

    Parameters
    ----------
    S : ndarray
        S-matrix (2*(2*nmax+1) x 2*(2*nmax+1))
    d : float
        Current slab thickness
    dinc : float
        Thickness increment
    lambda_wave : float
        Wavelength
    period : float
        Periodicity
    nmax : int
        Maximum mode index

    Returns
    -------
    Smod : ndarray
        Modified S-matrix with increased buffer
    dmod : float
        New slab thickness (d + dinc)
    """
    dmod = d + dinc
    k = 2 * np.pi / lambda_wave

    m = np.arange(-nmax, nmax + 1)
    m = np.concatenate([m, m])
    kxs = 2 * np.pi / period * m
    kys = ky(k, kxs)
    phase_vector = np.exp(-1j * kys * dinc)

    Smod = S * phase_vector[np.newaxis, :]  # S @ diag(PhaseVector)

    return Smod, dmod
