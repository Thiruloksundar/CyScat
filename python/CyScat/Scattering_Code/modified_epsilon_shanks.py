"""
Modified Epsilon Shanks Code
Coder: Curtis Jin
Date: 2010/NOV/21th Sunday
Contact: jsirius@umich.edu
Description: Implementation of recursive version of Shanks Transformation 
             It's NOT using determinant method
Translated to Python
Fixed: Added safe division checks to prevent inf/NaN propagation
"""

import numpy as np


def modified_epsilon_shanks(newentry, a1, a2):
    """
    Modified Epsilon Shanks transformation for accelerating convergence

    Parameters:
    -----------
    newentry : complex
        New entry to add to the sequence
    a1 : ndarray
        First auxiliary array
    a2 : ndarray
        Second auxiliary array

    Returns:
    --------
    S : ndarray
        Transformed sequence
    """
    kplus2 = len(a1)
    S = np.zeros(kplus2, dtype=complex)

    S[0] = np.inf
    S[1] = newentry

    for idx in range(1, kplus2 - 1):
        # MATLAB computes factors first, then checks if they're finite
        # In Python/NumPy: 1/inf = 0 (finite), 1/0 = inf (not finite)
        # This matches MATLAB behavior

        # Suppress divide warnings since we handle inf/nan explicitly
        with np.errstate(divide='ignore', invalid='ignore'):
            first_factor = 1 / (a2[idx] - a1[idx])
            second_factor = 1 / (S[idx] - a1[idx])
            third_factor = 1 / (a2[idx-1] - a1[idx])

        # Check if any factor is not finite (inf or NaN)
        # Note: 1/inf = 0 which IS finite, so this passes correctly
        if not np.isfinite(first_factor):
            S[idx+1] = a1[idx]
        elif not np.isfinite(second_factor):
            S[idx+1] = a1[idx]
        elif not np.isfinite(third_factor):
            S[idx+1] = a1[idx]
        else:
            S[idx+1] = 1 / (first_factor + second_factor - third_factor) + a1[idx]

    return S