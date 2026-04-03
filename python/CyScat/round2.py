"""
round2 - Rounding to specified decimal places

Python translation of MATLAB round2.m
"""

import numpy as np


def round2(decimal, number):
    """
    Round a number to specified decimal places.

    Parameters:
    -----------
    decimal : int
        Number of decimal places
    number : float or ndarray
        Number(s) to round

    Returns:
    --------
    rounded_value : float or ndarray
        Rounded value(s)
    """
    rounded_value = number * 10 ** decimal
    rounded_value = np.fix(rounded_value)
    rounded_value = rounded_value * 10 ** (-decimal)

    return rounded_value
