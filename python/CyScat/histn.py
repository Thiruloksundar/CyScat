"""
histn - Normalized histogram

Python translation of MATLAB histn.m
"""

import numpy as np
import matplotlib.pyplot as plt


def histn(y, m):
    """
    Compute and plot a normalized histogram.

    Parameters:
    -----------
    y : ndarray
        Input data
    m : int
        Number of bins

    Returns:
    --------
    yc : ndarray
        Bin centers
    h : ndarray
        Histogram counts (normalized)
    """
    y = np.asarray(y).flatten()
    w = np.ones_like(y)

    dy = (np.max(y) - np.min(y)) / m

    # Compute bin indices
    bin_idx = np.floor((y - np.min(y)) / dy) + 1
    bin_idx = np.clip(bin_idx, 1, m).astype(int)

    # Compute bin edges and centers
    yy = np.min(y) + dy * np.arange(m + 1)
    yc = (yy[:-1] + yy[1:]) / 2

    # Compute histogram using sparse-like accumulation
    h = np.zeros(m)
    for i, b in enumerate(bin_idx):
        h[int(b) - 1] += w[i]  # -1 for 0-based indexing

    # Plot normalized histogram
    width = yc[1] - yc[0] if len(yc) > 1 else 1
    plt.bar(yc, h / (width * len(w)), width=width * 0.8)

    return yc, h
