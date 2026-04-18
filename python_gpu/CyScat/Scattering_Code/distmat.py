"""
DISTMAT - Distance matrix for a set of points
Returns the point-to-point distance between all pairs of points in XY
(similar to PDIST in the Statistics Toolbox)

Original Author: Joseph Kirk
Email: jdkirk630 at gmail dot com
Release: 1.0
Release Date: 5/29/07
Translated to Python
"""

import numpy as np


def distmat(xy, opt=None):
    """
    Distance matrix for a set of points
    
    Parameters:
    -----------
    xy : ndarray
        NxP matrix of coordinates for N points in P dimensions
    opt : int, optional
        Integer between 1 and 4 representing the chosen method for 
        computing the distance matrix
        If None, automatically selected based on size
        
    Returns:
    --------
    dmat : ndarray
        NxN matrix where DMAT[i,j] is the distance from xy[i,:] to xy[j,:]
    opt : int
        Method used to compute the distance matrix
        
    Note:
        DISTMAT contains 4 methods for computing the distance matrix:
        OPT=1: Usually fastest for small inputs. Takes advantage of symmetric
               property to perform half as many calculations
        OPT=2: Usually fastest for medium inputs. Uses fully vectorized method
        OPT=3: Usually fastest for large inputs. Uses partially vectorized
               method with relatively small memory requirement
        OPT=4: Another compact calculation, but usually slower than others
    """
    n, dims = xy.shape
    numels = n * n * dims
    
    # Automatic option selection if not specified
    if opt is None:
        if numels > 5e4:
            opt = 3
        elif n < 20:
            opt = 1
        else:
            opt = 2
    else:
        opt = max(1, min(4, int(round(abs(opt)))))
    
    # Distance matrix calculation options
    if opt == 1:
        # Half as many computations (symmetric upper triangular property)
        k, kk = np.where(np.triu(np.ones((n, n)), 1))
        dmat = np.zeros((n, n))
        dmat[k, kk] = np.sqrt(np.sum((xy[k, :] - xy[kk, :]) ** 2, axis=1))
        dmat[kk, k] = dmat[k, kk]
    
    elif opt == 2:
        # Fully vectorized calculation (very fast for medium inputs)
        a = xy[np.newaxis, :, :]  # Shape: (1, n, dims)
        b = xy[:, np.newaxis, :]  # Shape: (n, 1, dims)
        dmat = np.sqrt(np.sum((a - b) ** 2, axis=2))
    
    elif opt == 3:
        # Partially vectorized (smaller memory requirement for large inputs)
        dmat = np.zeros((n, n))
        for k in range(n):
            dmat[k, :] = np.sqrt(np.sum((xy[k, :] - xy) ** 2, axis=1))
    
    elif opt == 4:
        # Another compact method, generally slower than the others
        a = np.arange(n)
        b = np.tile(a, (n, 1))
        dmat = np.sqrt(np.sum((xy[b.flatten(), :] - xy[b.T.flatten(), :]) ** 2, axis=1))
        dmat = dmat.reshape(n, n)
    
    return dmat, opt