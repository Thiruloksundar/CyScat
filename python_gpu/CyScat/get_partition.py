"""
GetPartition
Coder : Curtis Jin
Date  : 2011/MAY/20th FRIDAY
Contact : jsirius@umich.edu
Description : Code for getting the partition from S-matrix
              11 : 1,
              12 : 2,
              21 : 3,
              22 : 4

Python translation
"""

import numpy as np


def get_partition(S, partition_index):
    """
    Extract a partition (block) from the S-matrix.

    Parameters:
    -----------
    S : ndarray
        Full S-matrix
    partition_index : int
        1 for S11, 2 for S12, 3 for S21, 4 for S22

    Returns:
    --------
    SP : ndarray
        The requested partition
    """
    n = len(S)
    index1 = slice(0, n // 2)
    index2 = slice(n // 2, n)

    if partition_index == 1:
        SP = S[index1, index1]
    elif partition_index == 2:
        SP = S[index1, index2]
    elif partition_index == 3:
        SP = S[index2, index1]
    elif partition_index == 4:
        SP = S[index2, index2]
    else:
        raise ValueError('Wrong PartitionIndex!!')

    return SP


def smat_to_s11(S):
    """Extract S11 (reflection from side 1) from S-matrix."""
    return get_partition(S, 1)


def smat_to_s12(S):
    """Extract S12 (transmission from side 2 to 1) from S-matrix."""
    return get_partition(S, 2)


def smat_to_s21(S):
    """Extract S21 (transmission from side 1 to 2) from S-matrix."""
    return get_partition(S, 3)


def smat_to_s22(S):
    """Extract S22 (reflection from side 2) from S-matrix."""
    return get_partition(S, 4)
