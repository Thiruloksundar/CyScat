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

Julia translation
"""

"""
    get_partition(S::AbstractMatrix, partition_index::Int)

Extract a partition (block) from the S-matrix.

# Arguments
- `S::AbstractMatrix`: Full S-matrix
- `partition_index::Int`: 1 for S11, 2 for S12, 3 for S21, 4 for S22

# Returns
- `SP`: The requested partition
"""
function get_partition(S::AbstractMatrix, partition_index::Int)
    n = size(S, 1)
    half = n ÷ 2
    index1 = 1:half
    index2 = (half+1):n

    if partition_index == 1
        SP = S[index1, index1]
    elseif partition_index == 2
        SP = S[index1, index2]
    elseif partition_index == 3
        SP = S[index2, index1]
    elseif partition_index == 4
        SP = S[index2, index2]
    else
        error("Wrong PartitionIndex!!")
    end

    return SP
end

"""
    smat_to_s11(S)

Extract S11 (reflection from side 1) from S-matrix.
"""
smat_to_s11(S) = get_partition(S, 1)

"""
    smat_to_s12(S)

Extract S12 (transmission from side 2 to 1) from S-matrix.
"""
smat_to_s12(S) = get_partition(S, 2)

"""
    smat_to_s21(S)

Extract S21 (transmission from side 1 to 2) from S-matrix.
"""
smat_to_s21(S) = get_partition(S, 3)

"""
    smat_to_s22(S)

Extract S22 (reflection from side 2) from S-matrix.
"""
smat_to_s22(S) = get_partition(S, 4)
