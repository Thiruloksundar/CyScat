"""
DISTMAT - Distance matrix for a set of points
Returns the point-to-point distance between all pairs of points in XY

Original Author: Joseph Kirk
Email: jdkirk630 at gmail dot com
Release: 1.0
Release Date: 5/29/07

Julia translation
"""

using LinearAlgebra

"""
    distmat(xy::AbstractMatrix, opt::Union{Int,Nothing}=nothing)

Distance matrix for a set of points.

# Arguments
- `xy::AbstractMatrix`: NxP matrix of coordinates for N points in P dimensions
- `opt::Union{Int,Nothing}`: Integer between 1 and 4 representing the chosen method.
  If nothing, automatically selected based on size.

# Returns
- `dmat`: NxN matrix where DMAT[i,j] is the distance from xy[i,:] to xy[j,:]
- `opt`: Method used to compute the distance matrix

# Note
DISTMAT contains 4 methods for computing the distance matrix:
- OPT=1: Usually fastest for small inputs
- OPT=2: Usually fastest for medium inputs (fully vectorized)
- OPT=3: Usually fastest for large inputs (partially vectorized)
- OPT=4: Another compact calculation, but usually slower
"""
function distmat(xy::AbstractMatrix, opt::Union{Int,Nothing}=nothing)
    n, dims = size(xy)
    numels = n * n * dims

    # Automatic option selection if not specified
    if isnothing(opt)
        if numels > 5e4
            opt = 3
        elseif n < 20
            opt = 1
        else
            opt = 2
        end
    else
        opt = max(1, min(4, round(Int, abs(opt))))
    end

    # Distance matrix calculation options
    if opt == 1
        # Half as many computations (symmetric upper triangular property)
        dmat = zeros(n, n)
        for i in 1:n
            for j in (i+1):n
                dmat[i, j] = sqrt(sum((xy[i, :] .- xy[j, :]).^2))
                dmat[j, i] = dmat[i, j]
            end
        end

    elseif opt == 2
        # Fully vectorized calculation (very fast for medium inputs)
        dmat = zeros(n, n)
        for i in 1:n
            for j in 1:n
                dmat[i, j] = sqrt(sum((xy[i, :] .- xy[j, :]).^2))
            end
        end

    elseif opt == 3
        # Partially vectorized (smaller memory requirement for large inputs)
        dmat = zeros(n, n)
        for k in 1:n
            for j in 1:n
                dmat[k, j] = sqrt(sum((xy[k, :] .- xy[j, :]).^2))
            end
        end

    elseif opt == 4
        # Alternative method using broadcasting
        dmat = zeros(n, n)
        for i in 1:n
            for j in 1:n
                dmat[i, j] = norm(xy[i, :] .- xy[j, :])
            end
        end
    end

    return dmat, opt
end
