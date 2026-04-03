"""
histn - Normalized histogram

Julia translation of MATLAB histn.m
Original MATLAB code by Curtis Jin (jsirius@umich.edu)
"""

using Plots

"""
    histn(y, m::Int)

Compute and plot a normalized histogram.

# Arguments
- `y`: Input data (vector)
- `m::Int`: Number of bins

# Returns
- `yc`: Bin centers
- `h`: Histogram counts (normalized)
"""
function histn(y, m::Int)
    y = vec(collect(y))
    w = ones(length(y))

    dy = (maximum(y) - minimum(y)) / m

    # Compute bin indices
    bin_idx = floor.((y .- minimum(y)) ./ dy) .+ 1
    bin_idx = clamp.(bin_idx, 1, m)
    bin_idx = Int.(bin_idx)

    # Compute bin edges and centers
    yy = minimum(y) .+ dy .* (0:m)
    yc = (yy[1:end-1] .+ yy[2:end]) ./ 2

    # Compute histogram using accumulation
    h = zeros(m)
    for (i, b) in enumerate(bin_idx)
        h[b] += w[i]
    end

    # Plot normalized histogram
    width = length(yc) > 1 ? yc[2] - yc[1] : 1.0
    bar(yc, h ./ (width * length(w)), bar_width=width*0.8, legend=false)

    return yc, h
end
