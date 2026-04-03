"""
smatrix Code
Coder: Curtis Jin
Date: 2010/DEC/3rd Friday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : Scattering Matrix Generating Code

Julia translation
"""

using LinearAlgebra
using Dates
using Printf

include("ky.jl")
include("transall.jl")
include("sall.jl")
include("vall.jl")
include("scattering_coefficients_all.jl")
include("simulation_time_profile.jl")

"""
    smatrix(clocs, cmmaxs, cepmus, crads, period, lambda_wave, nmax, d, sp, interaction)

Generate Scattering Matrix.

# Arguments
- `clocs`: Cylinder locations
- `cmmaxs`: Maximum cylinder modes for each cylinder
- `cepmus`: Epsilon and mu values for cylinders
- `crads`: Cylinder radii
- `period`: Period of the structure
- `lambda_wave`: Wavelength
- `nmax`: Maximum mode number
- `d`: Distance/thickness
- `sp`: SMatrix parameters
- `interaction`: "On" or "Off" for interaction

# Returns
- `S`: Scattering matrix
- `STP`: Simulation time profile
"""
function smatrix(clocs, cmmaxs, cepmus, crads, period, lambda_wave, nmax, d, sp, interaction)
    # Make a copy of clocs to avoid modifying the original
    clocs = copy(clocs)

    # Parameter for progress bar
    no_cylinders = length(cmmaxs)
    total_steps = no_cylinders * (no_cylinders - 1) ÷ 2 + 2 * (2 * nmax + 1)

    k = 2π / lambda_wave
    # RT must be defined early — used by lu and matrix allocation below.
    RT = promote_type(eltype(clocs), typeof(real(lambda_wave)))
    phiinc = sp["phiinc"]

    # Open debugging log
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "--------------------\n")
        write(debug_file, "Simulation Started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")).\n")
    end

    println("--------------------------")
    println("Calculating T-Matrix")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Calculating T-Matrix\n")
    end

    st_tmatrix_start = time()
    if interaction == "On"
        t = transall(clocs, cmmaxs, period, lambda_wave, phiinc, sp, total_steps)
    else
        # To nullify interaction we set period to be negative
        t = transall(clocs, cmmaxs, -1, lambda_wave, phiinc, sp, total_steps)
    end

    st_tmatrix = time() - st_tmatrix_start

    println("Took $(st_tmatrix/60) minutes")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Took $(st_tmatrix/60) minutes\n")
    end

    println("Calculating S-Vector")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Calculating S-Vector\n")
    end

    s = sall(cmmaxs, cepmus, crads, lambda_wave)
    z = I - Diagonal(s) * t

    println("LU Decomposition")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "LU Decomposition\n")
    end
    st_lu_start = time()
    # lu() cannot handle Dual numbers (calls Float64 internally).
    # For AD passes use the matrix directly; \ dispatches generically.
    use_lu = (RT == Float64)
    lu_fact = use_lu ? lu(z) : z
    st_lu = time() - st_lu_start

    println("Took $(st_lu/60) minutes")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Took $(st_lu/60) minutes\n")
    end

    # CT uses RT defined above (early, before lu).
    CT = Complex{RT}
    s11matrix = zeros(CT, 2*nmax+1, 2*nmax+1)
    s12matrix = zeros(CT, 2*nmax+1, 2*nmax+1)
    s21matrix = zeros(CT, 2*nmax+1, 2*nmax+1)
    s22matrix = zeros(CT, 2*nmax+1, 2*nmax+1)

    println("Computing S11&S21 Partition")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Computing S11&S21 Partition\n")
    end

    st_up_start = time()
    # For S11&S21 Partition
    up_down = 1  # Going Up

    for nin in -nmax:nmax
        mid_idx = sp["MiddleIndex"]
        kxex = sp["kxs"][mid_idx + nin]
        # Extract scalar value if kxex is an array
        if isa(kxex, AbstractArray)
            kxex = kxex[1]
        end
        v = Diagonal(s) * vall(clocs, cmmaxs, lambda_wave, kxex, up_down)
        c = lu_fact \ v

        sj = scatteringcoefficientsall(clocs, cmmaxs, period, lambda_wave,
                                       nmax, c, up_down, d, sp)
        s11matrix[:, nin+nmax+1] = sj[:, 1]
        s21matrix[:, nin+nmax+1] = sj[:, 2]

        # Including the incident field
        incident_index = nmax + nin + 1  # Julia 1-indexed
        kyex = sp["kys"][mid_idx + nin]
        # Extract scalar value if kyex is an array
        if isa(kyex, AbstractArray)
            kyex = kyex[1]
        end
        s21matrix[incident_index, nin+nmax+1] += 1 * exp(-1im * kyex * d)
    end

    st_up = time() - st_up_start
    println("Took $(st_up/60) minutes")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Took $(st_up/60) minutes\n")
    end

    println("Computing S12&S22 Partition")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Computing S12&S22 Partition\n")
    end

    st_down_start = time()
    # For S12&S22 Partition
    up_down = -1  # Going Down
    clocs[:, 2] = clocs[:, 2] .- d

    for nin in -nmax:nmax
        mid_idx = sp["MiddleIndex"]
        kxex = sp["kxs"][mid_idx + nin]
        # Extract scalar value if kxex is an array
        if isa(kxex, AbstractArray)
            kxex = kxex[1]
        end
        v = Diagonal(s) * vall(clocs, cmmaxs, lambda_wave, kxex, up_down)
        c = lu_fact \ v

        sj = scatteringcoefficientsall(clocs, cmmaxs, period, lambda_wave,
                                       nmax, c, up_down, d, sp)
        s12matrix[:, nin+nmax+1] = sj[:, 1]
        s22matrix[:, nin+nmax+1] = sj[:, 2]

        # Including the incident field
        incident_index = nmax + nin + 1  # Julia 1-indexed
        kyex_inc = sp["kys"][mid_idx + nin]
        # Extract scalar value if kyex_inc is an array
        if isa(kyex_inc, AbstractArray)
            kyex_inc = kyex_inc[1]
        end
        s12matrix[incident_index, nin+nmax+1] += 1 * exp(-1im * kyex_inc * d)
    end

    st_down = time() - st_down_start
    println("Took $(st_down/60) minutes")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Took $(st_down/60) minutes\n")
    end

    S = [s11matrix s12matrix; s21matrix s22matrix]

    # DEBUG: Show S-matrix size before normalization
    println("DEBUG: S-matrix size before normalization: $(size(S, 1))x$(size(S, 2))")
    println("DEBUG: s11matrix size: $(size(s11matrix, 1))x$(size(s11matrix, 2))")
    println("DEBUG: nmax = $nmax")

    println("Normalizing Scattering Matrix")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Normalizing Scattering Matrix\n")
    end

    st_norm_start = time()
    # Normalization Process
    m = collect(-nmax:nmax)
    m = vcat(m, m)
    # Cast kxs to complex(RT) so ky() is compatible when k is a Dual.
    kxs = convert.(complex(RT), 2π / period .* m)
    kys_norm = ky(k, kxs)
    nor2 = sqrt.(kys_norm ./ k)
    nor1 = sqrt.(k ./ kys_norm)
    P2 = Diagonal(vec(nor2))
    P1 = Diagonal(vec(nor1))
    S = P2 * S * P1
    st_norm = time() - st_norm_start

    println("Took $(st_norm/60) minutes")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Took $(st_norm/60) minutes\n")
    end

    # DEBUG: Show S-matrix size after normalization
    println("DEBUG: S-matrix size after normalization: $(size(S, 1))x$(size(S, 2))")
    # Only compute debug SVD for plain Float64 (not AD Dual types)
    if eltype(clocs) == Float64
        svals_debug = svd(S).S
        println("DEBUG: Max singular value: $(maximum(svals_debug))")
        println("DEBUG: Min singular value: $(minimum(svals_debug))")
        println("DEBUG: Number of singular values: $(length(svals_debug))")
    end

    STP = simulation_time_profile(st_tmatrix, st_lu, st_up, st_down, st_norm)
    println("--------------------------")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "--------------------------\n")
    end

    println("Total Simulation Time: $(STP["TST"]/60) minutes")
    open("DebuggingNote.txt", "a") do debug_file
        write(debug_file, "Total Simulation Time: $(STP["TST"]/60) minutes\n")
        write(debug_file, "Simulation Terminated at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")).\n")
    end

    return S, STP
end