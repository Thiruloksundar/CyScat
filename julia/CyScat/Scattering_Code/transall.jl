"""
transall Code
Coder: Curtis Jin
Date: 2011/FEB/13th Sunday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : translational matrix generating code
             Using Modified Shanks Transformation
             Exploiting symmetric structure of T-Matrix

Julia translation
"""

using SpecialFunctions
using LinearAlgebra
using ToeplitzMatrices

include("modified_epsilon_shanks.jl")

"""
    transall(clocs, cmmaxs, period, lambda_wave, phiinc, sp, total_steps)

Generate translational matrix for all cylinders.
EXACT translation from MATLAB transall.m
"""
function transall(clocs, cmmaxs, period, lambda_wave, phiinc, sp, total_steps)
    no_cylinders = length(cmmaxs)
    tot_no_modes = Int(sum(cmmaxs .* 2 .+ 1))
    # Type-generic allocation for AD compatibility
    CT = Complex{eltype(clocs)}
    t = zeros(CT, tot_no_modes, tot_no_modes)

    # Pre-calculate the self-sum
    max_order = Int(maximum(cmmaxs))
    cms = collect(-2*max_order:2*max_order)
    self_sum_vector = transvector([0.0, 0.0], [0.0, 0.0], cms,
                                   period, lambda_wave, phiinc, sp)

    istart = 1  # Julia uses 1-based indexing
    progress_count = 1

    for icyl in 1:no_cylinders
        cmmaxob = Int(cmmaxs[icyl])

        # offset for indexing into self_sum_vector
        # Array has indices 1:(4*max_order+1), center at 2*max_order+1
        offset = 2 * max_order + 1

        # Extract t11 components
        c = self_sum_vector[offset:offset + 2*cmmaxob]
        r = reverse(self_sum_vector[offset - 2*cmmaxob:offset])
        # Ensure first elements match (required by ToeplitzMatrices)
        r[1] = c[1]

        t11 = Toeplitz(c, r)

        # Assign to main matrix
        t[istart:istart + 2*cmmaxob,
          istart:istart + 2*cmmaxob] = Matrix(t11)

        jstart = istart + 2*cmmaxob + 1

        for jcyl in (icyl + 1):no_cylinders
            cmmaxso = Int(cmmaxs[jcyl])

            # offset for tvector
            offset = cmmaxso + cmmaxob + 1

            # Calculate tvector
            cms = collect(-cmmaxob - cmmaxso:cmmaxob + cmmaxso)
            tvector = transvector(clocs[icyl, :], clocs[jcyl, :], cms,
                                  period, lambda_wave, phiinc, sp)

            # Extract t12
            start_idx = cmmaxso - cmmaxob + offset
            c = tvector[start_idx:end]
            r = reverse(tvector[1:start_idx])
            # Ensure first elements match (required by ToeplitzMatrices)
            r[1] = c[1]
            t12 = Toeplitz(c, r)

            # Extract t21
            start_idx = cmmaxob - cmmaxso + offset
            c = tvector[start_idx:end]
            r = reverse(tvector[1:start_idx])
            # Ensure first elements match (required by ToeplitzMatrices)
            r[1] = c[1]
            t21 = Toeplitz(c, r)

            # Modification matrix
            c_mod = collect(0:2*cmmaxso)
            c_mod = c_mod .+ (cmmaxob - cmmaxso)
            c_mod = (-1.0).^c_mod

            r_mod = collect(0:2*cmmaxob)
            r_mod = r_mod .+ (cmmaxob - cmmaxso)
            r_mod = (-1.0).^r_mod
            # Ensure first elements match (required by ToeplitzMatrices)
            r_mod[1] = c_mod[1]

            modification_matrix = Toeplitz(c_mod, r_mod)
            t21 = Matrix(modification_matrix) .* Matrix(t21)

            # Assign to main matrix
            t[istart:istart + 2*cmmaxob,
              jstart:jstart + 2*cmmaxso] = Matrix(t12)
            t[jstart:jstart + 2*cmmaxso,
              istart:istart + 2*cmmaxob] = t21

            jstart = jstart + 2*cmmaxso + 1
            progress_count += 1
        end

        istart = istart + 2*cmmaxob + 1
    end

    return t
end


"""
    transvector(clocob, clocso, cms, period, lambda_wave, phiinc, sp)

Calculate translation vector for a pair of cylinders.
"""
function transvector(clocob, clocso, cms, period, lambda_wave, phiinc, sp)
    epsloc = sp["epsloc"]
    rv = clocob .- clocso
    r = norm(rv)
    x = rv[1]
    y = rv[2]

    cms = vec(collect(cms))

    # Self-Sum & Spatial Sum
    if r < epsloc || abs(y) < sp["spectralCond"] || period < 0
        max_cm = Int(maximum(cms))
        min_cm = Int(minimum(cms))

        array_size = max_cm - min_cm + 1
        # FIX: promote over clocob and lambda_wave for Dual compatibility
        RT_local = promote_type(eltype(clocob), typeof(real(lambda_wave)))
        CT_local = Complex{RT_local}
        temp = zeros(CT_local, array_size)

        for cm in cms
            cm_int = Int(cm)
            idx = cm_int - min_cm + 1  # Julia 1-indexed
            temp[idx] = transper_internal(clocob, cm_int, clocso, 0, period,
                                         lambda_wave, phiinc, sp)
        end
        t = temp

    # Spectral Sum
    else
        kxs_all = vec(sp["kxs"])
        kys_all = vec(sp["kys"])
        angles_all = vec(sp["Angles"])

        # Create meshgrids
        n_kxs = length(kxs_all)
        n_cms = length(cms)

        cmgrid = repeat(cms', n_kxs, 1)
        kxsgrid = repeat(kxs_all, 1, n_cms)
        kysgrid = repeat(kys_all, 1, n_cms)
        anglesgrid = repeat(angles_all, 1, n_cms)

        # Determine sign of y
        sign_y = sign(y)
        if sign_y == 0
            sign_y = 1
        end

        # Calculate exponent
        exponent = (kxsgrid .* x .- kysgrid .* abs(y) .+
                   cmgrid .* (π .- anglesgrid .* sign_y))

        # Calculate t_matrix
        sign_power = sign_y.^cmgrid
        exp_term = exp.(1im .* exponent)

        # Protect against division by zero for grazing incidence modes
        kysgrid_safe = map(ky -> abs(ky) < 1e-10 ? sign(real(ky) + 1e-20) * 1e-10 + 0im : ky, kysgrid)
        t_matrix = sign_power .* exp_term ./ kysgrid_safe .* sp["TwoOverPeriod"]

        # Sum along first dimension (rows)
        t = vec(sum(t_matrix, dims=1))
    end

    return t
end


"""
    transper_internal(clocob, cmob, clocso, cmso, period, lambda_wave, phiinc, sp)

Calculate periodic translation coefficient.
"""
function transper_internal(clocob, cmob, clocso, cmso, period, lambda_wave, phiinc, sp)
    k = sp["k0"]

    epsseries = sp["epsseries"]
    epsloc = sp["epsloc"]
    jmax = sp["jmax"]

    spectral_internal = sp["spectral"]
    if spectral_internal < 0
        kshanks = sp["kshanksSpatial"]
        nrepeat = sp["nrepeatSpatial"]
    else
        kshanks = sp["kshanksSpectral"]
        nrepeat = sp["nrepeatSpectral"]
    end

    rv = clocob .- clocso
    r = norm(rv)
    x = rv[1]
    y = rv[2]

    # FIX: promote over ALL inputs that may carry a Dual tag.
    # eltype(clocob) alone is Float64 when differentiating w.r.t. lambda,
    # causing ComplexF64 arrays to reject Complex{Dual} entries.
    RT    = promote_type(eltype(clocob), typeof(real(lambda_wave)))
    CT_tp = Complex{RT}

    if period < 0
        if r < epsloc
            t = zero(CT_tp)
        else
            t = trans_internal(clocob, cmob, clocso, cmso, lambda_wave, sp)
        end
        return t
    end

    if period > 0
        # Spectral Condition Checker
        if abs(y) < sp["spectralCond"]
            spectral_internal = -1
            kshanks = sp["kshanksSpatial"]
            nrepeat = sp["nrepeatSpatial"]
        end

        # Shanks Transformation Initialization - CT_tp now carries Dual when needed
        if kshanks > 0
            ashankseps1 = fill(zero(CT_tp), kshanks + 2)
            ashankseps2 = fill(zero(CT_tp), kshanks + 2)
            ashankseps1[1] = convert(CT_tp, Inf)
            ashankseps2[1] = convert(CT_tp, Inf)
        end

        if spectral_internal < 0
            if r < epsloc
                t = zero(CT_tp)
            else
                t = trans_internal(clocob, cmob, clocso, cmso, lambda_wave, sp)
            end
            if kshanks > 0
                ashankseps1[2] = t
            end
            ts = one(CT_tp)
        else
            # Spectral case
            cm = cmso - cmob
            mid_idx = sp["MiddleIndex"]

            exponent = (sp["kxs"][mid_idx] * x +
                       sp["kys"][mid_idx] * abs(y) +
                       cm * (sign(y) * sp["Angles"][mid_idx] - π))

            # Extract scalar values
            exponent_val = isa(exponent, AbstractArray) ? exponent[1] : exponent
            kys_val = isa(sp["kys"][mid_idx], AbstractArray) ? sp["kys"][mid_idx][1] : sp["kys"][mid_idx]

            t = (sign(y)^cm * exp(-1im * exponent_val) /
                kys_val * sp["TwoOverPeriod"])

            if kshanks > 0
                ashankseps1[2] = t
            end
            ts = one(CT_tp)
        end

        # Looping
        j = 1
        irepeat = 0

        while irepeat < nrepeat && j < jmax
            # Sequence Calculation
            if spectral_internal < 0
                add = (exp(-1im * k * j * period * cos(phiinc)) *
                      trans_internal(clocob, cmob, clocso .+ j .* [period, 0], cmso, lambda_wave, sp) +
                      exp(-1im * k * (-j) * period * cos(phiinc)) *
                      trans_internal(clocob, cmob, clocso .- j .* [period, 0], cmso, lambda_wave, sp))
            else
                mid_idx = sp["MiddleIndex"]
                kxp = sp["kxs"][mid_idx + j]
                kxm = sp["kxs"][mid_idx - j]

                # Extract scalar values
                kxp_val = isa(kxp, AbstractArray) ? kxp[1] : kxp
                kxm_val = isa(kxm, AbstractArray) ? kxm[1] : kxm
                kys_p = sp["kys"][mid_idx + j]
                kys_m = sp["kys"][mid_idx - j]
                kys_p_val = isa(kys_p, AbstractArray) ? kys_p[1] : kys_p
                kys_m_val = isa(kys_m, AbstractArray) ? kys_m[1] : kys_m
                ang_p = sp["Angles"][mid_idx + j]
                ang_m = sp["Angles"][mid_idx - j]
                ang_p_val = isa(ang_p, AbstractArray) ? ang_p[1] : ang_p
                ang_m_val = isa(ang_m, AbstractArray) ? ang_m[1] : ang_m

                exponent1 = (kxp_val * x + kys_p_val * abs(y) +
                           cm * (sign(y) * ang_p_val - π))
                exponent2 = (kxm_val * x + kys_m_val * abs(y) +
                           cm * (sign(y) * ang_m_val - π))

                add = (sign(y)^cm * sp["TwoOverPeriod"] *
                      (exp(-1im * exponent1) / kys_p_val +
                       exp(-1im * exponent2) / kys_m_val))
            end

            # Shanks Transformation & Convergence Checker
            told = t
            t = t + add

            if kshanks <= 0
                rerror = abs(t - told)
            else
                T = typeof(t)

                if eltype(ashankseps1) != T
                    ashankseps1 = T.(ashankseps1)
                end

                if eltype(ashankseps2) != T
                    ashankseps2 = T.(ashankseps2)
                end

                S = modified_epsilon_shanks(t, ashankseps1, ashankseps2)
                ashankseps2 = copy(ashankseps1)
                ashankseps1 = copy(S)

                if j <= 2*kshanks + 1
                    rerror = 1.0
                else
                    tsold = ts
                    ts = S[end]
                    rerror = abs(ts - tsold)
                end
            end

            if rerror < epsseries
                irepeat += 1
            else
                irepeat = 0
            end

            j += 1
        end

        if j == jmax
            open("DebuggingNote.txt", "a") do f
                println("--------------------")
                write(f, "--------------------\n")
                println("j has reached jmax!!")
                write(f, "j has reached jmax!!\n")
            end
        end
    end

    if kshanks > 0 && period > 0
        t = ts
    end

    return t
end


"""
    trans_internal(clocob, cmob, clocso, cmso, lambda_wave, sp)

Calculate basic translation coefficient (non-periodic).
"""
function trans_internal(clocob, cmob, clocso, cmso, lambda_wave, sp)
    k = sp["k0"]
    rv = clocob .- clocso
    x = rv[1]
    y = rv[2]
    r = norm(rv)
    # FIX: promote over clocob and lambda_wave for Dual compatibility
    RT_ti = promote_type(eltype(clocob), typeof(real(lambda_wave)))
    CT_ti = Complex{RT_ti}

    epsloc = get(sp, "epsloc", 1e-4)

    if r < epsloc
        return zero(CT_ti)
    end

    phip = atan(y, x)
    kr = k * r
    n = Int(cmob - cmso)

    if kr < 1e-8
        return zero(CT_ti)
    end

    h = hankelh2(n, kr)

    if isnan(h) || isinf(h)
        return zero(CT_ti)
    end

    t = h * exp(1im * (cmso - cmob) * (phip - π))

    if isnan(t) || isinf(t)
        return zero(CT_ti)
    end

    return t
end