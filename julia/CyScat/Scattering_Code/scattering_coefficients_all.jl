"""
scatteringcoefficientsall Code
Coder: Curtis Jin
Date: 2010/DEC/3rd Friday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : Entries of the Scattering Matrix Generating Code

Julia translation - directly from MATLAB
"""

"""
    scatteringcoefficientsall(clocs, cmmaxs, period, lambda_wave, nmax,
                              current_coefficients, up_down, d, sp)

Generate scattering coefficients for all cylinders.
Exact translation from MATLAB scatteringcoefficientsall.m
"""
function scatteringcoefficientsall(clocs, cmmaxs, period, lambda_wave, nmax,
                                   current_coefficients, up_down, d, sp)
    # current_coefficients is already complex; use its element type directly
    CT = eltype(current_coefficients)
    scattering_coefficient_list = zeros(CT, 2*nmax + 1, 2)
    max_mode_index = Int(maximum(cmmaxs))

    for cm in -max_mode_index:max_mode_index
        # MATLAB: CylinderIndex = find(cmmaxs >= abs(cm));
        cylinder_index = findall(cmmaxs .>= abs(cm))

        # MATLAB: CoefficientIndex = zeros(length(CylinderIndex),1);
        coefficient_index = zeros(Int, length(cylinder_index))

        for j in 1:length(cylinder_index)
            # MATLAB: CoefficientIndex(j) = sum(2*cmmaxs(1:CylinderIndex(j)-1)+1)+cmmaxs(CylinderIndex(j))+1+cm;
            cyl_idx = cylinder_index[j]
            if cyl_idx == 1
                coefficient_index[j] = Int(cmmaxs[cyl_idx]) + 1 + cm
            else
                coefficient_index[j] = Int(sum(2 .* cmmaxs[1:cyl_idx-1] .+ 1)) + Int(cmmaxs[cyl_idx]) + 1 + cm
            end
        end

        # Add contribution from this mode
        scattering_coefficient_list .+= scatteringcoefficients_grid(
            clocs[cylinder_index, :], cm, period, lambda_wave, nmax,
            current_coefficients[coefficient_index], up_down, d, sp
        )
    end

    return scattering_coefficient_list
end


"""
    scatteringcoefficients_grid(clocs, cm, period, lambda_wave, nmax,
                                current_coefficients, up_down, d, sp)

Calculate scattering coefficients on a grid.
Exact translation from MATLAB scatteringcoefficientsGrid
"""
function scatteringcoefficients_grid(clocs, cm, period, lambda_wave, nmax,
                                     current_coefficients, up_down, d, sp)
    # MATLAB: Index1 = sp.MiddleIndex-nmax;
    # MATLAB: IndexEnd = sp.MiddleIndex+nmax;
    # MATLAB: Index = Index1:IndexEnd;
    index1 = sp["MiddleIndex"] - nmax
    index_end = sp["MiddleIndex"] + nmax
    index = index1:index_end

    # Ensure clocs is a matrix even for single cylinder
    if ndims(clocs) == 1
        clocs = reshape(clocs, 1, :)
    end

    # Ensure current_coefficients is a column vector
    current_coefficients = vec(current_coefficients)

    # Get the indexed arrays as column vectors (matching MATLAB)
    # In MATLAB, sp.kxs(Index) returns a column vector
    # We need to extract and reshape to column
    kxs_idx = vec(sp["kxs"][index, :])  # (n_modes,) column vector
    kys_idx = vec(sp["kys"][index, :])  # (n_modes,) column vector
    angles_idx = vec(sp["Angles"][index, :])  # (n_modes,) column vector

    # MATLAB: kxsclocx = sp.kxs(Index)*clocs(:,1)';
    # (n_modes,) * (1, n_cyl) -> need to reshape for proper matrix mult
    # In MATLAB: (7,1) * (1,3) = (7,3)
    # NOTE: Use transpose(), not ' (adjoint) to avoid complex conjugation
    kxs_clocx = kxs_idx * transpose(clocs[:, 1])

    # MATLAB: kysclocy = sp.kys(Index)*clocs(:,2)';
    kys_clocy = kys_idx * transpose(clocs[:, 2])

    # MATLAB: kysclocyminusd = sp.kys(Index)*(clocs(:,2)-d)';
    kys_clocy_minus_d = kys_idx * transpose(clocs[:, 2] .- d)

    # MATLAB: kysclocyplusd = sp.kys(Index)*(clocs(:,2)+d)';
    kys_clocy_plus_d = kys_idx * transpose(clocs[:, 2] .+ d)

    # MATLAB: [junk Angles] = meshgrid(currentcoefficients,sp.Angles(Index));
    # meshgrid(A, B) in MATLAB: X replicates A along rows, Y replicates B along columns
    # For Y (Angles): each column has sp.Angles(Index) values
    # Shape: (length(angles_idx), length(current_coefficients)) = (n_modes, n_cyl)
    n_modes = length(angles_idx)
    n_cyl = length(current_coefficients)
    angles_mesh = repeat(angles_idx, 1, n_cyl)  # Each column is angles_idx

    # MATLAB: periodOVERkysCOEFF = sp.TwoOverPeriod ./ sp.kys(Index) * transpose(currentcoefficients);
    # sp.TwoOverPeriod ./ sp.kys(Index) = scalar ./ (n_modes,) = (n_modes,)
    # (n_modes,) * (1, n_cyl) = (n_modes, n_cyl)
    # NOTE: Use transpose(), not ' (adjoint) to avoid complex conjugation of coefficients!
    period_over_kys = sp["TwoOverPeriod"] ./ kys_idx
    period_over_kys_coeff = period_over_kys * transpose(current_coefficients)

    scattering_coefficient_list = scatteringcoefficients_m_matrix(
        clocs, cm, period, lambda_wave, nmax, current_coefficients, up_down, d, sp,
        kxs_clocx, kys_clocy, kys_clocy_minus_d, kys_clocy_plus_d,
        angles_mesh, period_over_kys_coeff
    )

    return scattering_coefficient_list
end


"""
    scatteringcoefficients_m_matrix(...)

Calculate scattering coefficients matrix.
Exact translation from MATLAB scatteringcoefficientsmMatrix
"""
function scatteringcoefficients_m_matrix(clocs, cm, period, lambda_wave, nmax,
                                         current_coefficient, up_down, d, sp,
                                         kxs_clocx, kys_clocy, kys_clocy_minus_d,
                                         kys_clocy_plus_d, angles, period_over_kys_coeff)
    # S11 & S21 Partition
    if up_down > 0
        # MATLAB: exponent = kxsclocx - kysclocy + cm*(Angles+pi);
        exponent = kxs_clocx .- kys_clocy .+ cm .* (angles .+ π)

        # MATLAB: s11 = (-1).^cm .* exp(1i*exponent).*periodOVERkysCOEFF;
        s11 = ((-1.0)^cm) .* exp.(1im .* exponent) .* period_over_kys_coeff

        # MATLAB: s11 = sum(s11,2);
        s11 = sum(s11, dims=2)

        # MATLAB: exponent = kxsclocx + kysclocyminusd - cm*(Angles-pi);
        exponent = kxs_clocx .+ kys_clocy_minus_d .- cm .* (angles .- π)

        # MATLAB: s21 = exp(1i*exponent).*periodOVERkysCOEFF;
        s21 = exp.(1im .* exponent) .* period_over_kys_coeff

        # MATLAB: s21 = sum(s21,2);
        s21 = sum(s21, dims=2)

        # MATLAB: s = [s11,s21];
        s = hcat(vec(s11), vec(s21))

    # S22 & S12 Partition
    else
        # MATLAB: exponent = kxsclocx - kysclocyplusd + cm*(Angles+pi);
        exponent = kxs_clocx .- kys_clocy_plus_d .+ cm .* (angles .+ π)

        # MATLAB: s12 = (-1).^cm .* exp(1i*exponent).*periodOVERkysCOEFF;
        s12 = ((-1.0)^cm) .* exp.(1im .* exponent) .* period_over_kys_coeff

        # MATLAB: s12 = sum(s12,2);
        s12 = sum(s12, dims=2)

        # MATLAB: exponent = kxsclocx + kysclocy - cm*(Angles-pi);
        exponent = kxs_clocx .+ kys_clocy .- cm .* (angles .- π)

        # MATLAB: s22 = exp(1i*exponent).*periodOVERkysCOEFF;
        s22 = exp.(1im .* exponent) .* period_over_kys_coeff

        # MATLAB: s22 = sum(s22,2);
        s22 = sum(s22, dims=2)

        # MATLAB: s = [s12,s22];
        s = hcat(vec(s12), vec(s22))
    end

    return s
end
