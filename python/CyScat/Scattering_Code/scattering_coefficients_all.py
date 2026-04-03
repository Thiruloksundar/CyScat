"""
scatteringcoefficientsall Code
Coder: Curtis Jin
Date: 2010/DEC/3rd Friday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : Entries of the Scattering Matrix Generating Code
Translated to Python
"""

import numpy as np


def scatteringcoefficientsall(clocs, cmmaxs, period, lambda_wave, nmax, 
                              current_coefficients, up_down, d, sp):
    """
    Generate scattering coefficients for all cylinders
    
    Parameters:
    -----------
    clocs : ndarray
        Cylinder locations (Nx2 array)
    cmmaxs : ndarray
        Maximum mode numbers for each cylinder
    period : float
        Period of the structure
    lambda_wave : float
        Wavelength
    nmax : int
        Maximum mode number for plane waves
    current_coefficients : ndarray
        Current cylinder coefficients
    up_down : int
        Direction flag (1 for up, -1 for down)
    d : float
        Distance/thickness
    sp : dict
        SMatrix parameters
        
    Returns:
    --------
    scattering_coefficient_list : ndarray
        List of scattering coefficients (2*nmax+1 x 2)
    """
    scattering_coefficient_list = np.zeros((2*nmax + 1, 2), dtype=complex)
    max_mode_index = int(np.max(cmmaxs))
    
    for cm in range(-max_mode_index, max_mode_index + 1):
        # Find cylinders that have this mode
        cylinder_index = np.where(cmmaxs >= abs(cm))[0]
        
        # Calculate coefficient indices
        coefficient_index = []
        for j in range(len(cylinder_index)):
            cyl_idx = cylinder_index[j]
            idx = int(np.sum(2*cmmaxs[:cyl_idx] + 1) + cmmaxs[cyl_idx] + cm)
            coefficient_index.append(idx)
        
        coefficient_index = np.array(coefficient_index, dtype=int)
        
        # Add contribution from this mode
        scattering_coefficient_list += scatteringcoefficients_grid(
            clocs[cylinder_index, :], cm, period, lambda_wave, nmax,
            current_coefficients[coefficient_index], up_down, d, sp
        )
    
    return scattering_coefficient_list


def scatteringcoefficients_grid(clocs, cm, period, lambda_wave, nmax, 
                               current_coefficients, up_down, d, sp):
    """
    Calculate scattering coefficients on a grid
    
    Parameters:
    -----------
    clocs : ndarray
        Cylinder locations for selected cylinders
    cm : int
        Mode number
    period : float
        Period of the structure
    lambda_wave : float
        Wavelength
    nmax : int
        Maximum mode number
    current_coefficients : ndarray
        Current coefficients for selected cylinders
    up_down : int
        Direction flag
    d : float
        Distance/thickness
    sp : dict
        SMatrix parameters
        
    Returns:
    --------
    scattering_coefficient_list : ndarray
        Scattering coefficients (2*nmax+1 x 2)
    """
    index1 = sp['MiddleIndex'] - nmax
    index_end = sp['MiddleIndex'] + nmax
    index = np.arange(index1, index_end + 1)
    
    # Pre-calculate matrices
    kxs_index = sp['kxs'][index]
    kys_index = sp['kys'][index]
    angles_index = sp['Angles'][index]
    
    # Calculate dot products
    kxs_clocx = kxs_index @ clocs[:, 0:1].T
    kys_clocy = kys_index @ clocs[:, 1:2].T
    kys_clocy_minus_d = kys_index @ (clocs[:, 1:2] - d).T
    kys_clocy_plus_d = kys_index @ (clocs[:, 1:2] + d).T
    
    # Create angle mesh
    junk, angles_mesh = np.meshgrid(current_coefficients, angles_index)
    
    # Calculate coefficient
    # MATLAB: periodOVERkysCOEFF = sp.TwoOverPeriod ./ sp.kys(Index) * transpose(currentcoefficients);
    # This is: (scalar ./ column_vector) * row_vector → matrix of shape (len(kys), len(coeff))
    # Protect against division by zero for grazing incidence modes
    eps_ky = 1e-10
    kys_safe = kys_index.copy()
    near_zero_mask = np.abs(kys_safe) < eps_ky
    if np.any(near_zero_mask):
        kys_safe = np.where(near_zero_mask, eps_ky + 0j, kys_safe)
    period_over_kys_coeff = (sp['TwoOverPeriod'] / kys_safe.reshape(-1, 1)) @ current_coefficients.reshape(1, -1)
    
    scattering_coefficient_list = scatteringcoefficients_m_matrix(
        clocs, cm, period, lambda_wave, nmax, current_coefficients, up_down, d, sp,
        kxs_clocx, kys_clocy, kys_clocy_minus_d, kys_clocy_plus_d, 
        angles_mesh, period_over_kys_coeff
    )
    
    return scattering_coefficient_list


def scatteringcoefficients_m_matrix(clocs, cm, period, lambda_wave, nmax, 
                                   current_coefficient, up_down, d, sp,
                                   kxs_clocx, kys_clocy, kys_clocy_minus_d, 
                                   kys_clocy_plus_d, angles, period_over_kys_coeff):
    """
    Calculate scattering coefficients matrix
    
    Parameters:
    -----------
    All pre-calculated matrices and parameters
        
    Returns:
    --------
    s : ndarray
        Scattering coefficients [s11, s21] or [s12, s22]
    """
    # S11 & S21 Partition
    if up_down > 0:
        # S11 Partition
        exponent = kxs_clocx - kys_clocy + cm * (angles + np.pi)
        s11 = (-1)**cm * np.exp(1j * exponent) * period_over_kys_coeff
        s11 = np.sum(s11, axis=1, keepdims=True)
        
        # S21 Partition
        exponent = kxs_clocx + kys_clocy_minus_d - cm * (angles - np.pi)
        s21 = np.exp(1j * exponent) * period_over_kys_coeff
        s21 = np.sum(s21, axis=1, keepdims=True)
        
        s = np.hstack([s11, s21])
    
    # S22 & S12 Partition
    else:
        # S12 Partition
        exponent = kxs_clocx - kys_clocy_plus_d + cm * (angles + np.pi)
        s12 = (-1)**cm * np.exp(1j * exponent) * period_over_kys_coeff
        s12 = np.sum(s12, axis=1, keepdims=True)
        
        # S22 Partition
        exponent = kxs_clocx + kys_clocy - cm * (angles - np.pi)
        s22 = np.exp(1j * exponent) * period_over_kys_coeff
        s22 = np.sum(s22, axis=1, keepdims=True)
        
        s = np.hstack([s12, s22])
    
    return s