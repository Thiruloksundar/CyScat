"""
PositionGenerator
Coder: Curtis Jin
Date: 2011/MAR/31st Thursday
Contact: jsirius@umich.edu
Description: Generates Position
Translated to Python
"""

import numpy as np
import os
import scipy.io as sio
from scipy.stats import qmc
from Scattering_Code.distmat import distmat


def position_generator(GP, optional_filename=None):
    """
    Generate cylinder positions based on various methods
    
    Parameters:
    -----------
    GP : dict
        Global Parameters dictionary
    optional_filename : str, optional
        Optional filename for loading custom or saved positions
        
    Returns:
    --------
    modified_no_cylinders : int
        Final number of cylinders after removing overlaps
    initial_positions : ndarray
        Initial normalized positions (0-1 range)
    real_positions : ndarray
        Real positions in physical coordinates
    """
    
    no_cylinders = GP['NoCylinders']
    radius = GP['Radius']
    width = GP['Width']
    thickness = GP['Thickness']
    random_set = GP['RandomSet']
    rand_factor = GP['RandomFactor']
    min_inter_distance = GP['MinInterDistance']
    
    # Modulating the Position of the scatterer
    if random_set == 'Custom':
        print('Generating Particles(Custom)...')
        
        if not optional_filename:
            print('Please specify a file name!')
            show_error_message(
                ['You are in "Custom" mode.',
                 'Please specify a file name which contains the "InitialPositions" vector.'],
                'No file error!', 'error'
            )
            return 0, None, None
        
        if not os.path.exists(optional_filename):
            print(f'{optional_filename} does not exist!')
            show_error_message(
                [f'{optional_filename} does not exist!',
                 'Please enter the correct workspace file.'],
                'No matching file error!', 'error'
            )
            return 0, None, None
        
        print(f'Loading {optional_filename}...')
        mat_data = sio.loadmat(optional_filename)
        
        if 'InitialPositions' not in mat_data:
            show_error_message(
                ['You are in "Custom" mode.',
                 f'{optional_filename} does not contain "InitialPositions" structure.',
                 'Please enter a workspace file that contains "InitialPositions" structure,',
                 'or Regenerate the S-Matrix.'],
                'Wrong workspace file error!', 'error'
            )
            return 0, None, None
        
        positions = mat_data['InitialPositions']
    
    elif random_set == 'Latin':
        print('Generating Particles(Latin Hypercube)...')
        sampler = qmc.LatinHypercube(d=2)
        positions = sampler.random(n=no_cylinders)
    
    elif random_set == 'Sobol':
        print('Generating Particles(Sobol)...')
        sampler = qmc.Sobol(d=2)
        positions = sampler.random(n=no_cylinders)
    
    elif random_set == 'Halt':
        print('Generating Particles(Halton)...')
        sampler = qmc.Halton(d=2)
        positions = sampler.random(n=no_cylinders)
    
    elif random_set == 'RandPertDet':
        # Third method: Random perturbation of deterministic grid
        sqrt_cylinders = int(np.fix(np.sqrt(no_cylinders)))
        d = width
        a = radius
        
        X = np.linspace(0, 1, sqrt_cylinders + 1)[:-1]
        C = X + 1 / sqrt_cylinders / 2
        Cx, Cy = np.meshgrid(C, C)
        
        buffer = (width - 2*a) / sqrt_cylinders / 2 - a
        coeff = buffer * rand_factor
        coeff = coeff / (width - 2*a)
        
        Cx = Cx.flatten() + coeff * 2 * (np.random.rand(sqrt_cylinders**2) - 0.5)
        Cy = Cy.flatten() + coeff * 2 * (np.random.rand(sqrt_cylinders**2) - 0.5)
        
        initial_positions = np.column_stack([Cx, Cy])
        positions = initial_positions
        no_cylinders = sqrt_cylinders**2
    
    elif random_set == 'Load':
        if not optional_filename:
            show_error_message(
                ['You are in "Load" mode.',
                 'Please specify a file name of the workspace you want to use.'],
                'No file error!', 'error'
            )
            return 0, None, None
        
        if not os.path.exists(optional_filename):
            show_error_message(
                [f'{optional_filename} does not exist!',
                 'Please enter the correct workspace file.'],
                'No matching file error!', 'error'
            )
            return 0, None, None
        
        mat_data = sio.loadmat(optional_filename)
        
        if 'parameters' not in mat_data:
            show_error_message(
                [f'{optional_filename} does not contain "parameters" structure.',
                 'Please enter a workspace file that contains "parameters" structure,',
                 'or Regenerate the S-Matrix.'],
                'Wrong workspace file error!', 'error'
            )
            return 0, None, None
        
        if 'SMatrixData' not in mat_data:
            show_error_message(
                [f'{optional_filename} does not contain "SMatrixData" structure.',
                 'Please enter a workspace file that contains "SMatrixData" structure,',
                 'or Regenerate the S-Matrix.'],
                'Wrong workspace file error!', 'error'
            )
            return 0, None, None
        
        if 'RealPositions' not in mat_data:
            show_error_message(
                [f'{optional_filename} does not contain "RealPositions" structure.',
                 'Please enter a workspace file that contains "RealPositions" structure,',
                 'or Regenerate the S-Matrix.'],
                'Wrong workspace file error!', 'error'
            )
            return 0, None, None
        
        return 1, None, None
    
    else:
        print('Please specify a correct RandomSet type')
        return 0, None, None
    
    print(f'Initial number of cylinders: {no_cylinders}')
    
    a = radius
    Lx = width
    Ly = thickness
    
    # Scaling positions
    temp_positions = np.zeros_like(positions)
    temp_positions[:, 0] = positions[:, 0] * (Lx - 2*a)
    temp_positions[:, 1] = positions[:, 1] * (Ly - 2*a)
    
    # Shifting positions
    temp_positions = temp_positions + a
    
    if no_cylinders > 1:
        # Identifying Overlapping Cylinders
        dmat, _ = distmat(temp_positions)
        no_overlapping = 0
        overlapping_list = np.zeros(no_cylinders)
        
        for idx1 in range(no_cylinders):
            for idx2 in range(idx1+1, no_cylinders):
                if dmat[idx1, idx2] < min_inter_distance:
                    dmat[idx1, idx2] = 1
                    dmat[idx2, idx1] = 1
                    overlapping_list[idx1] = 1
                    overlapping_list[idx2] = 1
                    no_overlapping += 1
                else:
                    dmat[idx1, idx2] = 0
        
        # Removing the overlapping parameters
        print('Removing overlapping cylinders...')
        real_positions = []
        initial_positions_clean = []
        
        for idx in range(len(overlapping_list)):
            if overlapping_list[idx] == 0:
                real_positions.append(temp_positions[idx])
                initial_positions_clean.append(positions[idx])
        
        real_positions = np.array(real_positions)
        initial_positions = np.array(initial_positions_clean)
        modified_no_cylinders = len(real_positions)
        
        print(f'Final number of cylinders: {modified_no_cylinders}')
        
        if modified_no_cylinders == 0:
            show_error_message(
                ['All the cylinders are overlapping!!',
                 'Change the configuration and try it again.'],
                'No cylinder error!', 'error'
            )
            return 0, None, None
        
        # Sorting the positions in the order of distance from the origin
        r = np.sqrt(real_positions[:, 0]**2 + real_positions[:, 1]**2)
        sorted_indices = np.argsort(r)
        real_positions = real_positions[sorted_indices]
        initial_positions = initial_positions[sorted_indices]
    
    else:
        # Single cylinder case
        modified_no_cylinders = 1
        real_positions = temp_positions
        initial_positions = positions
    
    return modified_no_cylinders, initial_positions, real_positions


def show_error_message(messages, title, msg_type):
    """
    Display error message (placeholder for GUI message box)
    
    Parameters:
    -----------
    messages : list of str
        Error messages
    title : str
        Message box title
    msg_type : str
        Type of message ('error', 'warn', etc.)
    """
    print(f"[{msg_type.upper()}] {title}:")
    for msg in messages:
        print(f"  {msg}")