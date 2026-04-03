"""
SystemInitialization
Coder: Curtis Jin
Date: 2011/MAR/30th WEDNESDAY
Contact: jsirius@umich.edu
Description: Initializing the system
Translated to Python
"""

import os
import scipy.io as sio


def system_initialization(handles):
    """
    Initialize the system - create default settings if they don't exist
    
    Parameters:
    -----------
    handles : dict
        Dictionary containing GUI widget references
    """
    # Delete temporary workspace if it exists
    filename = 'system_temporary_workspace.mat'
    if os.path.exists(filename):
        os.remove(filename)
    
    # Create settings directory if it doesn't exist
    if not os.path.exists('Settings'):
        os.makedirs('Settings')
    
    # Check if settings file exists, create if not
    filename = 'Settings/system_settings_workspace.mat'
    if not os.path.exists(filename):
        # Create default settings
        plot_setting_parameters = {
            'IntensityPlotResolution': 500,
            'PRReflectionY': 10,
            'PRReflectionX': 1,
            'PRTransmissionY': 10,
            'PRTransmissionX': 1,
            'DrawBufferLine': 0
        }
        
        algorithm_setting_parameters = {
            'CylinderType': 'Dielectric',
            'MinInterDistance': 2,
            'CylinderModeTol': 1e-4,
            'Buffer': 0,
            'EvanescentModeTol': 1e-4,
            'Interaction': 'On'
        }
        
        computational_setting_parameters = {
            'epsseries': 1e-11,
            'epsloc': 1e-4,
            'nrepeatSpatial': 5,
            'nrepeatSpectral': 3,
            'jmax': 1000,
            'kshanksSpatial': 3,
            'kshanksSpectral': -1,
            'spectral': 1,
            'spectralCondParameter': 120
        }
        
        # Save to .mat file
        sio.savemat(filename, {
            'PlotSettingParameters': plot_setting_parameters,
            'AlgorithmSettingParameters': algorithm_setting_parameters,
            'ComputationalSettingParameters': computational_setting_parameters
        })