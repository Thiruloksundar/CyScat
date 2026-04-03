"""
Scattering_Code Package
Core scattering matrix computation modules

Original MATLAB code by Curtis Jin (jsirius@umich.edu)
Translated to Python
GPU-accelerated version using CuPy when available
"""

# GPU backend (import first to set up xp module)
from . import gpu_backend

# Import main functions for easy access
from .smatrix import smatrix
from .smatrix_parameters import smatrix_parameters
from .smatrix_cascade import smatrix_cascade
from .sall import sall
from .vall import vall
from .transall import transall
from .scattering_coefficients_all import scatteringcoefficientsall
from .ky import ky
from .modified_epsilon_shanks import modified_epsilon_shanks
from .simulation_time_profile import simulation_time_profile
from .distmat import distmat

# Field calculation functions
from .efieldall import efieldall
from .farefieldall import farefieldall
from .efieldallperiodic import efieldallperiodic
from .coefficients import coefficients

# Additional utilities
from .cascadertwo import cascadertwo
from .truncator import truncator
from .transper import transper

__all__ = [
    'gpu_backend',
    'smatrix',
    'smatrix_parameters',
    'smatrix_cascade',
    'sall',
    'vall',
    'transall',
    'scatteringcoefficientsall',
    'ky',
    'modified_epsilon_shanks',
    'simulation_time_profile',
    'distmat',
    'efieldall',
    'farefieldall',
    'efieldallperiodic',
    'coefficients',
    'cascadertwo',
    'truncator',
    'transper',
]