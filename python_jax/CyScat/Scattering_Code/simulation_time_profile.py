"""
SimulationTimeProfile Code
Coder: Curtis Jin
Date: 2011/JAN/28th Friday
Contact: jsirius@umich.edu
Description: Constructor for a simulation profile of S-Matrix generating code
Translated to Python
"""


def simulation_time_profile(st_tmatrix, st_lu, st_up, st_down, st_norm):
    """
    Create simulation time profile structure
    
    Parameters:
    -----------
    st_tmatrix : float
        Simulation time for T-Matrix generation
    st_lu : float
        Simulation time for LU Decomposition
    st_up : float
        Simulation time for computing S11&S21 Partition
    st_down : float
        Simulation time for computing S22&S12 Partition
    st_norm : float
        Simulation time for Normalizing the Scattering Matrix
        
    Returns:
    --------
    STP : dict
        Dictionary containing simulation time profile
    """
    tst = st_tmatrix + st_lu + st_up + st_down + st_norm
    
    # Generate SimulationProfile Structure
    STP = {
        'STTMatrix': st_tmatrix,  # Simulation Time for T-Matrix generation
        'STLU': st_lu,            # Simulation Time for LU Decomposition
        'STUp': st_up,            # Simulation Time for computing S11&S21 Partition
        'STDown': st_down,        # Simulation Time for computing S22&S12 Partition
        'STNorm': st_norm,        # Simulation Time for Normalizing the Scattering Matrix
        'TST': tst                # Total Simulation Time
    }
    
    return STP