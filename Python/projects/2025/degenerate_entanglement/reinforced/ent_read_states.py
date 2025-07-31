import argparse
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

def parse_arguments():
    """Parse command line arguments for quantum state loading."""
    parser = argparse.ArgumentParser(description="Load quantum states from HDF5 files")
    
    # Required parameters
    parser.add_argument("--L",      type=int,   required=True, help="System size L")
    parser.add_argument("--gamma",  type=float, required=True, help="Gamma parameter")
    parser.add_argument("--r",      type=int,   required=True, help="Random seed parameter r")
    
    # Optional parameters with defaults
    parser.add_argument("--data_dir",   type=str,               default="data", help="Base data directory (default: 'data')")
    parser.add_argument("--is_complex", action="store_true",    default=False,  help="Use complex coefficients (default: False)")
    parser.add_argument("--n_states",   type=int,               default=1,      help="Number of states to load (default: 1)")
    parser.add_argument("--lr",         type=float,             default=0.01,   help="Learning rate for optimization (default: 0.01)")
    parser.add_argument("--n_steps",    type=int,               default=1500,   help="Number of optimization steps (default: 1500)")
    
    # Very optional parameters
    parser.add_argument("--verbose_every", type=int, default=50, help="Print verbose output every N steps (default: 50)")
    parser.add_argument("--mixture_index", type=int, default=0, help="Mixture index to load (default: 0)")

    return parser.parse_args()


def load_quantum_states(L           : int, 
                    gamma           : float, 
                    r               : int, 
                    n_states        : int = 1, 
                    data_dir        : str = "data", 
                    mixture_index   : int = 0,
                    logger          : Any = None) -> Tuple[List[jnp.ndarray], List[float], List[jnp.ndarray], List[float], Dict[str, Any]]:
    """
    Load quantum states from HDF5 file.
    
    Parameters:
    - L: System size
    - gamma: Gamma parameter
    - r: Random seed parameter
    - n_states: Number of states to load
    - data_dir: Base data directory
    - mixture_index: Index of mixture to load
    - logger: Logger instance (optional)
    
    Returns:
    - org_states: List of original state matrices
    - org_entropies: List of original entropies
    - mix_states_real: List of mixture state matrices (real parts)
    - mix_entropies_real: List of mixture entropies
    - system_params: Dictionary of system parameters
    """
    try:
        from QES.general_python.common import HDF5Handler
        from QES.general_python.common import Directories
        from QES.general_python.common import flog
        
        if logger is None:
            logger = flog.get_global_logger()

    except ImportError as e:
        print(f"Warning: Could not import QES modules: {e}")
        raise NotImplementedError("QES modules not available. Implement fallback HDF5 loading.")
    
    # Calculate system parameters
    La      = L // 2
    dimA    = 2**La
    dimB    = 2**(L - La)
    
    system_params = {
        'L'         : int(L),
        'gamma'     : int(gamma),
        'r'         : int(r),
        'La'        : int(La),
        'dimA'      : int(dimA),
        'dimB'      : int(dimB),
        'n_states'  : int(n_states)
    }
    
    logger.info(f"System parameters: {system_params}", color="blue", lvl=1)
    
    # Setup directory and filename
    directory_save  = Directories(data_dir)
    filename        = f"L={int(L)},Gamma={int(gamma)},r={int(r)}.h5"

    logger.info(f"Filename: {filename}", color="blue", lvl=1)
    logger.info(f"Directory: {directory_save}", color="blue", lvl=1)

    # Load original states and entropies
    org_states = []
    org_entropies = []
    
    for i in range(n_states):
        try:
            state_data = HDF5Handler.read_hdf5(directory_save / filename, keys=[f"/eigvec/{i}/states"], verbose=True, logger=logger)
            entropy_data = HDF5Handler.read_hdf5(directory_save / filename, keys=[f"/eigvec/{i}/entropy"])
            org_states.append(state_data[f"/eigvec/{i}/states"])
            org_entropies.append(entropy_data[f"/eigvec/{i}/entropy"])

            logger.info(f"Loaded states for eigvec {i}: shape={org_states[i].shape}, entropy={org_entropies[i]}", 
                color="blue", lvl=2)

        except Exception as e:
            logger.error(f"Error loading original state {i}: {e}", color="red", lvl=1)
            raise
    
    # Load mixture states (real parts)
    mix_states_real     = []
    mix_entropies_real  = []

    for i in range(n_states):
        try:
            mix_state_data = HDF5Handler.read_hdf5(directory_save / filename, keys=[f"/eigvec/{i}/mixture/c/{mixture_index}/states"])
            mix_entropy_data = HDF5Handler.read_hdf5(directory_save / filename, keys=[f"/eigvec/{i}/mixture/c/{mixture_index}/entropy"])
            
            # Extract real parts and convert to JAX arrays
            mix_states = mix_state_data[f"/eigvec/{i}/mixture/c/{mixture_index}/states"]
            mix_states_real.append(jnp.asarray(np.real(mix_states), dtype=jnp.float32))
            mix_entropies_real.append(mix_entropy_data[f"/eigvec/{i}/mixture/c/{mixture_index}/entropy"])
            
            # Log the loaded mixture states
            logger.info(f"Loaded mixture states for eigvec {i}: shape={mix_states_real[i].shape}, entropy={mix_entropies_real[i]}", color="green", lvl=2)
                
        except Exception as e:
            logger.error(f"Error loading mixture state {i}: {e}", color="red", lvl=1)
            raise

    return (org_states, org_entropies), (mix_states_real, mix_entropies_real), system_params