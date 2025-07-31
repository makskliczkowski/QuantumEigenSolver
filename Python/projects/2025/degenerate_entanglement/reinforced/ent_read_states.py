import argparse
import numpy as np
from typing import List, Tuple, Dict, Any

def parse_arguments():
    """Parse command line arguments for quantum state loading."""
    parser          = argparse.ArgumentParser(description="Load quantum states from HDF5 files")
    d_dir           = "./data/hamiltonian/random/states/"
    s_dir           = "./data/hamiltonian/random/agent/"
    
    # Required parameters
    parser.add_argument("--L",          type=int,       default=6,      help="System size L")
    parser.add_argument("--gamma",      type=float,     default=6,      help="Gamma parameter")
    parser.add_argument("--r",          type=int,       default=42,     help="Random seed parameter r")
    
    # Optional parameters with defaults
    parser.add_argument("--data_dir",   type=str,       default=d_dir,  help=f"Base data directory (default: '{d_dir}')")
    parser.add_argument("--is_complex", type=int,       default=0,      help="Use complex coefficients (default: False)")
    parser.add_argument("--n_states",   type=int,       default=1,      help="Number of states to load (default: 1)")
    parser.add_argument("--lr",         type=float,     default=0.01,   help="Learning rate for optimization (default: 0.01)")
    parser.add_argument("--n_steps",    type=int,       default=1500,   help="Number of optimization steps (default: 1500)")
    
    # Optional for unitary agent
    parser.add_argument("--k",          type=int,       default=8,      help="Number of rotations (default: 8)")
    parser.add_argument("--unitary",    type=int,       default=0,      help="Use unitary agent (default: False)")
    parser.add_argument("--max_steps",  type=int,       default=100,    help="Maximum steps per episode (default: 100)")
    parser.add_argument("--upd_freq",   type=int,       default=10,     help="Update frequency (default: 10)")
    parser.add_argument("--sav_freq",   type=int,       default=100,    help="Save frequency (default: 100)")
    parser.add_argument("--batch_size", type=int,       default=32,     help="Batch size for training (default: 32)")
    parser.add_argument("--sav_dir",    type=str,       default=s_dir,  help=f"Directory to save agent data (default: '{s_dir}')")
    parser.add_argument("--ent_thr",    type=float,     default=0.1,    help="Entanglement threshold for training (default: 0.1)")
    
    parser.add_argument("--ent_weight", type=float,     default=1.0,    help="Weight for entropy in loss function (default: 1.0)")
    parser.add_argument("--pur_weight", type=float,     default=0.0,    help="Weight for purity in loss function (default: 0.0)")
    parser.add_argument("--rnk_weight", type=float,     default=0.0,    help="Weight for rank in loss function (default: 0.0)")
    parser.add_argument("--rot_weight", type=float,     default=0.0,    help="Weight for rotations in loss function (default: 0.0)")
    parser.add_argument("--non_weight", type=float,     default=0.0,    help="Weight for non-gaussianity in loss function (default: 0.0)")
    
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
                    logger          : Any = None) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float], Dict[str, Any]]:
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
            mix_states_real.append(np.asarray(np.real(mix_states), dtype=np.float32))
            mix_entropies_real.append(mix_entropy_data[f"/eigvec/{i}/mixture/c/{mixture_index}/entropy"])
            
            # Log the loaded mixture states
            logger.info(f"Loaded mixture states for eigvec {i}: shape={mix_states_real[i].shape}, entropy={mix_entropies_real[i]}", color="green", lvl=2)
                
        except Exception as e:
            logger.error(f"Error loading mixture state {i}: {e}", color="red", lvl=1)
            raise

    return (org_states, org_entropies), (mix_states_real, mix_entropies_real), system_params