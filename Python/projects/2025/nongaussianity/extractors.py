from ctypes import Union
from enum import Enum
import os, sys
from typing import Optional

#! -----
os.environ['BACKEND'] = 'numpy'
#! -----

# Add parent directories to sys.path for imports
for _ in range(5):
    parent = os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * (_ + 1))))
    if parent not in sys.path:
        sys.path.append(parent)

# ---------------------------------------------------------------------
import numpy as np
import time

#! General Python imports
try:
    from QES.general_python.common import flog, get_global_logger
    from QES.general_python.algebra.ran_wrapper import set_global_seed
except ImportError:
    raise ImportError("QES.general_python module is not available. Please ensure QES is installed correctly.")

#! QES imports
try:
    from QES.Algebra.Model.Noninteracting import syk
    from QES.Algebra.Model.Noninteracting.Conserving import free_fermions
except ImportError:
    raise ImportError("QES modules are not available. Please ensure QES is installed correctly.")

#! Global logger
logger = get_global_logger()

# ---------------------------------------------------------------------

class ModelHamiltonians(Enum):
    SYK2            = 'SYK2'
    FreeFermions    = 'FreeFermions'
    
def create_hamiltonian(ns           : int, 
                    model_type      : Union[str, ModelHamiltonians], 
                    t               : float = 1.0,
                    dtype           : type = np.float64, backend: str = 'numpy',
                    seed            : Optional[int] = None,
                    ):
    if model_type == ModelHamiltonians.SYK2:
        return syk.SYK2(ns, dtype=dtype, backend=backend, seed=seed, logger=logger)
    elif model_type == ModelHamiltonians.FreeFermions:
        return free_fermions.FreeFermions(ns, t, dtype=dtype, backend=backend, logger=logger, seed=seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")