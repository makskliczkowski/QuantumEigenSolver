'''
file: Algebra/hamil_types.py
This module contains the enumeration of Hamiltonian models and related functions.
A Hamiltonian model is defined as a class of Hamiltonians that share a common structure.

Date: 2025-02-01
'''

from enum import Enum, unique

# --- ED limits (using hexadecimal where applicable)
UI_LIMITS_MAXFULLED             = 0x40000
UI_LIMITS_MAXPRINT              = 0x8
UI_LIMITS_SI_STATENUM           = 100
UI_LIMITS_MIDDLE_SPEC_STATENUM  = 200

# ############################ EXISTING MODELS ############################
@unique
class Hamiltonians(Enum):
    '''
    Enumeration for different Hamiltonian models.
    '''
    NONE            = 0
    # interacting models:
    ISING_M         = 1                     # Quantum Ising model
    XYZ_M           = 2                     # XYZ model 
    HEI_KIT_M       = 3                     # Heisenberg model with Kitaev interaction
    QSM_M           = 4                     # Quantum Summer Model (QSM)
    RP_M            = 5                     # Rosenzweig-Porter Model
    ULTRAMETRIC_M   = 6                     # Ultrametric Model (UM)
    # quadratic (or noninteracting) models:
    FREE_FERMIONS_M = 100                   # Free Fermions Model
    AUBRY_ANDRE_M   = 101                   # Aubry-André Model
    SYK2_M          = 102                   # Sachdev-Ye-Kitaev (SYK) Model
    ANDERSON_M      = 103                   # Anderson Model
    POWER_LAW_RANDOM_BANDED_M = 104         # Power-law Random Banded Model

MY_MODELS_LIST = [
    Hamiltonians.ISING_M,
    Hamiltonians.XYZ_M,
    Hamiltonians.HEI_KIT_M,
    Hamiltonians.QSM_M,
    Hamiltonians.RP_M,
    Hamiltonians.ULTRAMETRIC_M,
    Hamiltonians.FREE_FERMIONS_M,
    Hamiltonians.AUBRY_ANDRE_M,
    Hamiltonians.SYK2_M,
    Hamiltonians.ANDERSON_M,
    Hamiltonians.POWER_LAW_RANDOM_BANDED_M,
    Hamiltonians.NONE,
]

# Maximum interacting model value defined as the value of ULTRAMETRIC_M.
MY_MODELS_MAX_INTERACTING = Hamiltonians.ULTRAMETRIC_M.value

################################################################################
#! TODO: Check the implementation of the following functions.
################################################################################

def check_noninteracting(model: Hamiltonians) -> bool:
    """
    Check if a model is noninteracting.
    
    In this implementation, a model is considered noninteracting if its value is
    greater than or equal to FREE_FERMIONS_M or if it is equal to RP_M.
    """
    return (model.value >= Hamiltonians.FREE_FERMIONS_M.value) or (model == Hamiltonians.RP_M)

def check_dense(model: Hamiltonians) -> bool:
    """
    Check if a model is dense.
    
    Dense models are defined as those corresponding to:
        - POWER_LAW_RANDOM_BANDED_M,
        - ULTRAMETRIC_M, or
        - RP_M.
    """
    return model in (Hamiltonians.POWER_LAW_RANDOM_BANDED_M, Hamiltonians.ULTRAMETRIC_M, Hamiltonians.RP_M)

################################################################################

from typing import Callable, List, Tuple, Optional
import numpy as np

def local_energy_int(k_map          : np.int64,
                    i               : int,
                    ns              : int,
                    operator_terms  : List[Tuple[Callable[[int, int, np.ndarray], Tuple[np.ndarray, np.ndarray]], np.ndarray, float]],
                    local_operators : Optional[List[Tuple[Callable[[int, int, int], Tuple[np.ndarray, np.ndarray]], float]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply operator terms to a state in an optimized manner.
    ----------
    k_map           : int
        The current state (for example, an integer representing a spin basis state).
    i               : int
        The site index for the operator terms. (i.e. the site where the operator acts locally)
    ns              : int
        The number of sites (modes) in the system.
    operator_terms  : list of tuples
        Each tuple is (operator_func, sites_array, multiplier). The operator_func must
        be vectorized (accepts state, ns, sites_array and returns (new_states, op_values)).
    local_operators : list of tuples, optional
        Each tuple is (operator_func, multiplier). For these operators the site argument
        will be taken from the local_site parameter.
    
    Returns
    -------
    new_states : np.ndarray
        Concatenated array of new states produced by each operator term.
    new_values : np.ndarray
        Concatenated array of corresponding operator values (each multiplied by its multiplier).
    """
    states_list = []
    values_list = []
    
    # Helper to process a general term.
    def process_term(term):
        '''
        Process a single operator term (general case).
        The operator_func must accept (k_map, ns, sites_array) and return (new_states, op_values).
        The sites_array is a numpy array of integers representing the sites (modes) where the operator acts.
        Returns the new states and operator values.
        Each term is a tuple of (operator_func, sites_array, multiplier).
        '''
        op_func, sites_array, multiplier    = term
        new_states, op_values               = op_func(k_map, ns, sites_array)
        return new_states, multiplier * op_values
    
    # Process operators
    for term in operator_terms:
        new_states, op_values = process_term(term)
        states_list.append(new_states)
        values_list.append(op_values)
    
    # Process local operators if provided - do it separately
    if local_operators is not None:
        local_value = np.array([0], dtype=local_operators[0][1].dtype)
        def process_local_term(term):
            '''
            Process a single local operator term.
            Returns the new states and operator values.
            Each term is a tuple of (operator_func, multiplier).
            
            Remmember that the state is unchanged in this case.
            The operator_func must accept (k_map, ns, i) and return (new_states, op_values).
            '''
            op_func, multiplier     = term
            _, op_values            = op_func(k_map, ns, i)
            return multiplier * op_values

        # Process local operators
        for term in local_operators:
            local_value += process_local_term(term)
        
        # Append local operator values to the list
        if local_value is not None:
            values_list.append(local_value)
            states_list.append(np.array([k_map]))
            
    # Concatenate the resulting arrays
    if states_list:
        return np.concatenate(states_list), np.concatenate(values_list)
    else:
        return np.array([]), np.array([])

################################################################################