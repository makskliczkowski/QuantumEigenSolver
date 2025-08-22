'''
This module provides classes and functions for working with local Hilbert spaces,
including the definition of local operators and their kernels.

Classes:
    - LocalOpKernels: A collection of operator kernels for different backends (int, numpy, jax).
    - LocalSpace: Represents a local Hilbert space, including its properties and available operators.

Enums:
    - LocalSpaceTypes: Enumerates different types of local spaces (e.g., spin-1/2, spin-1, spinless fermions).
'''
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple, Dict



####################################################################################################

# kernels always return (states_array, coeffs_array)
# int:   (state:int, ns:int,    sites:np.ndarray[int32], *extra) -> (np.ndarray[int64], np.ndarray[float/complex])
# numpy: (state:np.ndarray,     sites:np.ndarray[int32], *extra) -> (np.ndarray[...,],  np.ndarray[...])
# jax:   (state:jnp.ndarray,    sites:jnp.ndarray[int32],*extra) -> (jnp.ndarray[...,], jnp.ndarray[...])
@dataclass(frozen=True)
class LocalOpKernels:
    fun_int             : Callable
    fun_np              : Optional[Callable] = None
    fun_jax             : Optional[Callable] = None
    # how many site indices the op needs at call-time (0=global/all-or-preset, 1=local, 2=correlation)
    site_parity         : int = 1
    # whether applying the op can change the state (helps sparse assembly sizing)
    modifies_state      : bool = True
    # default extra args bound at operator creation (can be overridden)
    default_extra_args  : Tuple = ()

class LocalSpaceTypes(Enum):
    SPIN_1_2            = "spin-1/2"
    SPIN_1              = "spin-1"
    SPINLESS_FERMIONS   = "spinless-fermions"
    BOSONS              = "bosons"

@dataclass(frozen = True)
class LocalSpace:
    name                : str
    typ                 : LocalSpaceTypes
    local_dim           : int   # Local Hilbert space dimension (Nhl), e.g., 2S+1 for spins, cutoff+1 for bosons, 2 for spinless fermions
    sign_rule           : int   # Sign rule for fermions: +1 for bosons/spins, -1 for fermions
    cutoff              : int   # Cutoff for bosonic modes (if applicable), otherwise 0
    max_occupation      : int   # Maximum occupation number for bosonic modes (if applicable), otherwise 0 - 1 for fermions, cutoff for bosons
    onsite_operators    : Dict[str, LocalOpKernels] = field(default_factory=dict)

    def has_op(self, key: str) -> bool:
        return key in self.onsite_operators
    
    def get_op(self, key: str) -> LocalOpKernels:
        if not self.has_op(key):
            raise KeyError(f"Operator '{key}' not found.")
        return self.onsite_operators[key]
    
    def __str__(self):
        return f"LocalSpace(name={self.name}, local_dim={self.local_dim}, sign_rule={self.sign_rule}, cutoff={self.cutoff}, max_occupation={self.max_occupation}, operators={list(self.onsite_operators.keys())})"
    
    def __repr__(self):
        return f"LocalSpace(name={self.name},nhl={self.local_dim},occ_max={self.max_occupation})"

    #####################
    #! Set Operators for later
    #####################

    #####################
    #! Defaults
    #####################

    @staticmethod
    def default_spin_half():
        # Spin-1/2: dim=2, no cutoff, sign +1
        return LocalSpace(
            name                =   LocalSpaceTypes.SPIN_1_2.value,
            local_dim           =   2,
            sign_rule           =   +1,
            cutoff              =   0,
            max_occupation      =   1,
            typ                 =   LocalSpaceTypes.SPIN_1_2,
            onsite_operators    =   {},
        )

    @staticmethod
    def default_fermion_spinless(max_sites_hint: int = 0):
        # Spinless fermion: dim=2, Pauli exclusion, fermionic sign −1
        return LocalSpace(
            name                =   LocalSpaceTypes.SPINLESS_FERMIONS.value,
            local_dim           =   2,
            sign_rule           =   -1,
            cutoff              =   0,
            max_occupation      =   1,
            typ                 =   LocalSpaceTypes.SPINLESS_FERMIONS,
            onsite_operators    =   {}
        )

    @staticmethod
    def default_boson(cutoff: int = 4):
        # Boson with cutoff (dim = cutoff+1)
        return LocalSpace(
            name                =   LocalSpaceTypes.BOSONS.value,
            local_dim           =   cutoff+1,
            sign_rule           =   +1,
            cutoff              =   cutoff,
            max_occupation      =   cutoff,
            typ                 =   LocalSpaceTypes.BOSONS,
            onsite_operators    =   {}
        )
        
    @staticmethod
    def default():
        return LocalSpace.default_spin_half()

#####################################################################################################

class StateTypes(Enum):
    INTEGER = "integer"
    VECTOR  = "vector"

    def lower(self):
        return self.value.lower()

    def upper(self):
        return self.value.upper()

    def __str__(self):
        return self.value

#####################################################################################################
