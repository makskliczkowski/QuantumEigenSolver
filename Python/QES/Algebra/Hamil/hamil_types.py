'''
file: Algebra/hamil_types.py
This module contains the enumeration of Hamiltonian models and related functions.
A Hamiltonian model is defined as a class of Hamiltonians that share a common structure.

Date: 2025-02-01
'''

from enum import Enum, unique
import numba

# ED limits (using hexadecimal where applicable)
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

# @numba.experimental.jitclass
class DummyVector:
    """
    A dummy vector-like object that mimics a constant vector of length `ns` with all elements equal to `val`.
    Used for broadcasting scalar couplings in Hamiltonian construction.
    """
    def __init__(self, val, ns=None, backend=None):
        self.val = val
        self.ns = ns if ns is not None else 1
        self._backend = backend if backend is not None else None

    def __getitem__(self, item):
        # Support integer or slice indexing
        if isinstance(item, int):
            return self.val
        elif isinstance(item, slice):
            length = len(range(*item.indices(self.ns)))
            return [self.val] * length
        else:
            raise TypeError("Invalid index type.")

    def __len__(self):
        return self.ns

    def __iter__(self):
        for _ in range(self.ns):
            yield self.val

    def __repr__(self):
        return f"DummyVector(val={self.val}, ns={self.ns})"

    def __str__(self):
        return f"DummyVector({self.val}) * {self.ns}"

    def __array__(self, dtype=None):
        backend = self._backend if self._backend is not None else __import__('numpy')
        return backend.full(self.ns, self.val, dtype=dtype)

    def __array_priority__(self):
        return 100

    # Arithmetic operations
    def _binary_op(self, other, op):
        if isinstance(other, (int, float, complex)):
            return DummyVector(op(self.val, other), ns=self.ns, backend=self._backend)
        elif isinstance(other, DummyVector):
            if self.ns != other.ns:
                raise ValueError("DummyVector: size mismatch in operation.")
            return DummyVector(op(self.val, other.val), ns=self.ns, backend=self._backend)
        else:
            return NotImplemented

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_op(other, lambda a, b: b - a)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary_op(other, lambda a, b: b / a)

    def __floordiv__(self, other):
        return self._binary_op(other, lambda a, b: a // b)

    def __rfloordiv__(self, other):
        return self._binary_op(other, lambda a, b: b // a)

    def __mod__(self, other):
        return self._binary_op(other, lambda a, b: a % b)

    def __rmod__(self, other):
        return self._binary_op(other, lambda a, b: b % a)

    def __pow__(self, other):
        return self._binary_op(other, lambda a, b: a ** b)

    def __eq__(self, other):
        if isinstance(other, DummyVector):
            return (self.val == other.val) and (self.ns == other.ns)
        return False
    
    def to_array(self, dtype=None, backend=None):
        """
        Convert the DummyVector to a numpy array.
        """
        backend = backend if backend is not None else __import__('numpy')
        return backend.full(self.ns, self.val, dtype=dtype)