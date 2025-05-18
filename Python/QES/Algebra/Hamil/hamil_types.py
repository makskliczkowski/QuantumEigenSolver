'''
file: Algebra/hamil_types.py
This module contains the enumeration of Hamiltonian models and related functions.
A Hamiltonian model is defined as a class of Hamiltonians that share a common structure.

Date: 2025-02-01
'''
from general_python.algebra.utils import distinguish_type
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
    Constant vector of length `ns` with all entries == `val`.
    A thin dummy so scalar couplings can be broadcast like arrays.
    """

    #! construction
    def __init__(self, val, ns: int | None = None, *, backend=None):
        """
        Initialize the object with the given value, number of sites, and backend.
        Args:
            val:
                The value to be assigned to the object.
            ns (int, optional):
                The number of sites. If not provided, defaults to 1.
            backend (optional):
                The backend module to use. If not provided, defaults to 'numpy'.
        """
        
        self.val      = val
        self.ns       = int(ns) if ns is not None else 1
        self._backend = backend or __import__("numpy")

    # ---------------------------------------------------------------------------
    
    @property
    def dtype(self):
        return getattr(self.val, "dtype", type(self.val))

    def astype(self, dtype, copy: bool = False, *, backend=None):
        """
        Return a `DummyVector` with the same length but `val` cast to `dtype`.

        Parameters
        ----------
        dtype : str | numpy.dtype | jax.numpy.dtype | type
            Desired element dtype.
        copy  : bool, default False
            If False and the dtype is unchanged -> return self.
            If True  -> always return a *new* `DummyVector`.
        backend : optional
            Backend module (`numpy`, `jax.numpy`, …) controlling the cast.
            If None, use the instance's backend.

        Notes
        -----
        *The method never materialises a full array*, so it's O(1) in memory.
        """
        backend = backend or self._backend
        tgt_dt  = distinguish_type(dtype)

        # fast path: nothing to change
        if not copy and tgt_dt == self.dtype:
            return self

        new_val = backend.asarray(self.val, dtype=tgt_dt).item()
        return DummyVector(new_val, ns=self.ns, backend=backend)

    # ---------------------------------------------------------------------------
    
    def __array__(self, dtype=None):
        return self._backend.full(self.ns, self.val, dtype=dtype)

    def __array_priority__(self):
        return 100.0

    # ---------------------------------------------------------------------------
    
    def __repr__(self):
        return f"DummyVector(val={self.val!r}, ns={self.ns})"

    def __str__(self):
        return f"[{self.val}] * {self.ns}"

    # ---------------------------------------------------------------------------
    
    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            length = len(range(*idx.indices(self.ns)))
            return DummyVector(self.val, ns=length, backend=self._backend)
        elif isinstance(idx, int):
            return self.val
        else:
            raise TypeError("DummyVector supports int or slice indices only.")

    def __iter__(self):
        yield from (self.val for _ in range(self.ns))

    # ---------------------------------------------------------------------------
    
    def _binary(self, other, op):
        if isinstance(other, DummyVector):
            if self.ns != other.ns:
                raise ValueError("DummyVector: size mismatch")
            other_val = other.val
        else:  # assume scalar
            other_val = other
        return DummyVector(op(self.val, other_val), ns=self.ns, backend=self._backend)

    def __add__(self, other):       return self._binary(other, lambda a, b: a + b)
    def __radd__(self, other):      return self.__add__(other)
    def __sub__(self, other):       return self._binary(other, lambda a, b: a - b)
    def __rsub__(self, other):      return self._binary(other, lambda a, b: b - a)
    def __mul__(self, other):       return self._binary(other, lambda a, b: a * b)
    def __rmul__(self, other):      return self.__mul__(other)
    def __truediv__(self, other):   return self._binary(other, lambda a, b: a / b)
    def __rtruediv__(self, other):  return self._binary(other, lambda a, b: b / a)
    # (add more as needed)

    # ---------------------------------------------------------------------------
    
    def __eq__(self, other):
        return (
            isinstance(other, DummyVector)
            and self.ns == other.ns
            and self.val == other.val
        )

    def __hash__(self):
        return hash((self.val, self.ns))
    
    def to_array(self, dtype=None, backend=None):
        """
        Convert the DummyVector to a numpy array.
        """
        backend = backend if backend is not None else __import__('numpy')
        return backend.full(self.ns, self.val, dtype=dtype)
    
##################################################################################
#! END OF FILE