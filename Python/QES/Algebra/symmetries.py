'''
file: Algebra/symmetries.py
Description: This module contains functions to compute the symmetries of a given state at a given representation.
'''

# Import the necessary modules
import math
import numpy as np
import time
from functools import partial
from typing import Callable, Union, Tuple, List, Optional

# operator module for operator overloading
from Algebra.Operator.operator import Operator, SymmetryGenerators

# from general Python modules
from general_python.lattices.__lattice__ import Lattice, LatticeBC, LatticeDirection
from general_python.common.binary import rotate_left, rotate_right, flip_all, rev, rotate_left_ax, popcount, _BACKEND_DEF_SPIN, _BACKEND_REPR
from general_python.algebra.utils import get_backend as __backend, DEFAULT_BACKEND, maybe_jit

####################################################################################################

__LATTICE_NONE_ERROR    = "Lattice object is None."
__LATTICE_DIM2_ERROR    = "Lattice dimension must be at least 2..."
__LATTICE_DIM3_ERROR    = "Lattice dimension must be at least 3..."

####################################################################################################
# Tanslational Symmetries
####################################################################################################

def translation_x(lat, backend='default'):
    """
    Translation in the X direction.
    - For 1D: 
        rotates the entire state by one to the left:
        
        e.g. 0b1010 -> 0b0101 or [1, 0, 1, 0] -> [0, 1, 0, 1] or [1, 0, 0, 1] -> [0, 0, 1, 1]
        
    - For 2D: 
        treats the state as rows of length $L_x$ and cyclically shifts each row.
    - For 3D: 
        applies the same row-shift on every (y-)row in each z-slice.
    """
    
    if lat is None:
        raise ValueError(__LATTICE_NONE_ERROR)
    
    lx          = lat.lx
    ly          = lat.ly if lat.dim > 1 and hasattr(lat, 'ly') else 1
    lz          = lat.lz if lat.dim > 2 and hasattr(lat, 'lz') else 1
    ns          = lat.sites

    if lat.dim == 1:
        def op(state):
            '''
            This is a placeholder function that will be returned.
            '''
            # Integer state: binary representation (e.g. 0b0101 for 4 sites)
            new_state = rotate_left(state, ns, backend)
            return (new_state, 1.0)
        return op
    elif lat.dim == 2:
        row_mask = (1 << lx) - 1
        def op(state):
            '''
            This is a placeholder function that will be returned.
            '''
            # Integer state: binary representation (e.g. 0b0101 for 4 sites)
            if not isinstance(state, int):
                state_r     = state.reshape((ly, lx))
                new_state_r = rotate_left(state_r, ns, backend, axis=1)
                return (new_state_r.reshape(-1), 1.0)
            return (rotate_left_ax(state, lx, row_mask, ly, lz, -1, axis=1, backend=backend), 1.0)
        return op
    elif lat.dim == 3:
        def op(state):
            '''
            This is a placeholder function that will be returned.
            '''
            # Integer state: binary representation (e.g. 0b0101 for 4 sites)
            if not isinstance(state, int):
                state_r     = state.reshape((lz, ly, lx))
                new_state_r = rotate_left(state_r, ns, backend, axis=2)
                return (new_state_r.reshape(-1), 1.0)
            return (rotate_left_ax(state, lx, row_mask, ly, lz, 1, axis=2, backend=backend), 1.0)
        return op
    return None

def translation_y(lat, backend='default'):
    """
    Translation in the Y direction.
    - For 2D: treats the state as a (ly x lx) array and cyclically shifts the rows.
    - For 3D: treats the state as a (lz x ly x lx) array and cyclically shifts the rows within each z-slice.
    
    For integer states (binary encoded), it cyclically reassigns entire rows.
    """
    if lat is None:
        raise ValueError(__LATTICE_NONE_ERROR)
    if lat.dim == 1:
        raise ValueError(__LATTICE_DIM2_ERROR)

    lx = lat.lx
    ly = lat.ly if lat.dim > 1 and hasattr(lat, 'ly') else 1
    lz = lat.lz if lat.dim > 2 and hasattr(lat, 'lz') else 1
    ns = lat.sites

    if lat.dim == 2:
        def op(state):
            '''
            This is a placeholder function that will be returned.
            '''
            if isinstance(state, int):
                raise NotImplementedError("Translation in the Y direction is not implemented for integer states.")
            state_r = state.reshape((ly, lx))
            new_state_r = rotate_left(state_r, ns, backend, axis = 1)
            return (new_state_r.reshape(-1), 1.0)
        return op
    
    def op_3d(state):
        '''
        This is a placeholder function that will be returned.
        '''
        if isinstance(state, int):
            raise NotImplementedError("Translation in the Y direction is not implemented for integer states.")
        state_r = state.reshape((lz, ly, lx))
        new_state_r = rotate_left(state_r, ns, backend, axis = 1)
        return (new_state_r.reshape(-1), 1.0)
    return op_3d

def translation_z(lat, backend='default'):
    """
    Translation in the Z direction (applicable only for 3D lattices).
    - For array states: reshapes the state as (lz, ly, lx) and cyclically shifts the z-slices (roll along axis 0).
    - For integer states: performs a cyclic shift of the z-slices.
    """
    if lat is None:
        raise ValueError(__LATTICE_NONE_ERROR)
    if lat.dim != 3:
        raise ValueError(__LATTICE_DIM3_ERROR)
    
    lx = lat.lx
    ly = lat.ly
    lz = lat.lz
    ns = lat.sites

    if lat.dim == 3:
        def op(state):
            '''
            This is a placeholder function that will be returned.
            '''
            if isinstance(state, int):
                raise NotImplementedError("Translation in the Z direction is not implemented for integer states.")
            state_r = state.reshape((lz, ly, lx))
            new_state_r = rotate_left(state_r, ns, backend, axis = 0)
            return (new_state_r.reshape(-1), 1.0)
        return op

def translation(lat : Lattice,
                kx  : int,
                ky  : Optional[int] = 0,
                kz  : Optional[int] = 0,
                dim : Optional[int] = None,
                direction           = LatticeDirection.X,
                backend             = 'default'):
    """
    Generates a translation operator with a momentum phase factor.
    The phase is defined as exp(i * 2π * (k / L)) along the chosen direction.
    
    Parameters:
        - lat:
            lattice object.
        - kx, ky, kz: 
            momentum quantum numbers.
        - dim: 
            lattice dimension (if not provided, lat.get_Dim() is used).
        - direction: 
            one of 'x', 'y', or 'z' (default is 'x').
        - backend: 
            backend specifier for array operations.
    Returns:
        Operator: The translation operator with the defined momentum phase factor.    
    """
    if lat is None:
        raise ValueError(__LATTICE_NONE_ERROR)
    
    if dim is None:
        dim = lat.dim()

    lx = lat.lx
    ly = lat.ly if dim > 1 and hasattr(lat, 'ly') else 1
    lz = lat.lz if dim > 2 and hasattr(lat, 'lz') else 1
    kx = 2 * math.pi * kx / lx
    ky = 2 * math.pi * ky / ly if dim > 1 and ky is not None else 0
    kz = 2 * math.pi * kz / lz if dim > 2 and kz is not None else 0
    k  = kx
    
    if direction == LatticeDirection.X:
        op_fun      = translation_x(lat, backend)
    elif direction == LatticeDirection.Y:
        op_fun      = translation_y(lat, backend)
        k           = ky
    elif direction == LatticeDirection.Z:
        op_fun      = translation_z(lat, backend)
        k           = kz
    else:
        op_fun      = translation_x(lat, backend)
    phase = math.exp(1j * k)
    
    # get the symmetry generator type
    typek = SymmetryGenerators.Translation_x
    if direction == LatticeDirection.Y:
        typek = SymmetryGenerators.Translation_y
    elif direction == LatticeDirection.Z:
        typek = SymmetryGenerators.Translation_z
    
    name = f'T_{direction.name}'
    
    return Operator(lattice = lat, eigval = phase, fun = op_fun,
            type = typek, backend = backend, name = name)

####################################################################################################
# Reflection Symmetries
####################################################################################################

def _reflection(ns : int, backend : str = 'default'):
    """ 
    Reflection operator for a given state.
    """
    def op(state):
        return rev(state, ns, backend)
    return op

def reflection(sec : int, lat : Optional[Lattice] = None, ns : Optional[int] = None, backend : str = 'default'):
    """
    Generates a reflection operator with eigenvalue 'sec'.
    Parameters:
        - lat: lattice object.
        - sec: eigenvalue of the reflection operator.
        - base: base of the integer representation (default is 2).
    Returns:
        Operator: The reflection operator defined by the parameters.
    """
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(__LATTICE_NONE_ERROR)
    return Operator(lattice = lat, eigval = sec, fun=_reflection(ns, backend), typek=SymmetryGenerators.Reflection)

####################################################################################################
# Parity Symmetries
####################################################################################################

# --- Parity Z ---

def _flip_z(ns: int, backend: str = 'default', spin: bool = _BACKEND_DEF_SPIN, spin_value: float = _BACKEND_REPR):
    """
    Creates the behavior that checks the parity of the state by applying the Z-flip operator
    to the state and returning the phase factor. The state is assumed to be in the
    Pauli-Z basis.
    For integer states the binary popcount is used;
    for array-like states the helper popcount is applied.
    """
    def op(state):
        spin_ups    = ns - popcount(state, spin=spin, backend=backend, spin_value=spin_value)
        phase       = 1.0 - 2.0 * (spin_ups & 1)
        return (state, phase)
    return op


def parity_z(sec : int, ns : Optional[int] = None, lat : Optional[Lattice] = None,
            backend : str = 'default', spin: bool = _BACKEND_DEF_SPIN, spin_value : float = _BACKEND_REPR):
    """
    Generates a partity operator by applying the Pauli-Z flip operator to the state.
    The state is assumed to be in the Pauli-Z basis.
    Parameters:
        - lat: lattice object.
        - sec: eigenvalue of the parity operator.
    """
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(__LATTICE_NONE_ERROR)
    return Operator(lattice = lat, eigval = sec, fun=_flip_z(ns, backend, spin, spin_value), typek=SymmetryGenerators.ParityZ)

# --- Parity Y ---

def _flip_y(ns: int, backend: str = 'default', 
        spin: bool = _BACKEND_DEF_SPIN, spin_value: float = _BACKEND_REPR):
    """
    Creates the behavior that checks the parity of the state by applying the Y-flip operator
    to the state and returning the phase factor. The state is assumed to be in the
    Pauli-Z basis.
    For integer states the binary popcount is used;
    for array-like states the helper popcount is applied.
    """
    phase_factor    = 1j if ns % 2 == 0 else -1j
    
    def op(state):
        spin_ups    = ns - popcount(state, spin=spin, backend=backend, spin_value=spin_value)
        phase       = (1 - 2 * (spin_ups & 1)) * phase_factor
        return (flip_all(state, ns, backend=backend, spin=spin, spin_value=spin_value), phase)
    return op

def parity_y(sec : int, ns : Optional[int] = None, lat : Optional[Lattice] = None,
            backend : str = 'default', spin: bool = _BACKEND_DEF_SPIN, spin_value : float = _BACKEND_REPR):
    """
    Generates a parity operator by applying the Y-flip operator to the state.
    The state is assumed to be in the Pauli-Z basis.
    Parameters:
        - lat: lattice object.
        - ns: number of sites in the lattice.
        - sec: eigenvalue of the parity operator.
        - backend: backend specifier for array operations.
        - spin: boolean flag to indicate whether the state is a spin state.
        - spin_value: value of the spin.
    """
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(__LATTICE_NONE_ERROR)
    return Operator(lattice = lat, eigval = sec, fun=_flip_y(ns, backend, spin, spin_value), typek=SymmetryGenerators.ParityY)

# --- Parity X ---

def _flip_x(ns: int, backend: str = 'default',
            spin: bool = _BACKEND_DEF_SPIN, spin_value : float = _BACKEND_REPR):
    """
    Creates the behavior that checks the parity of the state by applying the X-flip operator
    to the state and returning the phase factor. The state is assumed to be in the
    Pauli-Z basis.
    For integer states the binary popcount is used;
    for array-like states the helper popcount is applied.
    """
        
    def op(state):
        return (flip_all(state, ns, backend=backend, spin=spin, spin_value=spin_value), 1)
    return op

def parity_x(sec : int, ns : Optional[int] = None, lat : Optional[Lattice] = None,
            backend : str = 'default', spin: bool = _BACKEND_DEF_SPIN, spin_value : float = _BACKEND_REPR):
    """
    Generates a parity operator by applying the X-flip operator to the state.
    The state is assumed to be in the Pauli-Z basis.
    Parameters:
        - lat: lattice object.
        - sec: eigenvalue of the parity operator.
    """
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(__LATTICE_NONE_ERROR)
    return Operator(lattice = lat, eigval = sec, fun=_flip_x(ns, backend, spin, spin_value), typek=SymmetryGenerators.ParityX)

####################################################################################################
# Choose Symmetry
####################################################################################################

def choose(sym_specifier : Tuple[SymmetryGenerators, int], 
        ns : Optional[int] = None, lat : Optional[Lattice] = None,
        backend : str = 'default', spin_value : Optional[float] = _BACKEND_REPR, spin : Optional[bool] = _BACKEND_DEF_SPIN):
    """
    Given a symmetry specification (a tuple of (SymmetryGenerators, eigenvalue))
    and a lattice, returns the corresponding symmetry operator.
    Parameters:
        - sym_specifier: 
            a tuple of (SymmetryGenerators, eigenvalue) specifying the symmetry.
        - ns: 
            number of sites in the lattice.
        - lat: 
            lattice object.
    Returns:
        Operator: The symmetry operator corresponding to the given specification.    
    
    Example sym_spec values:
        (SymmetryGenerators.T, k)           -> Translation with momentum sector k
        (SymmetryGenerators.R, sec)         -> Reflection with eigenvalue sec
        (SymmetryGenerators.PX, sec)        -> Parity (σ^x) with eigenvalue sec
        (SymmetryGenerators.PY, sec)        -> Parity (σ^y) with eigenvalue sec
        (SymmetryGenerators.PZ, sec)        -> Parity (σ^z) with eigenvalue sec
        (SymmetryGenerators.E, _)           -> Identity
    """
    gen, eig = sym_specifier
    
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(__LATTICE_NONE_ERROR)
    
    if gen == SymmetryGenerators.Translation_x:
        return translation(lat, kx=eig, dim = lat.dim, direction = LatticeDirection.X)
    elif gen == SymmetryGenerators.Translation_y:
        return translation(lat, kx=0, ky=eig, dim = lat.dim, direction = LatticeDirection.Y)
    elif gen == SymmetryGenerators.Translation_z:
        return translation(lat, kx=0, ky=0, kz=eig, dim = lat.dim, direction = LatticeDirection.Z)
    elif gen == SymmetryGenerators.Reflection:
        return reflection(sec=eig, lat=lat, ns=ns, backend=backend)
    elif gen == SymmetryGenerators.ParityX:
        return parity_x(sec=eig, lat=lat, ns=ns, backend=backend, spin=spin, spin_value=spin_value)
    elif gen == SymmetryGenerators.ParityY:
        return parity_y(sec=eig, lat=lat, ns=ns, backend=backend, spin=spin, spin_value=spin_value)
    elif gen == SymmetryGenerators.ParityZ:
        return parity_z(sec=eig, lat=lat, ns=ns, backend=backend, spin=spin, spin_value=spin_value)
    elif gen == SymmetryGenerators.E:
        return Operator(lat)
    # default return
    return Operator(lat)

####################################################################################################

from general_python.common.tests import GeneralAlgebraicTest

class SymmetryTests(GeneralAlgebraicTest):
    """
    Class for testing the symmetry functions.
    """
    
    def add_tests(self):
        """
        Adds all symmetry function tests to the test list.
        """
        self.tests.append(self.test_translation_x_int)
        self.tests_dict[self.test_translation_x_int.__name__] = self.test_translation_x_int

        self.tests.append(self.test_translation_x_array)
        self.tests_dict[self.test_translation_x_array.__name__] = self.test_translation_x_array

        self.tests.append(self.test_translation_y_array)
        self.tests_dict[self.test_translation_y_array.__name__] = self.test_translation_y_array

        self.tests.append(self.test_translation_z_array)
        self.tests_dict[self.test_translation_z_array.__name__] = self.test_translation_z_array

        self.tests.append(self.test_translation_operator)
        self.tests_dict[self.test_translation_operator.__name__] = self.test_translation_operator

        self.tests.append(self.test_reflection)
        self.tests_dict[self.test_reflection.__name__] = self.test_reflection

        self.tests.append(self.test_parity_z)
        self.tests_dict[self.test_parity_z.__name__] = self.test_parity_z

        self.tests.append(self.test_parity_y)
        self.tests_dict[self.test_parity_y.__name__] = self.test_parity_y

        self.tests.append(self.test_parity_x)
        self.tests_dict[self.test_parity_x.__name__] = self.test_parity_x

        self.tests.append(self.test_choose)
        self.tests_dict[self.test_choose.__name__] = self.test_choose

    # --- Translation Tests ---

    def test_translation_x_int(self, lat, state_int):
        """
        Test the translation_x operator on an INTEGER state.
        """
        from general_python.common.binary import int2binstr
        
        self._log("Testing translation_x on INTEGER state", self.test_count, "blue")
        op_func             = translation_x(lat, self.backend)
        new_state, phase    = op_func(state_int)
        self._log(f"Input (int): {int2binstr(state_int, lat.sites)}", self.test_count)
        self._log(f"Output (int): {int2binstr(new_state, lat.sites)}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_translation_x_array(self, lat, state_arr):
        """
        Test the translation_x operator on an ARRAY state.
        """
        self._log("Testing translation_x on ARRAY state", self.test_count, "blue")
        op_func             = translation_x(lat, self.backend)
        new_state, phase    = op_func(state_arr)
        self._log(f"Input (array): {state_arr}", self.test_count)
        self._log(f"Output (array): {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_translation_y_array(self, lat, state_arr):
        """
        Test the translation_y operator on an ARRAY state.
        (Integer version is not implemented.)
        """
        if lat.dim < 2:
            self._log("Skipping translation_y: lattice dim < 2", self.test_count, "yellow")
            self.test_count += 1
            return
        
        self._log("Testing translation_y on ARRAY state", self.test_count, "blue")
        op_func             = translation_y(lat, self.backend)
        new_state, phase    = op_func(state_arr)
        self._log(f"Input (array): {state_arr}", self.test_count)
        self._log(f"Output (array): {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count     += 1

    def test_translation_z_array(self, lat, state_arr):
        """
        Test the translation_z operator on an ARRAY state.
        """
        if lat.dim != 3:
            self._log("Skipping translation_z: lattice dim != 3", self.test_count, "yellow")
            self.test_count     += 1
            return
        self._log("Testing translation_z on ARRAY state", self.test_count, "blue")
        op_func             = translation_z(lat, self.backend)
        new_state, phase    = op_func(state_arr)
        self._log(f"Input (array): {state_arr}", self.test_count)
        self._log(f"Output (array): {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_translation_operator(self, lat, state, kx, ky, kz, direction):
        """
        Test the full translation operator (with momentum phase) using its apply() method.
        """
        self._log(f"Testing translation operator for direction {direction.name}", self.test_count, "blue")
        op                  = translation(lat, kx, ky, kz, dim=lat.dim, direction=direction, backend=self.backend)
        new_state, phase    = op.apply(state)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Output state: {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    # --- Reflection Test ---

    def test_reflection(self, lat, state):
        """
        Test the reflection operator.
        """
        self._log("Testing reflection operator", self.test_count, "blue")
        op = reflection(sec=1, lat=lat, backend=self.backend)
        new_state, phase = op.apply(state)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Reflected state: {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    # --- Parity Tests ---

    def test_parity_z(self, lat, state):
        """
        Test the parity_z operator.
        """
        self._log("Testing parity_z operator", self.test_count, "blue")
        op = parity_z(sec=1, lat=lat, backend=self.backend)
        new_state, phase = op.apply(state)
        expected_phase = 1.0 - 2.0 * ((lat.sites - popcount(state, backend=self.backend)) & 1)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Parity_z output: {new_state}", self.test_count)
        self._log(f"Expected phase: {expected_phase}, Obtained phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_parity_y(self, lat, state):
        """
        Test the parity_y operator.
        """
        self._log("Testing parity_y operator", self.test_count, "blue")
        op = parity_y(sec=1, lat=lat, backend=self.backend)
        new_state, phase = op.apply(state)
        expected_phase = (1 - 2 * ((lat.sites - popcount(state, backend=self.backend)) & 1)) * (1j ** lat.sites)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Parity_y output: {new_state}", self.test_count)
        self._log(f"Expected phase: {expected_phase}, Obtained phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_parity_x(self, lat, state):
        """
        Test the parity_x operator.
        """
        self._log("Testing parity_x operator", self.test_count, "blue")
        op = parity_x(sec=1, lat=lat, backend=self.backend)
        new_state, phase = op.apply(state)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Parity_x output: {new_state}", self.test_count)
        self._log(f"Phase (should be 1): {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    # --- Choose() Test ---

    def test_choose(self, lat, state):
        """
        Test the choose() function to select a symmetry operator.
        """
        self._log("Testing choose() function for symmetry operators", self.test_count, "blue")
        # Test Translation_x
        op_tx = choose((SymmetryGenerators.Translation_x, 1), lat=lat, backend=self.backend)
        st_tx, ph_tx = op_tx.apply(state)
        # Test Reflection
        op_ref = choose((SymmetryGenerators.Reflection, 1), lat=lat, backend=self.backend)
        st_ref, ph_ref = op_ref.apply(state)
        # Test Parity_x
        op_px = choose((SymmetryGenerators.ParityX, 1), lat=lat, backend=self.backend)
        st_px, ph_px = op_px.apply(state)
        self._log(f"Translation_x: Output: {st_tx}, Phase: {ph_tx}", self.test_count)
        self._log(f"Reflection:     Output: {st_ref}, Phase: {ph_ref}", self.test_count)
        self._log(f"Parity_x:       Output: {st_px}, Phase: {ph_px}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def run_tests(self):
        """
        Runs all symmetry function tests.
        """
        from general_python.common.binary import int2binstr
        self.test_count = 1
        separator       = "=" * 50
        self._log(separator, 0)
        self._log("TESTING SYMMETRY FUNCTIONS", 0, "green")
        self._log(separator, 0)

        # Create sample lattices.
        lat1d = Lattice(4)            # 1D lattice with 4 sites
        lat2d = Lattice(4, 4)           # 2D lattice: 4x4
        lat3d = Lattice(2, 2, 2)        # 3D lattice: 2x2x2

        # Create sample states.
        # For integer representations, generate random integers with appropriate bit-length.
        state_1d_int = np.random.randint(0, 2 ** lat1d.sites)
        state_2d_int = np.random.randint(0, 2 ** lat2d.sites)
        state_3d_int = np.random.randint(0, 2 ** lat3d.sites)
        # For array representations, convert the integer to a binary vector.
        state_1d_arr = np.array([int(b) for b in int2binstr(state_1d_int, lat1d.sites)])
        state_2d_arr = np.array([int(b) for b in int2binstr(state_2d_int, lat2d.sites)])
        state_3d_arr = np.array([int(b) for b in int2binstr(state_3d_int, lat3d.sites)])

        # Run translation_x tests.
        self.test_translation_x_int(lat1d, state_1d_int)
        self.test_translation_x_array(lat1d, state_1d_arr)
        self.test_translation_x_int(lat2d, state_2d_int)
        self.test_translation_x_array(lat2d, state_2d_arr)
        self.test_translation_x_int(lat3d, state_3d_int)
        self.test_translation_x_array(lat3d, state_3d_arr)

        # Run translation_y tests (only array versions).
        self.test_translation_y_array(lat2d, state_2d_arr)
        self.test_translation_y_array(lat3d, state_3d_arr)

        # Run translation_z test (3D only, array version).
        self.test_translation_z_array(lat3d, state_3d_arr)

        # Test translation operator with momentum.
        self.test_translation_operator(lat2d, state_2d_arr, 1, 0, 0, LatticeDirection.X)
        self.test_translation_operator(lat2d, state_2d_arr, 0, 1, 0, LatticeDirection.Y)
        self.test_translation_operator(lat3d, state_3d_arr, 0, 0, 1, LatticeDirection.Z)

        # Test reflection operator.
        self.test_reflection(lat2d, state_2d_arr)
        self.test_reflection(lat3d, state_3d_arr)

        # Test parity operators.
        self.test_parity_z(lat2d, state_2d_arr)
        self.test_parity_y(lat2d, state_2d_arr)
        self.test_parity_x(lat2d, state_2d_arr)

        # Test choose() function.
        self.test_choose(lat2d, state_2d_arr)

        self._log(separator, 0)
        total_time = time.time()  # For simplicity, you can record start time before tests.
        self._log(f"Total testing time: {total_time:.6f} sec", 0, "green")
        self._log(separator, 0)
        self._log("Testing completed.", 0, "green")
        
# -------------------------------------------------------------------------------------------------