"""
file : Algebra/hamil.py

High-level Hamiltonian class for the Quantum Energy Solver (QES) package. This class is used to
define the Hamiltonian of a system. It may be either a Many-Body Quantum Mechanics Hamiltonian or a
non-interacting system Hamiltonian. It may generate a Hamiltonian matrix but in addition it defines
how an operator acts on a state. The Hamiltonian class is an abstract class and is not meant to be
instantiated. It is meant to be inherited by other classes.

Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Date    : 2025-02-01
Version : 1.0.0
Changes :
    2025-02-01 (1.0.0) : First implementation of the Hamiltonian class. - MK
"""

import numpy as np
import scipy as sp
from typing import List, Tuple, Union, Optional, Callable
from abc import ABC
from functools import partial
import time
from tabulate import tabulate

###################################################################################################
from Algebra.hilbert import HilbertSpace, set_operator_elem
from Algebra.Operator.operator import Operator, OperatorTypeActing, create_add_operator
from Algebra.Operator.operator_matrix import operator_create_np
from Algebra.hamil_types import *
from Algebra.hamil_energy import local_energy_int_wrap, local_energy_np_wrap
###################################################################################################

###################################################################################################
from general_python.algebra.ran_wrapper import random_vector
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend, DEFAULT_INT_TYPE, DEFAULT_FLOAT_TYPE, DEFAULT_CPX_TYPE
import general_python.algebra.linalg as linalg

if _JAX_AVAILABLE:
    from Algebra.hamil_energy import local_energy_jax_wrap
    from Algebra.hilbert import process_matrix_elem_jax, process_matrix_batch_jax
###################################################################################################

if _JAX_AVAILABLE:
    import jax
    from jax import jit
    import jax.lax as lax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO, CSR

###################################################################################################
#! Pure (functional) Hamiltonian update functions.
###################################################################################################

if _JAX_AVAILABLE:
    import Algebra.hamil_jit_methods as hjm
    
    def _hamiltonian_functional_jax_sparse( ns                  : int,
                                            hilbert_space       : HilbertSpace,
                                            max_local_changes   : int,
                                            loc_energy, start=0):
        """
        JAX version: Functional Hamiltonian construction (SPARSE).
        Parameters:
        - ns (int)         : The number of sites.
        - hilbert_space    : The Hilbert space. Provides the mapping of the Hilbert space.
        - loc_energy       : The local energy function. Must be callable with the following signature:
                                loc_energy(k, k_map, i) -> Tuple[List[int], List[int], List[ham.dtype]]
        - max_local_changes: The maximum number of local changes.
        - start            : The starting index for the update (default is 0).
        """
        
        # get the number of elements in the Hilbert space
        nh          = hilbert_space.Nh      # number of elements in the Hilbert space
        _start      = int(start)            # starting index for the inner parameter loop
        _ns         = int(ns)               # number of modes (parameters)
        elems       = jnp.arange(_start, _ns, dtype=jnp.int32) # ns is rather small
        
        # jax.debug.print("elems: {}", elems)
        # Preallocate arrays for indices and values.
        max_est     = nh * max_local_changes * (ns - start)
        max_est_in  = max_local_changes * (ns - start)
        all_rows    = jnp.full((max_est,), 0, dtype=jnp.int64)
        all_cols    = jnp.full((max_est,), 0, dtype=jnp.int64)
        all_vals    = jnp.full((max_est,), 0.0, dtype=hilbert_space.dtype)
        current_nnz = 0
        batch_size  = 1024 if nh > 1024 else nh // 2**4
        # jax.debug.print("batch_size: {}", batch_size) 
        
        for batch_start in range(0, nh, batch_size):
            batch_end = min(nh, batch_start + batch_size)
            unique_cols_batch, summed_vals_batch, counts_batch = process_matrix_batch_jax(
                loc_energy, batch_start, batch_end, hilbert_space, elems, max_est_in
            )
            # For each row in the batch, extract only the valid (unpadded) entries.
            for i in range(unique_cols_batch.shape[0]):
                count       = int(counts_batch[i])
                row_idx     = batch_start + i
                valid_cols  = unique_cols_batch[i, :count]
                valid_vals  = summed_vals_batch[i, :count]
                row_ids     = jnp.full(valid_cols.shape, row_idx, dtype=jnp.int32)

                # Calculate the slice to insert into pre-allocated arrays
                insert_slice    = slice(current_nnz, current_nnz + count)
                all_rows        = all_rows.at[insert_slice].set(row_ids)
                all_cols        = all_cols.at[insert_slice].set(valid_cols)
                all_vals        = all_vals.at[insert_slice].set(valid_vals)
                current_nnz     += count
                
        # Trim the pre-allocated arrays to the actual number of non-zero elements
        all_rows    = all_rows[:current_nnz]
        all_cols    = all_cols[:current_nnz]
        all_vals    = all_vals[:current_nnz]
        # Stack row and column indices to form a (2, nnz) array.
        indices     = jnp.stack([all_rows, all_cols], axis=1)
        return BCOO((all_vals, indices), shape=(nh, nh))
    
    @partial(jit, static_argnames=('ns', 'hilbert_space', 'loc_energy', 'start', 'dtype'))
    def _hamiltonian_functional_jax(ns: int, hilbert_space: HilbertSpace, loc_energy, start=0, dtype=None):
        """
        JAX version: Functional Hamiltonian construction (DENSE).
        Parameters:
        - ns (int)         : The number of sites.
        - hilbert_space    : The Hilbert space.
        - loc_energy       : The local energy function.
        - start            : The starting index for the update (default is 0).
        - dtype            : The data type of the Hamiltonian matrix.
        """
        
        # get the number of elements in the Hilbert space
        nh      = jnp.int64(hilbert_space.Nh)
        _start  = jnp.int64(start)
        
        def outer_loop_body(ham, k):
            k_map = hilbert_space[k]

            def inner_loop_body(ham, i):
                ham = loc_energy(ham, hilbert_space, k, k_map, i)
                return ham, None

            ham, _  = lax.scan(inner_loop_body, ham, jnp.arange(_start, ns))
            return ham, None

        init_ham    = jnp.zeros((nh, nh), dtype=dtype)
        ham, _      = lax.scan(outer_loop_body, init_ham, jnp.arange(0, nh))
        return ham

    def hamiltonian_functional_jax(ns: int, hilbert_space: HilbertSpace, loc_energy, max_local_changes: int, 
            is_sparse: bool, start=0, dtype=None):
        """
        JAX version: Functional Hamiltonian construction.  Dispatcher.
        """
        if dtype is None:
            dtype = hilbert_space.dtype
        
        if is_sparse:
            return _hamiltonian_functional_jax_sparse(ns, hilbert_space, max_local_changes, loc_energy, start)
        else:
            return _hamiltonian_functional_jax(ns, hilbert_space, loc_energy, start, dtype)
    
    # ----------------------------------------------------------------------------------------------

####################################################################################################
#! Hamiltonian class - abstract class
####################################################################################################

class Hamiltonian(ABC):
    '''
    A general Hamiltonian class. This class is used to define the Hamiltonian of a system. It may be 
    either a Many-Body Quantum Mechanics Hamiltonian or a non-interacting system Hamiltonian. It may 
    generate a Hamiltonian matrix but in addition it defines how an operator acts on a state. The
    Hamiltonian class is an abstract class and is not meant to be instantiated. It is meant to be
    inherited by other classes.
    '''
    
    _ERR_EIGENVALUES_NOT_AVAILABLE  = "The eigenvalues are not available."
    _ERR_HAMILTONIAN_NOT_AVAILABLE  = "The Hamiltonian matrix is not available."
    _ERR_HAMILTONIAN_INITIALIZATION = "An error occurred during Hamiltonian initialization."
    _ERR_HAMILTONIAN_BUILD          = "An error occurred during Hamiltonian build."
    _ERR_HILBERT_SPACE_NOT_PROVIDED = "The Hilbert space must be provided."
    _ERR_COUP_VEC_SIZE              = "The coupling vector size is invalid."
    
    _ERRORS = {
        "eigenvalues_not_available"  : _ERR_EIGENVALUES_NOT_AVAILABLE,
        "hamiltonian_not_available"  : _ERR_HAMILTONIAN_NOT_AVAILABLE,
        "hamiltonian_initialization" : _ERR_HAMILTONIAN_INITIALIZATION,
        "hamiltonian_build"          : _ERR_HAMILTONIAN_BUILD,
        "hilbert_space_not_provided" : _ERR_HILBERT_SPACE_NOT_PROVIDED
    }
        
    # ----------------------------------------------------------------------------------------------
    
    @staticmethod
    def _set_backend(backend: str):
        '''
        Get the backend, scipy, and random number generator for the backend.
        
        Args:
            backend (str) : The backend to use.
        
        Returns:
            tuple : The backend, scipy, and random number generator for the backend.
        '''
        if isinstance(backend, str):
            bck = get_backend(backend, scipy=True, random=True)
            if isinstance(bck, tuple):
                _backend, _backend_sp = bck[0], bck[1]
                if isinstance(bck[2], tuple):
                    _rng, _rng_k = bck[2][0], bck[2][1]
                else:
                    _rng, _rng_k = bck[2], None
            else:
                _backend, _backend_sp   = bck, None
                _rng, _rng_k            = None, None
            return backend, _backend, _backend_sp, (_rng, _rng_k)
        if _JAX_AVAILABLE and backend == 'default':
            _backendstr = 'jax'
        else:
            _backendstr = 'np'
        return Hamiltonian._set_backend(_backendstr)
    
    # ----------------------------------------------------------------------------------------------
    
    def __init__(self,
                hilbert_space   : HilbertSpace,
                is_sparse       : bool  =   True,
                dtype                   =   None,
                backend         : str   =   'default',
                **kwargs):
        '''
        Initialize the Hamiltonian class.
        
        Args:
            hilbert_space (HilbertSpace)    : The Hilbert space of the system.
            is_sparse (bool)                : A flag to indicate whether the Hamiltonian is sparse or not.
            dtype (data-type)               : The data type of the Hamiltonian matrix.
            **kwargs                        : Additional arguments.
        '''
        
        self._backendstr, self._backend, self._backend_sp, (self._rng, self._rng_k) = Hamiltonian._set_backend(backend)
        self._is_jax        = _JAX_AVAILABLE and self._backend != np
        self._is_numpy      = not self._is_jax
        self._is_sparse     = is_sparse
        
        # get the backend, scipy, and random number generator for the backend
        self._dtypeint      = self._backend.int64
        self._hilbert_space = hilbert_space
        self._dtype         = dtype if dtype is not None else self._hilbert_space.dtype
        if self._hilbert_space is None:
            self._lattice   = kwargs.get("lattice", None)
            if self._lattice is None and kwargs.get('ns', None) is None:
                raise ValueError(Hamiltonian._ERR_HILBERT_SPACE_NOT_PROVIDED)
            # otherwise do that
            if self._lattice is not None:
                self._hilbert_space = HilbertSpace(lattice = self._lattice)
            else:
                self._hilbert_space = HilbertSpace(ns = kwargs.get('ns', None))
                    
        # get the lattice and the number of sites
        self._lattice       = self._hilbert_space.get_lattice()
        self._nh            = self._hilbert_space.get_Nh()
        self._ns            = self._hilbert_space.get_Ns()
        self._startns       = 0 # for starting hamil calculation
        self._logger        = self._hilbert_space.logger
        
        # for the Hamiltonian matrix properties, and energy properties    
        self._av_en_idx     = 0
        self._av_en         = 0.0
        self._std_en        = 0.0
        self._min_en        = 0.0
        self._max_en        = 0.0
        
        # for the matrix representation of the Hamiltonian
        self._hamil         = None
        self._eig_vec       = None
        self._eig_val       = None
        self._krylov        = None
        self._name          = "Hamiltonian"
        self._max_local_ch  = 1 # maximum number of local changes - through the loc_energy function
        self._max_local_ch_o= self._max_local_ch # maximum number of local changes - through the operator
        
        #! set the local energy functions and the corresponding methods
        self._ops_nmod_nosites      = [[] for _ in range(self.ns)]      # operators that do not modify the state and do not act on any site (through the function call)
        self._ops_nmod_sites        = [[] for _ in range(self.ns)]      # operators that do not modify the state and act on a given site(s)
        self._ops_mod_nosites       = [[] for _ in range(self.ns)]      # operators that modify the state and do not act on any site (through the function call)
        self._ops_mod_sites         = [[] for _ in range(self.ns)]      # operators that modify the state and act on a given site(s)
        self._loc_energy_int_fun    : Optional[Callable]    = None
        self._loc_energy_np_fun     : Optional[Callable]    = None
        self._loc_energy_jax_fun    : Optional[Callable]    = None
        
    # ----------------------------------------------------------------------------------------------
    
    def _log(self, msg : str, log = 'info', lvl : int = 0, color : str = "white"):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (int) : The logging level. Default is 'info'.
            lvl (int) : The level of the message.
        """
        msg = f"[{self._name}] {msg}"
        self._hilbert_space.log(msg, log = log, lvl = lvl, color = color)
    
    def __str__(self):
        '''
        Returns the string representation of the Hamiltonian class.
        
        Returns:
            str : The string representation of the Hamiltonian class.
        '''
        return f"Hamiltonian with {self._nh} elements and {self._ns} modes."
    
    def __repr__(self):
        '''
        Returns the detailed string representation of the Hamiltonian class.

        Provides information about the number of elements and modes, whether a Hilbert space is provided,
        details about the associated lattice, and the computational backend.
        '''
        hilbert_info = "with HilbertSpace" if self._hilbert_space is not None else "without HilbertSpace"
        lattice_info = f"on lattice {self._lattice}" if self._lattice is not None else "without lattice"
        backend_info = f"using backend {self._backendstr}"
        return (f"Hamiltonian: {self._nh} elements, {self._ns} modes, "
            f"{hilbert_info}, {lattice_info}, {backend_info}.")
    
    # ----------------------------------------------------------------------------------------------
    
    def randomize(self, **kwargs):
        ''' Randomize the Hamiltonian matrix.'''
        raise NotImplementedError("Randomization is not implemented for this Hamiltonian class.")
    
    def clear(self):
        '''
        Clears the Hamiltonian matrix.
        '''
        self._hamil      = None
        self._eig_vec    = None
        self._eig_val    = None
        self._krylov     = None
        self._av_en      = None
        self._std_en     = None
        self._min_en     = None
        self._max_en     = None
        self._av_en_idx  = None
        self._log("Hamiltonian cleared...", lvl = 2, color = 'blue')
    
    # ----------------------------------------------------------------------------------------------
    #! Getter methods
    # ----------------------------------------------------------------------------------------------
    
    @property
    def dtype(self):
        '''
        Returns the data type of the Hamiltonian matrix.
        '''
        return self._dtype
    
    @property
    def inttype(self):
        ''' Integer types for the model '''
        return self._dtypeint
    
    @property
    def backend(self):
        ''' Returns string backend '''
        return self._backendstr

    @property
    def quadratic(self):
        '''
        Returns a flag indicating whether the Hamiltonian is quadratic or not.
        '''
        return not self.manybody
    
    def is_quadratic(self):
        '''
        Returns a flag indicating whether the Hamiltonian is quadratic or not.
        '''
        return self.quadratic
    
    @property
    def manybody(self):
        '''
        Returns a flag indicating whether the Hamiltonian is many-body or not.
        '''
        return not self.quadratic
    
    def is_manybody(self):
        '''
        Returns a flag indicating whether the Hamiltonian is many-body or not.
        '''
        return self.manybody
    
    @property
    def sparse(self):
        '''
        Returns a flag indicating whether the Hamiltonian is sparse or not.
        '''
        return self._is_sparse
    
    def is_sparse(self):
        '''
        Returns a flag indicating whether the Hamiltonian is sparse or not.
        '''
        return self.sparse

    @property
    def max_local_changes(self):
        '''
        Returns the maximum number of local changes.
        '''
        return self._max_local_ch

    @property
    def name(self):
        '''
        Returns the name of the Hamiltonian.
        '''
        return self._name
    
    @name.setter
    def name(self, name : str):
        '''
        Sets the name of the Hamiltonian.
        
        Args:
            name (str) : The name of the Hamiltonian.
        '''
        self._name = name
    
    # --- LATTICE BASED
    
    @property
    def ns(self):
        '''
        Returns the number of sites.
        '''
        return self._lattice.ns if self._lattice is not None else self._ns

    @property
    def sites(self):
        '''
        Returns the number of sites.
        '''
        return self.ns
    
    @property
    def lattice(self):
        '''
        Returns the lattice associated with the Hamiltonian.
        '''
        return self._lattice
    
    # --- HILBERT BASED
    
    @property
    def modes(self):
        '''
        Returns the modes of the Hamiltonian.
        '''
        return self._hilbert_space.get_modes() if self._hilbert_space is not None else None
    
    @property
    def hilbert_space(self):
        '''
        Returns the Hilbert space associated with the Hamiltonian.
        '''
        return self._hilbert_space
    
    @property
    def hilbert_size(self):
        '''
        Returns the number of sites in the Hilbert space.
        '''
        return self._nh
    
    # --- EIGENVALUES AND EIGENVECTORS
    
    @property
    def hamil(self):
        '''
        Returns the Hamiltonian matrix.
        '''
        return self._hamil
    
    @hamil.setter
    def hamil(self, hamil):
        '''
        Sets the Hamiltonian matrix.
        
        Args:
            hamil : The Hamiltonian matrix.
        '''
        self._hamil = hamil
    
    @property
    def eig_vec(self):
        '''
        Returns the eigenvectors of the Hamiltonian.
        '''
        return self._eig_vec
    
    @property
    def eig_val(self):
        '''
        Returns the eigenvalues of the Hamiltonian.
        '''
        return self._eig_val
    
    @property
    def krylov(self):
        '''
        Returns the Krylov subspace vectors.
        '''
        return self._krylov
    
    # --- ENERGY PROPERTIES
    
    @property
    def av_en(self):
        '''
        Returns the average energy of the Hamiltonian.
        '''
        return self._av_en
    
    @property
    def std_en(self):
        '''
        Returns the standard deviation of the energy of the Hamiltonian.
        '''
        return self._std_en
    
    @property
    def min_en(self):
        '''
        Returns the minimum energy of the Hamiltonian.
        '''
        return self._min_en
    
    @property
    def max_en(self):
        '''
        Returns the maximum energy of the Hamiltonian.
        '''
        return self._max_en
    
    @property
    def diag(self):
        '''
        Returns the diagonal of the Hamiltonian matrix.
        Distinguish between JAX and NumPy/SciPy. 
        '''
        
        if _JAX_AVAILABLE and self._backend != np:
            if isinstance(self._hamil, BCOO):
                return self._hamil.diagonal()
            elif isinstance(self._hamil, jnp.ndarray):
                return jnp.diag(self._hamil)
            else:
                # dunnno what to do here
                return None
        else:
            return self._hamil.diagonal()
    
    # ----------------------------------------------------------------------------------------------
    #! Local energy getters
    # ----------------------------------------------------------------------------------------------
    
    @property
    def fun_int(self):
        '''
        Returns the local energy of the Hamiltonian - integer representation
        '''
        return self._loc_energy_int_fun
    
    def get_loc_energy_int_fun(self):
        '''
        Returns the local energy of the Hamiltonian
        
        Returns:
            A function that takes an integer k and returns the local energy for an integer representation.
        '''
        return self._loc_energy_int_fun
    
    @property
    def fun_npy(self):
        '''
        Returns the local energy of the Hamiltonian - NumPy representation
        '''
        return self._loc_energy_np_fun
    
    def get_loc_energy_np_fun(self):
        '''
        Returns the local energy of the Hamiltonian
        Returns:
            A function that takes an integer k and returns the local energy for a NumPy representation.
        '''
        return self._loc_energy_np_fun
    
    @property
    def fun_jax(self):
        '''
        Returns the local energy of the Hamiltonian - JAX representation
        '''
        return self._loc_energy_jax_fun
    
    def get_loc_energy_jax_fun(self):
        '''
        Returns the local energy of the Hamiltonian
        Returns:
            A function that takes an integer k and returns the local energy for a JAX representation.
        '''
        return self._loc_energy_jax_fun
    
    # ----------------------------------------------------------------------------------------------
    
    def get_loc_energy_arr_fun(self, backend: str = 'default'):
        '''
        Returns the local energy of the Hamiltonian
        Returns:
            A function that takes an integer k and returns the local energy for an array representation in
            a given backend - either NumPy or JAX.
        '''
        if (backend == 'default' or backend == 'jax' or backend == 'jnp') and _JAX_AVAILABLE:
            return self.fun_jax
        return self.fun_npy
    
    # ----------------------------------------------------------------------------------------------
    #! Memory properties
    # ----------------------------------------------------------------------------------------------
    
    @property
    def h_memory(self):
        """
        Returns the memory used by the Hamiltonian matrix in bytes.
        Works for both dense and sparse representations and for NumPy and JAX.
        """
        # Dense matrix: use nbytes if available, otherwise compute from shape.
        self._log(f"Checking the memory used by the Hamiltonian matrix of type {type(self._hamil)}", lvl=1)
        
        if not self._is_sparse:
            if hasattr(self._hamil, "nbytes"):
                return self._hamil.nbytes
            else:
                return int(np.prod(self._hamil.shape)) * self._hamil.dtype.itemsize
        self._log("It is not a dense matrix...", lvl=2)
        # Sparse matrix:
        # For NumPy (or when JAX is unavailable) we assume a scipy sparse matrix (e.g. CSR)
        if self._backend == np or not _JAX_AVAILABLE:
            memory = 0
            for attr in ('data', 'indices', 'indptr'):
                if hasattr(self._hamil, attr):
                    arr = getattr(self._hamil, attr)
                    if hasattr(arr, 'nbytes'):
                        memory += arr.nbytes
                    else:
                        memory += int(np.prod(arr.shape)) * arr.dtype.itemsize
            return memory
        # For JAX sparse matrices (e.g. BCOO), we assume they have data and indices attributes.
        data_arr        = self._hamil.data
        indices_arr     = self._hamil.indices
        if hasattr(data_arr, 'nbytes'):
            data_bytes = data_arr.nbytes
        else:
            data_bytes = int(np.prod(data_arr.shape)) * data_arr.dtype.itemsize
        if hasattr(indices_arr, 'nbytes'):
            indices_bytes = indices_arr.nbytes
        else:
            indices_bytes = int(np.prod(indices_arr.shape)) * indices_arr.dtype.itemsize
        return data_bytes + indices_bytes

    @property
    def h_memory_gb(self):
        """
        Returns the memory used by the Hamiltonian matrix in gigabytes.
        """
        return self.h_memory / (1024.0 ** 3)

    @property
    def eigenvalues_memory(self):
        """
        Returns the memory used by the eigenvalues array in bytes.
        Assumes the eigenvalues are stored in a dense array.
        """
        if self._eig_val is None:
            return 0
        if hasattr(self._eig_val, "nbytes"):
            return self._eig_val.nbytes
        else:
            return int(np.prod(self._eig_val.shape)) * self._eig_val.dtype.itemsize

    @property
    def eigenvalues_memory_gb(self):
        """
        Returns the memory used by the eigenvalues array in gigabytes.
        """
        return self.eigenvalues_memory / (1024.0 ** 3)

    @property
    def eigenvectors_memory(self):
        """
        Returns the memory used by the eigenvectors array in bytes.
        Assumes the eigenvectors are stored in a dense array.
        """
        # Note: If your eigenvectors are stored in a separate attribute (e.g. self._eig_vec), adjust accordingly.
        if self._eig_vec is None:
            return 0
        if hasattr(self._eig_vec, "nbytes"):
            return self._eig_vec.nbytes
        else:
            return int(np.prod(self._eig_vec.shape)) * self._eig_vec.dtype.itemsize

    @property
    def eigenvectors_memory_gb(self):
        """
        Returns the memory used by the eigenvectors array in gigabytes.
        """
        return self.eigenvectors_memory / (1024.0 ** 3)

    @property
    def memory(self):
        """
        Returns the total memory used by the Hamiltonian, eigenvalues, and eigenvectors in bytes.
        """
        return self.h_memory + self.eigenvalues_memory + self.eigenvectors_memory

    @property
    def memory_gb(self):
        """
        Returns the total memory used by the Hamiltonian, eigenvalues, and eigenvectors in gigabytes.
        """
        return self.memory / (1024.0 ** 3)
        
    # ----------------------------------------------------------------------------------------------
    #! Standard getters
    # ----------------------------------------------------------------------------------------------
    
    def get_mean_lvl_spacing(self, use_npy = True):
        '''
        Returns the mean level spacing of the Hamiltonian. The mean level spacing is defined as the
        average difference between consecutive eigenvalues.
        
        Returns:
            float : The mean level spacing of the Hamiltonian.
        '''
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        if (not _JAX_AVAILABLE or self._backend == np or use_npy):
            return self._backend.mean(self._backend.diff(self._eig_val))
        return hjm.mean_level_spacing(self._eig_val)
    
    def get_bandwidth(self):
        '''
        Returns the bandwidth of the Hamiltonian. The bandwidth is defined as the difference between
        the highest and the lowest eigenvalues - values are sorted in ascending order.
        '''
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        return self._eig_val[-1] - self._eig_val[0]
    
    def get_energywidth(self, use_npy = True):
        '''
        Returns the energy width of the Hamiltonian. The energy width is defined as trace of the
        Hamiltonian matrix squared.
        '''
        if self._hamil.size == 0:
            raise ValueError(Hamiltonian._ERR_HAMILTONIAN_NOT_AVAILABLE)
        if (not _JAX_AVAILABLE or self._backend == np or use_npy):
            return self._backend.trace(self._backend.dot(self._hamil, self._hamil))
        return hjm.energy_width(self._hamil)
    
    def get_eigvec(self, *args):
        '''
        Returns the eigenvectors of the Hamiltonian 
        - one can take all of them (matrix (Nh x Nh), each column is an eigenvector)
        - one can take one of them
        - one can take element
        '''
        if len(args) == 0:
            return self._eig_vec
        elif len(args) == 1 and len(self._eig_vec.shape) == 2:
            return self._eig_vec[:, args[0]]
        elif len(args) == 2 and len(self._eig_vec.shape) == 2:
            return self._eig_vec[args[0], args[1]]
        else:
            raise ValueError("Invalid arguments provided for eigenvector retrieval.")
    
    def get_eigval(self, *args):
        '''
        Returns the eigenvalues of the Hamiltonian
        - one can take all of them (a vector in ascending order)
        - one can take a single value
        '''
        if len(args) == 0:
            return self._eig_val
        elif len(args) == 1 and len(self._eig_val) > 0:
            return self._eig_val[args[0]]
        else:
            raise ValueError("Invalid arguments provided for eigenvalue retrieval.")

    # ----------------------------------------------------------------------------------------------
    #! Initialization methods
    # ----------------------------------------------------------------------------------------------
    
    def to_dense(self):
        '''
        Converts the Hamiltonian matrix to a dense matrix.
        '''
        self._is_sparse = False
        self._log("Converting the Hamiltonian matrix to a dense matrix... Run build...", lvl = 1)
        self.clear()
        
    def to_sparse(self):
        '''
        Converts the Hamiltonian matrix to a sparse matrix.
        '''
        self._is_sparse = True
        self._log("Converting the Hamiltonian matrix to a sparse matrix... Run build...", lvl = 1)
        self.clear()
    
    # ----------------------------------------------------------------------------------------------
    
    def init(self, use_numpy : bool = False):
        '''
        Initializes the Hamiltonian matrix. Uses Batched-coordinate (BCOO) sparse matrices if JAX is
        used, otherwise uses NumPy arrays. The Hamiltonian matrix is initialized to be a matrix of
        zeros if the Hamiltonian is not sparse, otherwise it is initialized to be an empty sparse
        matrix.
        
        Parameters:
            use_numpy (bool) : A flag indicating whether to use NumPy or JAX.
        '''
        self._log("Initializing the Hamiltonian matrix...", lvl = 2, log = "debug")
        
        jax_maybe_avail = _JAX_AVAILABLE and self._backend != np
        if jax_maybe_avail and use_numpy:
            self._log("JAX is available but NumPy is forced...", lvl = 3)
        
        if self.sparse:
            self._log("Initializing the Hamiltonian matrix as a sparse matrix...", lvl = 3, log = "debug")
            
            # --------------------------------------------------------------------------------------
            
            if not jax_maybe_avail or use_numpy:
                self._log("Initializing the Hamiltonian matrix as a CSR sparse matrix...", lvl = 3, log = "debug")
                self._hamil = sp.sparse.csr_matrix((self._nh, self._nh), dtype = self._dtype)
            else:
                self._log("Initializing the Hamiltonian matrix as a sparse matrix...", lvl = 3, log = "debug")
                # Create an empty sparse Hamiltonian matrix using JAX's BCOO format
                indices     = self._backend.zeros((0, 2), dtype=DEFAULT_INT_TYPE)
                data        = self._backend.zeros((0,), dtype=self._dtype)
                self._hamil = BCOO((data, indices), shape=(self._nh, self._nh))
                
            # --------------------------------------------------------------------------------------
            
        else:
            self._log("Initializing the Hamiltonian matrix as a dense matrix...", lvl = 3, log = "debug")
            if not _JAX_AVAILABLE or self._backend == np:
                self._hamil     = self._backend.zeros((self._nh, self._nh), dtype=self._dtype)
            else:
                # do not initialize the Hamiltonian matrix
                self._hamil     = None
        self._log("Hamiltonian matrix initialized.", lvl = 3, color = "green", log = "debug")
    
    # ----------------------------------------------------------------------------------------------
    
    def _set_some_coupling(self, coupling: Union[list, np.ndarray, float, complex, int, str]):
        '''
        Distinghuishes between different initial values for the coupling and returns it.
        One distinguishes between:
            - a full vector of a correct size
            - single value 
            - random string
        ---
        Parameters:
            - coupling : some coupling to be set
        ---
        Returns:
            array to be used latter with corresponding couplings
        '''
        if isinstance(coupling, list) and len(coupling) == self.ns:
            return self._backend.array(coupling)
        elif isinstance(coupling, (float, int, complex)):
            return self._backend.array([coupling] * self.ns)
        elif isinstance(coupling, str):
            return random_vector(self.ns, coupling, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_COUP_VEC_SIZE)
    
    # ----------------------------------------------------------------------------------------------
    #! Single particle Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------
    
    def _hamiltonian_single_particle(self):
        '''
        Generates the Hamiltonian matrix whenever the Hamiltonian is single-particle. 
        '''
        #!TODO implement this through the Hilbert space!
        pass
    
    # ----------------------------------------------------------------------------------------------
    #! Many body Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------

    def __hamiltonian_validate(self):
        ''' Check if the Hamiltonian matrix is valid. '''
        if self._hamil is None:
            self._log("Hamiltonian matrix is not initialized.", lvl=3, color="red", log = "debug")
        else:
            valid   = False
            # For dense matrices (NumPy/JAX ndarray) which have the 'size' attribute.
            if hasattr(self._hamil, "size"):
                if self._hamil.size > 0:
                    valid = True
            # For SciPy sparse matrices: check the number of nonzero elements.
            elif hasattr(self._hamil, "nnz"):
                if self._hamil.nnz > 0:
                    valid = True
            # For JAX sparse matrices (e.g., BCOO): verify if the data array has entries.
            elif hasattr(self._hamil, "data") and hasattr(self._hamil, "indices"):
                if self._hamil.data.shape[0] > 0:
                    valid = True
            
            if valid:
                self._log("Hamiltonian matrix calculated and valid.", lvl=3, color="green", log = "debug")
            else:
                self._log("Hamiltonian matrix calculated but empty or invalid.", lvl=3, color="red", log = "debug")
                self._hamil = None

    # ----------------------------------------------------------------------------------------------

    def _transform_to_backend(self):
        '''
        Transforms the Hamiltonian matrix to the backend.
        '''
        self._hamil     = linalg.transform_backend(self._hamil, self._is_sparse, self._backend)
        self._eig_val   = linalg.transform_backend(self._eig_val, False, self._backend)
        self._eig_vec   = linalg.transform_backend(self._eig_vec, False, self._backend)
        self._krylov    = linalg.transform_backend(self._krylov, False, self._backend)
        self._log(f"Hamiltonian matrix transformed to the backend {self._backendstr}", lvl=2, color="green")
        
    # ----------------------------------------------------------------------------------------------

    def build(self, verbose: bool = False, use_numpy: bool = False):
        '''
        Builds the Hamiltonian matrix.
        
        Args:
            verbose (bool) : A flag to indicate whether to print the progress of the build.
        '''
        if verbose:
            self._log("Building the Hamiltonian matrix...", lvl = 1)
            
        ################################
        # Initialize the Hamiltonian
        ################################
        init_start = time.perf_counter()
        try:
            self.init(use_numpy)
        except Exception as e:
            raise ValueError(f"{Hamiltonian._ERR_HAMILTONIAN_INITIALIZATION} : {str(e)}") from e
    
        if hasattr(self._hamil, "block_until_ready"):
            self._hamil = self._hamil.block_until_ready()
            
        init_duration = time.perf_counter() - init_start
        if verbose:
            self._log(f"Initialization completed in {init_duration:.6f} seconds", lvl = 2)
        
        ################################
        # Build the Hamiltonian matrix
        ################################
        ham_start = time.perf_counter()
        try:
            self._hamiltonian(use_numpy)
        except Exception as e:
            raise ValueError(f"{Hamiltonian._ERR_HAMILTONIAN_BUILD} : {str(e)}") from e

        ham_duration = time.perf_counter() - ham_start
        if self._hamil is not None and self._hamil.size > 0:
            if verbose:
                self._log(f"Hamiltonian matrix built in {ham_duration:.6f} seconds.", lvl = 1)
        else:
            raise ValueError(f"{Hamiltonian._ERR_HAMILTONIAN_BUILD} : The Hamiltonian matrix is empty or invalid.")

    # ----------------------------------------------------------------------------------------------
    #! Local energy methods - Abstract methods
    # ----------------------------------------------------------------------------------------------
    
    def loc_energy_int(self, k_map : int, i : int):
        '''
        Calculates the local energy.  MUST return NumPy arrays.

        Parameters:
            k_map (int) : The mapping of the k'th element.
            i (int)     : The i'th site.

        Returns:
            Tuple[np.ndarray, np.ndarray]   :  (row_indices, values)
                - row_indices               :  The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        '''
        if self._loc_energy_int_fun is None:
            self._set_local_energy_functions()
        return self._loc_energy_int_fun(k_map, i)
    
    def loc_energy_arr_jax(self, k : Union[int, np.ndarray]):
        '''
        Calculates the local energy based on the Hamiltonian. This method should be implemented by subclasses.
        Uses an array as a state input.
        Returns:
            Tuple[np.ndarray, np.ndarray]   :  (row_indices, values)
                - row_indices               :  The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        '''
        return self._loc_energy_jax_fun(k)
    
    def loc_energy_arr_np(self, k : Union[np.ndarray]):
        '''
        Calculates the local energy based on the Hamiltonian. This method should be implemented by subclasses.
        Uses an array as a state input.
        Returns:
            Tuple[np.ndarray, np.ndarray]   :  (row_indices, values)
                - row_indices               :  The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        '''
        return self._loc_energy_np_fun(k)
        
    def loc_energy_arr(self, k : Union[int, np.ndarray]) -> Tuple[List[int], List[int]]:
        '''
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by
        subclasses to provide a specific implementation.
        This is meant to check how does the Hamiltonian act on a state.
        Parameters:
            k (Union[int, np.ndarray]) : The k'th element of the Hilbert space - may use mapping if necessary.
        Returns:
            Tuple[List[int], List[int]] :  (row_indices, values)
                - row_indices               : The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        '''
        if self._is_jax or isinstance(k, jnp.ndarray):
            if self._loc_energy_jax_fun is None:
                self._set_local_energy_functions()
            return self.loc_energy_arr_jax(k)
        
        # go!
        if self._loc_energy_np_fun is None:
            self._set_local_energy_functions()
        return self.loc_energy_arr_np(k)
    
    def loc_energy(self, k : Union[int, np.ndarray], i : int = 0):
        '''
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by 
        subclasses to provide a specific implementation.
        
        This is meant to check how does the Hamiltonian act on a state at a given site.
        
        Parameters:
            k (Union[int, Backend.ndarray])         : The k'th element of the Hilbert space - may use mapping if necessary.
            i (int)                                 : The i'th site.
        '''
        if isinstance(k, (int, np.integer, int, jnp.integer)):
            return self.loc_energy_int(self._hilbert_space[k], i)
        elif isinstance(k, List):
            # concatenate the results
            rows, cols, data = [], [], []
            for k_i in k:
                # run the local energy calculation for a single element
                new_rows, new_cols, new_data = self.loc_energy(k_i, i)
                rows.extend(new_rows)
                cols.extend(new_cols)
                data.extend(new_data)
            return rows, cols, data
        # otherwise, it is an array (no matter which backend)
        return self.loc_energy_arr(k)

    # ----------------------------------------------------------------------------------------------
    # ! Hamiltonian matrix calculation
    # ----------------------------------------------------------------------------------------------

    def _hamiltonian(self, use_numpy : bool = False):
        '''
        Generates the Hamiltonian matrix. The diagonal elements are straightforward to calculate,
        while the off-diagonal elements are more complex and depend on the specific Hamiltonian.
        It iterates over the Hilbert space to calculate the Hamiltonian matrix. 
        
        Note: This method may be overridden by subclasses to provide a more efficient implementation
        '''
        if self._hilbert_space is None or self._nh == 0:
            raise ValueError(Hamiltonian._ERR_HILBERT_SPACE_NOT_PROVIDED)

        # -----------------------------------------------------------------------------------------
        matrix_type = "sparse" if self.sparse else "dense"
        self._log(f"Calculating the {matrix_type} Hamiltonian matrix...", lvl=1, color="blue", log = 'debug')
        # -----------------------------------------------------------------------------------------
        
        # Check if JAX is available and the backend is not NumPy
        jax_maybe_av = _JAX_AVAILABLE and self._backend != np
        
        # Choose implementation based on backend availability.sym_eig_py
        if not jax_maybe_av or use_numpy:
            self._log("Calculating the Hamiltonian matrix using NumPy...", lvl=2, log = 'info')
            
            # Calculate the Hamiltonian matrix using the NumPy implementation.
            self._hamil = operator_create_np(
                ns                  =   self._ns,
                hilbert_space       =   self._hilbert_space,
                local_fun           =   self._loc_energy_int_fun,
                max_local_changes   =   self._max_local_ch,
                is_sparse           =   self._is_sparse,
                start               =   self._startns,
                dtype               =   self._dtype
            )
        else:
            raise ValueError("JAX not yet implemented for the build...")
            self._log("Calculating the Hamiltonian matrix using JAX...", lvl=2, log = 'info')

            # Choose the correct local energy function for JAX.
            # local_fun = jit(self.loc_energy_int_jax) if self._is_sparse else self.loc_energy_ham

            # Calculate the Hamiltonian matrix using the JAX implementation.
            # self._hamil = hamiltonian_functional_jax(
            #     self._ns,
            #     self._hilbert_space,
            #     loc_energy          = local_fun,
            #     is_sparse           = self._is_sparse,
            #     max_local_changes   = self._max_local_ch,
            #     start               = self._startns,
            #     dtype               = self._dtype)
            
            # # Ensure the computation completes.
            # if not self._is_sparse:
            #     self._hamil = self._hamil.block_until_ready()

        # Check if the Hamiltonian matrix is calculated and valid using various backend checks
        self.__hamiltonian_validate()
    
    # ----------------------------------------------------------------------------------------------
    #! Calculators
    # ----------------------------------------------------------------------------------------------
    
    def _calculate_av_en(self):
        '''
        Calculates the properties of the Hamiltonian matrix that are related to the energy.
        '''
        
        if self._eig_val is None or self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        
        self._av_en     = self._backend.mean(self._eig_val)
        self._min_en    = self._backend.min(self._eig_val)
        self._max_en    = self._backend.max(self._eig_val)
        self._std_en    = self._backend.std(self._eig_val)
        self._nh        = self._eig_val.size
        
        # average energy index
        self._av_en_idx = self._backend.argmin(self._backend.abs(self._eig_val - self._av_en))
    
    def calculate_en_idx(self, en : float):
        '''
        Calculates the index of the energy level closest to the given energy.
        
        Args:
            en (float) : The energy level.
        
        Returns:
            int : The index of the energy level closest to the given energy.
        '''
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        return self._backend.argmin(self._backend.abs(self._eig_val - en))
    
    # ----------------------------------------------------------------------------------------------
    #! Diagonalization methods
    # ----------------------------------------------------------------------------------------------

    def diagonalize(self, verbose: bool = False, **kwargs):
        """
        Diagonalizes the Hamiltonian matrix using one of several methods.
        
        Supported methods:
        - 'eigh'        : Full diagonalization using a dense eigen-solver.
        - 'lanczos'     : Iterative Lanczos diagonalization via SciPy's eigsh.
        - 'shift-invert': Iterative shift-invert diagonalization via SciPy's eigsh.
        
        Additional kwargs:
        - k (int)       : Number of eigenpairs to compute (default: 6 for Lanczos/shift-invert).
        - sigma (float) : Shift value for shift-invert (default: 0.0).
        - which (str)   : Eigenvalue selection criteria ('SA' for Lanczos, 'LM' for shift-invert).
        
        Updates:
        - self._eig_val: Eigenvalues.
        - self._eig_vec: Eigenvectors.
        """
        diag_start  = time.perf_counter()
        method      = kwargs.get("method", "standard")
        backend     = self._backend if not isinstance(self._hamil, np.ndarray) and not sp.sparse.isspmatrix(self._hamil) else np
        if verbose:
            self._log(f"Diagonalization started using ({method})...", lvl = 1)
        try:
            if self._is_sparse or method.lower() in ["lanczos", "shift-invert"]:
                self._eig_val, self._eig_vec = linalg.eigsh(self._hamil, method, backend, **kwargs)
            else:
                self._eig_val, self._eig_vec = linalg.eigh(self._hamil, backend, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to diagonalize the Hamiltonian using method '{method}' : {e}") from e
        
        if _JAX_AVAILABLE:
            if hasattr(self._eig_val, "block_until_ready"):
                self._eig_val = self._eig_val.block_until_ready()
            if hasattr(self._eig_vec, "block_until_ready"):
                self._eig_vec = self._eig_vec.block_until_ready()
        
        diag_duration = time.perf_counter() - diag_start
        if verbose:
            self._log(f"Diagonalization ({method}) completed in {diag_duration:.6f} seconds.", lvl = 2)
        self._calculate_av_en()

    # ----------------------------------------------------------------------------------------------
    #! Setters
    # ----------------------------------------------------------------------------------------------
    
    def _set_hamil_elem(self, k, val, newk):
        '''
        Sets the element of the Hamiltonian matrix.
        
        Args:
            k (int) : The row index.
            val     : The value to set.
            newk    : The column index.
            
        We acted on the k'th state with the Hamiltonian which in turn gave us the newk'th state.
        First, we need to check the mapping of the k'th state as it may be the same as the newk'th state
        through the symmetries...
        
        After that, we check if they are the same and if they are we add the value to the diagonal element
        (k'th, not kMap as it only checks the representative).
        For instance, the mapping can be (0, 3, 5, 7) and k = 1, so mapping of 1 is 3 - we don't want to
        add the value to the 3'rd element of the Hamiltonian matrix but to the 1'st element.
        
        Otherwise, we need to find the representative of the newk'th state and add the value to the newk'th
        element of the Hamiltonian matrix. This is also used with the norm of a given state k as we need to 
        check in how many ways we can get to the newk'th state.
        '''
        
        try:
            set_operator_elem(self._hamil, self._hilbert_space, k, val, newk)
        except Exception as e:
            print(f"Error in _set_hamil_elem: Failed to set element at <newk(idx)|H|k>, newk={newk},k={k},value: {val}. Please verify that the indices and value are correct. Exception details: {e}")

    # ----------------------------------------------------------------------------------------------
    #! Energy related methods
    # ----------------------------------------------------------------------------------------------
    
    def reset_operators(self):
        '''
        Resets the Hamiltonian operators...
        '''
        self._ops_nmod_nosites      = [[] for _ in range(self.ns)]      # operators that do not modify the state and do not act on any site (through the function call)
        self._ops_nmod_sites        = [[] for _ in range(self.ns)]      # operators that do not modify the state and act on a given site(s)
        self._ops_mod_nosites       = [[] for _ in range(self.ns)]      # operators that modify the state and do not act on any site (through the function call)
        self._ops_mod_sites         = [[] for _ in range(self.ns)]      # operators that modify the state and act on a given site(s)
        self._loc_energy_int_fun    = None
        self._loc_energy_np_fun     = None
        self._loc_energy_jax_fun    = None
        self._max_local_ch          = self._max_local_ch_o
    
    def add(self, operator: Operator, multiplier: Union[float, complex, int], modifies: bool = False, sites = None):
        """
        Add an operator to the internal operator collections based on its locality.
        
        ---
        Parameters:
            operator: 
                The operator to be added. This can be any object representing an operation,
                typically in the context of a quantum system.
            sites (list[int]): 
                A list of site indices where the operator should act. If empty,
                the operator will be associated with site 0.
            multiplier (numeric): A scaling factor to be applied to the operator.
            is_local (bool, optional):
                Determines the type of operator. If True, the operator is
                considered local (i.e., it does not modify the state) and is
                appended to the local operator list. If False, it is added to
                the non-local operator list. Defaults to False.
        ---
        Behavior:
        
            - Determines the primary site for the operator based on the first element in the 'sites' list, or defaults to index 0 if 'sites' is empty.
            - Depending on the value of 'is_local', the operator is appended to either the local operator collection (_local_ops) or the non-local operator collection (_nonlocal_ops).
            - Logs a debug message indicating the addition of the operator along with its details.
        
        ---
        Returns:
            None
            
        --- 
        Example:
        >> operator    = sig_z
        >> sites       = [0, 1]
        >> hamiltonian.add(operator, sites, multiplier=1.0, is_local=True)
        >> This would add the operator 'sig_z' to the local operator list at site 0 with a multiplier of 1.0.
        """
        
        # if isinstance(operator, Callable):
            # if the operator is callable, we try to create the operator from it
            # raise ValueError("The operator is callable, but it should be an Operator object. TODO: implement this...")
        
        # check if the sites are provided, if one sets the operator, we would put it at a given site
        i           = 0 if (sites is None or len(sites) == 0) else sites[0]
        op_tuple    = create_add_operator(operator, multiplier, sites)
        modifies    = modifies or operator.modifies
        # if the operator is meant to be local, it does not modify the state
        if not modifies:
            if operator.type_acting == OperatorTypeActing.Global:
                self._ops_nmod_nosites[i].append((op_tuple))
                self._log(f"Adding non-modifying operator {operator} at site {i} (global) with multiplier {op_tuple[2]}", lvl = 2)
            else:
                self._ops_nmod_sites[i].append((op_tuple))
                self._log(f"Adding non-modifying operator {operator} at site {i} (sites: {str(op_tuple[1])}) with multiplier {op_tuple[2]}", lvl = 2)
        else:
            if operator.type_acting == OperatorTypeActing.Global:
                self._ops_mod_nosites[i].append((op_tuple))
                self._log(f"Adding modifying operator {operator} at site {i} (global) with multiplier {op_tuple[2]}", lvl = 2)
            else:
                self._ops_mod_sites[i].append((op_tuple))
                self._log(f"Adding modifying operator {operator} at site {i} (sites: {str(op_tuple[1])}) with multiplier {op_tuple[2]}", lvl = 2)

    def _set_local_energy_operators(self):
        '''
        This function is meant to be overridden by subclasses to set the local energy operators.
        The local energy operators are used to calculate the local energy of the Hamiltonian.
        Note:
            It is the internal function that knows about the structure of the Hamiltonian.
        '''
        self.reset_operators()
    
    def _set_local_energy_functions(self):
        """
        Private method that configures and sets local energy functions for different numerical backends.
        
        This method initializes three versions of the local energy function based on the available operator representations:
            - Integer operations: Constructs a version using the integer attributes (op.int) of both nonlocal and local operators.
            - NumPy operations: Constructs a version using the NumPy functions (op.npy) for numerical evaluations.
            - JAX operations: If JAX is available, constructs a version using the JAX functions (op.jax).
            
        The method uses the following wrapper functions to create the local energy functions:
        For each backend, the method:
        
            1. Iterates over the nonlocal and local operators for each site.
            2. Extracts the appropriate operator function (int, npy, or jax), along with corresponding sites and values.
            3. Wraps the extracted operators using the respective wrapper function (local_energy_int_wrap, local_energy_np_wrap, or local_energy_jax_wrap)
                to create the corresponding local energy function.
            4. Stores the resulting function into instance attributes (_loc_energy_int_fun, _loc_energy_np_fun, _loc_energy_jax_fun).
        
        --- 
        Note:
            - This function assumes that the instance has the attributes:
                - self.ns: number of sites.
                - self._nonlocal_ops: a list of nonlocal operator tuples for each site.
                - self._local_ops: a list of local operator tuples for each site.
            - JAX version is set only if the flag _JAX_AVAILABLE is True.
        """
        
        # set the integer functions
        try:
            operators_mod_int           =  [[(op.int, sites, vals) for (op, sites, vals) in self._ops_mod_sites[i]] for i in range(self.ns)]
            operators_mod_int_nsites    =  [[(op.int, sites, vals) for (op, sites, vals) in self._ops_mod_nosites[i]] for i in range(self.ns)]
            operators_nmod_int          =  [[(op.int, sites, vals) for (op, sites, vals) in self._ops_nmod_sites[i]] for i in range(self.ns)]
            operators_nmod_int_nsites   =  [[(op.int, sites, vals) for (op, sites, vals) in self._ops_nmod_nosites[i]] for i in range(self.ns)]
            self._loc_energy_int_fun    = local_energy_int_wrap(self.ns,
                                                    _op_mod_sites=operators_mod_int,
                                                    _op_mod_nosites=operators_mod_int_nsites,
                                                    _op_nmod_sites=operators_nmod_int,
                                                    _op_nmod_nosites=operators_nmod_int_nsites,
                                                    dtype=self._dtype)
        
        except Exception as e:
            self._log(f"Failed to set integer local energy functions: {e}", lvl=3, color="red", log='error')
            self._loc_energy_int_fun = None
        
        # set the NumPy functions
        try:
            # set the numpy functions
            operators_mod_np        = [[(op.npy, sites, vals) for (op, sites, vals) in self._ops_mod_sites[i]] for i in range(self.ns)]
            operators_mod_np_nsites = [[(op.npy, [], vals) for (op, _, vals) in self._ops_mod_nosites[i]] for i in range(self.ns)]
            operators_nmod_np       = [[(op.npy, sites, vals) for (op, sites, vals) in self._ops_nmod_sites[i]] for i in range(self.ns)]
            operators_nmod_np_nsites= [[(op.npy, [], vals) for (op, _, vals) in self._ops_nmod_nosites[i]] for i in range(self.ns)]
            self._loc_energy_np_fun = local_energy_np_wrap(self.ns,
                                                    operator_terms_list=operators_mod_np,
                                                    operator_terms_list_ns=operators_mod_np_nsites,
                                                    operator_terms_list_nmod=operators_nmod_np,
                                                    operator_terms_list_nmod_ns=operators_nmod_np_nsites,
                                                    n_max=self._max_local_ch, dtype=self._dtype)
        except Exception as e:
            self._log(f"Failed to set NumPy local energy functions: {e}", lvl=3, color="red", log='error')
            self._loc_energy_np_fun = None
            
        # set the jax functions
        if _JAX_AVAILABLE:
            try:
                operators_jax               = [[(op.jax, sites, vals) for (op, sites, vals) in self._ops_mod_sites[i]] for i in range(self.ns)]
                operators_jax_nosites       = [[(op.jax, None, vals) for (op, _, vals) in self._ops_mod_nosites[i]] for i in range(self.ns)]
                operators_local_jax         = [[(op.jax, sites, vals) for (op, sites, vals) in self._ops_nmod_sites[i]] for i in range(self.ns)]
                operators_local_jax_nosites = [[(op.jax, None, vals) for (op, _, vals) in self._ops_nmod_nosites[i]] for i in range(self.ns)]
                self._loc_energy_jax_fun    = local_energy_jax_wrap(self.ns,
                                                    operator_terms_list             = operators_jax,
                                                    operator_terms_list_ns          = operators_jax_nosites,
                                                    operator_terms_list_nmod        = operators_local_jax,
                                                    operator_terms_list_nmod_ns     = operators_local_jax_nosites,
                                                    n_max                           = self._max_local_ch,
                                                    dtype                           = self._dtype)
            except Exception as e:
                self._log(f"Failed to set JAX local energy functions: {e}", lvl=3, color="red", log='error')
        else:
            self._log("JAX is not available, skipping JAX local energy function setup.", lvl=3, color="yellow", log='warning')
            self._loc_energy_jax_fun   = None

        # log success
        
        self._max_local_ch              = max(self._max_local_ch_o,   max(len(op) for op in self._ops_mod_sites)    + \
                                                                    max(len(op) for op in self._ops_mod_nosites)  + \
                                                                    max(len(op) for op in self._ops_nmod_sites)   + \
                                                                    max(len(op) for op in self._ops_nmod_nosites))
        self._log(f"Max local changes set to {self._max_local_ch}", lvl=2, color="green", log='warning')
        self._log("Successfully set local energy functions...", lvl=2)

    def _local_energy_test(self, k_map = 0, i = 0):
        '''
        Tests the local energy calculation.
        
        Args:
            k_map (int) : The mapping of the k'th element.
            i (int)     : The i'th site.
        '''
        
        # integer function
        if self._loc_energy_int_fun is not None:
            idx, val = self._loc_energy_int_fun(k_map, i)
            self._log(f"{self._name} test_int({k_map},{i}): idx={idx}, vals={val}", lvl = 2, log = 'debug')
        
        # array functions
        if self._loc_energy_np_fun is not None:
            input_vec   = np.ones(self._ns, dtype = np.float32)
            idx, val    = self._loc_energy_np_fun(input_vec)
            self._log(f"{self._name} test_np({input_vec}): idx={idx}, vals={val}", lvl = 2, log = 'debug')
        if self._loc_energy_jax_fun is not None:
            input_vec   = jnp.ones(self._ns, dtype = jnp.float32)
            idx, val    = self._loc_energy_jax_fun(input_vec)
            self._log(f"{self._name} test_jax({input_vec}): idx={idx}, vals={val}", lvl = 2, log = 'debug')
    
    # ----------------------------------------------------------------------------------------------
    
# --------------------------------------------------------------------------------------------------

def test_generic_hamiltonian(ham: Hamiltonian, ns: int):
    '''
    Creates a generic Hamiltonian object based on the provided Hamiltonian class and number of sites.
    
    Args:
        ham (Hamiltonian):
            The Hamiltonian class to be used.
        ns (int):
            The number of sites.
    Returns:
    '''
    import Algebra.Operator.operators_spin as op_spin
    import general_python.common.binary as bin_mod

    sites   = [0, 2]

    #! GLOBAL

    #@ sig_x
    sig_x   = op_spin.sig_x(
        ns      = ns,
        sites   = sites
    )

    #@ sig_z
    sig_z   = op_spin.sig_z(
        ns      = ns,
        sites   = sites
    )

    #@ sig_x_0
    sig_z_0 = op_spin.sig_z(
        ns      = ns,
        sites   = [0]
    )

    #@ sig_x * 2
    sig_x_2     = 2 * sig_x

    #@ sig_z * sig_z_0
    sig_z_sig_z = sig_z_0 * sig_z 
    #! LOCAL
    sig_z_loc   = op_spin.sig_z(
        ns          = ns,
        type_act    = op_spin.OperatorTypeActing.Local
    )
    sig_x_loc   = op_spin.sig_x(
        ns          = ns,
        type_act    = op_spin.OperatorTypeActing.Local
    )

    #! CORRELATION
    sig_z_corr = op_spin.sig_z(
        ns          = ns,
        type_act    = op_spin.OperatorTypeActing.Correlation
    )

    # create a set of states and test it
    int_state   = np.random.randint(0, 2**ns)
    np_state    = np.random.choice([-1.0, 1.0], size=(ns,), replace=True).astype(np.float32)
    if _JAX_AVAILABLE:
        jnp_state   = jnp.array(np_state, dtype=jnp.float32)
    else:
        jnp_state   = np_state

    # print the operator names
    operators   = [sig_x, sig_z, sig_z_0, sig_x_2, sig_z_sig_z, sig_z_loc, sig_x_loc, sig_z_corr]
    for operator in operators:
        print(f"Operator: {operator.name}, {operator.type_acting}")
    # data = [
    #     ["Before", bin_mod.int2binstr(int_state, ns)],
    #     ["int_state", int_state],
    #     ["np_state", np_state],
    #     ["jnp_state", jnp_state]
    # ]
    # print(tabulate(data, headers=["State Name", "Value"], tablefmt="grid"))
    
    return (int_state, np_state, jnp_state), operators

# --------------------------------------------------------------------------------------------------