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

###################################################################################################
from Algebra.hilbert import HilbertSpace, set_operator_elem, Logger, Lattice
from Algebra.Operator.operator import Operator, OperatorTypeActing, create_add_operator
from Algebra.Operator.operator_matrix import operator_create_np
from Algebra.Hamil.hamil_types import *
from Algebra.Hamil.hamil_energy import local_energy_int_wrap, local_energy_np_wrap
###################################################################################################
import Algebra.Hamil.hamil_jit_methods as hjm
###################################################################################################
from general_python.algebra.ran_wrapper import random_vector
from general_python.algebra.utils import JAX_AVAILABLE, get_backend, ACTIVE_INT_TYPE, Array
import general_python.algebra.linalg as linalg

if JAX_AVAILABLE:
    import jax
    from jax import jit
    import jax.lax as lax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO, CSR
    from Algebra.Hamil.hamil_energy import local_energy_jax_wrap
    from Algebra.hilbert import process_matrix_elem_jax, process_matrix_batch_jax
else:
    import jax
    jax                     = None
    jnp                     = None
    lax                     = None
    BCOO                    = None
    CSR                     = None
    local_energy_jax_wrap   = None
    process_matrix_elem_jax = None
    process_matrix_batch_jax= None

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
    
    # Error messages for Hamiltonian class
    _ERR_EIGENVALUES_NOT_AVAILABLE  = "Eigenvalues are not available. Please diagonalize the Hamiltonian first."
    _ERR_HAMILTONIAN_NOT_AVAILABLE  = "Hamiltonian matrix is not available. Please build or initialize the Hamiltonian."
    _ERR_HAMILTONIAN_INITIALIZATION = "Failed to initialize the Hamiltonian matrix. Check Hilbert space, lattice, and parameters."
    _ERR_HAMILTONIAN_BUILD          = "Failed to build the Hamiltonian matrix. Ensure all operators and spaces are properly set."
    _ERR_HILBERT_SPACE_NOT_PROVIDED = "Hilbert space is not provided or is invalid. Please supply a valid HilbertSpace object."
    _ERR_NS_NOT_PROVIDED            = "'ns' (number of sites/modes) must be provided, e.g., via 'ns' kwarg or a Lattice object."
    _ERR_NEED_LATTICE               = "Lattice information is required but not provided. Please specify a lattice or number of sites."
    _ERR_COUP_VEC_SIZE              = "Invalid coupling vector size. Coupling must be a scalar, a string, or a list/array of length ns."
    _ERR_MODE_MISMATCH              = "Operation not supported for the current Hamiltonian mode (Many-Body/Quadratic). Check 'is_manybody' flag."
    
    _ERRORS = {
        "eigenvalues_not_available"  : _ERR_EIGENVALUES_NOT_AVAILABLE,
        "hamiltonian_not_available"  : _ERR_HAMILTONIAN_NOT_AVAILABLE,
        "hamiltonian_initialization" : _ERR_HAMILTONIAN_INITIALIZATION,
        "hamiltonian_build"          : _ERR_HAMILTONIAN_BUILD,
        "hilbert_space_not_provided" : _ERR_HILBERT_SPACE_NOT_PROVIDED,
        "need_lattice"               : _ERR_NEED_LATTICE,
        "coupling_vector_size"       : _ERR_COUP_VEC_SIZE,
        "mode_mismatch"              : _ERR_MODE_MISMATCH,
        "ns_not_provided"            : _ERR_NS_NOT_PROVIDED
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
        if JAX_AVAILABLE and backend == 'default':
            _backendstr = 'jax'
        else:
            _backendstr = 'np'
        return Hamiltonian._set_backend(_backendstr)
    
    # ----------------------------------------------------------------------------------------------
    
    def __init__(self,
                # concerns the definition of the system type
                is_manybody     : bool  =   True,                # True for many-body Hamiltonian, False for non-interacting
                hilbert_space   : Optional[HilbertSpace] = None, # Required if is_manybody=True
                ns              : Optional[int] = None,          # Number of sites/modes (if not provided, will be inferred from hilbert_space or lattice)
                lattice         : Optional[Union[str, List[int]]] = None, # Alternative way to specify ns and get the Hilbert space
                # concerns the matrix and computation
                is_sparse       : bool  =   True,                
                dtype                   =   None,
                backend         : str   =   'default',
                # logger and other kwargs
                logger          : Optional[Logger] = None,
                **kwargs):
        """
        Initialize the Hamiltonian class.

        Parameters
        ----------
        is_manybody : bool, optional
            If True, the Hamiltonian is treated as a many-body Hamiltonian.
            If False, it is treated as a non-interacting (single-particle) Hamiltonian. Default is True.
        hilbert_space : HilbertSpace or None, optional
            The Hilbert space object describing the system. Required if is_manybody=True.
        lattice : str or list of int or None, optional
            Lattice information or list of site indices. Used to infer the number of sites (ns) and optionally construct the Hilbert space.
        is_sparse : bool, optional
            If True, the Hamiltonian matrix is stored in a sparse format. Default is True.
        dtype : data-type, optional
            Data type for the Hamiltonian matrix elements. If None, inferred from Hilbert space or backend.
        backend : str, optional
            Computational backend to use ('default', 'np', 'jax', etc.). Default is 'default'.
        logger : Logger, optional
            Logger class, may be inherited from the Hilbert space
        **kwargs
            Additional keyword arguments, such as 'ns' (number of sites/modes), or 'lattice' for further customization.

        Raises
        ------
        ValueError
            If required information (such as Hilbert space or lattice) is missing or inconsistent.
        """
        
        self._backendstr, self._backend, self._backend_sp, (self._rng, self._rng_k) = Hamiltonian._set_backend(backend)
        self._is_jax                = JAX_AVAILABLE and self._backend != np
        self._is_numpy              = not self._is_jax
        self._is_sparse             = is_sparse
        self._is_manybody           = is_manybody
        self._is_quadratic          = not is_manybody
        self._particle_conserving   = False
        
        # get the backend, scipy, and random number generator for the backend
        self._dtypeint      = self._backend.int64
        self._dtype         = dtype
        self._hilbert_space = hilbert_space             # Hilbert space of the system, if any
        self._lattice       = lattice
        
        if ns is not None:
            # if the number of sites is provided, set it
            self._ns        = ns
            self._lattice   = lattice
        elif hilbert_space is not None:
            # if the Hilbert space is provided, get the number of sites
            self._ns        = hilbert_space.get_Ns()
            self._lattice   = hilbert_space.get_lattice()
            if self._dtype is None:
                self._dtype = hilbert_space.dtype
        elif lattice is not None:
            self._ns        = lattice.ns
            self._lattice   = lattice
        else:
            # if the number of sites is not provided, raise an error
            raise ValueError(Hamiltonian._ERR_NS_NOT_PROVIDED)
        
        if self._hilbert_space is None:
            # try to infer from lattice or number of sites
            if self._lattice is None:
                # if the lattice is not provided, create Hilbert space from number of sites
                if self._ns is None:
                    raise ValueError(Hamiltonian._ERR_NS_NOT_PROVIDED)
            self._hilbert_space = HilbertSpace(ns       = self._ns,
                                            lattice     = self._lattice,
                                            is_manybody = self._is_manybody,
                                            dtype       = self._dtype,
                                            backend     = self._backendstr,
                                            logger      = logger,
                                            **kwargs)
        else:
            # otherwise proceed 
            if self._is_manybody:
                if not self._hilbert_space._is_many_body:
                    raise ValueError(Hamiltonian._ERR_MODE_MISMATCH)
                self._hamil_sp = None
            else:
                if not self._hilbert_space._is_quadratic:
                    raise ValueError(Hamiltonian._ERR_MODE_MISMATCH)
                self._hamil = None
        
        if self._hilbert_space.get_Ns() != self._ns:
            raise ValueError(f"Ns mismatch: {self._hilbert_space.get_Ns()} != {self._ns}")
                
        # if the Hilbert space is provided, get the number of sites
        self._lattice       = self._hilbert_space.get_lattice()
                    
        if self._dtype is None:
            self._dtype     = self._hilbert_space.dtype
        self._nh            = self._hilbert_space.get_Nh()
        
        #! other properties
        self._startns       = 0 # for starting hamil calculation (potential loop over sites)
        self._logger        = self._hilbert_space.logger if logger is None else logger
        
        # for the Hamiltonian matrix properties, and energy properties    
        self._av_en_idx     = 0
        self._av_en         = 0.0
        self._std_en        = 0.0
        self._min_en        = 0.0
        self._max_en        = 0.0
        
        # for the matrix representation of the Hamiltonian
        self._hamil         = None  # will store the Hamiltonian matrix with Nh x Nh full Hilbert space
        
        #! single particle Hamiltonian info
        self._hamil_sp          = None  # will store Ns x Ns (2Ns x 2Ns for BdG) matrix for quadratic Hamiltonian
        self._delta_sp          = None
        self._constant_offset   = 0.0
        self._isfermions        = True
        self._isbosons          = False
        
        #! general Hamiltonian info
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
    
    def _log(self, msg : str, log : str = 'info', lvl : int = 0, color : str = "white"):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (str) : The logging level. Default is 'info'.
            lvl (int) : The level of the message.
        """
        msg = f"[{self._name}] {msg}"
        self._hilbert_space.log(msg, log = log, lvl = lvl, color = color)
    
    def __str__(self):
        '''
        Returns the string representation of the Hamiltonian class.
        
        Returns:
            str :
                The string representation of the Hamiltonian class.
        '''
        return f"{'Quadratic' if self._is_quadratic else ''} Hamiltonian with {self._nh} elements and {self._ns} modes."
    
    def __repr__(self):
        """
        Returns a detailed string representation of the Hamiltonian instance.

        Includes:
            - Hamiltonian type (Many-Body or Quadratic)
            - Number of Hilbert space elements (Nh)
            - Number of modes/sites (Ns)
            - Hilbert space and lattice info
            - Computational backend and dtype
            - Sparsity and memory usage (if available)
        """
        htype           = "Many-Body" if self._is_manybody else "Quadratic"
        hilbert_info    = (
                    f"HilbertSpace(Nh={self._nh})"
                    if self._hilbert_space is not None
                    else ""
                    )
        lattice_info    = (
                    f"Lattice({self._lattice})"
                    if self._lattice is not None
                    else ""
                    )
        backend_info    = f"backend='{self._backendstr}', dtype={self._dtype}"
        sparse_info     = "sparse" if self._is_sparse else "dense"

        return (
            f"<{htype} Hamiltonian | Nh={self._nh}, Ns={self._ns}, "
            f"{hilbert_info}, {lattice_info}, {backend_info}, {sparse_info}>"
        )
    
    # ----------------------------------------------------------------------------------------------
    
    def randomize(self, **kwargs):
        ''' Randomize the Hamiltonian matrix.'''
        raise NotImplementedError("Randomization is not implemented for this Hamiltonian class.")
    
    def clear(self):
        '''
        Clears the Hamiltonian matrix.
        '''
        self._hamil      = None
        self._hamil_sp   = None
        self._delta_sp   = None
        self._constant_sp= None
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
        return not self._is_quadratic
    
    def is_quadratic(self):
        '''
        Returns a flag indicating whether the Hamiltonian is quadratic or not.
        '''
        return self._is_quadratic
    
    @property
    def manybody(self):
        '''
        Returns a flag indicating whether the Hamiltonian is many-body or not.
        '''
        return self._is_manybody
    
    def is_manybody(self):
        '''
        Returns a flag indicating whether the Hamiltonian is many-body or not.
        '''
        return self._is_manybody
    
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
    def particle_conserving(self):
        '''
        Returns a flag indicating whether the Hamiltonian is particle conserving or not.
        '''
        return self._particle_conserving if not self._is_manybody else None
    
    @property
    def is_particle_conserving(self):
        '''
        Returns a flag indicating whether the Hamiltonian is particle conserving or not.
        '''
        return self._particle_conserving if not self._is_manybody else None
    
    @property
    def is_bdg(self):
        '''
        Returns a flag indicating whether the Hamiltonian is Bogoliubov-de Gennes (BdG) or not.
        '''
        return self.particle_conserving

    @property
    def max_local_changes(self):
        '''
        Returns the maximum number of local changes.
        '''
        return self._max_local_ch if self._is_many_body else 2

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
    
    #! LATTICE BASED
    
    @property
    def ns(self):
        '''
        Returns the number of sites.
        '''
        return self._ns # directly stored for the convenience

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
    
    #! HILBERT BASED
    
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
        return self._hilbert_space.get_Nh()
    
    #! EIGENVALUES AND EIGENVECTORS
    
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
    def hamil_sp(self):
        '''
        Returns the single-particle Hamiltonian matrix.
        '''
        return self._hamil_sp
    
    @hamil_sp.setter
    def hamil_sp(self, hamil_sp):
        '''
        Sets the single-particle Hamiltonian matrix.
        
        Args:
            hamil_sp : The single-particle Hamiltonian matrix.
        '''
        self._hamil_sp = hamil_sp
        
    @property
    def delta_sp(self):
        '''
        Returns the delta matrix of the single-particle Hamiltonian.
        '''
        return self._delta_sp
    
    @delta_sp.setter
    def delta_sp(self, delta_sp):
        '''
        Sets the delta matrix of the single-particle Hamiltonian.
        
        Args:
            delta_sp : The delta matrix of the single-particle Hamiltonian.
        '''
        self._delta_sp = delta_sp
    
    @property
    def constant_sp(self):
        '''
        Returns the constant matrix of the single-particle Hamiltonian.
        '''
        return self._constant_sp
    
    @constant_sp.setter
    def constant_sp(self, constant_sp):
        '''
        Sets the constant matrix of the single-particle Hamiltonian.
        
        Args:
            constant_sp : The constant matrix of the single-particle Hamiltonian.
        '''
        self._constant_sp = constant_sp
    
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
    
    #! ENERGY PROPERTIES
    
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
        
        target_hamiltonian: np.ndarray = self.hamil
    
        if JAX_AVAILABLE and self._backend != np:
            if isinstance(target_hamiltonian, BCOO):
                return target_hamiltonian.diagonal()
            elif isinstance(target_hamiltonian, jnp.ndarray):
                return jnp.diag(target_hamiltonian)
            else:
                # dunnno what to do here
                return None
        elif sp.sparse.issparse(target_hamiltonian):
            return target_hamiltonian.diagonal()
        elif isinstance(target_hamiltonian, np.ndarray):
            return target_hamiltonian.diagonal()
        else:
            # dunnno what to do here
            return None
    
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
    
    #! ----------------------------------------------------------------------------------------------
    
    def get_loc_energy_arr_fun(self, backend: str = 'default'):
        '''
        Returns the local energy of the Hamiltonian
        Returns:
            A function that takes an integer k and returns the local energy for an array representation in
            a given backend - either NumPy or JAX.
        '''
        if (backend == 'default' or backend == 'jax' or backend == 'jnp') and JAX_AVAILABLE:
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
        
        matrix_to_check = self.hamil if (self._is_manybody) else self.hamil_sp
        if matrix_to_check is None:
            raise ValueError(Hamiltonian._ERR_HAMILTONIAN_NOT_AVAILABLE)
        
        # Dense matrix: use nbytes if available, otherwise compute from shape.
        # self._log(f"Checking the memory used by the Hamiltonian matrix of type {type(self._hamil)}", lvl=1)
        memory = 0
        
        if not self._is_sparse:
            if hasattr(matrix_to_check, "nbytes"):
                return matrix_to_check.nbytes
            else:
                return int(np.prod(matrix_to_check.shape)) * matrix_to_check.dtype.itemsize
        else:
            self._log("It is not a dense matrix...", lvl=2, log='debug')
            
            # Sparse matrix:
            # For NumPy (or when JAX is unavailable) we assume a scipy sparse matrix (e.g. CSR)
            if self._is_numpy:
                memory = 0
                for attr in ('data', 'indices', 'indptr'):
                    if hasattr(matrix_to_check, attr):
                        arr = getattr(matrix_to_check, attr)
                        if hasattr(arr, 'nbytes'):
                            memory += arr.nbytes
                        else:
                            memory += int(np.prod(arr.shape)) * arr.dtype.itemsize
            elif self._is_jax:
                # For JAX sparse matrices (e.g. BCOO), we assume they have data and indices attributes.
                data_arr        = matrix_to_check.data
                indices_arr     = matrix_to_check.indices
                if hasattr(data_arr, 'nbytes'):
                    data_bytes  = data_arr.nbytes
                else:
                    data_bytes  = int(np.prod(data_arr.shape)) * data_arr.dtype.itemsize
                if hasattr(indices_arr, 'nbytes'):
                    indices_bytes = indices_arr.nbytes
                else:
                    indices_bytes = int(np.prod(indices_arr.shape)) * indices_arr.dtype.itemsize
                memory = data_bytes + indices_bytes
            else:
                return 0 # Unknown type, return 0
        return memory if self._is_manybody else 2 * memory # for BdG Hamiltonian

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
        if (not JAX_AVAILABLE or self._backend == np or use_npy):
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
        if (not JAX_AVAILABLE or self._backend == np or use_npy):
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
    
    def _set_some_coupling(self, coupling: Union[list, np.ndarray, float, complex, int, str]) -> Array:
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
            return DummyVector(coupling)
        elif isinstance(coupling, str):
            return random_vector(self.ns, coupling, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_COUP_VEC_SIZE)
    
    # ----------------------------------------------------------------------------------------------
    
    def init(self, use_numpy : bool = False):
        '''
        Initializes the Hamiltonian matrix. Uses Batched-coordinate (BCOO) sparse matrices if JAX is
        used, otherwise uses NumPy arrays. The Hamiltonian matrix is initialized to be a matrix of
        zeros if the Hamiltonian is not sparse, otherwise it is initialized to be an empty sparse
        matrix.
        
        Parameters:
            use_numpy (bool):
                A flag indicating whether to use NumPy or JAX.
        '''
        self._log("Initializing the Hamiltonian matrix...", lvl = 2, log = "debug")
        
        jax_maybe_avail = self._is_jax
        if jax_maybe_avail and use_numpy:
            self._log("JAX is available but NumPy is forced...", lvl = 3)
        
        if self._is_quadratic:
            # Initialize Quadratic Matrix (_hamil_sp)
            # Shape determined by subclass, assume (Ns, Ns) for now
            ham_shape = getattr(self, '_hamil_sp_shape', (self._ns, self._ns))
            self._log(f"Initializing Quadratic Hamiltonian structure {ham_shape} (Sparse={self.sparse})...", lvl=3, log="debug")
            
            if self.sparse:
                if self._is_numpy:
                    self._hamil_sp = sp.sparse.csr_matrix(ham_shape, dtype=self._dtype)
                else:
                    indices         = self._backend.zeros((0, 2), dtype=ACTIVE_INT_TYPE)
                    data            = self._backend.zeros((0,), dtype=self._dtype)
                    self._hamil_sp  = BCOO((data, indices), shape=ham_shape)
                    self._delta_sp  = BCOO((data, indices), shape=ham_shape)
            else:
                self._hamil_sp      = self._backend.zeros(ham_shape, dtype=self._dtype)
                self._delta_sp      = self._backend.zeros(ham_shape, dtype=self._dtype)
            self._hamil = None
        else:
            if self.sparse:
                self._log("Initializing the Hamiltonian matrix as a sparse matrix...", lvl = 3, log = "debug")
                
                # --------------------------------------------------------------------------------------
                
                if not jax_maybe_avail or use_numpy:
                    self._log("Initializing the Hamiltonian matrix as a CSR sparse matrix...", lvl = 3, log = "debug")
                    self._hamil = sp.sparse.csr_matrix((self._nh, self._nh), dtype = self._dtype)
                else:
                    self._log("Initializing the Hamiltonian matrix as a sparse matrix...", lvl = 3, log = "debug")
                    # Create an empty sparse Hamiltonian matrix using JAX's BCOO format
                    indices     = self._backend.zeros((0, 2), dtype=ACTIVE_INT_TYPE)
                    data        = self._backend.zeros((0,), dtype=self._dtype)
                    self._hamil = BCOO((data, indices), shape=(self._nh, self._nh))
                    
                # --------------------------------------------------------------------------------------
                
            else:
                self._log("Initializing the Hamiltonian matrix as a dense matrix...", lvl = 3, log = "debug")
                if not JAX_AVAILABLE or self._backend == np:
                    self._hamil     = self._backend.zeros((self._nh, self._nh), dtype=self._dtype)
                else:
                    # do not initialize the Hamiltonian matrix
                    self._hamil     = None
        self._log(f"Hamiltonian matrix initialized and it's {'many-body' if self._is_manybody else 'quadratic'}",
                    lvl = 3, color = "green", log = "debug")
    
    # ----------------------------------------------------------------------------------------------
    #! Many body Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------

    def _hamiltonian_validate(self):
        ''' Check if the Hamiltonian matrix is valid. '''
        
        matrix_to_check = self._hamil if (self._is_manybody) else self._hamil_sp
        
        if matrix_to_check is None:
            self._log("Hamiltonian matrix is not initialized.", lvl=3, color="red", log = "debug")
        else:
            valid   = False
            # For dense matrices (NumPy/JAX ndarray) which have the 'size' attribute.
            if hasattr(matrix_to_check, "size"):
                if matrix_to_check.size > 0:
                    valid = True
            # For SciPy sparse matrices: check the number of nonzero elements.
            elif hasattr(matrix_to_check, "nnz"):
                if matrix_to_check.nnz > 0:
                    valid = True
            # For JAX sparse matrices (e.g., BCOO): verify if the data array has entries.
            elif hasattr(matrix_to_check, "data") and hasattr(matrix_to_check, "indices"):
                if matrix_to_check.data.shape[0] > 0:
                    valid = True
            
            if valid:
                self._log("Hamiltonian matrix calculated and valid.", lvl=3, color="green", log = "debug")
            else:
                self._log("Hamiltonian matrix calculated but empty or invalid.", lvl=3, color="red", log = "debug")
                matrix_to_check = None

    def _transform_to_backend(self):
        '''
        Transforms the Hamiltonian matrix to the backend.
        '''
        if self._is_manybody:
            self._hamil = linalg.transform_backend(self._hamil, self._is_sparse, self._backend)
        else:
            self._hamil_sp = linalg.transform_backend(self._hamil_sp, self._is_sparse, self._backend)
            if self._particle_conserving:
                self._delta_sp = linalg.transform_backend(self._delta_sp, self._is_sparse, self._backend)

        self._eig_val   = linalg.transform_backend(self._eig_val, False, self._backend)
        self._eig_vec   = linalg.transform_backend(self._eig_vec, False, self._backend)
        self._krylov    = linalg.transform_backend(self._krylov, False, self._backend)
        self._log(f"Hamiltonian matrix transformed to the backend {self._backendstr}", lvl=2, color="green")
        
    # ----------------------------------------------------------------------------------------------

    def build(self, verbose: bool = False, use_numpy: bool = False):
        '''
        Builds the Hamiltonian matrix. It checks the internal masks 
        wheter it's many-body or quadratic...
        
        Args:
            verbose (bool) :
                A flag to indicate whether to print the progress of the build.
            use_numpy (bool) :
                Force numpy usage.
            
        '''
        if verbose:
            self._log(f"Building Hamiltonian (Type: {'Many-Body' if self._is_many_body else 'Quadratic'})...", lvl=1, color = 'orange')
        
        if self._is_many_body:
            # Ensure operators/local energy functions are defined
            if self._loc_energy_int_fun is None and self._loc_energy_np_fun is None and self._loc_energy_jax_fun is None:
                self._log("Local energy functions not set, attempting to set them via _set_local_energy_operators...", lvl=2, log="debug")
                try:
                    self._set_local_energy_operators()  # Should be implemented by MB subclass
                    self._set_local_energy_functions()  
                except Exception as e:
                    raise RuntimeError(f"Failed to set up operators/local energy functions: {e}")
        
        ################################
        #! Initialize the Hamiltonian
        ################################
        init_start = time.perf_counter()
        try:
            self.init(use_numpy)
        except Exception as e:
            raise ValueError(f"{Hamiltonian._ERR_HAMILTONIAN_INITIALIZATION} : {str(e)}") from e
        
        if self._is_quadratic:
            if hasattr(self._hamil_sp, "block_until_ready"):
                self._hamil_sp = self._hamil_sp.block_until_ready()
                
            if hasattr(self._delta_sp, "block_until_ready"):
                self._delta_sp = self._delta_sp.block_until_ready()

        if hasattr(self._hamil, "block_until_ready"):
            self._hamil = self._hamil.block_until_ready()
        
        # initialize duration
        init_duration = time.perf_counter() - init_start
        if verbose:
            self._log(f"Initialization completed in {init_duration:.6f} seconds", lvl = 2)
        
        ################################
        #! Build the Hamiltonian matrix
        ################################§
        ham_start = time.perf_counter()
        try:
            if self._is_manybody:
                self._hamiltonian(use_numpy)
            else:
                self._hamiltonian_quadratic(use_numpy)
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
        
        if not self._is_manybody:
            raise ValueError(Hamiltonian._ERR_MODE_MISMATCH)
        
        if self._hilbert_space is None or self._nh == 0:
            raise ValueError(Hamiltonian._ERR_HILBERT_SPACE_NOT_PROVIDED)

        if self._loc_energy_int_fun is None and (use_numpy or self._is_numpy):
            raise RuntimeError("MB build requires local energy functions (_loc_energy_int_fun).")
        
        # -----------------------------------------------------------------------------------------
        matrix_type = "sparse" if self.sparse else "dense"
        self._log(f"Calculating the {matrix_type} Hamiltonian matrix...", lvl=1, color="blue", log = 'debug')
        # -----------------------------------------------------------------------------------------
        
        # Check if JAX is available and the backend is not NumPy
        jax_maybe_av = self._is_jax
        
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

        # Check if the Hamiltonian matrix is calculated and valid using various backend checks
        self._hamiltonian_validate()
    
    # ----------------------------------------------------------------------------------------------
    #! Single particle Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------
    
    def _hamiltonian_quadratic(self, use_numpy : bool = False):
        '''
        Generates the Hamiltonian matrix whenever the Hamiltonian is single-particle. 
        This method needs to be implemented by the subclasses.
        '''
        pass
    
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
            
        if self._is_quadratic:
            if self.particle_conserving:
                self._log("Diagonalizing the quadratic Hamiltonian matrix without BdG...", lvl = 2, log = 'debug')
                self._hamil = self._hamil_sp
            else:
                self._log("Diagonalizing the quadratic Hamiltonian matrix with BdG...", lvl = 2, log = 'debug')
                if self._isfermions:
                    self._hamil = backend.block([   [ self._hamil_sp, self._delta_sp ],
                                                    [-self._delta_sp.conj(), -self._hamil_sp.conj().T ]])
        else:  # bosons – use ΣH to make it Hermitian
            sigma = backend.block([ [backend.eye(self.ns), backend.zeros_like(self._hamil)  ],
                                    [backend.zeros_like(self._hamil), -backend.eye(self.ns) ]])
            self._hamil = sigma @ backend.block([[ self._hamil_sp,  self._delta_sp          ],
                                    [self._delta_sp.conj().T, self._hamil_sp.conj().T       ]])
        
        # try to diagonalize the Hamiltonian matrix
        try:
            if self._is_sparse or method.lower() in ["lanczos", "shift-invert"]:
                self._eig_val, self._eig_vec = linalg.eigsh(self._hamil, method, backend, **kwargs)
            else:
                self._eig_val, self._eig_vec = linalg.eigh(self._hamil, backend, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to diagonalize the Hamiltonian using method '{method}' : {e}") from e
        
        if JAX_AVAILABLE:
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
    
        if not self._is_many_body:
            raise TypeError("Method 'add' is intended for Many-Body Hamiltonians to define local energy terms.")
        
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
            - JAX version is set only if the flag JAX_AVAILABLE is True.
        """
        
        if not self._is_manybody:
            self._log("Skipping local energy function setup for Quadratic Hamiltonian.", log='debug')
            return
        
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
        if JAX_AVAILABLE:
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
        
        self._max_local_ch              = max(self._max_local_ch_o, max(len(op) for op in self._ops_mod_sites)    + \
                                                                    max(len(op) for op in self._ops_mod_nosites)  + \
                                                                    max(len(op) for op in self._ops_nmod_sites)   + \
                                                                    max(len(op) for op in self._ops_nmod_nosites))
        self._log(f"Max local changes set to {self._max_local_ch}", lvl=2, color="green", log='debug')
        self._log("Successfully set local energy functions...", lvl=2, log ='debug')

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
    if JAX_AVAILABLE:
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