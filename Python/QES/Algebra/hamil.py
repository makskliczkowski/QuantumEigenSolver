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
from typing import List, Tuple, Union, Callable
from abc import ABC, abstractmethod
from functools import partial
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

###################################################################################################
from Algebra.hilbert import HilbertSpace
###################################################################################################

###################################################################################################
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend, maybe_jit
###################################################################################################
if _JAX_AVAILABLE:
    import jax
    from jax import jit
    import jax.lax as lax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

###################################################################################################
# Pure (functional) Hamiltonian update functions.
###################################################################################################

if _JAX_AVAILABLE:
    import Algebra.hamil_jit_methods as hjm
        
    @partial(jit, static_argnums=(3, 4))
    def _hamiltonian_inplace(ham, ns: int, nh: int, hilbert_space: HilbertSpace, loc_energy, start = 0):
        """
        JAX version: Functional "in-place" update via jax.lax.fori_loop.
        The arguments `hilbert_space` and `loc_energy` are treated as static.
        Parameters:
        - ham (jnp.ndarray) : The Hamiltonian matrix - can be sparse or dense.
        - ns (int)          : The number of sites.
        - nh (int)          : The number of elements in the Hilbert space.
        - hilbert_space     : The Hilbert space.
        - loc_energy        : The local energy function.
        - start             : The starting index for the update (default is 0).
        """
        def body_k(k, ham):
            k_map_jax_array = hilbert_space[k]
            k_map           = int(k_map_jax_array)
            def body_i(i, ham):
                return loc_energy(ham, hilbert_space, k, k_map, i)
            return lax.fori_loop(start, ns, body_i, ham)
        return lax.fori_loop(0, nh, body_k, ham)

def _hamiltonian_inplace_np(ham, ns: int, nh: int, hilbert_space: HilbertSpace, loc_energy, start = 0):
    """
    NumPy version: Updates the Hamiltonian array in place.
    Parameters:
    - ham (np.ndarray) : The Hamiltonian matrix.
    - ns (int)         : The number of sites.
    - nh (int)         : The number of elements in the Hilbert space.
    - hilbert_space    : The Hilbert space.
    - loc_energy       : The local energy function.
    - start            : The starting index for the update (default is 0).
    """
    for k in range(nh):
        k_map = hilbert_space[k]
        for i in range(start, ns):
            loc_energy(ham, hilbert_space, k, k_map, i)
    return ham

####################################################################################################
# Hamiltonian class - abstract class
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
    
    # ----------------------------------------------------------------------------------------------
    
    @staticmethod
    def __backend(backend: str):
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
                _backend, _backend_sp = bck, None
                _rng, _rng_k = None, None
            return backend, _backend, _backend_sp, (_rng, _rng_k)
        _backendstr = 'np' if backend is None or backend == np else 'jax'
        return Hamiltonian.__backend(_backendstr)
    
    # ----------------------------------------------------------------------------------------------
    
    def __init__(self, hilbert_space: HilbertSpace, is_sparse: bool = True, dtype=None, backend='default', **kwargs):
        '''
        Initialize the Hamiltonian class.
        
        Args:
            hilbert_space (HilbertSpace)    : The Hilbert space of the system.
            is_sparse (bool)                : A flag to indicate whether the Hamiltonian is sparse or not.
            dtype (data-type)               : The data type of the Hamiltonian matrix.
            **kwargs                        : Additional arguments.
        '''
        
        self._backendstr, self._backend, self._backend_sp, (self._rng, self._rng_k) = Hamiltonian.__backend(backend)

        # get the backend, scipy, and random number generator for the backend
        self._dtype         = dtype if dtype is not None else self._backend.float64
        self._hilbert_space = hilbert_space
        if self._hilbert_space is None:
            raise ValueError(Hamiltonian._ERR_HILBERT_SPACE_NOT_PROVIDED)
        
        self._is_sparse     = is_sparse

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
        self._hamil         = self._backend.array([]) # can be either sparse or dense depending on the system
        self._eig_vec       = self._backend.array([])
        self._eig_val       = self._backend.array([])
        self._krylov        = self._backend.array([]) # Krylov subspace vectors for the Lanczos algorithm
        self._name          = "Hamiltonian"
    
    def _log(self, msg : str, log : int = 1, lvl : int = 0, color : str = "white"):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (int) : The flag to log the message.
            lvl (int) : The level of the message.
        """
        if log > 0:
            msg = f"[Hamiltonian:{self._name}] {msg}"
            msg = self._logger.colorize(msg, color)
            self._logger.say(msg, log = log, lvl = lvl)
    
    # ----------------------------------------------------------------------------------------------
    
    def randomize(self, **kwargs):
        '''
        For random Hamiltonians, this method is used to generate a random Hamiltonian matrix from 
        scratch to capture new statistics. For non-random Hamiltonians, this method does nothing.
        '''
        pass 
    
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
        self._logger.say("Hamiltonian cleared.", log=1)

    
    # ----------------------------------------------------------------------------------------------
    #! Getter methods
    # ----------------------------------------------------------------------------------------------
    
    @property
    def dtype(self):
        '''
        Returns the data type of the Hamiltonian matrix.
        '''
        return self._dtype
    
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

    # ----------------------------------------------------------------------------------------------
    
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
        
    @property
    def ns(self):
        '''
        Returns the number of sites.
        '''
        return self._ns
    
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
    def lattice(self):
        '''
        Returns the lattice associated with the Hamiltonian.
        '''
        return self._lattice
    
    @property
    def hilbert_size(self):
        '''
        Returns the number of sites in the Hilbert space.
        '''
        return self._nh
    
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
        '''
        return self._hamil.diagonal()
    
    @property
    def h_memory(self):
        '''
        Returns the memory used by the Hamiltonian matrix.
        '''
        return self._hamil.nbytes
    
    # ----------------------------------------------------------------------------------------------
    #! Standard getters
    # ----------------------------------------------------------------------------------------------
    
    def get_mean_lvl_spacing(self):
        '''
        Returns the mean level spacing of the Hamiltonian. The mean level spacing is defined as the
        average difference between consecutive eigenvalues.
        
        Returns:
            float : The mean level spacing of the Hamiltonian.
        '''
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        if not _JAX_AVAILABLE or self._backend == np:
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
    
    def get_energywidth(self):
        '''
        Returns the energy width of the Hamiltonian. The energy width is defined as trace of the
        Hamiltonian matrix squared.
        '''
        if self._hamil.size == 0:
            raise ValueError(Hamiltonian._ERR_HAMILTONIAN_NOT_AVAILABLE)
        if not _JAX_AVAILABLE or self._backend == np:
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
    
    def init(self):
        '''
        Initializes the Hamiltonian matrix. Uses Batched-coordinate (BCOO) sparse matrices if JAX is
        used, otherwise uses NumPy arrays. The Hamiltonian matrix is initialized to be a matrix of
        zeros if the Hamiltonian is not sparse, otherwise it is initialized to be an empty sparse
        '''
        self._log("Initializing the Hamiltonian matrix...", lvl = 2)
        if self.sparse:
            if not _JAX_AVAILABLE or self._backend == np:
                self._log("Initializing the Hamiltonian matrix as a CSR sparse matrix...", lvl = 3)
                self._hamil = sp.sparse.csr_matrix((self._nh, self._nh), dtype = self._dtype)
            else:
                self._log("Initializing the Hamiltonian matrix as a sparse matrix...", lvl = 3)
                # Create an empty sparse Hamiltonian matrix using JAX's BCOO format
                indices     = self._backend.zeros((0, 2), dtype=int)
                data        = self._backend.zeros((0,), dtype=self._dtype)
                self._hamil = BCOO((data, indices), shape=(self._nh, self._nh))
        else:
            self._hamil     = self._backend.zeros((self._nh, self._nh), dtype = self._dtype)
        self._log("Hamiltonian matrix initialized.", lvl = 3, color = "green")
        
    # ----------------------------------------------------------------------------------------------
    #! Single particle Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------
    
    def _hamiltonian_sp(self):
        '''
        Generates the Hamiltonian matrix whenever the Hamiltonian is single-particle. 
        '''
        #!TODO implement this through the Hilbert space!
        pass
    
    # ----------------------------------------------------------------------------------------------
    #! Many body Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------

    def _hamiltonian(self):
        '''
        Generates the Hamiltonian matrix. The diagonal elements are straightforward to calculate,
        while the off-diagonal elements are more complex and depend on the specific Hamiltonian.
        It iterates over the Hilbert space to calculate the Hamiltonian matrix. 
        
        Note: This method may be overridden by subclasses to provide a more efficient implementation
        '''
        if self._hilbert_space is None or self._nh == 0:
            raise ValueError(Hamiltonian._ERR_HILBERT_SPACE_NOT_PROVIDED)

        self._log("Calculating the Hamiltonian matrix...", lvl = 1, color = 'blue')
        # Go through the Hilbert space and calculate the Hamiltonian matrix
        if not _JAX_AVAILABLE or self._backend == np:
            self._log("Calculating the Hamiltonian matrix using NumPy...", lvl = 2)
            self._hamil = _hamiltonian_inplace_np(self._hamil, self._ns, self._nh, 
                                            self._hilbert_space, self.loc_energy_ham, start = self._startns)
        else:
            self._log("Calculating the Hamiltonian matrix using JAX...", lvl = 2)
            self._hamil = _hamiltonian_inplace(self._hamil, self._ns, self._nh, 
                                            self._hilbert_space, self.loc_energy_ham, start=self._startns)
            self._hamil = self._hamil.block_until_ready()
        if self._hamil is not None and self._hamil.size > 0:
            self._log("Hamiltonian matrix calculated.", lvl = 3, color = "green")
        else:
            self._log("Hamiltonian matrix not calculated.", lvl = 3, color = "red")

    def build(self, verbose: bool = False):
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
            self.init()
        except Exception as e:
            raise ValueError(f"{Hamiltonian._ERR_HAMILTONIAN_INITIALIZATION} : {e}")
    
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
            self._hamiltonian()
        except Exception as e:
            self._log(f"{Hamiltonian._ERR_HAMILTONIAN_BUILD} : {e}", lvl = 2, color = "red")
        
        if hasattr(self._hamil, "block_until_ready"):
            self._hamil = self._hamil.block_until_ready()
        ham_duration = time.perf_counter() - ham_start
        if self._hamil is not None and self._hamil.size > 0:
            if verbose:
                self._log(f"Hamiltonian matrix built in {ham_duration:.6f} seconds.", lvl = 1)
        else:
            raise ValueError(Hamiltonian._ERR_HAMILTONIAN_BUILD)

    # ----------------------------------------------------------------------------------------------
    #! Local energy methods - Abstract methods
    # ----------------------------------------------------------------------------------------------
    
    @abstractmethod
    def loc_energy_ham(self, ham, hilbert : HilbertSpace, k : int, k_map : int, i : int):
        '''
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by 
        subclasses to provide a specific implementation.
        
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by
        subclasses to provide a specific implementation.
                
        Parameters:
            ham (jnp.ndarray or BCOO)     : The Hamiltonian matrix.
            k (int)                       : The k'th element of the Hilbert space.
            k_map (List)                  : The mapping of the k'th element.
            i (int)                       : The i'th site.
        '''
        pass
    
    @abstractmethod
    def loc_energy(self, k : Union[int, np.ndarray], i : int):
        '''
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by 
        subclasses to provide a specific implementation.
        
        This is meant to check how does the Hamiltonian act on a state at a given site.
        
        Parameters:
            k (Union[int, Backend.ndarray]) : The k'th element of the Hilbert space.
            i (int) : The i'th site.
        '''
        pass

    @property
    def quadratic(self):
        '''
        Returns a flag indicating whether the Hamiltonian is quadratic or not.
        '''
        return False

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
    
    @staticmethod
    def _diagonalize_backend(hamil, logging : Callable, backend : str = 'default'):
        '''
        Diagonalizes the Hamiltonian matrix using the backend's linear algebra library.
        '''
        try:
            backend = get_backend(backend)
            eig_val, eig_vec = backend.linalg.eigh(hamil)
            return eig_val, eig_vec
        except Exception as e:
            logging(f"An error occurred during diagonalization: {e}")
        return None, None
    
    @staticmethod
    def _diagonalize_lanczos_backend(hamil,
                    k       : int = 6,
                    which   : str = "SA",
                    logging : Callable = None,
                    backend : str = 'default'):
        '''
        Diagonalizes the Hamiltonian matrix using the Lanczos algorithm.
        Parameters:
        - k (int) : Number of eigenpairs to compute.
        - which (str) : Eigenvalue selection criteria ('SA' for smallest algebraic, 'LA' for largest algebraic).
        - sigma (float) : Shift value for shift-invert diagonalization.
        '''
        try:
            backend, backend_sp = get_backend(backend, scipy=True)
            _eig_val, _eig_vec = backend_sp.sparse.linalg.eigsh(hamil, k=k, which=which)
            return _eig_val, _eig_vec
        except Exception as e:
            if logging is not None:
                logging(f"An error occurred during Lanczos diagonalization: {e}")
        return None, None
    
    @staticmethod
    def _diagonalize_shift_invert_backend(hamil,
                    k       : int = 6,
                    sigma   : float = 0.0,
                    which   : str = "LM",
                    mode    : str = "normal",
                    logging : Callable = None,
                    backend : str = 'default'):
        '''
        Diagonalizes the Hamiltonian matrix using the shift-invert method.
        Parameters:
        - k (int) : Number of eigenpairs to compute.
        - sigma (float) : Shift value for shift-invert diagonalization.
        - which (str) : Eigenvalue selection criteria ('SA' for smallest algebraic, 'LA' for largest algebraic).
        - mode (str) : The mode of the shift-invert diagonalization ('normal' or 'cayley' or 'buckling').
        - backend (str) : The backend to use. Fallbacks to the default backend.
        '''
        try:
            backend, backend_sp = get_backend(backend, scipy=True)
            eig_val, eig_vec = backend_sp.sparse.linalg.eigsh(hamil,
                                            k = k, sigma = sigma, which = which, mode = mode)
            return eig_val, eig_vec
        except Exception as e:
            if logging is not None:
                logging(f"An error occurred during shift-invert diagonalization: {e}")
        return None, None
        
    def diagonalize(self, verbose: bool = False, **kwargs):
        """
        Diagonalizes the Hamiltonian matrix using one of several methods.
        
        Supported methods:
        - 'eigh'        : Full diagonalization using a dense eigen–solver.
        - 'lanczos'     : Iterative Lanczos diagonalization via SciPy's eigsh.
        - 'shift-invert': Iterative shift–invert diagonalization via SciPy's eigsh.
        
        Additional kwargs:
        - k (int)       : Number of eigenpairs to compute (default: 6 for Lanczos/shift-invert).
        - sigma (float) : Shift value for shift-invert (default: 0.0).
        - which (str)   : Eigenvalue selection criteria ('SA' for Lanczos, 'LM' for shift-invert).
        
        Updates:
        - self._eig_val: Eigenvalues.
        - self._eig_vec: Eigenvectors.
        """
        diag_start  = time.perf_counter()
        method      = kwargs.get("method", "eigh")
        
        if method == "eigh":
            self._eig_val, self._eig_vec = Hamiltonian._diagonalize_backend(self._hamil, self._log,
                                                                self._backendstr)
        elif method == "lanczos":

            k       = kwargs.get("k", 6)
            sigma   = kwargs.get("sigma", 0.0)
            which   = kwargs.get("which", "SA")
            self._eig_val, self._eig_vec = self._diagonalize_lanczos_backend(self._hamil, k=k, which=which,
                                                logging=self._log, backend=self._backendstr)
        elif method == "shift-invert":
            k       = kwargs.get("k", 6)
            sigma   = kwargs.get("sigma", 0.0)
            which   = kwargs.get("which", "LM")
            mode    = kwargs.get("mode", "normal")
            self._eig_val, self._eig_vec = self._diagonalize_shift_invert_backend(self._hamil, k=k, 
                                sigma=sigma, which=which, mode=mode, logging=self._log, backend=self._backendstr)
        else:
            raise ValueError(f"Unknown diagonalization method: {method}")
        
        if _JAX_AVAILABLE:
            if hasattr(self._eig_val, "block_until_ready"):
                self._eig_val = self._eig_val.block_until_ready()
            if hasattr(self._eig_vec, "block_until_ready"):
                self._eig_vec = self._eig_vec.block_until_ready()
        
        diag_duration = time.perf_counter() - diag_start
        if verbose:
            self._log(f"Diagonalization ({method}) completed in {diag_duration:.6f} seconds.")
        self._calculate_av_en()

    # ----------------------------------------------------------------------------------------------
    #! Setters
    # ----------------------------------------------------------------------------------------------
    
    @staticmethod
    def set_hamil_elem(ham, hilbert : HilbertSpace, k : int, val, newk : int):
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
        
        # check the mapping
        kmap = hilbert[k] if hilbert is not None else k
                
        # set the element
        try:
            if kmap == newk:
                # the element k is already the same as new_k and obviously we 
                # and we add this at k (not kmap as it only checks the representative)
                if not _JAX_AVAILABLE or isinstance(ham, (np.ndarray, sp.sparse.spmatrix)):
                    ham[newk, k] += val
                else:
                    ham = ham.at(newk, k).add(val)
            else:
                # otherwise we need to check the representative of the new k
                norm        = hilbert.norm(k) # get the norm of the k'th element of the Hilbert space
                idx, symeig = hilbert.find_representative_int(newk, norm) # find the representative of the new k
                if not _JAX_AVAILABLE or isinstance(ham, (np.ndarray, sp.sparse.spmatrix)):
                    ham[idx, k] += val * symeig
                else:
                    ham = ham.at[idx, k].add(val * symeig)
        except Exception as e:
            print(f"Error in set_hamil_elem: Failed to set element at <newk(idx)|H|k>, newk={newk},idx={idx},k={k},value: {val}. Please verify that the indices and value are correct. Exception details: {e}")

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
        
        self.set_hamil_elem(self._hamil, self._hilbert_space, k, val, newk)
        
    # ----------------------------------------------------------------------------------------------
        
# --------------------------------------------------------------------------------------------------