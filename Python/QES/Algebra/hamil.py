"""
High-level Hamiltonian class for the Quantum Energy Solver (QES) package. This class is used to
define the Hamiltonian of a system. It may be either a Many-Body Quantum Mechanics Hamiltonian or a
non-interacting system Hamiltonian. It may generate a Hamiltonian matrix but in addition it defines
how an operator acts on a state. The Hamiltonian class is an abstract class and is not meant to be
instantiated. It is meant to be inherited by other classes.
@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""
from typing import List, Tuple, Union, Callable
from abc import ABC, abstractmethod
import scipy.sparse as sp
import time
from functools import partial

# Import the necessary modules from the package
from .hilbert import HilbertSpace

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, random
    from jax.experimental.sparse import BCOO
    __JAX_AVAILABLE = True
    Backend         = jnp
    
    # Import the JAX-optimized methods
    import hamil_jit_methods as hjm
    
except ImportError:
    print("JAX is not available. Using NumPy instead.")
    __JAX_AVAILABLE = False

# import to set default backend
import numpy as np

def maybe_jit(f):
    '''
    Decorator to conditionally apply jax.jit to a function.
    '''
    return jit(f) if __JAX_AVAILABLE else f

###################################################################################################
# Pure (functional) Hamiltonian update functions.
###################################################################################################

if __JAX_AVAILABLE:
    @partial(jit, static_argnums=(3, 4))
    def hamiltonian_inplace(ham, ns: int, nh: int, hilbert_space: HilbertSpace, loc_energy):
        """
        JAX version: Functional "in-place" update via jax.lax.fori_loop.
        The arguments `hilbert_space` and `loc_energy` are treated as static.
        Parameters:
        - ham (jnp.ndarray) : The Hamiltonian matrix - can be sparse or dense.
        - ns (int)         : The number of sites.
        - nh (int)         : The number of elements in the Hilbert space.
        - hilbert_space    : The Hilbert space.
        - loc_energy       : The local energy function.
        """
        def body_k(k, ham):
            k_map = hilbert_space.get_mapping(k)
            def body_i(i, ham):
                return loc_energy(ham, k, k_map, i)
            return jax.lax.fori_loop(0, ns, body_i, ham)
        return jax.lax.fori_loop(0, nh, body_k, ham)
else:
    def hamiltonian_inplace(ham, ns: int, nh: int, hilbert_space, loc_energy):
        """
        NumPy version: Updates the Hamiltonian array in place.
        Parameters:
        - ham (np.ndarray) : The Hamiltonian matrix.
        - ns (int)         : The number of sites.
        - nh (int)         : The number of elements in the Hilbert space.
        - hilbert_space    : The Hilbert space.
        - loc_energy       : The local energy function.
        """
        for k in range(nh):
            k_map = hilbert_space.get_mapping(k)
            for i in range(ns):
                loc_energy(ham, k, k_map, i)
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
    
    def __init__(self, hilbert_space : HilbertSpace, is_sparse : bool = True, dtype = None, backend = None, **kwargs):
        '''
        Initialize the Hamiltonian class.
        
        Args:
            hilbert_space (HilbertSpace)    : The Hilbert space of the system.
            is_sparse (bool)                : A flag to indicate whether the Hamiltonian is sparse or not.
            dtype (data-type)               : The data type of the Hamiltonian matrix.
            **kwargs                        : Additional arguments.
        '''
        
        self._backend       = backend if backend is not None else Backend
        self._dtype         = dtype if dtype is not None else Backend.float64
        self._hilbert_space = hilbert_space
        self._is_sparse     = is_sparse
        
        if self._hilbert_space is None:
            raise ValueError("The Hilbert space must be provided.")
        else:
            self._lattice   = self._hilbert_space.get_lattice()
            self._nh        = self._hilbert_space.get_Nh()
            self._ns        = self._hilbert_space.get_Ns()
        
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
    
    # ----------------------------------------------------------------------------------------------
    
    def randomize(self):
        '''
        For random Hamiltonians, this method is used to generate a random Hamiltonian matrix from 
        scratch to capture new statistics. For non-random Hamiltonians, this method does nothing.
        '''        
        pass 
    
    # ----------------------------------------------------------------------------------------------
    
    # Getter methods
    
    # ----------------------------------------------------------------------------------------------
    
    @property
    def dtype(self):
        '''
        Returns the data type of the Hamiltonian matrix.
        '''
        return self._dtype
    
    @property
    @abstractmethod
    def quadratic(self):
        '''
        Returns a flag indicating whether the Hamiltonian is quadratic or not.
        '''
        return False
    
    @property
    def manybody(self):
        '''
        Returns a flag indicating whether the Hamiltonian is many-body or not.
        '''
        return not self.quadratic
    
    @property
    def sparse(self):
        '''
        Returns a flag indicating whether the Hamiltonian is sparse or not.
        '''
        return self._is_sparse

    # ----------------------------------------------------------------------------------------------
    
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
    def is_sparse(self):
        '''
        Returns a flag indicating whether the Hamiltonian is sparse or not.
        '''
        return self._is_sparse
    
    @property
    def hamil(self):
        '''
        Returns the Hamiltonian matrix.
        '''
        return self._hamil
    
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
    
    # Standard getters
    
    # ----------------------------------------------------------------------------------------------
    
    def get_mean_lvl_spacing(self):
        '''
        Returns the mean level spacing of the Hamiltonian. The mean level spacing is defined as the
        average difference between consecutive eigenvalues.
        
        Returns:
            float : The mean level spacing of the Hamiltonian.
        '''
        if self._eig_val.size == 0:
            raise ValueError("The eigenvalues are not available.")
        if __JAX_AVAILABLE:
            return hjm.mean_level_spacing(self._eig_val)
        return self._backend.mean(self._backend.diff(self._eig_val))
    
    def get_bandwidth(self):
        '''
        Returns the bandwidth of the Hamiltonian. The bandwidth is defined as the difference between
        the highest and the lowest eigenvalues - values are sorted in ascending order.
        '''
        if self._eig_val.size == 0:
            raise ValueError("The eigenvalues are not available.")
        return self._eig_val[-1] - self._eig_val[0]
    
    def get_energywidth(self):
        '''
        Returns the energy width of the Hamiltonian. The energy width is defined as trace of the
        Hamiltonian matrix squared.
        '''
        if self._hamil.size == 0:
            raise ValueError("The Hamiltonian matrix is empty.")
        if __JAX_AVAILABLE:
            return hjm.energy_width(self._hamil)
        return self._backend.trace(self._backend.dot(self._hamil, self._hamil))
    
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
    
    # Initialization methods
    
    # ----------------------------------------------------------------------------------------------
    
    def init(self):
        '''
        Initializes the Hamiltonian matrix. Uses Batched-coordinate (BCOO) sparse matrices if JAX is
        used, otherwise uses NumPy arrays. The Hamiltonian matrix is initialized to be a matrix of
        zeros.
        '''
        if self.sparse:
            if __JAX_AVAILABLE:
                # Create an empty BCOO sparse matrix
                data        = self._backend.array([],       dtype = self._dtype)
                indices     = self._backend.empty((0, 2),   dtype = self._backend.int32)
                self._hamil = BCOO((data, indices), shape = (self._nh, self._nh))
            else: # Otherwise, assume a NumPy backend (the sparse matrix is a CSR matrix)
                self._hamil = sp.csr_matrix((self._nh, self._nh), dtype = self._dtype)
        else:
            self._hamil     = self._backend.zeros((self._nh, self._nh), dtype = self._dtype)
        
    # ----------------------------------------------------------------------------------------------
    
    # Single particle Hamiltonian matrix
    
    # ----------------------------------------------------------------------------------------------
    
    def _hamiltonian_sp(self):
        '''
        Generates the Hamiltonian matrix whenever the Hamiltonian is single-particle. 
        '''
        #!TODO implement this through the Hilbert space! 
        pass
    
    # ----------------------------------------------------------------------------------------------
    
    # Many body Hamiltonian matrix
    
    # ----------------------------------------------------------------------------------------------

    def _hamiltonian(self):
        '''
        Generates the Hamiltonian matrix. The diagonal elements are straightforward to calculate,
        while the off-diagonal elements are more complex and depend on the specific Hamiltonian.
        It iterates over the Hilbert space to calculate the Hamiltonian matrix. 
        
        Note: This method may be overridden by subclasses to provide a more efficient implementation
        '''
        if self._hilbert_space is None or self._nh == 0:
            raise ValueError("The Hilbert space must be provided. The Hamiltonian matrix must be initialized.")

        # Go through the Hilbert space and calculate the Hamiltonian matrix
        if not __JAX_AVAILABLE:
            hamiltonian_inplace(self._hamil, self._ns, self._nh, self._hilbert_space, self.loc_energy_ham)
        else:
            self._hamil = hamiltonian_inplace(self._hamil, self._ns, self._nh, self._hilbert_space, self.loc_energy_ham)
            self._hamil = self._hamil.block_until_ready()
        
    def build(self, verbose : bool = False):
        '''
        Builds the Hamiltonian matrix.
        
        Args:
            verbose (bool) : A flag to indicate whether to print the progress of the build.
        '''
        if verbose:
            print("Building the Hamiltonian matrix...")
            
        ################################
        # Initialize the Hamiltonian
        ################################
        init_start = time.perf_counter()
        try:
            self.init()
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
    
        if __JAX_AVAILABLE and hasattr(self._hamil, "block_until_ready"):
            self._hamil = self._hamil.block_until_ready()
        init_duration = time.perf_counter() - init_start
        if verbose:
            print(f"Initialization completed in {init_duration:.6f} seconds")
        
        ################################
        # Build the Hamiltonian matrix
        ################################
        ham_start = time.perf_counter()
        try:
            self._hamiltonian()
        except Exception as e:
            print(f"An error occurred during the Hamiltonian build: {e}")
        
        if __JAX_AVAILABLE and hasattr(self._hamil, "block_until_ready"):
            self._hamil = self._hamil.block_until_ready()
        ham_duration = time.perf_counter() - ham_start
        if verbose:
            print(f"Hamiltonian matrix built in {ham_duration:.6f} seconds.")

    # ----------------------------------------------------------------------------------------------
    
    # Local energy methods - Abstract methods
    
    # ----------------------------------------------------------------------------------------------
    
    @abstractmethod
    def loc_energy_ham(self, ham, k : int, k_map : int, i : int):
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
    def local_energy(self, k : Union[int, np.ndarray, jnp.ndarray], i : int):
        '''
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by 
        subclasses to provide a specific implementation.
        
        This is meant to check how does the Hamiltonian act on a state at a given site.
        
        Parameters:
            k (Union[int, Backend.ndarray]) : The k'th element of the Hilbert space.
            i (int) : The i'th site.
        '''
        pass

    # ----------------------------------------------------------------------------------------------
    
    # Calculators 
    
    # ----------------------------------------------------------------------------------------------
    
    def _calculate_av_en(self):
        '''
        Calculates the properties of the Hamiltonian matrix that are related to the energy.
        '''
        
        if self._eig_val.size == 0:
            raise ValueError("The eigenvalues are not available.")
        
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
            raise ValueError("The eigenvalues are not available.")
        return self._backend.argmin(self._backend.abs(self._eig_val - en))
    
    # ----------------------------------------------------------------------------------------------
    
    # Diagonalization methods
    
    # ----------------------------------------------------------------------------------------------
    
    def _diagonalize_backend(self):
        '''
        Diagonalizes the Hamiltonian matrix using the backend's linear algebra library.
        '''
        try:
            self._eig_val, self._eig_vec = self._backend.linalg.eigh(self._hamil)
        except Exception as e:
            print(f"An error occurred during diagonalization: {e}")
    
    
    def diagonalize(self, verbose : bool = False, **kwargs):
        """
        Diagonalizes the Hamiltonian matrix using one of several methods.
        
        Supported methods (via the keyword argument 'method'):
        - 'eigh': Full diagonalization using a dense eigen–solver.
        - 'lanczos': Iterative Lanczos algorithm.
        - 'shift-invert': Iterative shift–invert diagonalization.
        
        Additional keyword arguments (e.g. 'k' for number of eigenpairs, 'sigma' for the shift)
        are passed to the underlying routine.
        
        Args:
            verbose (bool): If True, prints progress and timing.
            **kwargs: Additional keyword arguments. In particular:
                - method: One of 'eigh', 'lanczos', or 'shift-invert'.
                - k: (For Lanczos/shift-invert) number of eigenpairs to compute.
                - sigma: (For shift-invert) the shift value.
        
        Updates:
            self._eig_val: eigenvalues (1D array).
            self._eig_vec: eigenvectors (2D array, columns correspond to eigenstates).
        """
        if verbose:
            print("Diagonalizing the Hamiltonian matrix...")
        
        diag_start = time.perf_counter()
        
        # Choose the method; default to 'eigh' if none is provided.
        method = kwargs.get("method", "eigh")
        
        try:
            self._eig_val, self._eig_vec = self._backend.linalg.eigh(self._hamil)
        except Exception as e:
            print(f"An error occurred during diagonalization: {e}")
        
        if __JAX_AVAILABLE and hasattr(self._eig_val, "block_until_ready"):
            self._eig_val = self._eig_val.block_until_ready()
            self._eig_vec = self._eig_vec.block_until_ready()
        
        diag_duration = time.perf_counter() - diag_start
        if verbose:
            print(f"Diagonalization completed in {diag_duration:.6f} seconds.")
        
        # Calculate the properties of the Hamiltonian matrix related to the energy
        self._calculate_av_en()
    
    # ----------------------------------------------------------------------------------------------
    
    # Setters
    
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
        
        # check the mapping
        kmap = self._hilbert_space.get_mapping(k) if self._hilbert_space is not None else k
        
        # set the element
        try:
            if kmap == newk:
                # the element k is already the same as new_k and obviously we 
                # and we add this at k (not kmap as it only checks the representative)
                if __JAX_AVAILABLE:
                    self._hamil = self._hamil.at[k, k].add(val)
                else:
                    self._hamil[k, k] += val
            else:
                # otherwise we need to check the representative of the new k
                norm        = self._hilbert_space.norm(k) # get the norm of the k'th element of the Hilbert space
                idx, symeig = self._hilbert_space.find_representative(newk, norm) # find the representative of the new k
                if __JAX_AVAILABLE:
                    self._hamil = self._hamil.at[idx, k].add(val * symeig)
                else:
                    self._hamil[idx, k] += val * symeig
        except Exception as e:
            print(f"Error in _set_hamil_elem: Failed to set element at row {k} (mapping: {kmap}) and column {newk}; value: {val}. Please verify that the indices and value are correct. Exception details: {e}")

    # ----------------------------------------------------------------------------------------------
        