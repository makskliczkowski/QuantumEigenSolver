"""
file    : Model/Interacting/Spin/qsm.py
High-level Hamiltonian class for the Quantum Eigen Solver (QES) package.
This file contains the QSM model - a Hamiltonian for a quantum dot coupled to an external system.
The external system is modeled as a quantum spin chain.
Is an instance of the QSM model.
Derives from a general Hamiltonian.
For more information see the QSM model.
https://arxiv.org/pdf/2308.07431.pdf

Based on the C++ implementation. 
Author  : Maksymilian Kliczkowski
Date    : 2025-02-17
Version : 0.1
"""

import numpy as np
import numba
from typing import List, Tuple, Union, Optional
from functools import partial

# Assume these are available from the QES package:
import Algebra.hilbert as hilbert_module
import Algebra.hamil as hamil_module
import Algebra.Operator.operators_spin as operators_spin_module

##########################################################################################
import general_python.algebra.linalg as linalg
from general_python.algebra.ran_wrapper import RMT, random_matrix, random_vector
from general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, _JAX_AVAILABLE
##########################################################################################

# ----------------------------------------------------------------------------------------
#! DEFINE CONSTANTS
# ----------------------------------------------------------------------------------------

_QSM_CHECK_HS_NORM = True

# ----------------------------------------------------------------------------------------
#! INTEGER STATES
# ----------------------------------------------------------------------------------------

@numba.njit
def _local_energy_int(k_map     : int,
                        i       : int,
                        n       : int,
                        ns      : int,
                        neidot  : np.ndarray, h: np.ndarray, g0: float, au: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute the local energy interaction.
    Parameters:
        k : int
            The state index.
        k_map : int
            The mapped state (from the Hilbert space) corresponding to k.
        i : int
            An index parameter (typically i >= self.n).
        n : int
            Number of particles in the dot.
        ns : int
            Number of particles in the system.
        neidot : np.ndarray
            Array of random neighbors for the 'free' particles.
        h : np.ndarray
            Magnetic field vector for the particles outside the dot.
        g0 : float
            Coupling strength between the particles in the dot and the particles outside the dot.
        au : np.ndarray
            Array of coupling strengths.
    Returns:
        Tuple of arrays representing the row indices, column indices, and matrix values for this interaction.
    '''

    # Pre-allocate arrays
    new_rows    = np.empty(2, dtype=np.int64)
    new_vals    = np.empty(2, dtype=DEFAULT_NP_FLOAT_TYPE)
    
    # store here the rows, columns, and values
    part_idx    = i - n
    this_site   = np.array([i], dtype=DEFAULT_NP_INT_TYPE)
    
    # apply the sigma_z operator
    idx, val    = operators_spin_module.sigma_z_int_np(k_map, ns, this_site)
    vals        = h[part_idx] * val.astype(h.dtype)
    
    # apply the sigma_x * sigma_x operator
    next_site   = np.array([neidot[part_idx]], dtype=DEFAULT_NP_INT_TYPE)
    idx1, sxn   = operators_spin_module.sigma_x_int_np(k_map, ns, next_site)
    idx2, sxj   = operators_spin_module.sigma_x_int_np(idx1[0], ns, this_site)
    coupling_v  = g0 * au[part_idx] * sxj * sxn
    
    new_rows[:1] = idx
    new_vals[:1] = vals
    new_rows[1:] = idx2[0]
    new_vals[1:] = coupling_v
    return new_rows, new_vals

def _local_energy_int_wrap(n, ns, neidot, h, g0, au):
    '''Creates a JIT-compiled local energy interaction function.'''
    @numba.njit
    def wrapper(k, i):
        return _local_energy_int(k, i, n, ns, neidot, h, g0, au)
    return wrapper

# ----------------------------------------------------------------------------------------
#! ARRAY STATES
# ----------------------------------------------------------------------------------------

@numba.njit
def _local_energy_arr_loop(state, neidot, h, g0, n, au):
    """
    Compute the local energy interaction for a numpy array state.
    
    Parameters:
        state   : numpy array of spins (0 or 1)
        neidot  : numpy array of neighbor indices for free particles
        h, g0, n, au : model parameters
        sigma_x, sigma_z : functions to compute the spin operators
    
    Returns:
        Tuple [states], [energies] containing the new states and energies.
    """
    size        = state.shape[0]
    local_value = 0.0
    new_states  = np.empty((size+1, size), dtype=state.dtype)
    vals        = np.empty(size+1, dtype=h.dtype)
    
    for i in numba.prange(size):
        # Apply the local energy interaction.
        part_idx    = i - n
        this_site   = np.array([i], dtype=DEFAULT_NP_INT_TYPE)
        # Apply the sigma_z operator
        _, sz_i     = operators_spin_module.sigma_z_np(state, this_site)
        local_in    = h[part_idx] * sz_i
        
        # Apply the sigma_x * sigma_x operator
        next_site   = np.array([neidot[part_idx]], dtype=DEFAULT_NP_INT_TYPE)
        # s1, sx_n    = sigma_x(state, next_site)
        # s2, sx_i    = sigma_x(s1, this_site)
        s1, sx_n    = operators_spin_module.sigma_x_np(state, next_site)
        s2, sx_i    = operators_spin_module.sigma_x_np(s1, this_site)
        coupling_v  = g0 * au[part_idx] * sx_i * sx_n
        # Store the results
        local_value         += local_in
        new_states[i, :]    = s2
        vals[i]             = coupling_v

    new_states[size, :] = state
    vals[size]          = local_value

    return new_states, vals

# @numba.njit
def _local_energy_arr_np(state, neidot, h, g0, n, au):
    '''
    Compute the local energy interaction for a numpy array state.
    Parameters:
        state           : numpy array of spins (0 or 1)
        neidot          : numpy array of neighbor indices for free particles
        h, g0, n, au    : model parameters
    Returns:
        Tuple [states], [energies] containing the new states and energies.
    '''
    # sigma_x = operators_spin_module.sigma_x_np
    # sigma_z = operators_spin_module.sigma_z_np
    return _local_energy_arr_loop(state, neidot, h, g0, n, au)

# ----------------------------------------------------------------------------------------

if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import jit as jax_jit
    
    # @jax_jit
    def _local_energy_arr_lax(state     : jnp.ndarray,
                            neidot      : jnp.ndarray,
                            h           : jnp.ndarray,
                            g0          : float,
                            n           : int,
                            au          : jnp.ndarray,
                            sigma_x,
                            sigma_z) -> (jnp.ndarray, jnp.ndarray):
        """
        Compute the local energy interaction for a JAX array state.
        
        Parameters:
            state   : jnp.ndarray of spins (0 or 1), shape (S,)
            neidot  : jnp.ndarray of neighbor indices for free particles, shape (free_count,)
            h       : jnp.ndarray of magnetic field values, shape (free_count,)
            g0      : float, coupling strength
            n       : int, number of particles in the dot
            au      : jnp.ndarray of coupling strengths, shape (free_count,)
            sigma_x : function(state, site) -> (new_state, sign)
                    Operator helper to compute the σₓ operation.
            sigma_z : function(state, site) -> (state, eigenvalue)
                    Operator helper to compute the σ_z operation.
                    
        Returns:
            Tuple (all_states, all_values) where:
                all_states: jnp.ndarray of shape (free_count+1, S) containing the new states.
                all_values: jnp.ndarray of shape (free_count+1,) containing the corresponding energies.
        """
        
        # Get the size of the state - assumed to be the same for all states.
        size                    = state.shape[0]
        h_in                    = jnp.array(h)
        au_in                   = jnp.array(au)
        neidot_in               = jnp.array(neidot)
        # Assume free indices: i from n to size-1.
        def scan_fun(local_value, i):
            this_idx            = jnp.ones((1,), dtype = jnp.int32) * i
            part_idx            = n - i
            # Diagonal term: apply σ_z at site i.
            _, sz_i             = sigma_z(state, this_idx)
            local_in            = sz_i * h_in[part_idx]
            # Off-diagonal term: apply two successive σₓ operations.
            next_site           = jnp.ones((1,), dtype = jnp.int32) * neidot_in[part_idx]
            s1, sx_n            = sigma_x(state, next_site)
            s2, sx_i            = sigma_x(s1, this_idx)
            coupling_v          = g0 * au_in[part_idx] * sx_n * sx_i
            # Accumulate the σ_z contribution.
            new_local_value     = local_value + local_in
            # For each free site we output the new state (from σₓσₓ) and the coupling energy.
            return new_local_value, (s2, coupling_v)
        
        free_indices            = jnp.arange(n, size, dtype=jnp.int32)
        init_local_value        = 0.0
        final_local_value, (free_states, free_values) = jax.lax.scan(scan_fun,
                                                                    init_local_value,
                                                                    free_indices)
        # Concatenate the free sites' states with the original state.
        all_states = jnp.concatenate([free_states, state[None, :]], axis=0)
        all_values = jnp.concatenate([free_values, jnp.array([final_local_value])], axis=0)
        return all_states, all_values
    
    # @jax_jit
    def _local_energy_arr_jax(state, neidot, h, g0, n, au):
        '''
        Compute the local energy interaction for a JAX array state.
        Parameters:
            state   : numpy array of spins (0 or 1)
            neidot  : numpy array of neighbor indices for free particles
            h, g0, n, au : model parameters
        Returns:
            Tuple [states], [energies] containing the new states and energies.
        '''
        sigma_x = operators_spin_module.sigma_x_jnp
        sigma_z = operators_spin_module.sigma_z_jnp
        return _local_energy_arr_lax(state, neidot, h, g0, n, au, sigma_x, sigma_z)

def _local_energy_arr_wrap(n, neidot, h, g0, au, use_jax = False):
    '''Creates a JIT-compiled local energy interaction function.'''
    if use_jax and _JAX_AVAILABLE:
        # @jax_jit
        def wrapper(state):
            return _local_energy_arr_jax(state, neidot, h, g0, n, au)
        return wrapper
    else:
        # @numba.njit
        def wrapper(state):
            return _local_energy_arr_np(state, neidot, h, g0, n, au)
        return wrapper

##########################################################################################
#! HAMILTONIAN CLASS
##########################################################################################

class QSM(hamil_module.Hamiltonian):
    '''
    Hamiltonian for an ergodic quantum dot coupled to an external system.
    The external system is modeled as a quantum spin chain.
    '''

    #############################
    
    _ERR_PARTICLES_DONT_MATCH   = "QSM: the number of particles in the dot is larger than the number of particles in the system."
    _ERR_COUP_VEC_SIZE          = "QSM: the size of the coupling vector 'a' is not equal to the number of particles outside the dot."
    _ERR_RAND_BOX_SIZE          = "QSM: the size of the random box distribution vector 'xi' is not equal to the number of particles outside the dot."
    _ERR_MAG_FIELD_SIZE         = "QSM: the size of the magnetic field vector 'h' is not equal to the number of particles outside the dot."
    _ERR_EITHER_HIL_OR_NS       = "QSM: either the Hilbert space or the number of particles in the system must be provided."
    
    def __init__(self,
                ns                  : Optional[int]                             = None,
                hilbert_space       : Optional[hilbert_module.HilbertSpace]     = None,
                n                   : int                                       = 1,
                gamma               : float                                     = 1.0,
                g0                  : float                                     = 1.0,
                a                   : Union[List[float], None, float]           = None,
                h                   : Union[List[float], None, float]           = 1.0,
                xi                  : Union[List[float], None, float]           = 0.2,
                dtype               : type                                      = np.float32,
                backend             : str                                       = "default",
                **kwargs):
        '''
        Constructor for the QSM Hamiltonian.
        Parameters:
            hilbert_space : HilbertSpace
                Hilbert space for the Hamiltonian - this shall be the Hilbert space for spin 1/2 particles.
            n : int
                Number of particles in the dot (ergodic quantum dot).
            gamma : float
                Hilbert-Schmidt norm of the coupling operator normalizer.
            g0 : float
                Coupling strength between the particles in the dot and the particles outside the dot.
            a : Union[List[float], None, float]
                Coupling between the dot and outside world
            h : Union[List[float], None, float]
                Magnetic field vector for the particles outside the dot. If it is None, it will be
                initialized to [1.0] * (Ns - N). If it is a float, it will be initialized to [h] * (Ns - N).
                If it is a list, it will be used as is.
            xi : Union[List[float], None, float]
                Random box distribution vector for the particles outside the dot. If 
                it is None, it will be initialized to [0.2] * (Ns - N). If it is a float,
                it will be initialized to [xi] * (Ns - N).
            dtype : type
                Data type for the Hamiltonian matrix. Default is np.float64.
            backend : str
                Backend for the Hamiltonian matrix (default is "default").
        '''

        # Initialize the Hamiltonian
        if hilbert_space is None:
            if ns is None:
                raise ValueError(self._ERR_EITHER_HIL_OR_NS)
            hilbert_space = hilbert_module.HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)
        
        # Initialize the Hamiltonian
        super().__init__(hilbert_space, is_sparse=True, dtype=dtype, backend=backend, **kwargs)
        
        # setup the internal variables
        self._n     =  n                # Number of particles in the dot
        self._gamma = gamma             # Hilbert-Schmidt norm of the coupling operator normalizer
        self._g0    = g0                # Coupling strength between the particles in the dot and the particles outside the dot
        self._nout  = 0                 # Number of particles outside the dot
        self._a, self._h, self._xi      = None, None, None
        self.check_sizes(a, h, xi)      # Check the sizes of the coupling vector, random box distribution vector, and magnetic field vector       
        
        # Initialize the Hamiltonian
        self._hdot          = None
        self._dimin         = 0
        self._dimout        = 0
        self._neidot        = []
        self._au            = None
        self. _u            = None
        self._name          = "Quantum Sun Model"
        self._startns       = n
        self._is_sparse     = True
        self._max_local_ch  = 2
        self.init_particles()
        # test the Hamiltonian and allow jit to be built - trigger the jit compilation        
        self._hamil                     = None
        self._loc_energy_int_fun        = _local_energy_int_wrap(self._n, self.ns, self._neidot, self._h, self._g0, self._au)
        self._loc_energy_np_fun         = _local_energy_arr_wrap(self._n, self._neidot, self._h, self._g0, self._au, use_jax=False)
        self._std_en                    = None
        if _JAX_AVAILABLE:
            self._loc_energy_jax_fun    = _local_energy_arr_wrap(self._n, self._neidot, self._h, self._g0, self._au, use_jax=True)       
        self._local_energy_test()
    
    # ----------------------------------------------------------------------------------------------
    
    def __repr__(self):
        ''' Return the string representation of the QSM model. '''
        sep     = ","
        skip    = []
        # Check if all elements except the first in each list are equal to the first value
        different_alpha = ( not all(np.isclose(a, self.alphas[0]) for a in self.alphas[1:])
                            if len(self.alphas) > 1
                            else False)
        different_h = ( not all(np.isclose(h, self.h[0]) for h in self.h[1:])
                        if len(self.h) > 1
                        else False)
        different_xi = (not all(np.isclose(xi, self.xi[0]) for xi in self.xi[1:])
                        if len(self.xi) > 1
                        else False)

        name    =  sep + f"qsm,Ns={self.ns}"
        name    += sep + f"N={self.n}"
        name    += sep + f"gamm={self.gamma:.2f}"
        name    += sep + f"g0={self.g0:.2f}"

        if "alpha" not in skip:
            name    += sep + ("alpha=r" if different_alpha else f"alpha={self.alphas[0]:.3f}")
        if "h" not in skip:
            name    += sep + ("h=r" if different_h else f"h={self.h[0]:.3f}")
        if "xi" not in skip:
            name    += sep + ("xi=r" if different_xi else f"xi={self.xi[0]:.3f}")

        name += self._hilbert_space.get_sym_info()
        return name

    def __str__(self):
        ''' Return the string representation of the QSM model. '''
        return self.__repr__()

    # ----------------------------------------------------------------------------------------------
    
    @property
    def n(self) -> int:
        ''' Return the number of particles in the dot. '''
        return self._n
    
    @property
    def nout(self) -> int:
        ''' Return the number of particles outside the dot. '''
        return self._nout
    
    @property
    def gamma(self) -> float:
        ''' Return the Hilbert-Schmidt norm of the coupling operator normalizer. '''
        return self._gamma
    
    @property
    def g0(self) -> float:
        ''' Return the coupling strength between the particles in the dot and the particles outside the dot. '''
        return self._g0
    
    @property
    def alphas(self) -> List[float]:
        ''' Return the coupling vector alpha. '''
        return self._a
    
    @property
    def h(self) -> List[float]:
        ''' Return the magnetic field vector. '''
        return self._h
    
    @property
    def xi(self) -> List[float]:
        ''' Return the random box distribution vector. '''
        return self._xi

    # ----------------------------------------------------------------------------------------------

    def check_sizes(self, a, h, xi):
        '''
        Check the sizes of the coupling vector, random box distribution vector, and magnetic field vector.
        
        Parameters:
            a : Union[List[float], None, float]
                Coupling between the dot and outside world
            h : Union[List[float], None, float]
                Magnetic field vector for the particles outside the dot
            xi : Union[List[float], None, float]
                Random box distribution vector for the particles outside the dot
        '''
        
        if self.n >= self.ns:
            raise ValueError(self._ERR_PARTICLES_DONT_MATCH)

        # define the number of particles outside the dot
        self._nout = self.ns - self.n
        self.set_alpha(a, initialize = False)
        self.set_h(h, initialize = False)
        self.set_xi(xi, initialize = False)

    # ----------------------------------------------------------------------------------------------

    def __init_hdot(self):
        ''' Initialize the random Hamiltonian for the particles outside the dot. '''
        hdot = None
        if np.issubdtype(self._dtype, np.complexfloating):
            hdot  = random_matrix((self._dimin, self._dimin), typek = RMT.CUE,
                                        backend=self._backend, dtype=self._dtype)
        else:
            hdot  = random_matrix((self._dimin, self._dimin), typek = RMT.GOE,
                                        backend=self._backend, dtype=self._dtype)
        hdot      = self._gamma / np.sqrt(self._dimin + 1) * hdot
        
        if _QSM_CHECK_HS_NORM:
            _norm           = linalg.hilbert_schmidt_norm(hdot, backend=self._backend)
            self._log(f"H_dot norm: {_norm:.3e}", lvl = 2, log = 'debug')
            return hdot / np.sqrt(_norm)
        return hdot

    def __init_distances(self):
        ''' Initialize the random distances for the 'free' particles. '''
        nout        = self.nout
        xi          = self._xi
        u           = np.zeros(nout)
        for i in range(1, nout):
            u[i]    = np.random.uniform(low=i-xi[i], high=i+xi[i])
        self._u     = u
        return u
    
    def __init_a_distances(self):
        ''' Initialize the random distances for the 'free' particles. '''
        self._au = self._a ** self._u
    
    def init_particles(self):
        ''' Initialize the particles in the dot and outside the dot. '''
        self._dimin     = 2**(self.n)
        self._dimout    = 2**(self.nout)

        # initialize random neighbors for the 'free' particles
        self._neidot    = np.random.choice(list(range(self.n)), self.nout, replace=True)

        # initialize the random distances for the 'free' particles
        self.__init_distances()
        self.__init_a_distances()

        # log information
        self._log("alpha = [{}]".format(", ".join(f"{val:.3f}" for val in self._a)), lvl=1, log = 'debug')
        self._log("u = [{}]".format(", ".join(f"{val:.3f}" for val in self._u)), lvl=2, log = 'debug')
        self._log("alpha^u = [{}]".format(", ".join(f"{val:.3f}" for val in self._au)), lvl=2, log = 'debug')

        # initialize the Hamiltonian in the dot
        self._hdot      = self.__init_hdot()
        
        # based on the backend, convert the Hamiltonian to the appropriate type
        if self._backend != np:
            self._hdot      = self._backend.array(self._hdot, dtype = self._backend.float64)
        
        ######
        self._au        = np.array(self._au, dtype = np.float64)
        self._u         = np.array(self._u, dtype = np.float64)
        self._h         = np.array(self._h, dtype = np.float64)
        self._a         = np.array(self._a, dtype = np.float64)
        self._xi        = np.array(self._xi, dtype = np.float64)
        self._neidot    = np.array(self._neidot, dtype = np.int64)
    
    # ----------------------------------------------------------------------------------------------

    def set_alpha(self, a, initialize = True):
        ''' Set the coupling vector alpha. '''
        if isinstance(a, list) and len(a) == self.nout:
            self._a = a
        elif isinstance(a, float):
            self._a = [a] * self.nout
        elif isinstance(a, str):
            self._a = random_vector(self.nout, a, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_COUP_VEC_SIZE)
        
        # calculate the new coupling strength
        if initialize:
            self.__init_distances()
        
    def set_h(self, h, initialize = True):
        ''' Set the magnetic field vector. '''
        if isinstance(h, list) and len(h) == self.nout:
            self._h = h
        elif isinstance(h, float) or initialize:
            self._h = random_vector(self.nout, typek = f'r;{h-0.5};{h+0.5}', backend=self._backend, dtype=self._dtype)
        elif isinstance(h, str):
            self._h = random_vector(self.nout, h, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_MAG_FIELD_SIZE)
    
    def set_xi(self, xi, initialize = True):
        ''' Set the random box distribution vector. '''
        if isinstance(xi, list) and len(xi) == self.nout:
            self._xi = xi
        elif isinstance(xi, float):
            self._xi = [xi] * self.nout
        elif isinstance(xi, str):
            self._xi = random_vector(self.nout, xi, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_RAND_BOX_SIZE)
        
        # calculate the new random distances
        if initialize:
            self.__init_distances()
            self.__init_a_distances()
            
    # ----------------------------------------------------------------------------------------------
    
    def randomize(self, **kwargs):
        '''
        Randomize the QSM Hamiltonian. 
        Parameters:
            alpha : Union[List[float], None]
        Coupling between the dot and outside world
            h : Union[List[float], None]
        Magnetic field vector for the particles outside the dot
            xi : Union[List[float], None]
        Random box distribution vector for the particles outside the dot
        '''
        if "alpha" in kwargs:
            self.set_alpha(kwargs["alpha"], False)
        if "h" in kwargs:
            self.set_h(kwargs["h"], False)
        if "xi" in kwargs:
            self.set_xi(kwargs["xi"], False)
        self.init_particles()

    # ----------------------------------------------------------------------------------------------

    def get_mobility_edge(self, energy: float):
        ''' 
        Compute the mobility edge for the QSM model.
        It is calculated as in: Konrad Pawlik, Piotr Sierant, Lev Vidmar, Jakub Zakrzewski (2023)
        '''
        if self._nh == 0:
            self._log("Empty Hilbert, not counting anything.", "INFO", 1)
            return None
        
        if self._std_en is None:
            self._std_en    = np.std(self.eig_val)

        _std                = self._std_en / np.sqrt(self.ns)
        _eps                = (energy - self.eig_val[0]) / (self.eig_val[self._nh - 1] - self.eig_val[0])
        _bwd                = (self.eig_val[self._nh - 1] - self.eig_val[0])
        _bwd                = _bwd / float(self.ns)
        return np.exp(_bwd * _bwd * (_eps - 0.5) * (_eps - 0.5) / _std / _std / 4.0) / np.sqrt(2.0)

    # ----------------------------------------------------------------------------------------------

    def _hamiltonian(self, use_numpy : bool = False):
        '''
        Build the Hamiltonian of the QSM model.
        The Hamiltonian is built in the following way:
        1. The first term is the Hamiltonian of the dot particles. This is a random matrix.
        2. The second term is the coupling between the dot and the outside world. This is a spin-flip interaction.
        3. The third term is the magnetic field of the dot particles.
        '''
        
        if self._nh == 0:
            self._log("Empty Hilbert, not building anything.", "INFO", 1)
            return

        self._log(f"QSM: alpha={self._a[0]:.3f}, xi={self._xi[0]:.3f}", lvl = 2, log = 'debug')
        
        for i in range(self.nout):
            self._log(f"QSM: i={i} -> h={self._h[i]:.3e}, a^u={self._au[i]:.3e}", lvl = 3, log = 'debug')

        super()._hamiltonian(use_numpy)
        
        # add the Hamiltonian of the dot particles
        backend_changed = self._backend if not use_numpy else np
        eye         = linalg.sparse.identity(self._dimout, backend=backend_changed, dtype=self._dtype)
        kron_prod   = linalg.sparse.kron(self._hdot, eye, backend=backend_changed)
        self._hamil += kron_prod

    # ----------------------------------------------------------------------------------------------

####################################################################################################