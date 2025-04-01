"""
file    : Solver/MonteCarlo/sampler.py
author  : Maksymilian Kliczkowski
date    : 2025-02-01

Monte Carlo samplers for quantum state space.
This module provides samplers for efficiently exploring the Hilbert space 
of quantum systems using Monte Carlo techniques. The main component is the 
MCSampler class which implements Markov Chain Monte Carlo sampling for quantum
wavefunctions.
The module supports:
- Sampling from quantum state space according to Born distribution or modified distributions
- Multiple concurrent Markov chains
- Different initial state configurations (random, ferromagnetic, antiferromagnetic)
- Customizable state update proposals
- Both NumPy and JAX backends
Classes:
    SamplerErrors   : Error messages related to sampler operations
    SolverInitState : Enum for different types of initial states
    Sampler         : Abstract base class for samplers
    MCSampler       : Concrete MCMC sampler implementation for quantum states
    SamplerType     : Enum for different sampler types
Functions:
    get_sampler     : Factory function for creating samplers
"""


import random
import numpy as np
import numba
import scipy as sp
from numba import jit, njit, prange
from typing import Union, Tuple, Union, Callable, Optional, Any
from functools import partial

# flax for the network
from flax import linen as nn

# for the abstract class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique

# from algebra
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend, JIT, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_INT_TYPE, _KEY
from general_python.algebra.utils import DEFAULT_JP_CPX_TYPE, DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, DEFAULT_NP_CPX_TYPE
from general_python.algebra.ran_wrapper import choice, randint, uniform, randint_np, randint_jax
from general_python.common.directories import Directories
import general_python.common.binary as Binary

# from hilbert
from Algebra.hilbert import HilbertSpace

# JAX imports
if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.random as random_jp

#########################################################################
#! Errors
#########################################################################

class SamplerErrors(Exception):
    """
    Errors for the Sampler class.
    """
    NOT_GIVEN_SIZE_ERROR        = "The size of the system is not given"
    NOT_IMPLEMENTED_ERROR       = "This feature is not implemented yet"
    NOT_A_VALID_STATE_STRING    = "The state string is not a valid state string"
    NOT_A_VALID_STATE_DISTING   = "The state is not a valid state"
    NOT_A_VALID_SAMPLER_TYPE    = "The sampler type is not a valid sampler type"
    NOT_IN_RANGE_MU             = "The parameter \\mu must be in the range [0, 2]"
    NOT_HAVING_RNG              = "Either rng or seed must be provided"

class SolverInitState(Enum):
    """Enum for potential initial states """
    
    # -----------------------
    
    RND         = auto()    # random configuration
    F_UP        = auto()    # ferromagnetic up
    F_DN        = auto()    # ferromagnetic down
    AF          = auto()    # antiferromagnetic
    
    # -----------------------
    
    def __str__(self):
        """Return the name of the enum member."""
        return self.name

    # -----------------------

    @classmethod
    def from_str(cls, state_str: str):
        """
        Create an enum member from a string, ignoring case.
        Parameters:
            state_str (str)     : The string representation of the enum member.
        Returns:
            SolverInitState     : The enum member corresponding to the input string.
        """
        # Normalize input (upper-case) to match enum member names
        normalized = state_str.upper()
        if normalized in cls.__members__:
            return cls.__members__[normalized]
        raise ValueError(f"Unknown initial state: {state_str}")

    # -----------------------

#########################################################################
#! Propose a random flip
#########################################################################

if _JAX_AVAILABLE:
    @jax.jit
    def _propose_random_flip_jax(state: jnp.ndarray, rng_k):
        """
        Propose a random flip of a state using JAX.
        Parameters:
        - state (jnp.ndarray)           : The state array
        - rng_k (jax.random.PRNGKey)    : The random key for JAX
        - num (int)                     : The number of flips to propose
        Returns:
        - jnp.ndarray: The proposed flipped state
        """
        idx = randint_jax(key=rng_k, shape=(1,), low=0, high=state.size)[0]
        return Binary.flip_array_jax_spin(state, idx)

    @jax.jit
    def _propose_random_flips_jax(state: jnp.ndarray, rng_k, num = 1):
        """0
        Propose a random flip of a state using JAX.
        Parameters:
        - state (jnp.ndarray)           : The state array
        - rng_k (jax.random.PRNGKey)    : The random key for JAX
        - num (int)                     : The number of flips to propose
        Returns:
        - jnp.ndarray: The proposed flipped state
        """
        idx = randint_jax(key=rng_k, shape=(num,), low=0, high=state.size, dtype=DEFAULT_JP_INT_TYPE)
        return Binary.flip_array_jax_multi(state, idx, spin=Binary.BACKEND_DEF_SPIN)

@numba.njit
def _propose_random_flips_np(state: np.ndarray, rng, num = 1):
    """
    Propose a random flip of a state using numpy.
    """
    if state.ndim == 1:
        idx = randint_np(rng=rng, low=0, high=state.size, size=num)
        return Binary.flip_array_np_multi(state, idx,
                                        spin=Binary.BACKEND_DEF_SPIN, spin_value=Binary.BACKEND_REPR)
    n_chains, state_size = state.shape[0], state.shape[1]
    for i in range(n_chains):
        idx = randint_np(rng=rng, low=0, high=state_size, size=num)
        state[i] = Binary.flip_array_np_multi(state[i], idx,
                                        spin=Binary.BACKEND_DEF_SPIN, spin_value=Binary.BACKEND_REPR)
    return state

@numba.njit(parallel=True)
def _propose_random_flip_np(state: np.ndarray, rng: np.random.Generator):
    """
    Propose a random flip of a state using numpy.
    """
    if state.ndim == 1:
        idx = randint_np(rng=rng, low=0, high=state.size, size=1)[0]
        return Binary.flip_array_np(state, idx)
    
    n_chains, state_size = state.shape[0], state.shape[1]
    for i in prange(n_chains):
        idx         = randint_np(low=0, high=state_size, size=1)[0]
        state[i]    = Binary.flip_array_np_spin(state[i], idx)
    return state

def propose_random_flip(state: 'array-like', backend = 'default',
                        rng = None, rng_k = None, num = 1):
    """
    Propose a random flip of a state.
    """
    if backend == np or backend == 'numpy' or backend == 'np':
        return _propose_random_flip_np(state, rng) if num == 1 else _propose_random_flips_np(state, rng, num)
    return _propose_random_flip_jax(state, rng, rng_k) if num == 1 else _propose_random_flips_jax(state, rng, rng_k, num)

#########################################################################
#! Set the state of the system
#########################################################################

def _set_state_int(state        : int,
            modes               : int                           = 2,
            hilbert             : Optional[HilbertSpace]        = None,
            shape               : Union[int, Tuple[int, ...]]   = (1,),
            mode_repr           : float                         = Binary.BACKEND_REPR,
            backend             : str                           = 'default'
            ):
    '''
    Set the state configuration from the integer representation.
    
    Parameters:
    - state         : state configuration as an integer
    - modes         : modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape of the system (number of spins).
                    One may want to reshape the state to a given shape (for instance, 2D lattice).
    - mode_repr     : mode representation (default is 0.5 - for binary spins +-1)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    
    
    Transforms the integer to a given configuration 
    Notes:
        The states are given in binary or other representation 
            : 2 for binary (e.g. spin up/down)
            : 4 for fermions (e.g. fermions up/down)
                - The lower size bits (state & ((1 << size)-1)) encode the up orbital occupations.
                - The upper size bits (state >> size) encode the down orbital occupations.
            ...
        It uses the mode representation to determine the spin value:
        Examples:
        - spins 1/2 are created as +-0.5 when mode_repr = 0.5 (default) and _modes = 2.
                Thus, we need _size to represent the state.
        - fermions are created as 1/-1 when mode_repr = 1.0 and _modes = 2 
                and the first are up spins and the second down spins. Thus, we 
                need 2 * _size to represent the state and we have 0 and ones for the
                presence of the fermions.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    # check the size from the hilbert
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    out  = None
    
    ############################################################################
    if hilbert is None:
        if modes == 2:
            # set the state from tensor
            out = Binary.int2base(state, size, backend, 
                                spin_value=mode_repr, spin=Binary.BACKEND_DEF_SPIN).reshape(shape)
        elif modes == 4:
            # For fermions, split the state into up and down parts.
            up_int          = state & ((1 << size) - 1)
            down_int        = state >> size
            up_array        = np.array([1 if (up_int & (1 << i)) else 0 for i in range(size)],
                                dtype=DEFAULT_NP_FLOAT_TYPE)
            down_array      = np.array([1 if (down_int & (1 << i)) else 0 for i in range(size)],
                                dtype=DEFAULT_NP_FLOAT_TYPE)
            # Stack to form a (2, size) array: first row up, second row down.
            out             = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    return out
    
def _set_state_rand(modes       : int                           = 2,
                    hilbert     : Optional[HilbertSpace]        = None,
                    shape       : Union[int, Tuple[int, ...]]   = (1,),
                    mode_repr   : float                         = 0.5,
                    backend     : str                           = 'default',
                    rng         = None,
                    rng_key     = None
                    ):
    '''
    Generate a random state configuration.
    
    Parameters:
    - modes         : number of modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape of the state array (number of spins).
                      For fermions (modes == 4), an integer number of sites.
    - mode_repr     : mode representation (default is 0.5 for binary spins +-1, 1.0 for fermions)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    - rng           : random number generator for numpy
    - rng_key       : random key for JAX
    
    Returns:
    - A random state array.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    # check the size from hilbert
    size        = shape if isinstance(shape, int) else int(np.prod(shape))
    ran_state   = None
    ############################################################################
    if hilbert is None:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                ran_state = choice([-1, 1], shape, rng=rng, rng_k=rng_key, backend=backend)
            else:
                ran_state = choice([0, 1], shape, rng=rng, rng_k=rng_key, backend=backend)
        elif modes == 4:
            # Generate random occupancy for 2 * size orbitals.
            ran_state = choice([0, 1], 2 * size, rng=rng, rng_k=rng_key, backend=backend)
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    return ran_state.astype(DEFAULT_NP_FLOAT_TYPE).reshape(shape) * mode_repr

def _set_state_up(modes         : int                           = 2,
                  hilbert       : Optional[HilbertSpace]        = None,
                  shape         : Union[int, Tuple[int, ...]]   = (1,),
                  mode_repr     : float                         = 0.5,
                  backend       : str                           = 'default'
                  ):
    '''
    Generate an "all up" state configuration.
    
    Parameters:
    - modes         : number of modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape (number of spins).
                      For fermions (modes == 4), an integer number of sites.
    - mode_repr     : mode representation (default is 0.5 for binary spins +-1, 1.0 for fermions)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    
    Returns:
    - An "all up" state array.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    ############################################################################
    xp = get_backend(backend)
    if hilbert is None:
        if modes == 2:
            return xp.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE).reshape(shape) * mode_repr
        elif modes == 4:
            # For fermions, "up" means up orbitals occupied and down orbitals empty.
            up_array   = np.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE)
            down_array = np.zeros(size, dtype=DEFAULT_NP_FLOAT_TYPE)
            out        = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
            return out
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

def _set_state_down(modes       : int                           = 2,
                    hilbert     : Optional[HilbertSpace]        = None,
                    shape       : Union[int, Tuple[int, ...]]   = (1,),
                    mode_repr   : float                         = 0.5,
                    backend     : str                           = 'default'
                    ):
    '''
    Generate an "all down" state configuration.
    
    Parameters:
    - modes         : number of modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape (number of spins).
                      For fermions (modes == 4), an integer number of sites.
    - mode_repr     : mode representation (default is 0.5 for binary spins +-1, 1.0 for fermions)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    
    Returns:
    - An "all down" state array.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    ############################################################################
    xp = get_backend(backend)
    if hilbert is None:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                return xp.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE).reshape(shape) * (-mode_repr)
            else:
                return xp.zeros(size, dtype=DEFAULT_NP_FLOAT_TYPE).reshape(shape)
        elif modes == 4:
            # For fermions, "down" means up orbitals empty and down orbitals occupied.
            up_array   = np.zeros(size, dtype=DEFAULT_NP_FLOAT_TYPE)
            down_array = np.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE)
            out        = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
            return out
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

def _set_state_af(modes         : int                           = 2,
                  hilbert       : Optional[HilbertSpace]        = None,
                  shape         : Union[int, Tuple[int, ...]]   = (1,),
                  mode_repr     : float                         = 0.5,
                  backend       : str                           = 'default'
                  ):
    '''
    Generate an antiferromagnetic state configuration.
    
    Parameters:
    - modes         : number of modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape (number of spins).
                      For fermions (modes == 4), an integer number of sites.
    - mode_repr     : mode representation (default is 0.5 for binary spins +-1, 1.0 for fermions)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    
    Returns:
    - An antiferromagnetic state array.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    ############################################################################
    xp = get_backend(backend)
    if hilbert is None:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                af_state = np.array([1 if i % 2 == 0 else -1 for i in range(size)],
                                    dtype=DEFAULT_NP_FLOAT_TYPE)
            else:
                af_state = np.array([1 if i % 2 == 0 else 0 for i in range(size)],
                                    dtype=DEFAULT_NP_FLOAT_TYPE)
            return af_state.reshape(shape) * mode_repr
        elif modes == 4:
            # For fermions, antiferromagnetic state:
            # up orbitals: 1 at even sites, 0 at odd sites
            # down orbitals: 0 at even sites, 1 at odd sites
            up_array   = np.array([1 if i % 2 == 0 else 0 for i in range(size)],
                                  dtype=DEFAULT_NP_FLOAT_TYPE)
            down_array = np.array([0 if i % 2 == 0 else 1 for i in range(size)],
                                  dtype=DEFAULT_NP_FLOAT_TYPE)
            out        = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
            return out
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

def _state_distinguish(statetype,
                       modes       : int,
                       hilbert     : Optional[HilbertSpace],
                       shape       : Union[int, Tuple[int, ...]],
                       mode_repr   : float = Binary.BACKEND_REPR,
                       rng         = None,
                       rng_key     = None,
                       backend     : str = 'default'):
    """
    Distinguishes the type of the given state and returns the appropriate state configuration.
    
    Parameters:
    - statetype (int, jnp.ndarray, np.ndarray, str, SolverInitState): The state specification.
        * If int: Converts to configuration using _set_state_int.
        * If array: Returns the array directly.
        * If str: Converts to SolverInitState and processes accordingly.
        * If SolverInitState: Processes according to the enum value.
    - modes         : number of modes (2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system.
                      For modes == 2, the desired shape (number of spins);
                      For modes == 4, an integer number of sites.
    - mode_repr     : mode representation value.
    - rng           : random number generator for numpy.
    - rng_key       : random key for JAX.
    - backend       : computational backend ('default', 'numpy', or 'jax').
    
    Returns:
    - The corresponding state configuration as an ndarray.
    """
    if isinstance(statetype, (int, np.integer, jnp.integer)):
        return _set_state_int(statetype, modes, hilbert, shape, mode_repr, backend)
    elif isinstance(statetype, (np.ndarray, jnp.ndarray)):
        return statetype
    elif isinstance(statetype, str):
        try:
            state_enum = SolverInitState.from_str(statetype)
            return _state_distinguish(state_enum, modes, hilbert, shape, mode_repr, rng, rng_key, backend)
        except ValueError as e:
            raise ValueError(SamplerErrors.NOT_A_VALID_STATE_STRING) from e
    elif isinstance(statetype, SolverInitState):
        if statetype == SolverInitState.RND:
            return _set_state_rand(modes, hilbert, shape, mode_repr, backend, rng, rng_key)
        elif statetype == SolverInitState.F_UP:
            return _set_state_up(modes, hilbert, shape, mode_repr, backend)
        elif statetype == SolverInitState.F_DN:
            return _set_state_down(modes, hilbert, shape, mode_repr, backend)
        elif statetype == SolverInitState.AF:
            return _set_state_af(modes, hilbert, shape, mode_repr, backend)
    else:
        raise ValueError(SamplerErrors.NOT_A_VALID_STATE_DISTING)

########################################################################
#! Sampler class for Monte Carlo sampling
########################################################################

class Sampler(ABC):
    """
    A base class for the sampler.     
    """
    
    def __init__(self,
                shape       : Tuple[int, ...],
                upd_fun     : Callable,
                rng,
                rng_k                                           = None,
                seed        : Optional[int]                     = None,
                hilbert     : Optional[HilbertSpace]            = None,
                numsamples  : int                               = 1,
                numchains   : int                               = 1,
                initstate   : Union[np.ndarray, jnp.ndarray]    = None,
                backend     : str                               = 'default',
                **kwargs
            ):
        """
        Abstract base class for samplers.
        Parameters:
            shape (Tuple[int, ...])             : Shape of the system.
            upd_fun (Callable)                  : Update function for proposing new states.
            rng                                 : Random number generator.
            rng_k (Optional[jax.random.PRNGKey]): Random key for JAX.
            seed (Optional[int])                : Seed for random number generator.
            hilbert (Optional[HilbertSpace])    : Hilbert space instance.
            numsamples (int)                    : Number of samples.
            numchains (int)                     : Number of chains.
            initstate ('array-like')            : Initial state configuration.
            backend (str)                       : Computational backend ('default', 'numpy', or 'jax').
            kwargs (dict)                       : Additional arguments for state generation.
        Raises:
            SamplerErrors: If the backend is not valid or if the initial state is not valid.
            
        Attributes:
            _shape (Tuple[int, ...])            : Shape of the system.
            _size (int)                         : Total number of sites.
            _hilbert (Optional[HilbertSpace])   : Hilbert space instance.
            _rng, _rng_k                        : Random number generatozrs.
            _backend                            : Computational backend (e.g., numpy or jax).
            _numsamples (int)                   : Number of samples.
            _numchains (int)                    : Number of chains.
            _initstate                          : Initial state configuration.
            _states                             : Current state chains.
            _num_proposed, _num_accepted        : Counters for proposed and accepted moves.
            _upd_fun                            : Update function for proposing new states.
        """
        # check the backend
        if isinstance(backend, str):
            self._backendstr = backend
            
        if rng is not None:
            self._rng       = rng
            self._rng_k     = rng_k if rng_k is not None else _KEY
            self._backend   = get_backend(backend)
        elif seed is not None:
            self._backend, _, (self._rng, self._rng_k) = self.obtain_backend(backend, seed)
        else:
            raise SamplerErrors(SamplerErrors.NOT_HAVING_RNG)
        
        # set the backend
        self._isjax         = (not self._backend == np)
        self._backendstr    = 'np' if not self._isjax else 'jax'
        
        # handle the Hilbert space - may control state initialization
        self._hilbert       = hilbert
        
        # handle the states
        self._shape         = shape
        self._size          = int(np.prod(shape)) if isinstance(shape, tuple) else shape
        self._numsamples    = numsamples
        self._numchains     = numchains
        self._states        = None
        
        # handle the initial state
        self.set_initstate(initstate, **kwargs)
        
        # proposed state
        self._num_proposed  = self._backend.zeros(numchains, dtype=self._backend.int64)
        self._num_accepted  = self._backend.zeros(numchains, dtype=self._backend.int64)
        
        # handle the update function
        self._upd_fun       = upd_fun
        if self._upd_fun is None:
            if self._isjax:
                # Bind RNG arguments to the JAX updater and then wrap with JIT.
                self._upd_fun = _propose_random_flip_jax
            else:
                # For NumPy backend, bind the RNG to the updater.
                self._upd_fun = _propose_random_flip_np
    
    ###################################################################
    #! ABSTRACT
    ###################################################################
    
    @abstractmethod
    def sample(self, parameters=None, num_samples=None, num_chains=None):
        ''' Tries to sample the state from the Hilbert space. '''

    ###################################################################
    #! BACKEND
    ###################################################################

    @staticmethod
    def obtain_backend(backend: str, seed: Optional[int]):
        '''
        Obtain the backend for the calculations.
        Parameters:
        - backend       : backend for the calculations (default is 'default')
        - seed          : seed for the random number generator
        Returns:
        - Tuple         : backend, backend_sp, rng, rng_key 
        '''
        if isinstance(backend, str):
            bck = get_backend(backend, scipy=True, random=True, seed=seed)
            if isinstance(bck, tuple):
                _backend, _backend_sp = bck[0], bck[1]
                if isinstance(bck[2], tuple):
                    _rng, _rng_k = bck[2][0], bck[2][1]
                else:
                    _rng, _rng_k = bck[2], None
            else:
                _backend, _backend_sp = bck, None
                _rng, _rng_k = None, None
            return _backend, _backend_sp, (_rng, _rng_k), backend
        _backendstr = 'np' if (backend is None or (backend == 'default' and not _JAX_AVAILABLE) or backend == np) else 'jax'
        return Sampler.obtain_backend(_backendstr, seed)
    
    ###################################################################
    #! PROPERTIES
    ###################################################################
    
    @property
    def hilbert(self):
        return self._hilbert
    
    @property
    def numsamples(self):
        return self._numsamples
    
    @property
    def numchains(self):
        return self._numchains
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def size(self):
        return self._size
    
    @property
    def initstate(self):
        return self._initstate
    
    @property
    def upd_fun(self):
        return self._upd_fun
    
    @property
    def states(self):
        return self._states
    
    @property
    def rng(self):
        return self._rng
    
    @property
    def rng_key(self):
        return self._rng_k
    
    @property
    def backend(self):
        return self._backend
    
    @property
    def proposed(self):
        return self._num_proposed
    
    @property
    def num_proposed(self):
        return self._backend.sum(self._num_proposed)
    
    @property
    def accepted(self):
        return self._num_accepted
    
    @property
    def num_accepted(self):
        return self._backend.sum(self._num_accepted)
    
    @property
    def rejected(self):
        return self._num_proposed - self._num_accepted
    
    @property
    def num_rejected(self):
        return self._backend.sum(self._num_proposed - self._num_accepted)
    
    @property
    def accepted_ratio(self):
        return self.num_accepted / self.num_proposed if self.num_proposed > 0 else 0.0
    
    @property
    def isjax(self):
        return self._isjax
    
    ###################################################################
    #! SETTERS
    ###################################################################
    
    def reset(self):
        """
        Reset the sampler to its initial state.
        """
        self._num_proposed = 0
        self._num_accepted = 0
        self.set_chains(self._initstate, self._numchains)
    
    # ---
    
    def set_initstate(self, initstate, **kwargs):
        """
        Set the initial state of the system.
        
        Parameters:
            initstate (str, int, np.ndarray, jnp.ndarray) : The initial state specification
            kwargs (dict)                                 : Additional arguments for state generation
        Raises:
            NotImplementedError: If the requested state type is not implemented
        Returns:
            None
        """
        
        # handle the initial state
        if initstate is None or isinstance(initstate, str):
            if self._hilbert is None or True:
                if initstate is None:
                    self._initstate = _set_state_rand(
                        modes       =   kwargs.get('modes', 2),
                        hilbert     =   self._hilbert,
                        shape       =   self._shape,
                        mode_repr   =  kwargs.get('mode_repr', 0.5),
                        backend     =   self._backend,
                        rng         =   self._rng,
                        rng_key     =   self._rng_k)
                else:
                    self._initstate = _state_distinguish(initstate,
                                        modes   =   kwargs.get('modes', 2),
                                        hilbert =   self._hilbert,
                                        shape   =   self._shape,
                                        backend =   self._backend,
                                        rng     =   self._rng,
                                        rng_key =   self._rng_k)
            else:
                raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
        else:
            if isinstance(initstate, np.ndarray) and self._isjax:
                self._initstate = jnp.array(initstate)
            elif isinstance(initstate, jnp.ndarray) and not self._isjax:
                self._initstate = np.array(initstate)
            else:
                self._initstate = initstate
        self.set_chains(self._initstate, self._numchains)
    
    # ---
    
    def set_chains(self, initstate: Union[np.ndarray, jnp.ndarray], numchains: Optional[int] = None):
        '''
        Set the chains for the sampler.
        Parameters:
        - initstate (np.ndarray, jnp.ndarray): The initial state
        - numchains (int, optional): The number of chains
        '''
        if numchains is None:
            numchains = self._numchains
        if self._isjax:
            self._states    = jnp.stack([jnp.array(initstate)] * numchains, axis=0)
        else:
            self._states    = np.stack([np.array(initstate.copy())] * numchains, axis=0)
            self._tmpstates = self._states.copy()   # for the temporary states (for the proposal)
    
    # ---
    
    def set_numsamples(self, numsamples):
        '''
        Set the number of samples.
        Parameters:
        - numsamples (int): The number of samples
        '''
        self._numsamples = numsamples
        
    def set_numchains(self, numchains):
        '''
        Set the number of chains.
        Parameters:
        - numchains (int): The number of chains
        '''
        self._numchains = numchains
        self.set_chains(self._initstate, numchains)
    
#######################################################################    

class MCSampler(Sampler):
    """    
    MCSampler implements a Markov chain Monte Carlo sampler that samples basis states from 
    the Hilbert space according to the Born distribution, i.e.,
    
        :math:`p_{\\mu}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`.
    
    For :math:`\\mu=2`, this corresponds to the standard Born distribution, while 
    :math:`0 \\le \\mu < 2` allows for importance sampling (see [arXiv:2108.08631](https://arxiv.org/abs/2108.08631)).
    
    **Sampling Process (Step-by-Step):**
        1. **Initialization:** The sampler is set up with the system shape, a network (or callable) for evaluating 
            the log amplitude, an update proposer function, and MCMC parameters (mu, beta, thermalization and sweep steps).
        2. **Network Callable Setup:** The network is processed via `_set_net_callable` to extract a callable and parameters.
        3. **Thermalization:** Multiple Markov chains are thermalized by sweeping state updates for a prescribed number of steps.
        4. **Sampling Sweeps:** Additional sweeps are performed to collect samples using the Metropolis-Hastings acceptance rule.
        5. **Postprocessing:** Collected samples are reshaped, and the network is applied to compute log amplitudes and normalized probabilities.

    This class supports parallel sampling with multiple chains and works with both JAX and NumPy backends.
    """
    
    def __init__(self,
                net,
                shape       : Tuple[int, ...],
                rng,
                rng_k                               = None,
                upd_fun     : Optional[Callable]    = None,
                mu          : float                 = 2.0,
                beta        : float                 = 1.0,
                therm_steps : int                   = 100,
                sweep_steps : int                   = 100,
                seed                                = None,
                hilbert     : HilbertSpace          = None,
                numsamples  : int                   = 1,
                numchains   : int                   = 1,
                initstate                           = None,
                backend     : str                   = 'default',
                logprob_fact: float                 = 0.5,
                **kwargs):
        super().__init__(shape, upd_fun, rng, rng_k, seed, hilbert,
                numsamples, numchains, initstate, backend, **kwargs)
        
        if net is None:
            raise ValueError("A network (or callable) must be provided for evaluation.")

        # set the network
        self._net_callable, self._parameters = self._set_net_callable(net)

        # set the parameters - this for modification of the distribution
        self._mu            = mu
        if self._mu < 0.0 or self._mu > 2.0:
            raise ValueError(SamplerErrors.NOT_IN_RANGE_MU)
        
        self._beta          = beta
        self._therm_steps   = therm_steps
        self._sweep_steps   = sweep_steps
        self._logprob_fact  = logprob_fact
        self._logprobas     = None
        
        # Set update function if not provided.
        self._upd_fun = upd_fun
        if self._upd_fun is None:
            if self._isjax:
                # Bind RNG arguments to the JAX updater and then wrap with JIT.
                self._upd_fun = JIT(_propose_random_flip_jax)
            else:
                # For NumPy backend, bind the RNG to the updater.
                self._upd_fun = _propose_random_flip_np
    
    #####################################################################
    
    def __repr__(self):
        return (f"MCSampler(shape={self._shape}, mu={self._mu}, beta={self._beta}, "
                f"therm_steps={self._therm_steps}, sweep_steps={self._sweep_steps}, "
                f"numsamples={self._numsamples}, numchains={self._numchains}, backend={self._backendstr})")
    
    def __str__(self):
        return (f"MCSampler: Sampling from states of shape {self._shape} with "
                f"{self._numchains} chains and {self._numsamples} samples per chain. "
                f"Parameters: mu={self._mu}, beta={self._beta}, thermal steps={self._therm_steps}, "
                f"sweep steps={self._sweep_steps}.")
    
    ###################################################################
    
    def _set_net_callable(self, net):
        '''
        Set the network callable.
        Parameters:
            - net_callable : The network callable
        '''
        
        # this method shall return the callable and the variational parameters
        self._net = net
        if hasattr(self._net, 'get_apply'):
            network_callable, parameters = self._net.get_apply()    # if is flax model or similar
            return network_callable, parameters
        elif callable(self._net):
            return self._net, None
        raise ValueError("The network callable is not valid.")
    
    ###################################################################
    #! ACCEPTANCE PROBABILITY
    ###################################################################
    
    def _acceptance_probability_jax(self, current_val, candidate_val, beta: float = 1.0):
        '''
        Calculate the acceptance probability for the Metropolis-Hastings
        algorithm.
        Parameters:
            - current_val   : The value of the current state
            - candidate_val : The value of the candidate state
        
        Returns:
            - The acceptance probability as a float
        '''

        log_acceptance_ratio = beta * jnp.real(candidate_val - current_val)
        return jnp.minimum(1.0, jnp.exp(log_acceptance_ratio))
    
    def _acceptance_probability_np(self, current_val, candidate_val, beta: float = 1.0):
        '''
        Calculate the acceptance probability for the Metropolis-Hastings
        algorithm.
        Parameters:
            - current_val   : The value of the current state
            - candidate_val : The value of the candidate state
        
        Returns:
            - The acceptance probability as a float
        '''

        log_acceptance_ratio = beta * np.real(candidate_val - current_val)
        return np.minimum(1.0, np.exp(log_acceptance_ratio))
    
    def acceptance_probability(self, current_val, candidate_val, beta: float = 1.0):
        '''
        Calculate the acceptance probability for the Metropolis-Hastings
        algorithm.
        Parameters:
            - current_val   : The value of the current state
            - candidate_val : The value of the candidate state
        It calculates:
            $$ exp(\\beta (log(p') - log(p))) $$
        Returns:
            - The acceptance probability as a float
        '''
        if beta is None:
            beta = self._beta
            
        if self._isjax:
            return self._acceptance_probability_jax(current_val, candidate_val, beta)
        return self._acceptance_probability_np(current_val, candidate_val, beta)

    ###################################################################
    #! LOG PROBABILITY
    ###################################################################
    
    def _logprob_jax(self, x, mu: float, net_callable, net_params = None):
        '''
        Calculate the log probability of a state using JAX.
        Parameters:
            - x             : The state
            - mu            : The parameter mu
            - net_callable  : The network callable (returns (\\log\\psi(s)))
            - net_params    : The network parameters - may be None but for flax callables 
                            the parameters are necessary strictly!
        Returns:
            - The log probability as a float or complex number
        '''
        # if net_params is None:
            # If no parameters are needed, call net_callable with just y
            # return jax.vmap(lambda y: mu * net_callable(y), in_axes=(0,))(x)
            # return jax.vmap(lambda y: mu * jnp.real(net_callable(y)), in_axes=(0,))(x)
        return jax.vmap(lambda y: mu * (net_callable(net_params, y)), in_axes=(0,))(x)
        # return jax.vmap(lambda y: mu * jnp.real(net_callable(net_params, y)), in_axes=(0,))(x)
    
    def _logprob_np(self, x, mu, net_callable, net_params = None):
        '''
        Calculate the log probability of a state using NumPy.
        Parameters:
            - x             : The state
            - mu            : The parameter mu
            - net_callable  : The network callable (returns (\\log\\psi(s)))
            - net_params    : The network parameters - may be None but for flax callables
                            the parameters are necessary strictly!
        Returns:
            - The log probability as a float or complex number
        '''
        if net_params is None:
            # If no parameters are needed, call net_callable with just y
            return np.array([mu * net_callable(y) for y in x]).reshape(x.shape[0])
            # return np.array([mu * np.real(net_callable(y)) for y in x])
        return np.array([mu * net_callable(net_params, y) for y in x]).reshape(x.shape[0])
        # return np.array([mu * np.real(net_callable(net_params, y)) for y in x])
    
    def logprob(self, x, mu: float = 1.0, net_callable = None, net_params = None):
        '''
        Calculate the log probability of a state. This is just the function
        that overlaps the call to the network callable
        
        A network callable is needed. (it should mimic the probability
        calculation for a given state x - either in many body state or for a classical 
        configuration.
        Parameters:
            - x             : The state - either a many body state or a classical configuration
            - mu            : The parameter mu
            - net_callable  : The network callable (returns \\text{Re}(\\log\\psi(s)))
            - net_params    : The network parameters
        Returns:
            - The log probability as a float
        '''
        if net_callable is None:
            net_callable = self._net_callable
        if net_params is None:
            net_params = self._parameters
            
        if self._isjax:
            return self._logprob_jax(x, mu, net_callable, net_params)
        return self._logprob_np(x, mu, net_callable, net_params)

    ###################################################################
    #! UPDATE CHAIN
    ###################################################################
    
    def _sweep_chain_jax(self,
                        chain               : 'array-like',
                        logprobas           : 'array-like',
                        rng_k               : 'array-like',
                        num_proposed        : 'array-like',
                        num_accepted        : 'array-like',
                        params              : 'array-like',
                        update_proposer     : Callable,
                        log_proba_fun       : Callable,
                        accept_config_fun   : Callable,
                        net_callable_fun    : Callable,
                        steps               : int):
        '''
        Carry out the update chain using JAX. It uses JAX's fori_loop
        to iterate over the number of steps.
        Parameters
            - chain         : The initial state of the chain (array-like)
            - steps         : The number of steps to update the chain (jax.lax.fori_loop).
                            They are necessary to be given in the function.
            - logprobas     : The current log probabilities of the chain (array-like)
                            They are necessary to be given in the function as they decide on the 
                            acceptance probability. They are modified in the loop.
            - rng_k         : The random key (array-like)
            - num_proposed  : The number of proposed updates (array-like)
            - num_accepted  : The number of accepted updates (array-like)
            - params        : The network parameters (array-like) - if implemented for flax
            - update_proposer: The function that proposes a new state. Signature should be:
                            new_state = update_proposer(key, state, update_proposer_arg)
            - log_probability: The function to compute the log probability; signature:
                            new_logprob = log_probability(new_state, net_callable=..., net_params=params)
                            
        Returns
            - The updated chain after the specified number of steps.
        '''
        
        # get the current value of the chain
        # current_val = logprobas if logprobas is not None else net_callable(params, chain)
        current_val = logprobas if logprobas is not None else log_proba_fun(chain,
                                                                            net_callable=net_callable_fun, net_params=params)
        
        # setup the carry for the fori_loop
        carry       = (chain, current_val, rng_k, num_proposed, num_accepted)
        
        # define the body of the fori_loop
        def body(_, carry):
            '''
            Carry:
            - 0 : The current state of the chain
            - 1 : The current value of the chain
            - 2 : The random key
            - 3 : The number of proposed updates
            - 4 : The number of accepted updates
            '''
            
            # obtain the current key (0. current state of the chain, 1. current value of the chain,
            # 2. the random key, 3. the number of proposed updates, 4. the number of accepted updates
            chain_in, current_val_in, rng_k_in, num_proposed_in, num_accepted_in = carry

            # split the random key to get a new key for each chain element
            new_rng_ks      = random_jp.split(rng_k_in, num = chain_in.shape[0] + 1)
            carry_key       = new_rng_ks[-1]

            # update the chain by proposing a new state via the update_proposer
            new_chain       = jax.vmap(update_proposer, in_axes=(0, 0))(
                                    chain_in, new_rng_ks[:-1])

            # compute the acceptance probability (it is already partially called on mu and beta)
            logprobas_new   = log_proba_fun(new_chain, net_callable=net_callable_fun, net_params=params)
            
            # acceptance probability (already partially called on mu and beta)
            acc_probability = accept_config_fun(current_val_in, logprobas_new)

            # decide with dice rule
            new_rng_k, carry_key    = random_jp.split(carry_key,)
            accepted                = random_jp.bernoulli(new_rng_k, acc_probability)
            
            # Update the chain: if accepted, use new state; else keep old.
            # Use jnp.where to avoid an extra vmap.
            new_carry_states = jnp.where(accepted, new_chain, chain_in)
            new_carry_vals   = jnp.where(accepted, logprobas_new, current_val_in)
            
            # Update counters.
            num_proposed = num_proposed_in + chain_in.shape[0]
            num_accepted = num_accepted_in + jnp.sum(accepted)
            return (new_carry_states, new_carry_vals, carry_key, num_proposed, num_accepted)
        return jax.lax.fori_loop(0, steps, body, carry)

    def _sweep_chain_np(self,
                        chain               : np.ndarray,
                        logprobas           : np.ndarray,
                        num_proposed        : np.ndarray,
                        num_accepted        : np.ndarray,
                        params              : Any,
                        update_proposer     : Callable,
                        log_proba_fun       : Callable,
                        accept_config_fun   : Callable,
                        net_callable_fun    : Callable,
                        steps               : int):
        """
        NumPy version of sweeping a single chain.
        
        This function carries a tuple:
        (chain, current_logprob, rng_key, num_proposed, num_accepted)
        through a loop over a number of steps. For each step, it:
        - Proposes new candidate states using update_proposer (applied to each chain element)
        - Computes new log-probabilities via log_probability
        - Computes the acceptance probability using accept_config
        - Generates a uniform random number for each chain element to decide acceptance
        - Updates the chain and the carried current log-probability accordingly
        - Updates counters for the total proposals and acceptances.
        
        Parameters:
        chain           : Current state of the chain (NumPy array, shape (nChains, ...))
        logprobas       : Current log–probabilities for each chain element (1D NumPy array)
        rng_k           : (Not really used in the NumPy version; can be updated with a new seed)
        num_proposed    : Total number of proposals made so far (integer)
        num_accepted    : Total number of accepted proposals so far (integer)
        params          : Network parameters (passed to log_probability)
        update_proposer : Function that proposes a new state. Signature should be: new_state = update_proposer(key, state, update_proposer_arg)
        log_probability : Function to compute the log–probability; signature: new_logprob = log_probability(new_state, net_callable=..., net_params=params)
        accept_config   : Function to compute the acceptance probability from current and candidate log–probabilities.
        net_callable    : The network callable (e.g. returns Re(logψ(s)))
        steps           : Number of update steps to perform.
        
        Returns:
        A tuple (chain, current_logprobas, rng_k, num_proposed, num_accepted)
        """
        
        current_val     = logprobas if logprobas is not None else log_proba_fun(chain, net_callable=net_callable_fun, net_params=params)

        def loop_sweep(chain, current_val, num_proposed, num_accepted, tmpstates, rng):
            # Loop through the number of steps
            for _ in range(steps):
                
                # use the temporary states for the proposal 
                tmpstates               = chain.copy()
                
                # For each chain element, generate a new candidate using the update proposer.
                # Here, we simulate splitting rng_k by simply generating a new random integer key.
                # also, this flips the state inplace, so we need to copy the chain
                # and then update the chain with the new state.
                # Use NumPy's vectorized operations to generate new states
                new_chain               = update_proposer(tmpstates, rng)
                
                # Compute new log probabilities for candidates using the provided log probability function.
                new_logprobas           = log_proba_fun(new_chain, net_callable=net_callable_fun, net_params=params)
                
                # Compute acceptance probability for each chain element.
                acc_probability         = accept_config_fun(current_val, new_logprobas)
                
                # Decide acceptance by comparing against a random uniform sample.
                rand_vals               = self._rng.random(size=chain.shape[0] if len(chain.shape) > 1 else 1)
                accepted                = rand_vals < acc_probability
                num_proposed            += chain.shape[0] if len(chain.shape) > 1 else 1
                num_accepted[accepted]  += 1
                
                # Update: if accepted, take candidate, else keep old.
                new_val                 = np.where(accepted, new_logprobas, current_val)
                # Update the carry:
                # chain                   = np.array([new_chain[i] if accepted[i] else chain[i] for i in range(chain.shape[0])])
                chain                   = np.where(accepted[:, None], new_chain, chain)
                current_val             = new_val
                
                # Update rng_k with a new random integer (not used further)
            return chain, current_val, num_proposed, num_accepted, tmpstates
        
        # go through the number of steps, for each step: 
        # 1. propose a new state
        # 2. compute the log probability
        # 3. compute the acceptance probability
        # 4. decide with dice rule
        # 5. update the chain
        # 6. update the carry
        # 7. update rng_k with a new random integer (not used further)
        chain, current_val, num_proposed, num_accepted, self._tmpstates = loop_sweep(chain, 
                                                                    current_val, num_proposed, 
                                                                num_accepted, self._tmpstates, self._rng)
        return chain, current_val, num_proposed, num_accepted

    def _sweep_chain(self,
                    chain               : 'array-like',
                    logprobas           : 'array-like',
                    rng_k               : 'array-like',
                    num_proposed        : 'array-like',
                    num_accepted        : 'array-like',
                    params              : 'array-like',
                    update_proposer     : Callable,
                    log_proba_fun       : Callable,
                    accept_config_fun   : Callable,
                    net_callable_fun    : Callable,
                    steps               : int):
        '''
        Sweep the chain for a given number of steps.
        Parameters:
            - chain : The initial state of the chain (array-like)
            - steps : The number of steps to update the chain (jax.lax.fori_loop)
            - logprobas : The current log probabilities of the chain (array-like)
            - rng_k : The random key (array-like)
            - num_proposed : The number of proposed updates (array-like)
            - num_accepted : The number of accepted updates (array-like)
            - params : The network parameters (array-like) - if implemented for flax
            - update_proposer : The function that proposes a new state. Signature should be:
                            new_state = update_proposer(key, state, update_proposer_arg)
            - log_probability : The function to compute the log probability; signature:
                            new_logprob = log_probability(new_state, net_callable=..., net_params=params)
            - accept_config : The function to compute the acceptance probability from current and candidate log–probabilities.
            - net_callable : The network callable (e.g. returns Re(logψ(s)))

        Returns:
            - The updated chain after the specified number of steps.
        '''
        if logprobas is None:
            logprobas = self._logprobas
            if logprobas is None:
                logprobas = self.logprob(chain, net_callable=net_callable_fun, net_params=params)

        # check if the log_probability function is set, otherwsie use self function
        if log_proba_fun is None:
            log_proba_fun = self.logprob
            
        # check if the acceptance function is set, otherwise use self function
        if accept_config_fun is None:
            accept_config_fun = self.acceptance_probability
        
        # check if the net callable function is set, otherwise use self function
        if net_callable_fun is None:
            net_callable_fun = self._net_callable
            
        # check if the update_proposer function is set, otherwise use self function
        if update_proposer is None:
            update_proposer = self._upd_fun
        
        if self._isjax:
            return self._sweep_chain_jax(chain      =   chain,
                                        logprobas   =   logprobas,
                                        rng_k=rng_k, num_proposed=num_proposed, num_accepted=num_accepted,
                                        params=params, update_proposer=update_proposer, log_proba_fun=log_proba_fun,
                                        accept_config_fun=accept_config_fun, net_callable_fun=net_callable_fun, steps=steps)
        return self._sweep_chain_np(chain, logprobas, 
                    num_proposed, num_accepted,
                    params, update_proposer, log_proba_fun, accept_config_fun, net_callable_fun, steps)
    
    ###################################################################
    #! SAMPLING
    ###################################################################
    
    def _get_samples_jax(self,
        shape               : Tuple[int, ...],
        params              : 'array-like',
        num_samples         : int,
        therm_steps         : int,
        sweep_steps         : int,
        states              : 'array-like',
        logprobas           : 'array-like',
        net_callable        : Callable,
        update_proposer     : Callable,
        log_proba_fun       : Callable,
        accept_config       : Callable,
        rng_k               : 'array-like',
        num_proposed        : 'array-like',
        num_accepted        : 'array-like'):
        
        # thermalize the chains
        states, logprobas, rng_k, num_proposed, num_accepted = self._sweep_chain_jax(
            chain			    = states,
            logprobas		    = logprobas,
            rng_k			    = rng_k,
            num_proposed	    = num_proposed, 
            num_accepted	    = num_accepted,
            params			    = params,
            update_proposer	    = update_proposer,
            log_proba_fun	    = log_proba_fun,
            accept_config_fun	= accept_config,
            net_callable_fun	= net_callable,
            steps			    = therm_steps*sweep_steps)
        
        def scan_fun(carry, i):
            states, logprobas, rng_k, num_proposed, num_accepted = carry
            states, logprobas, rng_k, num_proposed, num_accepted = self._sweep_chain_jax(
                chain               = states,
                logprobas           = logprobas,
                rng_k               = rng_k,
                num_proposed        = num_proposed,
                num_accepted        = num_accepted,
                params              = params,
                update_proposer     = update_proposer,
                log_proba_fun       = log_proba_fun,
                accept_config_fun   = accept_config,
                net_callable_fun    = net_callable,
                steps               = sweep_steps)
            return (states, logprobas, rng_k, num_proposed, num_accepted), states

        meta, configs = jax.lax.scan(scan_fun, 
                    (states, logprobas, rng_k, num_proposed, num_accepted), None, length=num_samples)
        return meta, configs.reshape((num_samples, -1) + shape)
    
    def _get_samples_np(self,
                        params,
                        num_samples,
                        multiple_of = 1):
        """
        NumPy version of obtaining samples via MCMC.
        
        Parameters:
        - params         : Network parameters.
        - num_samples    : Number of sweeps (sample collection iterations) to perform.
        - multiple_of    : (Not used here, but could be used for device distribution.)
        
        Returns:
        A tuple (meta, configs) where:
            meta is a tuple of the final (states, logprobas, rng_k, num_proposed, num_accepted)
            configs is an array of shape (numSamples, -1) + self._shape containing the sampled configurations.
        """
        
        # Thermalize the chains by sweeping for (therm_steps * sweep_steps)
        states, logprobas, num_proposed, num_accepted = self._sweep_chain_np(
            chain               = self._states,
            logprobas           = self._logprobas,
            num_proposed        = self._num_proposed,
            num_accepted        = self._num_accepted,
            params              = params,
            update_proposer     = self._upd_fun,
            log_proba_fun       = self.logprob,
            accept_config_fun   = self.acceptance_probability,
            net_callable_fun    = self._net_callable,
            steps               = self._therm_steps * self._sweep_steps)
        
        # Now perform numSamples sweeps, collecting the resulting states
        meta            = []
        configs_list    = []
        
        # Perform the sampling sweeps - do the same as the thermalization but save the states
        for _ in range(num_samples):
            states, logprobas, num_proposed, num_accepted = self._sweep_chain_np(
                chain               = states,
                logprobas           = logprobas,
                num_proposed        = num_proposed,
                num_accepted        = num_accepted,
                params              = params,
                update_proposer     = self._upd_fun,
                log_proba_fun       = self.logprob,
                accept_config_fun   = self.acceptance_probability,
                net_callable_fun    = self._net_callable,
                steps               = self._sweep_steps)
            meta        = (states.copy(), logprobas.copy(), num_proposed, num_accepted)
            # Assume states is of shape (num_chains, *self._shape)
            configs_list.append(states.copy())
        
        # Concatenate configurations along the chain axis
        # Then reshape to (num_samples, -1) + self._shape (flattening the chain dimension per sample)
        configs = np.concatenate(configs_list, axis=0)
        return meta, configs.reshape((num_samples, -1) + self._shape)
    
    def sample(self, parameters=None, num_samples=None, num_chains=None):
        '''
        Sample the states from the Hilbert space according to the Born distribution.
        Parameters:
            - parameters : The parameters for the network
            - num_samples : The number of samples to generate
            - num_chains  : The number of chains to use
        Returns:
            - The sampled states
        '''
        
        # check the number of samples and chains
        if num_samples is None:
            num_samples = self._numsamples
        else:
            self.set_numsamples(num_samples)
            
        if num_chains is None:
            num_chains = self._numchains
        else:
            self.set_numchains(num_chains)
        
        # check the parameters - if not given, use the current parameters
        if parameters is None:
            if hasattr(self._net, 'get_params'):
                parameters = self._net.get_params()
            else:
                parameters = self._parameters
        else:
            if hasattr(self._net, 'set_params'):
                parameters = self._net.get_params()
        
        net_callable, parameters= self._set_net_callable(self._net)
        
        # set the log probabilities directly for the states that are already set
        self._logprobas         = self.logprob(self._states,
                                                mu              =   self._mu,
                                                net_callable    =   net_callable,
                                                net_params      =   parameters)
        
        if self._isjax:
            (self._states, self._logprobas, self._rng_k, self._num_proposed, self._num_accepted), configs =\
                self._get_samples_jax(
                    shape           =   self._shape,
                    params          =   parameters,
                    num_samples     =   num_samples,
                    therm_steps     =   self._therm_steps,
                    sweep_steps     =   self._sweep_steps,
                    states          =   self._states,
                    logprobas       =   self._logprobas,
                    net_callable    =   net_callable,
                    update_proposer =   self._upd_fun, log_proba_fun=self.logprob,
                    accept_config   =   self.acceptance_probability,
                    rng_k           =   self._rng_k,
                    num_proposed    =   self._num_proposed,
                    num_accepted    =   self._num_accepted)
            
            configs_log_ansatz  = jax.vmap(net_callable, in_axes=(None, 0))(parameters, configs)
            probs               = jnp.exp((1.0 / self._logprob_fact - self._mu) * jnp.real(configs_log_ansatz))
            norm                = jnp.sum(probs, axis=0, keepdims=True)
            probs               = probs / norm * self._numchains
            
            # flatten the configs to be of shape (num_samples * num_chains)
            configs             = configs.reshape(-1, *configs.shape[2:])
            configs_log_ansatz  = configs_log_ansatz.reshape(-1, *configs_log_ansatz.shape[2:])
            probs               = probs.reshape(-1, *probs.shape[2:])

            return (self._states, self._logprobas), (configs, configs_log_ansatz), probs
        
        # for numpy
        (self._states, self._logprobas, self._num_proposed, self._num_accepted), configs =\
            self._get_samples_np(parameters, num_samples, num_chains)
        configs_log_ansatz  = np.array([self._net(parameters, config) for config in configs])
        probs               = np.exp((1.0 / self._logprob_fact - self._mu) * np.real(configs_log_ansatz))
        norm                = np.sum(probs, axis=0, keepdims=True)
        probs               = probs / norm * self._numchains
        
        # flatten the configs to be of shape (num_samples * num_chains)
        configs             = configs.reshape(-1, *configs.shape[2:])
        configs_log_ansatz  = configs_log_ansatz.reshape(-1, *configs_log_ansatz.shape[2:])
        probs               = probs.reshape(-1, *probs.shape[2:])
        
        return (self._states, self._logprobas), (configs, configs_log_ansatz), probs
    
    ###################################################################
    #! SETTERS
    ###################################################################
    
    def set_mu(self, mu):
        '''
        Set the parameter mu.
        Parameters:
            - mu : The parameter mu
        '''
        self._mu = mu
        
    def set_beta(self, beta):
        '''
        Set the parameter beta.
        Parameters:
            - beta : The parameter beta
        '''
        self._beta = beta
        
    def set_therm_steps(self, therm_steps):
        '''
        Set the number of thermalization steps.
        Parameters:
            - therm_steps : The number of thermalization steps
        '''
        self._therm_steps = therm_steps

    ###################################################################
    #! GETTERS
    ###################################################################
    
    def get_mu(self):
        '''
        Get the parameter mu.
        '''
        return self._mu
    
    def get_beta(self):
        '''
        Get the parameter beta.
        '''
        return self._beta
    
    def get_therm_steps(self):
        '''
        Get the number of thermalization steps.
        '''
        return self._therm_steps
    
    def get_sweep_steps(self):
        '''
        Get the number of sweep steps.
        '''
        return self._sweep_steps

#######################################################################

@unique
class SamplerType(Enum):
    """
    Enum class for the sampler types.
    """
    MCSampler = auto()
    
    @staticmethod
    def from_str(s: str) -> 'SamplerType':
        """
        Convert a string to a SamplerType enum.
        """
        try:
            return SamplerType[s]
        except KeyError:
            raise ValueError(f"Invalid SamplerType: {s}") from None
    
#######################################################################
        
def get_sampler(typek: Union[str, SamplerType], *args, **kwargs) -> Sampler:
    """
    Get a sampler of the given type.
    
    Parameters:
    - typek (str or SamplerType): The type of sampler to get
    - args: Additional arguments for the sampler
    - kwargs: Additional keyword arguments for the sampler
    
    Returns:
    - Sampler: The requested sampler
    
    Raises:
    - ValueError: If the requested sampler type is not implemented
    """
    if isinstance(typek, str):                  # is a string to convert to enum
        typek = SamplerType.from_str(typek)
    elif typek == MCSampler:                    # is a sampler type from the enum
        typek = SamplerType.MCSampler
    elif isinstance(typek, Sampler):            # is already a sampler
        return typek
    else:
        raise ValueError(SamplerErrors.NOT_A_VALID_SAMPLER_TYPE)
    
    # set from the type enum
    if typek == SamplerType.MCSampler:
        return MCSampler(*args, **kwargs)
    
    raise ValueError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

#######################################################################