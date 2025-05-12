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

import numpy as np
import numba
import scipy as sp
from numba import jit, njit, prange
from typing import Union, Tuple, Callable, Optional, Any
from functools import partial

# flax for the network
from flax import linen as nn

# for the abstract class
from abc import ABC, abstractmethod
from enum import Enum, auto, unique

# from algebra
from general_python.algebra.utils import JAX_AVAILABLE, get_backend, DEFAULT_JP_INT_TYPE, DEFAULT_BACKEND_KEY
from general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE
from general_python.algebra.ran_wrapper import choice, randint, uniform, randint_np, randint_jax
from general_python.common.directories import Directories
import general_python.common.binary as Binary

# from hilbert
from Algebra.hilbert import HilbertSpace

#! JAX imports
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.random as random_jp
else:
    jax         = None
    import numpy as jnp
    random_jp   = None

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

if JAX_AVAILABLE:
    @jax.jit
    def _propose_random_flip_jax(state: jnp.ndarray, rng_k):
        r'''Propose `num` random flips of a state using JAX.

        Parameters:
            state (jnp.ndarray):
                The state array (or batch of states).
            rng_k (jax.random.PRNGKey):
                The random key for JAX.

        Returns:
            jnp.ndarray: The proposed flipped state(s).
        '''
        idx = randint_jax(key=rng_k, shape=(1,), low=0, high=state.size)[0]
        return Binary.jaxpy.flip_array_jax_spin(state, idx)

    @partial(jax.jit, static_argnums=(2,))
    def _propose_random_flips_jax(state: jnp.ndarray, rng_k, num = 1):
        r'''Propose `num` random flips of a state using JAX.

        Parameters:
            state (jnp.ndarray):
                The state array (or batch of states).
            rng_k (jax.random.PRNGKey):
                The random key for JAX.
            num (int):
                The number of flips to propose per state. Must be static for JIT.

        Returns:
            jnp.ndarray: The proposed flipped state(s).
        '''
        if state.ndim == 1:
            idx = randint_jax(key=rng_k, shape=(num,), low=0, high=state.size, dtype=DEFAULT_JP_INT_TYPE)
            return Binary.jaxpy.flip_array_jax_multi(state, idx, spin=Binary.BACKEND_DEF_SPIN)
        else:
            batch_size  = state.shape[0]
            state_size  = state.shape[1]
            # Generate indices for each state in the batch
            keys        = random_jp.split(rng_k, batch_size)
            # Use vmap to apply index generation and flipping per state
            def flip_single_state(single_state, key):
                idx     = randint_jax(key=key, shape=(num,), low=0, high=state_size, dtype=DEFAULT_JP_INT_TYPE)
                return Binary.jaxpy.flip_array_jax_multi(single_state, idx, spin=Binary.BACKEND_DEF_SPIN)

            return jax.vmap(flip_single_state)(state, keys)

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
    
    if hilbert is None:
        if modes == 2:
            # set the state from tensor
            out = Binary.int2base(state, size, backend, 
                                spin_value=mode_repr, spin=Binary.BACKEND_DEF_SPIN).reshape(shape)
        elif modes == 4:
            up_int          = state & ((1 << size) - 1)
            down_int        = state >> size
            up_array        = np.array([1 if (up_int & (1 << i)) else 0 for i in range(size)],
                                dtype=DEFAULT_NP_FLOAT_TYPE)
            down_array      = np.array([1 if (down_int & (1 << i)) else 0 for i in range(size)],
                                dtype=DEFAULT_NP_FLOAT_TYPE)
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
                    rng_k       = None
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
    - rng_k         : random key for JAX
    
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
    if hilbert is None:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                ran_state = choice([-1, 1], shape, rng=rng, rng_k=rng_k, backend=backend)
            else:
                ran_state = choice([0, 1], shape, rng=rng, rng_k=rng_k, backend=backend)
        elif modes == 4:
            ran_state = choice([0, 1], 2 * size, rng=rng, rng_k=rng_k, backend=backend)
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    return ran_state.astype(DEFAULT_NP_FLOAT_TYPE).reshape(shape) * mode_repr

def _set_state_up(modes         : int                           = 2,
                hilbert         : Optional[HilbertSpace]        = None,
                shape           : Union[int, Tuple[int, ...]]   = (1,),
                mode_repr       : float                         = 0.5,
                backend         : str                           = 'default'
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
                    rng_k       = None,
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
    - rng_k         : random key for JAX.
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
            return _state_distinguish(state_enum, modes, hilbert, shape, mode_repr, rng, rng_k, backend)
        except ValueError as e:
            raise ValueError(SamplerErrors.NOT_A_VALID_STATE_STRING) from e
    elif isinstance(statetype, SolverInitState):
        if statetype == SolverInitState.RND:
            return _set_state_rand(modes, hilbert, shape, mode_repr, backend, rng, rng_k)
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
            shape (Tuple[int, ...]):
                Shape of the system (e.g., lattice dimensions).
            upd_fun (Callable):
                Update function for proposing new states. Signature: `new_state = upd_fun(state, rng/rng_k, **kwargs)`.
                If None, defaults to single random spin flip.
            rng (np.random.Generator):
                NumPy random number generator (used if backend is NumPy or seed is provided).
            rng_k (Optional[jax.random.PRNGKey]):
                JAX random key (used if backend is JAX).
            seed (Optional[int]):
                Seed for initializing random number generators if `rng` and `rng_k` are not provided.
            hilbert (Optional[HilbertSpace]):
                Hilbert space instance (currently not fully utilized in provided code, but kept for structure).
            numsamples (int):
                Number of samples to generate per chain *after* thermalization.
            numchains (int):
                Number of parallel Markov chains.
            initstate ('array-like' or str or SolverInitState or int):
                Initial state configuration specification. Can be an array, an integer state index,
                a predefined string ('RND', 'F_UP', 'F_DN', 'AF'), or a SolverInitState enum. Defaults to 'RND'.
            backend (str):
                Computational backend ('numpy', 'jax', or 'default'). 'default' chooses JAX if available.
            **kwargs:
                Additional arguments passed to `_state_distinguish` for state generation (e.g., `modes`, `mode_repr`).

        Raises:
            SamplerErrors: If backend/initial state is invalid or RNG setup fails.

        Attributes:
            _shape (Tuple[int, ...])            : Shape of the system.
            _size (int)                         : Total number of sites/spins.
            _hilbert (Optional[HilbertSpace])   : Hilbert space instance.
            _rng, _rng_k                        : Random number generators.
            _backend                            : Computational backend module (np or jnp).
            _isjax (bool)                       : True if using JAX backend.
            _backendstr (str)                   : Name of the backend ('np' or 'jax').
            _numsamples (int)                   : Number of samples per chain.
            _numchains (int)                    : Number of chains.
            _initstate                          : Initial state configuration (single state).
            _states                             : Current states of all chains (shape: [numchains, *shape]).
            _tmpstates (np.ndarray)             : Temporary storage for NumPy proposals (if needed).
            _num_proposed, _num_accepted        : Counters for proposed and accepted moves per chain.
            _upd_fun                            : The actual update function used.
        """
        if isinstance(backend, str):
            self._backendstr = backend
            
        if rng is not None:
            self._rng       = rng
            self._rng_k     = rng_k if rng_k is not None else (DEFAULT_BACKEND_KEY if JAX_AVAILABLE else None)
            self._backend   = get_backend(backend)
        elif seed is not None:
            self._backend, _, (self._rng, self._rng_k) = self.obtain_backend(backend, seed)
        else:
            raise SamplerErrors(SamplerErrors.NOT_HAVING_RNG)
        
        is_valid_rng_k = (
            self._rng_k is not None and             
            isinstance(self._rng_k, jax.Array) and
            self._rng_k.shape == (2,) and           
            self._rng_k.dtype == jnp.uint32         
        )
        if not is_valid_rng_k:
            key_info = f"Value: {self._rng_k}, Type: {type(self._rng_k)}"
            if isinstance(self._rng_k, jax.Array):
                key_info += f", Shape: {self._rng_k.shape}, Dtype: {self._rng_k.dtype}"
            raise TypeError(f"Sampler's RNG key (self._rng_k) is not a valid JAX PRNGKey. {key_info}")
    
        # check JAX
        self._isjax         = (not self._backend == np)
        if self._isjax and self._rng_k is None:
            raise SamplerErrors(SamplerErrors.NOT_HAVING_RNG + " (JAX requires rng_k)")
        if not self._isjax and self._rng is None:
            raise SamplerErrors(SamplerErrors.NOT_HAVING_RNG + " (NumPy requires rng)")
        
        # set the backend
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
        self._modes         = kwargs.get('modes', 2)
        self._mode_repr     = kwargs.get('mode_repr', Binary.BACKEND_REPR)
        state_kwargs        = {
            'modes'     : self._modes,
            'mode_repr' : self._mode_repr
        }
        self._initstate_t   = initstate
        self.set_initstate(self._initstate_t, **state_kwargs)
        
        # proposed state
        int_dtype           = DEFAULT_JP_INT_TYPE if self._isjax else DEFAULT_NP_INT_TYPE
        self._num_proposed  = self._backend.zeros(numchains, dtype=int_dtype)
        self._num_accepted  = self._backend.zeros(numchains, dtype=int_dtype)
        
        # handle the update function
        self._upd_fun       = upd_fun
        if self._upd_fun is None:
            if self._isjax:
                # Bind RNG arguments to the JAX updater and then wrap with JIT.
                self._upd_fun = _propose_random_flip_jax
            else:
                # Use the Numba version (potentially parallelized)
                # Note:
                #   The parallel Numba version needs careful RNG handling if numchains > 1
                self._upd_fun = partial(_propose_random_flips_np, rng=self._rng, num=1)
    
    ###################################################################
    #! ABSTRACT
    ###################################################################
    
    @abstractmethod
    def sample(self, parameters=None, num_samples=None, num_chains=None):
        ''' Tries to sample the state from the Hilbert space. '''
        pass
    
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
        - Tuple         : backend, backend_sp, rng, rng_k 
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
        _backendstr = 'np' if (backend is None or (backend == 'default' and not JAX_AVAILABLE) or backend == np) else 'jax'
        return Sampler.obtain_backend(_backendstr, seed)
    
    ###################################################################
    #! PROPERTIES
    ###################################################################
    
    @property
    def hilbert(self): return self._hilbert
    @property
    def numsamples(self): return self._numsamples
    @property
    def numchains(self): return self._numchains
    @property
    def shape(self): return self._shape
    @property
    def size(self): return self._size
    @property
    def initstate(self): return self._initstate
    @property
    def upd_fun(self): return self._upd_fun
    @property
    def states(self): return self._states
    @property
    def rng(self): return self._rng
    @property
    def rng_k(self): return self._rng_k
    @property
    def backend(self): return self._backend
    @property
    def proposed(self): return self._num_proposed
    @property
    def num_proposed(self): return self._backend.sum(self._num_proposed)
    @property
    def accepted(self): return self._num_accepted
    @property
    def num_accepted(self): return self._backend.sum(self._num_accepted)
    @property
    def rejected(self): return self.proposed - self.accepted
    @property
    def num_rejected(self): return self.num_proposed - self.num_accepted
    @property
    def accepted_ratio(self):
        num_prop = self.num_proposed
        return self.num_accepted / num_prop if num_prop > 0 else self._backend.array(0.0)
    @property
    def isjax(self): return self._isjax
    
    ###################################################################
    #! SETTERS
    ###################################################################
    
    def reset(self):
        """
        Reset the sampler to its initial state.
        """
        int_dtype           = DEFAULT_JP_INT_TYPE if self._isjax else DEFAULT_NP_INT_TYPE
        self._num_proposed  = self._backend.zeros(self._numchains, dtype=int_dtype)
        self._num_accepted  = self._backend.zeros(self._numchains, dtype=int_dtype)
        state_kwargs        = {
            'modes'     : self._modes,
            'mode_repr' : self._mode_repr
        }
        self.set_initstate(self._initstate_t, **state_kwargs)
    
    # ---
    
    def set_initstate(self, initstate, **kwargs):
        """
        Set the initial state template of the system.

        Parameters:
            initstate (str, int, np.ndarray, jnp.ndarray, SolverInitState):
                The initial state specification. If None, defaults to 'RND'.
            **kwargs:
                Additional arguments passed to `_state_distinguish` (e.g., `modes`, `mode_repr`).

        Raises:
            SamplerErrors.NOT_A_VALID_STATE_DISTING:
                If the state type is invalid.
            ValueError:
                If underlying state generation fails.
        """

        if initstate is None:
            initstate = SolverInitState.RND 
        
        # handle the initial state
        if initstate is None or isinstance(initstate, (str, SolverInitState)):
            try:
                current_bcknd_str = 'jax' if self._isjax else 'numpy'
                if self._hilbert is None or True:
                    self._initstate = _state_distinguish(initstate,
                                        modes   =   kwargs.get('modes', 2),
                                        hilbert =   self._hilbert,
                                        shape   =   self._shape,
                                        backend =   current_bcknd_str,
                                        rng     =   self._rng,
                                        rng_k   =   self._rng_k)
                else:
                    raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
            except Exception as e:
                raise ValueError(f"Failed to set initial state: {e}") from e
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
        Set the chains for the sampler, replicating the initstate.

        Parameters:
            initstate (np.ndarray or jnp.ndarray):
                The single initial state configuration to replicate.
            numchains (int, optional):
                The number of chains. If None, uses the sampler's current `_numchains`.
        '''
        if numchains is None:
            numchains = self._numchains
        else:
            self._numchains = numchains

        # Ensure initstate is the correct type for the backend before stacking
        if self._isjax:
            self._states = jnp.stack([jnp.array(initstate)] * numchains, axis=0)
        else:
            self._states = np.stack([np.array(initstate.copy())] * numchains, axis=0)
    
    # ---
    
    def set_numsamples(self, numsamples):
        '''
        Set the number of samples.
        Parameters:
            numsamples (int):
                The number of samples
        '''
        self._numsamples = numsamples
        
    def set_numchains(self, numchains):
        '''
        Set the number of chains.
        Parameters:
            numchains (int):
                The number of chains
        '''
        self._numchains = numchains
        self.set_chains(self._initstate, numchains)
    
    ###################################################################
    #! GETTERS
    ###################################################################
    
    @abstractmethod
    def get_sampler_jax(self, num_samples=None, num_chains=None):
        """
        Get the JAX sampler instance.
        """
        pass
    
    @abstractmethod
    def get_sampler_np(self, num_samples=None, num_chains=None):
        """
        Get the NumPy sampler instance.
        """
        pass
    
#######################################################################    

class MCSampler(Sampler):
    r"""
    Markov Chain Monte Carlo sampler for quantum states.

    Implements MCMC sampling from the Hilbert space based on a probability distribution
    derived from the quantum state amplitudes :math:`\psi(s)`. The target distribution is typically
    proportional to :
    
        math:`|\psi(s)|^{\mu}`, where :math:`\mu` controls the sampling bias.

    The standard Born rule distribution
        
        :math:`p(s) = |\psi(s)|^2 / \sum_{s'} |\psi(s')|^2`
    
    
    corresponds to
    
        :math:`\mu=2`. Values :math:`0 \le \mu < 2`
        
    can be used for importance sampling techniques (see [arXiv:2108.08631](https://arxiv.org/abs/2108.08631)).
    Note:
        Is there an error in the paper, where they introduce \mu?

    The sampling process uses the Metropolis-Hastings algorithm:
        1. **Initialization:**
            Set up with system shape, network/callable for
                :math:`\log\psi(s)`,
            proposal function, MCMC parameters 
                (:math:`\mu`, :math:`\beta`, steps).
        2. **Network Callable:**
            Obtain the function to compute :math:`\log\psi(s)` and its parameters.
        3. **Thermalization:** 
            Run MCMC chains for `therm_steps * sweep_steps` updates to reach equilibrium.
        4. **Sampling:**
            Run MCMC chains for `num_samples * sweep_steps` updates, collecting configurations.
        5. **Output:**
            Return final chain states, collected samples, log-ansatz values, and importance weights.

    Supports JAX backend for performance via JIT compilation of core loops.
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
        r"""
        Initialize the MCSampler.

        Parameters:
            net (Callable or Flax Module):
                The network or function providing `log_psi(s)`. If a Flax module, assumes `net.apply` exists.
                If a callable, should have signature `log_psi = net(params, state)`.
            shape (Tuple[int, ...]):
                Shape of the system configuration (e.g., lattice dimensions).
            rng (np.random.Generator):
                NumPy random number generator.
            rng_k (Optional[jax.random.PRNGKey]):
                JAX random key.
            upd_fun (Optional[Callable]):
                State update proposal function. Defaults to single spin flip.
                Signature: `new_state = upd_fun(state, rng/rng_k)`
            mu (float):
                Exponent :math:`\mu` for the sampling distribution :math:`p(s) \propto |\psi(s)|^{\mu}`.
                Must be in range [0, 2]. Default is 2 (Born rule).
            beta (float):
                Inverse temperature factor :math:`\beta` for the Metropolis acceptance probability. Default is 1.0.
            therm_steps (int):
                Number of thermalization sweeps. Each sweep consists of `sweep_steps` MCMC updates per site (on average).
            sweep_steps (int):
                Number of MCMC updates per site within a single "sweep". Determines correlation between samples. Default is 1.
            seed (Optional[int]):
                Random seed for initialization if `rng`/`rng_k` not provided.
            hilbert (Optional[HilbertSpace]):
                Hilbert space object (optional).
            numsamples (int):
                Number of samples to collect per chain *after* thermalization.
            numchains (int):
                Number of parallel Markov chains.
            initstate ('array-like' or str or SolverInitState or int):
                Initial state specification (see Sampler docs). Defaults to 'RND'.
            backend (str):
                Computational backend ('numpy', 'jax', 'default').
            logprob_fact (float):
                Factor used in calculating importance sampling weights, often :math:`1/\mu`.
                The probability is calculated as :math:`\exp((\frac{1}{\text{logprob_fact}} - \mu) \text{Re}(\log\psi(s)))`.
                Default 0.5 corresponds to Born rule :math:`|\psi|^2` when :math:`\mu=1`. Needs careful setting based on :math:`\mu`.
            **kwargs:
                Additional arguments for the base Sampler class (e.g., `modes`, `mode_repr`).
        """
        
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
        
        self._beta                              = beta
        self._therm_steps                       = therm_steps
        self._sweep_steps                       = sweep_steps
        self._logprob_fact                      = logprob_fact
        self._logprobas                         = None
        self._total_therm_updates               = therm_steps * sweep_steps * self._size    # Total updates during thermalization
        self._total_sample_updates_per_sample   = sweep_steps * self._size                  # Updates between collected samples
        self._updates_per_sample                = self._sweep_steps                         # Steps between samples
        self._total_sample_updates_per_chain    = numsamples * self._updates_per_sample * self._numchains
        
        self._upd_fun = upd_fun
        if self._upd_fun is None:
            if self._isjax:
                # Bind RNG arguments to the JAX updater, already JIT-compiled.
                self._upd_fun = _propose_random_flip_jax
            else:
                # For NumPy backend, bind the RNG to the updater.
                self._upd_fun = _propose_random_flip_np
    
        self._logprob_fact = logprob_fact
        self._logprobas = None # Will store log prob of current states
    
    #####################################################################
    
    def __repr__(self):
        """
        Provide a string representation of the MCSampler object.

        Returns:
            str: A formatted string containing the key attributes of the MCSampler instance,
        """
        return (f"MCSampler(shape={self._shape}, mu={self._mu}, beta={self._beta}, "
                f"therm_steps={self._therm_steps}, sweep_steps={self._sweep_steps}, "
                f"numsamples={self._numsamples}, numchains={self._numchains}, backend={self._backendstr})")

    def __str__(self):
        total_therm_updates_display     = self._total_therm_updates * self.size                     # Total updates per site
        total_sample_updates_display    = self._numsamples * self._updates_per_sample * self.size   # Total sample updates per site
        return (f"MCSampler:\n"
                f"  - State shape: {self._shape} (Size: {self.size})\n"
                f"  - Backend: {self._backendstr}\n"
                f"  - Chains: {self._numchains}, Samples/Chain: {self._numsamples}\n"
                f"  - Params: mu={self._mu:.3f}, beta={self._beta:.3f}, logprob_fact={self._logprob_fact:.3f}\n"
                f"  - Thermalization: {self._therm_steps} sweeps x {self._sweep_steps} steps/sweep ({total_therm_updates_display} total site updates/chain)\n"
                f"  - Sampling: {self._updates_per_sample} steps/sample ({total_sample_updates_display} total site updates/chain)\n")
        
    ###################################################################
    
    def _set_net_callable(self, net):
        '''
        Set the network callable and extract parameters if applicable.

        Parameters:
            net (Callable or Flax Module):
                Input network object.

        Returns:
            Tuple[Callable, Any]: (network_callable, parameters)
        '''
        self._net = net
        
        # Check if it's a Flax Linen module (or similar with apply method)
        if isinstance(net, nn.Module):
            # Assume parameters are managed externally or within the module state
            # We need the apply function and the parameters.
            # This might require the user to pass parameters explicitly during sampling if they change.
            # Let's store the apply method and expect parameters at sample time.
            return net.apply, None # Parameters will be provided later
        elif hasattr(net, 'get_apply') and callable(net.get_apply):
            network_callable, parameters = net.get_apply()
            return network_callable, parameters
        elif hasattr(net, 'apply') and callable(net.apply):
            return net.apply, self.net.geta_params() if hasattr(net, 'geta_params') else self._parameters
        elif callable(net):
            return net, None # Assume no external parameters needed unless provided at sample time
        raise ValueError("Invalid network object provided. Needs to be callable or have an 'apply' method.")
    
    ###################################################################
    #! ACCEPTANCE PROBABILITY
    ###################################################################
    
    @staticmethod
    @jax.jit
    def _acceptance_probability_jax(current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        r'''
        Calculate the Metropolis-Hastings acceptance probability using JAX.

        ---
        Calculates:
            :math:`\min(1, \exp(\beta \mu \cdot \text{Re}(\text{val}_{\text{cand}} - \text{val}_{\text{curr}})))`.
        ---
        Parameters:
            current_val (jnp.ndarray):
                Log-probability (or related value) of the current state(s).
            candidate_val (jnp.ndarray):
                Log-probability (or related value) of the candidate state(s).
            beta (float):
                Inverse temperature :math:`\beta`. Static argument for JIT.
            mu (float):
                Exponent :math:`\mu`. Static argument for JIT.

        Returns:
            jnp.ndarray: The acceptance probability(ies).
        '''
        log_acceptance_ratio = beta * mu * (jnp.real(candidate_val) - jnp.real(current_val))
        return jnp.exp(log_acceptance_ratio)
    
    @staticmethod
    @numba.njit
    def _acceptance_probability_np(current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        '''
        Calculate the acceptance probability for the Metropolis-Hastings
        algorithm.
        Parameters:
            - current_val   : The value of the current state
            - candidate_val : The value of the candidate state
        
        Returns:
            - The acceptance probability as a float
        '''

        log_acceptance_ratio = beta * mu * np.real(candidate_val - current_val)
        return np.exp(log_acceptance_ratio)
    
    def acceptance_probability(self, current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        r'''
        Calculate the Metropolis-Hastings acceptance probability.

        Selects backend automatically. Uses instance `_beta` if `beta` is None.

        Parameters:
            current_val (array-like):
                Value (:math:`\log p(s)`) of the current state(s).
            candidate_val (array-like):
                Value (:math:`\log p(s')`) of the candidate state(s).
            beta (Optional[float]):
                Inverse temperature :math:`\beta`. Uses `self._beta` if None.
            mu (Optional[float]):
                Exponent :math:`\mu`. Uses `self._mu` if None.
        Returns:
            array-like: Acceptance probability(ies).
        '''
        use_beta = beta if beta is not None else self._beta

        if self._isjax:
            # We need beta to be static for the JIT version.
            # If beta can change dynamically, we cannot use the JIT version directly here.
            # Workaround:
            #   If beta is dynamic, call a non-jitted wrapper or re-jit if necessary.
            # Assuming beta is constant for a given sampling run for JIT benefits.
            
            if beta is not None and beta != self._beta:
                # If called with a different beta, might need a non-jitted path
                # or expect user to handle JIT compilation if beta changes often.
                # For simplicity, we assume the compiled version uses self._beta if beta is None.
                # If beta is provided *and different*, direct call might be slow if not JITted for that beta.
                # Let's use the JITted version with the provided beta if possible.
                # This requires careful handling of static arguments if beta changes frequently.
                # A simpler approach: always use self._beta in JITted functions called internally.
                # If external calls need different beta, they call this wrapper which might be slower.

                # Call JITted function with potentially non-static beta (will compile on first call with this beta)
                @partial(jax.jit, static_argnames=('beta',))
                def _accept_jax_dynamic_beta(cv, cdv, beta, mu):
                    log_acceptance_ratio = beta * jnp.real(cdv - cv) * mu
                    return jnp.exp(log_acceptance_ratio)
                return _accept_jax_dynamic_beta(current_val, candidate_val, use_beta, mu)
            else:
                return self._acceptance_probability_jax(current_val, candidate_val, beta=use_beta, mu=mu)
        else:
            return self._acceptance_probability_np(current_val, candidate_val, beta=use_beta, mu=mu)
        
    ###################################################################
    #! LOG PROBABILITY
    ###################################################################
    
    @partial(jax.jit, static_argnames=('net_callable',))
    def _logprob_jax(x, net_callable, net_params=None):
        r'''
        Calculate log probability :math:`\log \psi(s)` using JAX.
        
        This version is fully vectorized: it assumes that x is a batched
        input and applies the network callable via jax.vmap.
        
        Parameters:
            x (jnp.ndarray): Batched state configurations.
            net_callable (Callable): Function with signature: log_psi = net_callable(net_params, state)
            net_params (Any, optional): Parameters for the network callable.
        
        Returns:
            jnp.ndarray: The real part of log probabilities for the batch.
        '''
        batched_log_psi = jax.vmap(lambda s: net_callable(net_params, s))(x)
        return jnp.real(batched_log_psi)
    
    @staticmethod
    @numba.njit
    def _logprob_np(x, net_callable, net_params = None):
        '''
        Calculate the log probability of a state using NumPy.
        Parameters:
            - x             : The state
            - net_callable  : The network callable (returns (\\log\\psi(s)))
            - net_params    : The network parameters - may be None but for flax callables
                            the parameters are necessary strictly!
        Returns:
            - The log probability as a float or complex number
        '''
        return np.array([net_callable(net_params, y) for y in x]).reshape(x.shape[0])
    
    def logprob(self, x, net_callable = None, net_params = None):
        r'''Calculate the log probability used for MCMC steps.

        Computes :math:`\text{Re}(\log \psi(s))`.
        Uses instance defaults if arguments are None. Selects backend automatically.

        Parameters:
            x (array-like):
                State configuration(s).
            net_callable (Optional[Callable]):
                Network callable. Uses `self._net_callable` if None.
            net_params (Any):
                Network parameters. Uses `self._parameters` if None.

        Returns:
            array-like: Log probabilities.
        '''
        
        use_callable    = net_callable if net_callable is not None else self._net_callable
        use_params      = net_params if net_params is not None else self._parameters
        
        if use_callable is None:
            raise ValueError("No callable provided for log probability calculation.")
    
        # if isinstance(self._net, nn.Module) and use_params is None:
        #     # Try to get params from the network instance if possible
        #     if hasattr(self._net, 'params'):
        #         use_params = self._net.params
        #     elif hasattr(self._net, 'get_params') and callable(self._net.get_params):
        #         _, use_params = self._net.get_params()
        #     else:
        #         # Cannot automatically get params, raise error or warning
        #         raise ValueError("Network seems to require parameters, but none were provided or found.")
            
        if self._isjax:
            return MCSampler._logprob_jax(x, use_callable, use_params)
        return MCSampler._logprob_np(x, use_callable, use_params)

    ###################################################################
    #! UPDATE CHAIN
    ###################################################################
    
    @staticmethod
    @partial(jax.jit, static_argnames=('steps', 'update_proposer',
                                    'log_proba_fun', 'accept_config_fun',
                                    'net_callable_fun'))
    def _run_mcmc_steps_jax(chain_init,
                            current_val_init,
                            rng_k_init,
                            num_proposed_init,
                            num_accepted_init,
                            params,
                            steps               : int,
                            update_proposer     : Callable,
                            log_proba_fun       : Callable,
                            accept_config_fun   : Callable,
                            net_callable_fun    : Callable,
                            mu                  : float,
                            beta                : float = 1.0):
        r'''
        Runs multiple MCMC steps using lax.scan. JIT-compiled.
        The single-step logic is defined internally via closure.
        Parameters:
            chain_init (jnp.ndarray):
                Initial state of the chain (shape: [numchains, *shape]).
            current_val_init (jnp.ndarray):
                Initial log-probability values for each chain (1D array).
            rng_k_init (jax.random.PRNGKey):
                Initial random key for JAX.
            num_proposed_init (jnp.ndarray):
                Initial number of proposals made so far (1D array).
            num_accepted_init (jnp.ndarray):
                Initial number of accepted proposals so far (1D array).
            params (Any):
                Network parameters.
            steps (int):
                Number of MCMC steps to perform.
            update_proposer (Callable):
                Function that proposes a new state.
            log_proba_fun (Callable):
                Function to compute log probabilities.
            accept_config_fun (Callable):
                Function to compute acceptance probabilities.
            net_callable_fun (Callable):
                The network callable.
            mu (float):
                Exponent :math:`\mu` for the sampling distribution.
            beta (float):
                Inverse temperature :math:`\beta` for the Metropolis acceptance probability.
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]:
                Final chain states, log-probabilities, updated random key,
        '''

        num_chains = chain_init.shape[0]

        # Define the single-step function *inside* so it closes over the arguments
        # like params, update_proposer, log_proba_fun, etc.
        def _sweep_chain_jax_step_inner(carry, _):
            # Unpack the carry
            chain_in, current_val_in, rng_k_in, num_proposed_in, num_accepted_in = carry
            
            # jax.debug.print("chain_in: {}", chain_in.shape)
            # jax.debug.print("current_val_in: {}", current_val_in.shape)
            # jax.debug.print("rng_k_in: {}", rng_k_in.shape)
            # jax.debug.print("num_proposed_in: {}", num_proposed_in.shape)
            # jax.debug.print("num_accepted_in: {}", num_accepted_in.shape)
            # return carry, None
            # Split the key only once to generate proposal keys for all chains + one carry key
            keys                    = random_jp.split(rng_k_in, num_chains + 1)
            proposal_keys           = keys[:-1]
            new_rng_k               = keys[-1]
            
            #! Single MCMC update step logic
            # Propose update
            new_chain               = jax.vmap(update_proposer, in_axes=(0, 0))(chain_in, proposal_keys)
            if new_chain.ndim == chain_in.ndim + 1:
                new_chain = jnp.squeeze(new_chain, axis=-1)
                
            logprobas_new           = log_proba_fun(new_chain, net_callable=net_callable_fun, net_params=params)
            if logprobas_new.ndim == current_val_in.ndim + 1:
                logprobas_new = jnp.squeeze(logprobas_new, axis=-1)

            # Calculate acceptance probabilities (using accept_config_fun)
            acc_probability         = accept_config_fun(current_val_in, logprobas_new, beta, mu)
            uniform_draw            = random_jp.uniform(new_rng_k, shape=(num_chains,))
            accepted                = uniform_draw < acc_probability
        
            num_proposed_out        = num_proposed_in + 1
            num_accepted_out        = num_accepted_in + accepted.astype(num_accepted_in.dtype)
            
            accepted_expanded       = accepted[:, None]
            new_chain_final         = jnp.where(accepted_expanded, new_chain, chain_in)
            new_val_final           = jnp.where(accepted, logprobas_new, current_val_in)
            # jax.debug.print("accepted: {}", accepted.shape)
            # jax.debug.print("chain in: {}", chain_in.shape)
            # jax.debug.print("new_chain: {}", new_chain.shape)
            # jax.debug.print("logprobas_new: {}", logprobas_new.shape)
            # jax.debug.print("current_val_in: {}", current_val_in.shape)
            # return carry, None
            # def update(acc, old, new):
                # return jax.lax.select(acc, new, old)
            # new_carry_states        = jax.vmap(update, in_axes=(0, 0, 0))(accepted, chain_in, new_chain)
            # new_carry_vals          = jnp.where(accepted, logprobas_new, current_val_in)
            new_carry = (new_chain_final, new_val_final, new_rng_k, num_proposed_out, num_accepted_out)
            return new_carry, None        
        
        # Initial carry contains only the dynamic state passed into this function
        initial_carry   = (chain_init, current_val_init, rng_k_init, num_proposed_init, num_accepted_init)

        # Call lax.scan with the inner step function
        final_carry, _  = jax.lax.scan(_sweep_chain_jax_step_inner, initial_carry, None, length=steps)
        return final_carry
    
    @staticmethod
    @numba.njit
    def _run_mcmc_steps_np(chain            : np.ndarray,
                        logprobas           : np.ndarray,
                        num_proposed        : np.ndarray,
                        num_accepted        : np.ndarray,
                        params              : Any,
                        rng                 : np.random.Generator,
                        steps               : int,
                        mu                  : float,
                        beta                : float,
                        update_proposer     : Callable,
                        log_proba_fun       : Callable,
                        accept_config_fun   : Callable,
                        net_callable_fun    : Callable,
                        ):
        """
        NumPy version of sweeping a single chain.
        
        This function carries a tuple:
        (chain, current_logprob, rng_k, num_proposed, num_accepted)
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
        
        current_val     = logprobas if logprobas is not None else log_proba_fun(chain, mu=mu, net_callable=net_callable_fun, net_params=params)
        n_chains        = chain.shape[0] if len(chain.shape) > 1 else 1
        
        def loop_sweep(chain, current_val, num_proposed, num_accepted, rng):
            # Loop through the number of steps
            for _ in range(steps):
                
                # use the temporary states for the proposal
                # tmpstates               = chain.copy()
                
                # For each chain element, generate a new candidate using the update proposer.
                # Here, we simulate splitting rng_k by simply generating a new random integer key.
                # also, this flips the state inplace, so we need to copy the chain
                # and then update the chain with the new state.
                # Use NumPy's vectorized operations to generate new states
                new_chain               = update_proposer(chain, rng)
                
                # Compute new log probabilities for candidates using the provided log probability function.
                new_logprobas           = log_proba_fun(new_chain, net_callable=net_callable_fun, net_params=params)
                
                # Compute acceptance probability for each chain element.
                acc_probability         = accept_config_fun(current_val, new_logprobas, beta, mu)
                
                # Decide acceptance by comparing against a random uniform sample.
                rand_vals               = rng.random(size=n_chains)
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
            return chain, current_val, num_proposed, num_accepted
        
        # go through the number of steps, for each step: 
        # 1. propose a new state
        # 2. compute the log probability
        # 3. compute the acceptance probability
        # 4. decide with dice rule
        # 5. update the chain
        # 6. update the carry
        # 7. update rng_k with a new random integer (not used further)
        chain, current_val, num_proposed, num_accepted = loop_sweep(chain, current_val, num_proposed, num_accepted, rng)
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
        Sweep the chain for a given number of MCMC steps.
        One "step" typically involves proposing N updates, where N is the system size.

        Parameters:
            chain (array-like):
                Current states of the chains.
            logprobas (array-like):
                Log probabilities of the current states.
            rng_k_or_rng (Any):
                JAX PRNGKey or NumPy Generator.
            num_proposed (array-like):
                Array tracking proposed moves per chain.
            num_accepted (array-like):
                Array tracking accepted moves per chain.
            params (Any):
                Network parameters.
            steps (int):
                Number of MCMC update steps to perform.

        Returns:
            Tuple:
                (updated_chain, updated_logprobas, updated_rng_k/rng, updated_num_proposed, updated_num_accepted)
        '''
        use_log_proba_fun       = self.logprob if log_proba_fun is None else log_proba_fun
        use_accept_config_fun   = self.acceptance_probability if accept_config_fun is None else accept_config_fun
        use_net_callable_fun    = self._net_callable if net_callable_fun is None else net_callable_fun
        use_update_proposer     = self._upd_fun if update_proposer is None else update_proposer
        
        if logprobas is None:
            logprobas = self._logprobas
            if logprobas is None:
                logprobas = self.logprob(chain, net_callable=net_callable_fun, net_params=params)
        
        if self._isjax:
            # call the compiled version
            return self._run_mcmc_steps_jax(chain_init          =   chain,
                                            current_val_init    = logprobas,
                                            rng_k_init          = rng_k,
                                            num_proposed_init   = num_proposed,
                                            num_accepted_init   = num_accepted,
                                            params              = params,
                                            steps               = steps,
                                            update_proposer     = use_update_proposer,
                                            log_proba_fun       = use_log_proba_fun,
                                            accept_config_fun   = use_accept_config_fun,
                                            net_callable_fun    = use_net_callable_fun,
                                            mu                  = self._mu)
        # otherwise, numpy it is
        return self._run_mcmc_steps_np(chain                =   chain,
                                        logprobas           =   logprobas,
                                        rng                 =   rng_k,
                                        num_proposed        =   num_proposed,
                                        num_accepted        =   num_accepted,
                                        params              =   params,
                                        update_proposer     =   use_update_proposer,
                                        log_proba_fun       =   use_log_proba_fun,
                                        accept_config_fun   =   use_accept_config_fun,
                                        net_callable_fun    =   use_net_callable_fun,
                                        mu                  =   self._mu,
                                        beta                =   self._beta,
                                        steps               =   steps)
    
    ###################################################################
    #! SAMPLING
    ###################################################################
    
    @staticmethod
    @partial(jax.jit, static_argnames=('num_samples', 'total_therm_updates', 'updates_per_sample',
                                    'update_proposer', 'log_proba_fun',
                                    'accept_config_fun', 'net_callable_fun'))
    def _generate_samples_jax(
        states_init         : 'array-like',
        logprobas_init      : 'array-like',
        rng_k_init          : 'array-like',
        num_proposed_init   : 'array-like',
        num_accepted_init   : 'array-like',
        params              : Any,
        num_samples         : int,
        total_therm_updates : int,
        updates_per_sample  : int,
        mu                  : float,
        beta                : float,
        # Pass the actual function objects needed:
        update_proposer     : Callable,
        log_proba_fun       : Callable,
        accept_config_fun   : Callable,
        net_callable_fun    : Callable):
        '''
        JIT-compiled function for thermalization and sample collection using MCMC.
        Uses closures for scan bodies.
        '''

        #! Thermalization phase
        # Calls the static method _run_mcmc_steps_jax
        states_therm, logprobas_therm, rng_k_therm, num_proposed_therm, num_accepted_therm = \
            MCSampler._run_mcmc_steps_jax(
            chain_init          =       states_init,
            current_val_init    =       logprobas_init,
            rng_k_init          =       rng_k_init,
            num_proposed_init   =       num_proposed_init,
            num_accepted_init   =       num_accepted_init,
            params              =       params,
            steps               =       total_therm_updates,
            # Pass the functions through:
            update_proposer     =       update_proposer,
            log_proba_fun       =       log_proba_fun,
            accept_config_fun   =       accept_config_fun,
            net_callable_fun    =       net_callable_fun,
            mu                  =       mu,
            beta                =       beta
            )

        #! Sampling phase (using lax.scan for collection)
        def sample_scan_body(carry, _):
            # Unpack dynamic carry state
            states_carry, logprobas_carry, rng_k_carry, num_proposed_carry, num_accepted_carry = carry

            # Run MCMC steps between samples using the static method
            states_new, logprobas_new, rng_k_new, num_proposed_new, num_accepted_new = \
                MCSampler._run_mcmc_steps_jax(
                    chain_init          =   states_carry,
                    current_val_init    =   logprobas_carry,
                    rng_k_init          =   rng_k_carry,
                    num_proposed_init   =   num_proposed_carry,
                    num_accepted_init   =   num_accepted_carry,
                    params              =   params,
                    steps               =   updates_per_sample,
                    update_proposer     =   update_proposer,
                    log_proba_fun       =   log_proba_fun,
                    accept_config_fun   =   accept_config_fun,
                    net_callable_fun    =   net_callable_fun,
                    mu                  =   mu,
                    beta                =   beta
                )

            # Return updated dynamic carry and the collected sample (states_new)
            return (states_new, logprobas_new, rng_k_new, num_proposed_new, num_accepted_new), states_new

        # Run the scan to collect samples
        initial_scan_carry              = (states_therm, logprobas_therm, rng_k_therm, num_proposed_therm, num_accepted_therm)
        final_carry, collected_samples  = jax.lax.scan(
            f       = sample_scan_body, # Use the inner function defined above
            init    = initial_scan_carry,
            xs      = None,
            length  = num_samples
        )
        # final_carry contains the state after the last sample collection sweep
        return final_carry, collected_samples
    
    def _generate_samples_np(self,
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
        states, logprobas, num_proposed, num_accepted = self._run_mcmc_steps_np(
            chain               = self._states,
            logprobas           = self._logprobas,
            num_proposed        = self._num_proposed,
            num_accepted        = self._num_accepted,
            params              = params,
            update_proposer     = self._upd_fun,
            log_proba_fun       = self.logprob,
            accept_config_fun   = self.acceptance_probability,
            net_callable_fun    = self._net_callable,
            steps               = self._therm_steps * self._sweep_steps,
            rng                 = self._rng,
            mu                  = self._mu,
            beta                = self._beta)
        
        # Now perform numSamples sweeps, collecting the resulting states
        meta            = []
        configs_list    = []
        
        # Perform the sampling sweeps - do the same as the thermalization but save the states
        for _ in range(num_samples):
            states, logprobas, num_proposed, num_accepted = self._run_mcmc_steps_np(
                chain               = states,
                logprobas           = logprobas,
                num_proposed        = num_proposed,
                num_accepted        = num_accepted,
                params              = params,
                update_proposer     = self._upd_fun,
                log_proba_fun       = self.logprob,
                accept_config_fun   = self.acceptance_probability,
                net_callable_fun    = self._net_callable,
                steps               = self._sweep_steps,
                rng                 = self._rng,
                mu                  = self._mu,
                beta                = self._beta)
                
            meta        = (states.copy(), logprobas.copy(), num_proposed, num_accepted)
            # Assume states is of shape (num_chains, *self._shape)
            configs_list.append(states.copy())
        
        # Concatenate configurations along the chain axis
        # Then reshape to (num_samples, -1) + self._shape (flattening the chain dimension per sample)
        configs = np.concatenate(configs_list, axis=0)
        return meta, configs.reshape((num_samples, -1) + self._shape)
    
    ###################################################################
    #! STATIC JAX SAMPLING KERNEL
    ###################################################################
    
    @staticmethod
    @partial(jax.jit, static_argnames=(
        'num_samples', 'num_chains', 'total_therm_updates', 'updates_per_sample', 'shape',
        'update_proposer', 'log_proba_fun_base', 'accept_config_fun_base',
        'net_callable_fun'))
    def _static_sample_jax(
            # --- Initial State (Dynamic) ---
            states_init         : jnp.ndarray,
            rng_k_init          : jnp.ndarray,
            num_proposed_init   : jnp.ndarray,
            num_accepted_init   : jnp.ndarray,
            # --- Network/Model (Dynamic) ---
            params              : Any,
            # --- Configuration (Static) ---
            num_samples         : int,
            num_chains          : int,
            total_therm_updates : int,
            updates_per_sample  : int,
            shape               : Tuple[int,...],
            # --- Configuration (Dynamic Values) ---
            mu                  : float,
            beta                : float,
            logprob_fact        : float,
            # --- Function References (Static) ---
            update_proposer     : Callable,
            log_proba_fun_base  : Callable,     # e.g., MCSampler._logprob_jax
            accept_config_fun_base: Callable,   # e.g., MCSampler._acceptance_probability_jax
            net_callable_fun    : Callable
        ):
        r'''
        Static, JIT-compiled core logic for MCMC sampling in JAX. 
        
        Performs the following steps:
        1. Calculate Initial Log Probs
        2. Generate Samples via MCMC Kernel
        3. Post-processing
        4. Return Results
        Parameters:
            states_init (jnp.ndarray):
                Initial state(s) of the chain(s).
            rng_k_init (jnp.ndarray):
                Initial random key for JAX.
            num_proposed_init (jnp.ndarray):
                Initial number of proposals made so far.
            num_accepted_init (jnp.ndarray):
                Initial number of accepted proposals so far.
            params (Any):
                Network parameters.
            num_samples (int):
                Number of samples to generate.
            num_chains (int):
                Number of chains to use.
            total_therm_updates (int):
                Total number of thermalization updates.
            updates_per_sample (int):
                Number of updates per sample collection.
            shape (Tuple[int,...]):
                Shape of the state configurations.
            mu (float):
                Exponent :math:`\mu` for the sampling distribution.
            beta (float):
                Inverse temperature :math:`\beta` for the Metropolis acceptance probability.
            logprob_fact (float):
                Logarithmic factor for normalization.
            update_proposer (Callable):
                Function that proposes a new state.
            log_proba_fun_base (Callable):
                Function to compute log probabilities.
            accept_config_fun_base (Callable):
                Function to compute acceptance probabilities.
            net_callable_fun (Callable):
                The network callable.
        '''
        logprobas_init = log_proba_fun_base(states_init, net_callable_fun, params)
        if logprobas_init.ndim > 1:
            logprobas_init = jnp.squeeze(logprobas_init, axis=-1)

        #! Phase 1: Thermalization
        final_carry = MCSampler._run_mcmc_steps_jax(
            chain_init          =   states_init,
            current_val_init    =   logprobas_init,
            rng_k_init          =   rng_k_init,
            num_proposed_init   =   num_proposed_init,
            num_accepted_init   =   num_accepted_init,
            params              =   params,
            steps               =   total_therm_updates,
            update_proposer     =   update_proposer,
            log_proba_fun       =   log_proba_fun_base,
            accept_config_fun   =   accept_config_fun_base,
            net_callable_fun    =   net_callable_fun,
            mu                  =   mu,
            beta                =   beta
        )
        # Unpack thermalized state (we ignore counters for brevity)
        states_therm, logprobas_therm, rng_k_therm, _, _ = final_carry

        #! Phase 2: Sampling
        
        def sample_scan_body(carry, _):
            # Run a fixed number of updates per sample and update the chain carry.
            new_carry = MCSampler._run_mcmc_steps_jax(
                chain_init            = carry[0],
                current_val_init      = carry[1],
                rng_k_init            = carry[2],
                num_proposed_init     = carry[3],
                num_accepted_init     = carry[4],
                params                = params,
                steps                 = updates_per_sample,
                update_proposer       = update_proposer,
                log_proba_fun         = log_proba_fun_base,
                accept_config_fun     = accept_config_fun_base,
                net_callable_fun      = net_callable_fun,
                mu                    = mu,
                beta                  = beta
            )
            # Return updated carry and collect the new chain state.
            return new_carry, new_carry[0]

        # Initialize the carry for the scan with the thermalized state and counters.
        initial_scan_carry = (states_therm, logprobas_therm, rng_k_therm, num_proposed_init, num_accepted_init)

        # Run the scan for the specified number of sample steps.
        final_carry, collected_samples = jax.lax.scan(sample_scan_body, initial_scan_carry, None, length=num_samples)

        # Flatten the collected states from all sample steps.
        configs_flat        = collected_samples.reshape((-1,) + shape)
        
        # Evaluate the network in a fully batched (vectorized) manner to obtain log_ψ.
        batched_log_ansatz  = jax.vmap(lambda conf: net_callable_fun(params, conf))(configs_flat)

        # Compute the importance weights.
        log_prob_exponent    = (1.0 / logprob_fact - mu)
        probs                = jnp.exp(log_prob_exponent * jnp.real(batched_log_ansatz))
        total_samples        = num_samples * num_chains
        norm_factor          = jnp.where(jnp.sum(probs) > 1e-10, jnp.sum(probs), 1e-10)
        probs_normalized     = (probs / norm_factor) * total_samples

        # Prepare the final output tuples.
        final_state_tuple   = final_carry  # Contains the final chain state and updated counters.
        samples_tuple       = (configs_flat, batched_log_ansatz)
        return final_state_tuple, samples_tuple, probs_normalized
    
    ###################################################################
    
    def sample(self, parameters=None, num_samples=None, num_chains=None):
        '''
        Sample the states from the Hilbert space according to the Born distribution.
        Parameters:
            parameters:
                The parameters for the network
            num_samples:
                The number of samples to generate
            num_chains:
                The number of chains to use
        Returns:
            The sampled states
        '''
        
        # check the number of samples and chains
        used_num_samples        = num_samples if num_samples is not None else self._numsamples
        used_num_chains         = num_chains if num_chains is not None else self._numchains

        # Handle temporary state if num_chains differs from instance default
        current_states          = self._states
        current_proposed        = self._num_proposed
        current_accepted        = self._num_accepted
        reinitialized_for_call  = False

        current_states          = self._states
        current_proposed        = self._num_proposed
        current_accepted        = self._num_accepted
        reinitialized_for_call  = False
        if used_num_chains != self._numchains:
            print(f"Warning: Running sample with {used_num_chains} chains (instance default is {self._numchains}). State reinitialized for this call.")
            initstate_template  = self._initstate
            if self._isjax:
                if not isinstance(initstate_template, jnp.ndarray):
                    initstate_template = jnp.array(initstate_template)
                current_states      = jnp.repeat(initstate_template[jnp.newaxis, ...], used_num_chains, axis=0)
                current_proposed    = jnp.zeros(used_num_chains, dtype=DEFAULT_JP_INT_TYPE)
                current_accepted    = jnp.zeros(used_num_chains, dtype=DEFAULT_JP_INT_TYPE)
            else:
                if not isinstance(initstate_template, np.ndarray):
                    initstate_template = np.array(initstate_template)
                current_states      = np.stack([initstate_template.copy() for _ in range(used_num_chains)], axis=0)
                current_proposed    = np.zeros(used_num_chains, dtype=DEFAULT_NP_INT_TYPE)
                current_accepted    = np.zeros(used_num_chains, dtype=DEFAULT_NP_INT_TYPE)
            reinitialized_for_call = True
        
        # check the parameters - if not given, use the current parameters
        current_params = None
        if parameters is not None:
            current_params = parameters
        elif self._parameters is not None:
            current_params = self._net.get_params()
        elif hasattr(self._net, 'params'):
            current_params = self._net.params
        elif hasattr(self._net, 'get_params'):
            current_params = self._net.get_params()
        
        net_callable, current_params = self._set_net_callable(self._net)

        if self._isjax:
            if not isinstance(self._rng_k, jax.Array):
                raise TypeError(f"Sampler's RNG key is invalid: {type(self._rng_k)}")

            # Call the static, JIT-compiled function
            final_state_tuple, samples_tuple, probs = MCSampler._static_sample_jax(
                # Initial State (Dynamic)
                states_init             = current_states,
                rng_k_init              = self._rng_k,
                num_proposed_init       = current_proposed,
                num_accepted_init       = current_accepted,
                # Network/Model (Dynamic)
                params                  = current_params,
                # Configuration (Static Args)
                num_samples             = used_num_samples,
                num_chains              = used_num_chains,
                total_therm_updates     = self._total_therm_updates, # Based on instance settings
                updates_per_sample      = self._updates_per_sample,  # Based on instance settings
                shape                   = self._shape,
                # Configuration (Dynamic Values)
                mu                      = self._mu,
                beta                    = self._beta,
                logprob_fact            = self._logprob_fact,
                # Function References (Static Args) - Pass the static base methods
                update_proposer         = self._upd_fun, # The configured update function instance
                log_proba_fun_base      = MCSampler._logprob_jax,
                accept_config_fun_base  = MCSampler._acceptance_probability_jax,
                net_callable_fun        = self._net_callable
            )
        
            # Update Instance State (JAX)
            final_states, final_logprobas, final_rng_k, final_num_proposed, final_num_accepted = final_state_tuple
            self._rng_k                 = final_rng_k
            # if not reinitialized_for_call:
            self._states                = final_states
            self._logprobas             = final_logprobas
            self._num_proposed          = final_num_proposed
            self._num_accepted          = final_num_accepted

            final_state_info            = (self._states, self._logprobas)
            return final_state_info, samples_tuple, probs
        else:
            self._logprobas             = self.logprob(self._states,
                                            net_callable    =   net_callable,
                                            net_params      =   parameters)
            # for numpy
            (self._states, self._logprobas, self._num_proposed, self._num_accepted), configs =\
                self._generate_samples_np(parameters, num_samples, num_chains)
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

    def get_sampler_jax(self, num_samples: Optional[int] = None, num_chains: Optional[int] = None) -> Callable:
        """
        Returns a JIT-compiled static sampling function with baked-in configuration.

        This function partially applies the core static JAX sampling logic
        (`_static_sample_jax`) with configuration parameters (like mu, beta,
        step counts, function references) taken from the current sampler instance.

        The returned function is JIT-compiled for the specific `num_samples` and
        `num_chains` provided (or taken from the instance defaults). It has a
        simplified signature suitable for repeated calls where only the initial state,
        RNG key, and network parameters might change frequently.

        Parameters:
            num_samples (Optional[int]):
                Number of samples per chain to bake into the static function.
                If None, uses `self._numsamples`.
            num_chains (Optional[int]):
                Number of chains to bake into the static function.
                If None, uses `self._numchains`.

        Returns:
            Callable: A JIT-compiled function with the signature:
            `wrapped_sampler(states_init, rng_k_init, params,
                            num_proposed_init=None, num_accepted_init=None)`
            where `num_proposed_init` and `num_accepted_init` default to zero arrays
            of the correct shape and integer dtype if not provided.
            It returns the same tuple structure as `_static_sample_jax`:
            `(final_state_tuple, samples_tuple, probs_normalized)`

        Raises:
            RuntimeError: If the sampler backend is not 'jax'.
        """
        if not self._isjax:
            raise RuntimeError("Static JAX sampler getter only available for JAX backend.")

        # Use provided sample/chain counts or instance defaults
        static_num_samples  = num_samples if num_samples is not None else self._numsamples
        static_num_chains   = num_chains if num_chains is not None else self._numchains

        # Gather arguments to be baked in (partial application)
        # Configuration values from self
        baked_args = {
            #! Static Config
            "num_samples"               : static_num_samples,
            "num_chains"                : static_num_chains,
            "total_therm_updates"       : self._total_therm_updates,
            "updates_per_sample"        : self._updates_per_sample,
            "shape"                     : self._shape,
            "update_proposer"           : self._upd_fun,
            "log_proba_fun_base"        : MCSampler._logprob_jax,
            "accept_config_fun_base"    : MCSampler._acceptance_probability_jax,
            "net_callable_fun"          : self._net_callable,
            # Dynamic Config (but fixed for this partial function)
            "mu"                        : self._mu,
            "beta"                      : self._beta,
            "logprob_fact"              : self._logprob_fact,
        }

        # Create the partially applied function
        # Target function is MCSampler._static_sample_jax
        partial_sampler                 = partial(MCSampler._static_sample_jax, **baked_args)

        # Define the final wrapper function with the desired signature
        # This wrapper handles default values for initial counters
        int_dtype                       = DEFAULT_JP_INT_TYPE

        @jax.jit
        def wrapped_sampler(states_init         : jax.Array,
                            rng_k_init          : jax.Array,
                            params              : Any,
                            num_proposed_init   : Optional[jax.Array] = None,
                            num_accepted_init   : Optional[jax.Array] = None):
            """
            JIT-compiled static sampler with baked-in configuration.

            Args:
                states_init: Initial states [num_chains, *shape].
                rng_k_init: Initial JAX PRNGKey.
                params: Network parameters for this run.
                num_proposed_init: Initial proposal counts [num_chains] (defaults to zeros).
                num_accepted_init: Initial acceptance counts [num_chains] (defaults to zeros).

            Returns:
                (final_state_tuple, samples_tuple, probs_normalized)
            """
            # Default initial counters to zero if not provided
            if num_proposed_init is None:
                num_proposed_init = jnp.zeros(static_num_chains, dtype=int_dtype)
            if num_accepted_init is None:
                num_accepted_init = jnp.zeros(static_num_chains, dtype=int_dtype)

            # Validate shapes of inputs relative to static_num_chains
            if states_init.shape[0] != static_num_chains:
                raise ValueError(f"states_init first dimension ({states_init.shape[0]}) must match static num_chains ({static_num_chains})")
            if num_proposed_init.shape != (static_num_chains,):
                raise ValueError(f"num_proposed_init shape ({num_proposed_init.shape}) must match ({static_num_chains},)")
            if num_accepted_init.shape != (static_num_chains,):
                raise ValueError(f"num_accepted_init shape ({num_accepted_init.shape}) must match ({static_num_chains},)")

            # Call the partially applied core function
            final_state_tuple, samples_tuple, probs = partial_sampler(
                states_init         =   states_init,
                rng_k_init          =   rng_k_init,
                num_proposed_init   =   num_proposed_init.astype(int_dtype),
                num_accepted_init   =   num_accepted_init.astype(int_dtype),
                params              =   params
            )
            final_states, final_logprobas, final_rng_k, final_num_proposed, final_num_accepted = final_state_tuple
            # Update Instance State (JAX)
            final_state_info        = (self._states, self._logprobas)
            return final_state_info, samples_tuple, probs
        return wrapped_sampler
    
    def get_sampler_np(self, num_samples: Optional[int] = None, num_chains: Optional[int] = None) -> Callable:
        """
        Returns a NumPy-based sampling function with baked-in configuration.

        This function partially applies the core NumPy sampling logic
        (`_generate_samples_np`) with configuration parameters (like mu, beta,
        step counts, function references) taken from the current sampler instance.

        The returned function has a simplified signature suitable for repeated
        calls where only the initial state and network parameters might change frequently.

        Parameters:
            num_samples (Optional[int]):
                Number of samples per chain to bake into the static function.
                If None, uses `self._numsamples`.
            num_chains (Optional[int]):
                Number of chains to bake into the static function.
                If None, uses `self._numchains`.

        Returns:
            Callable: A NumPy-based function with the signature:
            `wrapped_sampler(states_init, params)`
            It returns the same tuple structure as `_generate_samples_np`:
            `(meta, configs)`

        Raises:
            RuntimeError: If the sampler backend is not 'numpy'.
        """
        if self._isjax:
            raise RuntimeError("NumPy sampler getter only available for NumPy backend.")

        # Use provided sample/chain counts or instance defaults
        static_num_samples  = num_samples if num_samples is not None else self._numsamples
        static_num_chains   = num_chains if num_chains is not None else self._numchains

        # Gather arguments to be baked in (partial application)
        # Configuration values from self
        baked_args = {
            #! Static Config
            "num_samples"               : static_num_samples,
            "num_chains"                : static_num_chains,
            "therm_steps"               : self._therm_steps
        }
        
        def wrapper(states_init, rng_k_init, param, num_proposed_init=None, num_accepted_init=None):
            """
            Wrapper function for NumPy sampling.

            Args:
                states_init: Initial states [num_chains, *shape].
                rng_k_init: Initial RNG key (not used in NumPy).
                params: Network parameters for this run.

            Returns:
                (meta, configs)
            """
            # Call the partially applied core function
            self._states        = states_init
            self._rng_k         = rng_k_init
            self._num_proposed  = num_proposed_init if num_proposed_init is not None else np.zeros(static_num_chains, dtype=DEFAULT_NP_INT_TYPE)
            self._num_accepted  = num_accepted_init if num_accepted_init is not None else np.zeros(static_num_chains, dtype=DEFAULT_NP_INT_TYPE)
            return self.sample(parameters=param, num_samples=baked_args["num_samples"], num_chains=baked_args["num_chains"])
        
        return wrapper

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
        
def get_sampler(typek: Union[str, SamplerType, Sampler], *args, **kwargs) -> Sampler:
    """
    Get a sampler of the given type.
    
    Parameters:
    - typek (str, SamplerType, or Sampler): The type of sampler to get or an existing sampler instance
    - args: Additional arguments for the sampler
    - kwargs: Additional keyword arguments for the sampler
    
    Returns:
    - Sampler: The requested sampler or the provided sampler instance
    
    Raises:
    - ValueError: If the requested sampler type is not implemented
    """
    if isinstance(typek, Sampler):              # is already a sampler instance
        return typek
    elif isinstance(type, MCSampler):           # is a sampler class
        return typek
    elif isinstance(typek, str):                # is a string to convert to enum
        typek = SamplerType.from_str(typek)
    elif typek == SamplerType.MCSampler:        # is a sampler type from the enum
        typek = SamplerType.MCSampler
    else:
        return typek
        # raise ValueError(SamplerErrors.NOT_A_VALID_SAMPLER_TYPE)
    
    # set from the type enum
    if typek == SamplerType.MCSampler:
        return MCSampler(*args, **kwargs)
    
    raise ValueError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

#######################################################################