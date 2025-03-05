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
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend, JIT, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_INT_TYPE
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
    import jax.random as random
    from jax import vmap

#########################################################################
#! Errors
#########################################################################

class SamplerErrors:
    """
    Errors for the Sampler class.
    """
    NOT_GIVEN_SIZE_ERROR        = "The size of the system is not given"
    NOT_IMPLEMENTED_ERROR       = "This feature is not implemented yet"
    NOT_A_VALID_STATE_STRING    = "The state string is not a valid state string"
    NOT_A_VALID_STATE_DISTING   = "The state is not a valid state"
    NOT_IN_RANGE_MU             = "The parameter \mu must be in the range [0, 2]"

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

@JIT
def _propose_random_flip_jax(state: jnp.ndarray, rng, rng_k):
    """
    Propose a random flip of a state using JAX.
    """
    idx = randint_jax(rng_module=rng, key=rng_k, shape=(1,), low=0, high=state.size)[0]
    return Binary.flip_array_jax_spin(state, idx)

@njit
def _propose_random_flip_np(state: np.ndarray, rng):
    
    """
    Propose a random flip of a state using numpy.
    """
    idx = randint_np(rng=rng, low=0, high=state.size, size=1)[0]
    return Binary.flip_array_np(state, idx)

def _propose_random_flip(state: 'array-like', backend = 'default', rng = None, rng_k = None):
    """
    Propose a random flip of a state.
    """
    if backend == np or backend == 'numpy' or backend == 'np':
        return _propose_random_flip_np(state, rng)
    return _propose_random_flip_jax(state, rng, rng_k)

#########################################################################
#! Set the state of the system
#########################################################################

#!TODO: implement the set_state_int and set_state_rand for the fermions

def _set_state_int(state        : int,
            modes               : int                           = 2,
            hilbert             : Optional[HilbertSpace]        = None,
            shape               : Union[int, Tuple[int, ...]]   = (1,),
            mode_repr           : float                         = Binary.BACKEND_REPR,
            backend             : str                           = 'default'
            ):
    '''
    Set the state configuration from the integer representation.
    - state         : state configuration
    - mode_repr    : mode representation (default is 0.5 - for binary spins +-1) 
    
    Transforms the integer to a given configuration 
    Notes:
        The states are given in binary or other representation 
            : 2 for binary
            : 4 for fermions
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
    size = shape if isinstance(shape, int) else np.prod(shape)
    out  = None
    
    ############################################################################
    if hilbert is None:
        if modes == 2:
            # set the state from tensor
            out = Binary.int2base(state, size, backend, spin_value=mode_repr, spin=Binary.BACKEND_DEF_SPIN).reshape(shape)
        elif modes == 4:
            # # first half is up and the second half is down
            # int_left    = state >> (size)              # the left part     - the last size bits   - move the bits to the right by size
            # int_right   = state & ((1 << size) - 1)    # the right part    - the first size bits  - get the modulo of the state by size
            
            # out = np.array([1 if (int_right & (1 << i)) else 0 for i in range(size)] +
            #           [1 if (int_left & (1 << i)) else 0 for i in range(size)],
            #           dtype=np.float32) * mode_repr
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        #!TODO : implement the Hilbert space representation
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    # return the state representation reshaped to the shape provided
    return out.reshape(shape)
    
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
    - modes          : number of modes (default is 2 for binary spins)
    - hilbert        : Hilbert space object (optional)
    - shape          : shape of the system
    - mode_repr     : mode representation (default is 0.5 - for binary spins +-1)
    - backend        : computational backend ('default', 'numpy', or 'jax')
    - rng            : random number generator for numpy
    - rng_key        : random key for jax
    
    Returns: random state array
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(Solver.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    # check the size from the hilbert
    size = shape if isinstance(shape, int) else np.prod(shape)
    
    if hilbert is None:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                ran_state = choice([-1, 1], size, rng=rng, rng_k=rng_key, backend=backend)
            else:
                ran_state = choice([0, 1], size, rng=rng, rng_k=rng_key, backend=backend)
            return ran_state * mode_repr
            
        elif modes == 4:
            ran_state = choice([0, 1], 2 * size, rng=rng, rng_k=rng_key, backend=backend)
            return ran_state * mode_repr
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        # ! TODO : implement the Hilbert space representation
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

def _set_state_up(modes         : int                           = 2,
                    hilbert     : Optional[HilbertSpace]        = None,
                    shape       : Union[int, Tuple[int, ...]]   = (1,),
                    mode_repr   : float                         = 0.5,
                    backend     : str                           = 'default'
                    ):
    '''
    Generate an "all up" state configuration.
    - modes          : number of modes (default is 2 for binary spins)
    - hilbert        : Hilbert space object (optional)
    - shape          : shape of the system
    - mode_repr      : mode representation (default is 0.5 - for binary spins +-1)
    - backend        : computational backend ('default', 'numpy', or 'jax')
    
    Returns: all up state array
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    # check the size from the hilbert
    size = shape if isinstance(shape, int) else np.prod(shape)
    
    # Get the appropriate backend module
    xp = get_backend(backend)
    
    if hilbert is None:
        if modes == 2:
            return xp.ones(size) * mode_repr
        elif modes == 4:
            return xp.array([1] * size + [0] * size) * mode_repr
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        # ! TODO : implement the Hilbert space representation
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    
def _set_state_down(modes       : int                           = 2,
                    hilbert     : Optional[HilbertSpace]        = None,
                    shape       : Union[int, Tuple[int, ...]]   = (1,),
                    mode_repr   : float                         = 0.5,
                    backend     : str                           = 'default'
                    ):
    '''
    Generate an "all down" state configuration.
    - modes          : number of modes (default is 2 for binary spins)
    - hilbert        : Hilbert space object (optional)
    - shape          : shape of the system
    - mode_repr     : mode representation (default is 0.5 - for binary spins +-1)
    - backend        : computational backend ('default', 'numpy', or 'jax')
    
    Returns: all down state array
    '''
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(Solver.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    # check the size from the hilbert
    size = shape if isinstance(shape, int) else np.prod(shape)
    
    # Get the appropriate backend module
    xp = get_backend(backend)
    
    if hilbert is None:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                return xp.ones(size) * (-mode_repr)
            else:
                return xp.zeros(size)
        elif modes == 4:
            return xp.array([0] * size + [1] * size) * mode_repr
        else:
            raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
    else:
        # ! TODO : implement the Hilbert space representation
        raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)

def _set_state_af(modes         : int                           = 2,
                hilbert         : Optional[HilbertSpace]        = None,
                shape           : Union[int, Tuple[int, ...]]   = (1,),
                mode_repr       : float                         = 0.5,
                backend         : str                           = 'default'
                ):
    '''
    Generate an antiferromagnetic state configuration.
    - modes          : number of modes (default is 2 for binary spins)
    - hilbert        : Hilbert space object (optional)
    - shape          : shape of the system
    - mode_repr     : mode representation (default is 0.5 - for binary spins +-1)
    - backend        : computational backend ('default', 'numpy', or 'jax')
    
    Returns: antiferromagnetic state array
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(Solver.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    # check the size from the hilbert
    size = shape if isinstance(shape, int) else np.prod(shape)
    
    # Get the appropriate backend module
    xp = get_backend(backend)
    
    if hilbert is None:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                return xp.array([1 if i % 2 == 0 else -1 for i in range(size)]) * mode_repr
            else:
                return xp.array([1 if i % 2 == 0 else 0 for i in range(size)]) * mode_repr
        elif modes == 4:
            return xp.array(
                [1 if i % 2 == 0 else 0 for i in range(size)] + 
                [0 if i % 2 == 0 else 1 for i in range(size)]
            ) * mode_repr
        else:
            raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
    else:
        # ! TODO : implement the Hilbert space representation
        raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
    
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
    - statetype (int, jnp.ndarray, np.ndarray, str, SolverInitState): The state specification
        * If int: Converts to binary representation using _set_state_int
        * If array: Returns the array directly
        * If str: Converts to SolverInitState enum and processes accordingly
        * If SolverInitState: Processes according to the enum value
    - modes (int)                       : Number of modes for the binary representation
    - hilbert (HilbertSpace, optional)  : Hilbert space representation
    - shape (int or tuple)              : Shape of the system
    - mode_repr (float)                 : Mode representation value (default: Binary.BACKEND_REPR)
    - rng                               : Random number generator for numpy or jax
    - rng_key                           : Random key for JAX
    - backend (str)                     : Computational backend ('default', 'numpy', or 'jax')
    
    Returns:
    - ndarray: The corresponding state configuration
    
    Raises:
    - ValueError: If statetype is not a valid state specification
    """
    if isinstance(statetype, (int, np.integer, jnp.integer)):
        return _set_state_int(statetype, modes, hilbert, shape, mode_repr, backend)
    elif isinstance(statetype, (jnp.ndarray, np.ndarray)):
        return statetype
    elif isinstance(statetype, str):
        try:
            state_enum = SolverInitState.from_str(statetype)
            return _state_distinguish(
                statetype   =   state_enum,
                modes       =   modes,
                hilbert     =   hilbert,
                shape       =   shape,
                mode_repr   =   mode_repr,
                backend     =   backend)
        except ValueError as e:
            raise ValueError(Solver.NOT_A_VALID_STATE_STRING) from e
    elif isinstance(statetype, SolverInitState):
        if statetype == SolverInitState.RND:
            return _set_state_rand(
                modes       =   modes,
                hilbert     =   hilbert,
                shape       =   shape,
                mode_repr   =   mode_repr,
                backend     =   backend,
                rng         =   rng,
                rng_key     =   rng_key
                
            )
        elif statetype == SolverInitState.F_UP:
            return _set_state_up(
                modes       =   modes,
                hilbert     =   hilbert,
                shape       =   shape,
                mode_repr   =   mode_repr,
                backend     =   backend
            )
        elif statetype == SolverInitState.F_DN:
            return _set_state_down(
                modes       =   modes,
                hilbert     =   hilbert,
                shape       =   shape,
                mode_repr   =   mode_repr,
                backend     =   backend
            )
        elif statetype == SolverInitState.AF:
            return _set_state_af(
                modes       =   modes,
                hilbert     =   hilbert,
                shape       =   shape,
                mode_repr   =   mode_repr,
                backend     =   backend
            )
    else:
        raise ValueError(SamplerErrors.NOT_A_VALID_STATE_DISTING)

########################################################################
#! Sampler class for Monte Carlo sampling
########################################################################

class Sampler(ABC):
    """
    A base class for the sampler.     
    """
    
    _ERR_NO_RNG_PROVIDED = "Either rng or seed must be provided"
    
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
        
        # check the backend
        if rng is not None and rng_k is not None:
            self._rng       = rng
            self._rng_k     = rng_k
            self._backend   = get_backend(backend)
        elif seed is not None:
            self._backend, _, (self._rng, self._rng_key) = self.obtain_backend(backend, seed)
        else:
            raise ValueError(Sampler._ERR_NO_RNG_PROVIDED)
        
        # set the backend
        self._isjax         = (not self._backend == np)
        
        # handle the Hilbert space - may control state initialization
        self._hilbert       = hilbert
        
        # handle the states
        self._shape         = shape
        self._size          = np.prod(shape) if isinstance(shape, tuple) else shape
        self._numsamples    = numsamples
        self._numchains     = numchains
        
        # handle the initial state
        self.set_initstate(initstate, **kwargs)
        
        # proposed state
        self._num_proposed  = self._backend.zeros(shape, dtype=self._backend.int64)
        self._num_accepted  = self._backend.zeros(shape, dtype=self._backend.int64)
        
        # handle the update function
        self._upd_fun       = upd_fun
        if self._upd_fun is None:
            if self._isjax:
                # Bind RNG arguments to the JAX updater and then wrap with JIT.
                updater = partial(_propose_random_flip_jax, rng=self._rng, rng_k=self._rng_key)
                self._upd_fun = JIT(updater)
            else:
                # For NumPy backend, bind the RNG to the updater.
                self._upd_fun = partial(_propose_random_flip_np, rng=self._rng)
    
    ###################################################################
    #! ABSTRACT
    ###################################################################
    
    @abstractmethod
    def sample(self, parameters=None, num_samples=None, num_chains=None):
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
        return self._rng_key
    
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
        self._num_rejected = 0
        self._set_chains(self._initstate, self._numchains)
    
    # ---
    
    def set_initstate(self, initstate, **kwargs):
        """
        Set the initial state of the system.
        
        Parameters:
        - initstate (str, int, np.ndarray, jnp.ndarray): The initial state specification
        - kwargs (dict): Additional arguments for state generation
        Raises:
        - NotImplementedError: If the requested state type is not implemented
        Returns:
        - None
        """
        
        # handle the initial state
        if initstate is None or isinstance(initstate, str):
            if self._hilbert is None or True:
                if initstate is None:
                    if self._isjax:
                        self._initstate = jnp.zeros(self._shape, dtype = DEFAULT_JP_FLOAT_TYPE)
                    else:
                        self._initstate = np.zeros(self._shape, dtype = DEFAULT_NP_FLOAT_TYPE)
                else:
                    self._initstate = _state_distinguish(initstate,
                                        modes   =   kwargs.get('modes', 2),
                                        hilbert =   self._hilbert, 
                                        shape   =   self._shape, 
                                        backend =   self._backend, 
                                        rng     =   self._rng, 
                                        rng_key =   self._rng_key)
            else:
                raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
        else:
            if isinstance(initstate, np.ndarray) and self._isjax:
                self._initstate = jnp.array(initstate)
            elif isinstance(initstate, jnp.ndarray) and not self._isjax:
                self._initstate = np.array(initstate)
            else:
                self._initstate = initstate

        self._set_chains(self._initstate, self._numchains)
    
    # ---
    
    def _set_chains(self, initstate: Union[np.ndarray, jnp.ndarray], numchains: Optional[int] = None):
        '''
        Set the chains for the sampler.
        Parameters:
        - initstate (np.ndarray, jnp.ndarray): The initial state
        - numchains (int, optional): The number of chains
        '''
        if numchains is None:
            numchains = self._numchains
        if self._isjax:
            self._states = jnp.stack([jnp.array(initstate)] * numchains, axis=0)
        else:
            self._states = np.stack([np.array(initstate.copy())] * numchains, axis=0)
    
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
        self._set_chains(self._initstate, numchains)
    
#######################################################################    

class MCSampler(Sampler):
    """    
    
    It provides the basic functionality to sample the basis states from the Hilbert space 
    according to the Born distribution.
    
        :math:`p_{\\mu}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`.

    For :math:`\\mu=2` this corresponds to sampling from the Born distribution. \
    :math:`0\\leq\\mu<2` can be used to perform importance sampling             \
    (see `[arXiv:2108.08631] <https://arxiv.org/abs/2108.08631>`_).
    
    It supports multiple Markov chains running in parallel (for batch sampling). In this
    implementation the update proposer (upd_fun) and a network (net) are provided. The sampler
    thermalizes all chains before collecting samples and uses a standard Metropolis-Hastings rule.
    """
    
    def __init__(self, 
                net,
                shape       : Tuple[int, ...],
                upd_fun     : Callable, 
                rng, 
                rng_k                               = None,
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
        
        if self._upd_fun is None:
            if self._isjax:
                self._upd_fun = JIT(partial(_propose_random_flip_jax, rng=self._rng, rng_k=self._rng_key))
            else:
                self._upd_fun = partial(_propose_random_flip_np, rng=self._rng)
                
        # Store the initial logprobability obtained from the net sampler
        self._logprobas     = None
    
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

        log_acceptance_ratio = beta * (candidate_val - current_val)
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

        log_acceptance_ratio = beta * (candidate_val - current_val)
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
    
    def _logprob_jax(self, x, mu: float, net_callable, net_params):
        '''
        Calculate the log probability of a state using JAX.
        Parameters:
            - x             : The state
            - mu            : The parameter mu
            - beta          : The parameter beta
            - net_callable  : The network callable (returns \\text{Re}(\\log\\psi(s)))
            - net_params    : The network parameters
        Returns:
            - The log probability as a float
        '''
        return jax.vmap(lambda y: mu * jnp.real(net_callable(net_params, y)), in_axes=(0,))(x)
    
    def _logprob_np(self, x, mu, net_callable, net_params):
        '''
        Calculate the log probability of a state using NumPy.
        Parameters:
            - x             : The state
            - mu            : The parameter mu
            - beta          : The parameter beta
            - net_callable  : The network callable (returns \\text{Re}(\\log\\psi(s)))
            - net_params    : The network parameters
        Returns:
            - The log probability as a float
        '''
        return np.array([mu * np.real(net_callable(net_params, y)) for y in x])
    
    def logprob(self, x, mu: float = 1.0, net_callable = None, net_params = None):
        '''
        Calculate the log probability of a state.
        Parameters:
            - x             : The state
            - mu            : The parameter mu
            - beta          : The parameter beta
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
                        chain           : 'array-like',
                        logprobas       : 'array-like',
                        rng_k           : 'array-like',
                        num_proposed    : 'array-like',
                        num_accepted    : 'array-like',
                        params          : 'array-like',
                        update_proposer : Callable,
                        log_probability : Callable,
                        accept_config   : Callable,
                        net_callable    : Callable,
                        steps           : int):
        '''
        Carry out the update chain using JAX. It uses JAX's fori_loop
        to iterate over the number of steps.
        Parameters
            - chain : The initial state of the chain (array-like)
            - steps : The number of steps to update the chain (jax.lax.fori_loop)
        Returns
            - The updated chain after the specified number of steps.
        '''
        
        # get the current value of the chain
        # current_val = logprobas if logprobas is not None else net_callable(params, chain)
        current_val = logprobas if logprobas is not None else log_probability(chain, net_callable=net_callable, net_params=params)
        carry       = (chain, current_val, rng_k, num_proposed, num_accepted)
        
        # define the body of the fori_loop
        def body(i, carry):
            '''
            Carry:
                - 0 : The current state of the chain
                - 1 : The current value of the chain
                - 2 : The random key
                - 3 : The number of proposed updates
                - 4 : The number of accepted updates
            '''
            
            # obtain the current key
            chain_in, current_val_in, rng_k_in, num_proposed_in, num_accepted_in = carry
            
            new_rng_keys    = random.split(rng_k_in, num = chain_in.shape[0] + 1)
            carry_key       = new_rng_keys[-1]
            
            # update the chain by proposing a new state via the update_proposer
            new_chain       = jax.vmap(update_proposer, in_axes=(0, 0, None))(new_rng_keys[:-1], chain_in)

            # compute the acceptance probability (it is already partially called on mu and beta)
            logprobas_new   = log_probability(new_chain, net_callable=net_callable, net_params=params)
            # acceptance probability (already partially called on mu and beta)
            acc_probability = accept_config(current_val_in, logprobas_new)
            
            # decide with dice rule
            new_rng_key, carry_key = random.split(carry_key,)
            accepted        = random.bernoulli(new_rng_key, acc_probability).reshape((-1,))
            
            # keep track of the updates
            num_proposed    = num_proposed_in + len(chain_in)
            num_accepted    = num_accepted_in + jax.lax.reduce_sum(accepted)
            
            # accept by jax
            def update(acc, old, new):
                return jax.lax.select(acc, new, old)
            new_carry_states= jax.vmap(update, in_axes=(0, 0, 0))(accepted, chain_in, new_chain)
            new_carry_vals  = jnp.where(accepted, logprobas_new, current_val_in)
            return (new_carry_states, new_carry_vals, carry_key, num_proposed, num_accepted)
        return jax.lax.fori_loop(0, steps, body, carry)

    def _sweep_chain_np(self, 
                        chain           : np.ndarray,
                        logprobas       : np.ndarray,
                        num_proposed    : int,
                        num_accepted    : int,
                        params          : Any,
                        update_proposer : Callable,
                        log_probability : Callable,
                        accept_config   : Callable,
                        net_callable    : Callable,
                        steps           : int):
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
        current_val = logprobas if logprobas is not None else log_probability(chain, net_callable=net_callable, net_params=params)
        for _ in range(steps):
            # For each chain element, generate a new candidate using the update proposer.
            # Here, we simulate splitting rng_k by simply generating a new random integer key.
            new_chain       = np.array([update_proposer(state) for state in chain])
            # Compute new log probabilities for candidates.
            new_logprobas   = log_probability(new_chain, net_callable=net_callable, net_params=params)
            # Compute acceptance probability for each chain element.
            acc_probability = accept_config(current_val, new_logprobas)
            # Decide acceptance by comparing against a random uniform sample.
            rand_vals       = np.random.uniform(size=chain.shape[0])
            accepted        = rand_vals < acc_probability
            num_proposed    += chain.shape[0]
            num_accepted    += np.sum(accepted)
            
            # Update: if accepted, take candidate, else keep old.
            new_state       = np.array([new_chain[i] if accepted[i] else chain[i] for i in range(chain.shape[0])])
            new_val         = np.where(accepted, new_logprobas, current_val)
            # Update the carry:
            chain           = new_state
            current_val     = new_val
            # Update rng_k with a new random integer (not used further)
        return chain, current_val, num_proposed, num_accepted

    def _sweep_chain(self, 
                    chain           : 'array-like',
                    logprobas       : 'array-like',
                    rng_k           : 'array-like',
                    num_proposed    : 'array-like',
                    num_accepted    : 'array-like',
                    params          : 'array-like',
                    update_proposer : Callable,
                    log_probability : Callable,
                    accept_config   : Callable,
                    net_callable    : Callable,
                    steps           : int):
        '''
        Sweep the chain for a given number of steps.
        Parameters:
            - chain : The initial state of the chain (array-like)
            - steps : The number of steps to update the chain (jax.lax.fori_loop)
        Returns:
            - The updated chain after the specified number of steps.
        '''
        if logprobas is None:
            logprobas = self._logprobas
            if logprobas is None:
                logprobas = self.logprob(chain, net_callable=net_callable, net_params=params)

        if self._isjax:
            return self._sweep_chain_jax(chain, logprobas, 
                    rng_k, num_proposed, num_accepted,
                    params, update_proposer, log_probability, accept_config, net_callable, steps)
        return self._sweep_chain_np(chain, logprobas, 
                    num_proposed, num_accepted,
                    params, update_proposer, log_probability, accept_config, net_callable, steps)
    
    ###################################################################
    #! SAMPLING
    ###################################################################
    
    def _get_samples_jax(self,
        shape           : Tuple[int, ...],
        params          : 'array-like',
        num_samples     : int,
        therm_steps     : int,
        sweep_steps     : int,
        states          : 'array-like',
        logprobas       : 'array-like',
        net_callable    : Callable,
        update_proposer : Callable,
        log_probability : Callable,
        accept_config   : Callable,
        rng_k           : 'array-like',
        num_proposed    : 'array-like',
        num_accepted    : 'array-like'):
        
        # thermalize the chains
        states, logprobas, rng_k, num_proposed, num_accepted = self._sweep_chain_jax(
            chain=states, logprobas=logprobas, rng_k=rng_k, 
            num_proposed=num_proposed, num_accepted=num_accepted,
            params=params, update_proposer=update_proposer, log_probability=log_probability, 
            accept_config=accept_config, net_callable=net_callable, steps=therm_steps*sweep_steps)
        
        def scan_fun(carry, i):
            states, logprobas, rng_k, num_proposed, num_accepted = carry
            states, logprobas, rng_k, num_proposed, num_accepted = self._sweep_chain_jax(
                chain=states, logprobas=logprobas, rng_k=rng_k, 
                num_proposed=num_proposed, num_accepted=num_accepted,
                params=params, update_proposer=update_proposer, log_probability=log_probability, 
                accept_config=accept_config, net_callable=net_callable, steps=sweep_steps)
            return (states, logprobas, rng_k, num_proposed, num_accepted), states

        meta, configs = jax.lax.scan(scan_fun, 
                    (states, logprobas, rng_k, num_proposed, num_accepted), None, length=num_samples)
        return meta, configs.reshape((num_samples, -1) + shape)
    
    def _get_samples_np(self, params, numSamples, multipleOf=1):
        """
        NumPy version of obtaining samples via MCMC.
        
        Parameters:
        - params         : Network parameters.
        - numSamples     : Number of sweeps (sample collection iterations) to perform.
        - multipleOf     : (Not used here, but could be used for device distribution.)
        
        Returns:
        A tuple (meta, configs) where:
            meta is a tuple of the final (states, logprobas, rng_k, num_proposed, num_accepted)
            configs is an array of shape (numSamples, -1) + self._shape containing the sampled configurations.
        """
        # Thermalize the chains by sweeping for (therm_steps * sweep_steps)
        states, logprobas, num_proposed, num_accepted = self._sweep_chain_np(
            self._states, self._logprobas, self._num_proposed, self._num_accepted,
            params, self._upd_fun, self.logprob, self.acceptance_probability, self._net_callable,
            self._therm_steps * self._sweep_steps)
        
        # Now perform numSamples sweeps, collecting the resulting states
        meta = []
        configs_list = []
        for i in range(numSamples):
            states, logprobas, num_proposed, num_accepted = self._sweep_chain_np(
                states, logprobas, num_proposed, num_accepted,
                params, self._upd_fun, self.logprob, self.acceptance_probability, self._net_callable,
                self._sweep_steps)
            meta.append((states.copy(), logprobas.copy(), num_proposed, num_accepted))
            # Assume states is of shape (num_chains, *self._shape)
            configs_list.append(states.copy())
        
        # Concatenate configurations along the chain axis
        # Then reshape to (numSamples, -1) + self._shape (flattening the chain dimension per sample)
        configs = np.concatenate(configs_list, axis=0)
        self.globNumSamples = configs.shape[0]
        return (states, logprobas, None, num_proposed, num_accepted), configs.reshape((numSamples, -1) + self._shape)
    
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
            parameters = self._net.get_parameters()
        else:
            self._net.set_parameters(parameters)
            
        net_callable, params = self._set_net_callable(self._net)
        
        # set the log probabilities
        self._logprobas = self.logprob(self._states, 
                                mu=self._mu, beta=self._beta, 
                                net_callable=net_callable, net_params=params)
        
        if self._isjax:
            (self._states, self._logprobas, self._rng_key, self._num_proposed, self._num_accepted), configs =\
                self._get_samples_jax(
                    shape=self._shape, params=params, num_samples=num_samples,
                    therm_steps=self._therm_steps, sweep_steps=self._sweep_steps,
                    states=self._states, logprobas=self._logprobas, net_callable=net_callable,
                    update_proposer=self._upd_fun, log_probability=self.logprob,
                    accept_config=self.acceptance_probability, rng_k=self._rng_key,
                    num_proposed=self._num_proposed, num_accepted=self._num_accepted)
            
            configs_log_ansatz  = jax.vmap(net_callable)(params, configs)
            probs               = jnp.exp((1.0 / self._logprob_fact - self._mu) * jnp.real(configs_log_ansatz))   
            norm                = jnp.sum(probs, axis=1, keepdims=True)
            probs               = probs / norm
            return configs, configs_log_ansatz, probs
        (self._states, self._logprobas, self._rng_key, self._num_proposed, self._num_accepted), configs =\
            self._get_samples_np(params, num_samples, num_chains)
        configs_log_ansatz  = self._net(configs)
        probs               = np.exp((1.0 / self._logprob_fact - self._mu) * np.real(configs_log_ansatz))
        norm                = np.sum(probs, axis=1, keepdims=True)
        probs               = probs / norm
        return configs, configs_log_ansatz, probs
    
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
    if isinstance(typek, str):
        typek = SamplerType.from_str(typek)
    elif typek == MCSampler:
        typek = SamplerType.MCSampler
    else:
        raise ValueError(SamplerErrors.NOT_A_VALID_SAMPLER_TYPE)
    
    if typek == SamplerType.MCSampler:
        return MCSampler(*args, **kwargs)
    
    raise ValueError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

#######################################################################