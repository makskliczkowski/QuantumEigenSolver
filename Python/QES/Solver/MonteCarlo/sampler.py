import numpy as np
import scipy as sp
from numba import jit, njit, prange
from typing import Union, Tuple, Union, Callable, Optional

# for the abstract class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique

# from algebra
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend, JIT
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

########################################################################

class Sampler(ABC):
    """
    A base class for the sampler. 
    
    It provides the basic functionality to sample the basis states from the Hilbert space 
    according to the Born distribution.
    
        :math:`p_{\\mu}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`.

    For :math:`\\mu=2` this corresponds to sampling from the Born distribution. \
    :math:`0\\leq\\mu<2` can be used to perform importance sampling             \
    (see `[arXiv:2108.08631] <https://arxiv.org/abs/2108.08631>`_).
    
    
    
    """
    
    def __init__(self,
                shape       : Tuple[int, ...],
                upd_fun     : Callable,
                rng,
                rng_k                                   = None,
                seed        : Optional[int]             = None,
                hilbert     : Optional[HilbertSpace]    = None,
                numsamples  : int                       = 1,
                numchains   : int                       = 1,
                initstate   : Union[np.ndarray, jnp.ndarray] = None,
                mu          : float                     = 2.0,
                beta        : Optional[float]           = 1.0,
                backend     : str                       = 'default',
                ):
        
        # check the backend
        if rng is not None and rng_k is not None:
            self._rng = rng
            self._rng_k = rng_k
        else:
            raise ValueError("rng and rng_k must be provided")
        
        # set the backend
        self._backend   = get_backend(backend)
        self._isjax     = not self._backend == np
        
        self._hilbert   = hilbert
        
        # handle the states
        self._shape     = shape
        self._numsamples= numsamples
        # handle the initial state