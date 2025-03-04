import numpy as np
import scipy as sp
from numba import jit, njit, prange
from typing import Union, Tuple, Union, Callable, Optional

# for the abstract class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique

# from algebra
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
from general_python.algebra.ran_wrapper import choice, randint, uniform
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

#######################################

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

#######################################

class Solver(ABC):
    """
    Abstract base class for the solvers in many body physics. This class is used to define the basic structure of the solvers.
    """
    
    ###################################
    defdir     = "./data"       # default directory for saving the data

    ###################################
    NOT_IMPLEMENTED_ERROR       = "The state is not implemented for the given modes."
    NOT_IMPLEMENTED_ERROR_SAVE  = "The saving is not implemented for the given modes."
    ###################################
    
    def __init__(self,
                size        : int = 1,
                modes       : int = 2,
                seed        : Optional[int] = None,
                hilbert     : Optional[HilbertSpace] = None,
                directory   : Union[str, Directories] = defdir,
                nthreads    : int = 1,
                backend     : str = 'default', 
                **kwargs):
        '''
        Initialize the solver.
        
        Parameters:
        - size          : size of the configuration (like lattice sites etc.)
        - modes         : number of modes for the binary representation
        - hilbert       : Hilbert space representation
        - directory     : directory for saving the data (potentially)
        - seed          : seed for the random number generator
        - backend       : backend for the calculations (default is 'default')
        '''
        self._size          = size                                                          # size of the configuration (like lattice sites etc.)
        self._modes         = modes                                                         # number of modes for the binary representation
        self._hilbert       = hilbert                                                       # Hilbert space representation
        if self._hilbert is not None:
            self._size  = hilbert.ns
            self._modes = hilbert.modes
        elif hilbert is None and size is not None:
            self._hilbert   = HilbertSpace(ns=size, modes=modes)
        elif hilbert is None and size is None:
            raise ValueError("The size of the system is not defined.")
        
        # directory creation
        self._dir           = directory                                                     # directory for saving the data (potentially)
        if not isinstance(directory, Directories):
            self._dir       = Directories(directory)
        self._dir.create_folder(False)
        
        # check the backend
        self._backend, self._backend_sp, (self._rng, self._rng_key) = self.obtain_backend(backend, seed)
        
        # set the precision
        if self._backend == np:
            self._eps   = np.finfo(np.float64).eps
            self._prec  = np.float32
        else:
            self._eps   = jnp.finfo(jnp.float64).eps
            self._prec  = jnp.float32

        # set the current state of the system
        self._currstate     = self._backend.zeros(size * (modes // 2), dtype=self._prec)    # current state of the system
        
        # statistical 
        self._lastloss      = None                                                          # last loss
        self._lastloss_std  = None                                                          # last loss standard deviation
        self._lastloss_mean = None                                                          # last loss mean
        self._lastloss_max  = None                                                          # last loss maximum
        self._lastloss_min  = None                                                          # last loss minimum
        self._currentloss   = None                                                          # current loss
        self._bestloss      = None                                                          # best loss
        
        self._replica_idx   = 1                                                             # replica index
        # initialize threads
        self._nthreads      = nthreads
        
        # allow for preconditioner and scheduler
        self._precond       = kwargs.get('preconditioner', None)
        self._scheduler     = kwargs.get('scheduler', None)
        self._solver        = kwargs.get('solver', None)
        self._optimizer     = kwargs.get('optimizer', None)
        self._early_stop    = kwargs.get('early_stop', None)
        self._arch_params   = kwargs.get('architecture_parameters', None)

    #####################################
    #! PROPERTIES AND GETTERS
    #####################################
    
    @property
    def size(self):
        '''Return the size of the configuration.'''
        return self._size
    
    @property
    def modes(self):
        '''Return the number of modes for the binary representation.'''
        return self._modes
    
    @property
    def hilbert(self):
        '''Return the Hilbert space representation.'''
        return self._hilbert
    
    @property
    def currstate(self):
        '''Return the current state of the system.'''
        return self._currstate
    
    @property
    def lastloss(self):
        '''Return the last loss.'''
        return self._lastloss
    
    @property
    def lastloss_std(self):
        '''Return the last loss standard deviation.'''
        return self._lastloss_std
    
    @property
    def lastloss_mean(self):
        '''Return the last loss mean.'''
        return self._lastloss_mean
    
    @property
    def lastloss_max(self):
        '''Return the last loss maximum.'''
        return self._lastloss_max
    
    @property
    def lastloss_min(self):
        '''Return the last loss minimum.'''
        return self._lastloss_min
    
    @property
    def replica_idx(self):
        '''Return the replica index.'''
        return self._replica_idx
    
    #####################################
    #! BACKEND
    #####################################
    
    @property
    def backend(self):
        '''Return the backend used for calculations.'''
        return self._backend
    
    @property
    def backend_sp(self):
        '''Return the backend (SciPy) used for calculations.'''
        return self._backend_sp
    
    @property
    def rng(self):
        '''Return the random number generator.'''
        return self._rng
    
    @property
    def rng_key(self):
        '''Return the random number generator key.'''
        return self._rng_key
    
    @property
    def random(self):
        '''Return random number'''
        return uniform(shape=(1,), backend=self._backend, rng=self._rng, rng_k=self._rng_key)[0]
    
    # ----------------------------------
    
    def reset_backend(self, backend: str = 'default', seed: Optional[int] = None):
        '''
        Reset the backend for the calculations.
        Parameters:
        - backend       : backend for the calculations (default is 'default')
        - seed          : seed for the random number generator
        '''
        self._backend, self._backend_sp, (self._rng, self._rng_key) = self.obtain_backend(backend, seed)
        self._currstate = self._backend.zeros(self._size * (self._modes // 2), dtype=Solver.prec)
        return self._backend, self._backend_sp, (self._rng, self._rng_key)
    
    @staticmethod
    def obtain_backend(backend: str, seed: Optional[int]):
        '''
        Set the backend for the calculations.
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
        return Solver.obtain_backend(_backendstr, seed)
    
    ###################################
    #! Setters
    ###################################
    
    def set_replica_idx(self, idx: int):
        '''
        Set the replica index.
        '''
        self._replica_idx = idx
        return self._replica_idx
    
    def set_early_stopping(self, *args, **kwargs):
        '''
        Set the early stopping criteria.
        '''
        #!TODO: implement the early stopping criteria
        self._early_stop = kwargs.get('early_stop', None)
        return self._early_stop
    
    def set_optimizer(self, *args, **kwargs):
        '''
        Set the optimizer.
        '''
        #!TODO: implement the optimizer
        self._optimizer = kwargs.get('optimizer', None)
        return self._optimizer
    
    def set_preconditioner(self, *args, **kwargs):
        '''
        Set the preconditioner.
        '''
        #!TODO: implement the preconditioner
        self._preconditioner = kwargs.get('preconditioner', None)
        return self._preconditioner
    
    def set_scheduler(self, *args, **kwargs):
        '''
        Set the scheduler.
        '''
        #!TODO: implement the scheduler        
        self._scheduler = kwargs.get('scheduler', None)
        return self._scheduler
    
    def set_solver(self, *args, **kwargs):
        '''
        Set the solver.
        '''
        #!TODO: implement the solver        
        self._solver = kwargs.get('solver', None)
        return self._solver
    
    ###################################
    #! Set the state of the system
    ###################################
    
    # @abstractmethod
    def _set_state_tens(self, state : Union[jnp.ndarray, np.ndarray], _mode_repr : float = 0.5):
        '''
        Set the state configuration from the tensor.
        - state         : state configuration
        - _mode_repr    : mode representation (default is 0.5 - for binary spins +-1)
        '''
        self._currstate = state

    #! TODO: implement the set_state_int and set_state_rand for the fermions
    def _set_state_int(self, state: int, _mode_repr : float = 0.5):
        '''
        Set the state configuration from the integer representation.
        - state         : state configuration
        - _mode_repr    : mode representation (default is 0.5 - for binary spins +-1) 
        
        Transforms the integer to a given configuration 
        Notes:
            The states are given in binary or other representation 
                : 2 for binary
                : 4 for fermions
                ...
            It uses the mode representation to determine the spin value:
            Examples:
            - spins 1/2 are created as +-0.5 when _mode_repr = 0.5 (default) and _modes = 2.
                    Thus, we need _size to represent the state.
            - fermions are created as 1/-1 when _mode_repr = 1.0 and _modes = 2 
                    and the first are up spins and the second down spins. Thus, we 
                    need 2 * _size to represent the state and we have 0 and ones for the
                    presence of the fermions.
        '''
        if self._hilbert is None:
            if self._modes == 2:
                # set the state from tensor
                self._set_state_tens(Binary.int2base(state, self._size, self.backend, spin_value = _mode_repr), _mode_repr)
            elif self.modes == 4:
                # # first half is up and the second half is down
                # int_left    = state >> (self.size)              # the left part     - the last _size bits   - move the bits to the right by _size
                # # int_right   = state % self.size               # the right part    - the first _size bits  - get the modulo of the state by _size
                # int_right   = state & ((1 << self.size) - 1)    # the right part    - the first _size bits  - get the modulo of the state by _size 
                #                                                 # for size not power of 2
                # self.set_state_tens(jnp.array(
                #             [1 if (int_right & (1 << i)) else 0 for i in range(self.size)] +
                #             [1 if (int_left & (1 << i)) else 0 for i in range(self.size)],
                #             dtype = Solver.prec) * _mode_repr)
                raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        else:
            #!TODO : implement the Hilbert space representation
            raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        
    def _set_state_rand(self, _mode_repr : float = 0.5):
        '''
        Set the state configuration randomly.
        - _mode_repr : mode representation (default is 0.5 - for binary spins +-1)
        '''
        if self.hilbert is None:
            if self.modes == 2:
                ran_state = choice([-1, 1], self.size, rng=self._rng, rng_k=self._rng_key, dtype=self._prec)
                self._set_state_tens(ran_state * _mode_repr)
                
            elif self.modes == 4:
                ran_state = choice([0, 1], 2 * self.size, rng=self._rng, rng_k=self._rng_key, dtype=self._prec)
                self._set_state_tens(ran_state * _mode_repr)
                # ! TODO : this is a specific implementation for the fermions 
                # ! TODO : create a specific basis so that symmetries can be implemented
            else:
                raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        else:
            # ! TODO : implement the Hilbert space representation
            raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
            
    def _set_state_up(self, _mode_repr : float = 0.5):
        '''
        Set the state configuration to all up.
        '''
        if self.hilbert is None:
            if self.modes == 2:
                state = self._backend.ones(self.size, dtype = self._prec) * _mode_repr
                self._set_state_tens(state, _mode_repr)
            elif self.modes == 4:
                state = self._backend.array(  [1 for _ in range(self.size)] +
                                    [0 for _ in range(self.size)],
                                    dtype = self._prec) * _mode_repr
                self._set_state_tens(state, _mode_repr)
            else:
                raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        else:
            # ! TODO : implement the Hilbert space representation
            raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        
    def _set_state_down(self, _mode_repr : float = 0.5):
        '''
        Set the state configuration to all down. 
        '''
        if self.hilbert is None:
            if self.modes == 2:
                state = self._backend.ones(self.size, dtype = self._prec) * (-_mode_repr)
                self._set_state_tens(state, _mode_repr)
            elif self.modes == 4:
                state = self._backend.array(  [0 for _ in range(self.size)] +
                                    [1 for _ in range(self.size)],
                                    dtype = self._prec) * _mode_repr
                self._set_state_tens(state, _mode_repr)
        else:
            # ! TODO : implement the Hilbert space representation
            raise NotImplementedError("The state is not implemented for the given modes.")
    
    def _set_state_af(self, _mode_repr : float = 0.5):
        '''
        Set the state configuration to antiferromagnetic.
        '''
        if self.hilbert is None:
            if self.modes == 2:
                state = self._backend.array([1 if i % 2 == 0 else -1 for i in range(self.size)], dtype = self._prec) * _mode_repr
                self._set_state_tens(state, _mode_repr)
            elif self.modes == 4:
                state = self._backend.array([1 if i % 2 == 0 else 0 for i in range(self.size)]
                                    + [0 if i % 2 == 0 else 1 for i in range(self.size)],
                                    dtype = self._prec) * _mode_repr
                self._set_state_tens(state, _mode_repr)
            else:
                raise NotImplementedError("The state is not implemented for the given modes.")
        else:
            # ! TODO : implement the Hilbert space representation
            raise NotImplementedError("The state is not implemented for the given modes.")
        
    # create a distinguish function
    def _state_distinguish(self, statetype, mode_repr = 0.5):
        """
        Distinguishes the type of the given state and sets the state accordingly.

        Parameters:
        state (int, jnp.ndarray, np.ndarray, str, Solver_init_state): The state to be distinguished. 
            - If an integer, sets the state using `set_state_int`.
            - If a jnp.ndarray or np.ndarray, sets the state using `set_state_tens`.
            - If a string, converts it to an Solver_init_state and sets the state accordingly:
                - 'RND' sets the state to random using `set_state_rand`.
                - 'F_UP' sets the state to up using `set_state_up`.
                - 'F_DN' sets the state to down using `set_state_down`.
                - 'AF' sets the state to antiferromagnetic using `set_state_af`.
            - If an Solver_init_state, sets the state accordingly:
                - Solver_init_state.RND sets the state to random using `set_state_rand`.
                - Solver_init_state.F_UP sets the state to up using `set_state_up`.
                - Solver_init_state.F_DN sets the state to down using `set_state_down`.
                - Solver_init_state.AF sets the state to antiferromagnetic using `set_state_af`.

        Raises:
        ValueError: If the state is not an integer, jnp.ndarray, np.ndarray, or a valid string representing an initial state.
        """
        if isinstance(statetype, int, np.integer, jnp.integer):
            self._set_state_int(statetype, mode_repr)
        elif isinstance(statetype, jnp.ndarray, np.ndarray):
            self._set_state_tens(statetype, mode_repr)
        elif isinstance(statetype, str):
            return self._state_distinguish(statetype)
        elif isinstance(statetype, SolverInitState):
            if statetype == SolverInitState.RND:
                self._set_state_rand(mode_repr)
            elif statetype == SolverInitState.F_UP:
                self._set_state_up(mode_repr)
            elif statetype == SolverInitState.F_DN:
                self._set_state_down(mode_repr)
            elif statetype == SolverInitState.AF:
                self._set_state_af(mode_repr)
        else:
            raise ValueError("The state must be an integer, a jnp.ndarray, or a valid string representing an initial state.")
            
    def set_state(self, state, mode_repr = 0.5, update = True):
        '''
        Set the state (either integer or vector) of the Monte Carlo solver.
        - state     : state of the system
        - mode_repr : mode representation (default is 0.5 - for binary spins +-1)
        - update    : update the current state of the system
        '''
        self._state_distinguish(state, mode_repr)
        #!TODO: handle the update of the current state of the system
        
    ###################################
    #! ABSTRACT METHODS
    ###################################
    
    @abstractmethod
    def clone(self):
        '''
        Clone the solver.
        '''
        pass
    
    @abstractmethod
    def swap(self, other):
        '''
        Swap the state of the solver with another solver.
        '''
        pass
    
########################################