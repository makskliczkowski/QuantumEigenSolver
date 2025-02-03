from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

# JAX imports
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
    epsilon    = jnp.finfo(jnp.float64).eps         # machine epsilon for float64 - might be changed to float32
    prec       = jnp.float32                        # precision of the calculations - might be changed to float64
    defdir     = "data"                             # default directory for saving the data

    ###################################
    NOT_IMPLEMENTED_ERROR       = "The state is not implemented for the given modes."
    NOT_IMPLEMENTED_ERROR_SAVE  = "The saving is not implemented for the given modes."
    ###################################
    
    def __init__(self, **kwargs):
        '''
        Initialize the solver.
        
        Parameters:
        - size          : size of the configuration (like lattice sites etc.)
        - modes         : number of modes for the binary representation
        '''
        self.size               = kwargs.get("size", 1)                                             # size of the configuration (like lattice sites etc.)
        self.modes              = kwargs.get("modes", 2)                                            # number of modes for the binary representation 
        self.currstate          = jnp.array([0.0 for _ in range(self.size)], dtype = Solver.prec)   # current state of the system
        if "hilbert" in kwargs:
            self.hilbert        = kwargs.get("hilbert", None)                                       # Hilbert space for the system
    
    ###################################
    
    # Set the state of the system
    
    ###################################
    
    @abstractmethod
    def set_state_tens(self, state : jnp.ndarray, _mode_repr : float = 0.5):
        '''
        Set the state configuration from the tensor.
        - state         : state configuration
        - _mode_repr    : mode representation (default is 0.5 - for binary spins +-1)
        '''
        pass
        
    #! TODO: implement the set_state_int and set_state_rand for the fermions
    def set_state_int(self, state : int, _mode_repr : float = 0.5):
        '''
        Set the state configuration from the integer.
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
        if self.hilbert is None:
            if self.modes == 2:
                self.set_state_tens(jnp.array([1 if (state & (1 << i)) else -1 for i in range(self.size)], dtype = Solver.prec) * _mode_repr)
            elif self.modes == 4:
                # first half is up and the second half is down
                int_left    = state >> (self.size)              # the left part     - the last _size bits   - move the bits to the right by _size
                # int_right   = state % self.size               # the right part    - the first _size bits  - get the modulo of the state by _size
                int_right   = state & ((1 << self.size) - 1)    # the right part    - the first _size bits  - get the modulo of the state by _size 
                                                                # for size not power of 2
                self.set_state_tens(jnp.array(
                            [1 if (int_right & (1 << i)) else 0 for i in range(self.size)] +
                            [1 if (int_left & (1 << i)) else 0 for i in range(self.size)],
                            dtype = Solver.prec) * _mode_repr)
                raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        else:
            # ! TODO : implement the Hilbert space representation
            raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        
    def set_state_rand(self, _mode_repr : float = 0.5):
        '''
        Set the state configuration randomly.
        - _mode_repr : mode representation (default is 0.5 - for binary spins +-1)
        '''
        if self.hilbert is None:
            if self.modes == 2:
                self.set_state_tens(jnp.array(jnp.random.choice([-1, 1], self.size),
                                            dtype = Solver.prec) * _mode_repr)
            elif self.modes == 4:
                self.set_state_tens(jnp.array(jnp.random.choice([0, 1], 2 * self.size),
                                            dtype = Solver.prec) * _mode_repr)
                # ! TODO : this is a specific implementation for the fermions 
                # ! TODO : create a specific basis so that symmetries can be implemented
            else:
                raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        else:
            # ! TODO : implement the Hilbert space representation
            raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
            
    def set_state_up(self, _mode_repr : float = 0.5):
        '''
        Set the state configuration to all up.
        '''
        if self.hilbert is None:
            if self.modes == 2:
                self.set_state_tens(jnp.ones(self.size, dtype = Solver.prec) * _mode_repr)
            elif self.modes == 4:
                self.set_state_tens(jnp.array([1 for _ in range(self.size)] +
                                            [0 for _ in range(self.size)],
                                            dtype = Solver.prec) * _mode_repr)
            else:
                raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        else:
            # ! TODO : implement the Hilbert space representation
            raise NotImplementedError(Solver.NOT_IMPLEMENTED_ERROR)
        
    def set_state_down(self, _mode_repr : float = 0.5):
        '''
        Set the state configuration to all down. 
        '''
        if self.hilbert is None:
            if self.modes == 2:
                self.set_state_tens(-jnp.ones(self.size, dtype = Solver.prec) * _mode_repr)
            elif self.modes == 4:
                self.set_state_tens(jnp.array([0 for _ in range(self.size)] 
                                            + [1 for _ in range(self.size)]
                                            , dtype = Solver.prec) * _mode_repr)
        else:
            # ! TODO : implement the Hilbert space representation
            raise NotImplementedError("The state is not implemented for the given modes.")
    
    def set_state_af(self):
        '''
        Set the state configuration to antiferromagnetic.
        '''
        if self.hilbert is None:
            if self.modes == 2:
                self.set_state_tens(jnp.array([1 if i % 2 == 0 else -1 for i in range(self.size)], dtype = jnp.float32))
            elif self.modes == 4:
                self.set_state_tens(jnp.array([1 if i % 2 == 0 else 0 for i in range(self.size)] 
                                    + [0 if i % 2 == 0 else 1 for i in range(self.size)]
                                    , dtype = jnp.float32))
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
        if isinstance(statetype, int):
            self.set_state_int(statetype, mode_repr)
        elif isinstance(statetype, jnp.ndarray):
            self.set_state_tens(statetype, mode_repr)
        elif isinstance(statetype, str):
            return self._state_distinguish(statetype)
        elif isinstance(statetype, SolverInitState): 
            if statetype == SolverInitState.RND:
                self.set_state_rand(mode_repr)
            elif statetype == SolverInitState.F_UP:
                self.set_state_up(mode_repr)
            elif statetype == SolverInitState.F_DN:
                self.set_state_down(mode_repr)
            elif statetype == SolverInitState.AF:
                self.set_state_af(mode_repr)
        else:
            raise ValueError("The state must be an integer, a jnp.ndarray, or a valid string representing an initial state.")
            
    def set_state(self, state, mode_repr = 0.5, update = True):
        '''
        Set the state (either integer or vector) of the Monte Carlo solver.
        - state : state of the system
        - mode_repr : mode representation (default is 0.5 - for binary spins +-1)
        - update : update the current state of the system
        '''    
        self._state_distinguish(state, mode_repr)
    
    ###################################
    