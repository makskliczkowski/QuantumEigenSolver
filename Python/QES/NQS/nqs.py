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

#########################################

from Solver.MonteCarlo.montecarlo import MonteCarloSolver, McsTrain
from Algebra.Operator.operator import Operator, OperatorFunction
from Algebra.hamil import Hamiltonian

#########################################

class NQS(MonteCarloSolver):
    '''
    Neural Quantum State (NQS) Solver.
    Implements a Monte Carlo-based training method for optimizing NQS models.
    Supports both NumPy and JAX backends for efficiency and flexibility.
    '''
    
    _ERROR_NO_HAMILTONIAN   = "A Hamiltonian must be provided!"
    
    def __init__(self,
                hamiltonian : Hamiltonian,
                hilbert     : Optional[HilbertSpace]    = None,
                size        : int                       = 1,
                lower_states: Optional[list]            = None,
                lower_betas : Optional[list]            = None,
                backend     : str                       = 'default',
                directory   : Optional[str]             = MonteCarloSolver.defdir,
                num_threads : Optional[int]             = 1,
                beta        : float                     = 1,
                seed        : Optional[int]             = None,
                replica     : int                       = 1,
                modes       : int                       = 2,
                nparticles  : Optional[int]             = None,
                **kwargs):
        '''
        Initialize the NQS solver.
        '''
        super().__init__(seed=seed, beta=beta, replica=replica, backend=backend, size=size,
                         hilbert=hilbert, modes=modes, directory=directory, nthreads=num_threads, **kwargs)        
        # set the Hamiltonian
        if hamiltonian is None:
            raise ValueError(self._ERROR_NO_HAMILTONIAN)
        self._hamiltonian   = hamiltonian
        
        # collect the Hilbert space information
        self._nh            = self._hilbert.Nh
        self._nparticles    = nparticles if nparticles is not None else 1
        self._nparticles2   = self._nparticles**2
        self._nvisible      = self._size
        self._nvisible2     = self._nvisible**2
        
        # set the lower states
        if lower_states is not None:
            self._lower_states = NQSLowerStates(lower_states, lower_betas, self)
        else:
            self._lower_states = None
            
        # state modifier
        self._modifier      = None
        
    #####################################
    #! TRAINING OVERRIDES
    #####################################
    
    def train_stop(self, i = 0, verbose = False, **kwargs):
        '''
        Stop the training process.
        '''
        return super().train_stop(i, verbose, **kwargs)
    
    def train_step(self, i = 0, verbose = False, start_st = None, par = None, update = True, timer = None, **kwargs):
        '''
        Perform a single training step.
        '''
        return super().train_step(i, verbose, start_st, par, update, timer, **kwargs)
    
    def train(self, nsteps = 1, verbose = False, start_st = None, par = None, update = True, timer = None, **kwargs):
        '''
        Train the NQS solver for a specified number of steps.
        '''
        return super().train(nsteps, verbose, start_st, par, update, timer, **kwargs)
    
    #####################################
    #! SET STATE
    #####################################
    
    def set_state(self, state: Union[np.ndarray, list], mode_repr: Optional[float] = 0.5, update = True):
        '''
        Set the state of the NQS solver.
        '''
        super().set_state(state, mode_repr, update)
        # handle the state update
        
    #####################################
    #! LOG_PROBABILITY_RATIO
    #####################################
    
    @abstractmethod
    def _log_probability_ratio(self, v1, v2 = None, **kwargs) -> Union[np.float64, jnp.float64, float, complex]:
        '''
        Compute the log probability ratio between two configurations.

        Parameters:
            v1: The initial state vector.
            v2: The secondary state vector (optional). If not provided, the current state is assumed.
            kwargs: Additional arguments for model-specific behavior.

        Returns:
            The log probability ratio as a numeric or complex value.

        Note:
            This method is intended to be implemented in derived classes since the model-specific details vary.
            In the context of the Metropolis-Hastings algorithm, the acceptance probability for a transition s → s'
            is given by:

            A(s', s) = min [1, (P(s') / P(s)) * (r(s' → s) / r(s → s'))]

            where the ratio r(s' → s) can be chosen such that it cancels out non-essential factors (e.g., setting r(s' → s)=1),
            simplifying the acceptance criterion. Typically, one computes:

            P_flip = || <ψ|s'> / <ψ|s> ||^2

            where <ψ|s> represents the wave function amplitude for state s. The derived implementation should compute
            the new <ψ|s'> efficiently.
        '''
        pass
    
    def log_probability_ratio(self, v1, v2 = None, **kwargs) -> Union[np.float64, jnp.float64, float, complex]:
        '''
        Compute the log probability ratio.
        Parameters:
            v1: First vector.
            v2      : Second vector (optional) - if not provided - current
                        state is used.
            kwargs  : Additional arguments.
        '''
        #!TODO: Add the state modifier
        return self._log_probability_ratio(v1, v2, **kwargs) + (0)
    
    def probability_ratio(self, v1, v2 = None, **kwargs):
        '''
        Compute the probability ratio.
        Parameters:
            v1: First vector.
            v2      : Second vector (optional) - if not provided - current state is used.
            kwargs  : Additional arguments.
        '''
        return self._backend.exp(self.log_probability_ratio(v1, v2, **kwargs))
    
    #####################################
    #! STATE MODIFIER
    #!TODO: Add the state modifier
    #####################################
    
    @property
    def modifier(self) -> Union[Operator, OperatorFunction]:
        '''
        Return the state modifier.
        '''
        return self._modifier
    
    @property
    def modified(self) -> bool:
        '''
        Return True if the state is modified, False otherwise.
        '''
        return self._modifier is not None
    
    def unset_modifier(self):
        '''
        Unset the state modifier.
        '''
        self._modifier = None
        self.log("State modifier unset.", log='info', lvl = 2, color = 'blue')
    
    def set_modifier(self, modifier: Union[Operator, OperatorFunction], **kwargs):
        '''
        Set the state modifier.
        '''
        self._modifier = modifier
        self.log(f"State modifier set to {modifier}.", log='info', lvl = 2, color = 'blue')

    #####################################
    #! UPDATES
    #####################################
    
    def update(self, **kwargs):
        '''
        Update the NQS solver after state modification.
        '''
        
    def unupdate(self, **kwargs):
        '''
        Unupdate the NQS solver after state modification.
        '''

    #####################################
    #! WEIGHTS
    #####################################
    
    @abstractmethod
    def set_weights(self, **kwargs):
        '''
        Set the weights of the NQS solver.
        '''
        pass
    
    def update_weights(self, f: Optional[Union['array-like', float]] = None, **kwargs):
        '''
        Update the weights of the NQS solver.
        '''
        pass
    
    def save_weights(self, dir = None, name = "weights"):
        return super().save_weights(dir, name)
    
    def load_weights(self, dir = None, name = "weights"):
        return super().load_weights(dir, name)
    
    #####################################
    #! GRADIENT
    #####################################
    
    
#########################################

class VariationalDerivatives:
    """
    Class to manage derivatives information for variational methods:
    
    Generally, stores the derivatives of the energy with respect to the variational parameters.
    
    \\frac{\\partial E}{\\partial \\theta_i} = \\frac{\\langle\\partial_i \\psi|H|\\psi\\rangle}{\\langle\\psi|\\psi\\rangle} = 
    \\langle E_{\\mathrm{loc}} O^*_i \\rangle - \\langle E_{\\mathrm{loc}} \\rangle \\langle O^*_i \\rangle = 
    \\langle (E_{\\mathrm{loc}} - \\langle E_{\\mathrm{loc}} \\rangle) O^*_i \\rangle
    
    This class provides an interface for computing and retrieving various derivatives
    used in the optimization process of the NQS.
    """
    
    def __init__(self, parent: NQS):
        """
        Initialize the derivatives container.

        Parameters:
            parent (NQS): The parent NQS solver instance.
        """
        self._parent            = parent
        self._derivatives_mean  = None
        self._energies_centered = None
        
    @property
    def parent(self) -> NQS:
        """Return the parent NQS solver instance."""
        return self._parent

class StochasticReconfiguration:
    """
    Class to manage the stochastic reconfiguration process for Neural Quantum State (NQS) solvers.
    
    This class provides an interface for computing and retrieving various derivatives
    used in the optimization process of the NQS.
    """
    
    def __init__(self, parent: NQS):
        """
        Initialize the stochastic reconfiguration container.

        Parameters:
            parent (NQS): The parent NQS solver instance.
        """
        self._parent            = parent
        self._derivatives_mean  = None
        self._energies_centered = None
        
    @property
    def parent(self) -> NQS:
        """Return the parent NQS solver instance."""
        return self._parent


#########################################

class NQSLowerStates:
    """
    Class to manage lower states information for Neural Quantum State (NQS) solvers.

    Lower states are used in both energy and gradient estimations for excited states.
    They are instrumental when modifying the Hamiltonian:
        H' = H + Σ β_i P_i   where   P_i = |f_i><f_i| / ⟨f_i|f_i⟩
    The probability ratios derived from the lower states are utilized to adjust the energy
    estimation and the gradient computation defined as:
        ⟨Δ_k* E_loc⟩ - ⟨Δ_k*⟩ ⟨E_loc⟩ + additional lower state corrections

    This class encapsulates the lower states, their associated penalty betas, and provides
    interface methods to access and manipulate these values in relation to a parent NQS solver.
    """
    #!TODO: Finish!

    def __init__(self,
                lower_states    : list,
                lower_betas     : list,
                parent          : NQS):
        """
        Initialize the lower states container.

        Parameters:
            lower_states (list): List of lower state configurations.
            lower_betas (list): List of penalty beta values corresponding to each lower state.
            parent (NQS): The parent NQS solver instance.
        """
        self._lower_states  = lower_states
        self._lower_betas   = lower_betas
        self._parent        = parent
        self._isset         = bool(lower_states)
        
        # containers for the lower states training
        
    
    @property
    def lower_states(self) -> list:
        """Return the list of lower state configurations."""
        return self._lower_states
    
    @property
    def lower_betas(self) -> list:
        """Return the list of penalty beta values for the lower states."""
        return self._lower_betas
    
    @property
    def parent(self) -> NQS:
        """Return the parent NQS solver instance."""
        return self._parent
    
    @property
    def isset(self) -> bool:
        """Return True if lower states have been set, False otherwise."""
        return self._isset
    
    def __len__(self) -> int:
        """Return the number of lower states."""
        return len(self._lower_states)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, float]:
        """
        Retrieve the lower state configuration and its beta value at a given index.

        Parameters:
            index (int): Index of the lower state.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the lower state configuration and its beta value.
        """
        return self._lower_states[index], self._lower_betas[index]