from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union
import jax.numpy as jnp
import numpy as np

# import solver specific libraries
from ..solver import SolverInitState, Solver

###################################

@dataclass
class McsTrain:
    """
    McsTrain is a class that encapsulates the parameters and
    methods for performing Monte Carlo training (or simulation).

    Attributes:
        epochs (int)    : Number of epochs. Default is 1 - only relevan when the parameters are variated.
        MC_sam (int)    : Number of Monte Carlo steps. Default is 10                - this is for samplers.
        MC_th (int)     : Number of Monte Carlo steps to thermalize. Default is 0   - this is for samplers.
        bsize (int)     : Size of a single block. Default is 4                      - this is for samplers - autocorrelation.
        dir (str)       : Directory for saving the data. Default is an empty string - this is for saving the data.
        nFlip (int)     : Number of flips for a single Monte Carlo step. Default is 1.
    """
    epochs      : int = 1       # number of epochs - when the training is performed
    mcsam       : int = 10      # number of Monte Carlo Steps - for samplers
    mcth        : int = 0       # number of mcSteps to thermalize - for samplers 
    bsize       : int = 4       # single block size (for autocorrelation) - for samplers
    dir         : str = ""      # saving directory for the data
    nflip       : int = 1       # number of flips for a single MC step
    nrepl       : int = 1       # number of replicas #!TODO : how to handle this?

    # ----------------
    
    def hi(self, prefix: str = "Train: ") -> None:
        """
        Prints a formatted string containing Monte Carlo simulation parameters.

        Args:
            prefix (str): A prefix string to be added at the beginning of the output. Default is "Train: ".

        Returns:
            None
        """
        outstr =    f"Monte Carlo Samples={self.mcsam}, Thermalization Steps={self.mcth}, "     \
                    f"Size of the single block={self.bsize}, "                                  \
                    f"Number of epochs={self.epochs}, "                                         \
                    f"Number of flips taken at each step={self.nflip}"
        print(f"{prefix}{outstr}")

    # ----------------
    
###################################

class MonteCarloSolver(Solver, ABC):
    '''
    Monte Carlo Solver is an abstract class that defines the basic structure of the Monte Carlo solver.
    The class is inherited by the specific Monte Carlo solvers.
    '''
    
    # define the static variables
    # ----------------
    
    def __init__(self, **kwargs):
        """
        Initializes the Monte Carlo solver with default parameters.
        Parameters:
        kwargs:
            - {replica} (int)   : Replica index (default is 1).
            - {beta}    (float) : Inverse temperature beta = 1/T.
            - {rng}     (int)   : Random number generator key
            - {size}    (int)   : Configuration size (like lattice sites etc.)
            - {modes}   (int)   : Number of spin modes (in MB systems 2 for spins etc.)
        **kwargs                : Arbitrary keyword arguments.
        Instance Variables:
        _accepted (int)         : Number of accepted steps.
        _total (int)            : Total number of steps.
        _acceptance_rate (float): Acceptance rate.
        _current_loss (float)   : Current loss value.
        _last_loss (float)      : Last loss value.
        _last_std_loss (float)  : Last standard deviation of the loss.
        _best_loss (float)      : Best loss value.
        _replica (int)          : Replica index (default is 1).
        _beta (float)           : Inverse temperature beta = 1/T.
        _rng_key [_rng]         : Random number generator key.
        _info (str)             : Information about the solver.
        """
        
        # call the parent class constructor with the arguments and keyword arguments passed
        super().__init__(size = kwargs.get("size", 1), modes = kwargs.get("modes", 2))
        
        # define the instance variables
        self._accepted          = 0            # number of accepted steps
        self._total             = 0            # total number of steps          
        self._acceptance_rate   = 0.0          # acceptance rate
        
        # define the loss related variables
        self._current_loss      = 0.0          # current loss value
        self._last_loss         = 0.0          # last loss value
        self._last_std_loss     = 0.0          # last standard deviation of the loss
        self._best_loss         = 1e10         # best loss value
        
        # temperature related variables
        self._replica           = kwargs.get("replica", 1)  # replica index
        self._beta              = kwargs.get("beta", 1.0)   # inverse temperature beta = 1/T
        
        # define the random number generator
        self._rng_key           = kwargs.get("rng", jnp.random.PRNGKey(0))
        
        # information
        self._info              = "Monte Carlo Solver"
        
        # container for the losses - this will be used throughout the training and may be a tensor eventually
        self._losses            = []            # serves for the placeholder
        # initialize the solver #!TODO : check whether this is necessary
        self.init()
    
    # INFORMATION RELATED ##################################################
    
    def __repr__(self):
        """
        Return a string representation of the object.

        The string includes the replica information, class name, additional info, 
        and the beta value formatted to two decimal places.

        Returns:
            str: A formatted string representing the object.
        """

        return f"[{self._replica}] I am a {self.__class__.__name__} object with {self._info} at β = {self._beta : .2f}."
    
    # ----------------
    
    def __str__(self):
        return self._info
    
    # ----------------
    
    @property
    def info(self):
        """
        Returns the information stored in the _info attribute.

        Returns:
            Any: The information stored in the _info attribute.
        """
        return self._info
    
    # ----------------
    
    @info.setter
    def info(self, value : str):
        """
        Sets the information stored in the _info attribute.

        Parameters:
        value : The information to be stored in the _info attribute.
        """
        self._info = value   
            
    # ----------------
    
    # INITIALIZATION ######################################################
    
    @abstractmethod
    def init(self):
        '''
        Initialize the Monte Carlo solver.
        '''
    
    # GETTERS #############################################################
    
    @abstractmethod
    def get_state(self):
        '''
        Get the state of the Monte Carlo solver.
        '''
    
    @property
    def beta(self):
        '''
        Returns the inverse temperature beta = 1/T.
        '''
        return self._beta
    
    @property
    def replica(self):
        '''
        Returns the replica index.
        '''
        return self._replica
    
    @property
    def acc_rate(self):
        '''
        Returns the acceptance rate. 
        - If the acceptance rate is not set, then it is calculated as the ratio of accepted to total steps.
        '''
        return self._acceptance_rate if self._acceptance_rate is not None else (self._accepted / self._total)

    # ----------------
    
    @beta.setter
    def beta(self, value):
        '''
        Set the inverse temperature beta = 1/T.
        '''
        self._beta = value
    
    # ----------------
    
    @replica.setter
    def replica(self, value):
        '''
        Set the replica index.
        '''
        self._replica = value
        
    # TRAINING ############################################################
    
    def _train_step_impl(self, start_st : SolverInitState | int | jnp.ndarray | None = None, 
                        par         : McsTrain | dict | None = None, 
                        update      : bool = True, **kwargs):
        '''
        Prepare the training step.

        '''
        # check whether correct parameters are passed
        self._total             = 0
        self._accepted          = 0
        self._acceptance_rate   = 0
        is_mcs_train            = (par is not None and isinstance(par, McsTrain))
        # set the parameters - if they are passed as a dictionary or as a class or as kwargs
        mcs                     = par.MC_sam    if is_mcs_train else kwargs.get("mcsam", par.get("mcsam", 1))                     
        mct                     = par.MC_th     if is_mcs_train else kwargs.get("mcth", par.get("mcth", 0))
        bsize                   = par.bsize     if is_mcs_train else kwargs.get("bsize", par.get("bsize", 4))
        nflip                   = par.nflip     if is_mcs_train else kwargs.get("nflip", par.get("nflip", 1))
        epochs                  = par.epochs    if is_mcs_train else kwargs.get("epochs", par.get("epochs", 1))
        directory               = par.dir       if is_mcs_train else kwargs.get("dir", par.get("dir", 1))
        
        # set the state
        self.set_state(start_st)
        
        # set the parameters
        return McsTrain(epochs  = epochs,
                        mcsam   = mcs,
                        mcth    = mct,
                        bsize   = bsize,
                        nflip   = nflip,
                        dir     = directory)
        
    @abstractmethod 
    def train_step(self, i      : int = 0,
                    verbose     : bool = False,
                    start_st    : SolverInitState | int | jnp.ndarray | None = None,
                    par         : McsTrain | dict | None = None,
                    update      : bool = True,
                    **kwargs):
        '''
        Perform a single training step.
        Parameters:
        - par           : Monte Carlo training parameters - those can be also included in kwargs (may be None)
        - verbose       : flag to print the information
        - rand_start    : flag to start from a random configuration
        - update        : flag to update the parameters (if the model needs this)
        '''
        pass
            
    @abstractmethod
    def train(self, par : McsTrain, verbose : bool, rand_start : bool, **kwargs):
        '''
        Perform the training.
        
        '''
        