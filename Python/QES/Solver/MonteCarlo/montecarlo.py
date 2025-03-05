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
from general_python.common.flog import get_global_logger, Logger
from general_python.common.timer import Timer
import general_python.common.binary as Binary

# from hilbert
from Algebra.hilbert import HilbertSpace

# JAX imports
if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jax import vmap

###################################
from Solver.solver import Solver, SolverInitState
###################################

@dataclass
class McsTrain:
    """
    McsTrain is a class that encapsulates the parameters and
    methods for performing Monte Carlo training (or simulation).

    Attributes:
        epochs (int)    : Number of epochs. Default is 1 - only relevant when the parameters are variated.
        mcsam (int)     : Number of Monte Carlo steps. Default is 10                - this is for samplers.
        mcth (int)      : Number of Monte Carlo steps to thermalize. Default is 0   - this is for samplers.
        bsize (int)     : Size of a single block. Default is 4                      - this is for samplers - autocorrelation.
        dir (str)       : Directory for saving the data. Default is an empty string - this is for saving the data.
        nflip (int)     : Number of flips for a single Monte Carlo step. Default is 1.
        nrepl (int)     : Number of replicas.
        accepted (int)  : Number of accepted steps.
        total (int)     : Total number of steps.
        acc_rate (float): Acceptance rate.
    """
    epochs  : int = 1           # number of epochs - when the training is performed - outside of the samplers
    mcsam   : int = 10          # number of Monte Carlo Steps - for samplers - inner loop for taking the samples 
    mcth    : int = 0           # number of mcSteps to thermalize - for samplers 
    bsize   : int = 4           # single block size (for autocorrelation) - for samplers (correlation)
    dir     : str = ""          # saving directory for the data
    nflip   : int = 1           # number of flips for a single MC step
    nrepl   : int = 1           # number of replicas
    accepted: int = 0           # number of accepted steps
    total   : int = 0           # total number of steps
    acc_rate: float = 0.0       # acceptance rate

    def hi(self, prefix: str = "Train: ") -> None:
        """
        Prints a detailed overview of the Monte Carlo simulation training configuration.

        This method displays all the relevant parameters used during the Monte Carlo simulation,
        including the number of samples, thermalization steps, block size, epochs, and flip counts.
        It also reports the directory for saving results, the number of replicas, and current metrics
        such as accepted steps, total steps, and the acceptance rate. This detailed output is useful
        for verifying the simulation settings and for debugging purposes.

        Args:
            prefix (str): A string to prepend to the beginning of the output message. Default is "Train: ".
        """
        outstr = (
            f"\nMonte Carlo Simulation Configuration:\n"
            f"  - Number of Monte Carlo Samples      : {self.mcsam}\n"
            f"  - Number of Thermalization Steps     : {self.mcth}\n"
            f"  - Block Size (for autocorrelation)   : {self.bsize}\n"
            f"  - Number of Epochs                   : {self.epochs}\n"
            f"  - Number of Flips per MC Step        : {self.nflip}\n"
            f"  - Saving Directory                   : {self.dir if self.dir else 'Not specified'}\n"
            f"  - Number of Replicas                 : {self.nrepl}\n"
            f"  - Accepted Steps                     : {self.accepted}\n"
            f"  - Total Steps                        : {self.total}\n"
            f"  - Acceptance Rate                    : {self.acc_rate:.4f}\n"
        )
        print(prefix + outstr)
    
###################################

@dataclass
class TrainStepVars:
    """
    TrainStepVars is a class that encapsulates the variables used during a training step.
    """
    losses      : list    = None
    losses_mean : list    = None
    losses_std  : list    = None
    finished    : bool    = False

###################################

class MonteCarloSolver(Solver):
    '''
    Monte Carlo Solver is an abstract class that defines the basic structure of the Monte Carlo solver.
    The class is inherited by the specific Monte Carlo solvers.
    '''
    
    # define the static variables
    # ----------------
    
    def __init__(self,
                sampler,
                seed    : Optional[int]                 = None,
                beta    : Optional[float]               = 1.0,
                replica : Optional[int]                 = 1,
                backend : Optional[str]                 = 'default',
                size    : Optional[int]                 = 1,
                hilbert : Optional[HilbertSpace]        = None,
                modes   : Optional[int]                 = 2,
                dir     : Optional[(str, Directories)]  = None,
                nthreads: Optional[int]                 = 1,
                **kwargs):
        """
        Initializes the Monte Carlo solver with default parameters.
        Parameters:
            - {sampler} (Sampler)   : Sampler object.
            - {seed}    (int)       : Random seed (default is None).
            - {replica} (int)       : Replica index (default is 1).
            - {beta}    (float)     : Inverse temperature beta = 1/T.
            - {rng}     (int)       : Random number generator key
            - {size}    (int)       : Configuration size (like lattice sites etc.)
            - {modes}   (int)       : Number of spin modes (in MB systems 2 for spins etc.)
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
        _nthreads (int)         : Number of threads.
        _hilbert (HilbertSpace) : Hilbert space object.
        _backend (str)          : Backend to use (default is 'default').
        _dir (str)              : Directory for saving the data.
        _rng_key [_rng]         : Random number generator key.
        _info (str)             : Information about the solver.
        """
        
        # call the parent class constructor with the arguments and keyword arguments passed
        super().__init__(size=size, modes=modes, seed=seed, nthreads=nthreads,
                         hilbert=hilbert, backend=backend, dir=dir, **kwargs)
        
        if sampler is None:
            raise ValueError("Sampler is not defined.")
        self._sampler           = sampler       # sampler object - for sampling the states
        
        # define the instance variables
        self._mcparams          = McsTrain(
            epochs      = kwargs.get("epochs", 1),
            mcsam       = kwargs.get("mcsam", 10),
            mcth        = kwargs.get("mcth", 0),
            bsize       = kwargs.get("bsize", 4),
            dir         = kwargs.get("dir", ""),
            nflip       = kwargs.get("nflip", 1),
            nrepl       = kwargs.get("nrepl", 1),
        )
        self._accepted          = 0             # number of accepted steps
        self._total             = 0             # total number of steps
        self._acceptance_rate   = None          # acceptance rate
        
        self._replica_idx       = replica       # replica index
        self._beta              = beta          # inverse temperature beta = 1/T
        
        # information
        self._info              = "Monte Carlo Solver"
        
        # create the logger
        self._logger            = get_global_logger() if self._hilbert is None else self._hilbert.logger
        
        # initialize the solver #!TODO : check whether this is necessary
        self.init()
    
    # ----------------------------------------------------------------------
    #! INFORMATION RELATED
    # ----------------------------------------------------------------------
    
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
    
    def log(self, msg : str, log : Union[int, str] = Logger.LEVELS_R['info'], lvl : int = 0, color : str = "white", append_msg = True):
        """
        Log the message.
        
        Args:
            msg (str)               : The message to log.
            log (Union[int, str])   : The flag to log the message (default is 'info').
            lvl (int)               : The level of the message.
            color (str)             : The color of the message.
            append_msg (bool)       : Flag to append the message.
        """
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        if append_msg:
            msg = f"[{self.__class__}] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log=log, lvl=lvl)

    # ----------------------------------------------------------------------
    #! INITIALIZATION
    # ----------------------------------------------------------------------
    
    def reset(self):
        '''
        Reset the Monte Carlo solver. It resets the losses and other parameters.
        1. Resets the losses, losses_mean, losses_std, totals, and accepteds lists.
        '''
        self._accepted      = 0
        self._total         = 0
    
    def init(self):
        '''
        Initialize the Monte Carlo solver. It initializes the losses and other parameters.
        '''
        self._losses        = []
        self._losses_mean   = []
        self._losses_std    = []
        
    # ----------------------------------------------------------------------
    #! GETTERS AND SETTERS
    # ----------------------------------------------------------------------
    
    def get_state(self):
        '''
        Get the state of the Monte Carlo solver.
        '''
        return self._currstate
    
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

    @property
    def accepted(self):
        '''
        Returns the number of accepted steps.
        '''
        return self._accepted
    
    @property
    def total(self):
        '''
        Returns the total number of steps.
        '''
        return self._total
    
    @property
    def losses(self):
        '''
        Returns the list of losses.
        '''
        return self._losses
    
    @property
    def losses_mean(self):
        '''
        Returns the list of mean losses.
        '''
        return self._losses_mean
    
    @property
    def losses_std(self):
        '''
        Returns the list of standard deviations of losses.
        '''
        return self._losses_std
    
    # ----------------
    
    @beta.setter
    def beta(self, value):
        '''
        Set the inverse temperature beta = 1/T.
        '''
        self._beta = value
    
    def set_beta(self, value):
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
    
    # ----------------------------------------------------------------------
    #! TRAINING
    # ----------------------------------------------------------------------  
    
    def _train_step_impl(self,
                        start_st    : Optional[Union[SolverInitState, int, jnp.ndarray]] = None,
                        par         : Union[McsTrain, dict, None] = None,
                        update      : bool = True, **kwargs):
        '''
        Prepare the training step.

        ''' 
        is_mcs_train            = (par is not None and isinstance(par, McsTrain))
        # set the parameters - if they are passed as a dictionary or as a class or as kwargs
        self._mcparams.mcsam    = par.MC_sam    if is_mcs_train else kwargs.get("mcsam", par.get("mcsam", self._mcparams.mcsam))
        self._mcparams.mcth     = par.MC_th     if is_mcs_train else kwargs.get("mcth", par.get("mcth", self._mcparams.mcth))
        self._mcparams.bsize    = par.bsize     if is_mcs_train else kwargs.get("bsize", par.get("bsize", self._mcparams.bsize))
        self._mcparams.nflip    = par.nflip     if is_mcs_train else kwargs.get("nflip", par.get("nflip", self._mcparams.nflip))
        self._mcparams.epochs   = par.epochs    if is_mcs_train else kwargs.get("epochs", par.get("epochs", self._mcparams.epochs))
        self._mcparams.dir      = par.dir       if is_mcs_train else kwargs.get("dir", par.get("dir", self._mcparams.dir))
        self._mcparams.nrepl    = par.nrepl     if is_mcs_train else kwargs.get("nrepl", par.get("nrepl", self._mcparams.nrepl))
        self._mcparams.dir      = par.dir       if is_mcs_train else kwargs.get("dir", par.get("dir", self._mcparams.dir))
        self._mcparams.accepted = 0
        self._mcparams.total    = 0
        self._mcparams.acc_rate = None
        
        # set the state
        self.set_state(start_st, update=update)
        # return the parameters
        return self._mcparams 
    
    @abstractmethod
    def train_stop(self,
                i           : int   = 0,
                verbose     : bool  = False,
                **kwargs):
        '''
        Determine when to stop training.
        Implementations should define the condition to stop based on the training progress.
        '''
        pass

    @abstractmethod
    def train_step(self,
                i           : int   = 0,
                verbose     : bool  = False,
                start_st    : Optional[Union[SolverInitState, int, jnp.ndarray]] = None,
                par         : Union[McsTrain, dict, None] = None,
                update      : bool  = True,
                timer       : Optional[Timer] = None,
                **kwargs) -> TrainStepVars:
        '''
        Perform a single training step.
        Parameters:
        - par           : Monte Carlo training parameters - those can be also included in kwargs (may be None)
        - verbose       : flag to print the information
        - rand_start    : flag to start from a random configuration
        - update        : flag to update the parameters (if the model needs this)
        
        Returns:
        - TrainStepVars : object containing the variables used during a training step
        '''
        pass
    
    @abstractmethod
    def train(self,
            par         : McsTrain,
            verbose     : bool,
            rand_start  : bool,
            timer       : Optional[Timer] = None,
            **kwargs) -> TrainStepVars:
        '''
        Perform the training.

        Parameters:
        - par         : Monte Carlo training parameters
        - verbose     : flag to print the information
        - rand_start  : flag to start from a random configuration
        - timer       : timer object
        - kwargs      : additional parameters
        
        Returns:
        - TrainStepVars : object containing the variables used during a training step
        '''
    
    # ----------------------------------------------------------------------
    #! Save and load
    # ----------------------------------------------------------------------
    
    def save_weights(self, dir: Union[str, Directories] = None, name: str = "weights"):
        '''
        Save the weights of the model.
        '''
        pass
    
    def load_weights(self, dir: Union[str, Directories] = None, name: str = "weights"):
        '''
        Load the weights of the model.
        '''
        pass
    
########################################
