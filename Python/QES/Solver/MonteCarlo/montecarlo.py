import numpy as np
import scipy as sp
from numba import jit, njit, prange
from typing import Union, Tuple, Union, Callable, Optional, Dict, Any

# for the abstract class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique

# from algebra
from general_python.algebra.utils import JAX_AVAILABLE, get_backend
from general_python.algebra.ran_wrapper import choice, randint, uniform
from general_python.common.directories import Directories
from general_python.common.flog import get_global_logger, Logger
from general_python.common.timer import Timer
import general_python.common.binary as Binary

# from hilbert
from Algebra.hilbert import HilbertSpace

# JAX imports
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jax import vmap

###################################
from Solver.solver import Solver
from Solver.MonteCarlo.sampler import Sampler, get_sampler, SolverInitState
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
    mcchain : int = 1           # number of chains - for samplers
    direct  : str = ""          # saving directory for the data
    nflip   : int = 1           # number of flips for a single MC step
    nrepl   : int = 1           # number of replicas
    accepted: int = 0           # number of accepted steps
    total   : int = 0           # total number of steps
    acc_rate: float = 0.0       # acceptance rate

    
    def hi(self, prefix: str = "Train: ") -> None:
        """Prints the configuration."""
        config_str = (
            f"\nMonte Carlo Simulation Configuration:\n"
            f"  - Samples       : {self.mcsam}\n"
            f"  - Thermal Steps : {self.mcth}\n"
            f"  - Block Size    : {self.bsize}\n"
            f"  - Epochs        : {self.epochs}\n"
            f"  - Flips per Step: {self.nflip}\n"
            f"  - Save Directory: {self.direct or 'Not specified'}\n"
            f"  - Replicas      : {self.nrepl}\n"
            f"  - Accepted      : {self.accepted}\n"
            f"  - Total Steps   : {self.total}\n"
            f"  - Acceptance    : {self.acc_rate:.4f}\n"
        )
        print(prefix + config_str)
    
###################################

@dataclass
class McsReturn:
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
    MonteCarloSolver: A base abstract class for Monte Carlo simulation solvers.
    This class defines the foundation for Monte Carlo solvers, providing common functionality
    for sampling, state management, training loops, and statistical tracking. Concrete
    implementations should inherit from this class and implement the abstract methods.
    The Monte Carlo solver works by generating samples according to specified sampling methods,
    tracking acceptance rates, and iteratively optimizing a model through training steps.

    Abstract Methods:
        train_stop: Determines termination condition for the training loop
        train_step: Performs a single Monte Carlo training iteration
        train: Executes the complete Monte Carlo training process
    Methods:
        reset(): Resets the solver's counters and statistical tracking
        set_sampler(): Configures the Monte Carlo sampler
        init(): Initializes the solver's state
        get_state(): Returns the current state of the solver
        set_beta(): Sets the inverse temperature parameter
        log(): Records messages to the logger with appropriate formatting
    Notes:
        - Concrete implementations must provide the training loop logic
        - The solver tracks statistical information about the Monte Carlo process
        - The class supports replica-based parallel simulations
        
    Monte Carlo Solver is an abstract class that defines the basic structure of the Monte Carlo solver.
    The class is inherited by the specific Monte Carlo solvers.
    '''
    
    _ERROR_MSG_SAMPLER  = "Sampler is not defined - it is necessary for the Monte Carlo solver."
    _ERROR_MSG_HILBERT  = "Hilbert space is not defined - it is necessary for the Monte Carlo solver."
    _ERROR_MSG_SHAPE    = "Shape is not defined - it is necessary for the Monte Carlo solver."
    
    # define the static variables
    # ----------------
    
    def __init__(self,
                sampler     : Sampler,
                seed        : Optional[int]                 = None,
                beta        : Optional[float]               = 1.0,
                mu          : Optional[float]               = 2.0,
                replica     : Optional[int]                 = 1,
                shape       : Optional[int]                 = 1,
                hilbert     : Optional[HilbertSpace]        = None,
                modes       : Optional[int]                 = 2,
                directory   : Optional[str]                 = None,
                nthreads    : Optional[int]                 = 1,
                backend     : Optional[str]                 = 'default',
                **kwargs):
        """
        Initializes the Monte Carlo solver with default parameters.
        Parameters:
            - {sampler} (Sampler)   : Sampler object.
            - {seed}    (int)       : Random seed (default is None).
            - {beta}    (float)     : Inverse temperature beta = 1/T.
            - {mu}      (float)     : Modification of the probability distribution.
            - {replica} (int)       : Replica index (default is 1).
            - {shape}   (int)       : Shape of the system.
            - {hilbert} (HilbertSpace): Hilbert space object.
            - {directory} (str)     : Directory for saving the data.
            - {nthreads} (int)      : Number of threads.
            - {backend} (str)       : Backend to use (default is 'default').
            - {modes}   (int)       : Number of spin modes (in MB systems 2 for spins etc.)
            - {upd_fun} (Callable)  : Update function for the sampler - if None, the default is used.
        """
        
        # call the parent class constructor with the arguments and keyword arguments passed
        super().__init__(shape      =   shape,
                        modes       =   modes,
                        seed        =   seed,
                        hilbert     =   hilbert,
                        directory   =   directory,
                        nthreads    =   nthreads,
                        backend     =   backend,
                        **kwargs)
        
        # define the instance variables
        self._mcparams          = McsTrain(
                            epochs      = kwargs.get("epochs", 1),
                            mcsam       = kwargs.get("mcsam", 10),
                            mcth        = kwargs.get("mcth", 0),
                            mcchain     = kwargs.get("mcchain", 1),
                            bsize       = kwargs.get("bsize", 4),
                            nflip       = kwargs.get("nflip", 1),
                            nrepl       = kwargs.get("nrepl", 1),
                            direct      = directory,
                            )
        
        if sampler is None:
            raise ValueError(self._ERROR_MSG_SAMPLER)
        
        # for the replica and Monte Carlo process
        self._replica_idx       = replica       # replica index
        self._beta              = beta          # inverse temperature beta = 1/T
        self._mu                = mu            # modification of the probability distribution
        self._sampler           = self.set_sampler(sampler, kwargs.get("upd_fun", None))
        
        self._accepted          = 0             # number of accepted steps
        self._total             = 0             # total number of steps
        self._acceptance_rate   = None          # acceptance rate
        
        # information
        self._info              = "a general Monte Carlo Solver"
        
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

        return f"[{self._replica_idx}] I am a {self.__class__.__name__} object with {self._info} at β = {self._beta : .2f}."
    
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
        self.init()
    
    def set_sampler(self, sampler: Union[Sampler, str], upd_fun: Optional[Callable] = None) -> Sampler:
        '''
        Set the sampler for the Monte Carlo solver.
        '''
        self._sampler = get_sampler(sampler,
                            shape       =   self._shape,
                            upd_fun     =   upd_fun,
                            rng         =   self._rng,
                            rng_k       =   self._rngJAX_RND_DEFAULT_KEY,
                            hilbert     =   self._hilbert,
                            numsamples  =   self._mcparams.mcsam,           #!TODO: this can be modified later
                            numchains   =   self._mcparams.mcchain,         #!TODO: this can be modified later
                            )
        # set the sampler properties
        if hasattr(self._sampler, "set_beta"):
            self._sampler.set_beta(self._beta)
        if hasattr(self._sampler, "set_mu"):
            self._sampler.set_mu(self._mu)
        return self._sampler
    
    # ----------------
    
    def init(self):
        '''
        Initialize the Monte Carlo solver. It initializes the losses and other parameters.
        '''
        self._losses        = []
        self._losses_mean   = []
        self._losses_std    = []
    
    def _update_mcparams(self,
                        par         : Union[McsTrain, Dict[str, Any], None],
                        extra_kwargs: Dict[str, Any]) -> None:
        """
        Update Monte Carlo training parameters from either a dataclass, dict, or kwargs.
        Parameters:
            - par         : Monte Carlo training parameters (McsTrain or dict)
            - extra_kwargs: Additional keyword arguments for parameters.
        1. Initializes a list of keys to update.
        2. Iterates through each key and updates the corresponding parameter.
        3. If the parameter is not provided, it uses the value from extra_kwargs or the default value.
        4. Updates the _mcparams attribute with the new parameters.
        """
        keys            = ['mcsam', 'mcth', 'bsize', 'nflip', 'epochs', 'directory', 'nrepl']
        updated_params  = {}
        for key in keys:
            if par is not None:
                if isinstance(par, McsTrain):
                    updated_params[key] = getattr(par, key, getattr(self._mcparams, key))
                elif isinstance(par, dict):
                    updated_params[key] = par.get(key, getattr(self._mcparams, key))
            else:
                updated_params[key] = extra_kwargs.get(key, getattr(self._mcparams, key))
        self._mcparams = McsTrain(**updated_params)
    
    # ----------------------------------------------------------------------
    #! GETTERS AND SETTERS
    # ----------------------------------------------------------------------
    
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
        return self._replica_idx
    
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
        self._replica_idx = value
    
    # ----------------
    
    @property
    def isjax(self):
        '''
        Returns True if the backend is JAX.
        '''
        return self._isjax
    
    @property
    def shape(self):
        '''
        Returns the shape of the system.
        '''
        return self._shape
    
    # ----------------------------------------------------------------------
    #! TRAINING
    # ----------------------------------------------------------------------  
    
    # @abstractmethod
    def train_stop(self,
                i           : int                           = 0,
                par         : Union[McsTrain, dict, None]   = None,
                curloss     : Optional[float]               = None,
                curstd      : Optional[float]               = None,
                verbose     : bool                          = False,
                **kwargs) -> bool:
        '''
        Determine when to stop training.
        Implementations should define the condition to stop based on the training progress.
        '''
        pass

    # @abstractmethod
    def train_step(self,
                i           : int   = 0,
                verbose     : bool  = False,
                start_st    : Optional[Union[SolverInitState, int]] = None,
                par         : Union[McsTrain, dict, None] = None,
                update      : bool  = True,
                timer       : Optional[Timer] = None,
                **kwargs) -> 'McsReturn':
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
    
    # @abstractmethod
    def train(self,
            par         : McsTrain,
            verbose     : bool,
            rand_start  : bool,
            timer       : Optional[Timer] = None,
            **kwargs) -> 'McsReturn':
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
    
    def save_weights(self, directory: Union[str, Directories] = None, name: str = "weights"):
        '''
        Save the weights of the model.
        '''
        pass
    
    def load_weights(self, directory: Union[str, Directories] = None, name: str = "weights"):
        '''
        Load the weights of the model.
        '''
        pass


########################################
