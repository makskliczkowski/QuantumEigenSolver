#file: Solver/MonteCarlo/parallel.py
# -----------------------------------------------------------------------------

import math
import numpy as np
import scipy as sp
from numba import jit, njit, prange
from typing import Union, Tuple, Union, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from threading import Lock

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

###################################
from Solver.MonteCarlo.montecarlo import MonteCarloSolver, McsTrain, TrainStepVars
###################################

# -----------------------------------------------------------------------------

@unique
class BetaSpacing(Enum):
    """
    Enum for different beta spacing strategies.
    """
    LINEAR      = auto()
    LOGARITHMIC = auto()
    GEOMETRIC   = auto()
    ADAPTIVE    = auto()
    
# -----------------------------------------------------------------------------

class ParallelTempering(ABC):
    """
    Implements Parallel Tempering for Monte Carlo simulations.

    Attributes:
        - solvers       : List of MonteCarloSolver instances.
        - betas         : List of inverse temperature values.
        - nsolvers      : Number of solvers.
        - finished      : List of booleans indicating finished solvers.
        - errors        : List of booleans indicating solvers with errors.
        - losses, mean_losses, std_losses: Containers for tracking losses.
        - best_losses, best_std_losses: Containers for best loss values.
        - accepted, total: Counters per solver.
        - thread_pool   : A ThreadPoolExecutor for parallel training.
        - swap_lock     : A lock to protect swap operations.
    """

    def __init__(self,
                solvers,
                betas   =   None,
                nsolvers=   None,
                spacing :   BetaSpacing = BetaSpacing.LINEAR,
                minbeta :   float = 1e-3,
                maxbeta :   float = 1.0):
        """
        Initialize ParallelTempering.

        Arguments:
            - solvers   : Either a single MonteCarloSolver or a list of them.
            - betas     : Optional list of beta values. If not provided, they are generated.
            - nsolvers  : Number of solvers. If solvers is a single solver, nsolvers must be provided.
            - spacing, minbeta, maxbeta: Parameters used to generate beta values if not provided.
        """
        if isinstance(solvers, MonteCarloSolver):
            if nsolvers is None or nsolvers < 1:
                raise ValueError("nsolvers must be provided and >= 1 when a single solver is given.")
            self._solvers = [solvers]       # List of solvers
            self.replicate(nsolvers)        # Replicate the solver
        elif isinstance(solvers, list):
            if len(solvers) < 1:
                raise ValueError("Solver list must not be empty.")
            self._solvers = solvers          # Otherwise, use the provided list
        else:
            raise TypeError("solvers must be a MonteCarloSolver instance or a list of them.")

        # set the number of solvers
        self._nsolvers = len(self._solvers)

        # Set betas; if none provided, generate them.
        if betas is None:
            self.betas = self.generate_betas(self._nsolvers, spacing, minbeta, maxbeta)
        else:
            if len(betas) != self._nsolvers:
                raise ValueError("The number of betas must match the number of solvers.")
            self.betas = betas

        # Set beta values in each solver
        for i in range(self._nsolvers):
            self._solvers[i].set_beta(self.betas[i])

        # Initialize counters and containers
        self._finished          = [False]   * self._nsolvers
        self._errors            = [False]   * self._nsolvers
        self._accepted          = [0]       * self._nsolvers
        self._total             = [0]       * self._nsolvers
        self._losses            = [[] for _ in range(self._nsolvers)]
        self._mean_losses       = [[] for _ in range(self._nsolvers)]
        self._std_losses        = [[] for _ in range(self._nsolvers)]

        # Best loss tracking 
        self._best_losses       = []
        self._best_std_losses   = []
        self._best_idx          = 0
        self._best_acc_idx      = 0
        self._best_loss         = float('inf')
        self._best_acc          = 0.0

        # Create thread pool and lock for swap
        self.thread_pool        = ThreadPoolExecutor(max_workers=self._nsolvers)
        self.swap_lock          = Lock()
        self._logger            = Logger(logfile=None, append_ts=True)

    def _log(self, msg : str, log : Union[int, str] = Logger.LEVELS_R['info'], lvl : int = 0, color : str = "white", append_msg = True):
        """
        Log a message with optional color and log level.
        """
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        if append_msg:
            msg = f"[HilbertSpace] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log=log, lvl=lvl)
    
    def init_containers(self):
        """
        Initialize containers for losses and other metrics.
        """
        self._losses            = [[] for _ in range(self._nsolvers)]
        self._mean_losses       = [[] for _ in range(self._nsolvers)]
        self._std_losses        = [[] for _ in range(self._nsolvers)]
        self._best_losses       = []
        self._best_std_losses   = []
        self._best_idx          = 0
        self._best_acc_idx      = 0
        self._best_loss         = float('inf')
        self._best_acc          = 0.0
        self._finished          = [False]   * self._nsolvers
        self._errors            = [False]   * self._nsolvers
        self._accepted          = [0]       * self._nsolvers
        self._total             = [0]       * self._nsolvers
    
    # -------------------------------------------------------------------------
    #! Static methods
    # -------------------------------------------------------------------------
    
    @staticmethod
    def generate_betas( n_betas: int,
                        spacing: BetaSpacing = BetaSpacing.LINEAR,
                        minbeta: float = 1e-3,
                        maxbeta: float = 1.0):
        """
        Generate a list of beta values between a specified minimum and maximum value, using a specified spacing method.

        Parameters:
            n_betas (int): The number of beta values to generate.
            spacing (BetaSpacing): The spacing method to use for generating beta values.
                Acceptable values are:
                    - BetaSpacing.LINEAR: Generates values linearly spaced between minbeta and maxbeta.
                    - BetaSpacing.GEOMETRIC: Generates values geometrically spaced between minbeta and maxbeta.
                    - BetaSpacing.LOGARITHMIC: Generates values using a logarithmic scaling between minbeta and maxbeta.
                    - BetaSpacing.ADAPTIVE: Not implemented; raises an error.
            minbeta (float, optional): The minimum beta value. Defaults to 1e-3.
            maxbeta (float, optional): The maximum beta value. Defaults to 1.0.

        Returns:
            List[float]: A list of beta values generated according to the specified spacing.

        Raises:
            NotImplementedError: If spacing is BetaSpacing.ADAPTIVE, since adaptive beta generation is not implemented.
            ValueError: If an unknown beta spacing type is provided.
        """
        if n_betas == 1:
            return [1.0]
        # generate betas based on the specified spacing
        betas = []
        if spacing == BetaSpacing.LINEAR:
            for i in range(n_betas):
                betas.append(minbeta + i * (maxbeta - minbeta) / (n_betas - 1))
        elif spacing == BetaSpacing.GEOMETRIC:
            ratio = (maxbeta / minbeta) ** (1.0 / (n_betas - 1))
            for i in range(n_betas):
                betas.append(minbeta * (ratio ** i))
        elif spacing == BetaSpacing.LOGARITHMIC:
            for i in range(n_betas):
                betas.append(minbeta + (maxbeta - minbeta) * math.log(1.0 + i) / math.log(1.0 + n_betas - 1))
        elif spacing == BetaSpacing.ADAPTIVE:
            raise NotImplementedError("Adaptive beta generation is not implemented yet.")
        else:
            raise ValueError("Unknown beta spacing type")
        return betas

    # -------------------------------------------------------------------------
    #! Replication
    # -------------------------------------------------------------------------
    
    def replicate(self, nsolvers: int):
        """
        Replicate the initial solver to create additional solvers.
        Assumes the solver provides a clone() method.
        
        Parameters:
            nsolvers (int): The number of solvers to create.
        Raises:
            ValueError: If nsolvers is less than 1.
        """
        if nsolvers < 1:
            raise ValueError("nsolvers must be >= 1")
        # Already have one solver in self.solvers
        for i in range(1, nsolvers):
            cloned = self._solvers[0].clone()
            cloned.set_replica_idx(i)
            self._solvers.append(cloned)

    # -------------------------------------------------------------------------
    #! Training
    # -------------------------------------------------------------------------

    def train_step(self,
                iteration       : int,
                train_params    : McsTrain,
                verbose         : bool,
                start_state,
                timer           : Timer = None,
                use_mpi         : bool = False):
        """
        Perform a training step for each solver in parallel.
        Parameters:
            iteration (int)         : The current iteration number.
            train_params (McsTrain) : Training parameters.
            verbose (bool)          : If True, print progress information.
            random_start (bool)     : If True, start with a random configuration.
            timer (Timer, optional) : Timer object for performance measurement.        
        """
        
        if use_mpi:
            try:
                from mpi4py import MPI
            except ImportError:
                raise RuntimeError("MPI is not available. Please install mpi4py or disable use_mpi.")
            comm        = MPI.COMM_WORLD
            rank        = comm.Get_rank()
            size        = comm.Get_size()
            local_start = (self._nsolvers // size) * rank
            if rank == size - 1:
                local_end = self._nsolvers
            else:
                local_end = local_start + (self._nsolvers // size)
        else:
            local_start = 0
            local_end   = self._nsolvers

        futures = []        
        # Submit tasks to the thread pool
        for j in range(local_start, local_end):            
            # Skip finished solvers or those with errors
            if self._finished[j]:
                continue

            def task(idx=j):
                try:
                    train_vars = self._solvers[idx].train_step(i=iteration,
                                            par=train_params, verbose=verbose, start_st=start_state, update=True, timer=timer)
                    return (idx, train_vars)
                except Exception as e:
                    self._log(f"Error in solver {idx}: {e}", log='error', color='red')
                    self._finished[idx]     = True
                    self._errors[idx]       = True
                    return (idx, True)
            futures.append(self.thread_pool.submit(task))

        # Wait for all tasks to complete
        wait(futures, return_when=ALL_COMPLETED)

        # Update counters from each solver
        for fut in futures:
            result = fut.result()
            if result is None:
                continue
            idx, finished_params = result
            # Skip updating if this solver is marked finished.
            if finished_params is None or getattr(finished_params, "finished", False):
                continue
            idx, finished_params    = fut.result()
            self._total[idx]        = self._solvers[idx].total
            self._accepted[idx]     = self._solvers[idx].accepted
            self._losses[idx]       += finished_params.losses
            self._mean_losses[idx]  += finished_params.losses_mean
            self._std_losses[idx]   += finished_params.losses_std

    # -------------------------------------------------------------------------
    #! Swapping
    # -------------------------------------------------------------------------

    def swap(self, i: int, j: int):
        """
        Swap configurations (or beta values) between solvers i and j if the swap is accepted.
        """
        if (i == j or i >= self._nsolvers or j >= self._nsolvers or
                self._finished[i] or self._finished[j] or self._errors[i] or self._errors[j]):
            return

        loss_i = self._solvers[i].lastloss
        loss_j = self._solvers[j].lastloss
        
        # Here we compute a swap acceptance probability based on loss and beta differences.
        temp_scaled_loss_diff   = (loss_i / self.betas[i]) - (loss_j / self.betas[j])
        delta                   = temp_scaled_loss_diff * (self.betas[i] - self.betas[j])
        absprob                 = np.exp(delta)
        if self._solvers[i].random() < absprob:
            with self.swap_lock:
                # Here you can choose to swap only configurations:
                self._solvers[i].swap(self._solvers[j])
                # Alternatively, if you want to swap betas and update counters, do:
                # self.betas[i], self.betas[j] = self.betas[j], self.betas[i]
                # self.solvers[i].set_beta(self.betas[i])
                # self.solvers[j].set_beta(self.betas[j])
                # And swap loss/counter values as needed.
                #!TODO: Implement this part.
                self._log(f"Swapped solvers {i} and {j} with acceptance probability {absprob:.2f}", color='green', lvl='debug')

    def swaps(self):
        """
        Iterate through solvers and perform swap operations.
        """
        i = 0
        while i < self._nsolvers - 1:
            if self._finished[i] or self._errors[i]:
                i += 1
                continue
            # Assume j is the next solver to check
            j = i + 1
            
            # Find next valid solver j
            while j < self._nsolvers and (self._finished[j] or self._errors[j]):
                j += 1
            if j < self._nsolvers and i != j:
                self.swap(i, j)
                i = j
            else:
                break

    # -------------------------------------------------------------------------
    #! Training methods
    # -------------------------------------------------------------------------

    def _update_best_loss(self, idx: int, verbose: bool = False):
        '''
        Update the best loss and accuracy indices.
        '''
        
        if len(self._best_losses) < idx + 1:
            self._best_losses.append(float('inf'))
        if len(self._best_std_losses) < idx + 1:                                                
            self._best_std_losses.append(float('inf'))
        
        best_loss       = float('inf')
        best_std_loss   = float('inf')
        best_loss_t     = float('inf')
        best_std_loss_t = float('inf')
        best_acc        = 0.0
        for j in range(self._nsolvers):
            curr_l_t        = self._solvers[j].lastloss
            curr_l          = np.abs(curr_l_t)
            curr_l_std_t    = self._solvers[j].lastloss_std
            curr_l_std      = np.abs(curr_l_std_t)
            if np.abs(curr_l_t) < best_loss:
                best_loss       = curr_l
                best_loss_t     = curr_l_t
                best_std_loss   = curr_l_std
                best_std_loss_t = curr_l_std_t
                self._best_idx  = j
                self._best_loss = best_loss
                self._best_losses[idx] = best_loss
                self._best_std_losses[idx] = best_std_loss

            if self._total[j] <= 0:
                continue
            
            curr_acc = self._accepted[j] / self._total[j]
            if curr_acc > best_acc:
                self._best_acc = curr_acc
                self._best_acc_idx = j
        if verbose:
            self._log(f"Best loss: {best_loss_t:.4f} ± {best_std_loss_t:.4f} (Solver {self._best_idx})", color='green')
            self._log(f"Best accuracy: {self._best_acc:.4f} (Solver {self._best_acc_idx})", color='green')
            
    # -------------------------------------------------------------------------
    #! Training methods
    # -------------------------------------------------------------------------
        
    def train_single(self, train_params: McsTrain, verbose: bool, rand_start: bool = False, timer: Timer = None):
        """
        Train only the first solver (when only one solver is used).
        Parameters:
            train_params (McsTrain) : Training parameters.
            verbose (bool)          : If True, print progress information.
            state_start             : Initial state for training.
            timer (Timer, optional) : Timer object for performance measurement.
        """
        train_vars = self._solvers[0].train(par=train_params, verbose=verbose, rand_start=rand_start, timer=timer)
        self._mean_losses[0]    = train_vars.losses_mean
        self._std_losses[0]     = train_vars.losses_std
        
    # -------------------------------------------------------------------------

    def train(self, train_params: McsTrain, verbose: bool = False, random_start: bool = False, timer: Timer = None, use_mpi: bool = False):
        """
        Train the Parallel Tempering model.
        If there is only one solver, use train_single.
        Otherwise, run training steps and perform swaps.
        
        Parameters:
            train_params (McsTrain) : Training parameters.
            verbose (bool)          : If True, print progress information.
            random_start (bool)     : If True, start with a random configuration.
            timer (Timer, optional) : Timer object for performance measurement.        
        """
        if self._nsolvers == 1:
            return self.train_single(train_params, verbose, random_start, timer)

        if timer is None:
            timer = Timer()

        train_params.hi()

        # Reset each solver (using nblck from train_params)
        for solver in self._solvers:
            solver.reset(train_params.mcsam)

        # Get the configuration size from the first solver
        config_size = self._solvers[0].size

        # Initialize all the containers
        self.init_containers()

        # Main training loop
        for i in range(1, train_params.MC_sam + 1):
            try:
                self.train_step(i, train_params, verbose, random_start, timer, use_mpi)
            except Exception as e:
                self._log(f"Error in training step {i}: {e}", log='error', color='red')
                break
            
            # Perform swaps every 20% of the total iterations 
            if i % int(config_size / 5) == 0:
                self.swaps()

            self._update_best_loss(i, verbose)

            # If all solvers finished, break early.
            if self.finished:
                self._log("All solvers have finished training.", color='green')
                break
        # Finally, save the best weights
        self._solvers[self._best_idx].save_weights(Directories(train_params.dir, "Weights"), "best_weights")

    # -------------------------------------------------------------------------
    #!Getters
    # -------------------------------------------------------------------------
    
    @property
    def solvers(self):
        '''List of solvers.'''
        return self._solvers
    @property
    def betas(self):
        '''Inverse temperature values.'''
        return self._betas
    @betas.setter
    def betas(self, value):
        self._betas = value
    @property
    def nsolvers(self):
        '''Number of solvers.'''
        return self._nsolvers
    @property
    def finished(self):
        '''Check if all solvers have finished.'''
        return all(self._finished)

    def shutdown(self):
        """Shutdown the thread pool."""
        self.thread_pool.shutdown(wait=True)

    def __del__(self):
        """Ensure thread pool is shut down when object is deleted."""
        self.shutdown()

# -----------------------------------------------------------------------------