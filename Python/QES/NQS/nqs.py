import numpy as np
import inspect
import numba

# typing and other imports
from typing import Union, Tuple, Union, Callable, Optional
from math import isclose
from functools import partial

# for the abstract class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique

# from general_python imports
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
from general_python.algebra.ran_wrapper import choice, randint, uniform
from general_python.common.directories import Directories
import general_python.ml.networks as Networks 
import general_python.common.binary as Binary

# from hilbert
from Algebra.hilbert import HilbertSpace

# JAX imports
if _JAX_AVAILABLE:
    import jax
    from jax import jit as jax_jit, grad, vmap, random
    from jax import numpy as jnp
    from jax.tree_util import tree_flatten, tree_unflatten, tree_map
    from jax.flatten_util import ravel_pytree

    # use flax
    import flax
    import flax.linen as nn
    from flax.core.frozen_dict import freeze, unfreeze
    
# try to import autograd for numpy
try:
    import autograd.numpy as anp
    from autograd import grad as np_grad
    from autograd.misc.flatten import flatten_func
    AUTOGRAD_AVAILABLE = True
except ImportError:
    AUTOGRAD_AVAILABLE = False
    
#########################################

from Solver.MonteCarlo.montecarlo import MonteCarloSolver, McsTrain, McsReturn, Sampler
from Algebra.Operator.operator import Operator, OperatorFunction

# Hamiltonian imports
from Algebra.hamil import Hamiltonian

#########################################

# for the gradients and stuff
import NQS.nqs_utils as NQSUtils

#########################################

def apply_to_all(func, states):
    """
    Apply a function to all elements in a list of states.

    Parameters:
        func (callable): The function to apply.
        states (list): List of states to apply the function to.

    Returns:
        list: List of results after applying the function.
    """
    return [func(state) for state in states]

def apply_to_all_np(func, states):
    """
    Apply a function to all elements in a list of states using NumPy.

    Parameters:
        func (callable): The function to apply.
        states (list): List of states to apply the function to.

    Returns:
        list: List of results after applying the function.
    """
    return np.array([func(state) for state in states])

#########################################

class NQS(MonteCarloSolver):
    '''
    Neural Quantum State (NQS) Solver.
    Implements a Monte Carlo-based training method for optimizing NQS models.
    Supports both NumPy and JAX backends for efficiency and flexibility.
    '''
    
    _ERROR_NO_HAMILTONIAN   = "A Hamiltonian must be provided!"
    _ERROR_HAMILTONIAN_TYPE = "Hamiltonian must be either a Hamiltonian class or a callable function!"
    _ERROR_HAMILTONIAN_ARGS = "Hamiltonian function must accept a state vector only!"
    _ERROR_ALL_DTYPE_SAME   = "All weights must have the same dtype!"
    
    _TOL_HOLOMORPHIC        = 1e-14
    
    def __init__(self,
                net         : Union[Callable, str, nn.Module],
                sampler     : Union[Callable, str, Sampler],
                hamiltonian : Hamiltonian,
                lower_states: Optional[list]            = None,
                lower_betas : Optional[list]            = None,
                nparticles  : Optional[int]             = None,
                seed        : Optional[int]             = None,
                beta        : float                     = 1,
                mu          : float                     = 0,
                replica     : int                       = 1,
                shape       : Union[list, tuple]        = (1,),
                hilbert     : Optional[HilbertSpace]    = None,
                modes       : int                       = 2,
                directory   : Optional[str]             = MonteCarloSolver.defdir,
                backend     : str                       = 'default',
                nthreads    : Optional[int]             = 1,
                **kwargs):
        '''
        Initialize the NQS solver.
        '''
        super().__init__(sampler    =   sampler, 
                        seed        =   seed, 
                        beta        =   beta, 
                        mu          =   mu, 
                        replica     =   replica,
                        shape       =   shape, 
                        hilbert     =   hilbert, 
                        modes       =   modes, 
                        directory   =   directory, 
                        backend     =   backend, 
                        nthreads    =   nthreads)
        
        # pre-set the Hamiltonian
        if hamiltonian is None:
            raise ValueError(self._ERROR_NO_HAMILTONIAN)
        self._hamiltonian   = hamiltonian
        
        #######################################
        #! collect the Hilbert space information
        #######################################
        self._nh            = self._hilbert.Nh if self._hilbert is not None else None
        self._nparticles    = nparticles if nparticles is not None else self._size
        self._nvisible      = self._size
        self._nparticles2   = self._nparticles**2
        self._nvisible2     = self._nvisible**2
        
        #######################################
        #! set the lower states
        #######################################
        
        if lower_states is not None:
            self._lower_states = NQSLowerStates(lower_states, lower_betas, self)
        else:
            self._lower_states = None
        
        #######################################
        #! state modifier (for later)
        #######################################
        self._modifier      = None
        
        #######################################
        #! handle the network
        #######################################
        self._initialized   = False
        self._weights       = None
        self._dtype         = None
        self._net           = self._choose_network(net, **kwargs)   # initialize network type
        self.init_network(self._backend.ones(self._shape))          # run the network
        self._init_gradients()
        self._init_functions()
    
    #####################################
    #! NETWORK
    #####################################
    
    def _choose_network(self, net, **kwargs) -> Networks.GeneralNet:
        '''
        Initialize the variational parameters ansatz via the provided network - it simply creates
        the network instance. To truly initialize the network, use the init_network method.
        Parameters:
            net     : The network to be used (can be a string or a callable).
            kwargs  : Additional arguments for the network.
        Returns:
            The initialized network.
        '''
        if issubclass(type(net), nn.Module):
            self.log(f"Network {net} provided from the flax module.", log='info', lvl = 2, color = 'blue')
        return Networks.choose_network(network_type=net, input_shape=self._shape, backend=self._backend, dtype=self._dtype, **kwargs)
    
    #####################################
    #! INITIALIZATION OF THE NETWORK AND FUNCTIONS
    #####################################
    
    def _check_holomorphic(self, s) -> bool:
        """
        Check if the network provided is holomorphic.

        A network is considered holomorphic if the gradient of its real part equals 
        i times the gradient of its imaginary part. This method computes the gradients 
        of the real and imaginary parts of the network's output with respect to the 
        network parameters, flattens these gradients, and checks if they are equal 
        (up to a small tolerance) when combined appropriately.

        Parameters
        ----------
        s : array-like
            The state vector, assumed to have at least the shape such that s[0, 0, ...]
            is valid.

        Returns
        -------
        bool
            True if the holomorphic condition is met, False otherwise.
        """

        # Extract the sample state for gradient computation.
        sample_state    = s[0, 0, ...]

        if self._isjax and _JAX_AVAILABLE:
            # Flatten the parameters tree into a 1D array.
            def make_flat(x):
                leaves, _   = tree_flatten(x)
                return jnp.concatenate([p.ravel() for p in leaves])
            
            # Compute gradients of the real and imaginary parts.
            grads_real      = make_flat(jax.grad(lambda a,b: jnp.real(self.net.apply(a,b)))(self._weights, sample_state)["params"])
            grads_imag      = make_flat(jax.grad(lambda a,b: jnp.imag(self.net.apply(a,b)))(self._weights, sample_state)["params"] )
            # Flatten the gradients.
            flat_real       = make_flat(grads_real)
            flat_imag       = make_flat(grads_imag)
            
            norm_diff       = jnp.linalg.norm(flat_real - 1.j * flat_imag) / flat_real.shape[0]
            return jnp.isclose(norm_diff, 0.0, atol = self._TOL_HOLOMORPHIC)
        else:
            # Using numpy-based gradients.
            def make_flat(x):
                leaves, _ = flatten_func(x)
                return np.concatenate([p.ravel() for p in leaves])
            
            grads_real      = make_flat(np_grad(lambda a,b: anp.real(self.net.apply(a,b)))(self._weights, sample_state)["params"] )
            grads_imag      = make_flat(np_grad(lambda a,b: anp.imag(self.net.apply(a,b)))(self._weights, sample_state)["params"] )
            
            flat_real       = make_flat(grads_real)
            flat_imag       = make_flat(grads_imag)
            
            norm_diff       = np.linalg.norm(flat_real - 1.j * flat_imag) / flat_real.shape[0]
            return np.isclose(norm_diff, 0.0, atol= self._TOL_HOLOMORPHIC)
        
    def _check_analitic(self):
        '''
        Check if the network is analitic, this means that we check
        whether the function has an analitic gradient - like RBMs or 
        other networks that have a closed form gradient.
        '''
        pass
    
    def _init_gradients(self):
        '''
        Initialize the gradients.
        1. Check if the backend is JAX or NumPy.
        2. If JAX, set the gradient function to JAX's grad, if NumPy, set the gradient function to NumPy's grad.
        3. If the network is complex, set the gradient function to JAX's grad with holomorphic=True, otherwise set it to JAX's grad with holomorphic=False.
        '''
        self._forces        = None
        self._gradients     = None
        
        # self._flat_grad_fun, self._dict_grad_type = NQSUtils.decide_grads(iscpx=self._iscpx,
        #                                 isjax=self._isjax, isanalitic=self._isanalitic, isholomorphic=self._holomorphic)

    def _init_functions(self):
        '''
        Initialize the functions for the gradient and network evaluation.
        1. Check if the backend is JAX or NumPy.
        2. If so, set the evaluation and gradient functions to the appropriate JAX or NumPy functions.
        '''
        if self._isjax:
            self._eval_func         = self._eval_jax
            self._grad_func         = self._grad_jax
        else:
            self._eval_func         = self._eval_np
            self._grad_func         = self._grad_np
        
        # set the local energy function
        self._init_hamiltonian(self._hamiltonian)

    def _init_hamiltonian(self, hamiltonian):
        '''
        Initialize the Hamiltonian.
        Parameters:
            hamiltonian: The Hamiltonian to be used.
        '''
        if hamiltonian is None:
            raise ValueError(self._ERROR_NO_HAMILTONIAN)
        self._hamiltonian   = hamiltonian
        # Hamiltonian can be either a class containing the Hamiltonian
        # or a function that returns the local energy given a state vector s
        # set this callable function
        if not isinstance(self._hamiltonian, Hamiltonian):
            if not callable(self._hamiltonian):
                raise ValueError(self._ERROR_HAMILTONIAN_TYPE)
            # check if it accepts a state vector only
            elif len(inspect.signature(self._hamiltonian).parameters) != 1:
                raise ValueError(self._ERROR_HAMILTONIAN_ARGS)
            else:
                self._local_en_func = self._hamiltonian
        else:
            if self._isjax:
                self._local_en_func = self._hamiltonian.get_loc_energy_jax_fun()
            else:
                self._local_en_func = self._hamiltonian.get_loc_energy_np_fun()
        
        # set the local energy function - jit or numba
        # if the backend is JAX, use jax.jit
        # if the backend is NumPy, use numba.jit
        # if self._isjax:
        #     self._local_en_func = jax_jit(self._local_en_func)
        # else:
        #     self._local_en_func = numba.jit(self._local_en_func)
            
    def init_network(self, s):
        '''
        In1tialize the network truly. This means that the weights are initialized correctly 
        and the dtypes are checked. In addition, the network is checked if it is holomorphic or not.
        Parameters:
            s: The state vector, can be any, but it is used to initialize the network.
        
        Note:
            1. Check if the network is already initialized.
            2. If not, initialize the weights using the network's init method.
            3. Check the dtypes of the weights and ensure they are consistent.
            4. Check if the network is complex and holomorphic.
            5. Check the shape of the weights and store them.
            6. Calculate the number of parameters in the network.
            7. Set the initialized flag to True.
            8. If the network is not initialized, raise a ValueError.
        '''

        if not self._initialized:
            
            # initialize the network 
            self._weights   = self._net.init(self._rng_key)
            
            # check the dtypes of the weights
            # dtypes          = [a.dtype for a in tree_flatten(self._weights)[0]] \
            #                     if self._isjax                                  \
            #                     else [a.dtype for a in flatten_func(self._weights)[0]]
            dtypes          = self._net.dtypes
            
            # check if all dtypes are the same
            if not all([a == dtypes[0] for a in dtypes]):
                raise ValueError(self._ERROR_ALL_DTYPE_SAME)
            
            # check if the network is complex
            self._iscpx     = not (dtypes[0] == np.single or dtypes[0] == np.double)
            
            # check if the network is holomorphic
            # if the value is set to None, we check if the network is holomorphic
            # through calculating the gradients of the real and imaginary parts
            # of the network. Otherwise, we use the value provided.
            if self._net.holomorphic is None:
                # self._holomorphic   = self._check_holomorphic(s)
                self._holomorphic   = True
            else:
                self._holomorphic   = self._net.holomorphic
            
            # check the shape of the weights
            self._paramshape        = self._net.shapes
            
            # number of parameters
            self._nparams           = self._net.nparams
            # if self._isjax:
            #     self._nparams = jnp.sum(jnp.array([p.size for p in tree_flatten(self.parameters["params"])[0]]))
            # else:
            #     self._nparams = np.sum(np.array([p.size for p in flatten_func(self.parameters["params"])[0]]))
    
    #####################################
    #! EVALUATION
    #####################################
    
    def _eval_np(self, net, params, batch_size, data):
        '''
        Evaluate the network using NumPy.
        Parameters:
            net         : The network to be evaluated.
            params      : The parameters of the network.
            batch_size  : The size of the batch.
            data        : The data to be evaluated.
        Returns:
            The evaluated network output.
        '''
        return NQSUtils.eval_batched_np(batch_size=batch_size, func=net, params=params, data=data)[:data.shape[0]]

    def _eval_jax(self, net, params, batch_size, data):
        '''
        Evaluate the network using JAX.
        Parameters:
            net         : The network to be evaluated.
            params      : The parameters of the network.
            batch_size  : The size of the batch.
            data        : The data to be evaluated.
        Returns:
            The evaluated network output.
        '''
        return NQSUtils.eval_batched_jax(batch_size=batch_size, func=net, params=params, data=data)[:data.shape[0]]
    
    def __call__(self, s, **kwargs):
        '''
        Evaluate the network using the provided state. This
        will return the log ansatz of the state coefficient. Uses
        the default backend for this class - using self._eval_func.
        
        Parameters:
            s: The state vector.
            kwargs: Additional arguments for model-specific behavior.
        Returns:
            The evaluated network output.
        '''
        return self._eval_func(self._net, self._weights, s, kwargs.get('batch_size', 1))
    
    #####################################
    #! EVALUATE FUNCTION VALUES
    #####################################
    
    def _local_energy(self, s):
        '''
        Evaluate the local energy of the system.
        Parameters:
            s: The state vector.
        Returns:
            The evaluated local energy. In principle returns a tuple 
            (new_states, new_vals) where new_states are the new states
            and new_vals are the values of the local energy.
            After that, a probability ratio is computed that modifies the 
            new_vals.
        '''
        return self._local_en_func(s)
    
    def _evaluate_local_energy(self, s, log_values, probabilities = None):
        '''
        '''

        def scan(c, x):
            new_states, new_vals    = self._local_energy(x)
            # use the new states to compute the probability ratio
            new_log_vals_rati       = self.log_probability_ratio(log_values, new_states)
            
            # compute the new values
            
            
            return c, new_vals
        
    def _evaluate_fun(self, s, log_values, funct: Callable, probabilities = None):
        '''
        '''
        # return funct(self._net, self._weights, s)
    
    def evaluate_fun(self, s_and_psi = None,
                probabilities = None, sampler = None, functions : Optional[list] = None,
                **kwargs):
        '''
        '''
        if sampler is None:
            sampler = self._sampler
        
        # get the parameters, if not provided set the default values
        num_samples = kwargs.get('num_samples', None)
        num_chains  = kwargs.get('num_chains', None)
        
        if s_and_psi is None or not isinstance(s_and_psi, tuple):
            # create the samples if not provided
            _, (s, p), probabilities = self._sampler.sample(self._net.get_parameters(), num_samples=num_samples,
                                                            num_chains=num_chains)
        else:
            s, p = s_and_psi
        
        # if we already have the samples, choose the function.add()
        # Namely, if the list of functions is empty, we shall use the 
        # local energy function to obtain the estimate of the local energy
        
        if functions is None or len(functions) == 0:
            return self._evaluate_local_energy(s, p, probabilities)
        
        # otherwise, we shall use the functions provided
        # to evaluate other operators
        return [self._evaluate_fun(s, p, f, probabilities) for f in functions]    
    
    #####################################
    #! GRADIENTS
    #####################################
    
    def _grad_jax(self, net, params, batch_size, data, flat_grad_fun):
        '''
        Compute the gradients using JAX. This function uses JAX's
        vmap and scan functions to compute the gradients in a batched manner.
        Parameters:
            net         : The network to be evaluated.
            params      : The parameters of the network.
            batch_size  : The size of the batch.
            data        : The data to be evaluated.
            flat_grad   : The function to compute the gradients.
        Note: 
            For the networks that have a closed form gradient, we can use the
            flat_grad function to compute the gradients analytically. This 
            shall be set previously in the init function.
        '''
        
        # create the batches
        sb = NQSUtils.create_batches_jax(data, batch_size)
        
        # compute the gradients using JAX's vmap and scan
        def scan_fun(c, x):
            return c, jax.vmap(lambda y: flat_grad_fun(net, params, y), in_axes=(0,))(x)
        g = jax.lax.scan(scan_fun, None, sb)[1]
        g = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), g)
        return tree_map(lambda x: x[:data.shape[0]], g)
    
    def _grad_np(self, net, params, batch_size, data, flat_grad):
        '''
        Compute the gradients using NumPy. This function uses NumPy's
        loop to compute the gradients in a batched manner.
        Parameters:
            net         : The network to be evaluated.
            params      : The parameters of the network.
            batch_size  : The size of the batch.
            data        : The data to be evaluated.
            flat_grad   : The function to compute the gradients.
        Note:
            For the networks that have a closed form gradient, we can use the
            flat_grad function to compute the gradients analytically. This
            shall be set previously in the init function.
        Returns:
            The computed gradients.
        !TODO: Add the precomputed gradient vector - memory efficient
        '''
        sb = NQSUtils.create_batches_np(data, batch_size)
        
        # compute the gradients using NumPy's loop
        g = np.zeros((len(sb),) + self._paramshape[0][1], dtype=self._dtype)
        for i, b in enumerate(sb):
            g[i] = flat_grad(net, params, b)
        return g
    
    def _grad(self, data, **kwargs):
        '''
        Compute the gradients.
        '''
        return self._grad_func(net=self._net, batch_size=kwargs.get('batch_size', 1),
                        params=self._weights, data=data,
                        flat_grad=self._flat_grad_fun)
    
    #####################################
    #! TRAINING OVERRIDES
    #####################################
    #!TODO 
    
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
    #! LOG_PROBABILITY_RATIO
    #####################################
    
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
    
    def log_probability_ratio(self, v1, v2 = None, **kwargs) -> Union[np.float64, jnp.float64, float, complex]:
        '''
        Compute the log probability ratio.
        Parameters:
            v1: First vector.
            v2      : Second vector (optional) - if not provided - current
                        state is used.
            kwargs  : Additional arguments.
        '''
        base_ratio = self._log_probability_ratio(v1, v2, **kwargs)
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
    
    def set_weights(self, **kwargs):
        '''
        Set the weights of the NQS solver.
        '''
        
        #! TODO: Add the weights setter
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
    
    
    #####################################
    #! OVERLOADS
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