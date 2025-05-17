#   file    : QES/NQS/nqs.py
#   author  : Maksymilian Kliczkowski
#   date    : 2025-04-01
#   version : 0.1

'''
'''

import os
import numpy as np
import inspect
import numba
import h5py
import json

# typing and other imports
from typing import Union, Tuple, Union, Callable, Optional, Any, Sequence, List
from functools import partial

# from general_python imports
from general_python.algebra.utils import JAX_AVAILABLE, Array
from general_python.algebra.ran_wrapper import choice, randint, uniform
from general_python.common.directories import Directories
from general_python.common.timer import timeit
import general_python.ml.networks as Networks
import general_python.common.binary as Binary

# from hilbert
import Algebra.hilbert as Hilbert

# JAX imports
if JAX_AVAILABLE:
    import jax
    from jax import jit as jax_jit
    from jax import numpy as jnp
    
    # jax tree
    try:
        import jax.tree as jax_tree
        from jax.tree import tree_flatten, tree_unflatten, tree_map
    except ImportError:
        import jax.tree_util as jax_tree
        from jax.tree_util import tree_map
    from jax.tree_util import tree_flatten, tree_unflatten
    from jax.flatten_util import ravel_pytree

    # use flax
    import flax
    import flax.linen as nn
    import flax.training.train_state
else:
    jax             = None
    jax_jit         = None
    jnp             = None
    random          = None
    flax            = None
    nn              = None
    flax_jit        = None
    flax_train      = None
    flax_core       = None
    flax_linen      = None
    tree_map        = None
    jax_tree        = None
    tree_flatten    = None
    tree_unflatten  = None

#########################################

from Solver.MonteCarlo.montecarlo import MonteCarloSolver, McsTrain, McsReturn, Sampler
from Algebra.Operator.operator import Operator, OperatorFunction

# Hamiltonian imports
from Algebra.hamil import Hamiltonian

#########################################

# for the gradients and stuff
import general_python.ml.net_impl.utils.net_utils as net_utils
import general_python.ml.net_impl.net_general as net_general
if JAX_AVAILABLE:
    import general_python.ml.net_impl.interface_net_flax as net_flax

# schedulers and preconditioners and solvers
import general_python.ml.schedulers as scheduler_mod
import general_python.algebra.solvers as solvers_mod

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
    _ERROR_JAX_WITH_FLAX    = "JAX backend is only compatible with Flax networks!"
    
    def __init__(self,
                # information on the NQS
                net         : Union[Callable, str, net_general.GeneralNet],
                sampler     : Union[Callable, str, Sampler],
                hamiltonian : Hamiltonian,
                # information on the Monte Carlo solver
                batch_size  : Optional[int]                     = 1,
                # information on the NQS
                lower_states: Optional[list]                    = None,
                lower_betas : Optional[list]                    = None,
                nparticles  : Optional[int]                     = None,
                # information on the Monte Carlo solver
                seed        : Optional[int]                     = None,
                beta        : float                             = 1,
                mu          : float                             = 2.0,
                replica     : int                               = 1,
                # information on the NQS - Hilbert space
                shape       : Union[list, tuple]                = (1,),
                hilbert     : Optional[Hilbert.HilbertSpace]    = None,
                modes       : int                               = 2,
                # information on the Monte Carlo solver
                directory   : Optional[str]                     = MonteCarloSolver.defdir,
                backend     : str                               = 'default',
                nthreads    : Optional[int]                     = 1,
                **kwargs):
        '''
        Initialize the NQS solver.
        
        Parameters:
            net:
                The neural network to be used.
            sampler:
                The sampler to be used.
            hamiltonian:
                The Hamiltonian to be used.
            batch_size:
                The batch size for training.
            lower_states:
                The lower states for the NQS.
            lower_betas:
                The lower betas for the NQS.
            nparticles:
                The number of particles in the system.
            seed:
                Random seed for initialization.
            beta:
                Beta parameter for the NQS.
            mu:
                Mu parameter for the NQS.
            replica:
                Number of replicas in the system.
            shape:
                Shape of the input data.
            hilbert:
                Hilbert space object (optional).
            modes:
                Number of modes in the system.
            directory:
                Directory for saving results (default: MonteCarloSolver.defdir).
            backend:
                Backend to be used ('default', 'jax', etc.).
            nthreads:
                Number of threads to use (default: 1).
        
        
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
        self._hamiltonian       = hamiltonian
        
        #######################################
        #! collect the Hilbert space information
        #######################################
        
        self._nh                = self._hilbert.Nh if self._hilbert is not None else None
        self._nparticles        = nparticles if nparticles is not None else self._size
        self._nvisible          = self._size
        self._nparticles2       = self._nparticles**2
        self._nvisible2         = self._nvisible**2
        
        #######################################
        #! state modifier (for later)
        #######################################
        
        self._modifier          = None
        self._modifier_func     = None
        
        #######################################
        #! handle the network
        #######################################
        
        self._batch_size        = batch_size
        self._initialized       = False
        self._dtype             = None
        self._analytic          = False                                 # analytic gradients
        self._holomorphic       = True                                  # holomorphic network
        self._net : Networks.GeneralNet = self._choose_network(net, input_shape=self._shape, backend=self._backend_str, **kwargs)
        if not self._net.initialized:
            self.init_network()
        self._isjax             = getattr(self._net, 'is_jax', True)    # Assumes net object knows its backend type
        self._iscpx             = self._net.is_complex
        self._holomorphic       = self._net.is_holomorphic
        # self._holomorphic       = False #! Enforce False
        self._analytic          = self._net.has_analytic_grad
        self._dtype             = self._net.dtype
        
        #######################################
        #! handle gradients
        #######################################
        
        self._params_slice_metadata     = None  # metadata for the parameter slices, e.g., (slice, name)
        self._params_leaf_info          = None  # leaf information for the parameters - useful for the gradients
        self._params_tree_def           = None  # tree definition for the parameters, meaning the structure of the parameters
        self._params_total_size         = None  # total size of the parameters
        #! ------------------------------------
        self._flat_grad_func             = None  # function to calculate the gradients (returns a flat vector)
        self._dict_grad_type            = None  # dictionary with the type of gradients (real/imaginary)
        self._init_gradients()
        self._init_param_metadata()
        
        #######################################
        #! handle the functions for the ansatz - we will store them not to recompile them
        #######################################
        
        self._params                    = None  # parameters of the network 
        self._local_en_func             = None  # function to calculate the local energy (Callable[state])
        self._ansatz_func               = None  # function to calculate the ansatz for a given state (using net, Callable[state, params])
        self._eval_func                 = None  # function to batch evaluate the ansatz (Callable[states, ansatz, params])
        self._apply_func                = None  # function to apply callable with the ansatz (Callable[states, ansatz, params])
        # set the gradient function if needed
        self._grad_func                 = None
        
        #######################################
        #! handle the optimizer
        #######################################
        
        self._init_functions(batch_size=self._batch_size)
        self._initialized               = True
        
        #######################################
        #! set the lower states
        #######################################
        
        if lower_states is not None and lower_betas is not None:
            self._lower_states_manager = NQSLower(lower_states, lower_betas, self)
        else:
            self._lower_states_manager = NQSLower([], [], self)
        
    #####################################
    #! NETWORK
    #####################################
    
    def _choose_network(self, net_spec: Any, input_shape: tuple, backend: str, **kwargs) -> Networks.GeneralNet:
        """
        Selects, instantiates, and initializes the network object.
        Ensures the returned network is initialized and conforms to GeneralNet.
        
        Parameters
        ----------
        net_spec : Any
            The network specification, which can be a string, callable, or a GeneralNet instance.
        input_shape : tuple
            The shape of the input data for the network.
        backend : str
            The backend to be used for the network ('jax' or 'numpy').
        **kwargs : dict
            Additional keyword arguments for network instantiation.
        Returns
        -------
        Networks.GeneralNet
            An initialized instance of the specified network.
            
        Raises
        ------
        TypeError
            If the provided network specification is of an unsupported type.
        """
        # Force backend to jax if needed
        if issubclass(type(net_spec), nn.Module) or isinstance(net_spec, Networks.FlaxInterface):
            backend = 'jax'
            if not JAX_AVAILABLE:
                raise ImportError("Flax requires JAX.")
            
        #! Check if the network is already a GeneralNet instance
        if isinstance(net_spec, Networks.GeneralNet):
            # If already a GeneralNet instance, ensure it's initialized
            network_instance = net_spec
            if not network_instance.initialized:
                network_instance.init()
            # TODO: Verify backend compatibility?
            return network_instance
        elif issubclass(type(net_spec), nn.Module) or callable(net_spec):
            # Assume it's a Flax module class or similar callable needing wrapping
            if backend != 'jax':
                raise ValueError(f"Flax module {net_spec} requires 'jax' backend.")
            # FlaxInterface handles its own initialization inside its __init__
            return Networks.FlaxInterface(net_module=net_spec, input_shape=input_shape, backend=backend, **kwargs)
        elif isinstance(net_spec, str):
            # Use your existing Networks.choose_network factory
            network_instance = Networks.choose_network(network_type=net_spec, input_shape=input_shape, backend=backend, **kwargs)
            if not network_instance.initialized:
                network_instance.init() # Ensure factory initializes if needed
            return network_instance
        else:
            raise TypeError(f"Unsupported network specification type: {type(net_spec)}")
    
    #####################################
    #! INITIALIZATION OF THE NETWORK AND FUNCTIONS
    #####################################
    
    def reset(self):
        """
        Resets the initialization state of the object and reinitializes the underlying network.
        This method marks the object as not initialized and forces a reinitialization of the associated
        neural network by calling its `force_init` method.
        """
        
        self._initialized = False
        self._net.force_init()
    
    # ---
    
    def _init_param_metadata(self):
        """
        Initializes the parameter metadata for the neural quantum state (NQS).
        This includes the parameter slices, names, and tree structure.
        It uses the `jax.tree_util` module to flatten the parameters and
        create a mapping of slices to names.
        """
        
        if self._params_tree_def is None and self._isjax:
        
            current_params              = self._net.get_params()
            
            # get the parameter tree definition
            _, self._params_tree_def    = tree_flatten(current_params)
                        
            # get the leaf information for the parameters
            self._params_leaf_info      = net_utils.jaxpy.prepare_leaf_info(current_params)
        
            # get the parameter slices and names
            self._params_slice_metadata = net_utils.jaxpy.prepare_unflatten_metadata_from_leaf_info(self._params_leaf_info)
            
            # get the sizes
            self._params_sizes          = [slice_info.size for slice_info in self._params_slice_metadata]
            
            # get the shapes
            self._params_shapes         = [slice_info.shape for slice_info in self._params_slice_metadata]
            
            # get the info about the type
            self._params_iscpx          = [slice_info.is_complex for slice_info in self._params_slice_metadata]
            
            # get the total size of the parameters
            self._params_total_size     = self._params_slice_metadata[-1].start + self._params_slice_metadata[-1].size if self._params_slice_metadata else 0
    
    # ---
    
    def _init_gradients(self):
        """
        Initializes the gradient computation method for the neural quantum state (NQS).
        It sets up the gradient of the logarithmic wave function. It checks whether the network is:
        - complex
        - holomorphic
        - has an analytic gradient solution
        This method determines the appropriate gradient computation function and gradient type
        based on the properties of the system, such as whether the system is complex-valued,
        uses JAX for computation, employs analytic gradients, or is holomorphic.
        Attributes:
            self._flat_grad_func : A function for computing gradients in a flattened format.
            self._dict_grad_type: A dictionary specifying the type of gradients to be used.
        Dependencies:
            - net_utils.decide_grads: A utility function that selects the gradient computation
                method based on the provided flags.
        Flags:
            - self._iscpx: Boolean indicating if the system is complex-valued.
            - self._isjax: Boolean indicating if JAX is used for computation.
            - self._analytic: Boolean indicating if analytic gradients are used.
            - self._holomorphic: Boolean indicating if the system is holomorphic.
        """

        self._flat_grad_func, self._dict_grad_type = net_utils.decide_grads(iscpx=self._iscpx,
                            isjax=self._isjax, isanalytic=self._analytic, isholomorphic=self._holomorphic)
        
        if self._analytic:
            
            # Get analytical function handle from the network object
            self.log(f"Analytical gradient function check...", log='info', lvl = 2, color = 'blue')
            
            analytical_pytree_fun, _ = self._net.get_gradient(use_jax=True)
            if analytical_pytree_fun is None:
                raise ValueError("Analytical gradient selected but network did not provide grad_func.")
            # self._flat_grad_func = net_utils.jaxpy.get_analytical_gradient_function(analytical_pytree_fun, self._params_tree_def)
            self.log(f"Analytical gradient function provided.", log='info', lvl = 2, color = 'blue')
    
    # ---

    def _init_functions(self, batch_size: Optional[int] = None):
        '''
        Initialize the functions for the gradient and network evaluation.
        1. Check if the backend is JAX or NumPy.
        2. If so, set the evaluation and gradient functions to the appropriate JAX or NumPy functions.
        '''
        
        #! set the batch size
        self._batch_size = batch_size if batch_size is not None else self._batch_size
        
        #! set the evaluation function, look for the ansatz function and it's modifier
        if not self.modified:
            self._ansatz_func, self._params = self._net.get_apply(use_jax=self._isjax)
        else:
            self._ansatz_func, self._params = self._net.get_apply_modified(use_jax=self._isjax)
        #! gradient is unchanged with modifier
        self._grad_func, self._params = self._net.get_gradient(use_jax=self._isjax)
        
        if self._isjax:
            # ansatz evaluation function already JITted
            self._eval_func             = jax.jit(net_utils.jaxpy.eval_batched_jax, static_argnums=(0, 1))
            if self._batch_size > 1:
                self._apply_func        = jax.jit(net_utils.jaxpy.apply_callable_batched_jax, static_argnums=(0, 4, 6))
            else:
                self._apply_func        = jax.jit(net_utils.jaxpy.apply_callable_jax, static_argnums=(0, 4))
        else:
            # ansatz evaluation function already compiled
            self._eval_func             = net_utils.numpy.eval_batched_np
            self._apply_func            = net_utils.numpy.apply_callable_batched_np
        
        # set the local energy function
        self._init_hamiltonian(self._hamiltonian)

    # ---
    
    def _init_hamiltonian(self, hamiltonian):
        '''
        Initialize the Hamiltonian.
        Parameters:
            hamiltonian: The Hamiltonian to be used.
        '''
        if hamiltonian is None:
            raise ValueError(self._ERROR_NO_HAMILTONIAN)
        self._hamiltonian = hamiltonian
        
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

    # ---
    
    def init_network(self):
        '''
        Initialize the network truly. This means that the weights are initialized correctly 
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
            self._params            = self._net.init(self._rng)
            dtypes                  = self._net.dtypes
            
            # check if all dtypes are the same
            if not all([a == dtypes[0] for a in dtypes]):
                raise ValueError(self._ERROR_ALL_DTYPE_SAME)
            
            # check if the network is complex
            self._iscpx             = not (dtypes[0] == np.single or dtypes[0] == np.double)
            
            # check if the network is holomorphic
            # if the value is set to None, we check if the network is holomorphic
            # through calculating the gradients of the real and imaginary parts
            # of the network. Otherwise, we use the value provided.
            self._holomorphic       = self._net.check_holomorphic()
            self.log(f"Network is holomorphic: {self._holomorphic}", log='info', lvl = 2, color = 'blue')
            self._analytic          = self._net.has_analytic_grad
            self.log(f"Network has analytic gradient: {self._analytic}", log='info', lvl = 2, color = 'blue')
            
            # check the shape of the weights
            self._paramshape        = self._net.shapes
            
            # number of parameters
            self._nparams           = self._net.nparams
            self._initialized       = True
    
    # ---
    
    def init_state(self):
        '''
        Initialize the state of the network. This is done by creating a new
        TrainState object with the network's parameters and the provided shape.
        Returns:
            flax.training.train_state.TrainState
                The initialized TrainState object.
        Note:
            This method is only applicable if the backend is JAX and the network
            is a Flax network. If the backend is not JAX, this method will return None.
            
        '''

        if JAX_AVAILABLE and self._isjax and issubclass(type(self._net), net_flax.FlaxInterface):
            params   = self._net.init(self._rngJAX_RND_DEFAULT_KEY, jnp.ones(self._shape, dtype=jnp.int32))
            return flax.training.train_state.TrainState.create(
                apply_fn = self._ansatz_func,
                params   = params,
                tx       = None
            )
        else:
            return None
    
    #####################################
    #! EVALUATION OF THE ANSATZ BATCHED (\psi(s))
    #####################################
    
    def eval_jax(self, states, batch_size = None, params = None):
        """
        Evaluates the neural network (log ansatz) for the given quantum states using JAX.
        This method applies the network function to the provided quantum states, using
        JAX for computation. The evaluation can be performed in batches for memory efficiency.
        Parameters
        ----------
        states : array_like
            The quantum states for which to evaluate the network.
        states : array_like
            The quantum states (configurations) to evaluate the network on.
        batch_size : int, optional
            The size of batches to use for the evaluation. If None, uses the default batch size
            stored in self._batch_size.
        params : dict, optional
            The parameters (weights) to use for the network evaluation. If None, uses the
            current parameters stored in network._params.
        Returns
        -------
        array_like
            The output of the neural network for the given states, representing the log of the 
            wavefunction amplitudes.
        """

        # evaluate the network (log ansatz) using JAX
        return net_utils.jaxpy.eval_batched_jax(batch_size, self._ansatz_func, params, states)
    
    def eval_np(self, states, batch_size = None, params = None):
        """
        Evaluates the neural network (log ansatz) for the given quantum states using NumPy.
        This method applies the network function to the provided quantum states, using
        NumPy for computation. The evaluation can be performed in batches for memory efficiency.
        Parameters
        ----------
        states : array_like
            The quantum states for which to evaluate the network.
        batch_size : int, optional
            The size of batches to use for the evaluation. If None, uses the default batch size
            stored in self._batch_size.
        params : dict, optional
            The parameters (weights) to use for the network evaluation. If None, uses the
            current parameters stored in self._params.
        Returns
        -------
        array_like
            The output of the neural network for the given states, representing the log of the 
            wavefunction amplitudes.
        """

        # evaluate the network (log ansatz) using NumPy
        return net_utils.numpy.eval_batched_np(batch_size, self._ansatz_func, params, states)
    
    def evaluate_ansatz(self, states, batch_size = None, params = None):
        '''
        Evaluate the network using the provided state. This
        will return the log ansatz of the state coefficient.
        
        Parameters:
            states      : The state vector.
            batch_size  : The size of batches to use for the evaluation.
            params      : The parameters (weights) to use for the network evaluation.
        Returns:
            The evaluated network output.
        '''
        
        if params is None:
            params = self.get_params()
            
        if batch_size is None:
            batch_size = self._batch_size
        
        if self._isjax:
            return self.eval_jax(states, batch_size=batch_size, params=params)
        return self.eval_np(states, batch_size=batch_size, params=params)
    
    def __call__(self, states, **kwargs):
        '''
        Evaluate the network using the provided state. This
        will return the log ansatz of the state coefficient. Uses
        the default backend for this class - using self._eval_func.
        
        Parameters:
            s:
                The state vector.
            kwargs:
                Additional arguments for model-specific behavior.
        Returns:
            The evaluated network output.
        '''
        if self._isjax:
            return self.eval_jax(states, batch_size=self._batch_size, params=self.get_params())
        return self.eval_np(states, batch_size=self._batch_size, params=self.get_params())
    
    #####################################
    #! EVALUATE FUNCTION VALUES - LOCAL ENERGY AND OTHER FUNCTIONS (OPERATORS)
    #####################################
    
    @staticmethod
    @partial(jax_jit, static_argnames=['func', 'logproba_fun', 'batch_size'])
    def evaluate_fun_jax(func         : Callable,
                        states        : jnp.ndarray,
                        probabilities : jnp.ndarray,
                        logproba_in   : jnp.ndarray,
                        logproba_fun  : Callable,
                        parameters    : Union[dict, list, jnp.ndarray],
                        batch_size    : Optional[int] = None):
        """
        Evaluates a given function on a set of states and probabilities, with optional batching.
        Args:
            func (Callable):
                The function to be evaluated.
            states (jnp.ndarray):
                The input states for the function.
            probabilities (jnp.ndarray):
                The probabilities associated with the states.
            logproba_in (jnp.ndarray):
                The logarithm of the probabilities for the input states.
            logproba_fun (Callable):
                A function to compute the logarithm of probabilities.
            parameters (Union[dict, list, jnp.ndarray]):
                Parameters to be passed to the function.
            batch_size (Optional[int], optional):
                The size of batches for evaluation. 
                If None, the function is evaluated without batching. Defaults to None.
        Returns:
            The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
        """
        if batch_size is None or batch_size == 1:
            funct_in = net_utils.jaxpy.apply_callable_jax
            return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters)
        else:
            funct_in = net_utils.jaxpy.apply_callable_batched_jax
            return funct_in(func            = func,
                            states          = states,
                            sample_probas   = probabilities,
                            logprobas_in    = logproba_in,
                            logproba_fun    = logproba_fun,
                            parameters      = parameters,
                            batch_size      = batch_size)
    
    @staticmethod
    def evaluate_fun_np(func          : Callable,
                        states        : np.ndarray,
                        probabilities : np.ndarray,
                        logproba_in   : np.ndarray,
                        logproba_fun  : Callable,
                        parameters    : Union[dict, list, np.ndarray],
                        batch_size    : Optional[int] = None):
        """
        Evaluates a given function on a set of states and probabilities, with optional batching.
        
        Args:
            func (Callable):
                The function to be evaluated.
            states (np.ndarray):
                The input states for the function.
            probabilities (np.ndarray):
                The probabilities associated with the states.
            logproba_in (np.ndarray):
                The logarithm of the probabilities for the input states.
            logproba_fun (Callable):
                A function to compute the logarithm of probabilities.
            parameters (Union[dict, list, np.ndarray]):
                Parameters to be passed to the function.
            batch_size (Optional[int], optional):
                The size of batches for evaluation. 
                If None, the function is evaluated without batching. Defaults to None.
        Returns:
            The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
        """
        
        if batch_size is None:
            funct_in = net_utils.numpy.apply_callable_np
            return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters)
        # otherwise, we shall use the batched version
        funct_in = net_utils.numpy.apply_callable_batched_np
        return funct_in(func            = func,
                        states          = states,
                        sample_probas   = probabilities,
                        logprobas_in    = logproba_in,
                        logproba_fun    = logproba_fun,
                        parameters      = parameters,
                        batch_size      = batch_size)

    @staticmethod
    def _evaluate_fun(func          : Callable,
                    states          : np.ndarray,
                    probabilities   : np.ndarray,
                    logproba_in     : np.ndarray,
                    logproba_fun    : Callable,
                    parameters      : Union[dict, list, np.ndarray],
                    batch_size      : Optional[int] = None,
                    is_jax          : bool          = True):
        """
        Evaluates a given function on a set of states and probabilities, with optional batching.
        
        Args:
            func (Callable):
                The function to be evaluated.
            states (Union[np.ndarray, jnp.ndarray]):
                The input states for the function.
            probabilities (Union[np.ndarray, jnp.ndarray]):
                The probabilities associated with the states.
            logproba_in (Union[np.ndarray, jnp.ndarray]):
                The logarithm of the probabilities for the input states.
            logproba_fun (Callable):
                A function to compute the logarithm of probabilities.
            parameters (Union[dict, list, np.ndarray]):
                Parameters to be passed to the function.
            batch_size (Optional[int], optional):
                The size of batches for evaluation. 
                If None, the function is evaluated without batching. Defaults to None.
            is_jax (bool, optional):
                Flag indicating if JAX is used for computation. Defaults to True.
        Returns:
            The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
        """
                    
        if batch_size is None:
            funct_in = net_utils.jaxpy.apply_callable_jax if is_jax else net_utils.numpy.apply_callable_np
            return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters)
        # otherwise, we shall use the batched version
        funct_in = net_utils.jaxpy.apply_callable_batched_jax if is_jax else net_utils.numpy.apply_callable_batched_np
        return funct_in(func            = func,
                        states          = states,
                        sample_probas   = probabilities,
                        logprobas_in    = logproba_in,
                        logproba_fun    = logproba_fun,
                        parameters      = parameters,
                        batch_size      = batch_size)
    
    @staticmethod
    def _evaluate_fun_s(func        : list[Callable],
                        sampler     : Sampler,
                        num_samples : int,
                        num_chains  : int,
                        logproba_fun: Callable,
                        parameters  : dict,
                        batch_size  : Optional[int] = None,
                        is_jax      : bool = True):
        """
        Evaluates a given function using samples generated by a sampler.

        This method utilizes a sampler to generate states, ansatze, and their 
        associated probabilities, and then evaluates the provided function 
        using these samples.

        Args:
            func (Callable)                 : The function to be evaluated. It should accept 
                                            states, probabilities, ansatze, logproba_fun, and parameters 
                                            as inputs.
            sampler (Sampler)               : The sampler object used to generate samples.
            num_samples (int)               : The total number of samples to generate.
            num_chains (int)                : The number of independent Markov chains to use 
                                            in the sampling process.
            logproba_fun (Callable)         : A function that computes the logarithm 
                                            of the probability for given states.
            parameters (dict)               : A dictionary of parameters to be passed to 
                                            the function being evaluated.
            batch_size (Optional[int])      : The size of batches to process at a 
                                            time. If None, the entire dataset is processed at once.
                                            Defaults to None.
            is_jax (bool, optional)         : Flag indicating if JAX is used for computation. 
                                            Defaults to True.
        Returns:    
            Any: The result of evaluating the provided function `func` using 
            the generated samples.
        """

        _, (states, ansatze), probabilities = sampler.sample(parameters=parameters, num_samples=num_samples, num_chains=num_chains)
        evaluated_results = [NQS._evaluate_fun(f, states, probabilities, ansatze, logproba_fun, parameters, batch_size, is_jax) for f in func]
        return (states, ansatze), probabilities, evaluated_results
    
    def evaluate_fun(self,
                functions       : Optional[list] = None,
                states_and_psi  : Optional[Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]] = None,
                probabilities   : Optional[Union[np.ndarray, jnp.ndarray]] = None,
                **kwargs):
        """
        Evaluate a set of functions based on the provided states, wavefunction, and probabilities.
        This method computes the output of one or more functions using the provided states, 
        wavefunction (ansatze), and probabilities. If states and wavefunction are not provided, 
        it uses a sampler to generate the required data.
        
        Args:
            functions (Optional[list]):
                A list of functions to evaluate. If not provided, 
                defaults to using the local energy function (`self._local_en_func`).
            states_and_psi (Optional[Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]]):
                A tuple containing the states and the corresponding wavefunction (ansatze). 
                If not provided, the sampler is used to generate these.
            probabilities (Optional[Union[np.ndarray, jnp.ndarray]]):
                Probabilities associated with the states. If not provided, defaults to an array of ones with the same 
                shape as the wavefunction ansatze.
            **kwargs:
                Additional keyword arguments:
                    - batch_size (int)          : The batch size for evaluation. Defaults to self._batch_size.
                    - num_samples (int)         : Number of samples to generate if using the sampler.
                    - num_chains (int)          : Number of chains to use if using the sampler.
        Returns:
            Union[Any, list]:
                The output of the evaluated functions. If a single function is 
                provided, the result is returned directly. If multiple functions are provided, 
                a list of results is returned.
        """
        
        output          = [None]
        batch_size      = kwargs.get('batch_size', self._batch_size)
        params          = kwargs.get('params', self._net.get_params())
        
        # check if the functions are provided
        if functions is None or len(functions) == 0:
            functions = [self._local_en_func]
        
        # check if the states and psi are provided
        states, ansatze = None, None
        if states_and_psi is not None:
            if isinstance(states_and_psi, tuple):
                states, ansatze = states_and_psi
            else:
                states          = states_and_psi
                ansatze         = self(states)
            
            # check if the probabilities are provided
            if probabilities is None:
                probabilities = self._backend.ones_like(ansatze).astype(ansatze.dtype)
            
            if self._isjax:
                output = [NQS.evaluate_fun_jax(func     = f,
                                        states          = states,
                                        probabilities   = probabilities,
                                        logproba_in     = ansatze,
                                        logproba_fun    = self._ansatz_func,
                                        parameters      = params,
                                        batch_size      = batch_size) for f in functions]
            else:
                # otherwise, we shall use the numpy version
                output = [NQS.evaluate_fun_np(func      = f,
                                        states          = states,
                                        probabilities   = probabilities,
                                        logproba_in     = ansatze,
                                        logproba_fun    = self._ansatz_func,
                                        parameters      = params,
                                        batch_size      = batch_size) for f in functions]
        else:
            # get other parameters from kwargs
            num_samples = kwargs.get('num_samples', self._sampler.num_samples)
            num_chains  = kwargs.get('num_chains', self._sampler.numchains)
            
            # otherwise, we shall use the sampler
            (states, ansatze), probabilities, output    = \
                    self._evaluate_fun_s(func           = functions,
                                        sampler         = self._sampler,
                                        num_samples     = num_samples,
                                        num_chains      = num_chains,
                                        logproba_fun    = self._ansatz_func,
                                        parameters      = params,
                                        batch_size      = batch_size,
                                        is_jax          = self._isjax)
        
        # check if the output is a list
        if isinstance(output, list) and len(output) == 1:
            output = output[0]
        return (states, ansatze), probabilities, output

    def __getitem__(self, funct: Callable):
        '''
        Use this to apply a function to the state with the ansatz.
        Parameters:
            funct: The function to apply to the state.
        Returns:
            The result of the function applied to the state.
        Note:
            This method is used to apply a function to the state with the ansatz.
            It uses the sampler to sample the states and ansatz, and then applies
            the function to the sampled states and ansatz.
        '''
        # sample the states and ansatz
        (_, _), (configs, configs_ansatze), probabilities = self._sampler.sample()
        return self._evaluate_fun(funct,
                                configs, probabilities, configs_ansatze,
                                self._ansatz_func, self._net.get_params(), self._batch_size, self._isjax)
    
    #####################################
    #! SAMPLE
    #####################################
    
    def sample(self, num_samples = None, num_chains = None, reset: bool = True):
        '''
        Sample the NQS using the provided sampler. This will return
        the sampled states and the corresponding probabilities.
        Parameters:
            num_samples (int):
                The number of samples to generate.
            num_chains (int):
                The number of chains to use for sampling.
            reset (bool):
                Whether to reset the sampler before sampling. This 
                corresponds to reinitializing the state before sampling the new ones.
        Returns:
            The sampled states and the corresponding probabilities.
            (last configs, last ansatze), (all configs, all ansatze), (all probabilities)
        '''
        if reset:
            self._sampler.reset()
        return self._sampler.sample(num_samples=num_samples, num_chains=num_chains)
    
    #####################################
    #! GRADIENTS
    #####################################
    
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 3, 4))
    def grad_jax(
            net_apply                   : Callable,
            params                      : Any,
            states                      : jnp.ndarray,
            single_sample_flat_grad_fun : Callable[[Callable, Any, Any], jnp.ndarray],
            batch_size                  : int = 1) -> jnp.ndarray:
        '''
        Compute the batch of flattened gradients using JAX (JIT compiled).

        Returns the gradients (e.g., :math:`O_k = \\nabla \\ln \\psi(s)`)
        for each state in the batch. The output format (complex/real)
        depends on the `single_sample_flat_grad_fun` used.

        Parameters
        ----------
        net_apply : Callable
            The network's apply function `f(p, x)`. Static argument for JIT.
        params : Any
            Network parameters `p`.
        states : jnp.ndarray
            Input states `s_i`, shape `(num_samples, ...)`.
        single_sample_flat_grad_fun : Callable[[Callable, Any, Any], jnp.ndarray]
            JAX-traceable function computing the flattened gradient for one sample.
            Signature: `fun(net_apply, params, single_state) -> flat_gradient_vector`.
            Static argument for JIT.
        batch_size : int
            Batch size. Static argument for JIT.

        Returns
        -------
        jnp.ndarray
            Array of flattened gradients, shape `(num_samples, num_flat_params)`.
            Dtype matches the output of `single_sample_flat_grad_fun`.
        '''
        
        # The dtype (complex/real) depends on single_sample_flat_grad_fun
        gradients_batch, shapes, sizes, is_cpx = net_utils.jaxpy.compute_gradients_batched(net_apply, params, states, single_sample_flat_grad_fun, batch_size)
        return gradients_batch, shapes, sizes, is_cpx
    
    @staticmethod
    def grad_np(net, params, batch_size, states, flat_grad):
        '''
        !TODO: Add the precomputed gradient vector - memory efficient
        '''
        sb = net_utils.numpy.create_batches_np(states, batch_size)
        
        # compute the gradients using NumPy's loop
        g = np.zeros((len(sb),) + params.shape[1:], dtype=np.float64)
        for i, b in enumerate(sb):
            g[i] = flat_grad(net, params, b)
        return g
    
    def gradient(self, states, batch_size = None, params = None):
        '''
        Compute the gradients of the ansatz logarithmic wave-function using JAX or NumPy.
        
        Parameters:
            states      : The state vector.
            batch_size  : The size of batches to use for the evaluation.
            params      : The parameters (weights) to use for the network evaluation.
        Returns:
            The computed gradients.
        '''
        
        # check if the batch size is provided
        batch_size = batch_size if batch_size is not None else self._batch_size
        if batch_size is None:
            batch_size = 1
            
        # check if the parameters are provided
        params = self._net.get_params() if params is None else params
        
        if self._isjax:
            if not self._analytic:
                return self.grad_jax(self._ansatz_func, params, states, self._flat_grad_func, batch_size)
            return self.grad_jax(self._grad_func, params, states, self._flat_grad_func, batch_size)
        
        if not self._analytic:
            return self.grad_np(self._ansatz_func, params, batch_size, states, self._flat_grad_func)
        return self.grad_np(self._grad_func, params, batch_size, states, self._flat_grad_func)
    
    #####################################
    #! UPDATE PARAMETERS
    #####################################
        
    def transform_flat_params(self,
                            flat_params : jnp.ndarray,
                            shapes      : list,
                            sizes       : list,
                            is_cpx      : bool) -> Any:
        """
        Transform a flat parameter vector into a PyTree structure.

        Parameters
        ----------
        flat_params : jnp.ndarray
            The flat parameter vector to transform.
        shapes : list
            The shapes of the original parameters.
        sizes : list
            The sizes of the original parameters.
        is_cpx : bool
            Whether the parameters are complex.

        Returns
        -------
        Any
            The transformed PyTree structure.
        """
        if not self._isjax:
            raise NotImplementedError("Only JAX backend supported.")
        # transform shapes to NamedTuple
        slices = net_utils.jaxpy.prepare_slice_info(shapes, sizes, is_cpx)
    
        # Transform the flat parameters back to the original PyTree structure
        return net_utils.jaxpy.transform_flat_params_jit(flat_params,
                                                        self._params_tree_def, 
                                                        slices,
                                                        self._params_total_size)
    
    def update_parameters(self, d_par: Any, mult: Any, shapes, sizes, iscpx):
            """
            Update model parameters using a flat vector (in real representation) or a PyTree.

            Parameters
            ----------
            d_par : Any
                The parameter update. Can be:
                    1.  A 1D JAX array (`jnp.ndarray`) containing the update in the
                        flattened **real representation** format (matching the structure
                        defined by the model's parameters, including [Re, Im] for complex leaves).
                    2.  A PyTree (dict, list, custom) matching the exact structure of
                        the model's parameters.
            """
            if not self._isjax:
                raise NotImplementedError("Only JAX backend supported.")

            #! Handle Input Types
            if isinstance(d_par, jnp.ndarray):
                update_tree = self.transform_flat_params(d_par * mult, shapes, sizes, iscpx)
            elif isinstance(d_par, (dict, list)) or hasattr(d_par, '__jax_flatten__'):
                # Validate PyTree structure (necessary for correctness)
                flat_leaves_dpar, tree_d_par = tree_flatten(d_par * mult)
                if tree_d_par != self._params_tree_def:
                    raise ValueError("Provided `d_par` PyTree structure does not match model parameters structure.")
                update_tree = d_par
            else:
                raise TypeError(f"Unsupported type for parameter update `d_par`: {type(d_par)}. "
                                "Expected PyTree or 1D JAX array.")

            #! Update Parameters
            current_params  = self._net.get_params()
            new_params      = net_utils.jaxpy.add_tree(current_params, update_tree)
            self._net.set_params(new_params)
    
    #####################################
    #! TRAINING OVERRIDES
    #####################################
    
    def _sample_for_shapes(self, *args, **kwargs):
        """
        Sample a configuration to get the shapes of the parameters.
        This is used to initialize the shapes of the parameters.
        """
        #! Sample the configurations
        numchains   = self._sampler.numchains
        numsamples  = self._sampler.numsamples
        params      = self.get_params()
        (_, _), (configs, configs_ansatze), probabilities = self._sampler.sample(params)
                                                                    # num_chains=1, num_samples=1)
        
        #! Get the shapes of the parameters
        _, _, (shapes, sizes, iscpx) = NQS._single_step_jax(
                params,
                configs,
                configs_ansatze,
                probabilities,
                *args, **kwargs)
        
        #! Set the number of chains and samples
        self._sampler.set_numchains(numchains)
        self._sampler.set_numsamples(numsamples)
        return shapes, sizes, iscpx
        
    @staticmethod
    def _single_step_jax(
            # Dynamic Inputs (Data & State)
            params                      : Any,                              # Current PyTree parameters
            configs                     : jnp.ndarray,                      # Batch of sampled configurations (N_samples, ...shape)
            configs_ansatze             : jnp.ndarray,                      # Batch of sampled ansatze (N_samples, ...)
            probabilities               : jnp.ndarray,                      # Batch of sampled probabilities (N_samples, ...)
            # Static Inputs (Functions & Config)
            apply_fn                    : Callable,                         # Network apply function: apply_fn(params, state) -> log_psi
            local_energy_fun            : Callable,                         # Computes local energy: local_energy_fun(state, params) -> E_loc
            flat_grad_fun               : Callable,                         # Computes grad O_k: fun(apply_fn, params, state) -> flat_grad
            apply_fun                   : Callable,                         # Function to evaluate the local energy
            # Static for evaluation
            batch_size                  : Optional[int]             = None, # Batch size for evaluation
            t                           : Optional[float]           = None, # Time for the jax,
            int_step                    : Optional[int]             = 0,    # Step for the
        ) -> Tuple[Array, Array]:
        """
        Performs a single training step for Neural Quantum States (NQS) using JAX.
        This function computes the local energies, gradients, and the parameter update vector (d_par).
        It supports both standard force vector calculation and Stochastic Reconfiguration (SR) for parameter updates.
        Args:
            params (Any):
                Current parameters of the neural network (PyTree).
            configs (jnp.ndarray):
                Batch of sampled configurations, shape (N_samples, ...).
            configs_ansatze (jnp.ndarray):
                Batch of sampled ansatze, shape (N_samples, ...).
            probabilities (jnp.ndarray):
                Batch of sampled probabilities, shape (N_samples, ...).
            apply_fn (Callable):
            local_energy_fun (Callable):
                Function to compute local energy `E_loc = local_energy_fun(states, params)`. Static.
            flat_grad_fun (Callable):
                Function to compute flattened gradient for one sample. Static.
        Notes:
            - If `use_sr` is True, the SR solver computes the parameter update vector by solving the linear system:
                S * d_par = F,
                where S is the Fisher matrix and F is the force vector.
            - If `use_sr` is False, the parameter update vector is computed directly as the force vector.
            - The Fisher matrix S is computed using centered gradients, and the force vector F is computed using
                centered energies.
            - In case of SR solver failure, the function returns a NaN vector for `d_par_flat` and sets
                `sr_info['converged']` to False.
        Raises:
            ValueError: If `use_sr` is True but no SR solver function is provided in `sr_options`.
        """
        
        batch_size = batch_size if batch_size is not None else 1
        
        #! 1. Compute Local Energy
        (v, means, stds) = apply_fun(func            = local_energy_fun,
                                    states          = configs,
                                    sample_probas   = probabilities,
                                    logprobas_in    = configs_ansatze,
                                    logproba_fun    = apply_fn,
                                    parameters      = params,
                                    batch_size      = batch_size)
        
        #! 2. Compute Gradients (O_k = ∇ log ψ)
        # The output `flat_grads` will have the dtype determined by single_sample_flat_grad_fun
        # For complex NQS, this is typically complex. Shape: (batch_size, n_params_flat)
        flat_grads, shapes, sizes, iscpx = net_utils.jaxpy.compute_gradients_batched(
                            net_apply                   = apply_fn,
                            params                      = params,
                            states                      = configs,
                            single_sample_flat_grad_fun = flat_grad_fun,
                            batch_size                  = batch_size)
        
        return (v, means, stds), flat_grads, (shapes, sizes, iscpx)

    def wrap_single_step_jax(self, batch_size: Optional[int] = None):
        """
        Wraps the single-step JAX function for use in optimization or sampling routines.
        This method prepares and returns a JIT-compiled function that performs a single optimization or sampling step
        using the neural quantum state (NQS) ansatz and associated functions. It handles parameter transformation,
        function initialization, and batching.
        Args:
            batch_size (Optional[int]): The batch size to use for sampling configurations. If not provided,
                the default batch size (`self._batch_size`) is used.
        Returns:
            Callable: A JIT-compiled function with the signature:
                wrapped(y, t, configs, configs_ansatze, probabilities, int_step=0)
            where:
                - y: Flat parameter vector.
                - t: Current step or time (used as a static argument for JIT).
                - configs: Sampled configurations.
                - configs_ansatze: Ansatz-specific configurations.
                - probabilities: Probability weights for the configurations.
                - int_step: (Optional) Integer step counter (default: 0).
        Notes:
            - The returned function automatically transforms flat parameters into the required tree structure.
            - All necessary NQS functions are (re)initialized to ensure correct compilation.
            - This wrapper is intended for use in iterative algorithms such as VMC or optimization loops.
        """
        
        batch_size                  = batch_size if batch_size is not None else self._batch_size
        
        #! reinitialize the functions - it may happen that they recompile but that doesn't matter
        self._init_functions(batch_size=batch_size)
        apply_fn                    = self._ansatz_func
        local_energy_fun            = self._local_en_func
        flat_grad_fun               = self._flat_grad_func
        apply_fun                   = self._apply_func
        
        #! Sample the configurations
        # self._init_param_metadata()
        shapes, sizes, iscpx        = self._sample_for_shapes(apply_fn, local_energy_fun, flat_grad_fun, apply_fun, batch_size=self._batch_size, t=0)
        # shapes, sizes, iscpx = self._params_shapes, self._params_sizes, self._params_iscpx
        tree_def, flat_size, slices = self._params_tree_def, self._params_total_size, net_utils.jaxpy.prepare_slice_info(shapes, sizes, iscpx)

        #! Create the function to be used
        single_step_jax = partial(
                    NQS._single_step_jax,
                    apply_fn            =   apply_fn,
                    local_energy_fun    =   local_energy_fun,
                    flat_grad_fun       =   flat_grad_fun,
                    apply_fun           =   apply_fun,
                    batch_size          =   batch_size
                )
        
        @partial(jax.jit, static_argnames=('t',))
        def wrapped(y, t, configs, configs_ansatze, probabilities, int_step = 0):
            # if isinstance(y, jnp.ndarray):
            params = net_utils.jaxpy.transform_flat_params_jit(y, tree_def, slices, flat_size)
            # else:
            #     params = y
            return single_step_jax(params,
                                    configs,
                                    configs_ansatze,
                                    probabilities,
                                    t               = t,
                                    int_step        = int_step)
        return wrapped

    #####################################
    
    def train(self,
            nsteps  : int = 1,
            verbose : bool = False,
            use_sr  : bool = True,
            **kwargs) -> list:
        """
        Train the NQS solver for a specified number of steps.

        Parameters:
            nsteps: Number of training steps.
            verbose: Whether to print progress.
            use_sr: Whether to use stochastic reconfiguration.

        Returns:
            List of mean energies for each step.
        """
        energies = []
        for step in range(nsteps):
            if self._isjax:
                pass
                # self._state, mean_energy, std_energy, _ = self.single_step_jax(
                #         params      = self._net.get_params(),
                #         configs     = 
            else:
                self._params, mean_energy, std_energy, _ = self.train_step_np(
                    params=self._params,
                    sampler=self._sampler,
                    hamiltonian=self._hamiltonian,
                    batch_size=self._batch_size,
                    use_sr=use_sr,
                    reg=kwargs.get('reg', 1e-7),
                    lr=kwargs.get('lr', 1e-2))

            energies.append(mean_energy)
            if verbose:
                print(f"Step {step + 1}/{nsteps}: Mean Energy = {mean_energy}, Std Energy = {std_energy}")

        return energies
    
    #####################################
    #! LOG_PROBABILITY_RATIO
    #####################################
    
    @staticmethod
    def log_prob_ratio( top_log_ansatz   :   Callable,
                        top_params,
                        bot_log_ansatz   :   Callable,
                        bot_params,
                        states           :   jnp.ndarray):
        top_log = top_log_ansatz(top_params, states)
        bot_log = bot_log_ansatz(bot_params, states)
        return top_log - bot_log
    
    #####################################
    #! STATE MODIFIER
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
        
        # reset the ansatz function
        self._ansatz_func, self._params = self._net.get_apply(self._isjax)
    
    def set_modifier(self, modifier: Union[Operator, OperatorFunction], **kwargs):
        '''
        Set the state modifier.
        '''
        
        #! The modifier should be an inverse mapping - for a given state, it shall return all the states that lead to it 
        #! through the application of the operator.
        self._modifier = modifier
        self.log(f"State modifier set to {modifier}.", log='info', lvl = 2, color = 'blue')
        
        if isinstance(modifier, Operator):
            if self._isjax and hasattr(modifier, 'jax'):
                self._modifier_func = modifier.jax
            elif hasattr(modifier, 'npy'):
                self._modifier_func = modifier.npy
            else:
                raise ValueError("The operator does not have a JAX or NumPy implementation.")
        else:
            self._modifier_func = modifier
        model_callable, self._params = self._net.get_apply(self._isjax)
        # it modifies this ansatz function, one needs to recompile and take it
        def _ansatz_func(params, x):
            # log_psi = model_callable(params, x)
            # If x is a batch of states, apply modifier to each state using jax.lax.map
            def apply_mod(s):
                return self._modifier_func(s)
            # Returns (st, val) for each s in x
            st, val = jax.lax.map(apply_mod, x)

            # st        : (M, *shape)
            # weights   : (M,)
            # compute log ψ for each
            log_psi_mod = jax.vmap(lambda s: model_callable(params, s))(st)[:, 0]
            # multiply by the value
            log_psi_mod = log_psi_mod + jnp.log(val.astype(log_psi_mod.dtype))
            # now return the sum simply - this shall be sum of ansatzes but they are logarithmic
            # so we need to multiply them log(x_1) + log(x_2) = log(x_1 * x_2)
            return jnp.array([jnp.prod(log_psi_mod, axis=0)])

        if self._isjax:
            self._ansatz_func = jax.jit(_ansatz_func)
        else:
            self._ansatz_func = _ansatz_func
    
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
    
    def set_weights(self, weights: dict = None, file_path: str = None, fmt: str = "h5"):
        """
        Set the weights of the NQS solver.

        Args:
            weights:
                dict of parameter arrays to set.
            file_path:
                path to load weights from ('.h5' or '.json').
            fmt:
                format of file ('h5' or 'json').
        """
        if file_path:
            self._params = self.load_weights(file_path, fmt)
        elif weights is not None:
            self._params = weights
        else:
            raise ValueError("Provide either weights dict or file_path.")
        self.set_params(self._params)
    
    def update_weights(self, f: any = None, weights: dict = None):
        """
        Update the weights of the NQS solver.

        Args:
            f: scalar multiplier, callable, or None. If scalar, multiplies all params by f.
               If callable, applies f(param) to each param.
            weights: dict of same structure; added elementwise to current params.
        """
        # apply function f
        if f is not None:
            if isinstance(f, (int, float)):
                self._params = jax.tree_map(lambda x: x * f, self._params)
            elif callable(f):
                self._params = jax.tree_map(lambda x: f(x), self._params)
            else:
                raise ValueError("f must be a number or a callable.")
        # add weights
        if weights is not None:
            self._params = jax.tree_multimap(lambda x, y: x + y, self._params, weights)
        self._net.set_params(self._params)
    
    def save_weights(self, file_path: str, fmt: str = "h5") -> str:
        """
        Save current weights to disk.

        Args:
            file_path: destination path (without extension).
            fmt: 'h5' or 'json'.
        Returns:
            Full path of saved file.
        """
        path = file_path + (".h5" if fmt == "h5" else ".json")
        if fmt == "h5":
            with h5py.File(path, "w") as f:
                for k, v in self._params.items():
                    f.create_dataset(k, data=jax.device_get(v))
        elif fmt == "json":
            serial = {k: jax.device_get(v).tolist() for k, v in self._params.items()}
            with open(path, "w") as f:
                json.dump(serial, f)
        else:
            raise ValueError("Unsupported format: {}".format(fmt))
        return path
    
    def load_weights(self, file_path: str, fmt: str = "h5") -> dict:
        """
        Load weights from disk.

        Args:
            file_path: path to file (with extension).
            fmt: 'h5' or 'json'.
        Returns:
            Dict of parameter arrays.
        """
        path = file_path
        if fmt == "h5":
            with h5py.File(path, "r") as f:
                loaded = {k: jnp.array(f[k][()]) for k in f.keys()}
        elif fmt == "json":
            with open(path, "r") as f:
                data = json.load(f)
            loaded = {k: jnp.array(v) for k, v in data.items()}
        else:
            raise ValueError("Unsupported format: {}".format(fmt))
        return loaded
    
    #####################################
    
    def get_params(self, unravel: bool = False) -> Any:
        """Returns the current parameters from the network object."""
        params = self._net.get_params()
        if unravel:
            return jnp.concatenate([p.ravel() for p in tree_flatten(params)[0]])
        return params
    
    def set_params(self, new_params: Any, shapes: list = None, sizes: list = None, iscpx: bool = False):
        """Sets new parameters in the network object."""
        params = new_params
        # check if the parameters are provided
        if params is None:
            params = self._net.get_params()
        elif isinstance(params, jnp.ndarray):
            shapes = shapes if shapes is not None else self._params_shapes
            sizes  = sizes if sizes is not None else self._params_sizes
            iscpx  = iscpx if iscpx is not None else self._params_is_cpx
            params = self.transform_flat_params(params, shapes, sizes, iscpx)

        # set the parameters
        self._net.set_params(params)
    
    #####################################
    #! GETTERS AND PROPERTIES
    #####################################
    
    @property
    def net(self):
        '''
        Return the neural network.
        '''
        return self._net
    
    @property
    def sr(self):
        '''
        Return the stochastic reconfiguration object.
        '''
        return self._stochastic_reconf
    
    @property
    def sampler(self):
        '''
        Return the sampler object.
        '''
        return self._sampler
    
    @property
    def num_params(self):
        '''
        Return the number of parameters in the neural network.
        '''
        return self._params_total_size
    
    @property
    def nvisible(self):
        '''
        Return the number of visible units in the neural network.
        '''
        return self._nvisible
    
    @property
    def size(self):
        '''
        Return the size of the neural network.
        '''
        return self._size
    
    @property
    def batch_size(self):
        '''
        Return the batch size.
        '''
        return self._batch_size
    
    @property
    def backend(self):
        '''
        Return the backend used for the neural network.
        '''
        return self._backend
    
    #! Callers
    
    @property
    def ansatz(self):
        '''
        Return the neural network apply function.
        '''
        return self._ansatz_func
    
    @property
    def flat_grad(self):
        '''
        Return the flat gradient function.
        '''
        return self._flat_grad_func
    
    @property
    def local_energy(self):
        '''
        Return the local energy function.
        '''
        return self._local_en_func
    
    #####################################

    def clone(self):
        '''
        Clone the NQS solver.
        '''
        return NQS(self._net.clone(), self._sampler.clone(), self._backend, **self._kwargs)
    
    def swap(self, other):
        return super().swap(other)
    
    #####################################
    
    def __repr__(self):
        return f"NQS(ansatz={self._net},sampler={self._sampler},backend={self._backend_str})"
    
    def __str__(self):
        return f"NQS(ansatz={self._net},sampler={self._sampler},backend={self._backend_str})"    

    #####################################
    
    def eval_observables(
        self,
        operators      : Sequence,                          # list of AttrAccess wrappers (sig_z, sig_x, …)
        true_values    : Optional[Sequence[float]] = None,  # same length or None
        *,
        n_chains       : int,
        n_samples      : int,
        batch_size     : int,
        logger,
        get_energy     : bool = False,
        plot           : bool = False,
        **plot_kwargs):
        """
        Sample once from `nqs` and evaluate a set of observables.

        Returns
        -------
        results : dict
            keys: operator objects (as given); values: dict(mean, std, raw)
        energy  : dict | None
            mean/std/raw for local energy if `energy_fun` is provided
        timings : dict
            elapsed times for 'sample', 'observables', and 'energy' phases
        """
        from general_python.common.plot import Plotter
        
        timings = {}
        params  = self.get_params()
        true_en = plot_kwargs.get('true_en', None)

        #! 1) sampling
        ((_,_), (configs, configs_ans), probs), timings['sample'] = timeit(self.sample, num_chains=n_chains, num_samples=n_samples)


        #! 2) observables
        results = {}
        for i, op in enumerate(operators):
            (vals, mu, sig), timings_op = timeit(
                self.evaluate_fun_jax,
                func           = op.jax,
                states         = configs,
                probabilities  = probs,
                logproba_in    = configs_ans,
                logproba_fun   = self.ansatz,
                parameters     = params,
                batch_size     = batch_size,
            )
            timings[f"obs_{i}"]     = timings_op
            results[op]             = dict(raw=vals, mean=mu, std=sig)
            color                   = ["red","blue","green","orange","purple","brown"][i%6]
            logger.info(f"{op}: ⟨O⟩ = ({mu:.4f}) ± ({sig:.4f})  (N={len(vals)})", color=color)

            #! compare to true value
            if true_values is not None and true_values[i] is not None:
                ref = true_values[i]
                rel = abs(mu-ref)/abs(ref)*100
                logger.info(f"ref = {ref:.4f} - rel.err = {rel:.2f} %", lvl=2)

        #! 3) energy (optional)
        energy = None
        if get_energy:
            (e_vals, e_mu, e_sig), timings['energy'] = timeit(
                self.evaluate_fun_jax,
                func           = self.local_energy,
                states         = configs,
                probabilities  = probs,
                logproba_in    = configs_ans,
                logproba_fun   = self.ansatz,
                parameters     = params,
                batch_size     = batch_size,
            )
            energy = dict(raw=e_vals, mean=e_mu, std=e_sig)
            logger.info(f"Energy: E = ({e_mu:.4e}) ± ({e_sig:.4f}) (N={len(e_vals)})", color='cyan')
            if true_en is not None:
                rel = abs(e_mu-true_en)/abs(true_en)*100
                logger.info(f"ref = {true_en:.4e} - rel.err = {rel:.2f} %", lvl=2)

        #! 4) optional quick-look plot
        if plot:
            bins    = plot_kwargs.get('bins', 50)
            n_ops   = len(operators)
            fig, ax = Plotter.get_subplots(
                nrows       = n_ops + (energy is not None),
                ncols       = 1,
                figsize     = (4, 1.5*(n_ops+1)),
                dpi         = 120,
            )

            for i, op in enumerate(operators):
                vals            = np.real(np.asarray(results[op]['raw']))
                mean            = np.nanmean(vals)
                # std             = np.nanstd(vals)
                binsin          = 30
                hist, binsin    = np.histogram(vals, bins=binsin, density=True)
                ax[i].hist(vals, bins=binsin, density=True, color='gray', alpha=0.7)
                ax[i].stairs(hist, binsin, color='gray', alpha=0.7)
                Plotter.vline(ax[i], mean, color='k', lw=1, label=f'{op}$ = {mean:.3f}$')
                if true_values is not None and true_values[i] is not None:
                    Plotter.vline(ax[i], true_values[i], color='r', lw=1, alpha=0.5, label=f'$O_{{\\rm true}}={true_values[i]:.3f}$')
                # minmaxop    = (mean - std, mean + std)
                minmaxop    = None
                Plotter.set_ax_params(ax[i], ylabel=r'$P(\langle O \rangle)$', xlim=minmaxop, yscale='log')
                Plotter.set_legend(ax[i], fontsize=8)

            if energy is not None:
                idx             = -1
                vals            = np.real(np.asarray(energy['raw']))
                mean            = np.mean(vals)
                std             = np.std(vals)
                minimum         = np.min(vals)
                maximum         = np.max(vals)
                binsin          = min(len(vals)//10, bins)
                hist, binsin    = np.histogram(vals, bins=binsin, density=True)
                ax[idx].stairs(hist, binsin, color='gray', alpha=0.7)
                Plotter.vline(ax[idx], mean, color='k', lw=1, label=f'$\\bar E ={mean:.3e}$')
                if true_en is not None:
                    Plotter.vline(ax[idx], true_en, color='r', lw=1, alpha=0.5, label=f'$E_{{\\rm true}}={true_en:.3e}$')
                Plotter.set_ax_params(ax[idx], xlabel=r'$\langle O \rangle$', ylabel=r'$P(\langle E_{\rm loc} \rangle)$', xlim=None, yscale='log')
                if mean > (minimum + maximum) / 2:
                    axin = ax[idx].inset_axes([0.2, 0.2, 0.3, 0.4])
                    Plotter.set_legend(ax[idx], fontsize=8, loc='upper left')
                else:
                    axin = ax[idx].inset_axes([0.6, 0.2, 0.3, 0.4])
                    Plotter.set_legend(ax[idx], fontsize=8, loc='upper right')
                axin.scatter(np.arange(len(vals)), vals, s=0.5, color='gray', alpha=0.7)
                if true_en is not None:
                    Plotter.hline(axin, true_en, color='r', lw=1, alpha=0.5, label=f'$ref={true_en:.3e}$')
                Plotter.set_ax_params(axin, xlim=(0, len(vals)), yscale='linear', ylabel=r'$\langle E_{\rm loc} \rangle$')
                Plotter.set_label_cords(axin, which='x', inX=0.5, inY=1.2)
                
            fig.suptitle(f"Sampled observables (N={len(configs)})", fontsize=10)
            fig.tight_layout()
            directory = os.path.join(os.curdir, 'data', 'nqs_train', 'figs')
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(os.curdir + '/data/nqs_train/figs/obs.png', dpi=300)
        for k, v in timings.items():
            if isinstance(v, float):
                logger.info(f"{k}: {v:.2e} s", color='green')
            else:
                logger.info(f"{k}: {v:.2e} s", color='blue')
        return results, energy, timings

    #####################################
    
#########################################
#! NQS Lower States
#########################################

class NQSLower:
    def __init__(self,
                lower_nqs_instances     : List[NQS],                # List of NQS objects for lower states
                lower_betas             : List[float],              # Penalty terms beta_i
                parent_nqs              : NQS):

        if len(lower_nqs_instances) != len(lower_betas):
            raise ValueError("Number of lower NQS instances must match number of betas.")
        
        self._backend               = parent_nqs.backend
        self._isjax                 = parent_nqs.isjax
        
        # assert that all are the same backend
        if not all([nqs.backend == parent_nqs.backend for nqs in lower_nqs_instances]):
            raise ValueError("All lower NQS instances must have the same backend as the parent NQS.")
        
        self._parent_nqs                    = parent_nqs
        self._parent_apply_fn               = parent_nqs.ansatz
        self._parent_params                 = parent_nqs.get_params()                               # will likely be updated during training
        self._parent_evaluate               = parent_nqs.evaluate_fun_jax if self._isjax else parent_nqs.evaluate_fun_np
        self._log_p_ratio_fn                = parent_nqs.log_prob_ratio        
        
        #! handle the lower states
        self._lower_nqs             = lower_nqs_instances
        self._lower_betas           = self._backend.array(lower_betas)
        self._num_lower_states      = len(lower_nqs_instances)
        self._is_set                = self._num_lower_states > 0

        if not self._is_set:
            return

        # Store apply functions and parameters for each lower state
        self._lower_apply_fns       = [nqs.ansatz for nqs in self._lower_nqs]               # this is static as it is not updated during training
        self._lower_params          = [nqs.get_params() for nqs in self._lower_nqs]         # this is static as it is not updated during training
        if self._isjax:
            self._lower_evaluate    = [nqs.evaluate_fun_jax for nqs in self._lower_nqs]     # this is static as it is not updated during training
        else:
            self._lower_evaluate    = [nqs.evaluate_fun_np for nqs in self._lower_nqs]

        #! Placeholder for ratios - these would be computed during the excited state's MC sampling
        # Shape: (num_lower_states, num_samples_excited_state)
        self._ratios_psi_exc_div_psi_lower = None       # Psi_W / Psi_W_j (current excited / lower_j)
        self._ratios_psi_lower_div_psi_exc = None       # Psi_W_j / Psi_W (lower_j / current excited)

        # The C++ code has `train_lower_` for MC parameters for sampling *within* lower states.
        # For the JAX implementation, this might translate to parameters for sampling
        # from each P(s) ~ |psi_lower_j(s)|^2 if needed for <Psi_W / Psi_W_j>_j terms.
        # This part is complex as it implies separate MC runs or combined sampling.

    def get_p_ratio_wrapper(self, lower_idx: int):
        """
        Wrapper to compute log(Psi_1 / Psi_2) for a given lower state.
        """
        
        if 0 <= lower_idx < self._num_lower_states:
            # compute log(Psi_exc / Psi_lower_j)
            ansatz_top = self._parent_apply_fn
            params_top = self._parent_nqs.get_params()          # not static - will be updated during training
            
            ansatz_low = self._lower_apply_fns[lower_idx]
            params_low = self._lower_params[lower_idx]
        else:
            # reverse order - compute log(Psi_lower_j / Psi_exc)
            ansatz_top = self._lower_apply_fns[lower_idx]
            params_top = self._lower_params[lower_idx]

            ansatz_low = self._parent_apply_fn
            params_low = self._parent_nqs.get_params()          # not static - will be updated during training

        @jax.jit
        def wrapme(states):
            log_ratio = self._log_p_ratio_fn(ansatz_top, params_top, ansatz_low, params_low, states)
            return jnp.exp(log_ratio)
        return wrapme

    ######################################
    #! Properties and Getters
    ######################################
    
    @property
    def is_set(self) -> bool:
        return self._is_set

    @property
    def num_lower_states(self) -> int:
        return self._num_lower_states

    @property
    def lower_betas(self) -> jnp.ndarray:
        return self._lower_betas

    ######################################
    #! Getters
    ######################################

    def get_lower_state_ansatz_val(self, lower_idx: int, states: jnp.ndarray) -> jnp.ndarray:
        """
        Computes log(psi_lower_j(s)) for a given lower state j and batch of states s.
        
        lower_idx:
            index of the lower state.
        states: 
            batch of sampled configurations.
        Returns:
            log(psi_lower_j(s)) for all samples s.
        """
        if not self._is_set or lower_idx >= self._num_lower_states:
            raise IndexError("Lower state index out of bounds.")
        apply_fn    = self._lower_apply_fns[lower_idx]
        params      = self._lower_params[lower_idx]
        return apply_fn(params, states)

    # --- Methods to compute quantities needed for excited state ---

    @partial(jax.jit, static_argnums=(0,))
    def _compute_log_psi_ratios_for_single_lower_state(self,
                                                       lower_idx: int,
                                                       excited_log_psi_s: jnp.ndarray,
                                                       states_s: jnp.ndarray) -> jnp.ndarray:
        """
        Computes log( Psi_excited(s) / Psi_lower_j(s) ) for all samples s.
        excited_log_psi_s: log amplitudes of the current excited state for the samples.
        states_s: the sampled configurations.
        """
        log_psi_lower_j_s = self.get_lower_state_ansatz_val(lower_idx, states_s)
        return excited_log_psi_s - log_psi_lower_j_s # log(A/B) = logA - logB

    def compute_all_log_psi_ratios(self,
                                   excited_log_psi_s: jnp.ndarray, # log psi_exc(s_i) for N_samples
                                   states_s: jnp.ndarray           # configurations s_i (N_samples, n_visible)
                                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes log_ratio_exc_div_lower = log(Psi_exc(s) / Psi_lower_j(s)) and
                 log_ratio_lower_div_exc = log(Psi_lower_j(s) / Psi_exc(s))
        for all lower states j and all samples s.

        Returns:
            Tuple of JAX arrays:
            - log_ratios_exc_div_lower (num_lower_states, num_samples)
            - log_ratios_lower_div_exc (num_lower_states, num_samples)
        """
        if not self.is_set:
            return jnp.array([]), jnp.array([])

        all_log_ratios_exc_div_lower = []
        for j in range(self.num_lower_states):
            log_ratios_j = self._compute_log_psi_ratios_for_single_lower_state(j, excited_log_psi_s, states_s)
            all_log_ratios_exc_div_lower.append(log_ratios_j)

        log_ratios_exc_div_lower_arr = jnp.stack(all_log_ratios_exc_div_lower, axis=0)
        log_ratios_lower_div_exc_arr = -log_ratios_exc_div_lower_arr # log(B/A) = -log(A/B)

        return log_ratios_exc_div_lower_arr, log_ratios_lower_div_exc_arr

    # The C++ `collectLowerEnergy` seems to sample from P_j(s) = |psi_j(s)|^2
    # and then computes < P_exc(s) >_j where P_exc is |psi_exc(s')/psi_exc(s)|^2 * <s|H_proj|s'>.
    # This is intricate. For JAX, we'll likely compute everything using samples from P_exc(s).
    # The energy penalty term H_i = beta_i * |f_i><f_i| / <f_i|f_i> becomes
    # E_penalty_i(s) = beta_i * sum_s' <s| |f_i><f_i| |s'> / <f_i|f_i> * (psi_exc(s')/psi_exc(s))
    #                = beta_i * |<s|f_i>|^2 / <f_i|f_i> * (psi_exc(f_i)/psi_exc(s))
    # If f_i is a single configuration, <s|f_i> = delta_s,f_i.
    # More generally, if f_i is a known wavefunction psi_lower_i:
    # E_penalty_i(s) = beta_i * |psi_lower_i(s)|^2 / <|psi_lower_i|^2> * (sum_s' psi_lower_i(s')^* psi_exc(s') / (psi_lower_i(s)^* psi_exc(s)))
    # This simplifies to: beta_i * |psi_lower_i(s) / psi_exc(s)|^2. (Assuming <|psi_lower_i|^2> = 1 after normalization)
    # This is the local energy contribution from the penalty term for state i.
    # E_penalty_local(s) = sum_i beta_i * |psi_lower_i(s) / psi_exc(s)|^2
    #                  = sum_i beta_i * exp(2 * Re[log(psi_lower_i(s)) - log(psi_exc(s))])

    @partial(jax.jit, static_argnums=(0,))
    def compute_local_penalty_energies(self,
                                       log_psi_exc_s: jnp.ndarray, # (N_samples,)
                                       states_s: jnp.ndarray       # (N_samples, n_visible)
                                       ) -> jnp.ndarray:           # (N_samples,)
        """
        Computes the local energy contribution from all penalty terms for each sample.
        E_penalty_local(s) = sum_i beta_i * |psi_lower_i(s) / psi_exc(s)|^2
        """
        if not self.is_set:
            return jnp.zeros_like(log_psi_exc_s)

        _, log_ratios_lower_div_exc = self.compute_all_log_psi_ratios(log_psi_exc_s, states_s)
        # log_ratios_lower_div_exc has shape (num_lower_states, N_samples)

        # |psi_lower_i(s) / psi_exc(s)|^2 = exp(2 * Re[log_ratios_lower_div_exc_i(s)])
        abs_sq_ratios = jnp.exp(2 * jnp.real(log_ratios_lower_div_exc)) # (num_lower_states, N_samples)

        # Multiply by betas and sum over lower states
        # betas shape (num_lower_states,), need to reshape for broadcasting
        penalties_per_lower_state = self._lower_betas[:, None] * abs_sq_ratios # (num_lower_states, N_samples)
        total_local_penalty = jnp.sum(penalties_per_lower_state, axis=0) # (N_samples,)
        return total_local_penalty

#########################################
#! TESTS
#########################################

def test_net_ansatz(nqs: NQS, nsamples = 10):
    '''
    Tests the NQS ansatz of the provided neural quantum state solver.
    '''
    
    ns          = nqs.shape[0]
    states      = np.random.choice([-1,1], size=(nsamples, ns), replace=True)
    if nqs.isjax:
        states  = jnp.array(states)
    net         = nqs.net
    
    # params      = net.get_params()
    ansatz      = net(states)
    ansatz      = ansatz.reshape(-1, 1)
    return ansatz, ansatz.shape

#########################################