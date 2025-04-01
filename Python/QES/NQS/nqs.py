import numpy as np
import inspect
import numba

# typing and other imports
from typing import Union, Tuple, Union, Callable, Optional
from functools import partial

# from general_python imports
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
from general_python.algebra.ran_wrapper import choice, randint, uniform
from general_python.common.directories import Directories
import general_python.ml.networks as Networks 
import general_python.common.binary as Binary

# from hilbert
import Algebra.hilbert as Hilbert

# JAX imports
if _JAX_AVAILABLE:
    import jax
    from jax import jit as jax_jit, grad, vmap, random
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
    from flax.core.frozen_dict import freeze, unfreeze

#########################################

from Solver.MonteCarlo.montecarlo import MonteCarloSolver, McsTrain, McsReturn, Sampler
from Algebra.Operator.operator import Operator, OperatorFunction

# Hamiltonian imports
from Algebra.hamil import Hamiltonian

#########################################

# for the gradients and stuff
import general_python.ml.net_impl.utils.net_utils as net_utils
import general_python.ml.net_impl.net_general as net_general
import general_python.algebra.solvers.stochastic_rcnfg as sr

# schedulers and preconditioners
import general_python.algebra.preconditioners as precond_mod
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
                mu          : float                             = 0,
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
        #! set the lower states
        #######################################
        
        if lower_states is not None:
            self._lower_states = NQSLowerStates(lower_states, lower_betas, self)
        else:
            self._lower_states = None
        
        #######################################
        #! state modifier (for later)
        #######################################
        
        self._modifier          = None
        
        #######################################
        #! handle the network
        #######################################
        
        self._batch_size        = batch_size
        self._initialized       = False
        self._weights           = None
        self._dtype             = None
        self._net               = self._choose_network(net, **kwargs)   # initialize network type
        self._analytic          = False
        self._holomorphic       = True
        self.init_network()
        
        #######################################
        #! handle gradients
        #######################################
        
        self._flat_grad_fun     = None
        self._dict_grad_type    = None
        self._init_gradients()
        
        #######################################
        #! handle the functions
        #######################################
        
        self._ansatz_func       = None
        self._local_en_func     = None
        self._eval_func         = None
        # if the network has an analytic gradient
        self._grad_func         = None
        
        #######################################
        #! handle the optimizer
        #######################################
        
        self._preconditioner    = precond_mod.choose_precond(
            precond_type = kwargs.get('preconditioner', None),
            backend      = self._backend)
        
        self._lr_scheduler      = kwargs.get('lr_scheduler', None)
        self._reg_scheduler     = kwargs.get('reg_scheduler', None)
        
        self._stochastic_reconf = sr.StochasticReconfiguration(None, self._backend)
        
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
            if not self._isjax:
                raise ValueError(self._ERROR_JAX_WITH_FLAX)
            
        return Networks.choose_network(network_type=net, input_shape=self._shape, backend=self._backend, dtype=self._dtype, **kwargs)
    
    #####################################
    #! INITIALIZATION OF THE NETWORK AND FUNCTIONS
    #####################################
    
    def reset(self):
        """
        Resets the initialization state of the object and reinitializes the underlying network.
        This method marks the object as not initialized and forces a reinitialization of the associated
        neural network by calling its `force_init` method.
        """
        
        self._initialized       = False
        self._net.force_init()
        
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
            self._flat_grad_fun : A function for computing gradients in a flattened format.
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

        self._flat_grad_fun, self._dict_grad_type = net_utils.decide_grads(iscpx=self._iscpx,
                            isjax=self._isjax, isanalitic=self._analytic, isholomorphic=self._holomorphic)

    # ---

    def _init_functions(self):
        '''
        Initialize the functions for the gradient and network evaluation.
        1. Check if the backend is JAX or NumPy.
        2. If so, set the evaluation and gradient functions to the appropriate JAX or NumPy functions.
        '''
        
        # set the evaluation function
        self._ansatz_func, self._params = self._net.get_apply(use_jax=self._isjax)
        
        if self._isjax:
            self._eval_func         = net_utils.jaxpy.eval_batched_jax
            self._grad_func         = self._net.get_gradient(use_jax=True)
            # jit the ansatz function
            self._ansatz_func       = self._ansatz_func
        else:
            self._eval_func         = net_utils.numpy.eval_batched_np
            self._grad_func         = self._net.get_gradient(use_jax=False)
            # numba the ansatz function
            self._ansatz_func       = self._ansatz_func
        
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
            self._holomorphic       = self._net.check_holomorphic()
            self.log(f"Network is holomorphic: {self._holomorphic}", log='info', lvl = 2, color = 'blue')
            self._analytic          = self._net.has_analitic_grad
            self.log(f"Network has analytic gradient: {self._analytic}", log='info', lvl = 2, color = 'blue')
            
            # check the shape of the weights
            self._paramshape        = self._net.shapes
            
            # number of parameters
            self._nparams           = self._net.nparams
            # if self._isjax:
            #     self._nparams = jnp.sum(jnp.array([p.size for p in tree_flatten(self.parameters["params"])[0]]))
            # else:
            #     self._nparams = np.sum(np.array([p.size for p in flatten_func(self.parameters["params"])[0]]))
    
    #####################################
    #! EVALUATION OF THE ANSATZ
    #####################################
    
    def _eval_jax(self, states, batch_size = None, params = None):
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
            current parameters stored in self._weights.
        Returns
        -------
        array_like
            The output of the neural network for the given states, representing the log of the 
            wavefunction amplitudes.
        """

        if params is None:
            params = self._weights
            
        if batch_size is None:
            batch_size = self._batch_size
        
        # evaluate the network (log ansatz) using JAX
        return net_utils.jaxpy.eval_batched_jax(batch_size, self._ansatz_func, params, states)
    
    def _eval_np(self, states, batch_size = None, params = None):
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
            current parameters stored in self._weights.
        Returns
        -------
        array_like
            The output of the neural network for the given states, representing the log of the 
            wavefunction amplitudes.
        """
        
        if params is None:
            params = self._weights
            
        if batch_size is None:
            batch_size = self._batch_size
        
        # evaluate the network (log ansatz) using NumPy
        return net_utils.numpy.eval_batched_np(batch_size, self._ansatz_func, params, states)
    
    def evaluate(self, states, batch_size = None, params = None):
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
        if self._isjax:
            return self._eval_jax(states, batch_size=batch_size, params=params)
        return self._eval_np(states, batch_size=batch_size, params=params)
    
    def __call__(self, states, **kwargs):
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
        return self.evaluate(states, **kwargs)
    
    #####################################
    #! EVALUATE FUNCTION VALUES
    #####################################
    
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
            func (Callable)                                 : The function to be evaluated.
            states (Union[np.ndarray, jnp.ndarray])         : The input states for the function.
            probabilities (Union[np.ndarray, jnp.ndarray])  : The probabilities associated with the states.
            logproba_in (Union[np.ndarray, jnp.ndarray])    : The logarithm of the probabilities for the input states.
            logproba_fun (Callable)                         : A function to compute the logarithm of probabilities.
            parameters (Union[dict, list, np.ndarray])      : Parameters to be passed to the function.
            batch_size (Optional[int], optional)            : The size of batches for evaluation. 
                                                            If None, the function is evaluated without batching. Defaults to None.
            is_jax (bool, optional)                         : Flag indicating if JAX is used for computation. Defaults to True.
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
        
        # get from kwargs
        batch_size      = kwargs.get('batch_size', self._batch_size)
        
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
                probabilities = self._backend.ones_like(ansatze)
            
            output = [self._evaluate_fun(func           = f,
                                        states          = states,
                                        probabilities   = probabilities,
                                        logproba_in     = ansatze,
                                        logproba_fun    = self._ansatz_func,
                                        parameters      = self._net.get_params(),
                                        batch_size      = batch_size,
                                        is_jax          = self._isjax) for f in functions]
        else:
            # get other parameters from kwargs
            num_samples = kwargs.get('num_samples', None)
            num_chains  = kwargs.get('num_chains', None)
            
            # otherwise, we shall use the sampler
            (states, ansatze), probabilities, output    = \
                    self._evaluate_fun_s(func           = functions,
                                        sampler         = self._sampler,
                                        num_samples     = num_samples,
                                        num_chains      = num_chains,
                                        logproba_fun    = self._ansatz_func,
                                        parameters      = self._net.get_params(),
                                        batch_size      = batch_size,
                                        is_jax          = self._isjax)
        
        # check if the output is a list
        if isinstance(output, list) and len(output) == 1:
            output = output[0]
        return (states, ansatze), probabilities, output

    #####################################
    #! SAMPLE
    #####################################
    
    def sample(self, num_samples = 1, num_chains = 1):
        '''
        Sample the NQS using the provided sampler. This will return
        the sampled states and the corresponding probabilities.
        Parameters:
            num_samples : The number of samples to generate.
            num_chains  : The number of chains to use for sampling.
            kwargs      : Additional arguments for the sampler.
        Returns:
            The sampled states and the corresponding probabilities.
            (last configs, last ansatze), (all configs, all ansatze), (all probabilities)
        '''
        return self._sampler.sample(num_samples=num_samples, num_chains=num_chains)    
    
    #####################################
    #! GRADIENTS
    #####################################
    
    @staticmethod
    @partial(jax.jit, static_argnums=(0,2,4))
    def _grad_jax(net_apply, params, batch_size, states, flat_grad_fun):
        '''
        Compute the gradients of the ansatz logarithmic wave-function using JAX. 
        
        Compute gradient of the logarithmic wave function coefficients, \
        :math:`\\nabla\\ln\\psi(s)`, for computational configurations :math:`s`.
        '''
        
        # create the batches
        sb = net_utils.jaxpy.create_batches_jax(states, batch_size)

        # compute the gradients using JAX's vmap and scan
        # use the provided flat_grad_fun to compute the gradients
        # this is a function that computes the gradients of the network
        def scan_fun(c, x):
            return c, jax.vmap(lambda y: flat_grad_fun(net_apply, params, y), in_axes=(0,))(x)
        
        # use jax's scan to compute the gradients of the logarithmic wave function
        g = jax.lax.scan(scan_fun, None, sb)[1]
        g = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), g)
        
        # only take the non-padded values
        return tree_map(lambda x: x[:states.shape[0]], g)
    
    @staticmethod
    def _grad_np(net, params, batch_size, states, flat_grad):
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
                return self._grad_jax(self._ansatz_func, params, batch_size, states, self._flat_grad_fun)
            return self._grad_jax(self._grad_func, params, batch_size, states, self._flat_grad_fun)
        
        if not self._analytic:
            return self._grad_np(self._ansatz_func, params, batch_size, states, self._flat_grad_fun)
        return self._grad_np(self._grad_func, params, batch_size, states, self._flat_grad_fun)
        
    #####################################
    #! UPDATE PARAMETERS
    #####################################
    
    def _update_unflatten(self, d_par):
        
        # Reshape parameter update according to net tree structure
        p_tree_shape    = []
        start           = 0
        for s in self._net.shapes:
            if self._iscpx:
                p_tree_shape.append((d_par[start:start + s[0]] + 1.j * d_par[start + s[0]:start + 2*s[0]]).reshape(s[1]))
                start   += 2 * s[0]
            else:
                p_tree_shape.append(d_par[start:start + s[0]].reshape(s[1]))
                start   += s[0]
        
        # Return unflattened parameters
        return tree_unflatten(self._net.tree_def, p_tree_shape)
    
    def update_parameters(self, d_par: Union[dict, list, np.ndarray]):
        '''
        Update the parameters of the model.
        
        Parameters:
            d_par : The new parameters to update the model with, can be a dictionary,
                    list, or numpy array.
        '''
        
        unflattened     = self._update_unflatten(d_par)
        new_parameters  = tree_map(jax.lax.add,
                                    self._net.get_params()['params'],
                                    unflattened)
        
        # set the new parameters already
        self._weights   = new_parameters
        self._net.set_params(new_parameters)
    
    def optimization(self,
                    energies,
                    derivatives,
                    mean_loss   = None,
                    mean_deriv  = None,
                    use_sr      = True,
                    use_s       = False,
                    use_minsr   = False):
        '''
        Perform the optimization step using stochastic reconfiguration.
        Parameters:
            energies:
                The energy values for the configurations.
            derivatives:
                The derivatives of the energy values.
            mean_loss:
                The mean loss value (optional).
            mean_deriv:
                The mean derivative value (optional).
            use_sr:
                Flag to use stochastic reconfiguration (default: True).
            use_s:
                Flag to calculate the S matrix (default: False).
            use_minsr:
                Flag to use the minimum SR method (default: False).
        '''
        
        # set the values in the stochastic reconfiguration
        self._stochastic_reconf.set_values(energies, derivatives, mean_loss, mean_deriv, use_s, use_minsr)
        if use_sr:
            return self._stochastic_reconf.solve(use_s, use_minsr)
        return self._stochastic_reconf.forces
        
    #####################################
    #! TRAINING OVERRIDES
    #####################################
    #!TODO 
    
    def train_stop(self, i = 0, verbose = False, **kwargs):
        '''
        Stop the training process.
        '''
        
        # best            = 
        
        
        return super().train_stop(i, verbose, **kwargs)
    
    def train_step(self, i = 0, verbose = False, start_st = None, par = None, update = True, timer = None, **kwargs):
        '''
        Perform a single training step.
        '''
        
        # get the batch size
        batch_size      = kwargs.get("batch_size", self._batch_size)
        use_sr          = kwargs.get("use_sr", True)
        use_s           = kwargs.get("use_s",  False)
        use_minsr       = kwargs.get("use_minsr", False)
        
        # get the values
        (configs, _), _, (v, means, _) = self.evaluate_fun(batch_size = batch_size)

        # compute the gradients based on the same configurations
        g               = self.gradient(configs, batch_size = batch_size)
        
        # handle the lower states - if provided
        if self._lower_states is not None:
            for lower in self._lower_states:
                pass
        
        # calculate the final solution
        solution        = self.optimization(
                                    energies    = v,
                                    derivatives = g,
                                    mean_loss   = means,
                                    mean_deriv  = None,
                                    use_sr      = use_sr,
                                    use_s       = use_s,
                                    use_minsr   = use_minsr
                                )
        
        
    def train(self,
            nsteps                  = 1,
            verbose                 = False,
            start_st                = None,
            par       : McsTrain    = None,
            update    : bool        = True,
            timer                   = None,
            **kwargs):
        '''
        Train the NQS solver for a specified number of steps.
        '''
        
        energies    = []
        batch_size  = kwargs.get("batch_size", self._batch_size)
        use_sr      = kwargs.get("use_sr", True)
        use_s       = kwargs.get("use_s",  False)
        use_minsr   = kwargs.get("use_minsr", False)
        lr          = par.get("lr", 1e-2) if par is not None else 1e-2
        
        for _ in range(nsteps):
            


            # get the variational derivatives
            
            


            # update weights
            self.update_parameters(lr * solution)
            energies.append(means)
            
        return energies
    
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
    #! GETTERS AND PROPERTIES
    #####################################
    
    @property
    def net(self):
        '''
        Return the neural network.
        '''
        return self._net
    
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
        if index < 0 or index >= len(self._lower_states):
            raise IndexError("Index out of range for lower states.")
        return self._lower_states[index], self._lower_betas[index]
    
#################################################

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