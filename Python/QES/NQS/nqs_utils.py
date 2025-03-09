'''
file    : NQS/nqs_utils.py
author  : Maksymilian Kliczkowski
date    : 2025-03-01


-----------------


# Some info:

This file contains functions for creating batches of data, evaluating functions on batches, 
and handling various network types and their corresponding operations.
It includes implementations for both JAX and NumPy backends, as well as functions for computing gradients
using both analytical and numerical methods.
It also provides a wrapper for selecting the appropriate gradient function based on the input parameters.
The code is designed to be flexible and can handle complex and real-valued functions,
as well as holomorphic and non-holomorphic networks.

## Holomorphic networks:
If the function (for example, a variational ansatz for a quantum state)
is holomorphic with respect to its complex parameters, then the derivative with respect to the complex variable 
is well defined in the usual sense. The gradient can be computed using standard complex differentiation rules,
and the real and imaginary parts of the gradient are not independent—they satisfy the Cauchy-Riemann conditions.

## Non-holomorphic networks:
When we say the gradient is not holomorphic,
it means that the function is not complex differentiable in the standard sense.
In this case, the function does not satisfy the Cauchy-Riemann equations and the differentiation with respect
to the complex parameters must be done by treating the real and imaginary parts as independent variables. This results
in a gradient that generally has extra degrees of freedom compared to the holomorphic case and requires more care in its computation.

For example, if you have a wave function ansatz ψ(s;θ), where θ is complex, a holomorphic ansatz would allow
you to compute derivatives with respect to θ directly. However, if the ansatz is non-holomorphic,
you need to compute the derivatives with respect to Re(θ) and Im(θ) separately and then combine them appropriately.
'''

import numpy as np
from functools import partial
from typing import Union, Tuple, Union, Callable, Optional, Any

import collections
import time
from math import isclose
from typing import Sequence

from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
from general_python.algebra.ran_wrapper import choice, randint, uniform

if _JAX_AVAILABLE:
    import jax
    from jax import jit, grad, vmap, random
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

#########################################################################

if _JAX_AVAILABLE:
    def create_batches_jax( data        : jnp.ndarray,
                            batch_size  : int):
        """ JAX version of create_batches """
        
        # For example, if data.shape[0] is 5 and batch_size is 3, then:
        #   ((5 + 3 - 1) // 3) =  (7 // 3) = 2 batches needed, so 2*3 = 6 samples in total.
        #   Then append_data = 6 - 5 = 1 extra sample needed.
        append_data = batch_size * ((data.shape[0] + batch_size - 1) // batch_size) - data.shape[0]
        
        # Create a list of padding widths.
        # First dimension: pad (0, append_data) so that we add 'append_data' rows at the end.
        # For the rest of the dimensions, pad with (0, 0) meaning no padding.
        pads        = [(0, append_data)] + [(0, 0)] * (len(data.shape) - 1)
        # Pad the array along the first dimension using 'edge' mode (repeats the last element),
        # then reshape the array into batches.
        # The reshape uses -1 to infer the number of batches, followed by the batch_size,
        # and then the remaining dimensions.
        return jnp.pad(data, pads, mode='edge').reshape(-1, batch_size, *data.shape[1:])

def create_batches_np( data : np.ndarray,
                    batch_size : int):
    """ Numpy version of create_batches """
    append_data = batch_size * ((data.shape[0] + batch_size - 1) // batch_size) - data.shape[0]
    pads        = [(0, append_data)] + [(0, 0)] * (len(data.shape) - 1)
    return np.pad(data, pads, mode='edge').reshape(-1, batch_size, *data.shape[1:])

def create_batches( data,
                    batch_size  : int,
                    backend     : str = 'default'):
    """
    Create batches of data with a specified batch size. If the data cannot be evenly divided into batches,
    the data is padded with the last element of the data.
    The function supports both JAX and NumPy backends.
    
    Parameters
    ----------
    data : jnp.ndarray
        The input array (for example, an array of samples).
    batch_size : int
        The desired size of each batch.
    
    Returns
    -------
    jnp.ndarray
        The padded and reshaped array where the first dimension is split into batches.
    
    Example
    -------
    >>> # Suppose data has shape (5, 2) and batch_size is 3.
    >>> data = jnp.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
    >>> # This function pads data to shape (6,2) and then reshapes it to (2, 3, 2)
    >>> create_batches_jax(data, 3)
    DeviceArray([[[ 1,  2],
                    [ 3,  4],
                    [ 5,  6]],
                    [[ 7,  8],
                    [ 9, 10],
                    [ 9, 10]]], dtype=int32)
    >>> This means that batch is appended in front of the data.
    """
    backend = get_backend(backend)
    if backend == np:
        data = np.asarray(data)
        return create_batches_np(data, batch_size)
    return create_batches_jax(data, batch_size)

##########################################################################

if _JAX_AVAILABLE:
    def eval_batched_jax(batch_size : int,
                        func        : Any,
                        params      : Any,
                        data        : jnp.ndarray):
        """ JAX version of eval_batched """
        
        # Create batches of data
        batches = create_batches(data, batch_size)
        
        def scan_fun(c, x):
            return c, jax.vmap(lambda y: func(params, y), in_axes=(0,))(x)

        # Evaluate the function on each batch using vmap
        return jax.lax.scan(scan_fun, None, jnp.array(batches))[1].reshape((-1,))

def eval_batched_np(batch_size  : int,
                    func        : Any,
                    params      : Any,
                    data        : np.ndarray):
    """
    NumPy version of eval_batched.
    
    Parameters
    ----------
    batch_size : int
        The size of each batch.
    func : Callable
        The function to evaluate on each individual sample.
    data : np.ndarray
        The input data array.
    
    Returns
    -------
    np.ndarray
        A 1D array containing the results of applying func to each sample.
    
    Example
    -------
    >>> data = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
    >>> # With batch_size=3, data is padded to shape (6,2) and reshaped to (2,3,2)
    >>> result = eval_batched_np(3, lambda x: np.sum(x), data)
    >>> # result is concatenated to a 1D array with 6 elements.
    """
    # Create batches from the data using our NumPy version of create_batches.
    batches = create_batches(data, batch_size, backend='numpy')
    # For each batch, apply func to each sample using a list comprehension.
    # Then, concatenate the results into one array.
    return np.concatenate([np.array([func(params, x) for x in batch]) for batch in batches])

def eval_batched(batch_size : int,
                func        : Callable,
                params      : Any,
                data        : np.ndarray,
                backend     : str = 'default'):
    """ Evaluate a function on batches of data using either JAX or NumPy.
    Parameters
    ----------
    batch_size : int
        The size of each batch.
    func : Callable
        The function to evaluate on each batch.
    data : np.ndarray
        The input data to be processed.
    backend : str, optional
        The backend to use for evaluation ('default' uses NumPy if available).

    Returns
    -------
    np.ndarray
        The concatenated results of the function applied to each batch.
    """
    backend = get_backend(backend)
    if backend == np:
        return eval_batched_np(batch_size, func, params, data)
    return eval_batched_jax(batch_size, func, params, data)

##########################################################################
#! GRADIENTS
##########################################################################

if _JAX_AVAILABLE:

    def flat_gradient_analytical_jax(fun: Any, params, arg) -> jnp.ndarray:
        """
        Compute a flattened complex gradient using an analytical method (JAX version).
        
        This function assumes that 'fun' provides an attribute 'analytical_gradient' that 
        returns a pytree of gradients.
        
        Parameters
        ----------
        fun : object
            The network/function object. Must provide an analytical_gradient method.
        params : Any
            The network parameters (pytree).
        arg : Any
            The input state.
        
        Returns
        -------
        jnp.ndarray
            A single flattened complex gradient.
        
        Example
        -------
        >>> # Assume fun.analytical_gradient exists.
        >>> flat_grad = flat_gradient_analytical_jax(fun, params, state)
        """
                
        # Call the analytical gradient function.
        grad_val    = fun.gradient(params, arg)
        # Flatten the gradient pytree: each leaf is reshaped to 1D.
        flat_grad   = tree_flatten(tree_map(lambda x: x.ravel(), grad_val))[0]
        # Concatenate all flattened arrays into one vector.
        return jnp.concatenate(flat_grad)

    def flat_gradient_numerical_jax(fun: Any, params, arg) -> jnp.ndarray:
        """
        Compute a flattened complex gradient using numerical differentiation (JAX version).
        
        Uses jax.grad on both the real and imaginary parts of fun.apply.
        
        Parameters
        ----------
        fun : object
            The network/function object with an apply method.
        params : Any
            The network parameters (pytree).
        arg : Any
            The input state.
        
        Returns
        -------
        jnp.ndarray
            A flattened complex gradient.
        
        Example
        -------
        >>> flat_grad = flat_gradient_numerical_jax(fun, params, state)
        """
        # Compute gradient of the real part.
        gr = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)["params"]
        gr = tree_flatten(tree_map(lambda x: x.ravel(), gr))[0]
        # Compute gradient of the imaginary part.
        gi = grad(lambda p, y: jnp.imag(fun.apply(p, y)))(params, arg)["params"]
        gi = tree_flatten(tree_map(lambda x: x.ravel(), gi))[0]
        # Concatenate and combine into a single complex vector.
        return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

    def flat_gradient_jax(fun: Any, params: Any, arg: Any, analytical: bool = False) -> jnp.ndarray:
        """
        Wrapper for computing a flattened complex gradient using JAX.
        
        If analytical=True and an analytical gradient method exists, it is used;
        otherwise numerical differentiation is applied.
        
        Parameters
        ----------
        fun : object
            The network/function object.
        params : Any
            The network parameters.
        arg : Any
            The input state.
        analytical : bool, optional
            Whether to use the analytical gradient if available (default: False).
        
        Returns
        -------
        jnp.ndarray
            The flattened complex gradient.
        
        Example
        -------
        >>> grad_vec = flat_gradient_jax(fun, params, state, analytical=True)
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        if analytical:
            return flat_gradient_analytical_jax(fun, params, arg)
        return flat_gradient_numerical_jax(fun, params, arg)

    # -----------------------------------------------------------------------------
    #! Non-holomorphic Gradients: JAX
    # -----------------------------------------------------------------------------

    def flat_gradient_cpx_nonholo_analytical_jax(fun: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute an analytical flattened complex gradient for non-holomorphic networks (JAX).

        Assumes fun returns the analytical gradient as a pytree.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        grad_val    = fun.gradient(params, arg)
        flat_grad   = tree_flatten(tree_map(lambda x: x.ravel(), grad_val))[0]
        return jnp.concatenate(flat_grad)

    def flat_gradient_cpx_nonholo_numerical_jax(fun: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute a flattened complex gradient for non-holomorphic networks using numerical differentiation (JAX).

        Adjusts the sign of the imaginary part.
        """
        gr = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)["params"]
        gr = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gr))[0]
        gi = grad(lambda p, y: jnp.imag(fun.apply(p, y)))(params, arg)["params"]
        gi = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gi))[0]
        return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

    def flat_gradient_cpx_nonholo_jax(fun: Any, params: Any, arg: Any, analytical: bool = False) -> jnp.ndarray:
        """
        Wrapper for computing a flattened complex gradient for non-holomorphic networks using JAX.
        Parameters
        ----------
        fun : object
            The network/function object.
        params : Any
            The network parameters.
        arg : Any
            The input state.
        analytical : bool, optional
            If True, use the analytical gradient if available.
        Returns
        -------
        jnp.ndarray
            The flattened complex gradient.
        Example
        -------        
        """
        if analytical:
            return flat_gradient_cpx_nonholo_analytical_jax(fun, params, arg)
        return flat_gradient_cpx_nonholo_numerical_jax(fun, params, arg)

    # -----------------------------------------------------------------------------
    #! Real Gradients: JAX
    # -----------------------------------------------------------------------------

    def flat_gradient_real_analytical_jax(fun: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute an analytical flattened real gradient (JAX).

        Assumes fun provides an 'analytical_gradient_real' method.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        grad_val    = fun.gradient(params, arg).astype(jnp.float32)
        flat_grad   = tree_flatten(tree_map(lambda x: x.ravel(), grad_val))[0]
        return jnp.concatenate(flat_grad).astype(jnp.float32)

    def flat_gradient_real_numerical_jax(fun: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute a flattened real gradient using numerical differentiation (JAX).
        """
        g = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)["params"]
        g = tree_flatten(tree_map(lambda x: x.ravel(), g))[0]
        return jnp.concatenate(g).astype(jnp.float32)

    def flat_gradient_real_jax(fun: Any, params: Any, arg: Any, analytical: bool = False) -> jnp.ndarray:
        """
        Wrapper for computing a flattened real gradient using JAX.
        Parameters
        ----------
        fun : object
            The network/function object.        
        params : Any
            The network parameters.
        arg : Any
            The input state.
        analytical : bool, optional
            If True, use the analytical gradient if available.
        Returns
        ------- 
        jnp.ndarray
            The flattened real gradient.
        Example
        -------
        >>> grad_vec = flat_gradient_real_jax(fun, params, state, analytical=True)        
        """
        if analytical:
            return flat_gradient_real_analytical_jax(fun, params, arg)
        return flat_gradient_real_numerical_jax(fun, params, arg)

    # -----------------------------------------------------------------------------
    #! Holomorphic Gradients: JAX
    # -----------------------------------------------------------------------------

    def flat_gradient_holo_analytical_jax(fun: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute an analytical flattened gradient for holomorphic networks (JAX).

        Assumes fun provides an 'analytical_gradient_holo' method.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        grad_val    = fun.gradient(params, arg)
        flat_grad   = tree_flatten(tree_map(lambda x: x.ravel(), grad_val))[0]
        return jnp.concatenate(flat_grad)

    def flat_gradient_holo_numerical_jax(fun: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute a flattened gradient for holomorphic networks using numerical differentiation (JAX).

        Each parameter's raveled value is repeated once with a multiplier of 1.j.
        """
        g = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)["params"]
        # Create a list with each flattened leaf repeated with an imaginary component.
        g = tree_flatten(tree_map(lambda x: [x.ravel(), 1.j * x.ravel()], g))[0]
        return jnp.concatenate(g)

    def flat_gradient_holo_jax(fun: Any, params: Any, arg: Any, analytical: bool = False) -> jnp.ndarray:
        """
        Wrapper for computing a flattened gradient for holomorphic networks using JAX.
        """
        if analytical:
            return flat_gradient_holo_analytical_jax(fun, params, arg)
        return flat_gradient_holo_numerical_jax(fun, params, arg)

    # -----------------------------------------------------------------------------
    #! Dictionary of Gradients: JAX
    # -----------------------------------------------------------------------------

    def dict_gradient_analytical_jax(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute an analytical dictionary of complex gradients using JAX.

        Assumes fun provides an 'analytical_dict_gradient' method.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        return fun.gradient(params, arg)

    def dict_gradient_numerical_jax(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute a dictionary of complex gradients using numerical differentiation (JAX).
        """
        gr = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)["params"]
        gr = tree_map(lambda x: x.ravel(), gr)
        gi = grad(lambda p, y: jnp.imag(fun.apply(p, y)))(params, arg)["params"]
        gi = tree_map(lambda x: x.ravel(), gi)
        return tree_map(lambda x, y: x + 1.j * y, gr, gi)

    def dict_gradient_jax(fun: Any, params: Any, arg: Any, analytical: bool = False) -> Any:
        """
        Wrapper for computing a dictionary of complex gradients using JAX.
        """
        if analytical:
            return dict_gradient_analytical_jax(fun, params, arg)
        return dict_gradient_numerical_jax(fun, params, arg)

    def dict_gradient_real_analytical_jax(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute an analytical dictionary of real gradients using JAX.

        Assumes fun provides 'analytical_dict_gradient_real'.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        return fun.gradient(params, arg)

    def dict_gradient_real_numerical_jax(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute a dictionary of real gradients using numerical differentiation (JAX).
        """
        g = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)["params"]
        return tree_map(lambda x: x.ravel(), g)

    def dict_gradient_real_jax(fun: Any, params: Any, arg: Any, analytical: bool = False) -> Any:
        """
        Wrapper for computing a dictionary of real gradients using JAX.
        """
        if analytical:
            return dict_gradient_real_analytical_jax(fun, params, arg)
        return dict_gradient_real_numerical_jax(fun, params, arg)

# ==============================================================================
#! NumPy (Autograd) Implementations: Analytical and Numerical Gradient Functions
# ==============================================================================

if AUTOGRAD_AVAILABLE:
    def flat_gradient_analytical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened complex gradient using an analytical method (NumPy version).

        Assumes fun is callable and has an 'analytical_gradient' attribute.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        grad_val    = fun.gradient(params, arg)
        flat_grad   = np.concatenate([v.ravel() for v in grad_val.values()])
        return flat_grad

    def flat_gradient_numerical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened complex gradient using numerical differentiation (NumPy/Autograd).

        Uses autograd's np_grad for both real and imaginary parts.
        """
        gr = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        gi = np_grad(lambda p, y: anp.imag(fun.apply(p, y)))(params, arg)["params"]
        return np.concatenate([gr.ravel(), 1.j * gi.ravel()])

    def flat_gradient_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> np.ndarray:
        """
        Wrapper for computing a flattened complex gradient using NumPy.
        """
        if analytical:
            return flat_gradient_analytical_np(fun, params, arg)
        else:
            return flat_gradient_numerical_np(fun, params, arg)

    def flat_gradient_cpx_nonholo_analytical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute an analytical flattened complex gradient for non-holomorphic networks (NumPy).
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        if not hasattr(fun, "analytical_gradient"):
            raise NotImplementedError("Analytical gradient not implemented for this function.")
        grad_val = fun.gradient(params, arg)
        return np.concatenate([v.ravel() for v in grad_val.values()])

    def flat_gradient_cpx_nonholo_numerical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened complex gradient for non-holomorphic networks using numerical differentiation (NumPy).
        """
        gr = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        gi = np_grad(lambda p, y: anp.imag(fun.apply(p, y)))(params, arg)["params"]
        return np.concatenate([gr.ravel(), 1.j * gi.ravel()])

    def flat_gradient_cpx_nonholo_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> np.ndarray:
        """
        Wrapper for computing a flattened complex gradient for non-holomorphic networks using NumPy.
        """
        if analytical:
            return flat_gradient_cpx_nonholo_analytical_np(fun, params, arg)
        return flat_gradient_cpx_nonholo_numerical_np(fun, params, arg)

    def flat_gradient_real_analytical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute an analytical flattened real gradient using NumPy.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        if not hasattr(fun, "analytical_gradient_real"):
            raise NotImplementedError("Analytical real gradient not implemented for this function.")
        grad_val = fun.analytical_gradient_real(params, arg)
        return np.concatenate([v.ravel() for v in grad_val.values()])

    def flat_gradient_real_numerical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened real gradient using numerical differentiation (NumPy).
        """
        g = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        return g.ravel()

    def flat_gradient_real_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> np.ndarray:
        """
        Wrapper for computing a flattened real gradient using NumPy.
        """
        if analytical:
            return flat_gradient_real_analytical_np(fun, params, arg)
        else:
            return flat_gradient_real_numerical_np(fun, params, arg)

    def flat_gradient_holo_analytical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute an analytical flattened gradient for holomorphic networks using NumPy.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        grad_val = fun.gradient(params, arg)
        return np.concatenate([v.ravel() for v in grad_val.values()])

    def flat_gradient_holo_numerical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened gradient for holomorphic networks using numerical differentiation (NumPy).
        """
        g = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        return np.concatenate([g.ravel(), 1.j * g.ravel()])

    def flat_gradient_holo_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> np.ndarray:
        """
        Wrapper for computing a flattened gradient for holomorphic networks using NumPy.
        """
        if analytical:
            return flat_gradient_holo_analytical_np(fun, params, arg)
        else:
            return flat_gradient_holo_numerical_np(fun, params, arg)

    def dict_gradient_analytical_np(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute an analytical dictionary of complex gradients using NumPy.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        if not hasattr(fun, "analytical_dict_gradient"):
            raise NotImplementedError("Analytical dict gradient not implemented for this function.")
        return fun.analytical_dict_gradient(params, arg)

    def dict_gradient_numerical_np(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute a dictionary of complex gradients using numerical differentiation (NumPy).
        """
        gr = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        gi = np_grad(lambda p, y: anp.imag(fun.apply(p, y)))(params, arg)["params"]
        return {key: gr[key].ravel() + 1.j * gi[key].ravel() for key in gr}

    def dict_gradient_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> Any:
        """
        Wrapper for computing a dictionary of complex gradients using NumPy.
        """
        if analytical:
            return dict_gradient_analytical_np(fun, params, arg)
        else:
            return dict_gradient_numerical_np(fun, params, arg)

    def dict_gradient_real_analytical_np(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute an analytical dictionary of real gradients using NumPy.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        if not hasattr(fun, "analytical_dict_gradient_real"):
            raise NotImplementedError("Analytical dict real gradient not implemented for this function.")
        return fun.analytical_dict_gradient_real(params, arg)

    def dict_gradient_real_numerical_np(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute a dictionary of real gradients using numerical differentiation (NumPy).
        """
        g = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        return g  # Preserving structure

    def dict_gradient_real_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> Any:
        """
        Wrapper for computing a dictionary of real gradients using NumPy.
        """
        if analytical:
            return dict_gradient_real_analytical_np(fun, params, arg)
        else:
            return dict_gradient_real_numerical_np(fun, params, arg)

# ==============================================================================
#! Global Wrapper Functions (for both backends)
# ==============================================================================

def flat_gradient(fun: Any, params: Any, arg: Any,
            backend: str = "jax", analytical: Optional[bool] = False) -> Any:
    """
    Compute a flattened complex gradient using either JAX or NumPy.

    Parameters
    ----------
    fun : Callable
        The function or analytical gradient function.
    params : Any
        The network parameters.
    arg : Any
        The input state.
    backend : str, optional
        Backend to use ("jax" or "numpy").
    analytical : bool, optional
        If True, use the analytical gradient if available.

    Returns
    -------
    A flattened complex gradient (jnp.ndarray or np.ndarray).
    """
    backend = get_backend(backend)
    if backend == np:
        return flat_gradient_np(fun, params, arg, analytical)
    return flat_gradient_jax(fun, params, arg, analytical)

def flat_gradient_cpx_nonholo(fun: Any, params: Any, arg: Any,
                            backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a flattened complex gradient for non-holomorphic networks using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np:
        return flat_gradient_cpx_nonholo_np(fun, params, arg, analytical)
    return flat_gradient_cpx_nonholo_jax(fun, params, arg, analytical)

def flat_gradient_real(fun: Any, params: Any, arg: Any,
                    backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a flattened real gradient using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np:
        return flat_gradient_real_np(fun, params, arg, analytical)
    return flat_gradient_real_jax(fun, params, arg, analytical)

def flat_gradient_holo(fun: Any, params: Any, arg: Any,
                    backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a flattened gradient for holomorphic networks using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np:
        return flat_gradient_holo_np(fun, params, arg, analytical)
    return flat_gradient_holo_jax(fun, params, arg, analytical)

def dict_gradient(fun: Any, params: Any, arg: Any,
                backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a dictionary of complex gradients using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np:
        return dict_gradient_np(fun, params, arg, analytical)
    return dict_gradient_jax(fun, params, arg, analytical)

def dict_gradient_real(fun: Any, params: Any, arg: Any,
                    backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a dictionary of real gradients using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np:
        return dict_gradient_real_np(fun, params, arg, analytical)
    return dict_gradient_real_jax(fun, params, arg, analytical)

# ==============================================================================

def decide_grads(iscpx, isjax, isanalitic, isholomorphic):
    """
    Decide which gradient function to use based on the input flags.

    Parameters
    ----------
    iscpx : bool
        Flag indicating if the function is complex.
    isjax : bool
        Flag indicating if JAX should be used.
    isanalitic : bool
        Flag indicating if the analytical gradient should be used.
    isholomorphic : bool
        Flag indicating if the function is holomorphic.

    Returns
    -------
    Callable
        The appropriate gradient function.
    """
    if not isjax:  # NumPy backend
        if iscpx:  # Complex functions
            if isholomorphic:  # Holomorphic
                if isanalitic:
                    return flat_gradient_holo_analytical_np
                else:
                    return flat_gradient_holo_numerical_np
            else:  # Non-holomorphic
                if isanalitic:
                    return flat_gradient_cpx_nonholo_analytical_np
                else:
                    return flat_gradient_cpx_nonholo_numerical_np
        else:  # Real functions
            if isanalitic:
                return flat_gradient_real_analytical_np
            else:
                return flat_gradient_real_numerical_np
    else:  # JAX backend
        if iscpx:  # Complex functions
            if isholomorphic:  # Holomorphic
                if isanalitic:
                    return flat_gradient_holo_analytical_jax
                else:
                    return flat_gradient_holo_numerical_jax
            else:  # Non-holomorphic
                if isanalitic:
                    return flat_gradient_cpx_nonholo_analytical_jax
                else:
                    return flat_gradient_cpx_nonholo_numerical_jax
        else:  # Real functions
            if isanalitic:
                return flat_gradient_real_analytical_jax
            else:
                return flat_gradient_real_numerical_jax
            
# ==============================================================================