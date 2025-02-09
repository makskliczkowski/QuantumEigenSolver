'''
High-level methods for JIT compilation of Hamiltonian operations.

This module contains optimized functions that utilize JAX's JIT compilation features
to enhance the performance of Hamiltonian computations within the QES package.

Functions included in this module aim to improve the efficiency and speed of
Hamiltonian evaluations, especially for complex many-body systems, by leveraging
the capabilities of JAX for automatic differentiation and just-in-time compilation.

This module also aims to provide a seamless interface for transitioning between
standard NumPy operations and JAX-optimized operations without compromising code clarity.

In addition, helper functions for managing the Hamiltonian matrices and integrating
with Hilbert space representations will be included to streamline computations.

Future development may include the implementation of utility functions to facilitate
matrix operations, eigenvalue solvers, and advanced Hamiltonian manipulation routines.
'''

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, random
    from jax.experimental.sparse import BCOO
    
    # -----------------------------------------------------------------
    
    @jit
    def mean_level_spacing(eigenvalues):
        '''
        Compute the mean level spacing of a set of energy eigenvalues.

        Parameters
        ----------
        eigenvalues : array-like
            A 1D array of energy eigenvalues.

        Returns
        -------
        float
            The mean level spacing of the energy eigenvalues.
        '''
        return jnp.mean(jnp.diff(jnp.sort(eigenvalues)))
    
    # -----------------------------------------------------------------
    
    @jit
    def energy_width(hamil):
        '''
        Compute the energy width of a Hamiltonian matrix.

        Parameters
        ----------
        hamil : array-like
            A 2D array representing the Hamiltonian matrix.

        Returns
        -------
        float
            The energy width of the Hamiltonian matrix.
        '''
        return jnp.dot(hamil, hamil).trace()
    
    # -----------------------------------------------------------------
    
except ImportError:
    pass
    