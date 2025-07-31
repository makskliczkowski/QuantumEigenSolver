import os, sys
import math
import numpy as np
from typing import Optional, List, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
for i in range(1, 5): # Check up to 4 levels up
    dir_up = os.path.abspath(os.path.join(script_dir, *(['..'] * i)))
    if dir_up not in sys.path:
        sys.path.append(dir_up)
    sys.path.append(os.path.join(dir_up, "QES", "general_python"))

# ------------------------------------------------------------------------------------

from ent_loss_jax import *

# ------------------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    print("Please install the required packages: jax, jaxlib, optax, flax")
    sys.exit(1)
    
# ------------------------------------------------------------------------------------
#! Step function for simple optimization
# ------------------------------------------------------------------------------------

class MultiStateOptimizerSimple:
    
    def __init__(self, 
                prepared_states_list    : List[jnp.ndarray],
                dimA                    : int,
                dimB                    : Optional[int] = None,
                gamma                   : int           = 0, 
                is_complex              : bool          = True,
                seed                    : int           = 42,
                logger                  : Optional[any] = None
                ):
        """
        Initialize the optimizer with multiple prepared state matrices.
        
        Parameters:
        - prepared_states_list  : List of JAX arrays, each containing prepared states
        - dimA                  : Dimension of the first state space
        - dimB                  : Dimension of the second state space (if applicable)
        - gamma                 : Regularization strength
        - is_complex            : Whether to use complex coefficients
        - seed                  : Random seed for reproducibility
        """
        self.gamma                  = gamma
        self.is_complex             = is_complex
        dtype                       = jnp.complex64 if is_complex else jnp.float32
        self.prepared_states        = jnp.stack(prepared_states_list).astype(dtype)
        self.n_matrices             = self.prepared_states.shape[0]
        self.seed                   = seed
        
        if self.n_matrices == 0:
            raise ValueError("prepared_states_list must contain at least one prepared state matrix")
        
        # Validate that all matrices have the same number of rows (first dimension)
        row_dims = [states.shape[0] for states in self.prepared_states]
        if not all(dim == row_dims[0] for dim in row_dims):
            raise ValueError("All prepared state matrices must have the same number of rows")
        
        self.l                      = np.log2(row_dims[0])
        self.la                     = np.log2(dimA)
        self.lb                     = np.log2(dimB) if dimB is not None else self.l - self.la
        self.dimA                   = dimA
        self.dimB                   = dimB if dimB is not None else 2**(self.lb)
        
        # Validate that all matrices have the same number of columns (second dimension)
        col_dims = [states.shape[1] for states in self.prepared_states]
        if not all(dim == col_dims[0] for dim in col_dims):
            raise ValueError("All prepared state matrices must have the same number of columns")
        
        if gamma != col_dims[0]:
            raise ValueError("Gamma must be equal to the number of columns in prepared states")
        
        self.coeffs         = None
        self.loss_history   = []
        self.logger         = logger

    # --------------------------------------------------------------------------------
    #! Private methods
    # --------------------------------------------------------------------------------    
    
    def _initialize_coefficients(self) -> jnp.ndarray:
        """Initialize random coefficients."""
        if self.is_complex:
            dtype_in        = jnp.complex64
            key_re, key_im  = jax.random.split(jax.random.PRNGKey(self.seed))
            coeffs_re       = jax.random.normal(key_re, (self.gamma,), dtype=jnp.float32)
            coeffs_im       = jax.random.normal(key_im, (self.gamma,), dtype=jnp.float32)
            coeffs          = coeffs_re + 1j * coeffs_im
        else:
            key             = jax.random.PRNGKey(self.seed)
            coeffs          = jax.random.normal(key, (self.gamma,), dtype=jnp.float32)
        
        return normalize_state(coeffs)

    # ------------------------------------------------------------------------------------
    #! Loss functions
    # ------------------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def loss_multi(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the loss for multiple prepared states.
        
        Parameters:
            coeffs : Coefficients for the prepared states.
            
        Returns:
            jnp.ndarray : Loss value.
        """
        losses = [loss_entanglement_single(coeffs, states, self.dimA, self.dimB) for states in self.prepared_states]
        return jnp.mean(jnp.array(losses))

    # ------------------------------------------------------------------------------------
    #! Steps
    # ------------------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,2))
    def _step_real(self, coeffs: jnp.ndarray, lr: float) -> Tuple[jnp.ndarray, float]:
        """Single optimization step for real coefficients."""
        loss_fn         = lambda c: self.loss_multi(c)
        loss            = loss_fn(coeffs)
        grad            = jax.grad(loss_fn)(coeffs)
        coeffs_new      = coeffs - lr * grad
        return normalize_state(coeffs_new), loss
    
    @partial(jax.jit, static_argnums=(0,2))
    def _step_complex(self, coeffs: jnp.ndarray, lr: float) -> Tuple[jnp.ndarray, float]:
        """Single optimization step for complex coefficients."""
        def loss_wrapped(real_imag):
            c = real_imag[:real_imag.size//2] + 1j * real_imag[real_imag.size//2:]
            return self.loss_multi(c)

        # Reconstruct complex coefficients
        real_imag       = jnp.concatenate([jnp.real(coeffs), jnp.imag(coeffs)])
        grad            = jax.grad(loss_wrapped)(real_imag)

        # Gradient descent in real+imag space
        updated         = real_imag - lr * grad
        c_new           = updated[:coeffs.size] + 1j * updated[coeffs.size:]

        return normalize_state(c_new), self.loss_multi(c_new)

    # ------------------------------------------------------------------------------------
    
    def optimize(self, 
                lr              : float = 0.01, 
                n_steps         : int   = 500, 
                verbose_every   : int   = 50) -> Tuple[jnp.ndarray, List[float]]:
        """
        Optimize coefficients across all prepared state matrices.
        
        Parameters:
            lr              : Learning rate for the optimization
            n_steps         : Number of optimization steps
            verbose_every   : Print loss every N steps

        Returns:
            Final coefficients and loss history
        """
        
        # Initialize coefficients if not already done
        if self.coeffs is None:
            self.coeffs = self._initialize_coefficients()
        
        self.loss_history = []
        if self.logger is not None:
            self.logger.info(f"Starting optimization with {self.n_matrices} state matrices", color = "blue")
            self.logger.info(f"Coefficient dimension: {self.gamma}", lvl = 1, color = "blue")
            self.logger.info(f"Complex coefficients?: {self.is_complex}", lvl = 1, color = "blue")
        else:
            print(f"Starting optimization with {self.n_matrices} state matrices")
            print(f"\tCoefficient dimension: {self.gamma}")
            print(f"\tComplex coefficients?: {self.is_complex}")
        
        for step in range(n_steps):
            if self.is_complex:
                self.coeffs, loss = self._step_complex(self.coeffs, lr)
            else:
                self.coeffs, loss = self._step_real(self.coeffs, lr)
            
            self.loss_history.append(float(loss))
            
            if step % verbose_every == 0 and step > 0:
                if self.logger is not None:
                    self.logger.info(f"Step {step}, Loss: {loss:.6f}", color = "green", lvl = 1)
                else:
                    print(f"\tStep {step}, Loss: {loss:.6f}")

        print(f"Optimization completed. Final loss: {self.loss_history[-1]:.6f}")
        return self.coeffs, self.loss_history

    # ------------------------------------------------------------------------------------

    def get_final_states(self) -> List[jnp.ndarray]:
        """
        Get the final quantum states for each prepared state matrix.
        
        Returns:
        - List of final quantum states, one for each input matrix
        """
        if self.coeffs is None:
            raise ValueError("No coefficients available. Run optimize() first.")
        
        final_states = []
        for prepared_states in self.prepared_states:
            final_state = prepared_states @ self.coeffs
            final_states.append(final_state)
        
        entropies    = []
        for state in final_states:
            schmidt_values = schmidt_jax(state, self.dimA, self.dimB, use_eig=False)[0]
            entropies.append(vn_entropy_jax(schmidt_values))

        return final_states, entropies

    # ------------------------------------------------------------------------------------
    
    def reset(self, new_seed: Optional[int] = None):
        """Reset the optimizer with new random coefficients."""
        if new_seed is not None:
            self.seed = new_seed
        self.coeffs         = None
        self.loss_history   = []

# ----------------------------------------------------------------------------------------

if __name__ == "__main__":    
    # load the states from the HDF5 file
    from QES.general_python.common import flog
    from ent_read_states import load_quantum_states, parse_arguments
    
    args        = parse_arguments()
    logger      = flog.get_global_logger()
    
    (org_states, org_entropies), (mix_states_real, mix_entropies_real), system_params = load_quantum_states(
        L               = args.L,
        gamma           = args.gamma,
        r               = args.r,
        n_states        = args.n_states,
        data_dir        = args.data_dir,
        mixture_index   = args.mixture_index,
        logger          = logger
    )

    # create the optimizer
    optimizer = MultiStateOptimizerSimple(
        prepared_states_list   = mix_states_real,
        dimA                   = system_params['dimA'],
        dimB                   = system_params['dimB'],
        gamma                  = system_params['gamma'],
        seed                   = system_params['r'],
        is_complex             = args.is_complex,
    )

    final_coeffs, loss_history = optimizer.optimize(
        lr              = args.lr,
        n_steps         = args.n_steps,
        verbose_every   = args.verbose_every
    )
    
    logger.info(f"Final coefficients: {final_coeffs}", color="green", lvl = 1)
    # final state entropies
    final_states, final_entropies = optimizer.get_final_states()
    for i, entropy in enumerate(final_entropies):
        logger.info(f"Final entropy for state {i}: {entropy}", color="green", lvl = 1)
