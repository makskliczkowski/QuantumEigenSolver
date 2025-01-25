import jax.numpy as jnp

class MonteCarloSolver:
    '''
    
    
    '''
    
    # define the static variables
    _epsilon = jnp.finfo(jnp.float64).eps      # machine epsilon for float64
    
    def __init__(self, *args, **kwargs):
        '''
        '''
        
        # define the instance variables
        self._accepted          = 0            # number of accepted steps
        self._total             = 0            # total number of steps          
        self._acceptance_rate   = 0.0          # acceptance rate
        
        # define the loss related variables
        self._current_loss      = 0.0          # current loss value
        self._last_loss         = 0.0          # last loss value
        self._best_loss         = 1e10         # best loss value
        
        # temperature related variables
        self._beta              = 1.0          # inverse temperature beta = 1/T
        
        # define the random number generator
        self._rng_key           = None         # random number generator key
        
        # information 
        self._info              = "Monte Carlo Solver"
        
        