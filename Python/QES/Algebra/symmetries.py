'''
Description: This module contains functions to compute the symmetries of a given state at a given representation.
'''

from . import Operator
from ..general_python import Lattice
from typing import Callable, Union, Tuple, List

# Import the necessary modules
import jax.numpy as jnp
import numpy as np
from functools import partial

####################################################################################################

def translation(lattice : Lattice, state_type) -> Callable:
    '''
    Generate the translation operator for a given lattice
    '''
    # generate the cyclic shift
    cyclicShift = 