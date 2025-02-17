"""
file        : Algebra/Operator/operators_spin.py

This module implements spin operators for quantum systems.
It includes functions for sigma_x, sigma_y, sigma_z, sigma_plus (raising),
sigma_minus (lowering), their products, and a Fourier-transformed sigma_k operator.
The implementation is based on the provided C++ code and uses a general Operator class.

Author      : Maksymilian Kliczkowski, WUST, Poland
Date        : February 2025
Version     : 1.0

Future Work:
- Implement sigma_x, sigma_y, sigma_z operator functions.
- Define functionality for raising and lowering operator functions.
- Implement products of spin operators.
- Add tests for each functionality implemented.
"""

import math
import numpy as np

from Algebra.Operator.operator import Operator

################################################################################
from general_python.algebra.utils import DEFAULT_BACKEND, get_backend as __backend
from general_python.common.binary import 
################################################################################


################################################################################
#! Standard Pauli matrices
################################################################################

# Define the Pauli matrices for reference
_SIG_X = np.array([[0, 1],
                [1, 0]], dtype=float)   
_SIG_Y = np.array([[0, -1j],
                [1j, 0]], dtype=float)
_SIG_Z = np.array([[1,  0],
                [0, -1]], dtype=float)

################################################################################