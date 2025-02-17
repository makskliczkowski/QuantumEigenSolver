"""
file    : Model/Interacting/Spin/qsm.py
High-level Hamiltonian class for the Quantum Energy Solver (QES) package.
This file contains the QSM model - a Hamiltonian for a quantum dot coupled to an external system.
The external system is modeled as a quantum spin chain.
Based on the C++ implementation.
Author  : Maksymilian Kliczkowski
Date    : 2025-02-17
Version : 0.1
"""

import numpy as np
import math
from typing import List, Tuple, Union

# Assume these are available from the QES package:
from Algebra.hilbert import HilbertSpace
from Algebra.hamil import Hamiltonian 
