"""
file: Algebra/__init__.py
This package contains modules for algebraic operations in the Quantum EigenSolver project.

Modules:
    - operator: Defines classes and functions for handling general operators in quantum mechanics.
    - hilbert : High-level Hilbert space class for quantum many-body systems.
"""

# ----------------------------------------------------------------------------------------------
#! Hilbert space and operator classes
# ----------------------------------------------------------------------------------------------

from Algebra.hilbert import HilbertSpace
from Algebra.Operator.operator import Operator, SymmetryGenerators
from Algebra.Operator.operators_spin import sig_x, sig_y, sig_z, sig_p, sig_m, sig_k

# ----------------------------------------------------------------------------------------------
#! Hamiltonian models
# ----------------------------------------------------------------------------------------------
from Algebra.hamil import Hamiltonian
from Algebra.hamil_types import Hamiltonians, check_noninteracting, check_dense

# ----------------------------------------------------------------------------------------------

__all__ = ["Operator", "HilbertSpace",
        "Hamiltonian", "Hamiltonians", "check_noninteracting", "check_dense"]
# ----------------------------------------------------------------------------------------------