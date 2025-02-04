"""

Quantum EigenSolver Package
@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""

'''
Import the lattices and allow for the user to choose the lattice type.
'''
from .general_python.lattices import Lattice, SquareLattice, HexagonalLattice, choose_lattice


'''
Import elements related to the algebra of the system like:
- HilbertSpace : The Hilbert space of the system. It is defined within the algebra module.
- Hamiltonian  : The Hamiltonian of the system. It is defined within the algebra module.
'''
from .Algebra.hilbert import HilbertSpace


# Import all modules

# Export the modules
__all__ = [
    "Lattice",
    "SquareLattice",
    "HexagonalLattice",
    "choose_lattice",
    "HilbertSpace"
]