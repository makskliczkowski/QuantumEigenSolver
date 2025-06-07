"""
QES Algebra Module
==================

This package contains modules for algebraic operations in the Quantum EigenSolver project.

Modules:
--------
- hilbert: High-level Hilbert space class for quantum many-body systems
- hamil: Hamiltonian construction and manipulation
- symmetries: Symmetry operations and group theory
- Operator: General operators in quantum mechanics
- Model: Predefined quantum models (interacting and non-interacting)
- Hilbert: Hilbert space utilities
- Properties: Physical properties calculations

Classes:
--------
- HilbertSpace: Main class for quantum many-body Hilbert spaces
- Hamiltonian: Hamiltonian matrix construction and operations
- Operator: General quantum mechanical operators

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

# Import main classes
try:
    from .hilbert import HilbertSpace
    from .hamil import Hamiltonian
    from .symmetries import *
    
    __all__ = ['HilbertSpace', 'Hamiltonian']
except ImportError:
    # Handle missing dependencies gracefully
    __all__ = []