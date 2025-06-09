"""
QES Operator Module
===================

This module provides quantum mechanical operators for various particle types.

Modules:
--------
- operator: Base operator classes and utilities
- operator_matrix: Matrix representations of operators
- operators_spin: Spin operators (Pauli matrices, etc.)
- operators_spin_jax: JAX implementations of spin operators
- operators_spinless_fermions: Fermionic operators for spinless particles
- operators_spinless_fermions_jax: JAX implementations of fermionic operators

Classes:
--------
- Operator: Base class for quantum operators
- SymmetryGenerators: Symmetry operation generators

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

try:
    from .operator import Operator, SymmetryGenerators
    from .operator_matrix import *
    from .operators_spin import *
    from .operators_spin import *
    
    __all__ = ['Operator', 'SymmetryGenerators']
except ImportError:
    __all__ = []