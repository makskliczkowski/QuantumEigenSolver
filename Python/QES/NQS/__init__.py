"""
QES Neural Quantum States (NQS) Module
======================================

This module provides implementation of Neural Quantum States for 
variational quantum many-body calculations.

Modules:
--------
- nqs: Core Neural Quantum State implementations
- nqs_train: Training algorithms for NQS
- tdvp: Time-Dependent Variational Principle methods
- REF: Reference implementations and examples

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

# Import main classes when available
try:
    from .nqs import *
    from .nqs_train import *
    from .tdvp import *
except ImportError:
    # Handle cases where dependencies might not be available
    pass

__all__ = []