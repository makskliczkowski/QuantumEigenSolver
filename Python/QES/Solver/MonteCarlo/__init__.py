"""
Monte Carlo Solver Module
=========================

This module provides Monte Carlo methods for quantum many-body systems.

Modules:
--------
- montecarlo: Core Monte Carlo algorithms
- parallel: Parallel Monte Carlo implementations
- sampler: Sampling algorithms for quantum states

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

try:
    from .montecarlo import *
    from .parallel import *
    from .sampler import *
    
    __all__ = []
except ImportError:
    __all__ = []