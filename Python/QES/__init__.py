"""
Quantum EigenSolver Package
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
Date        : 2025-02-01
file        : QES/__init__.py
"""

__version__         = "0.1.0"
__author__          = "Maksymilian Kliczkowski"
__email__           = "maksymilian.kliczkowski@pwr.edu.pl"
__license__         = "CC-BY-4.0"
__description__     = "Quantum Eigen Solver: Comprehensive framework for quantum eigenvalue problem solving"

__all__ = [
    "PRNGKey",
    "backend_mgr",
    "qes_reseed",
    "qes_next_key",
    "qes_split_keys",
    "qes_split2",
    "qes_seed_scope",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
]

from contextlib import contextmanager

# IMPORTANT:
# - No sys.path hacks here. The package should be installed/editable (`pip install -e .`).
# - Use absolute, package-local imports.
from general_python.algebra.utils import PRNGKey, backend_mgr, get_global_logger, get_global_backend

####################################################################################################

def qes_reseed(seed: int):
    """
    Reseed the global random number generators.
    """
    backend_mgr.reseed(seed)

def qes_next_key():
    """
    Get a fresh JAX subkey and advance the global key.
    """
    return backend_mgr.next_key()

def qes_split_keys(n: int):
    """
    Split the current JAX PRNG key into two subkeys.
    """
    return backend_mgr.split_keys(n)

@contextmanager
def qes_seed_scope(seed: int, *, touch_numpy_global: bool = False, touch_python_random: bool = False):
    """
    Context manager to temporarily set the random seed for various backends.
    """
    with backend_mgr.seed_scope(seed, touch_numpy_global=touch_numpy_global, touch_python_random=touch_python_random) as suite:
        yield suite

# -------------------------------------------------------------------------------------------------
#! End of QES package initialization
# -------------------------------------------------------------------------------------------------