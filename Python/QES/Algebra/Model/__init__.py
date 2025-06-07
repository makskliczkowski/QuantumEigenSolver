"""
QES Model Module
================

This module provides implementations of various quantum many-body models.

Submodules:
-----------
- Interacting: Models with particle interactions
- Noninteracting: Free particle and non-interacting models

Classes:
--------
Various quantum model implementations including:
- Spin models (Heisenberg, Ising, XY, etc.)
- Fermionic models (Hubbard, t-J, etc.)
- Bosonic models

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from . import Interacting as intr
from . import Noninteracting as nintr

__all__ = ['intr', 'nintr']