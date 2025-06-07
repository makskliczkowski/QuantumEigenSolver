"""
Interacting Models Module
=========================

This module contains quantum many-body models with particle interactions.

Submodules:
-----------
- Spin: Spin models (Heisenberg, Ising, XY, etc.)
- Fermion: Fermionic models with interactions (planned)

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from . import Spin as spin

__all__ = ['spin']
