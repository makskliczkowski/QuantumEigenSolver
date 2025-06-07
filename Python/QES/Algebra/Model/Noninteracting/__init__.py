"""
Non-Interacting Models Module
=============================

This module contains free particle and non-interacting quantum models.

Submodules:
-----------
- Conserving: Models with particle number conservation
- Nonconserving: Models without particle number conservation (planned)

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from . import Conserving as c

__all__ = ['c']