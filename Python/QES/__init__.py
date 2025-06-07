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

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

####################################################################################################

import general_python

# Conditionally import Solver - skip during documentation build
import os
if os.environ.get('SPHINX_BUILD') != '1':
    try:
        import Solver
    except ImportError:
        pass  # Skip during documentation build

# -------------------------------------------------------------------------------------------------
#! End of QES package initialization
# -------------------------------------------------------------------------------------------------