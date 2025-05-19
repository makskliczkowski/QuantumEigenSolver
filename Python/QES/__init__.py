"""
Quantum EigenSolver Package
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
Date        : 2025-02-01
file        : QES/__init__.py
"""
import os
import sys
# add the QES directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

####################################################################################################

import general_python
import Solver

# -------------------------------------------------------------------------------------------------