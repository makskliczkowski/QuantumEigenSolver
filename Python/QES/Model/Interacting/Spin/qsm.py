"""
file    : Model/Interacting/Spin/qsm.py
High-level Hamiltonian class for the Quantum Energy Solver (QES) package.
This file contains the QSM model - a Hamiltonian for a quantum dot coupled to an external system.
The external system is modeled as a quantum spin chain.
Based on the C++ implementation.
Author  : Maksymilian Kliczkowski
Date    : 2025-02-17
Version : 0.1
"""

import numpy as np
import math
from typing import List, Tuple, Union

# Assume these are available from the QES package:
from Algebra.hilbert import HilbertSpace
from Algebra.hamil import Hamiltonian 
from Algebra.Operator.operators_spin import sig_x, sig_y, sig_z

##########################################################################################
from general_python.algebra.linalg import kron, identity
##########################################################################################

__QSM_CHECK_HS_NORM = True

class QSM(Hamiltonian):
    """
    Quantum Sun Model (QSM) Hamiltonian. It describes a quantum dot coupled to an external system.
    The external system is modeled as a quantum spin chain.
    """
    
    def __init__(self,
                ns      : int,
                nin     : int,
                gamma   : float,
                g0      : float,
                a       : list,
                h       : list,
                xi      : list,
                hilbert_space   = None,
                backend         = 'default'):
        """
        Construct a QSM model.
        
        Parameters
        ----------
        ns : int
            Total number of spins.
        nin : int
            Number of spins in the "dot" (internal subsystem).
        gamma : float
            Normalization parameter for the dot Hamiltonian.
        g0 : float
            Coupling strength between the dot and the outside system.
        a : list of float
            Coupling constants between the dot and outside spins.
        h : list of float
            Magnetic field acting on the outside spins.
        xi : list of float
            Random distribution parameters for the coupling.
        hilbert_space : HilbertSpace, optional
            The Hilbert space object (if not provided, one is constructed internally).
        backend : str, optional
            The backend for computations.
        """
        
        # Initialize the base Hamiltonian (assume Hamiltonian's __init__ handles hilbert_space)
        if hilbert_space is None:
            hilbert_space = HilbertSpace(ns = ns)
        super().__init__(hilbert_space=hilbert_space, backend=backend)
        self._ns        = ns
        self._nin       = nin       # number of spins in the dot (internal subsystem)
        self._nout      = ns - nin  # number of spins in the outside system
        self._gamma     = gamma
        self._g0        = g0
        if isinstance(a, float):
            a = [a] * self._nout
        elif len(a) != self._nout:
            self.__log(f"Length of a ({len(a)}) does not match the number of outside spins ({self._nout}).", log = 1, lvl = 1, color = 'red')
            a = [a[0]] * self._nout
        self._a         = a
        
        if isinstance(h, float):
            h = [h] * self._nout
        elif len(h) != self._nout:
            self.__log(f"Length of h ({len(h)}) does not match the number of outside spins ({self._nout}).", log = 1, lvl = 1, color = 'red')
            h = [h[0]] * self._nout

        self._h         = h
        self._xi        = xi

        self.dimIn      = 2 ** self._nin
        self.dimOut     = 2 ** self._nout
        # Generate random dot Hamiltonian (GOE) and normalize.
        self.Hdot       = random_GOE(self.dimIn)
        self.Hdot       = self.gamma / math.sqrt(self.dimIn + 1) * self.Hdot

        self.check_sizes()
        self.initialize_particles()
        # Assume self.Nh is defined by the Hilbert space.
        self.Nh = self.hilbert_space.Nh if self.hilbert_space is not None else 2 ** Ns
