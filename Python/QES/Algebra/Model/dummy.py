'''
file:   : Model/dummy.py
author: : Maksymilian Kliczkowski
email:  : maksymilian.kliczkowski@pwr.edu.pl
This file defines a dummy Hamiltonian class for testing purposes. 

Description:
        This module implements a dummy Hamiltonian class for testing and development purposes.
        It provides a simplified implementation of quantum Hamiltonian mechanics with integer
        and array representations of quantum states. The class includes support for different
        computational backends including Numba and JAX (when available).
        
        The DummyHamiltonian class inherits from the Hamiltonian base class and implements
        a simple parametric Hamiltonian with diagonal and off-diagonal elements for 
        benchmarking and testing matrix operations within the QES framework.
'''

import numpy as np
import numba

# typing
from typing import Union, Optional

# Assume these are available from the QES package:
from Algebra.hilbert import HilbertSpace
from Algebra.hamil import Hamiltonian
import Algebra.Operator.operators_spin as operators_spin_module

##########################################################################################
import general_python.algebra.linalg as linalg
from general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, _JAX_AVAILABLE
from general_python.common import binary as _binary
##########################################################################################

# ----------------------------------------------------------------------------------------

class DummyHamiltonian(Hamiltonian):
    '''
    Dummy Hamiltonian class for testing
    '''
    
    def __init__(self, 
                hilbert_space   : HilbertSpace, 
                param           : Union[float, complex] = 1.0, 
                ns              : Optional[int]         = None, 
                backend         : str                   = 'default', 
                dtype                                   = complex):
        '''
        Dummy Hamiltonian class for testing
        Parameters
        ----------
        hilbert_space   : HilbertSpace
            Hilbert space object
        param           : float or complex
            Parameter for the Hamiltonian
        '''
        
        # Initialize the Hamiltonian
        if hilbert_space is None:
            if ns is None:
                raise ValueError(self._ERR_EITHER_HIL_OR_NS)
            hilbert_space = HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)
        
        super().__init__(hilbert_space=hilbert_space, backend=backend, dtype=dtype)
        
        self._param = param
        if (self.dtype == np.float64 or self.dtype == np.float32 or self.dtype == float) and isinstance(param, complex):
            self._param = np.complex128(param)
        
        # functions for local energy
        self._set_local_energy_operators()
        self._set_local_energy_functions()
        
    # ------------------------------------------------------------------------------------
    
    def __repr__(self):
        return f'Dummy Hamiltonian with {self.ns} sites and parameter {self._param}'
    
    def __str__(self):
        return f'dummy,{self.ns},p={self._param}'
    
    # ------------------------------------------------------------------------------------
    
    @property
    def param(self):
        ''' Parameter for the Hamiltonian '''
        return self._param
    
    @param.setter
    def param(self, value):
        self._param = value
        self.reset_operators()
        self._set_local_energy_operators()
        self._set_local_energy_functions()
        
    # ------------------------------------------------------------------------------------
    
    def _set_local_energy_operators(self):
        ''' Set local energy operators for the Hamiltonian '''
        
        if self._hilbert_space.Nhl == 2:
            self._log('Using spin operators', log = 'info', lvl = 1, color = 'green')
            
            for i in range(self.ns):
                
                # site indices
                sites = [i, (i + 1) % self.ns]
                
                # create global sigma_x that acts on neighboring sites
                sigma_x = operators_spin_module.sig_x(ns = self.ns, sites = sites)

                # create global sigma_z that acts on the site i
                sigma_z = operators_spin_module.sig_z(ns = self.ns, sites = [i])

                # add the operators with the value of the parameter times the site index
                self.add(
                    operator    = sigma_x,
                    multiplier  = self._param * i,
                    modifies    = True,
                    sites       = sites
                )
                
                self.add(
                    operator    = sigma_z,
                    multiplier  = self._param / 2.0 * i % 2,
                    modifies    = False,
                    sites       = [i]
                )
        else:
            pass
        
        
    # ------------------------------------------------------------------------------------
    
# ----------------------------------------------------------------------------------------