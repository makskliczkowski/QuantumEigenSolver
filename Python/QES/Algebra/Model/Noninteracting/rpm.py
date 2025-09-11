"""
Rosenzweig - Porter model (RPM) Hamiltonian and related utilities.

File    : Model/Noninteracting/rpm.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
"""

import numpy as np
from typing import List, Optional, Union

#! QES package imports
import Algebra.hilbert as hilbert_module
import Algebra.hamil as hamil_module

#! Random matrix wrapper and linear algebra utilities
from general_python.algebra.ran_wrapper import set_global_seed, GOE, GUE

# -------------------------------------------------------------------

class RosenzweigPorter(hamil_module.Hamiltonian):
    """
    Rosenzweig-Porter model Hamiltonian.
    """
    
    def __init__(self,
                ns              : int,
                gamma           : float,
                dtype           : type                      = np.float64,
                backend         : str                       = "default",
                many_body       : bool                      = True,
                seed            : Optional[int]             = None,
                **kwargs):

        # initialize Hilbert space
        self._many_body     = many_body
        self._ns            = ns if many_body else np.log2(ns)
        self._nh            = 2**self._ns if many_body else ns

        _hilbert_space      = hilbert_module.HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)
        super().__init__(is_manybody = True, hilbert_space=_hilbert_space, is_sparse=False, seed=seed, dtype=dtype, backend=backend, **kwargs)
        self._is_sparse     = False
        
        #! couplings
        self._gamma         = gamma
        self._gamma_power   = self._nh ** self._gamma
        self._gamma_power_i = self._nh ** (-0.5 * self._gamma)
        
        # storage for random blocks
        self._hamil         = None
        self._diagonal      = None
        self._std_en        = None
        self._seed          = kwargs.get('seed', None)
        set_global_seed(self._seed, backend=self._backend)

        # set the Hamiltonian operators
        self._max_local_ch_o = 0
        self._set_local_energy_operators()
        self._set_local_energy_functions()

    # ---------------------------------------------------------------

    @property
    def ns(self) -> int:
        return self._ns
    
    @property
    def gamma(self) -> float:
        return self._gamma
        return self._many_body

    def randomize(self, **kwargs):
        '''
        Randomize the Hamiltonian.
        This method is used to generate a new random Hamiltonian
        with the same parameters as the original one but with different
        random blocks.
        '''
        if kwargs.get('seed', None) is not None:
            set_global_seed(kwargs['seed'], backend=self._backend)
        
        # initialize the blocks
        # self._diagonal  = self._rng.normal(loc=0.0, scale=1.0, size=self._nh).astype(self._dtype)
        self._diagonal = self._backend.asarray(np.random.normal(loc=0.0, scale=1.0, size=self._nh), dtype=self._dtype)
    
    # ---------------------------------------------------------------
    
    @staticmethod
    def repr(**kwargs) -> str:
        ns  = kwargs.get('ns', '?')
        g   = kwargs.get('gamma','1.0')
        return f"RPM(ns={ns},g={g:.3f})"

    def __repr__(self):
        return self.repr(ns = self.ns, gamma = self.gamma)

    def __str__(self):
        return self.__repr__()

    # ---------------------------------------------------------------

    def _hamiltonian(self, use_numpy: bool = False):
        """
        Build the full ultrametric Hamiltonian by combining the central dot Hamiltonian H0
        with the hierarchical blocks Hk.
        
        First, the Hamiltonian is initialized. There shall be no operators
        acting on states apart from the random blocks.
        The Hamiltonian is constructed as a Kronecker product of the central dot Hamiltonian H0
        and the identity matrix of size 2^L.
        """
        
        if self._nh == 0:
            raise ValueError("UltrametricModel: Hamiltonian not initialized.")
        
        # initialize zero Hamiltonian container
        try:
            backend_changed = self._backend if not use_numpy else np
            
            #! off-diagonal part
            self._hamil     = backend_changed.zeros((self._nh, self._nh), dtype=self._dtype)
            if self._iscpx:
                self._hamil += backend_changed.asarray(GUE(self._nh), dtype=self._dtype) * self._gamma_power_i
            else:
                self._hamil += backend_changed.asarray(GOE(self._nh), dtype=self._dtype) * self._gamma_power_i
            
            #! add diagonal disorder
            self.randomize()
            self._hamil += backend_changed.diag(self._diagonal)
            
            self._hamiltonian_validate()
        except Exception as e:
            raise e

    # ---------------------------------------------------------------
    
    def _set_local_energy_operators(self):
        """
        Is empty for the Ultrametric model.
        The Hamiltonian is a sum of blocks, and the local energy operators
        are not defined in the same way as in other models.
        The local energy operators are not needed for the Ultrametric model.
        The Hamiltonian is a sum of blocks, and the local energy operators
        are not defined in the same way as in other models.
        """
        pass
    
    # ---------------------------------------------------------------

# -------------------------------------------------------------------