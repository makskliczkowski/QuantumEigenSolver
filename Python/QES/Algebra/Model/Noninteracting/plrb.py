"""
file    : Model/Noninteracting/plrb.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
"""

import numpy as np
from typing import List, Optional, Union

#! QES package imports
import Algebra.hilbert as hilbert_module
import Algebra.hamil as hamil_module

#! Random matrix wrapper and linear algebra utilities
from general_python.algebra.ran_wrapper import set_global_seed
from general_python.algebra.utils import Array

class PowerLawRandomBanded(hamil_module.Hamiltonian):
    r"""
    PowerLawRandomBanded is a Hamiltonian model for a system of non-interacting spins
    with a power-law distribution of random banded blocks.
    This model is a generalization of the Ultrametric model, where the Hamiltonian
    is constructed from a central dot Hamiltonian and hierarchical blocks with
    a power-law decay of the coupling strength.
    The Hamiltonian is defined as:
    """

    def __init__(self,
                ns              : int,
                a               : float                                     = 1.0,
                b               : float                                     = 1.0,                       
                hilbert_space   : Optional[hilbert_module.HilbertSpace]     = None,
                dtype           : type                                      = np.float64,
                backend         : str                                       = "default",
                many_body       : bool                                      = True,
                seed            : Optional[int]                             = None,
                **kwargs):

        # initialize Hilbert space
        self._many_body     = many_body
        self._ns            = ns if many_body else np.log2(ns)
        self._nh            = 2**self._ns if many_body else ns

        _hilbert_space      = hilbert_module.HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)
        super().__init__(is_manybody = True, 
            hilbert_space=_hilbert_space, is_sparse=True, seed=seed,
            dtype=dtype, backend=backend, **kwargs)

        self._is_sparse     = False
        self._a             = a
        self._b             = b

        # storage for random blocks
        self._hamil         = None
        self._std_en        = None
        self._seed          = kwargs.get('seed', None)
        set_global_seed(self._seed, backend=self._backend)

        # set the Hamiltonian operators
        self._max_local_ch_o = 0
        self._set_local_energy_operators()
        self._set_local_energy_functions()

    @property
    def ns(self) -> int:
        return self._ns
    
    @property
    def a(self) -> Union[List[float], float, None]:
        return self._a

    @property
    def b(self) -> Union[List[float], float, None]:
        return self._b

    @property
    def many_body(self) -> bool:
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
        self._hamil = self._set_Hk(self._hamil, backend=self._backend)

    # ---------------------------------------------------------------

    def _set_Hk(self, _hedit: Array, backend = np, rng: Optional[np.random.Generator] = None) -> Array:
        

        if self._nh == 0 or rng is None:
            raise ValueError("PLRB: Hamiltonian not initialized.")
        
        # Power-law random banded Hamiltonian construction
        H      = backend.zeros((self._nh, self._nh), dtype=backend.float64)
        power  = 2.0 * self._a
        binv   = 1.0 / self._b

        for i in range(self._nh):
            for j in range(i, self._nh):
                if i == j:
                    val     = rng.uniform(-1.0, 1.0)
                else:
                    dist    = abs(i - j) * binv
                    denom   = backend.sqrt(1.0 + dist**power)
                    val     = rng.uniform(-1.0, 1.0) / denom
                H[i, j] = val
                H[j, i] = val # symmetric

        return H
    
    # ---------------------------------------------------------------
    
    def __repr__(self):
        return f"PLRB(ns={self.ns},a={self.a:.3f},b={self.b:.3f})"

    def __str__(self):
        return f"PLRB(ns={self.ns},a={self.a:.3f},b={self.b:.3f})"

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
        backend_changed = self._backend if not use_numpy else np
        self._hamil     = backend_changed.zeros((self._nh, self._nh), dtype=self._dtype)
        self._hamil     = self._set_Hk(self._hamil, backend_changed, rng=np.random.default_rng(self._seed))
        self._hamiltonian_validate()
        
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