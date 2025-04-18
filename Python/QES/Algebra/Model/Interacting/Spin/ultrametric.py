"""
file    : Model/Interacting/Spin/ultrametric.py
High-level Hamiltonian class for the Quantum Eigen Solver (QES) package.
This file contains the Ultrametric (UM) model - a hierarchical random-matrix Hamiltonian.
The UM model is defined by a quantum dot of N spins coupled to hierarchical blocks of increasing size.
Derives from a general Hamiltonian.
For more debugrmation see: hierarchical ultrametric ensembles in many-body physics.
"""

import numpy as np
from typing import List, Optional, Union

#! QES package imports
import Algebra.hilbert as hilbert_module
import Algebra.hamil as hamil_module

#! Random matrix wrapper and linear algebra utilities
from general_python.algebra.ran_wrapper import RMT, random_matrix
from general_python.algebra.utils import Array
import general_python.algebra.linalg as linalg

class UltrametricModel(hamil_module.Hamiltonian):
    r"""
    Hierarchical ultrametric random-matrix Hamiltonian.

    `math: H = H0 \otimes I_{2^L}
      + J * sum_{k=1}^L alpha[k-1] * (Hk \otimes I_{2^(L-k)})`

    where H0 is GOE of size 2^N, and each Hk is GOE of size 2^{N+k},
    normalized by gamma/sqrt(dim+1).
    """

    _ERR_PARTICLE_MISMATCH  = "UM: the number of dot spins 'n' must be less than total 'ns'."
    _ERR_ALPHA_SIZE         = "UM: 'alphas' length must equal number of hierarchical levels (ns-n)."

    def __init__(self,
                ns              : int,
                hilbert_space   : Optional[hilbert_module.HilbertSpace]     = None,
                n               : int                               = 1,
                J               : float                             = 1.0,
                alphas          : Union[List[float], float, None]   = None,
                gamma           : float                             = 1.0,
                dtype           : type                              = np.float64,
                backend         : str                               = "default",
                **kwargs):
        # ns: total spins; n: spins in dot; L = ns - n hierarchical levels
        if n >= ns:
            raise ValueError(self._ERR_PARTICLE_MISMATCH)

        L               = ns - n
        # initialize Hilbert space
        hilbert_space   = hilbert_module.HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)
        super().__init__(hilbert_space, is_sparse=True, dtype=dtype, backend=backend, **kwargs)

        self._ns        = ns
        self._n         = n
        self._L         = L
        self._J         = J
        self._gamma     = gamma
        self._is_sparse = False
        self._dimdot    = 2**n
        self._dimout    = 2**L
        
        # parse alphas
        self._set_alphas(alphas)

        self._log(f"UltrametricModel initialized.", lvl=2, log='debug')
        self._log(f"ns (total spins): {self._ns}", lvl=3, log='debug')
        self._log(f"n (dot spins): {self._n}", lvl=3, log='debug')
        self._log(f"L (out spins): {self._L}", lvl=3, log='debug')
        self._log(f"J (coupling): {self._J:.3f}", lvl=3, log='debug')
        self._log(f"gamma (normalization): {self._gamma:.3f}", lvl=3, log='debug')
        self._log(f"alphas (hierarchy): {self._alphas}", lvl=3, log='debug')
            
        # storage for random blocks
        self._hamil         = None
        self._std_en        = None
        self._H0            = None
        # build blocks
        self._init_H0()
        
        # set the Hamiltonian operators
        self._max_local_ch_o = 0
        self._set_local_energy_operators()
        self._set_local_energy_functions()

    @property
    def ns(self) -> int:
        return self._ns

    @property
    def n(self) -> int:
        return self._n

    @property
    def levels(self) -> int:
        return self._L

    @property
    def J(self) -> float:
        return self._J

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def alphas(self) -> List[float]:
        return self._alphas

    def randomize(self, **kwargs):
        '''
        Randomize the Hamiltonian.
        This method is used to generate a new random Hamiltonian
        with the same parameters as the original one but with different
        random blocks.
        '''
        # initialize the Hdot
        self._init_H0()
        # initialize the blocks
        self._hamil = self._set_Hk(self._hamil, self._H0, backend=self._backend)

    # ---------------------------------------------------------------
    
    def _set_alphas(self, alphas: Union[List[float], float, None]):
        '''
        Set the alphas for the hierarchical blocks.
        - If alphas is None, default geometric decay is used.
        - If alphas is a float, all blocks are set to the same value.
        - If alphas is a list, it must have the same length as the number of levels.
        
        Parameters
        ----------
        alphas : Union[List[float], float, None]
            The alphas for the hierarchical blocks.
            If None, default geometric decay is used.
            If float, all blocks are set to the same value.
            If list, must have the same length as the number of levels.
        Raises
        ------
        ValueError
            If alphas is a list and its length does not match the number of levels.
        '''
        #! default geometric decay if none
        if alphas is None:
            self._alphas = [0.5**k for k in range(1, self._L+1)]
        elif isinstance(alphas, float):
            self._alphas = [alphas]*self._L
        elif isinstance(alphas, list) and len(alphas)==self._L:
            self._alphas = alphas
        else:
            raise ValueError(self._ERR_ALPHA_SIZE)

    # ---------------------------------------------------------------
    
    def _init_H0(self):
        """
        Initialize the central dot Hamiltonian H0 of size 2^n.
        The Hamiltonian is a Gaussian Orthogonal Ensemble (GOE) matrix
        with elements drawn from a normal distribution.
        
        The Hamiltonian is normalized by gamma/sqrt(dim+1).
        The matrix is generated using the random_matrix function from the
        general_python.algebra.ran_wrapper module.
        
        """
        dim0 = 2**self._n
        if np.issubdtype(self._dtype, np.complexfloating):
            hdot  = random_matrix((dim0, dim0), typek = RMT.CUE,
                                        backend=self._backend, dtype=self._dtype)
        else:
            hdot  = random_matrix((dim0, dim0), typek = RMT.GOE,
                                        backend=self._backend, dtype=self._dtype)
        self._H0 = (self._gamma/np.sqrt(dim0+1)) * hdot

    # ---------------------------------------------------------------

    def _set_Hk(self, _hedit: Array, _hdot: Array, backend = np):
        """
        Initialize hierarchical blocks Hk of increasing size 2^{n+k}.
        The Hamiltonians are Gaussian Orthogonal Ensemble (GOE) matrices
        with elements drawn from a normal distribution.
        """
        
        # initialize the blocks
        #! Hk = gamma/sqrt(dim+1) * R
        #! where R is a GOE matrix of size 2^{n+k}
        #! and dim = 2^{n+k}
        
        # initialize the matrix
        _hedit = backend.zeros((self._nh, self._nh), dtype=self._dtype)
        
        # set the Hamiltonian by creating blocks
        for k in range(1, self._L + 1):
            # inner dimension
            dimk    = 2**(self._n + k)
            self._log(f"UM: Hk[{k}] = {dimk}", lvl = 3, log = 'debug')
            mult    = (self._gamma/np.sqrt(dimk+1)) * self._alphas[k-1]**(k) * self._J
            
            for i in range(0, 2**(self._L - k)):
                # start index for the block
                start_idx   = i * dimk
                # end index for the block
                end_idx     = (i + 1) * dimk
                R           = random_matrix((dimk, dimk), typek=RMT.GOE,
                                backend=backend, dtype=self._dtype)
                # set the block
                _hedit[start_idx:end_idx, start_idx:end_idx] += R * mult
        
        # set the H0 at each block
        for i in range(0, 2**self._L):
            start_idx   = i * self._dimdot
            end_idx     = (i + 1) * self._dimdot
            _hedit[start_idx:end_idx, start_idx:end_idx] += _hdot
        
        return _hedit
    
    # ---------------------------------------------------------------
    
    def __repr__(self):
        return (f"UltrametricModel, ns={self.ns}, n={self.n}, J={self.J:.3f}, "
                f"gamma={self.gamma:.3f}, alphas={self.alphas}")

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
        self._log(f"UltrametricModel: Hamiltonian initialized.", lvl=2, log='debug')
        self._log(f"UltrametricModel: H0 size: {self._H0.shape}", lvl=3, log='debug')
        self._hamil     = self._set_Hk(self._hamil, self._H0, backend_changed)
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