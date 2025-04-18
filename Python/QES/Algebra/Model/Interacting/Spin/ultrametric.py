"""
file    : Model/Interacting/Spin/ultrametric.py
High-level Hamiltonian class for the Quantum Eigen Solver (QES) package.
This file contains the Ultrametric (UM) model - a hierarchical random-matrix Hamiltonian.
The UM model is defined by a quantum dot of N spins coupled to hierarchical blocks of increasing size.
Derives from a general Hamiltonian.
For more information see: hierarchical ultrametric ensembles in many-body physics.
"""

import numpy as np
from typing import List, Optional, Union

#! QES package imports
import Algebra.hilbert as hilbert_module
import Algebra.hamil as hamil_module

#! Random matrix wrapper and linear algebra utilities
from general_python.algebra.ran_wrapper import RMT, random_matrix
import general_python.algebra.linalg as linalg


class UltrametricModel(hamil_module.Hamiltonian):
    """
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
        hilbert_space = hilbert_module.HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)
        super().__init__(hilbert_space, is_sparse=True, dtype=dtype, backend=backend, **kwargs)

        self._ns        = ns
        self._n         = n
        self._L         = L
        self._J         = J
        self._gamma     = gamma
        self._is_sparse = False
        
        # parse alphas
        self._set_alphas(alphas)

        # storage for random blocks
        self._hamil     = None
        self._std_en    = None
        self._H0        = None
        self._Hk        = None
        
        # build blocks
        self._init_H0()
        self._init_Hk()
        
        # set the Hamiltonian operators
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

    # ---------------------------------------------------------------
    
    def _set_alphas(self, alphas: Union[List[float], float, None]):
        #! default geometric decay if none
        if alphas is None:
            self._alphas = [0.5**k for k in range(1, self._L+1)]
        elif isinstance(alphas, float):
            self._alphas = [alphas]*self._L
        elif isinstance(alphas, list) and len(alphas)==self._L:
            self._alphas = alphas
        else:
            raise ValueError(self._ERR_ALPHA_SIZE)

    def _init_H0(self):
        """
        Initialize the central dot Hamiltonian H0 of size 2^n.
        The Hamiltonian is a Gaussian Orthogonal Ensemble (GOE) matrix
        with elements drawn from a normal distribution.
        """
        dim0    = 2**self._n
        if np.issubdtype(self._dtype, np.complexfloating):
            hdot  = random_matrix((dim0, dim0), typek = RMT.CUE,
                                        backend=self._backend, dtype=self._dtype)
        else:
            hdot  = random_matrix((dim0, dim0), typek = RMT.GOE,
                                        backend=self._backend, dtype=self._dtype)
        self._H0 = (self._gamma/np.sqrt(dim0+1)) * hdot

    def _init_Hk(self):
        """
        Initialize hierarchical blocks Hk of increasing size 2^{n+k}.
        The Hamiltonians are Gaussian Orthogonal Ensemble (GOE) matrices
        with elements drawn from a normal distribution.
        """
        
        # initialize the blocks
        #! Hk = gamma/sqrt(dim+1) * R
        #! where R is a GOE matrix of size 2^{n+k}
        Hk = self._backend.zeros((self._nh, self._nh), dtype=self._dtype)
        for k in range(0, self._L):
            # inner dimension
            dimk = 2**(self._n + (k + 1))
            hkin = self._backend.zeros((dimk, dimk), dtype=self._dtype)
            for i in range(0, 2**(self._L - k)):
                R = random_matrix((dimk, dimk), typek=RMT.GOE,
                                backend=self._backend, dtype=self._dtype)
                hkin[i * dimk:(i + 1) * dimk, i * dimk:(i + 1) * dimk] += R
            Hk += (self._gamma/np.sqrt(dimk+1)) * hkin * self._alphas[k]**(k+1)
        self._Hk = Hk

    # ---------------------------------------------------------------
    
    def __repr__(self):
        return (f"UltrametricModel, ns={self.ns}, n={self.n}, J={self.J:.3f}, "
                f"gamma={self.gamma:.3f}, alphas={self.alphas}")

    def _hamiltonian(self, use_numpy: bool = False):
        """
        Build the full ultrametric Hamiltonian.
        """
        
        if self._nh == 0:
            raise ValueError("UltrametricModel: Hamiltonian not initialized.")
        
        self._log(f"UM: alpha={self._alphas[0]:.3f}", lvl = 2, log = 'debug')
        self._log(f"UM: J={self._J:.3f}", lvl = 2, log = 'debug')
        
        # initialize zero Hamiltonian container
        super()._hamiltonian(use_numpy)
        
        backend_changed = self._backend if not use_numpy else np
        eye             = linalg.sparse.identity(self._dimout, backend=backend_changed, dtype=self._dtype)
        kron_prod       = linalg.sparse.kron(self._hdot, eye, backend=backend_changed)
        self._hamil     = kron_prod + self._J * self._Hk
        
    # ---------------------------------------------------------------
    
    def _set_local_energy_operators(self):
        """
        Sets up the local energy operators for the quantum system.
        This method initializes and registers energy operators acting on the spin sites.
        It creates two types of operators:
            - A local spin-z operator (sig_z) acting on individual sites.
            - A spin-x correlation operator (sig_x) acting on pairs of sites.
        For each site index i in the range [self.n, self.ns):
            - Computes an offset index (part_idx = i - self.n) to access the corresponding
                parameters from self._h, self._au, and self._neidot.
            - Registers the local operator with a multiplier given by self._h[part_idx].
            - Registers the correlation operator (modifying the state) with a multiplier given
              by the product self.g0 * self._au[part_idx], acting between site i and the site
                indicated by self._neidot[part_idx].
        Additionally, debug log messages are generated at the start of processing each site,
        and a success message is logged after all operators have been set.
        Returns:
            None
        """

        pass