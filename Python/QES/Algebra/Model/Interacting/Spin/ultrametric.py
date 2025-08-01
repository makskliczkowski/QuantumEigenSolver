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
from general_python.algebra.ran_wrapper import RMT, random_matrix, set_global_seed
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
                n               : int                                       = 1,
                J               : float                                     = 1.0,
                alphas          : Union[List[float], float, None]           = None,
                gamma           : float                                     = 1.0,
                dtype           : type                                      = np.float64,
                backend         : str                                       = "default",
                **kwargs):
        # ns: total spins; n: spins in dot; L = ns - n hierarchical levels
        if n >= ns:
            raise ValueError(self._ERR_PARTICLE_MISMATCH)

        L               = ns - n
        # initialize Hilbert space
        _hilbert_space  = hilbert_module.HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)
        super().__init__(is_manybody=True, hilbert_space=_hilbert_space, is_sparse=True, dtype=dtype, backend=backend, **kwargs)

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

        self._log("UltrametricModel initialized.", lvl=2, log='debug')
        self._log(f"ns (total spins): {self._ns}", lvl=3, log='debug')
        self._log(f"n (dot spins): {self._n}", lvl=3, log='debug')
        self._log(f"L (out spins): {self._L}", lvl=3, log='debug')
        self._log(f"J (coupling): {self._J:.3f}", lvl=3, log='debug')
        self._log(f"gamma (normalization): {self._gamma:.3f}", lvl=3, log='debug')
        self._log(f"alphas (hierarchy): {self._alphas}", lvl=3, log='debug')
            
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
        # initialize the blocks
        self._hamil = self._set_Hk(self._hamil, backend=self._backend)

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

    def _set_Hk(self, _hedit: Array, backend = np):
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
        _hedit      = backend.zeros((self._nh, self._nh), dtype=self._dtype)

        dim0        = 2**self._n
        num_blocks  = 2**self._L
        # multiplier for H0
        mult0       = self._gamma/np.sqrt(dim0 + 1)
        for i in range(num_blocks):
            start   = i*dim0; end = (i+1)*dim0
            R0      = random_matrix((dim0, dim0), typek=RMT.GOE,
                                    backend=backend, dtype=self._dtype)
            _hedit[start:end, start:end] += mult0 * R0
        
        # set the Hamiltonian by creating blocks
        for k in range(1, self._L + 1):
            # inner dimension
            dimk     = 2**(self._n + k)
            rest     = 2**(self._L - k)
            au_k     = self._alphas[k-1]**k
            multk    = self._J * self._gamma * au_k / np.sqrt(dimk+1)
            
            # independent blocks per diagonal
            for j in range(rest):
                start   = j*dimk
                end     = (j+1)*dimk
                Rk      = random_matrix((dimk, dimk), typek=RMT.GOE,
                                    backend=backend, dtype=self._dtype)
                _hedit[start:end, start:end] += multk * Rk

        return _hedit
    
    # ---------------------------------------------------------------
    
    def __repr__(self):
        prec    = 3          # decimal places for floats
        tol     = 1e-10      # tolerance for uniformity check
        sep     = ","        # parameter separator
        
        parts   = [
            f"Ultrametric(ns={self.ns}",
            f"N={self.n}",
            self.fmt("J", self.J),
            self.fmt("g", self.gamma),
        ]

        # handle alphas array or scalar
        a = self.alphas
        try:
            amin, amax = min(a), max(a)
            if abs(amax - amin) < tol:
                parts.append(f"a={amin:.{prec}f}")
            else:
                parts.append(f"a[min={amin:.{prec}f}, max={amax:.{prec}f}]")
        except TypeError:
            parts.append(self.fmt("a", a))

        return sep.join(parts) + ")"
        
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
        backend_changed = self._backend if not use_numpy else np
        self._hamil     = backend_changed.zeros((self._nh, self._nh), dtype=self._dtype)
        self._hamil     = self._set_Hk(self._hamil, backend_changed)
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