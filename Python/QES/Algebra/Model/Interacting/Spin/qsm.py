"""
file    : Model/Interacting/Spin/qsm.py
High-level Hamiltonian class for the Quantum Eigen Solver (QES) package.
This file contains the QSM model - a Hamiltonian for a quantum dot coupled to an external system.
The external system is modeled as a quantum spin chain.
Is an instance of the QSM model.
Derives from a general Hamiltonian.
For more information see the QSM model.
https://arxiv.org/pdf/2308.07431.pdf

Based on the C++ implementation. 
Author  : Maksymilian Kliczkowski
Date    : 2025-02-17
Version : 0.1
"""

import numpy as np
import math
from typing import List, Tuple, Union, Optional

# Assume these are available from the QES package:
from Algebra.hilbert import HilbertSpace
from Algebra.hamil import Hamiltonian
from Algebra.Operator.operators_spin import _sigma_z_int, _sigma_x_int

##########################################################################################
from general_python.algebra.linalg import kron, identity, hilbert_schmidt_norm
from general_python.algebra.ran_wrapper import choice, randint, RMT, random_matrix, random_vector
from scipy.sparse import csr_matrix
##########################################################################################

_QSM_CHECK_HS_NORM = True

class QSM(Hamiltonian):
    '''
    Hamiltonian for an ergodic quantum dot coupled to an external system.
    The external system is modeled as a quantum spin chain.
    '''
    
    _ERR_PARTICLES_DONT_MATCH   = "QSM: the number of particles in the dot is larger than the number of particles in the system."
    _ERR_COUP_VEC_SIZE          = "QSM: the size of the coupling vector 'a' is not equal to the number of particles outside the dot."
    _ERR_RAND_BOX_SIZE          = "QSM: the size of the random box distribution vector 'xi' is not equal to the number of particles outside the dot."
    _ERR_MAG_FIELD_SIZE         = "QSM: the size of the magnetic field vector 'h' is not equal to the number of particles outside the dot."
    _ERR_EITHER_HIL_OR_NS       = "QSM: either the Hilbert space or the number of particles in the system must be provided."
    
    def __init__(self,
                ns                  : Optional[int] = None,
                hilbert_space       : Optional[HilbertSpace] = None,
                n                   : int = 1,
                gamma               : float = 1.0,
                g0                  : float = 1.0,
                a                   : Union[List[float], None, float] = None,
                h                   : Union[List[float], None, float] = 1.0,
                xi                  : Union[List[float], None, float] = 0.2,
                dtype               : type = np.float32,
                backend             : str = "default",
                **kwargs):
        '''
        Constructor for the QSM Hamiltonian.
        Parameters:
            hilbert_space : HilbertSpace
                Hilbert space for the Hamiltonian - this shall be the Hilbert space for spin 1/2 particles.
            n : int
                Number of particles in the dot (ergodic quantum dot).
            gamma : float
                Hilbert-Schmidt norm of the coupling operator normalizer.
            g0 : float
                Coupling strength between the particles in the dot and the particles outside the dot.
            a : Union[List[float], None, float]
                Coupling between the dot and outside world
            h : Union[List[float], None, float]
                Magnetic field vector for the particles outside the dot. If it is None, it will be
                initialized to [1.0] * (Ns - N). If it is a float, it will be initialized to [h] * (Ns - N).
                If it is a list, it will be used as is.
            xi : Union[List[float], None, float]
                Random box distribution vector for the particles outside the dot. If 
                it is None, it will be initialized to [0.2] * (Ns - N). If it is a float,
                it will be initialized to [xi] * (Ns - N).
            dtype : type
                Data type for the Hamiltonian matrix. Default is np.float64.
            backend : str
                Backend for the Hamiltonian matrix (default is "default").
        '''

        # Initialize the Hamiltonian
        
        if hilbert_space is None:
            if ns is None:
                raise ValueError(self._ERR_EITHER_HIL_OR_NS)
            hilbert_space = HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)
        
        # Initialize the Hamiltonian            
        super().__init__(hilbert_space, is_sparse=True, dtype=dtype, backend=backend, **kwargs)
        
        # setup the internal variables
        self._n     =  n                # Number of particles in the dot
        self._gamma = gamma             # Hilbert-Schmidt norm of the coupling operator normalizer
        self._g0    = g0                # Coupling strength between the particles in the dot and the particles outside the dot
        self._nout  = 0                 # Number of particles outside the dot
        self._a, self._h, self._xi      = None, None, None
        self.check_sizes(a, h, xi)      # Check the sizes of the coupling vector, random box distribution vector, and magnetic field vector       
        
        # Initialize the Hamiltonian
        self._hdot      = None
        self._dimin     = 0
        self._dimout    = 0
        self._neidot    = []
        self._au        = None
        self. _u        = None
        self._name      = "Quantum Sun Model"
        self.init_particles()
    
    # ----------------------------------------------------------------------------------------------
    
    def __repr__(self):
        ''' Return the string representation of the QSM model. '''
        sep     = ","
        skip    = []
        # Check if all elements except the first in each list are equal to the first value
        different_alpha = ( not all(np.isclose(a, self.alphas[0]) for a in self.alphas[1:])
                            if len(self.alphas) > 1
                            else False)
        different_h = ( not all(np.isclose(h, self.h[0]) for h in self.h[1:])
                        if len(self.h) > 1
                        else False)
        different_xi = (not all(np.isclose(xi, self.xi[0]) for xi in self.xi[1:])
                        if len(self.xi) > 1
                        else False)

        name    =  sep + f"qsm,Ns={self.ns}"
        name    += sep + f"N={self.n}"
        name    += sep + f"gamm={self.gamma:.2f}"
        name    += sep + f"g0={self.g0:.2f}"

        if "alpha" not in skip:
            name    += sep + ("alpha=r" if different_alpha else f"alpha={self.alphas[0]:.3f}")
        if "h" not in skip:
            name    += sep + ("h=r" if different_h else f"h={self.h[0]:.3f}")
        if "xi" not in skip:
            name    += sep + ("xi=r" if different_xi else f"xi={self.xi[0]:.3f}")

        name += self._hilbert_space.get_sym_info()
        return name

    # ----------------------------------------------------------------------------------------------
    
    @property
    def n(self) -> int:
        ''' Return the number of particles in the dot. '''
        return self._n
    
    @property
    def nout(self) -> int:
        ''' Return the number of particles outside the dot. '''
        return self._nout
    
    @property
    def gamma(self) -> float:
        ''' Return the Hilbert-Schmidt norm of the coupling operator normalizer. '''
        return self._gamma
    
    @property
    def g0(self) -> float:
        ''' Return the coupling strength between the particles in the dot and the particles outside the dot. '''
        return self._g0
    
    @property
    def alphas(self) -> List[float]:
        ''' Return the coupling vector alpha. '''
        return self._a
    
    @property
    def h(self) -> List[float]:
        ''' Return the magnetic field vector. '''
        return self._h
    
    @property
    def xi(self) -> List[float]:
        ''' Return the random box distribution vector. '''
        return self._xi

    # ----------------------------------------------------------------------------------------------

    def check_sizes(self, a, h, xi):
        '''
        Check the sizes of the coupling vector, random box distribution vector, and magnetic field vector.
        
        Parameters:
            a : Union[List[float], None, float]
                Coupling between the dot and outside world
            h : Union[List[float], None, float]
                Magnetic field vector for the particles outside the dot
            xi : Union[List[float], None, float]
                Random box distribution vector for the particles outside the dot
        '''
        
        if self.n >= self.ns:
            raise ValueError(self._ERR_PARTICLES_DONT_MATCH)

        # define the number of particles outside the dot
        self._nout = self.ns - self.n
        self.set_alpha(a, initialize = False)
        self.set_h(h, initialize = False)
        self.set_xi(xi, initialize = False)

    # ----------------------------------------------------------------------------------------------

    def __init_hdot(self):
        ''' Initialize the random Hamiltonian for the particles outside the dot. '''
        hdot = None
        if np.issubdtype(self._dtype, np.complexfloating):
            hdot  = random_matrix((self._dimin, self._dimin), typek = RMT.CUE,
                                        backend=self._backend, dtype=self._dtype)
        else:
            hdot  = random_matrix((self._dimin, self._dimin), typek = RMT.GOE,
                                        backend=self._backend, dtype=self._dtype)
        hdot      = self._gamma / np.sqrt(self._dimin + 1) * hdot
        
        if _QSM_CHECK_HS_NORM:
            _norm           = hilbert_schmidt_norm(hdot, backend=self._backend)
            self._log(f"H_dot norm: {_norm}", lvl = 2)
            return hdot / np.sqrt(_norm)
        return hdot

    def __init_distances(self):
        ''' Initialize the random distances for the 'free' particles. '''
        nout        = self.nout
        xi          = self._xi
        u           = np.ones(nout)
        for i in range(1, nout):
            u[i]    = np.random.uniform(low=i-xi[i], high=i+xi[i])
        self._u     = u
        return u
    
    def __init_a_distances(self):
        ''' Initialize the random distances for the 'free' particles. '''
        self._au = self._a ** self._u
    
    def init_particles(self):
        ''' Initialize the particles in the dot and outside the dot. '''
        self._dimin     = 2**(self.n)
        self._dimout    = 2**(self.nout)

        # initialize random neighbors for the 'free' particles
        self._neidot    = np.random.choice(list(range(self.n)), self.nout, replace=True)

        # initialize the random distances for the 'free' particles
        self.__init_distances()
        self.__init_a_distances()

        # log information
        self._log(f"alpha={self._a}", lvl = 1)
        self._log(f"u={self._u}", lvl = 1)
        self._log(f"alpha^u={self._au}", lvl = 1)

        self._hdot      = self.__init_hdot()

    # ----------------------------------------------------------------------------------------------

    def set_alpha(self, a, initialize = True):
        ''' Set the coupling vector alpha. '''
        if isinstance(a, list) and len(a) == self.nout:
            self._a = a
        elif isinstance(a, float):
            self._a = [a] * self.nout
        elif isinstance(a, str):
            self._a = random_vector(self.nout, a, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_COUP_VEC_SIZE)
        
        # calculate the new coupling strength
        if initialize:
            self.__init_distances()
        
    def set_h(self, h, initialize = True):
        ''' Set the magnetic field vector. '''
        if isinstance(h, list) and len(h) == self.nout:
            self._h = h
        elif isinstance(h, float):
            self._h = [h] * self.nout
        elif isinstance(h, str):
            self._h = random_vector(self.nout, h, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_MAG_FIELD_SIZE)
    
    def set_xi(self, xi, initialize = True):
        ''' Set the random box distribution vector. '''
        if isinstance(xi, list) and len(xi) == self.nout:
            self._xi = xi
        elif isinstance(xi, float):
            self._xi = [xi] * self.nout
        elif isinstance(xi, str):
            self._xi = random_vector(self.nout, xi, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_RAND_BOX_SIZE)
        
        # calculate the new random distances
        if initialize:
            self.__init_distances()
            self.__init_a_distances()
            
    # ----------------------------------------------------------------------------------------------
    
    def randomize(self, **kwargs):
        '''
        Randomize the QSM Hamiltonian. 
        Parameters:
            alpha : Union[List[float], None]
        Coupling between the dot and outside world
            h : Union[List[float], None]
        Magnetic field vector for the particles outside the dot
            xi : Union[List[float], None]
        Random box distribution vector for the particles outside the dot
        '''
        if "alpha" in kwargs:
            self.set_alpha(kwargs["alpha"], False)
        if "h" in kwargs:
            self.set_h(kwargs["h"], False)
        if "xi" in kwargs:
            self.set_xi(kwargs["xi"], False)
        self.init_particles()

    # ----------------------------------------------------------------------------------------------

    def get_mobility_edge(self, energy: float):
        ''' 
        Compute the mobility edge for the QSM model.
        It is calculated as in: Konrad Pawlik, Piotr Sierant, Lev Vidmar, Jakub Zakrzewski (2023)
        '''
        if self._nh == 0:
            self._log("Empty Hilbert, not counting anything.", "INFO", 1)
            return None
        
        if self._std_en is None:
            self._std_en = np.std(self.eig_val)

        _std            = self._std_en / np.sqrt(self.ns)
        _eps            = (energy - self.eig_val[0]) / (self.eig_val[self._nh - 1] - self.eig_val[0])
        _bwd            = (self.eig_val[self._nh - 1] - self.eig_val[0])
        _bwd            = _bwd / float(self.ns)
        return np.exp(_bwd * _bwd * (_eps - 0.5) * (_eps - 0.5) / _std / _std / 4.0) / np.sqrt(2.0)

    # ----------------------------------------------------------------------------------------------

    def _hamiltonian(self):
        '''
        Build the Hamiltonian of the QSM model.
        The Hamiltonian is built in the following way:
        1. The first term is the Hamiltonian of the dot particles. This is a random matrix.
        2. The second term is the coupling between the dot and the outside world. This is a spin-flip interaction.
        3. The third term is the magnetic field of the dot particles.
        '''
        
        if self._nh == 0:
            self._log("Empty Hilbert, not building anything.", "INFO", 1)
            return

        self._log(f"QSM: alpha={self._a[0]:.3f}, xi={self._xi[0]:.3f}", lvl = 2)
        
        for i in range(self.nout):
            self._log(f"QSM: i={i} -> h={self._h[i]:.3e}, a^u={self._au[i]:.3e}", lvl = 3)

        super()._hamiltonian()

        eye         = identity(self._dimout, backend=self._backend, dtype=self._dtype)
        kron_prod   = kron(self._hdot, eye, backend=self._backend)
        self._hamil = self._backend.array(self.hamil + kron_prod)

    # ----------------------------------------------------------------------------------------------
    #! ABSTRACT METHODS OVERRIDE
    # ----------------------------------------------------------------------------------------------
    
    def loc_energy_ham(self, ham, hilbert : HilbertSpace, k : int, k_map : int, i : int):
        '''
        Compute the local energy Hamiltonian.
        Parameters:
            ham : List[Tuple[int, int, float]]
                Hamiltonian matrix
            k : int
                Index of the Hilbert space element
            k_map : int
                Index of the Hilbert space element mapped to the Hamiltonian
            i : int
                Index of the particle
        '''
        if i < self._n:
            return
        
        part_idx    = i - self.n
        
        # check the spin of the particle
        idx, val    = _sigma_z_int(k_map, self.ns, [i])
        # apply the magnetic field of the particle
        self.set_hamil_elem(ham, hilbert, k, self.h[part_idx] * val, idx)
        
        # apply the spin flips
        n           = self._neidot[part_idx]
        idx1, sxn   = _sigma_x_int(k_map, self.ns, [n])
        idx2, sxj   = _sigma_x_int(idx1, self.ns, [i])
        
        # apply the coupling between the dot and the outside world
        self.set_hamil_elem(ham, hilbert, k, self.g0 * self._au[part_idx] * sxj * sxn, idx2)
        
    def loc_energy(self, *args):
        """
        !TODO: Add description and implementation
        """
        return 0.0
    
    # ----------------------------------------------------------------------------------------------


####################################################################################################