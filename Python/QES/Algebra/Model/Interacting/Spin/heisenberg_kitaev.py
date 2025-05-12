"""
file    : Model/Interacting/Spin/heisenberg_kitaev.py
High-level Hamiltonian class for the Quantum Eigen Solver (QES) package.

Based on the C++ implementation. 
Author  : Maksymilian Kliczkowski
Date    : 2025-02-17
Version : 0.1
"""

import numpy as np
import numba
from typing import List, Tuple, Union, Optional

# Assume these are available from the QES package:
import Algebra.hilbert as hilbert_module
import Algebra.hamil as hamil_module
import Algebra.Operator.operators_spin as operators_spin_module

##########################################################################################
import general_python.algebra.linalg as linalg
from general_python.lattices.lattice import Lattice, LatticeType
from general_python.lattices.honeycomb import HoneycombLattice

# ----------------------------------------------------------------------------------------
#! DEFINE CONSTANTS
# ----------------------------------------------------------------------------------------

HEI_KIT_Z_BOND_NEI = 1
HEI_KIT_Y_BOND_NEI = 2
HEI_KIT_X_BOND_NEI = 0

##########################################################################################
#! HAMILTONIAN CLASS
##########################################################################################

class HeisenbergKitaev(hamil_module.Hamiltonian):
    '''
    Hamiltonian for an ergodic quantum dot coupled to an external system.
    The external system is modeled as a quantum spin chain.
    '''

    #############################
    
    _ERR_EITHER_HIL_OR_NS       = "QSM: either the Hilbert space or the number of particles in the system must be provided."
    
    def __init__(self,
                lattice             : Optional[Lattice],
                hilbert_space       : Optional[hilbert_module.HilbertSpace] = None,
                hx                  : Union[List[float], None, float]       = 1.0,
                hz                  : Union[List[float], None, float]       = 1.0,
                kx                  : Union[List[float], None, float]       = 1.0,
                ky                  : Union[List[float], None, float]       = 1.0,
                kz                  : Union[List[float], None, float]       = 1.0,
                j                   : Union[List[float], None, float]       = 1.0,
                dlt                 : Union[List[float], None, float]       = 1.0,
                dtype               : type                                  = np.float32,
                backend             : str                                   = "default",
                logger              : Optional['Logger']                    = None,
                **kwargs):
        '''
        Constructor for the QSM Hamiltonian.
        
        ---
        Parameters:
            lattice : Optional[Lattice]:
                The lattice structure for the Hamiltonian.
            hilbert_space : Optional[hilbert_module.HilbertSpace]:
                The Hilbert space associated with the Hamiltonian.
            hx : Union[List[float], None, float]:
                Magnetic field for the x-direction.
            hz : Union[List[float], None, float]:
                Magnetic field for the z-direction.
            kx : Union[List[float], None, float]:
                Coefficient for the x-direction Kitaev interaction.
            ky : Union[List[float], None, float]:
                Coefficient for the y-direction Kitaev interaction.
            kz : Union[List[float], None, float]:
                Coefficient for the z-direction Kitaev interaction.
            j : Union[List[float], None, float]:
                Coupling constant.
            dlt : Union[List[float], None, float]:
                Delta parameter.
            dtype : type:
                Data type for the Hamiltonian (default: np.float32).
            backend : str:
                Backend to use for computations (default: "default").
            **kwargs :
                Additional keyword arguments.
        '''
        
        # Initialize the Hamiltonian
        self._lattice                   = lattice
        super().__init__(is_manybody=True, hilbert_space=hilbert_space, lattice=lattice, is_sparse=True,
                         dtype=dtype, backend=backend, logger=logger, **kwargs)

        # Initialize the Hamiltonian
        if hilbert_space is None:
            if self.ns is None:
                raise ValueError(self._ERR_EITHER_HIL_OR_NS)
            hilbert_space = hilbert_module.HilbertSpace(ns=self.ns, backend=backend, dtype=dtype, nhl=2) 
        
        # setup the fields
        self._hx                        = hx
        self._hz                        = hz
        self._kx                        = kx
        self._ky                        = ky
        self._kz                        = kz
        self._j                         = j
        self._dlt                       = dlt

        if self._lattice.typek not in [LatticeType.HONEYCOMB, LatticeType.HEXAGONAL]:
            self._log(f"The type of the lattice {self._lattice} is not standard. Check your intentions...", lvl = 2)
        self._neibz                     = [[]]
        self._neiby                     = [[]]
        self._neibx                     = [[]]
        self._neiadd                    = [[]]
        
        # Initialize the Hamiltonian
        self._name                      = "Heisenberg Kitaev Model"
        self._is_sparse                 = True
        self._max_local_ch              = 6
        
        self.set_couplings()
        
        # functions for local energy calculation in a jitted way (numpy and jax)
        self._set_local_energy_operators()
        self._set_local_energy_functions()

        # test the Hamiltonian and allow jit to be built - trigger the jit compilation        

        # self._loc_energy_np_fun         = _local_energy_arr_wrap(self.ns, self._hx, self._hz, self._j, self._dlt,
        #                                                         self._kx, self._ky, self._kz, 
        #                                                         self._neibz, self._neiby, self._neibx, self._neiadd)
        # if JAX_AVAILABLE:
            # self._loc_energy_jax_fun    = _local_energy_arr_wrap(self._n, self._neidot, self._h, self._g0, self._au, use_jax=True)        
        # self._local_energy_test()
    
    # ----------------------------------------------------------------------------------------------
    
    def __repr__(self):
        """
        Returns a string representation of the model.
        """
        sep     = ","
        prec    = 3

        def param_str(param_name, values):
            # Check that the list is non-empty and uniform.
            if values and all(x == values[0] for x in values):
                # Format the common value using the desired precision.
                return f"{param_name}={values[0]:.{prec}f}"
            else:
                return f"{param_name}=r"

        # Build the representation string.
        info_str = f"{sep}heikit,Ns={self.Ns}"
        info_str += sep + param_str("J", self.J)
        info_str += sep + param_str("Kx", self.Kx)
        info_str += sep + param_str("Ky", self.Ky)
        info_str += sep + param_str("Kz", self.Kz)
        info_str += sep + param_str("dlt", self.delta)
        info_str += sep + param_str("hz", self.hz)
        info_str += sep + param_str("hx", self.hx)

        # Append symmetry information from the hilbertSpace.
        info_str += self.hilbertSpace.getSymInfo()
        info_str += sep + str(self.hilbertSpace.getBC())

        # If this class inherits from a Hamiltonian base class that further augments info,
        # you might call that method here. For now, we simply return the constructed string.
        return info_str

    # ----------------------------------------------------------------------------------------------
    
    def __str__(self):
        ''' Return the string representation of the QSM model. '''
        return self.__repr__()

    # ----------------------------------------------------------------------------------------------
    #! INIT
    # ----------------------------------------------------------------------------------------------
    
    def set_couplings(self,
                        hx      : Union[List[float], None, float]       = None,
                        hz      : Union[List[float], None, float]       = None,
                        kx      : Union[List[float], None, float]       = None,
                        ky      : Union[List[float], None, float]       = None,
                        kz      : Union[List[float], None, float]       = None,
                        j       : Union[List[float], None, float]       = None,
                        dlt     : Union[List[float], None, float]       = None):
        '''
        Sets the couplings based on their initial value (list, string, value)
        '''
        self._hx            = self._set_some_coupling(self._hx  if hx  is None else hx )
        self._hz            = self._set_some_coupling(self._hz  if hz  is None else hz )
        self._kx            = self._set_some_coupling(self._kx  if kx  is None else kx )
        self._ky            = self._set_some_coupling(self._ky  if ky  is None else ky )
        self._kz            = self._set_some_coupling(self._kz  if kz  is None else kz )
        self._j             = self._set_some_coupling(self._j   if j   is None else j  )
        self._dlt           = self._set_some_coupling(self._dlt if dlt is None else dlt)
    
    def _set_local_energy_operators(self):
        """
        Set up the local and non-local energy operators for the spin lattice.
        This method constructs the operator lists representing local (single-site)
        and correlation (two-site) interactions for the system. It iterates over
        each site and performs the following steps:
        
            - Initializes lists to store local (operators_local) and non-local
                (operators) operator tuples.
            - For each site:
                - Creates local operators (sig_x and sig_z) acting on the site.
                - Creates correlation operators for sig_x, sig_y, and sig_z.
                - Appends local operators to the local operators list with their associated
                    field strengths (from self._hx and self._hz).
            - For each site, iterates over forward nearest neighbors as provided by the
                lattice object:
                - Retrieves the neighbor indices using lattice.get_nn_forward.
                - Computes the interaction multipliers based on Heisenberg coupling terms
                    (self._j and self._dlt) and Kitaev interaction contributions (self._kx,
                    self._ky, self._kz), with adjustments made according to the bond directions
                    (e.g., HEI_KIT_Z_BOND_NEI, HEI_KIT_Y_BOND_NEI).
                - Appends the corresponding correlation operator tuples to the operator lists.
            - Logs detailed debug messages at various levels throughout the process.
            
        The resulting operator tuples are stored as:
            - self._local_ops: a list containing tuples of (operator, [site index], coefficient)
                for local energy contributions.
            - self._nonlocal_ops: a list containing tuples of (operator, [site index, neighbor index],
                coefficient) for two-site interactions.
                
        Note:
            This method updates internal state and does not return a value.
        """
        super()._set_local_energy_operators()
        
        # operators
        lattice         = self._lattice
        #! define the operators beforehand
        op_sx_l         =   operators_spin_module.sig_x(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Local)
        op_sz_l         =   operators_spin_module.sig_z(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Local)
        
        # Kitaev and Heisenberg terms
        op_sx_sx_c      =   operators_spin_module.sig_x(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Correlation)
        op_sy_sy_c      =   operators_spin_module.sig_y(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Correlation)
        op_sz_sz_c      =   operators_spin_module.sig_z(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Correlation)
        
        for i in range(self.ns):
            self._log(f"Starting i: {i}", lvl = 1, log = 'debug')
            
            # now check the local operators
            self.add(op_sz_l, multiplier = self._hz[i], modifies = False, sites = [i])
            self.add(op_sx_l, multiplier = self._hx[i], modifies = True, sites = [i])
            
            self._log(f"Adding local Sz at {i} with value {self._hz[i]:.2f}", lvl = 2, log = 'debug')
            self._log(f"Adding local Sx at {i} with value {self._hx[i]:.2f}", lvl = 2, log = 'debug')
            
            # now check the correlation operators
            nn_forward_num = lattice.get_nn_forward_num(i)
            for nn in range(nn_forward_num):
                nei = lattice.get_nn_forward(i, num=nn)

                # check the direction of the bond
                if nei < 0:
                    continue
                
                # Heisenberg - value of SzSz (multiplier)
                sz_sz = self._j[i] * self._dlt[i]
                # Heisenberg - value of SxSx (multiplier)
                sx_sx = self._j[i] * self._dlt[i]
                # Heisenberg - value of SySy (multiplier)
                sy_sy = self._j[i] * self._dlt[i]
                
                if nn == HEI_KIT_Z_BOND_NEI:
                    sz_sz += self._kz[i]
                elif nn == HEI_KIT_Y_BOND_NEI:
                    sy_sy += self._ky[i]
                else:
                    sx_sx += self._kx[i]

                # append the operators
                self.add(op_sz_sz_c, sites = [i, nei], multiplier = sz_sz, modifies = False)
                self._log(f"Adding SzSz at {i},{nei} with value {sz_sz:.2f}", lvl = 2, log = 'debug')
                self.add(op_sx_sx_c, sites = [i, nei], multiplier = sx_sx, modifies = True)
                self._log(f"Adding SySy at {i},{nei} with value {sy_sy:.2f}", lvl = 2, log = 'debug')
                self.add(op_sy_sy_c, sites = [i, nei], multiplier = sy_sy, modifies = True)
                self._log(f"Adding SxSx at {i},{nei} with value {sx_sx:.2f}", lvl = 2, log = 'debug')
        self._log("Successfully set local energy operators...", lvl=1, log='info')

    # ----------------------------------------------------------------------------------------------

####################################################################################################