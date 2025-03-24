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
from general_python.algebra.ran_wrapper import RMT, random_matrix, random_vector
from general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, _JAX_AVAILABLE
##########################################################################################

# ----------------------------------------------------------------------------------------
#! DEFINE CONSTANTS
# ----------------------------------------------------------------------------------------

HEI_KIT_Z_BOND_NEI = 1
HEI_KIT_Y_BOND_NEI = 2
HEI_KIT_X_BOND_NEI = 0

# ----------------------------------------------------------------------------------------
#! ARRAY STATES
# ----------------------------------------------------------------------------------------

@numba.njit
def _local_energy_arr_loop(state,
                        hx          : np.ndarray,
                        hz          : np.ndarray,
                        j_cpl       : np.ndarray,
                        delta_cpl   : np.ndarray,
                        kx_cpl      : np.ndarray,
                        ky_cpl      : np.ndarray,
                        kz_cpl      : np.ndarray,
                        neib_z      : np.ndarray,
                        neib_y      : np.ndarray,
                        neib_x      : np.ndarray,
                        addit_n     : np.ndarray):
    """
    Compute the local energy interaction for a numpy array state.
    
    Parameters:
        state   : numpy array of spins (0 or 1)
        neidot  : numpy array of neighbor indices for free particles
        h, g0, n, au : model parameters
        sigma_x, sigma_z : functions to compute the spin operators
    
    Returns:
        Tuple [states], [energies] containing the new states and energies.
    """
    size        = state.shape[0]
    local_value = 0.0
    
    states_size = 1 + (1 + neib_x.shape[0] + addit_n.shape[0]) * size
    new_states  = np.empty((states_size, size), dtype=state.dtype)
    vals        = np.empty(states_size, dtype=hx.dtype)
    
    for i in numba.prange(len(state)):
        # Apply the local energy interaction.
        part_idx    = i - n
        this_site   = np.array([i], dtype=DEFAULT_NP_INT_TYPE)
        
        # Apply the sigma_z operator
        _, val      = sigma_z(state, this_site)
        vals        = h[part_idx] * val.astype(h.dtype)
        
        # Apply the sigma_x * sigma_x operator
        next_site   = neidot[part_idx]
        idx1, sxn   = sigma_x(state, int(next_site))
        idx2, sxj   = sigma_x(idx1, this_site)
        coupling_v  = g0 * au[part_idx] * sxj * sxn
        # Store the results        
        local_value += vals
        states.append(idx2[0])
        values.append(coupling_v)
    return states + local_state, values + [local_value]

@numba.njit
def _local_energy_arr_np(state, neidot, h, g0, n, au):
    '''
    Compute the local energy interaction for a numpy array state.
    Parameters:
        state   : numpy array of spins (0 or 1)
        neidot  : numpy array of neighbor indices for free particles
        h, g0, n, au : model parameters
    Returns:
        Tuple [states], [energies] containing the new states and energies.
    '''
    return _local_energy_arr_loop(state, neidot, h, g0, n, au)

if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import jit as jax_jit
    
    @jax_jit
    def _local_energy_arr_lax(state     : jnp.ndarray,
                            neidot      : jnp.ndarray,
                            h           : jnp.ndarray,
                            g0          : float,
                            n           : int,
                            au          : jnp.ndarray,
                            sigma_x,
                            sigma_z) -> (jnp.ndarray, jnp.ndarray):
        """
        Compute the local energy interaction for a JAX array state.
        
        Parameters:
            state   : jnp.ndarray of spins (0 or 1), shape (S,)
            neidot  : jnp.ndarray of neighbor indices for free particles, shape (free_count,)
            h       : jnp.ndarray of magnetic field values, shape (free_count,)
            g0      : float, coupling strength
            n       : int, number of particles in the dot
            au      : jnp.ndarray of coupling strengths, shape (free_count,)
            sigma_x : function(state, site) -> (new_state, sign)
                    Operator helper to compute the σₓ operation.
            sigma_z : function(state, site) -> (state, eigenvalue)
                    Operator helper to compute the σ_z operation.
                    
        Returns:
            Tuple (all_states, all_values) where:
                all_states: jnp.ndarray of shape (free_count+1, S) containing the new states.
                all_values: jnp.ndarray of shape (free_count+1,) containing the corresponding energies.
        """
        
        # Get the size of the state - assumed to be the same for all states.
        size = state.shape[0]
        
        # Assume free indices: i from n to size-1.
        def scan_fun(local_value, i):
            part_idx            = i - n  # index into h and au arrays
            
            # Diagonal term: apply σ_z at site i.
            _, eigen_z          = sigma_z(state, i)
            vals                = h[part_idx] * eigen_z
            # Off-diagonal term: apply two successive σₓ operations.
            next_site           = neidot[part_idx]
            state_temp, v1      = sigma_x(state, next_site)
            state_x, v2         = sigma_x(state_temp, i)
            coupling_v          = g0 * au[part_idx] * v1 * v2
            # Accumulate the σ_z contribution.
            new_local_value     = local_value + vals
            # For each free site we output the new state (from σₓσₓ) and the coupling energy.
            return new_local_value, (state_x, coupling_v)
        
        free_indices            = jnp.arange(n, size, dtype=jnp.int32)
        init_local_value        = 0.0
        final_local_value, (free_states, free_values) = jax.lax.scan(scan_fun,
                                                                    init_local_value,
                                                                    free_indices)
        # Concatenate the free sites' states with the original state.
        all_states = jnp.concatenate([free_states, state[None, :]], axis=0)
        all_values = jnp.concatenate([free_values, jnp.array([final_local_value])], axis=0)
        return all_states, all_values
    
    @jax_jit
    def _local_energy_arr_jax(state, neidot, h, g0, n, au):
        '''
        Compute the local energy interaction for a JAX array state.
        Parameters:
            state   : numpy array of spins (0 or 1)
            neidot  : numpy array of neighbor indices for free particles
            h, g0, n, au : model parameters
        Returns:
            Tuple [states], [energies] containing the new states and energies.
        '''
        sigma_x = _operators_spin.sigma_x_jnp
        sigma_z = _operators_spin.sigma_z_jnp
        return _local_energy_arr_lax(state, neidot, h, g0, n, au, sigma_x, sigma_z)

def _local_energy_arr_wrap(n, neidot, h, g0, au, use_jax = False):
    '''Creates a JIT-compiled local energy interaction function.'''
    if use_jax and _JAX_AVAILABLE:
        @jax_jit
        def wrapper(state):
            return _local_energy_arr_jax(state, neidot, h, g0, n, au)
        return wrapper
        
    @numba.njit
    def wrapper(state):
        return _local_energy_arr_np(state, neidot, h, g0, n, au)
    return wrapper

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
        self._lattice                   = lattice
        super().__init__(hilbert_space, lattice=lattice, is_sparse=True, dtype=dtype, backend=backend, **kwargs)

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
        self._set_local_energy_functions()

        
        

        # test the Hamiltonian and allow jit to be built - trigger the jit compilation        

        # self._loc_energy_np_fun         = _local_energy_arr_wrap(self.ns, self._hx, self._hz, self._j, self._dlt,
        #                                                         self._kx, self._ky, self._kz, 
        #                                                         self._neibz, self._neiby, self._neibx, self._neiadd)
        # if _JAX_AVAILABLE:
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
    
    def _set_some_coupling(self, coupling):
        '''
        Distinghuishes between different initial values for the coupling and returns it.
        
        Parameters:
            - coupling : some coupling to be set
        Returns:
            array to be used latter with corresponding couplings
        '''
        if isinstance(coupling, list) and len(coupling) == self.ns:
            return self._backend.array(coupling)
        elif isinstance(coupling, (float, int)):
            return self._backend.array([coupling] * self.ns)
        elif isinstance(coupling, str):
            return random_vector(self.ns, coupling, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_COUP_VEC_SIZE)

    def _set_local_energy_functions(self):
        
        # operators
        operators       = [[] for _ in range(self.ns)]
        operators_local = [[] for _ in range(self.ns)]
        lattice         = self._lattice
        
        for i in range(self.ns):
            self._log(f"Starting i: {i}", lvl = 1, log = 'debug')
            
            op_sx_l     =   operators_spin_module.sig_x(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Local)
            op_sz_l     =   operators_spin_module.sig_z(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Local)
            
            # Kitaev and Heisenberg terms
            op_sx_sx_c  =   operators_spin_module.sig_x(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Correlation)
            op_sy_sy_c  =   operators_spin_module.sig_y(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Correlation)
            op_sz_sz_c  =   operators_spin_module.sig_z(lattice = self._lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Correlation)
            
            # now check the local operators 
            operators_local[i].append((op_sz_l, [i], self._hz[i]))
            operators[i].append((op_sx_l, [i], self._hx[i]))
            
            self._log(f"Adding local Sz at {i} with value {self._hz[i]:.2f}", lvl = 2, log = 'debug')
            self._log(f"Adding local Sx at {i} with value {self._hx[i]:.2f}", lvl = 2, log = 'debug')
            
            # now check the correlation operators
            nn_forward_num = lattice.get_nn_forward_num(i)
            # Kitaev - Z direction and Heisenberg - SzSz
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
                operators_local[i].append((op_sz_sz_c, [i, nei], sz_sz))
                self._log(f"Adding SzSz at {i},{nei} with value {sz_sz:.2f}", lvl = 2, log = 'debug')
                operators[i].append((op_sx_sx_c, [i, nei], sx_sx))
                self._log(f"Adding SySy at {i},{nei} with value {sy_sy:.2f}", lvl = 2, log = 'debug')
                operators[i].append((op_sy_sy_c, [i, nei], sy_sy))
                self._log(f"Adding SxSx at {i},{nei} with value {sx_sx:.2f}", lvl = 2, log = 'debug')
            # finished
            self._log(f"Finished i with len_local: {len(operators_local[i])}, len_normal: {len(operators[i])}", lvl = 1, log = 'debug')
            
        
        operators_int       = [[(op.int, sites, vals) for (op, sites, vals) in operators[i]] for i in range(self.ns)]
        operators_local_int = [[(op.int, sites, vals) for (op, sites, vals) in operators_local[i]] for i in range(self.ns)]
        self._loc_energy_int_fun    = hamil_module.local_energy_int_wrap(self.ns, operators_int, operators_local_int)
        self._loc_energy_jax_fun    = None
        self._loc_energy_np_fun     = None
        # if _JAX_AVAILABLE:
            # self._loc_energy_int_jax_fun = hamil_module.local_energy_int_wrap(ns, operators, operators_local, use_jax=True)
        
        self._log("Successfully set local energy functions...", log=2)

    # ----------------------------------------------------------------------------------------------


####################################################################################################