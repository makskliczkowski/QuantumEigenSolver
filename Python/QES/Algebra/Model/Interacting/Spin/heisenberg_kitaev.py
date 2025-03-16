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
#! INTEGER STATES
# ----------------------------------------------------------------------------------------

@numba.njit
def _apply_heisenberg(org_idx, new_idx, nei, sx_i, sy_i, sz_i, this_site, ns):
    if nei < 0:
        return 0,0.0,0.0,0.0
    
    # nei             = np.array(, dtype=DEFAULT_NP_INT_TYPE)
    # apply sig_z
    _, sz_j         = operators_spin_module.sigma_z_int_np(org_idx, ns, [nei])
    # apply sig_y
    sy_j            = sx_i * 1.0j if (sz_j[0] > 0) else -1.0j * sx_i
    # apply sig_x
    new_new_idx, sx_j = operators_spin_module.sigma_x_int_np(new_idx, ns, this_site)
    # return new_state, s^z_i * s^z_j, s^y_i * s^y_j, s^x_i * s^x_j
    return new_new_idx, sz_i * sz_j, np.real(sy_i * sy_j), sx_i * sx_j

@numba.njit
def _local_energy_int(k_map         : int,
                        i           : int,
                        ns          : int,
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
                        neiadd      : np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the local energy contribution from a given set of parameters.

    Parameters
    ----------
    k_map : int
        The integer representation of a basis state.
    i : int
        The bit position to flip for the off-diagonal element.
    Returns
    -------
    tuple
        A tuple containing two arrays:
        - new_rows: An array of row indices (int) representing the positions in the matrix.
        - new_vals: An array of values representing the energy contributions at those positions.
        The first element of each array corresponds to the diagonal element (k_map),
        while the second corresponds to the off-diagonal element (k_map with bit i flipped).
    """

    # Pre-allocate arrays
    # - 1 for S_z and S_zS_z
    # - 1 for S_x
    # - len(neib_z) + len(neib_y) + len(neib_x) + len(addit_n) for S_xS_x and S_yS_y
    # num_states_m= 2 + neib_x.shape[0] + neib_y.shape[0] + addit_n.shape[0]
    nn_x_in     = neib_x[i]
    nn_y_in     = neib_y[i]
    nn_z_in     = neib_z[i]
    nn_a_in     = neiadd[i]
    
    num_states  = 2 + nn_x_in.shape[0] + nn_y_in.shape[0] + nn_z_in.shape[0] + nn_a_in.shape[0]
    new_rows    = np.empty(num_states, dtype=np.int64)
    new_vals    = np.empty(num_states, dtype=DEFAULT_NP_FLOAT_TYPE)
    
    # store here the rows, columns, and values
    this_site   = np.array([i], dtype=DEFAULT_NP_INT_TYPE)
    
    # apply the sigma_z operator (and use this for later as well)
    _, sz_i     = operators_spin_module.sigma_z_int_np(k_map, ns, this_site)
    h_z_val     = hz[i] * sz_i
    
    # start with setting hz
    new_rows[0] = k_map
    new_rows[0] = h_z_val
    
    # apply the sigma_x operator (and use this for later as well)
    new_idx, sx_i       = operators_spin_module.sigma_x_int_np(k_map, ns, this_site)
    h_x_val             = sx_i * hx[i]
    new_rows[1]         = new_idx
    new_vals[1]         = h_x_val
    
    # get the value for sy_i
    sy_i        = sx_i * 1.0j if (sz_i[0] > 0) else -1.0j * sx_i
    
    # define heisenberg part function
    iter_in             = 2
    # go through the neighbors corresponding to sig_z - sig_z : Kitaev - Z direction
    for ii, nei in enumerate(nn_z_in):
        # if nei < 0:
            # continue
        
        new_new_idx, sz_val, sy_val, sx_val = _apply_heisenberg(k_map, new_idx, nei,
                                                            sx_i, sy_i, sz_i, this_site, ns)
        
        # add-up the values to the local contribution - add Kitaev - Z direction
        new_vals[0]             += (j_cpl[i] * delta_cpl[i] + kz_cpl[i]) * sz_val
        new_rows[iter_in + ii]  = new_new_idx
        new_vals[iter_in + ii]  += j_cpl[i] * (sx_val + sy_val)
    
    iter_in += nn_z_in.shape[0]
    
    # # go through the neighbors corresponding to sig_y - sig_y : Kitaev - Y direction
    for ii, nei in enumerate(nn_y_in):
        # if nei < 0:
            # continue

        new_new_idx, sz_val, sy_val, sx_val = _apply_heisenberg(k_map, new_idx, nei,
                                                            sx_i, sy_i, sz_i, this_site, ns)
        
        # add-up the values to the local contribution - add Kitaev - Y direction
        new_vals[0]             += j_cpl[i] * delta_cpl[i] * sz_val
        new_rows[iter_in + ii]  = new_new_idx
        new_vals[iter_in + ii]  += j_cpl[i] * sx_val + (j_cpl[i] + ky_cpl[i]) * sy_val

    iter_in += nn_y_in.shape[0]
    
    # go through the neighbors corresponding to sig_x - sig_x : Kitaev - X direction
    for ii, nei in enumerate(nn_x_in):
        # if nei < 0:
            # continue
        
        new_new_idx, sz_val, sy_val, sx_val = _apply_heisenberg(k_map, new_idx, nei,
                                                            sx_i, sy_i, sz_i, this_site, ns)
        
        # add-up the values to the local contribution - add Kitaev - X direction
        new_vals[0]             += j_cpl[i] * delta_cpl[i] * sz_val
        new_rows[iter_in + ii]  = new_new_idx
        new_vals[iter_in + ii]  += (j_cpl[i] + kx_cpl[i]) * sx_val + j_cpl[i] * sy_val
    
    iter_in += nn_z_in.shape[0]
    
    # apply any additional Heisenberg neighbors
    for ii, nei in enumerate(nn_a_in):
        # if nei < 0:
            # continue
        
        new_new_idx, sz_val, sy_val, sx_val = _apply_heisenberg(k_map, new_idx, nei,
                                                            sx_i, sy_i, sz_i, this_site, ns)
        
        # add-up the values to the local contribution - add Kitaev - X direction
        new_vals[0]             += j_cpl[i] * delta_cpl[i] * sz_val
        new_rows[iter_in + ii]  = new_new_idx
        new_vals[iter_in + ii]  += j_cpl[i] * (sx_val + sy_val)
    return new_rows, new_vals

def _local_energy_int_wrap(ns       : int,
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
    '''Creates a JIT-compiled local energy interaction function.'''
    hx          = np.array(hx, dtype=DEFAULT_NP_FLOAT_TYPE)
    hz          = np.array(hz, dtype=DEFAULT_NP_FLOAT_TYPE)
    j_cpl       = np.array(j_cpl, dtype=DEFAULT_NP_FLOAT_TYPE)
    delta_cpl   = np.array(delta_cpl, dtype=DEFAULT_NP_FLOAT_TYPE)
    kx_cpl      = np.array(kx_cpl, dtype=DEFAULT_NP_FLOAT_TYPE)
    ky_cpl      = np.array(ky_cpl, dtype=DEFAULT_NP_FLOAT_TYPE)
    kz_cpl      = np.array(kz_cpl, dtype=DEFAULT_NP_FLOAT_TYPE)
    neib_z      = np.array(neib_z, dtype = DEFAULT_NP_INT_TYPE)
    neib_y      = np.array(neib_y, dtype = DEFAULT_NP_INT_TYPE)
    neib_x      = np.array(neib_x, dtype = DEFAULT_NP_INT_TYPE)
    addit_n     = np.array(addit_n, dtype = DEFAULT_NP_INT_TYPE)
    @numba.njit
    def wrapper(k, i):
        return _local_energy_int(k, i, ns,
                    hx, hz, j_cpl, delta_cpl, kx_cpl, ky_cpl, kz_cpl,
                    neib_z, neib_y, neib_x, neiadd=addit_n)
    return wrapper

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
            if self._ns is None:
                raise ValueError(self._ERR_EITHER_HIL_OR_NS)
            hilbert_space = hilbert_module.HilbertSpace(ns=self._ns, backend=backend, dtype=dtype, nhl=2)
        
        # setup the fields
        self._hx                        = hx
        self._hz                        = hz
        self._kx                        = kx
        self._ky                        = ky
        self._kz                        = kz
        self._j                         = j
        self._dlt                       = dlt
        self.set_couplings()
        
        # set the vectors for the neighbors at a given site index in a given direction
        
        if self._lattice.typek not in [LatticeType.HONEYCOMB, LatticeType.HEXAGONAL]:
            self._log(f"The type of the lattice {self._lattice} is not standard. Check your intentions...", lvl = 2)
        self._neibz                     = [[]]
        self._neiby                     = [[]]
        self._neibx                     = [[]]
        self._neiadd                    = [[]]
        self._set_neighbors()
        
        # use the lattice to obtain all the neighbors to use them later in the energy calculation
        
        # Initialize the Hamiltonian
        self._name                      = "Heisenberg Kitaev Model"
        self._is_sparse                 = True
        self._max_local_ch              = 6
        # test the Hamiltonian and allow jit to be built - trigger the jit compilation        
        self._loc_energy_int_fun        = _local_energy_int_wrap(self._ns, self._hx, self._hz, self._j, self._dlt,
                                                                self._kx, self._ky, self._kz, 
                                                                self._neibz, self._neiby, self._neibx, self._neiadd)
        # self._loc_energy_np_fun         = _local_energy_arr_wrap(self._ns, self._hx, self._hz, self._j, self._dlt,
        #                                                         self._kx, self._ky, self._kz, 
        #                                                         self._neibz, self._neiby, self._neibx, self._neiadd)
        # if _JAX_AVAILABLE:
            # self._loc_energy_jax_fun    = _local_energy_arr_wrap(self._n, self._neidot, self._h, self._g0, self._au, use_jax=True)        
        # self._local_energy_test()
    
    # ----------------------------------------------------------------------------------------------
    
    def __repr__(self):
        ''' Return the string representation of the QSM model. '''
        return ''
    
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
        if isinstance(coupling, list) and len(coupling) == self._ns:
            return self._backend.array(coupling)
        elif isinstance(coupling, (float, int)):
            return self._backend.array([coupling] * self._ns)
        elif isinstance(coupling, str):
            return random_vector(self._ns, coupling, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_COUP_VEC_SIZE)

    def _set_neighbors(self):
        '''
        Presets the neighbors based on the lattice - to be used in the energy calculation
        '''
        
        self._neibz                     = []
        self._neiby                     = []
        self._neibx                     = []
        self._neiadd                    = []
        # go through sites
        for i in range(self._ns):
            
            # check number on nearest neighbors
            nn_forward_num              = self._lattice.get_nn_forward_num(i)
            nn_z_in                     = []
            nn_y_in                     = []
            nn_x_in                     = []
            nn_add_in                   = []
            # go through them
            for nn in range(nn_forward_num):
                
                # check if ok
                nei = self._lattice.get_nn_forward(i, num = nn)
                # if self._ns > nei >= 0:
                if nn == HEI_KIT_Z_BOND_NEI:
                    nn_z_in.append(nei)
                elif nn == HEI_KIT_Y_BOND_NEI:
                    nn_y_in.append(nei)
                elif nn == HEI_KIT_X_BOND_NEI:
                    nn_x_in.append(nei)
                else:
                    nn_add_in.append(nei)
            
            self._neibz.append(nn_z_in)
            self._neiby.append(nn_y_in)
            self._neibx.append(nn_x_in)
            self._neiadd.append(nn_add_in if len(nn_add_in) > 0 else [-1])
        self._log(f"Succesfully set the neighbors for the couplings...", log = 2)
    
    # ----------------------------------------------------------------------------------------------


####################################################################################################