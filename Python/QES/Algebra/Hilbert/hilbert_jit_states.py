'''
file    : Algebra/Hilbert/hilbert_jit_states.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
'''


from Algebra.Hilbert.hilbert_jit_methods import *
import numpy as np
from numba import njit, prange
from numba.typed import List
from typing import Union, Optional, Callable, Tuple

#! jax
import Algebra.Hilbert.hilbert_jit_states_jax as jnp
from general_python.algebra.utilities import pfaffian as pfaffian
from general_python.algebra.utilities import hafnian as hafnian
from general_python.common.binary import int2binstr, check_int_l

# Signature: pfaff(A, n)    (antisymmetric, even n)
PfFunc              = Callable[[Array, int], Union[float, complex]]
# Signature: hafnian(A)     (symmetric,   even n)
HfFunc              = Callable[[Array], Union[float, complex]]
# Signature: callable for $\Psi$
CallableCoefficient = Callable[[Array, Array, Union[int, np.ndarray], int], Union[float, complex]]

_TOLERANCE          = 1e-10
_USE_EIGEN          = True
####################################################################################################
#! NUMBA METHODS
####################################################################################################

@njit(cache=True, inline='always')
def _popcount(x: int) -> int:
    """
    Returns the number of set bits (1-bits) in the binary representation of an integer.

    Args:
        x (int): The integer whose set bits are to be counted.

    Returns:
        int: The number of set bits in the binary representation of x.

    Example:
        >>> _popcount(7)
        3  # (7 in binary is 0b111)
    """
    cnt = 0
    while x:
        x   &= x - 1
        cnt += 1
    return cnt

@njit(cache=True)
def _extract_occupied(ns    : int,
                    basis   : Union[int, np.ndarray]) -> Array:
    """
    Extracts the indices of occupied sites from a basis state representation.

    Parameters
    ----------
    ns : int
        The number of sites (not used in the function body, but may be required for interface consistency).
    basis : Union[int, numpy.ndarray]
        The basis state, either as an integer (bit representation) or as a NumPy array.

    Returns
    -------
    numpy.ndarray
        An array of indices (dtype int64) indicating the positions of occupied sites.

    Notes
    -----
    - If `basis` is an integer, its binary representation is used to determine occupied sites (bits set to 1).
    - If `basis` is a NumPy array, nonzero entries are considered occupied.
    """

    if isinstance(basis, (int, np.integer)):
        x   = int(basis)
        occ = np.empty(_popcount(x), dtype=np.int64)
        k   = 0
        while x:
            lsb      = x & -x
            occ[k]   = lsb.bit_length() - 1
            k       += 1
            x       ^= lsb
        return occ
    else:
        return np.where(basis > 0)[0].astype(np.int64)

#############################################################################
#! Slater determinants
#############################################################################

@njit(cache=True, inline='always')
def _slater_from_mask(U, occ, mask, ns, use_eigen=_USE_EIGEN):
    n_fock  = _popcount(mask)
    N       = occ.shape[0]
    
    if n_fock != N:
        return 0.0
    # print(f"{mask}/{2**ns} : ", int2binstr(mask, ns))
    M       = np.empty((N, N), dtype=U.dtype)
    row     = 0
    for site in range(ns):
        if not check_int_l(mask, site, ns):
            continue
        for col in range(N):
            M[row, col] = U[occ[col], site]
        row += 1
        # if row == N:
        #     break
    #! eiggen
    if use_eigen:
        eigvals = np.linalg.eigvals(M)
        return np.prod(eigvals)
    #! determinant
    sign, logdet = np.linalg.slogdet(M)
    return 0.0 if sign == 0 else sign * np.exp(logdet)

@njit(cache=True, inline='always')
def _slater_from_vec(U, occ, vec, ns, use_eigen=_USE_EIGEN):
    n_fock  = vec.sum()
    N       = occ.shape[0]
    
    if n_fock != N:
        return 0.0
    
    M       = np.empty((N, N), dtype=U.dtype)
    row     = 0
    for site in range(ns):
        if vec[site] == 0:
            continue
        for col in range(N):
            M[row, col] = U[occ[col], site]
        row += 1
        if row == N:
            break
        
    #! eiggen
    if use_eigen:
        eigvals = np.linalg.eigvals(M)
        return np.prod(eigvals)
    #! determinant
    sign, logdet = np.linalg.slogdet(M)
    return 0.0 if sign == 0 else sign * np.exp(logdet)

@njit(cache=True, inline='always')
def calculate_slater_det(sp_eigvecs         : np.ndarray,
                        occupied_orbitals   : np.ndarray,
                        org_basis_state     : Union[int, np.ndarray],
                        ns                  : int,
                        use_eigen           : bool = False):
    r"""
    Compute the amplitude of an $N$-fermion Slater determinant that
    connects a Fock basis state expressed in *site* operators with a
    product state expressed in *eigen-orbital* operators.

    A single-particle Hamiltonian diagonalises as

    .. math::
        H \;=\;\sum_{n} E_{n}\, b_{n}^{\dagger} b_{n},

    where :math:`b_{n}^{\dagger}` creates a particle in eigen-orbital
    :math:`n`.  Eigen and site operators are related by the unitary matrix
    :math:`U` of eigenvectors,

    .. math::
        b_{i}^{\dagger} = \sum_{x=0}^{n_s-1} U_{x i}\, a_{x}^{\dagger},\qquad
        a_{x}^{\dagger} = \sum_{i}           U_{x i}\, b_{i}^{\dagger}.

    Occupying orbitals :math:`\{\alpha_k\}` yields the product state

    .. math::
        |\Psi_{\alpha}\rangle = \prod_{k=1}^{N} b_{\alpha_k}^{\dagger}\,|0\rangle.

    Its overlap with a site Fock configuration
    :math:`|x\rangle = a_{x_1}^{\dagger}\!\dots a_{x_N}^{\dagger}|0\rangle`
    is the determinant

    .. math::
        \Psi(x) \;=\; \frac{1}{\sqrt{N!}}\,
        \det\!\bigl[M_{jk}\bigr],\qquad
        M_{jk} \;=\;
        \langle 0|\,a_{x_j}\,b_{\alpha_k}^{\dagger}|0\rangle
        = U_{x_j\,\alpha_k}.

    This routine builds :math:`M` and returns :math:`\det M`.

    Parameters
    ----------
    sp_eigvecs : ndarray, shape (ns, n_orb)
        Unitary matrix :math:`U`; columns are eigenvectors.
    occupied_orbitals : ndarray, shape (N,)
        Integer indices :math:`\alpha_k` of the occupied orbitals.
        They are not in binary representation (mask) but rather
        give you the true indices of the occupied orbitals.

        Example:
        
        >>> occupied_orbitals = np.array([0, 1, 2]) # 3 particles
        >>> org_basis_state = np.array([1, 0, 0, 1, 1, 0, 0, 0]) # 8 sites
        >>> ns = 8
        >>> psi_x = calculate_slater_det(sp_eigvecs, occupied_orbitals, org_basis_state, ns)
    
    org_basis_state : int or ndarray
        Site-basis Fock state with ``ns`` modes; either an integer bitmask
        or a length-``ns`` 0/1 array.
        If it is an integer, it is treated as a bitmask.
        Example:
        >>> org_basis_state     = 0b11000000
        >>> org_basis_state     = np.array([1,1,0,0,0,0,0,0]) # 8 sites
        >>> ns                  = 8
    ns : int
        Number of lattice sites / modes, :math:`n_s`.

    Returns
    -------
    complex
        Determinant value.  Returns ``0.0`` if the particle number in
        ``org_basis_state`` differs from ``len(occupied_orbitals)``;
        returns ``1.0`` for the vacuum state.
    """

    if isinstance(org_basis_state, (int, np.integer)):
        return _slater_from_mask(sp_eigvecs, occupied_orbitals, org_basis_state, ns, use_eigen)
    elif isinstance(org_basis_state, np.ndarray):
        return _slater_from_vec(sp_eigvecs, occupied_orbitals, org_basis_state, ns, use_eigen)
    return 0.0

#############################################################################
#! Bogolubov - de'Gennes - no particle number conservation
#############################################################################

# ---------------------------------------------------------------------------
# a)  Pairing matrices  F = V·U^{-1}   (fermions)   /   G = V·U^{-1} (bosons)
# ---------------------------------------------------------------------------

@njit(cache=True)
def bogolubov_decompose(eig_val, eig_vec, tol = _TOLERANCE) -> Tuple[Array, Array, Array]:
    """
    Decomposes eigenvalues and eigenvectors into Bogoliubov (u, v) components.
    
    Given the eigenvalues and eigenvectors of a Bogoliubov-de Gennes (BdG) Hamiltonian,
    this function selects the positive eigenvalues (above a given tolerance), extracts
    the corresponding eigenvectors, and splits them into their (u, v) components.
    Each column of the resulting U and V matrices is normalized such that
    U†U + V†V = I.
    
    Args:
        eig_val (Array):
            Array of eigenvalues.
        eig_vec (Array):
            Array of eigenvectors, shape (2N, 2N).
        tol (float, optional):
            Tolerance for selecting positive eigenvalues. Defaults to _TOLERANCE.
    Returns:
        Tuple[Array, Array, Array]: Tuple containing:
            - U (Array): Matrix of u components, shape (N, N).
            - V (Array): Matrix of v components, shape (N, N).
            - eig_val_pos (Array): Array of positive eigenvalues, shape (N,).
    Raises:
        ValueError: If the number of positive eigenvalues does not match N (degeneracy or sign problem).
    """

    keep = eig_val > tol
    if keep.sum() != eig_val.shape[0] // 2:
        raise ValueError("degeneracy or sign problem; adjust tol")
    
    eig_val_pos = eig_val[keep]         # shape (N,)
    eig_vec_pos = eig_vec[:, keep]      # shape (2N, N)
    N           = eig_val_pos.shape[0]

    # split into (u,v) components
    U = eig_vec_pos[:N, :].copy()
    V = eig_vec_pos[N:, :].copy()

    # column-wise normalisation (guarantees U^\dagU + V^\dag V = I)
    for k in range(N):
        s = np.sqrt(np.linalg.norm(U[:, k])**2 + np.linalg.norm(V[:, k])**2)
        U[:, k] /= s
        V[:, k] /= s

    return U, V, eig_val_pos

@njit(cache=True)
def pairing_matrix(u_mat: Array, v_mat: Array) -> Array:
    """
    Solve V = F·U  →  F = V·U^{-1}
    without forming U^{-1} explicitly.
    
    Valid for both fermionic (antisymmetric F) and bosonic (symmetric G)
    Bogoliubov transformations.
    
    Params:
    
    """
    # Equivalent to v_mat @ np.linalg.inv(u_mat)
    return np.linalg.solve(u_mat.T, v_mat.T).T

# ---------------------------------------------------------------------------
# b)  Fermions :  Ψ(x) = Pf[ F_{ij} ] (x has an *even* length)
# ---------------------------------------------------------------------------

@njit(cache=True)
def calculate_bogoliubov_amp(F          : Array, # pairing  (ns, ns)
                            basis       : Union[int, np.ndarray],
                            ns          : int,
                            pfaff       : PfFunc,
                            enforce     = True):
    """
    Calculate the amplitude of a Bogoliubov vacuum state for a given basis configuration.

    This function computes the amplitude of a Bogoliubov quasiparticle vacuum state
    for a specified occupation basis, using the provided pairing matrix `F` and a
    Pfaffian evaluation function. The amplitude is zero if the number of occupied
    sites is odd, and one if there are no occupied sites.

    Parameters
    ----------
    F : Array
        Pairing matrix of shape (ns, ns) representing the Bogoliubov transformation.
    basis : Union[int, np.ndarray]
        Basis configuration specifying the occupied sites. Can be an integer or an array.
    ns : int
        Number of single-particle states.
    pfaff : PfFunc
        Function to compute the Pfaffian of a skew-symmetric matrix.
    enforce : bool, optional
        If True (default), enforces the submatrix to be skew-symmetric by symmetrization.

    Returns
    -------
    float
        The amplitude of the Bogoliubov vacuum state for the given basis configuration.
    """
    occ     = _extract_occupied(ns, basis)
    m       = occ.size
    if m == 0:
        return 1.0
    if m & 1:
        return 0.0

    # build the submatrix F_{ij} for the occupied sites
    # for the Bogoliubov transformation vacuum state
    # F_{ij} = F_{i_p j_q} where p,q are the occupied sites
    sub = np.empty((m, m), dtype=F.dtype)
    for p in range(m):
        ip = occ[p]
        for q in range(m):
            sub[p, q] = F[ip, occ[q]]
    if enforce:
        sub[:] = 0.5*(sub - sub.T)
    return pfaff(sub, m)

@njit(cache=True)
def calculate_bogoliubov_amp_exc(F      : Array,     # (ns, ns)
                                U       : Array,     # (ns, Nqp)
                                qp_inds : Array,     # [m1,…,mk]
                                basis   : Union[int,Array],
                                ns      : int,
                                pfaff   : PfFunc) -> complex:
    """
    Calculate the Bogoliubov amplitude for an excited state.

    This function constructs a skew-symmetric matrix based on the provided Fock matrix,
    Bogoliubov transformation matrix, and indices of quasiparticle excitations, and then
    computes its Pfaffian to obtain the amplitude.

    Args:
        F (Array):
            The Fock matrix of shape (ns, ns).
        U (Array):
            The Bogoliubov transformation matrix of shape (ns, Nqp).
        qp_inds (Array):
            Array of indices [m1, ..., mk] specifying the excited quasiparticle states.
        basis (Union[int, np.ndarray]):
            Basis state, either as an integer or an array of occupation numbers.
        ns (int):
            Number of single-particle states.
        pfaff (PfFunc):
            Function to compute the Pfaffian of a skew-symmetric matrix.

    Returns:
        complex: The Bogoliubov amplitude for the specified excitation.

    Notes:
        - If the total number of occupied states and excitations is odd, the amplitude is zero.
        - If there are no occupied states and no excitations, the amplitude is one.
    """
    occ     = _extract_occupied(ns, basis)
    n       = occ.size
    k       = qp_inds.size
    if (n + k) & 1:
        return 0.0
    if n == k == 0:
        return 1.0

    dim = n + k
    M   = np.empty((dim, dim), dtype=F.dtype)

    # F block
    for p in range(n):
        ip = occ[p]
        for q in range(n):
            M[p, q] = F[ip, occ[q]]

    # cross blocks (±U)
    for p in range(n):
        ip = occ[p]
        for j in range(k):
            m = qp_inds[j]
            M[p, n+j]   =  U[ip, m]
            M[n+j, p]   = -U[ip, m]
    # lower-right k \times k = 0
    return pfaff(M, dim)

# ---------------------------------------------------------------------------
# c)  Bosons :  Ψ(x) = Hf[ G_{ij} ] (x has an *even* length)
# ---------------------------------------------------------------------------

@njit(cache=True)
def calculate_bosonic_gaussian_amp(G        : Array,    # (ns, ns)
                                basis       : Union[int,Array],
                                ns          : int,
                                haf         : HfFunc    = None):
    """
    Calculate the amplitude of a bosonic Gaussian state in a given Fock basis.

    This function computes the amplitude of a bosonic Gaussian state, represented by the matrix `G`,
    for a specified Fock basis state. The calculation uses the Hafnian function for the relevant
    submatrix of `G` corresponding to the occupied modes.

    Args:
        G (Array):
            The (ns, ns) complex symmetric matrix representing the Gaussian state.
        basis (Union[int, np.ndarray]):
            The Fock basis state, either as an integer or an array of occupations.
        ns (int):
            The number of modes.
        haf (HfFunc, optional):
            Function to compute the Hafnian of a matrix. If None, a default Hafnian function is used.

    Returns:
        complex: 
            The amplitude of the bosonic Gaussian state in the specified Fock basis.

    Notes:
        - If the number of occupied modes is zero, returns 1.0.
        - If the number of occupied modes is odd, returns 0.0 (since the amplitude vanishes).
        - Otherwise, computes the Hafnian of the submatrix of `G` corresponding to the occupied modes.
    """
    occ = _extract_occupied(ns, basis)
    m   = occ.size
    if m == 0:
        return 1.0
    if m & 1:                              # odd # bosons ⇒ 0
        return 0.0

    sub = np.empty((m, m), dtype=G.dtype)
    for p in range(m):
        ip = occ[p]
        for q in range(m):
            sub[p, q] = G[ip, occ[q]]
    return haf(sub)

#############################################################################
#! Permanents
#############################################################################

@njit(cache=True)
def _calculate_permanent_core_np(M: np.ndarray):
    """ 
    Core Ryser's formula calculation (Numba). 
    """
    n = M.shape[0]
    if n == 0:
        return 1.0 + 0.0j
    
    total_sum           = 0.0 + 0.0j
    for k in range(1, 1 << n):
        sum_prod        = 1
        popcount_k      = 0
        # Calculate product over rows of sum over columns in S
        for i in range(n): # Rows
            row_sum     = 0.0 + 0.0j
            temp_k_cols = k
            col_idx     = 0
            while temp_k_cols > 0:
                if temp_k_cols & 1: # Check if col_idx is in subset S
                    row_sum += M[i, col_idx]
                temp_k_cols >>= 1
                col_idx += 1
            sum_prod *= row_sum

        # Calculate popcount |S|
        temp_k_pop = k
        while temp_k_pop > 0:
            temp_k_pop &= (temp_k_pop - 1)
            popcount_k += 1

        # Sign (-1)^(n - |S|)
        sign        =   -1.0 if (n - popcount_k) % 2 else 1.0
        total_sum   +=  sign * sum_prod
    return total_sum

@njit(cache=True)
def calculate_permanent(sp_eigvecs          : np.ndarray,   # U matrix (Ns x Norb)
                        occupied_orbitals   : np.ndarray,   # Indices {α_k}, shape (N,)
                        org_basis_state     : Union[int, np.ndarray],
                        ns                  : int
                        ) -> np.complex128:
    """
    Calculates the permanent using NumPy/Numba (via Ryser's formula).

    Represents the amplitude <Fock(R)|State_B(O)> for BOSONS.

    Args:
        sp_eigvecs (np.ndarray):
            Eigenvector matrix U (shape: Ns x Norb).
        occupied_orbitals (np.ndarray):
            1D array (length N) of occupied orbital indices {α_k}.
        org_basis_state (Union[int, np.ndarray]):
            Represents the Fock state R (int bitmask or 0/1 array).
        ns (int):
            Total number of sites/modes.

    Returns:
        np.complex128: The value of the permanent perm(M).
    """
    # --- Input processing and particle number check (Identical to Slater) ---
    n_particles     = occupied_orbitals.shape[0]
    if isinstance(org_basis_state, (int, np.integer)):
        occupied_modes      = np.empty(n_particles, dtype=np.int64)
        n_particles_fock    = 0
        idx_count           = 0
        temp_basis          = int(org_basis_state)
        
        for i in range(ns):
            if (temp_basis >> i) & 1:
                n_particles_fock += 1
                if idx_count < n_particles:
                    occupied_modes[idx_count] = i
                idx_count += 1
    elif isinstance(org_basis_state, np.ndarray):
        if org_basis_state.ndim == 1 and org_basis_state.size == ns:
            basis_state         = org_basis_state.astype(np.bool_) if org_basis_state.dtype != np.bool_ else org_basis_state
            n_particles_fock    = np.sum(basis_state)
            if n_particles_fock == n_particles:
                occupied_modes = np.nonzero(basis_state)[0]
        else: 
            n_particles_fock = -1
    else:
        n_particles_fock = -1

    if n_particles_fock != n_particles:
        return 0.0
    if n_particles == 0:
        return 1.0

    #! Construct the Matrix M (Identical to Slater)
    M = np.empty((n_particles, n_particles), dtype=sp_eigvecs.dtype)
    for j in range(n_particles):
        site_j = occupied_modes[j]
        for k in range(n_particles):
            orbital_k   = occupied_orbitals[k]
            M[j, k]     = sp_eigvecs[site_j, orbital_k]

    #! Calculate Permanent using Ryser's formula
    perm_val = _calculate_permanent_core_np(M)
    return perm_val

#############################################################################
#! Many body state through summation
#############################################################################

# @njit(cache=True, parallel=True)
def _fill_batched_space(   matrix_arg               : np.ndarray,            
                            calculator_func         : CallableCoefficient,   
                            target_basis_states     : Array,
                            target_basis_states_idx : Array,
                            result_vector_slice     : Array,
                            ns                      : int,                  
                        ):
    """
    Numba-jitted loop to compute amplitudes for many-body state construction.

    Args:
        matrix_arg (np.ndarray):
            Matrix needed by calculator (e.g., sp_eigvecs U, f_mat).
        occupied_orbitals (np.ndarray):
            Indices needed by calculator (e.g., orbitals α_k, sites i_p).
            For the particle occupation representation
        calculator_func (Callable):
            Numba-jitted function (e.g., calculate_slater_det_np).
            Must have signature:
                func(matrix, orbitals, basis_state_int/numpy, ns) : CallableCoefficient
        target_basis_states (np.ndarray):
            1D array of integer Fock states to compute amplitudes for.
        result_vector (np.ndarray):
            modified in-place with results.
        ns (int):
            Number of sites.
    """
    # how many states
    n_target_states = target_basis_states.shape[0]

    if target_basis_states_idx and len(target_basis_states_idx) == len(target_basis_states):
        for i in prange(n_target_states):
            basis_state_int                             = target_basis_states[i]
            basis_state_int_idx                         = target_basis_states_idx[i]
            # Call the appropriate calculator function
            amplitude                                   = calculator_func(matrix_arg, occupied_orbitals, basis_state_int, ns)
            result_vector_slice[basis_state_int_idx]    = amplitude
    else:
        for i in prange(n_target_states):
            basis_state_int                             = target_basis_states[i]
            basis_state_int_idx                         = i
            # Call the appropriate calculator function
            amplitude                                   = calculator_func(matrix_arg, occupied_orbitals, basis_state_int, ns)
            result_vector_slice[basis_state_int_idx]    = amplitude
        
def many_body_state_mapping(matrix_arg          : Array,
                            calculator_func     : CallableCoefficient,
                            mapping_array       : Array,
                            ns                  : int,
                            dtype               = np.complex128) -> Array:
    """
    Given ``mapping_array[j] = bitstring of Hilbert basis state j``,
    return the coefficient vector ``ψ_j`` **in the same order** as
    ``mapping_array``.

    Suitable when a high-level Hilbert-space class already stores an
    enumeration different from lexicographic bit order.
    """
    n_states     = mapping_array.shape[0]
    result_vec   = np.empty(n_states, dtype=dtype)

    # indices are 0…n_states-1, so we can reuse many_body_state_loop
    _fill_batched_space(matrix_arg,
                        calculator_func,
                        mapping_array, # target_states
                        None,
                        result_vec, # slice = whole output
                        ns)
    return result_vec

# ###########################################################################
#! Full Hilbert–space version (loops over all integers)
# ###########################################################################

# @njit(cache=True, fastmath=True)
def _fill_full_space(matrix_arg : Array,
                    calculator  : Callable[[Array, int, int], complex],
                    ns          : int,
                    result      : Array) -> None:
    """
    Fills the `result` array with values computed by the provided `calculator` function for each state in the Hilbert space.

    Args:
        matrix_arg (Array): Input array or matrix to be used by the calculator function.
        calculator (Callable[[Array, int, int], complex]): Function that computes the value for each state, given `matrix_arg`, the state index, and `ns`.
        ns (int): Number of states or a parameter required by the calculator function.
        result (Array): Output array to be filled with computed values. Must be pre-allocated with the correct size.

    Returns:
        None: The function modifies the `result` array in place.
    """
    nh = result.size
    for st in range(nh):
        st          = np.int64(st)
        result[st]  = calculator(matrix_arg, st, ns)

def many_body_state_full(matrix_arg     : Array,
                        calculator      : Callable[[Array, int, int], complex],
                        ns              : int,
                        resulting_s     : Optional[Array] = None,
                        dtype           = np.complex128) -> Array:
    """
    Generates the full many-body quantum state vector for a system with `ns` sites.

    Parameters
    ----------
    matrix_arg : Array
        Input matrix or array containing parameters or data required by the calculator function.
    calculator : Callable[[Array, int, int], complex]
        A function that computes the amplitude for a given basis state, taking the matrix_arg,
        the basis index, and the number of sites as arguments.
    resulting_s : Optional[Array], default=None
        If provided, this array will be filled with the resulting state vector.
        If None, a new array will be created.
        The size of the resulting_s array must match the expected size (2**ns).
        If the size does not match, a ValueError will be raised.
    ns : int
        Number of sites (qubits, spins, etc.) in the system.
    dtype : data-type, optional
        Desired data-type for the output array. Default is np.complex128.

    Returns
    -------
    Array
        The full many-body state vector as a 1D array of length 2**ns, with each entry representing
        the amplitude of a basis state in the computational basis.
    """
    nh              = 1 << ns
    if resulting_s is not None:
        if resulting_s.size != nh:
            raise ValueError(f"resulting_s must be of size {nh}, but got {resulting_s.size}")
        out         = resulting_s.astype(dtype, copy=False)
    else:
        out         = np.empty(nh, dtype=dtype)
    
    _fill_full_space(matrix_arg, calculator, ns, out)
    return out

def many_body_state_closure(calculator_func: Callable[[Array, int, int], Union[float, complex]],
                            matrix_arg: Optional[Array] = None) -> Callable[[int], Union[float, complex]]:
    """
    Creates a closure function that computes the amplitude for a given basis state.

    Args:
        matrix_arg (Array):
            Input array or matrix to be used by the calculator function.
        calculator_func (Callable[[Array, int, int], Union[float, complex]]):
            Function that computes the value for each state, given `matrix_arg`, the state index, and `ns`.
    Returns:
        Callable[[int], Union[float, complex]]:
            A function that takes a single integer argument (the state index) and returns the computed amplitude.
    """
    
    if matrix_arg is not None:
        const = matrix_arg
        # @njit(inline='always')
        def closure(U: Array, state: int, ns: int):
            return calculator_func(U, const, state, ns)
        return closure

    # no extra constant
    @njit(inline='always', cache=True)
    def closure(U: Array, state: int, ns: int):
        return calculator_func(U, state, ns)
    return closure

############################################################################

@numba.njit(cache=True, inline='always')
def nrg_particle_conserving(eigvals: np.ndarray, occ: np.ndarray) -> float:
    """
    Calculates the total energy for a particle-conserving system given eigenvalues and occupied states.

    Args:
        eigvals (np.ndarray):
            Array of eigenvalues corresponding to single-particle states.
        occ (np.ndarray):
            Array of indices indicating which states are occupied.

    Returns:
        float: The total energy of the occupied states.
    """
    tot = 0.0
    for k in occ:
        tot += eigvals[k]
    return tot

@numba.njit(cache=True, inline='always')
def nrg_bdg(eigvals: np.ndarray, Ns: int, occ: np.ndarray) -> float:
    """
    Calculates the total energy contribution from selected Bogoliubov-de Gennes (BdG) eigenvalues.

    Given an array of eigenvalues, the number of single-particle states, and an array of occupation indices,
    this function sums the energy differences between positive and negative energy branches for the specified
    occupied states.

    Args:
        eigvals (np.ndarray):
            Array of BdG eigenvalues, ordered such that negative and positive branches are contiguous.
        Ns (int):
            Number of single-particle states (half the total number of BdG states).
        occ (np.ndarray):
            Array of occupation indices (integers from 0 to Ns-1) specifying which states are occupied.

    Returns:
        float: The total energy contribution from the specified occupied states.
    """
    tot     = 0.0
    mid     = Ns - 1           # last negative index
    for i in occ:              # i = 0 … Ns-1   (positive branch)
        tot += eigvals[mid + i + 1] - eigvals[mid - i]
    return tot

############################################################################
#! Test function
############################################################################

def test_many_body_states(seed: int = 12345) -> None:
    """
    Generates tiny (ns=4) random unitaries / BdG matrices and verifies that
    our accelerated amplitudes match slow NumPy references.
    
    The test includes:
    - Slater determinants
    - Bogoliubov vacuum states
    - Bogoliubov excited states
    - Bosonic Gaussian states
    """
    np.random.seed(seed)
    ns      = 4
    nh      = 1 << ns
    # --- random unitary -------------------------------------------------
    X       = np.random.randn(ns, ns) + 1j*np.random.randn(ns, ns)
    U, _    = np.linalg.qr(X)
    occ_orb = np.array([0, 1]) # 2 particles, |1, 1, 0, 0> state
    
    # use this to test the full state
    psi_sd  = np.empty(nh, dtype=complex)
    
    # 1) full state (Slater)
    psi_sd  = many_body_state_full(U,
                lambda Um, st, ns_: calculate_slater_det(Um, occ_orb, st, ns_), ns)
    # 2) brute reference
    psi_ref = np.empty(nh, dtype=complex)
    for st in range(nh):
        sites = _extract_occupied(ns, st)
        if sites.size != occ_orb.size:
            psi_ref[st] = 0.0
        else:
            M = U[sites][:, occ_orb]
            psi_ref[st] = np.linalg.det(M)
    assert np.allclose(psi_sd, psi_ref)

    # 2) fermionic BdG vacuum
    # craft simple pairing: Δ (nearest-neighbour) on ring
    K           = np.diag(np.ones(ns-1), 1)+np.diag(np.ones(ns-1), -1)
    Δ           = 0.3*(np.roll(np.eye(ns), 1, 1)-np.roll(np.eye(ns), -1, 1))
    Hbdg        = np.block([[ K,  Δ],
                    [-Δ, -K]])
    E, W        = np.linalg.eigh(Hbdg)
    Ub, Vb, Ep  = bogolubov_decompose(E, W)
    F           = pairing_matrix(Ub, Vb)
    pf          = lambda A,n: pfaffian.Pfaffian._pfaffian_parlett_reid(A, n)
    psi_bdg     = many_body_state_full(F,
                    lambda Fm, st, ns_: calculate_bogoliubov_amp(Fm, st, ns_, pf),
                ns)
    
    # vacuum check: amplitude for empty state must be 1
    assert np.isclose(psi_bdg[0], 1.0)

    # 3) add one qp (index 0)
    psi_exc = many_body_state_full((F, Ub),
                lambda arg, st, ns_: calculate_bogoliubov_amp_exc(
                        arg[0], arg[1], np.array([0]), st, ns_, pf),
                        ns)
    
    # norm ratio equals ⟨γ₀γ₀†⟩ = 1
    assert np.isclose(np.sum(np.abs(psi_exc)**2) /
                    np.sum(np.abs(psi_bdg)**2), 1.0)

    # 4) bosonic Gaussian vacuum 
    # fake bosonic U,V by copying fermionic |Ub|,|Vb|
    G = pairing_matrix(np.abs(Ub), np.abs(Vb))
    psi_bos = many_body_state_full(G,
                lambda Gm, st, ns_: calculate_bosonic_gaussian_amp(Gm, st, ns_),
                ns)
    # empty state amplitude again 1
    assert np.isclose(psi_bos[0], 1.0)
    print("✓ all Gaussian-state tests passed")

if __name__ == "__main__":
    test_many_body_states()
