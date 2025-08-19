import os, sys
import itertools
import numpy as np
from math import comb as nCk
from enum import Enum
from typing import Optional, List, Union, Tuple, Sequence, Iterable, Callable, Literal

########################################################################
#! RESOLVE PATH
########################################################################

from pathlib import Path
cwd         = Path.cwd()
mod_path    = Path(__file__).resolve()
qes_path    = Path(__file__).parent.parent.parent.parent
lib_path    = qes_path / 'QES'
print("Current working directory:", cwd)
print("Module path:", mod_path)
print("QES path:", qes_path)
print("Library path:", lib_path, "\n\tWith folders:", os.listdir(lib_path))
sys.path.insert(0, str(lib_path))

#! -----
os.environ['BACKEND'] = 'numpy'
#! -----

# ---------------------------------------------------------------------
import numpy as np

#! General Python imports
try:
    from QES.general_python.common import get_global_logger
    from QES.general_python.physics.sp import correlation_matrix as Corr
    from QES.general_python.physics import entropy as Entropy, density_matrix as Dens
except ImportError:
    raise ImportError("QES.general_python module is not available. Please ensure QES is installed correctly.")

#! QES imports
try:
    from QES.Algebra.Model.Noninteracting import syk
    from QES.Algebra.Model.Noninteracting.Conserving import free_fermions, aubry_andre
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
except ImportError:
    raise ImportError("QES modules are not available. Please ensure QES is installed correctly.")

#! Global logger
logger = get_global_logger()

# ---------------------------------------------------------------------

def to_one_hot(positions    : Iterable[int],
                size        : int,
                *,
                asbool      : bool = True) -> np.ndarray:
    '''
    Transform a list of positions into a one-hot encoded array.

    Parameters
    ----------
    positions : Iterable[int]
        The positions to encode.
    size : int
        The size of the output array.
    asbool : bool, optional
        If True, return a boolean array. Default is True.

    Returns
    -------
    np.ndarray
        The one-hot encoded array.
    '''
    y               = np.zeros(shape=size, dtype = np.int32)
    y[positions]    = 1
    if asbool:
        return y.astype(bool)
    return y

# ---------------------------------------------------------------------
#! Hamiltonian Models
# ---------------------------------------------------------------------

class ModelHamiltonians(Enum):
    SYK2            = 'SYK2'
    FreeFermions    = 'FreeFermions'
    AubryAndre      = 'AubryAndre'

def create_hamiltonian(ns           : int, 
                    model_type      : Union[str, ModelHamiltonians], 
                    t               : float         = 1.0,
                    dtype           : type          = np.float64, backend: str = 'numpy',
                    seed            : Optional[int] = None,
                    **kwargs) -> QuadraticHamiltonian:
    '''
    Create a Hamiltonian of the specified type.
    '''
    if model_type == ModelHamiltonians.SYK2:
        return syk.SYK2(ns, dtype=dtype, backend=backend, seed=seed, logger=logger)
    elif model_type == ModelHamiltonians.FreeFermions:
        return free_fermions.FreeFermions(ns, t, dtype=dtype, backend=backend, logger=logger, seed=seed)
    elif model_type == ModelHamiltonians.AubryAndre:
        return aubry_andre.AubryAndre(lattice = kwargs.get('lattice', None),
                lx      = kwargs.get('lx', ns),
                ly      = kwargs.get('ly', 1),
                lz      = kwargs.get('lz', 1),
                dtype   = dtype,
                backend = backend,
                logger  = logger, seed = seed,
                lmbd    = kwargs.get('lmbd', 1.0),
                J       = kwargs.get('J', 1.0),
                beta    = kwargs.get('beta', 1.0))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
# --------------------------------------------------------------------
#! Orbital selection
# --------------------------------------------------------------------

def _energy_single(hamil: QuadraticHamiltonian, config: np.ndarray) -> float:
    return float(hamil.many_body_energy(config))

def choose_orbitals(arrangement     : Sequence[int],
                    filling         : int,
                    hamil           : QuadraticHamiltonian,
                    *,
                    number          : Optional[int]                 = None,
                    e_window        : Optional[float]               = None,
                    rng             : Optional[np.random.Generator] = None,
                    constraints     : List[Callable]                = []
                    ):
    '''
    Choose orbitals based on the given criteria.

    Parameters
    ----------
    arrangement : Sequence[int]
        The arrangement of orbitals.
    filling : int
        The number of orbitals to fill.
    hamil : Any
        The Hamiltonian object.
    number : Optional[int], optional
        The number of orbitals to choose. If None, all valid orbitals are chosen.
    e_window : Optional[float], optional
        The energy window for selecting orbitals. If None, no energy filtering is applied.
    rng : Optional[np.random.Generator], optional
        Random number generator for sampling.
        
    Returns
    -------
    List[Tuple[np.ndarray, float]]
        A list of tuples containing the chosen orbital configurations and their energies.
    '''
    
    if isinstance(arrangement, int):
        # interpret as orbitals 0..Ns-1
        ns                      = arrangement
        domain: Iterable[int]   = range(ns) # 0..Ns-1
    else:
        # use as given (list/tuple/range are fine; no copy)
        domain = arrangement
        try:
            ns      = len(arrangement)
        except TypeError:
            domain  = list(arrangement)
            ns      = len(domain)

    if ns <= 0 or filling <= 0 or filling > ns:
        raise ValueError("Invalid arrangement or filling.")

    # generator of all combinations - fast path
    combos  = itertools.combinations(domain, filling)
    if number is None and e_window is None:
        results = []
        for c in combos:
            cfg     = np.fromiter(c, dtype=int)
            ok      = True
            for constraint in constraints:
                ok = ok and constraint(cfg)
                
            if ok:
                e = float(hamil.many_body_energy(cfg))
                results.append((cfg, e))
                
        return results

    # Energy window filtering
    if e_window is None:
        emin, emax = -np.inf, np.inf
    else:
        if isinstance(e_window, (tuple, list)) and len(e_window) == 2:
            emin, emax = float(e_window[0]), float(e_window[1])
        else:
            emin, emax = -np.inf, float(e_window)

    if rng is not None and number is not None and isinstance(arrangement, int) and number <= max(1, int(0.01 * ns)):
        
        target      = int(number)
        total_combs = nCk(ns, filling)
        target      = min(target, total_combs)       # cap to available unique combos

        results     : List[Tuple[np.ndarray, float]] = []
        seen        = set()                              # dedupe sampled configs
        
        # pick a batch size that amortizes overhead but stays modest
        batch_size  = min(max(64, 4 * target), 4096)

        # safety guard: avoid infinite search if acceptance is tiny
        max_batches = 10000
        batches     = 0

        while len(results) < target and batches < max_batches:
            # draw B samples of size `filling` from 0..ns-1, without replacement within each row
            batches    += 1
            samples     = rng.choice(ns, size=(batch_size, filling), replace=False)
            samples.sort(axis=1)                     # canonicalize rows

            # dedupe within batch and against global set
            uniq = np.unique(samples, axis=0)
            for cfg in uniq:
                ok = True
                for constraint in constraints:
                    ok = ok and constraint(cfg)
                    
                if not ok:
                    continue
                
                key = cfg.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                e   = _energy_single(hamil, cfg)
                if emin <= e <= emax:
                    results.append((cfg.copy(), e))
                    if len(results) >= target:
                        break

        return results

    # general streaming path (combinations)    
    if rng is not None:
        domain = list(domain)  # ensure it's a list for shuffling
        rng.shuffle(domain)
    
    results = []
    taken   = 0
    for c in itertools.combinations(domain, filling):
        cfg = np.fromiter(c, dtype=int)
        ok = True
        for constraint in constraints:
            ok = ok and constraint(cfg)
            
        if not ok:
            continue
        
        e = float(hamil.many_body_energy(cfg))
        if emin <= e <= emax:
            results.append((cfg, e))
            taken += 1
            if number is not None and taken >= number:
                break

    return results

def q_constraint(ns: int, equals: float, prec: float) -> bool:
    coef = 2 * np.pi / ns
    def check(cfg: Sequence[int]) -> bool:
        sum_value = np.sum(np.array(cfg) * coef)
        while sum_value >= 2 * np.pi:
            sum_value -= 2 * np.pi
        isclose = np.isclose(sum_value, equals, atol=prec)
        return isclose
    return check

# --------------------------------------------------------------------
#! Orbitals for given subsystem
# --------------------------------------------------------------------

def prepare_trans_mat_orbitals(hamil: QuadraticHamiltonian, occ, *, ordered: bool = True, mask_a: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the transformation matrices for the Hamiltonian. Use the occupied orbitals indices. Afterwards, 
    one can use it to calculate, for instance, the entanglement entropy.
    """

    if ordered:
        transform: QuadraticHamiltonian.PCTransform = hamil.prepare_transformation(occ=occ)
    else:
        if mask_a is None:
            occ = np.sort(np.random.choice(hamil.ns, size=occ, replace=False))
        transform: QuadraticHamiltonian.PCTransform = hamil.prepare_transformation(occ=occ)
    return transform

# --------------------------------------------------------------------

def prepare_corr_matrix_sp(W_A: np.ndarray, W_A_CT: Optional[np.ndarray] = None, occ: Union[np.ndarray, List[int]] = None, **kwargs) -> np.ndarray:
    """
    Prepare the correlation matrix for the single-particle case.
    
    Parameters
    ----------
    W : np.ndarray
        The eigenvector matrix.
    W_A : np.ndarray
        The A part of the eigenvector matrix.
    W_A_CT : Optional[np.ndarray]
        The conjugate transpose of W_A.
    occ : Union[np.ndarray, List[int]], optional
        The occupation vector or indices.
    raw : bool, optional
        If True, use the occupation vector directly; otherwise, map to ±1 for entanglement Hamiltonian.
    
    Returns
    -------
    np.ndarray
        The correlation matrix.
    """
    if occ.dtype == np.bool_:
        comb_1_hot  = occ
    else:
        comb_1_hot  = to_one_hot(occ, size=W_A.shape[0], asbool=True)
        
    return Corr.corr_single(W_A, occ=comb_1_hot, W_A_CT=W_A_CT, **kwargs)

# --------------------------------------------------------------------

def prepare_corr_matrix_mb(many_body_state, hamil: QuadraticHamiltonian, la: Union[np.ndarray, List[int]], **kwargs) -> np.ndarray:
    """
    Prepare the correlation matrix for the many-body state.
    
    Parameters
    ----------
    many_body_state : np.ndarray
        The many-body state vector.
    hamil : QuadraticHamiltonian
        The Hamiltonian object.
    la : int, optional
        The number of occupied orbitals (default is 0, which means all).
    
    Returns
    -------
    np.ndarray
        The correlation matrix.
    """
    if la.dtype == np.bool_:
        comb_1_hot  = la
    else:
        comb_1_hot  = to_one_hot(np.arange(la), size=hamil.ns, asbool=True)
    
    corr = Corr.corr_from_statevector(many_body_state, Ns=hamil.ns, mode="slater", **kwargs)
    return corr[comb_1_hot, comb_1_hot] if isinstance(comb_1_hot, (list, np.ndarray)) else corr[:comb_1_hot, :comb_1_hot]

# --------------------------------------------------------------------
#! Main measure
# --------------------------------------------------------------------

def nongaussianity(eigvals: np.ndarray, return_stats: bool = False, eps: float = 1e-10):
    """
    Compute the nongaussianity measure from the eigenvalues.

    Parameters
    ----------
    eigvals      : array_like
        Eigenvalues e_a. Only entries with -1 < e_a < 1 contribute to NG.
    return_stats : bool
        If True, also return (gaussianity, m2, m4, m6).
    eps          : float
        Clip guard to avoid log(0).

    Returns
    -------
    NG [, gaussianity, m2, m4, m6]
    """
    e        = np.asarray(eigvals, dtype=np.float64)
    e2       = e * e
    m2       = np.sum(e2) / e.size
    m4       = np.sum(e2 * e2) / e.size
    m6       = np.sum(e2 * e2 * e2) / e.size

    # gaussianity = m4 / m2^2 - 1, with safe handling if m2 == 0
    denom    = m2 * m2
    gaussian = (m4 / denom - 1.0) if denom > 0.0 else 0.0

    # NG term: only for -1 < e < 1
    mask     = (e > -1.0) & (e < 1.0)
    if np.any(mask):
        em    = e[mask]
        ap    = 0.5 * (1.0 + em)     # (1+e)/2
        am    = 0.5 * (1.0 - em)     # (1-e)/2
        ap    = np.clip(ap, eps, 1.0)
        am    = np.clip(am, eps, 1.0)
        NG    = np.sum(ap * np.log(ap) + am * np.log(am))
    else:
        NG    = 0.0

    return (-NG, gaussian, m2, m4, m6) if return_stats else NG

# --------------------------------------------------------------------