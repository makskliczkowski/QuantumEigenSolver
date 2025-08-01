import sys
import os
os.environ['BACKEND'] = 'numpy'

# Add the parent directory (project root) to sys.path
script_dir                  = os.path.dirname(os.curdir)
parent_dir                  = os.path.abspath(os.path.join(script_dir, '..', '..'))
parent_parent_dir           = os.path.abspath(os.path.join(parent_dir, '..'))
parent_parent_parent_dir    = os.path.abspath(os.path.join(parent_parent_dir, '..'))

for path in [parent_dir, parent_parent_dir, parent_parent_parent_dir]:
    if path not in sys.path:
        sys.path.append(path)

import traceback
import numpy as np
import numba
import time
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Dict, Any, Union

# ------------------------------------------------------------------
try:
    #! General
    from QES.general_python.common.binary import get_global_logger, Array
    from QES.general_python.common import Plotter, colorsCycle, linestylesCycle
    from QES.general_python.common import Directories
    from QES.general_python.maths.math_utils import Fitter, FitterParams
    from QES.general_python.maths.statistics import Fraction
    # ------------------------------------------------------------------
    from QES.general_python.common.hdf5man import HDF5Manager
    from QES.general_python.physics import entropy
    # ------------------------------------------------------------------
    from QES.Algebra.Model.Interacting.Spin.ultrametric import UltrametricModel
    from QES.Algebra.Properties import statistical
    # ------------------------------------------------------------------
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

try:
    logger = get_global_logger()
except Exception as e:
    print(f"Error initializing logger: {e}")
    #! Fallback logger if global logger fails
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Error initializing logger: {e}")


# ------------------------------------------------------------------
#! Read the time evolution data from the HDF5 files
# ------------------------------------------------------------------

@dataclass
class TimeEvolutionData:
    time                : np.ndarray
    initial_state_e     : np.ndarray
    evolution           : np.ndarray
    evolution_rm        : np.ndarray
    diag_ens            : np.ndarray
    distribution        : Tuple[np.ndarray, np.ndarray]
    distribution_rm     : Tuple[np.ndarray, np.ndarray]
    moments             : Tuple[np.ndarray, ...]
    moments_rm          : Tuple[np.ndarray, ...]
    fft_om              : np.ndarray
    fft                 : np.ndarray
    fft_n               : float

def read_time_evolution_uniform(
            directories     : List[Directories],
            alpha           : float,
            ns              : int,
            n               : int,
            operator        : str,
            window          : int = 31,
        ):
    """
    Read the time evolution data from the HDF5 files for a uniform model.
    Parameters
    ----------
    directories : List[Directories]
        List of directories to search for the HDF5 files.
    alpha : float
        The alpha parameter for the model.
    ns : int
        The number of sites in the model.
    n : int
        The number of particles in the model.
    operator : str
        The operator to use for the time evolution.
    window : int
        The window size for the moving average.
    """
    
    time_start      = time.time()
    logger.info(f"Reading time evolution data for alpha={alpha:.3f}, ns={ns}, n={n}, operator={operator}", lvl=1, color='green')
    
    #! keys for Python data
    time_key        = '/time'
    time_evo_key    = f'/time_evolution/{operator}/expectation'
    time_evo_d_key  = f'/time_evolution/{operator}/diag_ens'
    time_evo_e_key  = f'/time_evolution/quench/energy'
    model           = UltrametricModel(ns = ns, n = n, alphas = alpha)
    modelstr        = str(model)
    directories     = [Directories(d, modelstr) for d in directories]
    filelist        = [x for d in directories for x in d.list_files(filters = [lambda x: str(x).endswith('h5')])]
    filelist_time   = [str(x) for x in filelist if "evo_" in str(x)]
    filelist_time   = sorted(filelist_time)

    #! read data from the Python files
    preamble        = f"[a={alpha:.3f},ns={ns},n={n},op={operator}]"
    if len(filelist_time):
        data_evo_time           = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_time, dataset_keys=[time_key])
        data_evo_time           = [x[time_key] for x in data_evo_time][0] # same time for all files (100000 time steps)
        logger.info(f"{preamble},data_evo_time shape: {data_evo_time.shape}", lvl=2, color='blue')

        data_evo_all            = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_time, dataset_keys=[time_evo_key])
        data_evo_all            = [x[time_evo_key] for x in data_evo_all]
        data_evo_samples        = [len(x) for x in data_evo_all]
        logger.info(f"{preamble},data_evo_samples: {data_evo_samples} for {modelstr}", lvl=3, color='blue')

        data_evo_all            = [y for x in data_evo_all for y in x if y.shape[0] > 0]
        data_evo_all            = np.array(data_evo_all, dtype=float)
        logger.info(f"{preamble},reshaped to: {data_evo_all.shape}", lvl=4, color='blue')
        # data_evo_av             = data_evo_av.reshape(-1, data_evo_av.shape[-1], order='C')
        
        #! diagonal ensemble
        data_evo_d_av           = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_time, dataset_keys=[time_evo_d_key])
        data_evo_d_av           = [x[time_evo_d_key] for x in data_evo_d_av]
        data_evo_d_av           = [y for x in data_evo_d_av for y in x]
        data_evo_d_av           = np.array(data_evo_d_av, dtype=float)

        #! energy of initial state
        data_evo_e_av           = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_time, dataset_keys=[time_evo_e_key])
        data_evo_e_av           = [x[time_evo_e_key] for x in data_evo_e_av]
        data_evo_e_av           = [y for x in data_evo_e_av for y in x]
        data_evo_e_av           = np.array(data_evo_e_av, dtype=float)

        #! remove diagonal ensemble
        data_evo_av             = data_evo_all - data_evo_d_av[:, np.newaxis]

        #! remove structure over the moving window
        data_evo_av_rm          = statistical.spectral_structure(data_evo_av, window = window)
    else:
        logger.error(f"No time evolution data found for {modelstr} in directories: {directories}")
        return None
    
    #! create the distribution of the time evolution values after removing the diagonal ensemble
    distr, distr_bins       = np.histogram(data_evo_av, bins=100, density=True)
    distr_bins              = 0.5 * (distr_bins[1:] + distr_bins[:-1])
    
    distr_rm, distr_bins_rm = np.histogram(data_evo_av_rm, bins=100, density=True)
    distr_bins_rm           = 0.5 * (distr_bins_rm[1:] + distr_bins_rm[:-1])
    
    #! moments of the values
    mu, mu2, mu3, mu4       = np.mean(data_evo_av, axis = 1),               \
                            np.mean(data_evo_av**2, axis = 1),              \
                            np.mean(data_evo_av**3, axis = 1),              \
                            np.mean(data_evo_av**4, axis = 1)
    
    mu_rm, mu2_rm, mu3_rm, mu4_rm = np.mean(data_evo_av_rm, axis = 1),      \
                            np.mean(data_evo_av_rm**2, axis = 1),           \
                            np.mean(data_evo_av_rm**3, axis = 1),           \
                            np.mean(data_evo_av_rm**4, axis = 1)

    logger.info(f"{preamble} starting FFT calculation after t={time.time() - time_start:.2f}s", lvl=2, color='green')
    #! FFT
    # 1) calculate the time delta and number of time steps
    # 2) calculate the FFT frequencies and take only the positive frequencies
    # 3) calculate the FFT of the time evolution data, normalize it, and take the positive frequencies
    # 4) normalize the FFT by the number of samples 
    time_delta              = data_evo_time[1] - data_evo_time[0]
    time_num                = data_evo_time.shape[-1]
    time_norm               = 2 / np.sqrt(time_num)
    data_evo_fft_om         = np.fft.fftfreq(time_num, d=time_delta)[1:time_num//2] * 2 * np.pi
    data_evo_fft_base       = np.fft.fft(data_evo_av, axis=1)[:, 1:time_num//2] * time_norm
    data_evo_fft            = np.abs(data_evo_fft_base)**2 / data_evo_time.shape[0]**2
    data_evo_fft            = np.mean(data_evo_fft, axis = 0)
    data_evo_fft_n          = np.trapz(data_evo_fft, x=data_evo_fft_om, axis=0)
    data_evo_fft            = data_evo_fft / data_evo_fft_n
    
    logger.info(f"time delta: {time_delta}, time num: {time_num}, shape: {data_evo_fft_om.shape}", lvl = 2)
    logger.info(f"data_evo_fft shape: {data_evo_fft.shape}", lvl = 3, color = 'blue')
    
    data_out                = TimeEvolutionData(
        time                = data_evo_time,
        initial_state_e     = data_evo_e_av,
        evolution           = data_evo_av,
        evolution_rm        = data_evo_av_rm,
        diag_ens            = data_evo_d_av,
        distribution        = (distr_bins, distr),
        distribution_rm     = (distr_bins_rm, distr_rm),
        moments             = (mu, mu2, mu3, mu4),
        moments_rm          = (mu_rm, mu2_rm, mu3_rm, mu4_rm),
        fft_om              = data_evo_fft_om,
        fft                 = data_evo_fft,
        fft_n               = data_evo_fft_n
    )

    logger.info(f"Finished reading time evolution data for {modelstr} in {time.time() - time_start:.2f}s", lvl=2, color='yellow')
    return data_out

# ------------------------------------------------------------------
#! ENERGY RELATED READOUTS
# ------------------------------------------------------------------


@dataclass
class EnergyData:
    
    @dataclass
    class EnergyClass:
        eigenvalues         : np.ndarray
        sigma_2_e           : float
        bandwidth           : float
        mean_energies       : np.ndarray
        mean_energies_idx   : np.ndarray
        
    @dataclass
    class EntropyClass:
        von_neumann         : np.ndarray
        tsallis             : np.ndarray
        schmidt_gap         : np.ndarray

    energy                  : EnergyClass
    gap_ratios              : Tuple[float, float]
    mean_level_spacing      : Tuple[float, float]
    ldos                    : np.ndarray
    iprs                    : Dict[float, np.ndarray]
    participation_entropies : Dict[float, np.ndarray]
    entropy                 : EntropyClass = None

def read_energy(
            directories     : List[Directories],
            alpha           : float,
            ns              : int,
            n               : int,
        ):
    
    #! keys for Python data
    logger.info(f"Reading energy data for alpha={alpha:.3f}, ns={ns}, n={n}", lvl=1, color='green')
    time_start      = time.time()
    energy_key      = '/energies'
    ldos_key        = '/ldos'
    vn_key          = '/entropy/quench/von_neumann/1.0'
    tsalis_key      = '/entropy/quench/tsallis/2.0'
    schmidt_gap_key = '/entropy/quench/schmidt_gap'
    model           = UltrametricModel(ns = ns, n = n, alphas = alpha)
    modelstr        = str(model)
    directories     = [Directories(d, modelstr) for d in directories]
    # check whether the directories exist
    directories     = [d for d in directories if d.exists()]
    filelist        = [x for d in directories for x in d.list_files(filters = [lambda x: str(x).endswith('h5')])]
    filelist_energy = [str(x) for x in filelist if "stat_" in str(x) or "energy_" in str(x)]
    filelist_energy = sorted(filelist_energy)
    #! read data from the Python files
    try:
        if len(filelist_energy):
            data_energy             = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_energy, dataset_keys=[energy_key])
            data_energy             = [x[energy_key] for x in data_energy]
            data_energy             = [y for x in data_energy for y in x if y.shape[0] > 0] # flatten and remove empty arrays
            data_energy             = np.array(data_energy, dtype=float)
            if len(data_energy) == 0:
                logger.error(f"No energy data found for {modelstr} in directories: {directories}", color='red', lvl=1)
                return None
            
            hilber_dim              = data_energy.shape[1] if data_energy.ndim > 1 else 1
            
            # calculate the mean level spacing and bandwidth
            differences             = np.diff(data_energy, axis=1)
            sigma_2_e               = np.mean(np.var(data_energy, axis=1), axis=0)                          # mean variance of the energy levels
            bandwidth_data          = np.mean(np.max(data_energy, axis=1) - np.min(data_energy, axis=1))    # mean bandwidth of the energy levels
            mean_lvl_spacing        = np.mean(np.mean(differences, axis = 1), axis=0)                       # mean level spacing over all samples
            mean_lvl_spacing_c_100  = np.mean(np.mean(differences[:, hilber_dim//2 - 50: hilber_dim//2 + 50], axis = 1), axis=0) if hilber_dim > 100 else mean_lvl_spacing
                    
            # gap ratio
            gap_ratios              = np.minimum(differences[:, :-1], differences[:, 1:]) / np.maximum(differences[:, :-1], differences[:, 1:])
            mean_gap_ratio          = np.nanmean(np.mean(gap_ratios, axis=1) if gap_ratios.size > 0 else np.nan)
            mean_gap_ratio_c_100    = np.nanmean(np.mean(gap_ratios[:, hilber_dim//2 - 50: hilber_dim//2 + 50], axis=1) if hilber_dim > 100 else np.nan)
            mean_energies           = np.mean(data_energy, axis=1)
            mean_energies_idx       = np.argmin(np.abs(data_energy - mean_energies[:, np.newaxis]), axis=1)
            
            #! LDOS
            data_ldos               = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_energy, dataset_keys=[ldos_key])
            data_ldos               = [x[ldos_key] for x in data_ldos]
            data_ldos               = [y for x in data_ldos for y in x if x.shape[0] > 0]
            data_ldos               = np.array(data_ldos, dtype=float)

            #! IPRS from the LDOS
            qs                      = [0.5, 1.0, 2.0, 3.0, 4.0] # divide by 2, as we have the LDOS, which is already squared - IPRS for quench
            iprs                    = { q: statistical.inverse_participation_ratio(data_ldos.T, q = q / 2) for q in qs }
            participation_entropies = { q: entropy.participation_entropy(data_ldos.T, q = q / 2) for q in qs }
            
            #! Entropies
            vn_entropies           = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_energy, dataset_keys=[vn_key])
            if len(vn_entropies) > 0:
                vn_entropies           = [x[vn_key] for x in vn_entropies if vn_key in x]
                vn_entropies           = [y for x in vn_entropies for y in x if x.shape[0] > 0]
                vn_entropies           = np.array(vn_entropies, dtype=float)
            else:
                vn_entropies           = np.array([])

            tsalis_entropies       = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_energy, dataset_keys=[tsalis_key])
            if len(tsalis_entropies) > 0:
                tsalis_entropies       = [x[tsalis_key] for x in tsalis_entropies if tsalis_key in x]
                tsalis_entropies       = [y for x in tsalis_entropies for y in x if x.shape[0] > 0]
                tsalis_entropies       = np.array(tsalis_entropies, dtype=float)
            else:
                tsalis_entropies       = np.array([])

            schmidt_gaps           = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_energy, dataset_keys=[schmidt_gap_key])
            if len(schmidt_gaps) > 0:
                schmidt_gaps           = [x[schmidt_gap_key] for x in schmidt_gaps if schmidt_gap_key in x]
                schmidt_gaps           = [y for x in schmidt_gaps for y in x if x.shape[0] > 0]
                schmidt_gaps           = np.array(schmidt_gaps, dtype=float)
            else:
                schmidt_gaps           = np.array([])
        else:
            return None
    except Exception as e:
        logger.error(f"Error reading energy data: {e}", color='red', lvl=2)
        tb_str = traceback.format_exc()
        logger.error(f"Traceback:\n{tb_str}", color='red', lvl=3)
        return None
    
    data_out                    = EnergyData(
        energy                  = EnergyData.EnergyClass(
            eigenvalues         = data_energy,
            sigma_2_e           = sigma_2_e,
            bandwidth           = bandwidth_data,
            mean_energies       = mean_energies,
            mean_energies_idx   = mean_energies_idx
        ),
        gap_ratios              = (mean_gap_ratio, mean_gap_ratio_c_100),
        mean_level_spacing      = (mean_lvl_spacing, mean_lvl_spacing_c_100),
        ldos                    = data_ldos,
        iprs                    = iprs,
        participation_entropies = participation_entropies,
        entropy                 = EnergyData.EntropyClass(von_neumann = vn_entropies,
                                                        tsallis = tsalis_entropies,
                                                        schmidt_gap = schmidt_gaps)
    )

    logger.info(f"Finished reading energy data for {modelstr} in {time.time() - time_start:.2f}s", lvl=2, color='yellow')
    return data_out

# ------------------------------------------------------------------
#! Read the histogram data from the HDF5 files
# ------------------------------------------------------------------

@dataclass
class HistogramData:
    v : np.ndarray  # values of the histogram, can be the same as counts
    b : np.ndarray  # bins of the histogram
    c : np.ndarray  # counts of the histogram

@dataclass
class ObservableData:
    histogram       : HistogramData
    histogram_typ   : HistogramData
    cdf             : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    cdf_typ         : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def read_histogram(
            directories         : List[Directories],
            alpha               : float,
            ns                  : int,
            n                   : int,
            operator            : str,
            window              : int           = 31,
            cutoff_small        : float         = 1e-12
        ):
    
    #! keys for Python data
    logger.info(f"Reading histogram data for alpha={alpha:.3f}, ns={ns}, n={n}, operator={operator}", lvl=1, color='green')
    time_start      = time.time()
    hist_key        = f'/historgram/{operator}/average/values'
    hist_typ_key    = f'/historgram/{operator}/typical/values'
    hist_count_key  = '/historgram/counts'
    hist_edges_key  = '/historgram/edges'
    model           = UltrametricModel(ns = ns, n = n, alphas = alpha)
    modelstr        = str(model)
    directories     = [Directories(d, modelstr) for d in directories]
    filelist        = [x for d in directories for x in d.list_files(filters = [lambda x: str(x).endswith('h5')])]
    filelist_hist   = [str(x) for x in filelist if "hist_" in str(x)]
    filelist_hist   = sorted(filelist_hist)
    
    if len(filelist_hist):
        data_hist_bins          = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_hist, dataset_keys=[hist_edges_key])
        data_hist_bins          = np.mean([x[hist_edges_key] for x in data_hist_bins], axis=0) #! bins should be the same for all files

        #! read the histogram data
        
        # average
        data_hist_av            = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_hist, dataset_keys=[hist_key])
        data_hist_av            = np.mean([x[hist_key] for x in data_hist_av], axis=0)
        mask                    = (data_hist_av > cutoff_small) & np.isfinite(data_hist_av)
        if mask.any():
            data_hist_av         = data_hist_av[mask]       # remove zeros
            data_hist_bins       = data_hist_bins[mask]     
        else:
            data_hist_av         = np.zeros_like(data_hist_bins)
            data_hist_bins       = data_hist_bins
            
        # typical
        data_hist_typ           = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_hist, dataset_keys=[hist_typ_key])
        data_hist_typ           = np.mean([(x[hist_typ_key]) for x in data_hist_typ], axis=0)
        data_hist_typ           = np.exp(data_hist_typ) # convert to log scale

        # counts
        data_hist_count         = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_hist, dataset_keys=[hist_count_key])
        data_hist_count         = np.mean([x[hist_count_key] for x in data_hist_count], axis=0)
        
        #! normalize
        # data_hist_av            = data_hist_av / np.trapz(data_hist_av, x=data_hist_bins)
        # data_hist_typ           = data_hist_typ / np.trapz(data_hist_typ, x=data_hist_bins)

        oms, smooth, cdf, g     = statistical.spectral_cdf(data_hist_bins, data_hist_av, BINVAL=window)
        omst, smootht, cdft, gt = statistical.spectral_cdf(data_hist_bins, data_hist_av, BINVAL=window)
        
        data_out                = ObservableData(
            histogram               = HistogramData(
                                        v = data_hist_av,
                                        b = data_hist_bins,
                                        c = data_hist_count),
            histogram_typ           = HistogramData(
                                        v = data_hist_typ,
                                        b = data_hist_bins,
                                        c = data_hist_count
                                    ),
            cdf                     = (oms, smooth, cdf, g),
            cdf_typ                 = (omst, smootht, cdft, gt)
        )
        
        logger.info(f"Finished reading histogram data for {modelstr} in {time.time() - time_start:.2f}s", lvl=2, color='yellow')
        return data_out
    return None

# ------------------------------------------------------------------
#! Matrix elements readout
# ------------------------------------------------------------------

@dataclass
class DiagonalMatrixData:
    diagonal        : Tuple[np.ndarray, np.ndarray, np.ndarray]
    histogram       : Tuple[np.ndarray, np.ndarray]
    gaussianity     : float
    binder_cumulant : np.ndarray
    microcanonical  : np.ndarray

def read_diagonal_matrix_elements(
            directories     : List[Directories],
            alpha           : float,
            ns              : int,
            n               : int,
            operator        : str,
            mean_en_indices : np.ndarray = None
        ):
    """
    Read the diagonal matrix elements of a given operator for a uniform model.
    
    Parameters
    ----------
    directories : List[Directories]
        List of directories to search for the HDF5 files.
    alpha : float
        The alpha parameter for the model.
    ns : int
        The number of sites in the model.
    n : int
        The number of particles in the model.
    operator : str
        The operator to read the diagonal matrix elements for.
    
    Returns
    -------
    dict
        A dictionary containing the diagonal matrix elements and their statistics.
    """
    
    #! keys for Python data
    logger.info(f"Reading diagonal matrix elements for alpha={alpha:.3f}, ns={ns}, n={n}, operator={operator}", lvl=1, color='green')
    time_start      = time.time()
    diag_key        = f'/{operator}'
    model           = UltrametricModel(ns = ns, n = n, alphas = alpha)
    modelstr        = str(model)
    directories     = [Directories(d, modelstr) for d in directories]
    filelist        = [x for d in directories for x in d.list_files(filters = [lambda x: str(x).endswith('h5')])]
    filelist_diag   = [str(x) for x in filelist if "diag_" in str(x)]
    filelist_diag   = sorted(filelist_diag)

    if len(filelist_diag):
        data_diag          = HDF5Manager.load_data_from_multiple_files(file_paths=filelist_diag, dataset_keys=[diag_key])
        data_diag          = [x[diag_key] for x in data_diag]
        data_diag          = [y for x in data_diag for y in x if y.shape[0] > 0] # flatten and remove empty arrays
        data_diag          = np.array(data_diag, dtype=float)
        
        #! calculate statistics
        mean_diag          = np.mean(data_diag, axis=1)
        std_diag           = np.std(data_diag, axis=1)
        
        #! histogram of the diagonal matrix elements
        hist_diag, edges_diag   = np.histogram(data_diag, bins=100, density=True)
        
        #! gaussianity of the diagonal matrix elements
        mean_squared    = np.mean(data_diag**2, axis=1)
        squared_mean    = np.mean(np.abs(data_diag), axis=1)**2
        gaussianity     = np.mean(mean_squared / squared_mean)

        #! binder cumulant
        mean_fourth     = np.mean(data_diag**4, axis=1)
        binder_cumulant = mean_fourth / (3 * mean_squared**2)

        #! microcanonical prediction
        if mean_en_indices is not None:
            hilbert_dim = data_diag.shape[-1]
            n_states_me = 20
            # mean energy indices is vector of indices in axis-1 of diagonal matrix elements corresponding to the mean energy
            if hilbert_dim > n_states_me:
                left_indices    = np.maximum(mean_en_indices - n_states_me // 2, 0)
                right_indices   = np.minimum(mean_en_indices + n_states_me // 2, hilbert_dim)
                mean_diag       = np.array([
                    np.mean(data_diag[i, left:right], axis=0)
                    for i, (left, right) in enumerate(zip(left_indices, right_indices))
                ])
            else:
                logger.warning(f"Not enough states for microcanonical prediction: {hilbert_dim} < {n_states_me}")
                mean_diag   = np.mean(data_diag, axis=1)
        else:
            mean_diag = np.zeros(data_diag.shape[0])

        data_out = DiagonalMatrixData(
            diagonal        = (data_diag, mean_diag, std_diag),
            histogram       = (hist_diag, edges_diag),
            gaussianity     = gaussianity,
            binder_cumulant = binder_cumulant,
            microcanonical  = mean_diag
        )
        
        logger.info(f"Finished reading diagonal matrix elements for {modelstr} in {time.time() - time_start:.2f}s", lvl=2, color='yellow')
        return data_out
    
    return None

# ------------------------------------------------------------------
#! Extract the K-functions over whole spectrum from LDOS for initial states
# ------------------------------------------------------------------

@numba.njit(inline='always')
def find_bin(bins, omega):
    left, right = 0, bins.shape[0] - 2
    while left <= right:
        mid = (left + right) // 2
        if bins[mid] <= omega < bins[mid + 1]:
            return mid
        elif omega < bins[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return -1

@numba.njit(fastmath=True)
def k_function(ldos     :   np.ndarray,
            energies    :   np.ndarray,
            bins        :   np.ndarray,
            target      :   float = 0.0,
            tol         :   float = 0.015
            ):
    r"""
    Compute 
        k(omega) = \sum _{ij} ldos[i] * ldos[j] * delta (omega - |E[j]-E[i]|)
    using a histogram binning method.
    
    Parameters
    ----------
    ldos : float64
        Local density of states (LDOS) at each energy level.
    energies : float64
        Energies of the system.
    bins : float64
        Bins for histogramming the energy differences.
    """
    
    nE   = energies.shape[0]
    nbin = bins.shape[0] - 1
    kf   = np.zeros(nbin, ldos.dtype)
    cnt  = np.zeros(nbin, np.int64)

    for i in numba.prange(nE):
        ei = energies[i]
        li = ldos[i]
        for j in range(i, nE):
        # for j in range(0, nE):
            # absolute energy difference
            ej      = energies[j]
            
            # if abs((ei + ej) / 2.0 - target) < tol:
            #     continue

            dE      = abs(ei - ej)
            omega   = dE if dE >= 0.0 else -dE
            
            # find bin index: largest b such that bins[b] <= omega
            idx     = find_bin(bins, omega)

            # clamp to valid range [0, nbin−1]
            if idx < 0:
                # idx = 0
                continue
            elif idx >= nbin:
                # idx = nbin - 1
                continue

            kf[idx]  += li * ldos[j]
            cnt[idx] += 1

    return kf, cnt

def extract_k_functions(data_ldos: np.ndarray, data_energy: np.ndarray, data_hist_bins: np.ndarray, energies_at: np.ndarray = None):
    """
    Extract k-functions from the LDOS and energy data.

    Parameters
    ----------
    data_ldos : np.ndarray
        The LDOS data.
    data_energy : np.ndarray
        The energy data.
    data_hist_bins : np.ndarray
        The histogram bins.
    energies_at : np.ndarray, optional
        Specific energies to extract k-functions at.

    Returns
    -------
    k_functions : Dict[str, np.ndarray]
        Dictionary with k-functions for each energy.
    """
    logger.info("Extracting k-functions from LDOS and energy data", lvl=1, color='green')
    time_start = time.time()
    
    # if energies_at is None:
        # energies_at = np.mean(data_energy, axis = 1)  # Use mean energy if not specified
    
    k_function_list         = [statistical.k_function(data_ldos[i], data_energy[i], data_hist_bins, None) for i in range(len(data_energy))]
    # k_function_list         = [k_function(data_ldos[i], data_energy[i], data_hist_bins, None) for i in range(len(data_energy))]
    k_functions             = np.array([x[0] for x in k_function_list])
    k_counts                = np.array([x[1] for x in k_function_list])
    k_counts[k_counts == 0] = 1
    k_functions             = (k_functions / k_counts)
    k_functions             = np.mean(k_functions, axis=0)
    # k_functions             = k_functions / np.trapz(k_functions, x=data_hist_bins[1:])
    logger.info(f"Extracted k-functions in {time.time() - time_start:.2f}s", lvl=2, color='yellow')
    return k_functions, k_counts

# ------------------------------------------------------------------
#! Extract the power spectrum from multiplying the k functions times the O^2
# ------------------------------------------------------------------

def extract_k_times_o2(k_functions, data_hist, data_hist_bins):
    '''
    Extract k-functions for the O2 operator from the histogram data.
    Parameters
    ----------
    k_functions : np.ndarray
        The k-functions extracted from the LDOS and energy data.
    data_hist : np.ndarray
        The histogram data - spectral function.
    data_hist_bins : np.ndarray
        The histogram bins.
    Returns
    -------
    k_times_hist : np.ndarray
        The k-functions multiplied by the histogram data.
    k_times_hist_norm : float
        The normalization factor for the k-functions multiplied by the histogram data.
    '''
    k_times_hist            = k_functions * data_hist[1:]
    k_times_hist_norm       = np.trapz(k_times_hist, x=data_hist_bins[1:])
    k_times_hist           /= k_times_hist_norm
    return k_times_hist, k_times_hist_norm

# ------------------------------------------------------------------