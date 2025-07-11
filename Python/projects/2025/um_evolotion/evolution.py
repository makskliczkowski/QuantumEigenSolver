#!/usr/bin/env python3
"""
Script to perform time-evolution statistics for an ultrametric spin model.
Parameters:
    save_dir               Base directory for output data
    alpha_start            Initial coupling exponent \(\alpha\)
    alpha_step             Step size for \(\alpha\)
    alpha_num              Number of \(\alpha\) values
    n_realizations         Number of disorder realizations per system size
    sites_start            Minimum number of spins
    sites_end              Maximum number of spins (inclusive)
    n                      Model parameter \(n\)
    time_num               Number of time steps (default: 1e5)
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import time
import numpy as np
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

#! -------------------------------------------------------
os.environ['BACKEND'] = 'numpy'
#! -------------------------------------------------------

# Add parent directories to sys.path for imports
for _ in range(4):
    parent = os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * (_ + 1))))
    if parent not in sys.path:
        sys.path.append(parent)

# project imports
from QES.general_python.run_scripts.slurm import SlurmMonitor

from QES.general_python.common.binary import get_global_logger

from QES.general_python.algebra.linalg import overlap
from QES.general_python.common import Directories
from QES.general_python.common import HDF5Handler
from QES.general_python.run_scripts.prepareini import ManyBodyEstimator
from QES.general_python.run_scripts import calculate_optimal_workers

from QES.general_python.maths.statistics import HistogramAverage

#! VON NEUMANN, TSALLIS, PARTICIPATION ENTROPY
from QES.general_python.physics import entropy as entropy, density_matrix as density_matrix

#! MODEL
from QES.Algebra.Model.Interacting.Spin.ultrametric import UltrametricModel

from QES.Algebra.Properties import time_evolution, statistical, time_evo
import QES.Algebra.Operator.operators_spin as op_spin

logger = get_global_logger()

# ------------------------------------------------------------------
#! Read the parameters

def _single_realisation(
        model                   : UltrametricModel,
        r                       : int,
        sigma_es                : np.ndarray,   # save the energy width
        bandwidths              : np.ndarray,   # save the bandwidth
        energies                : np.ndarray,   # save the energies
        ldos                    : np.ndarray,   # save the local density of states
        iprs                    : dict,         # save the inverse participation ratio
        entros                  : dict,         # save the participation entropy
        quench_energies         : np.ndarray,   # save the quench energies
        time_vals               : dict,         # save the time evolution values
        time_steps              : np.ndarray,   # save the time steps
        operators               : dict,         # operators to compute - has the names etc
        operators_mat           : dict,         # operators matrices
        omegas_allocated        : np.ndarray,   # allocated omegas for histogram
        vals_allocated          : np.ndarray,   # allocated values for histogram
        h_av                    : dict,         # histogram averages
        h_typ                   : dict,         # histogram typical values
        diagonals_operators     : dict,         # save the diagonal elements of the operators
        diagonal_ensembles      : dict,         # save the diagonal ensembles
        vn_entropies            : np.ndarray,   # save the von Neumann entropies for the quench state
        tsalis                  : np.ndarray,   # save the Tsallis entropies for the quench state
        schmidt_gaps            : np.ndarray,   # save the Schmidt gaps for the quench state
        iprs_quench             : dict,         # save the IPR for the quench state
        par_ent_quench          : dict,         # save the participation entropy for the quench state
        # optional parameters - info about the job
        start_time              : float = None, # start time for the job - when it started
        job_time                : float = None, # job time for the job - how long it can run
        ):

    if SlurmMonitor.is_overtime(limit=1200, start_time=start_time, job_time=job_time):  # 20 minutes buffer
        logger.warning(f"SLURM job approaching timeout, skipping realization {r}")
        return False
    
    #! constants
    alpha           = model.alphas[0]
    ns              = model.ns
    hilbert_size    = 2**ns
    batch_limit     = lambda ns: ns > 14
    batch_num       = lambda ns: 10 if batch_limit(ns) else 1
    
    time_start_r    = time.perf_counter()
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r}", lvl=3, color='red')

    #! build & diagonalize
    # model.clear()
    model.build()
    model.diagonalize(verbose=True)
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} diag done in {time.perf_counter() - time_start_r:.2f} s", lvl=3, color='red')
    
    #! compute the bandwidth
    time_start_stat = time.perf_counter()
    mean_energy                 = model.av_en
    bandwidth                   = model.get_bandwidth()
    sigma_e                     = model.get_energywidth()
    bandwidths[r]               = bandwidth
    sigma_es[r]                 = sigma_e
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} bandwidth = {bandwidth:.2f} done in {time.perf_counter() - time_start_stat:.2f} s", lvl=3, color='red')
    
    #! IPR
    time_start_ipr = time.perf_counter()
    for q in iprs.keys():
        iprs[q][r, :] = statistical.inverse_participation_ratio(model.eig_vec, q=q)
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} IPR done in {time.perf_counter() - time_start_ipr:.2f} s", lvl=3, color='red')

    #! participation entropy
    time_start_entro = time.perf_counter()
    for q in entros.keys():
        entros[q][r, :] = entropy.entropy(model.eig_vec, q=q, typek=entropy.Entanglement.PARTIC)
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} entropy done in {time.perf_counter() - time_start_entro:.2f} s", lvl=3, color='red')

    #! select the quench state
    quench_state            = np.zeros(hilbert_size, dtype=np.float64)
    diff                    = np.abs(model.hamil.diagonal() - mean_energy)
    idx                     = int(np.argmin(diff))  # find the closest eigenvalue
    quench_state[idx]       = 1.0                   # set the quench state
    quench_energies[r]      = overlap(quench_state, model.hamil)
    overlaps                = model.eig_vec.T @ quench_state
    soverlaps               = np.square(np.abs(overlaps))
    
    if True:
        time_start_entro_quench = time.perf_counter()
        rho                 = density_matrix.rho_numba(overlaps, dimA=2, dimB=2**(ns - 1))
        #! calculate the entropies for the initial state in the eigenbasis
        schmidt_val, _      = density_matrix.schmidt_numba(rho, 2, 2**(ns - 1), eig = False)
        #! von Neumann entropy
        vn_entropies[r]     = entropy.entropy(schmidt_val, q=1.0, typek=entropy.Entanglement.VN)
        #! Tsallis entropy
        tsalis[r]           = entropy.entropy(schmidt_val, q=2.0, typek=entropy.Entanglement.TSALLIS)
        #! Schmidt gaps
        schmidt_gaps[r]     = schmidt_val[1] - schmidt_val[0] if len(schmidt_val) > 1 else 0.0 
        #! Iprs for the quench state
        for q in iprs_quench.keys():
            iprs_quench[q][r]       = statistical.inverse_participation_ratio(overlaps[np.newaxis, :], q=q)[0]
        for q in par_ent_quench.keys():
            par_ent_quench[q][r]    = entropy.entropy(overlaps[np.newaxis, :], q=q, typek=entropy.Entanglement.PARTIC)[0]
        logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} entropies for quench state done in {time.perf_counter() - time_start_entro_quench:.2f} s", lvl=3, color='white')
        
    #! compute the ldos
    ldos[r, :]              = statistical.ldos(energies = model.eig_val, overlaps = overlaps)
    energies[r, :]          = model.eig_val
        
    #! accumulate into the single histogram
    for name in operators.keys():
        logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} computing histogram for operator {name}", lvl=3, color='blue')
        
        matrix_elements                 = overlap(model.eig_vec, operators_mat[name])
        diagonals_operators[name][r, :] = matrix_elements.diagonal()
        diagonal_ensembles[name][r]     = time_evo.diagonal_ensemble(soverlaps=soverlaps, diag_mat=matrix_elements.diagonal())

        time_start_hist = time.perf_counter()
        cnt = statistical.f_function(
                start               = 0,
                stop                = len(model.eig_vec),
                overlaps            = matrix_elements,
                eigvals             = model.eig_val,
                omegas_allocated    = omegas_allocated,
                vals_allocated      = vals_allocated,
                energy_target       = mean_energy,
                bw                  = bandwidth,
                energy_diff_cut     = 0.015)
        logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} histogram done in {time.perf_counter() - time_start_hist:.2f} s", lvl=4, color='blue')
        
        #! remove zeros and nan
        omegas                  = omegas_allocated[:cnt]
        vals                    = vals_allocated[:cnt]
        mask                    = np.logical_and(vals > 1e-10, np.isfinite(vals))
        omegas                  = omegas[mask]
        vals                    = vals[mask]
        h_av[name].append(omegas, vals)
        h_typ[name].append(omegas, np.log(vals))
        
        #! time evolution - compute the expectation values
        time_start_evo = time.perf_counter()
        if not batch_limit(ns):
            # Small systems: compute all at once
            quench_states_t         = time_evo.time_evo_block(model.eig_vec, model.eig_val, quench_overlaps=overlaps, times=time_steps)
            quenched_values_t       = time_evo.time_evo_evaluate(quench_states_t, operators_mat[name])
            time_vals[name][r, :]   = np.real(quenched_values_t)
        else:
            # Large systems: batch processing
            batch_count             = batch_num(ns)
            batch_size              = (time_num + batch_count - 1) // batch_count
            for i in range(batch_count):
                start_idx   = i * batch_size
                end_idx     = min(start_idx + batch_size, time_num)
                
                if start_idx >= end_idx:
                    break
                    
                # Check timeout during batching
                if SlurmMonitor.is_overtime(limit=600):  # 10 minutes buffer
                    logger.warning(f"SLURM timeout during batch {i}, stopping early")
                    break
                
                # Compute time evolution for this batch
                batch_times             = time_steps[start_idx:end_idx]
                quench_states_batch     = time_evo.time_evo_block(model.eig_vec, model.eig_val, quench_overlaps=overlaps, times=batch_times)
                quenched_values_batch   = time_evo.time_evo_evaluate(quench_states_batch, operators_mat[name])
                time_vals[name][r, start_idx:end_idx] = np.real(quenched_values_batch)
                
                # Clean up memory
                del quench_states_batch, quenched_values_batch

        logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} time evolution done in {time.perf_counter() - time_start_evo:.2f} s", lvl=4, color='blue')

    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} done in {time.perf_counter() - time_start_r:.2f} s", lvl=4, color='red')
    return True

def _single_alpha(alpha             : float, 
                ns                  : int, 
                n                   : int, 
                time_num            : int, 
                uniform             : bool, 
                data_dir            : Directories,
                rand_num            : int,
                n_realisations      : int,
                mean_lvl_spacing    : float,
                bandwidth_data      : float,
                operators           : dict,
                # allocated data
                vals_allocated      : np.ndarray,
                omegas_allocated    : np.ndarray,
                sigma_es            : np.ndarray,
                bandwidths          : np.ndarray,
                energies            : np.ndarray,
                ldos                : np.ndarray,
                iprs                : dict,
                entros              : dict,
                diagonals_operators : dict,
                diagonal_ensembles  : dict,
                histograms_av       : dict,
                histograms_typ      : dict,
                time_vals           : dict,
                operators_mat       : dict,
                start_time          : float = None,
                job_time            : float = None,
                seed                : int   = None
                ) -> None:

    logger.info(f"ns = {ns}, alpha = {alpha:.2f}", lvl=2, color='green')
    time_start_a        = time.perf_counter()
    hilbert_size        = 2**ns
    
    #! constants
    heisenberg_time     = 1.0 / mean_lvl_spacing * 2.0 * np.pi
    time_delta          = 2.0 * np.pi / bandwidth_data
    time_start          = heisenberg_time - time_num * time_delta / 2
    if (time_start < 0.0):
        time_start      = heisenberg_time / 100
    
    #! allocate the data
    quench_energies     = np.zeros((n_realisations,), dtype=np.float64)
    
    #! time evolution
    if uniform:
        time_steps      = np.linspace(time_start, time_start + time_delta * time_num, num=time_num)
    else:
        time_steps      = np.logspace(-2, np.log10(hilbert_size * 100), num=time_num)
            
    #! histogram
    o_min, o_max        = 0.01 / hilbert_size, 3 * bandwidth_data
    n_bins              = int(20 * ns)
    edges               = HistogramAverage.uniform_log(n_bins, v_max=o_max, v_min=o_min)
    for name in operators.keys():
        h_av    = { k : HistogramAverage(n_bins, edges=edges) for k in operators.keys() }
        h_typ   = { k : HistogramAverage(n_bins, edges=edges) for k in operators.keys() }

    #! entropies 
    vn_entropies        = np.zeros((n_realisations,), dtype=np.float64)  # von Neumann entropies
    tsalis              = np.zeros((n_realisations,), dtype=np.float64)  # Tsallis entropies
    schmidt_gaps        = np.zeros((n_realisations,), dtype=np.float64)  # Schmidt gaps
    par_ent_quench      = { q : np.zeros((n_realisations), dtype=np.float64) for q in [0.5, 1.0, 2.0] }  # participation entropy for the quench state
    iprs_quench         = { q : np.zeros((n_realisations), dtype=np.float64) for q in [0.5, 1.0, 2.0] }  # IPR for the quench state

    #! model
    model               = UltrametricModel(ns = ns, n = n, alphas = alpha, backend = np, seed = seed)
    
    #! directories
    data_dir_in = data_dir.join('uniform' if uniform else 'log', f'{str(model)}')
    data_dir_in.mkdir()
    
    completed_realizations = 0
    #! go through the realisations
    for r in range(n_realisations):
        success = _single_realisation(
            model                   = model,
            r                       = r,
            sigma_es                = sigma_es,
            bandwidths              = bandwidths,
            energies                = energies,
            ldos                    = ldos,
            iprs                    = iprs,
            entros                  = entros,
            quench_energies         = quench_energies,
            time_vals               = time_vals,
            time_steps              = time_steps,
            operators               = operators,
            operators_mat           = operators_mat,
            omegas_allocated        = omegas_allocated,
            vals_allocated          = vals_allocated,
            h_av                    = h_av,
            h_typ                   = h_typ,
            diagonals_operators     = diagonals_operators,
            diagonal_ensembles      = diagonal_ensembles,
            start_time              = start_time,
            job_time                = job_time,
            vn_entropies            = vn_entropies,
            tsalis                  = tsalis,
            schmidt_gaps            = schmidt_gaps,
            iprs_quench             = iprs_quench,
            par_ent_quench          = par_ent_quench,
        )
        
        if success:
            completed_realizations += 1
        else:
            logger.warning(f"Realization {r} failed or was skipped due to timeout")
            break
    logger.info(f"Completed {completed_realizations}/{n_realisations} realizations for ns={ns}, alpha={alpha:.2f}")
    
    #! store the combined histogram for this (ns,alpha)
    for name in operators:
        histograms_av[name][alpha][ns]  = h_av[name]
        histograms_typ[name][alpha][ns] = h_typ[name]
    
    #! save the data
    if completed_realizations > 0:
        logger.info(f"ns = {ns}, alpha = {alpha:.2f} saving data to {data_dir_in} - random number: {rand_num}", lvl=2, color='green')
        data_stat = {
            'bandwidth'                         : bandwidths[:completed_realizations],
            'sigma_e'                           : sigma_es[:completed_realizations],
            'energies'                          : energies[:completed_realizations],
            'ldos'                              : ldos[:completed_realizations],
            'entropy/quench/von_neumann/1.0'    : vn_entropies[:completed_realizations],
            'entropy/quench/tsallis/2.0'        : tsalis[:completed_realizations],
            'entropy/quench/schmidt_gap'        : schmidt_gaps[:completed_realizations],
            'completed_realizations'            : completed_realizations,
        }
        data_stat.update({f'iprs/quench/{q}'                        : iprs_quench[q][:completed_realizations] for q in iprs_quench.keys()})
        data_stat.update({f'participation/quench/{q}'               : par_ent_quench[q][:completed_realizations] for q in par_ent_quench.keys()})
        data_stat.update({f'iprs/{q}'                               : iprs[q][:completed_realizations] for q in iprs.keys()})
        data_stat.update({f'entropy/eigenbasis/participation/{q}'   : entros[q][:completed_realizations] for q in entros.keys()})

        #! statistical
        HDF5Handler.save_hdf5(directory =   data_dir_in, 
                            filename    =   f'stat_{rand_num}.h5',
                            data        =   data_stat)
        data_hist = {
            'historgram/edges'          : histograms_av[name][alpha][ns].bin_edges,
            'historgram/counts'         : histograms_av[name][alpha][ns].counts(),
            'completed_realizations'    : completed_realizations,
        }
        for name in operators.keys():
            data_hist.update({
                f'historgram/{name}/average/values' : histograms_av[name][alpha][ns].averages_av(),
                f'historgram/{name}/typical/values' : histograms_av[name][alpha][ns].averages_av(is_typical=True),
            })
        HDF5Handler.save_hdf5(directory =   data_dir_in,
                            filename    =   f'hist_{rand_num}.h5', 
                            data        =   data_hist)
        #! operator
        data_op = { op : diag[:completed_realizations] for op, diag in diagonals_operators.items() }
        data_op.update({ 'completed_realizations' : completed_realizations })
        HDF5Handler.save_hdf5(directory =   data_dir_in,
                            filename    =   f'diag_{rand_num}.h5', 
                            data        =   data_op)
        
        #! time evolution
        data_time = {
            'time'                          : time_steps,
            'time_evolution/quench/energy'  : quench_energies[:completed_realizations],
            'completed_realizations'        : completed_realizations,
        }
        data_time.update({
            f'time_evolution/{name}/expectation' : time_vals[name] for name in operators.keys()
        })
        data_time.update({
            f'time_evolution/{name}/diag_ens' : diagonal_ensembles[name] for name in operators.keys()
        })
        HDF5Handler.save_hdf5(directory =   data_dir_in,
                            filename    =   f'evo_{rand_num}.h5', 
                            data        =   data_time)
    logger.info(f"ns = {ns}, alpha = {alpha:.2f} done in {time.perf_counter() - time_start_a:.2f} s", lvl=2, color='green')

#! -------------------------------------------------------

def prepare_evolution(
                data_dir        : Directories,
                sites, 
                alphas,
                n_realisations  : dict, 
                time_num        : int, 
                operators_map   : dict,
                n               : int           = 1,
                uniform         : bool          = True,
                n_random        : int           = 0,
                bw_df           : pd.DataFrame  = None,
                mls_df          : pd.DataFrame  = None,
                start_time      : float         = None,
                job_time        : float         = None,
                seed            : int           = None) -> tuple:
    """
    Prepares and allocates data structures for quantum evolution simulations over multiple system sizes and disorder strengths.
    This function initializes and allocates arrays and dictionaries required for simulating quantum evolution, 
    including observables, energies, bandwidths, and histograms. It loops over the specified system sizes (`sites`) 
    and disorder strengths (`alphas`), and for each combination, prepares the necessary data for further computation 
    by calling the `_single_alpha` function.
    Parameters
    ----------
    data_dir : Directories
        Object or structure containing paths to data directories.
    sites : iterable
        List or array of system sizes (number of sites) to simulate.
    alphas : iterable
        List or array of disorder strengths (alpha values) to consider.
    n_realisations : dict
        Dictionary mapping system size to the number of disorder realisations to simulate.
    time_num : int
        Number of time points for time evolution.
    operators_map : dict
        Dictionary mapping operator names to operator constructor functions.
    n : int, optional
        Parameter for the simulation, default is 1.
    uniform : bool, optional
        Whether to use uniform distribution for disorder, default is True.
    n_random : int, optional
        Number of random samples to use, default is 0.
    bw_df : pd.DataFrame, optional
        DataFrame containing precomputed bandwidth data, indexed by alpha and system size.
    mls_df : pd.DataFrame, optional
        DataFrame containing precomputed mean level spacing data, indexed by alpha and system size.
    start_time : float, optional
        Start time of the job, for logging or timing purposes.
    job_time : float, optional
        Maximum allowed job time, for resource management.
    seed : int, optional
        Random seed for reproducibility.
    Returns
    -------
    tuple
        A tuple containing:
            - histograms_av: dict
                Nested dictionary of averaged histograms for each operator and alpha.
            - histograms_typ: dict
                Nested dictionary of typical histograms for each operator and alpha.
            - bandwidths: np.ndarray
                Array of bandwidths for each realisation.
    Notes
    -----
    This function is intended to be used as a preparatory step before running quantum evolution simulations.
    It does not perform the actual evolution but sets up all necessary data structures and calls `_single_alpha`
    for each (system size, alpha) pair.
    """
    
    #! allocate the data
    max_ns              = sites[-1]
    max_nh              = 2**(max_ns)
    max_allocated       = max_nh * (max_nh - 1) // 2
    vals_allocated      = np.zeros(max_allocated, dtype=np.float64)
    omegas_allocated    = np.zeros(max_allocated, dtype=np.float64)

    histograms_av       = { op: {alpha: {} for alpha in alphas} for op in operators_map }
    histograms_typ      = { op: {alpha: {} for alpha in alphas} for op in operators_map }
    
    #! loop over the sites
    for ins, ns in enumerate(sites):
        logger.info(f"ns = {ns}", lvl=1, color='blue')
        time_start_ns       = time.perf_counter()
        hilbert_size        = 2**ns
        n_realisations_in   = n_realisations[ns]
        
        # operators_map[sig_z_l(ns).name] = sig_z_l
        operators           = {k : v(ns) for k, v in operators_map.items()}
        operators_mat       = {k : op.matrix(dim=hilbert_size) for k, op in operators.items()}
        
        #! energy and bandwidth
        ldos                = np.zeros(shape=(n_realisations_in, hilbert_size), dtype=np.float64)
        iprs                = {q : np.zeros(shape=(n_realisations_in, hilbert_size), dtype=np.float64) for q in [2.0, 0.5]          }
        entros              = {q : np.zeros(shape=(n_realisations_in, hilbert_size), dtype=np.float64) for q in [1.0]               }

        energies            = np.zeros(shape=(n_realisations_in, hilbert_size), dtype=np.float64)
        diagonals_operators = { k : np.zeros(shape=(n_realisations_in, hilbert_size), dtype=np.float64) for k in operators.keys()   }
        bandwidths          = np.zeros(shape=(n_realisations_in,), dtype=np.float64)
        sigma_es            = np.zeros(shape=(n_realisations_in,), dtype=np.float64)
        
        time_vals           = { k : np.zeros(shape=(n_realisations_in, time_num), dtype=np.float64) for k in operators.keys()       }
        diagonal_ensembles  = { k : np.zeros(shape=(n_realisations_in,), dtype=np.float64) for k in operators.keys()                }
        logger.info(f"ns = {ns} allocation done in {time.perf_counter() - time_start_ns:.2f} s", lvl=1, color='blue')
        
        #! go through the alphas
        for ia, alpha in enumerate(alphas):
            try:
                mean_lvl_spacing    = mls_df.loc[f'{alpha:.2f}', ns]
            except KeyError:
                logger.error(f"Mean level spacing not found for ns = {ns}, alpha = {alpha:.2f}")
                mean_lvl_spacing    = 1 / hilbert_size * 2.0 * np.pi
            try:
                bandwidth_data      = bw_df.loc[f'{alpha:.2f}', ns]
            except KeyError:
                logger.error(f"Bandwidth not found for ns = {ns}, alpha = {alpha:.2f}")
                bandwidth_data      = (1 + (alpha**2 * (1 - alpha**(2 * ns))) / (1 - alpha**2))
                
            _single_alpha(
                alpha                   = alpha,
                ns                      = ns,
                n                       = n,
                time_num                = time_num,
                uniform                 = uniform,
                data_dir                = data_dir,
                rand_num                = n_random,
                n_realisations          = n_realisations_in,
                mean_lvl_spacing        = mean_lvl_spacing,
                bandwidth_data          = bandwidth_data,
                operators               = operators,
                # allocated data
                vals_allocated          = vals_allocated,
                omegas_allocated        = omegas_allocated,
                sigma_es                = sigma_es,
                bandwidths              = bandwidths,
                energies                = energies,
                ldos                    = ldos,
                iprs                    = iprs,
                entros                  = entros,
                time_vals               = time_vals,
                diagonals_operators     = diagonals_operators,
                diagonal_ensembles      = diagonal_ensembles,
                histograms_av           = histograms_av,
                histograms_typ          = histograms_typ,
                operators_mat           = operators_mat,
                start_time              = start_time,
                job_time                = job_time,
                seed                    = seed
            )
            
    return histograms_av, histograms_typ, bandwidths

#! -------------------------------------------------------

def make_sig_z_global(ns):
    return op_spin.sig_z(
        ns          = ns,
        type_act    = op_spin.OperatorTypeActing.Global,
        sites       = [0])

#! -------------------------------------------------------

def run_parallel_evolution(alphas_chunks, base_dir, sites, n_reals, time_num, operators_map, n, rand_num, bw_df, mls_df, start_time, remaining_time, max_workers):
    """Run evolution in parallel with proper error handling"""
    
    if max_workers == 1:
        logger.info("Running in single-threaded mode")
        return prepare_evolution(
            data_dir        = base_dir,
            sites           = sites,
            alphas          = alphas_chunks[0],  # All alphas in single chunk
            n_realisations  = n_reals,
            time_num        = time_num,
            operators_map   = operators_map,
            n               = n,
            uniform         = True,
            n_random        = rand_num,
            bw_df           = bw_df,
            mls_df          = mls_df,
            start_time      = start_time,
            job_time        = remaining_time,
        )
    
    logger.info(f"Using {max_workers} workers for parallel processing")
    results = []
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(
                    prepare_evolution,
                    data_dir        = base_dir,
                    sites           = sites,
                    alphas          = chunk,
                    n_realisations  = n_reals,
                    time_num        = time_num,
                    operators_map   = operators_map,
                    n               = n,
                    uniform         = True,
                    n_random        = rand_num,
                    bw_df           = bw_df,
                    mls_df          = mls_df,
                    start_time      = start_time,
                    job_time        = remaining_time,
                ): chunk for chunk in alphas_chunks
            }
            
            # Collect results
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Chunk {chunk} completed successfully")
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk}: {e}")
                    # Continue with other chunks instead of failing completely
                    continue
    
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        # Fallback to single-threaded processing
        logger.info("Falling back to single-threaded processing")
        return run_parallel_evolution(
            [np.concatenate(alphas_chunks)], base_dir, sites, n_reals, time_num,
            operators_map, n, rand_num, bw_df, mls_df, start_time, remaining_time, 1
        )
    
    if not results:
        raise RuntimeError("No results obtained from parallel processing")
    
    # If multiple results, you might need to combine them
    # This depends on what prepare_evolution returns
    return results[0] if len(results) == 1 else results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel time-evolution statistics')
    parser.add_argument('save_dir',                         type    =   str,                            help    =   'Directory to save data')
    parser.add_argument('alpha_start',                      type    =   float,                          help    =   'Initial alpha value')
    parser.add_argument('alpha_step',                       type    =   float,                          help    =   'Step for alpha')
    parser.add_argument('alphas_number',                    type    =   int,                            help    =   'Number of alpha steps')
    parser.add_argument('number_of_realizations',           type    =   int,    default =   10,         help    =   'Realizations per ns')
    parser.add_argument('sites_start',                      type    =   int,                            help    =   'Minimum number of spins')
    parser.add_argument('sites_end',                        type    =   int,                            help    =   'Maximum number of spins (inclusive)')
    parser.add_argument('n',                                type    =   int,    default =   1,          help    =   'Model parameter n')
    parser.add_argument('time_num',                         type    =   int,    default =   int(1e5),   help    =   'Number of time points')
    parser.add_argument('memory_per_worker',                type    =   float,  default =   2.0,        help    =   'Memory reserved per worker in GB')
    parser.add_argument("max_memory",                       type    =   float,  default =   80.0,       help    =   'Maximum memory in GB')
    parser.add_argument('-S',        '--seed',              type    =   int,    default =   None,       help    =   'Random seed for reproducibility')
    parser.add_argument('-c',        '--max_cores',         type    =   int,    default =   psutil.cpu_count(), help='Maximum number of cores to use')
    parser.add_argument('-f',        '--force_single_thread',                   action  =   'store_true',       help='Force single-threaded execution')
    args = parser.parse_args()
    
    #! -------------------------------------------------------
    rng                 = np.random.default_rng(seed=args.seed)
    rand_num            = rng.integers(0, int(1e5))
    bw_df               = pd.read_csv("./model/bw.csv", index_col=0, header=None, dtype=float)
    mls_df              = pd.read_csv("./model/mls.csv", index_col=0, header=None, dtype=float)
    bw_df.index         = [f'{x:.2f}' for x in bw_df.index]
    mls_df.index        = [f'{x:.2f}' for x in mls_df.index]
    bw_df.columns       = list(range(7, 17))
    mls_df.columns      = list(range(7, 16))
    logger.info(f"Bandwidths:\n{bw_df}", lvl=2, color='blue')
    logger.info(f"Mean level spacing:\n{mls_df}", lvl=2, color='blue')
    logger.breakline(1)
    #! -------------------------------------------------------    
    base_dir            = Directories(args.save_dir)
    base_dir.mkdir()
    
    #! process the arguments
    n               = args.n
    alphas          = np.arange(args.alpha_start,
                                args.alpha_start + args.alpha_step * args.alphas_number,
                                args.alpha_step)
    sites           = list(range(args.sites_start, args.sites_end + 1))
    operators_map   = { op_spin.sig_z(ns=0, type_act=op_spin.OperatorTypeActing.Global, sites=[0]).name: make_sig_z_global }
    n_reals         = {ns: args.number_of_realizations for ns in sites}
    time_num        = args.time_num

    #! -------------------------------------------------------

    logger.info(f"Number of realisations: {n_reals}")
    logger.info(f"Number of sites: {len(sites)}")
    logger.info(f"Alphas: {alphas}")
    logger.info(f"Sites: {sites}")
    logger.info(f"Operators: {operators_map.keys()}")
    logger.info(f"Time steps: {time_num}")

    #! -------------------------------------------------------
    avail_gb            = min(psutil.virtual_memory().available / (1024**3), args.max_memory)
    memory_per_worker   = max(args.memory_per_worker, 1.0)
    memory_per_worker   = min(avail_gb, max(1.0, max(ManyBodyEstimator.estimate_matrix_memory(Ns = sites[-1]), memory_per_worker))) # minimum 1 GB

    try:
        estimated_memory    = ManyBodyEstimator.estimate_matrix_memory(Ns=sites[-1])
        memory_per_worker   = min(avail_gb, max(1.0, max(estimated_memory, memory_per_worker)))
    except Exception as e:
        logger.warning(f"Could not estimate memory requirements: {e}")
        logger.info("Using default memory per worker")

    if args.force_single_thread:
        max_workers = 1
    else:
        max_workers         = calculate_optimal_workers(alphas, avail_gb, memory_per_worker, args.max_cores)
    
    logger.info(f"Available memory: {avail_gb:.2f} GB")
    logger.info(f"Memory per worker: {memory_per_worker:.2f} GB")
    logger.info(f"Max workers: {max_workers}")
    #! -------------------------------------------------------
    
    # split alphas into chunks
    n_alphas_per_worker = len(alphas) // max_workers
    alphas_chunks       = np.array_split(alphas, max_workers)
    logger.info(f"Alphas chunks: {alphas_chunks}")
    #! -------------------------------------------------------
    
    # Split alphas into chunks
    try:
        alphas_chunks   = np.array_split(alphas, max_workers)
        logger.info(f"Alpha chunks: {[len(chunk) for chunk in alphas_chunks]}")

        # Get remaining time
        start_time      = time.perf_counter()
        remaining_time  = SlurmMonitor.get_remaining_time()
        if remaining_time == -1:
            remaining_time = 60 * 60 * 24 * 2  # default to 2 days
        logger.info(f"Remaining time: {remaining_time:.2f} s ({remaining_time/3600:.2f} hours)", color='green')

        # Run evolution
        results = run_parallel_evolution(alphas_chunks, base_dir, sites, n_reals, time_num,
            operators_map, n, rand_num, bw_df, mls_df, start_time, remaining_time, max_workers)

        logger.info(f"All computations completed in {time.perf_counter() - start_time:.2f} s", color='green')

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

