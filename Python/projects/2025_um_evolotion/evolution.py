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
import pandas as pd
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

#! -------------------------------------------------------
os.environ['BACKEND'] = 'numpy'
#! -------------------------------------------------------

script_dir = os.path.dirname(os.curdir)
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# project imports
from QES.general_python.common.binary import get_global_logger
from QES.general_python.common import Directories
from QES.general_python.common import HDF5Handler
from QES.general_python.maths.statistics import HistogramAverage
from QES.general_python.physics import entropy
from QES.Algebra.Model.Interacting.Spin.ultrametric import UltrametricModel
from QES.general_python.algebra.linalg import overlap
from QES.Algebra.Properties import time_evo, statistical
import QES.Algebra.Operator.operators_spin as op_spin

logger      = get_global_logger()

# ------------------------------------------------------------------
#! Read the parameters

def _single_realisation(
        model                   : UltrametricModel,
        r                       : int,
        sigma_es                : np.ndarray,
        bandwidths              : np.ndarray,
        energies                : np.ndarray,
        ldos                    : np.ndarray,
        iprs                    : dict,
        entros                  : dict,
        quench_energies         : np.ndarray,
        time_vals               : dict,
        time_steps              : np.ndarray,
        operators               : dict,
        operators_mat           : dict,
        omegas_allocated        : np.ndarray,
        vals_allocated          : np.ndarray,
        h_av                    : dict,
        h_typ                   : dict,
        diagonals_operators     : dict,
        diagonal_ensembles      : dict,
        ):
    
    #! constants
    alpha           = model.alphas[0]
    ns              = model.ns
    hilbert_size    = 2**ns
    vn_num          = min(500, hilbert_size)
    qs_ipr          = { 2.0, 0.5 }
    qs_entro        = { 1.0 }
    batch_limit     = lambda ns: ns > 13
    batch_num       = lambda ns: 100 if batch_limit(ns) else 1
    
    time_start_r    = time.perf_counter()
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r}", lvl=3, color='red')

    #! build & diagonalize
    # model.clear()
    model.build()
    model.diagonalize(verbose=True)
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} diag done in {time.perf_counter() - time_start_r:.2f} s", lvl=3, color='red')
    
    #! compute the bandwidth
    time_start_stat = time.perf_counter()
    mean_energy     = model.av_en
    idx_mean        = int(np.argmin(np.abs(model.eig_val - mean_energy)))
    idx_mn, idx_mx  = int(idx_mean - vn_num // 2), int(idx_mean + vn_num // 2)
    idx_mn, idx_mx  = np.clip(idx_mn, 0, hilbert_size), np.clip(idx_mx, 0, hilbert_size)
    bandwidth       = model.get_bandwidth()
    sigma_e         = model.get_energywidth()
    bandwidths[r]   = bandwidth
    sigma_es[r]     = sigma_e
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} bandwidth = {bandwidth:.2f} done in {time.perf_counter() - time_start_stat:.2f} s", lvl=3, color='red')
    
    #! IPR
    time_start_ipr = time.perf_counter()
    for q in qs_ipr:
        iprs[q][r, :] = statistical.inverse_participation_ratio(model.eig_vec, q=q)
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} IPR done in {time.perf_counter() - time_start_ipr:.2f} s", lvl=3, color='red')

    #! participation entropy
    time_start_entro = time.perf_counter()
    for q in qs_entro:
        entros[q][r, :] = entropy.entropy(model.eig_vec, q=q, typek=entropy.Entanglement.PARTIC)
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} entropy done in {time.perf_counter() - time_start_entro:.2f} s", lvl=3, color='red')

    #! von Neumann entropy
    # von_neumann[r, :]   = entropy.entropy(model.eig_vec[:, idx_mn:idx_mx], q=1.0, typek=entropy.Entanglement.VN)
    #! Tsallis entropy
    # tsalis[r, :]        = entropy.entropy(model.eig_vec[:, idx_mn:idx_mx], q=2.0, typek=entropy.Entanglement.TSALLIS)
    
    #! select the quench state
    quench_state            = np.zeros(hilbert_size, dtype=np.float64)
    diff                    = np.abs(model.hamil.diagonal() - mean_energy)
    idx                     = int(np.argmin(diff))  # find the closest eigenvalue
    quench_state[idx]       = 1.0                   # set the quench state
    quench_energies[r]      = overlap(quench_state, model.hamil)
    overlaps                = model.eig_vec.T @ quench_state
    soverlaps               = np.square(np.abs(overlaps))
    
    if not batch_limit(ns):
        quench_states_t     = time_evo.time_evo_block(model.eig_vec, model.eig_val, quench_overlaps=overlaps, times=time_steps)
    else:
        quench_states_t     = None
        
    #! compute the ldos
    ldos[r, :]              = statistical.ldos(energies = model.eig_val, overlaps = overlaps)
    energies[r, :]          = model.eig_val
        
    #! accumulate into the single histogram
    for name in operators.keys():
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
        logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} histogram done in {time.perf_counter() - time_start_hist:.2f} s", lvl=3, color='blue')
        
        #! remove zeros and nan
        omegas                  = omegas_allocated[:cnt]
        vals                    = vals_allocated[:cnt]
        mask                    = np.logical_and(vals > 1e-10, np.isfinite(vals))
        omegas                  = omegas[mask]
        vals                    = vals[mask]
        h_av[name].append(omegas, vals)
        h_typ[name].append(omegas, np.log(vals))
        
        #! time evolution
        time_start_evo = time.perf_counter()
        for i in range(batch_num(ns)):
            start                           = i * time_num // batch_num(ns)
            stop                            = (i + 1) * time_num // batch_num(ns)
            if quench_states_t is None:
                quench_states_t = time_evo.time_evo_block(model.eig_vec, model.eig_val, quench_overlaps=overlaps, times=time_steps[start:stop])
            quenched_values_t               = time_evo.time_evo_evaluate(quench_states_t, operators_mat[name])
            time_vals[name][r, start:stop]  = np.real(quenched_values_t)
            
        # time_vals[name][r, :] = np.real(quenched_values_t)
        logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} time evolution done in {time.perf_counter() - time_start_evo:.2f} s", lvl=3, color='blue')
    logger.info(f"ns = {ns}, alpha = {alpha:.2f}, r = {r} done in {time.perf_counter() - time_start_r:.2f} s", lvl=4, color='red')

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
                ) -> None:

    logger.info(f"ns = {ns}, alpha = {alpha:.2f}", lvl=2, color='green')
    time_start_a        = time.perf_counter()
    hilbert_size        = 2**ns
    vn_num              = min(500, hilbert_size)
    qs_ipr              = { 2.0, 0.5 }
    qs_entro            = { 1.0 }
    
    #! constants
    # heisenberg_time     = 1.0 / mls_df.loc[f'{alpha:.2f}', ns] * 2.0 * np.pi
    # bandwidth_data      = bw_df.loc[f'{alpha:.2f}', ns]
    
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
    o_max               = 3 * bandwidth_data
    o_min               = 0.01 / hilbert_size
    n_bins              = int(20 * ns)
    edges               = HistogramAverage.uniform_log(n_bins, v_max=o_max, v_min=o_min)
    for name in operators.keys():
        h_av    = { k : HistogramAverage(n_bins, edges=edges) for k in operators.keys() }
        h_typ   = { k : HistogramAverage(n_bins, edges=edges) for k in operators.keys() }

    #! model
    model               = UltrametricModel(
                                ns      = ns,
                                n       = n,
                                alphas  = alpha,
                                backend = np,
                                seed    = None)
    
    #! directories
    data_dir_in = data_dir.join('uniform' if uniform else 'log', f'{str(model)}')
    data_dir_in.mkdir()
    
    #! go through the realisations
    for r in range(n_realisations):
        _single_realisation(
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
            diagonal_ensembles      = diagonal_ensembles)
    
    #! store the combined histogram for this (ns,alpha)
    for name in operators:
        histograms_av[name][alpha][ns]  = h_av[name]
        histograms_typ[name][alpha][ns] = h_typ[name]
    
    #! save the data
    if True:
        data_stat = {
            'bandwidth'                 : bandwidths,
            'sigma_e'                   : sigma_es,
            'energies'                  : energies,
            'ldos'                      : ldos,
            # 'entropy/von_neumann/1.0'   : von_neumann,
            # 'entropy/tsallis/1.0'       : tsalis,
        }
        data_stat.update({f'iprs/{q}' : iprs[q] for q in qs_ipr})
        data_stat.update({f'entropy/participation/{q}' : entros[q] for q in qs_entro})
        
        #! statistical
        HDF5Handler.save_hdf5(directory =   data_dir_in, 
                            filename    =   f'stat_{rand_num}.h5',
                            data        =   data_stat)
        data_hist = {
            'historgram/edges'  : histograms_av[name][alpha][ns].bin_edges,
            'historgram/counts' : histograms_av[name][alpha][ns].counts(),
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
        data_op = { op : diag for op, diag in diagonals_operators.items() }
        HDF5Handler.save_hdf5(directory =   data_dir_in,
                            filename    =   f'diag_{rand_num}.h5', 
                            data        =   data_op)
        
        #! time evolution
        data_time = {
            'time'                          : time_steps,
            'time_evolution/quench/energy'  : quench_energies
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
                n               : int   = 1,
                uniform         : bool  = True,
                n_random        : int   = 0,
                bw_df           : pd.DataFrame = None,
                mls_df          : pd.DataFrame = None,
                ) -> tuple:
    
    #! allocate the data
    max_ns              = sites[-1]
    max_nh              = 2**(max_ns)
    max_allocated       = max_nh * (max_nh - 1) // 2
    vals_allocated      = np.zeros(max_allocated, dtype=np.float64)
    omegas_allocated    = np.zeros(max_allocated, dtype=np.float64)
    
    qs_ipr              = { 2.0, 0.5 }
    qs_entro            = { 1.0 }
    batch_limit         = lambda ns: ns > 13

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
        iprs                = {q : np.zeros(shape=(n_realisations_in, hilbert_size), dtype=np.float64) for q in qs_ipr              }
        entros              = {q : np.zeros(shape=(n_realisations_in, hilbert_size), dtype=np.float64) for q in qs_entro            }
        # von_neumann         = np.zeros(shape=(n_realisations_in, vn_num), dtype=np.float64)
        # tsalis              = np.zeros(shape=(n_realisations_in, tsalis_num), dtype=np.float64)
        
        energies            = np.zeros(shape=(n_realisations_in, hilbert_size), dtype=np.float64)
        diagonals_operators = { k : np.zeros(shape=(n_realisations_in, hilbert_size), dtype=np.float64) for k in operators.keys()   }
        bandwidths          = np.zeros(shape=(n_realisations_in,), dtype=np.float64)
        sigma_es            = np.zeros(shape=(n_realisations_in,), dtype=np.float64)
        
        time_vals           = { k : np.zeros(shape=(n_realisations_in, time_num), dtype=np.float64) for k in operators.keys()       }
        diagonal_ensembles  = { k : np.zeros(shape=(n_realisations_in,), dtype=np.float64) for k in operators.keys()                }
        idx                 = 0
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
            )
            
    return histograms_av, histograms_typ, bandwidths

#! -------------------------------------------------------

def make_sig_z_global(ns):
    return op_spin.sig_z(
        ns          = ns,
        type_act    = op_spin.OperatorTypeActing.Global,
        sites       = [0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel time-evolution statistics')
    parser.add_argument('save_dir',                                                             help    =   'Directory to save data')
    parser.add_argument('alpha_start',              type    =   float,                          help    =   'Initial alpha value')
    parser.add_argument('alpha_step',               type    =   float,                          help    =   'Step for alpha')
    parser.add_argument('alphas_number',            type    =   int,                            help    =   'Number of alpha steps')
    parser.add_argument('number_of_realizations',   type    =   int,    default =   10,         help    =   'Realizations per ns')
    parser.add_argument('sites_start',              type    =   int,                            help    =   'Minimum number of spins')
    parser.add_argument('sites_end',                type    =   int,                            help    =   'Maximum number of spins (inclusive)')
    parser.add_argument('n',                        type    =   int,    default =   1,          help    =   'Model parameter n')
    parser.add_argument('time_num',               type    =   int,    default =   int(1e5),     help    =   'Number of time points')
    parser.add_argument('memory_per_worker',      type    =   float,  default =   2.0,          help    =   'Memory reserved per worker in GB')
    args = parser.parse_args()
    
    
    #! -------------------------------------------------------
    rng                 = np.random.default_rng()
    rand_num            = rng.integers(0, int(1e5))
    bw_df               = pd.read_csv("./bw.csv", index_col=0, header=None, dtype=float)
    mls_df              = pd.read_csv("./mls.csv", index_col=0, header=None, dtype=float)
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

    #! -------------------------------------------------------
    memory_per_worker   = args.memory_per_worker
    avail_gb            = psutil.virtual_memory().available / (1024**3)
    max_workers         = max_workers= max(1, min(len(alphas), int(avail_gb / memory_per_worker)))
    # max_workers         = 1
    logger.info(f"Available memory: {avail_gb:.2f} GB")
    logger.info(f"Memory per worker: {memory_per_worker:.2f} GB")
    logger.info(f"Max workers: {max_workers}")
    #! -------------------------------------------------------
    
    # split alphas into chunks
    n_alphas_per_worker = len(alphas) // max_workers
    alphas_chunks       = [alphas[i:i + n_alphas_per_worker] for i in range(0, len(alphas), n_alphas_per_worker)]
    alphas_chunks       = [chunk for chunk in alphas_chunks if len(chunk) > 0]
    logger.info(f"Alphas chunks: {alphas_chunks}")
    #! -------------------------------------------------------
    
    if max_workers > 1:
        logger.info(f"Using {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
                    mls_df          = mls_df
                ): chunk for chunk in alphas_chunks
            }
            
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    histograms_av, histograms_typ, bandwidths = future.result()
                except Exception as e:
                    logger.error(f"Error chunk {chunk}: {e}")
                else:
                    logger.info(f"Chunk {chunk} done")
    else:
        histograms_av, histograms_typ, bandwidths = prepare_evolution(
            data_dir        = base_dir,
            sites           = sites,
            alphas          = alphas,
            n_realisations  = n_reals,
            time_num        = time_num,
            operators_map   = operators_map,
            n               = n,
            uniform         = True,
            n_random        = rand_num,
            bw_df           = bw_df,
            mls_df          = mls_df
        )
    logger.info(f"All done")
        
    #! -------------------------------------------------------

