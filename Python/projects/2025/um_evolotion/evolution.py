#!/usr/bin/env python3
import sys
import time
import psutil
import argparse
import numpy as np
import pandas as pd
import numpy as np
from typing import Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# project imports
try:
    from evolution_data import (
            EvolutionData, 
            HistogramAverage,
            Timer,
            Directories, 
            UltrametricModel, PowerLawRandomBanded, Hamiltonian,
            create_model,
            file_path
        )
    
    # log
    from QES.general_python.common.flog import get_global_logger
    
    # linear algebra methods
    from QES.general_python.algebra.linalg import overlap
    
    # runner
    from QES.general_python.run_scripts.slurm import SlurmMonitor, SimulationParams
    from QES.general_python.run_scripts import calculate_optimal_workers, calculate_realisations_per_parameter
    from QES.general_python.run_scripts.prepareini import ManyBodyEstimator

    # histograms etc.
    from QES.general_python.algebra.ran_wrapper import set_global_seed

    # VON NEUMANN, TSALLIS, PARTICIPATION ENTROPY
    from QES.general_python.physics import entropy as entropy, density_matrix as density_matrix
except ImportError as e:
    print("Error importing evolution_data or related modules:", e)

# ------------------------------------------------------------------

try:
    from QES.Algebra.Properties import statistical, time_evo
except ImportError as e:
    print("Error importing:", e)
    
try:
    import QES.Algebra.Operator.operators_spin as op_spin
except ImportError as e:
    print("Error importing operators_spin:", e)

logger = get_global_logger()

# ------------------------------------------------------------------
#! Read the parameters

#% checkers
batch_limit = lambda ns: ns >= 14
batch_num   = lambda ns: 10 if batch_limit(ns) else 1

# ------------------------------------------------------------------

def _single_realisation(model : Union[UltrametricModel, PowerLawRandomBanded], r : int, sim_params : SimulationParams = None, edata: EvolutionData = None) -> bool:

    if SlurmMonitor.is_overtime(limit=3600, start_time=sim_params.start_time, job_time=sim_params.job_time): # 20 minutes buffer
        return False
    
    #! constants
    time_start_r = time.perf_counter()
    logger.info(f"ns = {edata.ns}, alpha = {edata.alpha:.2f}, r = {r}", lvl=3, color='red')

    #! build & diagonalize
    with Timer(f"Build & Diagonalize: r = {r}, alpha = {edata.alpha:.2f}, ns = {edata.ns}", logger=logger, logger_args = {'lvl':3, 'color':'red'}):
        model.build()
        model.diagonalize(verbose=True)

    #! compute the bandwidth
    with Timer(f"Energy: r = {r}, alpha = {edata.alpha:.2f}, ns = {edata.ns}", logger=logger, logger_args = {'lvl':3, 'color':'red'}):
        edata.mean_energy       = model.av_en
        bandwidth               = model.get_bandwidth()
        sigma_e                 = model.get_energywidth()
        edata.bandwidths[r]     = bandwidth
        edata.sigma_es[r]       = sigma_e

    #! IPR
    with Timer(f"IPR: r = {r}, alpha = {edata.alpha:.2f}, ns = {edata.ns}", logger=logger, logger_args = {'lvl':3, 'color':'red'}):
        for q in edata.iprs.keys():
            edata.iprs[q][r, :] = statistical.inverse_participation_ratio(model.eig_vec, q=q)

    #! participation entropy
    with Timer(f"Entropy: r = {r}, alpha = {edata.alpha:.2f}, ns = {edata.ns}", logger=logger, logger_args = {'lvl':3, 'color':'red'}):
        for q in edata.entros.keys():
            edata.entros[q][r, :] = entropy.entropy(model.eig_vec, q=q, typek=entropy.Entanglement.PARTIC)

    #! select the quench state
    quench_state                = time_evo.create_initial_quench_state(time_evo.QuenchTypes.SEEK, edata.hilbert_size,
                                                            edata.ns, edata.mean_energy, model.hamil.diagonal(), backend=np)
    edata.quench_energies[r]    = overlap(quench_state, model.hamil)
    overlaps                    = model.eig_vec.T @ quench_state
    soverlaps                   = np.square(np.abs(overlaps))

    with Timer(f"Entropy: r = {r}, alpha = {edata.alpha:.2f}, ns = {edata.ns}", logger=logger, logger_args = {'lvl':3, 'color':'red'}):
        rho                         = density_matrix.rho_numba(overlaps, dimA=2, dimB=2**(edata.ns - 1))
        #! calculate the entropies for the initial state in the eigenbasis
        schmidt_val, _              = density_matrix.schmidt_numba(rho, 2, 2**(edata.ns - 1), eig = False)
        #! von Neumann entropy
        edata.vn_entropies[r]       = entropy.entropy(schmidt_val, q=1.0, typek=entropy.Entanglement.VN)
        #! Tsallis entropy
        edata.tsalis[r]             = entropy.entropy(schmidt_val, q=2.0, typek=entropy.Entanglement.TSALLIS)
        #! Schmidt gaps
        edata.schmidt_gaps[r]       = schmidt_val[1] - schmidt_val[0] if len(schmidt_val) > 1 else 0.0
        #! Iprs for the quench state
        for q in edata.iprs_quench.keys():
            edata.iprs_quench[q][r] = statistical.inverse_participation_ratio(overlaps[np.newaxis, :], q=q)[0]
        for q in edata.par_ent_quench.keys():
            edata.par_ent_quench[q][r] = entropy.entropy(overlaps[np.newaxis, :], q=q, typek=entropy.Entanglement.PARTIC)[0]
    
    #! compute the ldos
    edata.ldos[r, :]            = statistical.ldos(energies = model.eig_val, overlaps = overlaps)
    edata.energies[r, :]        = model.eig_val

    #! compute the indices for the histograms
    if edata.indices_omegas is None or edata.indices_omegas.shape[0] <= 0:
        capacity                = edata.hilbert_size # idx_i, idx_j_start, idx_j_end
        edata.indices_omegas    = np.empty((capacity, 3), dtype=np.int64)
        
    with Timer(f"Histogram indices: r = {r}, alpha = {edata.alpha:.2f}, ns = {edata.ns}, nh = {edata.hilbert_size}", logger=logger, logger_args = {'lvl':3, 'color':'red'}):
        edata.indices_omegas, cnt = statistical.extract_indices_window(
            start           = 0,
            stop            = edata.hilbert_size,
            eigvals         = model.eig_val,
            bw              = bandwidth,
            energy_diff_cut = 0.015,
            energy_target   = edata.mean_energy,
            whole_spectrum  = False
        )
        indices_local = edata.indices_omegas[:cnt]

    #! accumulate into the single histogram
    for name in edata.operators.keys():

        with Timer(f"{name} Matrix Elements", logger=logger, logger_args = {'lvl':4, 'color':'blue'}):
            matrix_elements                         = overlap(model.eig_vec, edata.operators_mat[name])
            edata.diagonals_operators[name][r, :]   = matrix_elements.diagonal()
            edata.diagonal_ensembles[name][r]       = time_evo.diagonal_ensemble(soverlaps=soverlaps, diag_mat=matrix_elements.diagonal())

        #! F-Function
        with Timer(f"{name} F-Function", logger=logger, logger_args = {'lvl':4, 'color':'blue'}):
            _, counts, sums = statistical.f_function(overlaps=matrix_elements, eigvals=model.eig_val, indices_alloc=indices_local, bins=edata.edges, uniform_log_bins=True)
            edata.h_av[name].add(sums, counts)
            _, counts, sums = statistical.f_function(overlaps=matrix_elements, eigvals=model.eig_val, indices_alloc=indices_local, bins=edata.edges, typical=True, uniform_log_bins=True)
            edata.h_typ[name].add(sums, counts)
            # full
            _, counts, sums = statistical.f_function(overlaps=matrix_elements, eigvals=model.eig_val, indices_alloc=None, bins=edata.edges, uniform_log_bins=True)
            edata.h_av_full[name].add(sums, counts)
            
        #! K-Function
        with Timer(f"{name} K-Function", logger=logger, logger_args = {'lvl':4, 'color':'blue'}):
            _, counts, sums = statistical.k_function(ldos=edata.ldos[r], eigvals=model.eig_val, 
                                                    indices_alloc=indices_local, bins=edata.edges, uniform_log_bins=True)
            edata.k_functions.add(sums, counts)
            _, counts, sums = statistical.k_function(ldos=edata.ldos[r], eigvals=model.eig_val, 
                                                    indices_alloc=None, bins=edata.edges, uniform_log_bins=True)
            edata.k_functions_full.add(sums, counts)

        #! S-Function
        with Timer(f"{name} S-Function", logger=logger, logger_args = {'lvl':4, 'color':'blue'}):
            _, counts, sums = statistical.s_function(ldos=edata.ldos[r], overlaps=matrix_elements,
                                        eigvals=model.eig_val, indices_alloc=indices_local, bins=edata.edges, uniform_log_bins=True)
            edata.s_functions[name].add(sums, counts)
            _, counts, sums = statistical.s_function(ldos=edata.ldos[r], overlaps=matrix_elements,
                                        eigvals=model.eig_val, indices_alloc=None, bins=edata.edges, uniform_log_bins=True)
            edata.s_functions_full[name].add(sums, counts)

        #! fidelity susceptibility
        if name in edata.fidelity_susc_av:
            edata.fidelity_susc_av[name][r, :] = statistical.fidelity_susceptibility(
                energies  = model.eig_val,
                V         = matrix_elements,
                mu        = bandwidth / edata.hilbert_size
            )
            edata.fidelity_susc_typ[name][r, :] = statistical.fidelity_susceptibility(
                energies  = model.eig_val,
                V         = matrix_elements,
                mu        = 1e-10
            )
    
    #! time evolution - compute the expectation values
    if not batch_limit(edata.ns):
        logger.info(f"ns = {edata.ns}, alpha = {edata.alpha:.2f}, r = {r} computing time evolution for operator {name} in one go", lvl=4, color='blue')
        # Small systems: compute all at once
        with Timer(f"Time Evolution: overlaps", logger=logger, logger_args = {'lvl':5, 'color':'blue'}):
            evolved_overlaps   = np.exp(-1j * np.outer(model.eig_val, edata.time_steps)) * overlaps[:, np.newaxis]
            quench_states_t    = model.eig_vec @ evolved_overlaps
            # quench_states_t     = time_evo.time_evo_block_optimized(eig_vec=model.eig_vec, eig_val=model.eig_val, overlaps=overlaps, time_steps=edata.time_steps)

        with Timer(f"Survival Probability", logger=logger, logger_args = {'lvl':5, 'color':'blue'}):
            edata.survival_proba[r, :]  = statistical.survival_prob(psi0 = quench_state, psi_t = quench_states_t)

        for name in edata.operators.keys():
            with Timer(f"{name} Time Evolution", logger=logger, logger_args = {'lvl':5, 'color':'blue'}):
                quenched_values_t               = np.einsum('ij,ji->i', np.conj(quench_states_t.T), edata.operators_mat[name] @ quench_states_t)
                edata.time_vals[name][r, :]     = np.real(quenched_values_t)

            if edata.uniform:
                with Timer(f"FFT", logger=logger, logger_args = {'lvl':6, 'color':'yellow'}):
                    # remove the diagonal ens
                    time_norm                       = 2 / np.sqrt(edata.time_num)
                    data_evo_av                     = edata.time_vals[name][r, :] - edata.diagonal_ensembles[name][r]
                    #! calculate FFT
                    data_evo_fft_base               = np.fft.rfft(data_evo_av) * time_norm
                    data_evo_fft                    = np.abs(data_evo_fft_base)**2 / edata.time_num**2
                    data_evo_fft_n                  = np.trapz(data_evo_fft, x=edata.fft_omegas, axis=0)
                    data_evo_fft                    = data_evo_fft / data_evo_fft_n
                    edata.fft_results[name][r, :]   = data_evo_fft[1:-1] # Exclude the zero frequency and Nyquist
                    edata.fft_n[name][r]            = data_evo_fft_n
    else:
        #! Large systems: batch processing
        batch_count             = batch_num(edata.ns)
        batch_size              = (edata.time_num + batch_count - 1) // batch_count
        logger.info(f"ns = {edata.ns}, alpha = {edata.alpha:.2f}, r = {r} computing time evolution for operator {name} in batches of size {batch_size}", lvl=4, color='blue')
        for i in range(batch_count):
            start_idx   = i * batch_size
            end_idx     = min(start_idx + batch_size, edata.time_num)
            
            if start_idx >= end_idx:
                break
                
            # Check timeout during batching
            if SlurmMonitor.is_overtime(limit=600):  # 10 minutes buffer
                logger.warning(f"SLURM timeout during batch {i}, stopping early")
                break
            
            with Timer(f"Time Evolution: batch {i+1}/{batch_count}", logger=logger, logger_args = {'lvl':4, 'color':'blue'}):
                # Compute time evolution for this batch
                batch_times             = edata.time_steps[start_idx:end_idx]
                evolved_overlaps_batch  = np.exp(-1j * np.outer(model.eig_val, batch_times)) * overlaps[:, np.newaxis]
                quench_states_batch     = model.eig_vec @ evolved_overlaps_batch
                for name in edata.operators.keys():
                    with Timer(f"{name} Time Evolution: batch {i+1}/{batch_count}", logger=logger, logger_args = {'lvl':5, 'color':'blue'}):
                        values_batch                                = np.einsum('ij,ji->i', np.conj(quench_states_batch.T), edata.operators_mat[name] @ quench_states_batch)
                        edata.time_vals[name][r, start_idx:end_idx] = np.real(values_batch)
                del quench_states_batch, evolved_overlaps_batch
    logger.info(f"ns = {edata.ns}, alpha = {edata.alpha:.2f}, r = {r} done in {time.perf_counter() - time_start_r:.2f} s", lvl=4, color='red')
    return True

#! -------------------------------------------------------

def _single_alpha(modelstr : 'str' = 'um', evolution_data : EvolutionData = None, sim_params : SimulationParams = None):
    
    time_start_a = time.perf_counter()
    logger.info(f"Starting: ns = {evolution_data.ns}, alpha = {evolution_data.alpha:.2f}, worker = {sim_params.worker_id}", lvl=3, color='orange')
    
    with Timer(f"Model creation: ns = {evolution_data.ns}, alpha = {evolution_data.alpha:.2f}", logger=logger, logger_args = {'lvl':3, 'color':'green'}):
        model = create_model(model=modelstr, ns=evolution_data.ns, param=evolution_data.alpha, seed=sim_params.seed, **evolution_data.additional_params)
    
    #! directories
    data_dir_in = sim_params.data_dir.join('uniform' if evolution_data.uniform else 'log', f'{str(model)}')
    data_dir_in.mkdir()
    
    completed_realizations = 0
    #! go through the realisations
    for r in range(evolution_data.realizations):
        if _single_realisation(model=model, r=r, sim_params=sim_params, edata=evolution_data):
            completed_realizations += 1
        else:
            logger.warning(f"Realization {r} failed or was skipped due to timeout")
            break
        
    # log me
    logger.info(f"Completed {completed_realizations}/{evolution_data.realizations} realizations for ns={evolution_data.ns}, alpha={evolution_data.alpha:.2f}.",
                lvl=2, color='green' if completed_realizations == evolution_data.realizations else 'yellow')
    
    #! save the data
    with Timer(f"Saving data: ns = {evolution_data.ns}, alpha = {evolution_data.alpha:.2f}", logger=logger, logger_args = {'lvl':3, 'color':'green'}):
        if completed_realizations > 0:
            evolution_data.save_all(directory=data_dir_in, rand_num=sim_params.rand_num, completed=completed_realizations)
        else:
            logger.warning(f"No completed realizations to save for ns={evolution_data.ns}, alpha={evolution_data.alpha:.2f}.")

    logger.info(f"ns = {evolution_data.ns}, alpha = {evolution_data.alpha:.2f} done in {time.perf_counter() - time_start_a:.2f} s", lvl=2, color='green')

#! -------------------------------------------------------

def _mean_lvl_and_bw(alpha: float, ns: int, hilbert_size: int):
    
    if ns not in mls_df.columns:
        if ns < mls_df.columns.min():
            mult = 2 ** (int(mls_df.columns.min())-ns)
            ns = mls_df.columns.min()
        else:
            mult = 2 ** (int(mls_df.columns.max())-ns)
            ns = mls_df.columns.max()
    else:
        mult = 1.0
        
    try:
        mean_lvl_keys       = np.array(mls_df.index, dtype=float)
        closest_idx         = np.abs(mean_lvl_keys - alpha).argmin()
        logger.info(f"Using params: alpha = {alpha:.2f}, alpha_idx = {mean_lvl_keys[closest_idx]:.2f}, ns = {ns}", lvl=2, color='blue')
        mean_lvl_spacing    = mls_df.iloc[closest_idx][ns] * mult
    except KeyError:
        logger.error(f"Mean level spacing not found for ns = {ns}, alpha = {alpha:.2f}")
        mean_lvl_spacing    = 1 / hilbert_size * 2.0 * np.pi
        
    #! bandwidth
    if ns not in bw_df.columns:
        if ns < bw_df.columns.min():
            ns = bw_df.columns.min()
        else:
            ns = bw_df.columns.max()
    try:
        bandwidth_keys      = np.array(bw_df.index, dtype=float)
        closest_idx         = np.abs(bandwidth_keys - alpha).argmin()
        bandwidth_data      = bw_df.iloc[closest_idx][ns]
    except KeyError:
        logger.error(f"Bandwidth not found for ns = {ns}, alpha = {alpha:.2f}")
        bandwidth_data      = (1 + (alpha**2 * (1 - alpha**(2 * ns))) / (1 - alpha**2))
    return mean_lvl_spacing, bandwidth_data

def prepare_evolution(
                sites, 
                alphas,
                n               : int               = 1,
                uniform         : bool              = True,
                n_realisations  : dict              = None,
                # job parameters
                sim_params      : SimulationParams  = None,
                modelstr        : str               = 'um') -> tuple:
    """
    """
    
    #! loop over the sites
    for ins, ns in enumerate(sites):
        logger.info(f"ns = {ns}", lvl=1, color='blue')
        n_realisations_in       = n_realisations[ns]
        hilbert_size            = 2**ns
        
        # operators_map[sig_z_l(ns).name] = sig_z_l
        operators               = {k : v(ns) for k, v in operators_map.items()}
        operators_mat           = {k : op.matrix(dim=hilbert_size) for k, op in operators.items()}
        evolution_data          = EvolutionData(
                                        ns                  = ns,
                                        alpha               = None,  # placeholder, will be set in the loop
                                        realizations        = n_realisations_in,
                                        hilbert_size        = hilbert_size,
                                        time_num            = time_num,
                                        # other
                                        mean_lvl_space      = None,  # placeholder, will be set in the loop
                                        bandwidth_data      = None,  # placeholder, will be set in the loop
                                        # operators
                                        operators           = operators,
                                        operators_mat       = operators_mat,
                                        # simulation flags
                                        uniform             = uniform,
                                        additional_params   = { 'n': n }
                                    )
        #! go through the alphas - sequentially
        for ia, alpha in enumerate(alphas):

            mean_lvl_spacing, bandwidth_data    = _mean_lvl_and_bw(alpha, ns, hilbert_size)
            evolution_data.alpha                = alpha
            evolution_data.mean_lvl_space       = mean_lvl_spacing
            evolution_data.bandwidth_data       = bandwidth_data
            evolution_data.initialize_histograms_and_time()

            sim_params_in                       = sim_params.copy()
            sim_params_in.worker_id             = ins * len(alphas) + ia  # Unique worker ID for each (ns, alpha)

            try:
                _single_alpha(evolution_data = evolution_data, modelstr = modelstr, sim_params = sim_params_in)
            except Exception as e:
                logger.error(f"Error occurred for ns = {ns}, alpha = {alpha:.2f}: {e}")
                continue
            
            # time left for the job
            if sim_params_in.start_time is not None and sim_params_in.job_time is not None:
                remaining_time = sim_params_in.job_time - (time.perf_counter() - sim_params_in.start_time)
                if remaining_time <= 0:
                    logger.warning(f"Job time exceeded for ns = {ns}, alpha = {alpha:.2f}. Stopping early.")
                    break
                logger.info(f"Remaining time for ns = {ns}, alpha = {alpha:.2f}: {remaining_time:.2f} seconds")

#! -------------------------------------------------------

def make_sig_z_global(ns):
    return op_spin.sig_z(
        ns          = ns,
        type_act    = op_spin.OperatorTypeActing.Global,
        sites       = [0])

def make_sig_x_global(ns):
    return op_spin.sig_x(
        ns          = ns,
        type_act    = op_spin.OperatorTypeActing.Global,
        sites       = [0, 5])

#! -------------------------------------------------------

def run_parallel_evolution(alphas_chunks, base_dir, sites, n_reals, 
                time_num, operators_map, n, rand_num, start_time, remaining_time, 
                max_workers, seed, modelstr, uniform=False):
    """Run evolution in parallel with proper error handling"""
        
    sim_params = SimulationParams(
        data_dir            = base_dir,
        seed                = time.time_ns() % (2**32 - 1) if seed is None else seed,  # Unique seed for single-threaded run
        rand_num            = rand_num,
        worker_id           = 0,  # Placeholder, will be set per job
        start_time          = start_time,
        job_time            = remaining_time)
    
    if max_workers == 1:
        logger.info("Running in single-threaded mode")
        
        return prepare_evolution(
            sites           = sites,
            alphas          = alphas_chunks[0],  # All alphas in single chunk
            # other parameters
            n               = n,
            uniform         = uniform,
            n_realisations  = n_reals,
            # simulation
            sim_params      = sim_params,
            modelstr        = modelstr,
        )
    
    logger.info(f"Using {max_workers} workers for parallel processing")
    results = []
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(
                    prepare_evolution,
                    sites           = sites,
                    alphas          = chunk,
                    # other parameters
                    n               = n,
                    uniform         = True,
                    n_realisations  = n_reals,
                    # simulation
                    sim_params      = sim_params,
                    modelstr        = modelstr,
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
    parser = argparse.ArgumentParser(description='Parallel time-evolution statistics', allow_abbrev=False)
    parser.add_argument('--save_dir',        type=str,   required=True, help='Directory to save data')
    parser.add_argument('--alpha_start',     type=float, required=True, help='Initial alpha value')
    parser.add_argument('--alpha_step',      type=float, required=True, help='Step for alpha')
    parser.add_argument('--alphas_number',   type=int,   required=True, help='Number of alpha steps')
    parser.add_argument('--sites_start',     type=int,   required=True, help='Minimum number of spins')
    parser.add_argument('--sites_end',       type=int,   required=True, help='Maximum number of spins (inclusive)')
    
    # OPTIONAL arguments with defaults
    parser.add_argument('--number_of_realizations', type=str,    default='10',            help='Realizations per ns (default: 10)')
    parser.add_argument('--n',                      type=int,    default=1,             help='Model parameter n (default: 1)')
    parser.add_argument('--time_num',               type=int,    default=int(1e5),      help='Number of time points (default: 100000)')
    parser.add_argument('--memory_per_worker',      type=float,  default=2.0,           help='Memory reserved per worker in GB (default: 2.0)')
    parser.add_argument('--max_memory',             type=float,  default=196.0,          help='Maximum memory in GB (default: 80.0)')
    parser.add_argument('--uniform',                type=int,    default=1,             help='Use uniform times for the evolution')
    parser.add_argument('-m',        '--model',             type    =   str,    default =   'um',       choices=['um', 'plrb', 'rpm'], help='Model type: um (ultrametric) or plrb (power-law random banded)')
    parser.add_argument('-S',        '--seed',              type    =   int,    default =   None,       help    =   'Random seed for reproducibility')
    parser.add_argument('-c',        '--max_cores',         type    =   int,    default =   psutil.cpu_count(), help='Maximum number of cores to use')
    parser.add_argument('-f',        '--force_single_thread',                   action  =   'store_true',       help='Force single-threaded execution')
    args = parser.parse_args()
    
    if args.sites_start > args.sites_end:
        parser.error(f"--sites_start ({args.sites_start}) must be <= --sites_end ({args.sites_end})")
    
    #! -------------------------------------------------------
    if args.seed is not None:
        logger.info(f"Using provided seed: {args.seed}", lvl=1, color='green')
        seed    = args.seed
        rng     = np.random.default_rng(seed=args.seed)
        set_global_seed(seed, backend=np)
        logger.info(f"Random seed set to {seed}", lvl=2, color='green')
    else:
        logger.info("No seed provided, using current time for random seed", lvl=1, color='yellow')
        seed    = time.time_ns() % (2**32 - 1)
        rng     = np.random.default_rng(seed=seed)
        set_global_seed(seed, backend=np)
        logger.info(f"Random seed set to {seed}", lvl=2, color='yellow')

    rand_num            = rng.integers(0, int(1e5))
    bw_df               = pd.read_csv(f"{file_path}/model/bw.csv", index_col=0, header=None, dtype=float)
    mls_df              = pd.read_csv(f"{file_path}/model/mls.csv", index_col=0, header=None, dtype=float)
    bw_df.index         = [f'{x:.2f}' for x in bw_df.index]
    mls_df.index        = [f'{x:.2f}' for x in mls_df.index]
    bw_df.columns       = list(range(7, 17))
    mls_df.columns      = list(range(7, 16))
    logger.info(f"Bandwidths:\n{bw_df}", lvl=2, color='blue')
    logger.info(f"Mean level spacing:\n{mls_df}", lvl=2, color='blue')
    logger.breakline(1)
    #! -------------------------------------------------------    
    base_dir            = Directories(args.save_dir if args.save_dir else './data')
    base_dir.mkdir()
    
    #! process the arguments
    n               = args.n
    alphas          = np.array([args.alpha_start + i * args.alpha_step for i in range(args.alphas_number)])
    sites           = list(range(args.sites_start, args.sites_end + 1))
    
    #! operators
    operators_map   = {}
    operators_map.update({ op_spin.sig_z(ns=0, type_act=op_spin.OperatorTypeActing.Global, sites=[0]).name: make_sig_z_global })
    operators_map.update({ op_spin.sig_x(ns=0, type_act=op_spin.OperatorTypeActing.Global, sites=[0,5]).name: make_sig_x_global })
    
    n_reals         = calculate_realisations_per_parameter(sites, args.number_of_realizations)
    time_num        = args.time_num
    modelstr        = args.model

    #! -------------------------------------------------------

    logger.info(f"Number of realisations: {n_reals}", color='green')
    logger.info(f"Number of sites: {len(sites)}", color='green')
    logger.info(f"Alphas: {alphas}", color='green')
    logger.info(f"Sites: {sites}", color='green')
    logger.info(f"Operators: {operators_map.keys()}", color='green')
    logger.info(f"Time steps: {time_num}", color='green')
    logger.info(f"Memory per worker: {args.memory_per_worker} GB", color='green')
    logger.info(f"Max memory: {args.max_memory} GB", color='green')
    logger.info(f"Random number: {rand_num}", color='green')
    logger.info(f"Max cores: {args.max_cores}", color='green')
    logger.info(f"Model: {modelstr}", color='green')

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
        max_workers = calculate_optimal_workers(alphas, avail_gb, memory_per_worker, args.max_cores)

    logger.info(f"Available memory: {avail_gb:.2f} GB", color='orange', lvl=1)
    logger.info(f"Memory per worker: {memory_per_worker:.2f} GB", color='orange', lvl=1)
    logger.info(f"Max workers: {max_workers}", color='orange', lvl=1)
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
            operators_map, n, rand_num, start_time, 
            remaining_time, max_workers, seed, modelstr, uniform=args.uniform)

        logger.info(f"All computations completed in {time.perf_counter() - start_time:.2f} s", color='green')

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

#! -------------------------------------------------------
