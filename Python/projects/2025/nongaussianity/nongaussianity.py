''' 
This script produces the result for non-Gaussianity in quantum states
A non-Gaussian state requires higher-order correlators that cannot be reduced to two-point correlators by Wick's theorem.
Thus, any measure of deviation from Wick factorization can serve as a non-Gaussianity measure.
'''

import os
import sys
import time
import math
import numpy as np
import argparse
import itertools
import matplotlib.pyplot as plt

# Import the external module for non-Gaussianity calculations
from nongaussianity_ext import *
from QES.general_python.common.timer import Timer
from QES.general_python.common.directories import Directories
from QES.general_python.common.hdf5_lib import HDF5Handler
from QES.general_python.common.plot import Plotter

#########################################

def parse_gammas(gamma_str: str, ns: int):
    ''' Parse the gamma parameter from command line arguments. '''
    if isinstance(gamma_str, str):
        # split the string by commas
        gamma_str   = list(gamma_str.split(','))
        gammas      = []
        for g in gamma_str:
            if "^" in g:
                # calculate N^\xi gammas
                gamma = ns ** int(g.split('^')[1])
            elif not "*" in g and float(g) >= 1:
                gamma = int(g)
            elif "*" in g:
                gamma = int(ns * float(g.split("*")[1]))
            else:
                gamma = int(ns * float(g))
            gammas.append(gamma)
    else:
        raise ValueError(f"Invalid gamma value: {args.gammas}. Must be a comma-separated list of integers or floats.")
    return gammas

def orbitals_selection(orbitals_en: List, gamma: int, nreal_comb: int, rng: np.random.Generator):

    # create the shuffle to take the orbitals from
    o_arrange           = np.arange(len(orbitals_en))
    o_combs_set         = set()

    # Add a safeguard in case we ask for more combinations than exist
    max_combs           = math.comb(len(orbitals_en), gamma)
    if nreal_comb > max_combs:
        # If all combinations are requested, generating them sequentially is most efficient
        logger.warning(f"Requested {nreal_comb} combinations, but only {max_combs} exist. Using all available.", color='yellow')
        nreal_comb      = max_combs
        o_combs         = itertools.combinations(o_arrange, gamma)
    else:
        while len(o_combs_set) < nreal_comb:
            # 1. Randomly choose 'gamma' indices without replacement
            comb        = rng.choice(o_arrange, size=gamma, replace=False)
            # 2. Sort and convert to tuple for canonical representation
            comb_tuple  = tuple(sorted(comb))
            # 3. Add to set (duplicates are automatically ignored)
            o_combs_set.add(comb_tuple)
            
        # Convert to list for iteration
        o_combs = list(o_combs_set)
    return o_combs

def generate_coeffs(is_complex: bool, num_coeffs_to_gen: int, gamma: int, rng: np.random.Generator):    
    
    if is_complex:
        rand_block  = np.zeros((num_coeffs_to_gen, gamma), dtype=np.complex128)
        for i in range(num_coeffs_to_gen):
            rand_block[i, :] = QuadraticSelection.haar_random_coeff(gamma=gamma, rng=rng)
            
        # Generate a block of random numbers for all realizations
        # rand_block  = rng.normal(size=(num_coeffs_to_gen, gamma)) + 1j * rng.normal(size=(num_coeffs_to_gen, gamma))
        # norms       = np.linalg.norm(rand_block, axis=1, keepdims=True)
        all_coeffs  = rand_block
    else:
        # Generate a block of random numbers for all realizations
        rand_block  = np.zeros((num_coeffs_to_gen, gamma), dtype=np.float64)
        for i in range(num_coeffs_to_gen):
            rand_block[i, :] = QuadraticSelection.haar_random_coeff_real(gamma=gamma, rng=rng)
        all_coeffs  = rand_block
    return all_coeffs

#########################################

def main():
    ap = argparse.ArgumentParser(description='Calculate non-Gaussianity in quantum states.')
    ap.add_argument('--lx',     type=int,       default=10,     help='Size of the system [X] (number of modes)')
    ap.add_argument('--ly',     type=int,       default=1,      help='Size of the system [Y] (number of modes)')
    ap.add_argument('--lz',     type=int,       default=1,      help='Size of the system [Z] (number of modes)')
    ap.add_argument('--la',     type=float,     default=0.5,    help='Size of the system [A] (number of modes)')
    ap.add_argument('--seed',   type=int,       default=None,   help='Random seed for reproducibility')
    ap.add_argument('--occ',    type=float,     default=0.5,    help='Occupation number. If > 1, treated as integer.')
    ap.add_argument('--ordered',type=bool,      default=True,   help='Whether to use ordered modes')
    # couplings
    ap.add_argument('--J',      type=float,     default=1.0,    help='Coupling strength')
    ap.add_argument('--lmbd',   type=float,     default=1.0,    help='Aubry-André model parameter')
    ap.add_argument('--beta',   type=float,     default=1.0/np.sqrt(2), help='Beta parameter for AA model')
    ap.add_argument('--dtype',  type=str,       default='float', help='Data type for calculations')
    # orbital selection and then random mixtures
    ap.add_argument('--norbitals',  type=int,   default=0,      help='Number of orbitals to select from the full spectrum.')
    ap.add_argument('--maxorbs',    type=int,   default=1000,   help='Maximum number of orbitals to select.')
    ap.add_argument('--nreal_comb', type=int,   default=5,      help='Number of realizations to generate for combinations.')
    ap.add_argument('--nreal',      type=int,   default=5,      help='Number of realizations to generate.')
    ap.add_argument('--gammas',     type=str,   default='1,2',  help='Gamma parameter for the random mixture')
    # other
    ap.add_argument('--e_window',   type=float, default=None,   help='Energy window for orbital selection')
    ap.add_argument('--savestates', type=bool,  default=False,  help='Whether to save the states')
    ap.add_argument('--savepath',   type=str,   default='./',   help='Path to save the data')
    ap.add_argument('--verbose',    type=bool,  default=False,   help='Whether to print verbose output')

    ap.add_argument('--hamil',  type    = str,
                                default = 'SYK2',
                                help    = 'Type of Hamiltonian to use')

    args        = ap.parse_args()

    # get the command line arguments
    seed        = args.seed if args.seed is not None else int(time.time() * 1000) % 10000
    lx, ly, lz  = args.lx, args.ly, args.lz
    ns          = args.lx * args.ly * args.lz
    nh          = 2**ns

    # Determine the number of occupied modes
    nocc        = int(args.occ) if args.occ >= 1 else int(args.occ * ns)

    # For system bipartitioning
    la          = int(args.la) if args.la >= 1 else int(args.la * ns)
    lb          = ns - la
    dimA        = 2**la
    dimB        = 2**lb
    normaliz    = np.log(2) * la
    verbose     = args.verbose
    if verbose:
        logger.info(f"Using verbose output", lvl=1, color='blue')

    # other
    rng         = np.random.default_rng(seed)
    e_window    = (-args.e_window, args.e_window) if args.e_window is not None else None
    ran_file    = f"{rng.integers(0, 10000)}"
    gammas      = parse_gammas(args.gammas, ns)

    timer       = Timer(name='nongaussianity', verbose=True, precision='ms')
    t_start     = Timer.now()
    logger.info(f"Using gamma values: {gammas}", lvl=1, color='blue')
    logger.info(f"Using random seed: {seed}", lvl=1, color='blue')
    logger.info(f"Using dtype: {args.dtype}", lvl=1, color='blue')
    logger.info(f"Using ordered: {args.ordered}", lvl=1, color='blue')

    #########################################
    #! 1) Create hamiltonian
    #########################################
    
    model_type  = ModelHamiltonians[args.hamil]
    hamil = create_hamiltonian(
        ns          = ns,
        model_type  = model_type,
        seed        = seed,
        t           = args.J,
        J           = args.J,
        lmbd        = args.lmbd,
        beta        = args.beta,
        lx          = lx,
        ly          = ly,
        lz          = lz,
        dtype       = np.float64 if 'float' in args.dtype else np.complex128,
    )
    
    if model_type != ModelHamiltonians.FreeFermions:
        hamil.build(verbose=args.verbose, use_numpy=True)
        hamil.diagonalize(verbose=args.verbose)

    t_ham = timer.now()
    logger.info(f"Hamiltonian {hamil}, dtype: {hamil.dtype}", lvl=2)
    logger.info(f"Finished building Hamiltonian after t={timer.format_time(t_ham, t_start)}", lvl=1, color='green')

    # save directory
    savedir     = Directories(args.savepath, "data_ent", f"{hamil}")
    savedir.mkdir()
    logger.title(f'Calculating non-Gaussianity for {ns} modes with {nocc} occupied modes - particle conserving', 100, '#', lvl = 0, color = 'red')
    logger.info(f"Saving results to {savedir} using r={ran_file}", lvl=1, color='blue')

    ##########################################
    #! 2) Choose orbitals for the later calculations
    ##########################################

    arrangement         = range(ns)
    orbitals_en         = choose_orbitals(
        arrangement         = arrangement,
        filling             = nocc,
        hamil               = hamil,
        number              = min(args.norbitals, args.maxorbs) if args.norbitals > 0 else args.maxorbs,
        e_window            = e_window,
        rng                 = rng,
        constraints         = [], #[q_constraint(ns, 0.0, 1e-5)] if model_type == ModelHamiltonians.FreeFermions else [],
        choose_momentum     = False #model_type == ModelHamiltonians.FreeFermions
    )
    t_orb               = timer.now()
    logger.info(f"Selected orbitals: {len(orbitals_en)} in t={timer.format_time(t_orb, t_ham)}", lvl=1, color='blue')
    
    orbitals_one_hot    = [to_one_hot(occ[0], ns) for occ in orbitals_en]
    t_orb_one_hot       = timer.now()
    logger.info(f"Pre-processed orbitals to 1-hot representation in t={timer.format_time(t_orb_one_hot, t_orb)}", lvl=2, color='blue')

    ##########################################
    #! 3) Prepare transformation matrices
    ##########################################
    
    transform_pc        = prepare_trans_mat_orbitals(hamil, occ=la, ordered=args.ordered)
    W, W_A, W_A_CT, W_CT= transform_pc.W, transform_pc.W_A(), transform_pc.W_A_CT(), transform_pc.W.conj().T
    t_transform         = timer.now()
    logger.info(f"Shape of W: {W.shape}, dtype: {W.dtype}", lvl=2)
    logger.info(f"Shape of W_A: {W_A.shape}", lvl=3)
    logger.info(f"Shape of W_A_CT: {W_A_CT.shape}", lvl=3)
    logger.info(f"Execution time: {timer.format_time(timer.now(), t_transform)}", lvl=2, color = 'yellow')

    ##########################################
    #! 4) Calculate the random mixtures
    ##########################################

    calculate_mb = ns <= 14 # many-body states are feasable
    
    # store the results in the HDF5
    HDF5Handler.save_hdf5(
            directory   = savedir,
            filename    = f'nongaussianity_{ran_file}.h5',
            data        = {
                "gamma" : np.array(gammas, dtype=np.int64),
                "norm"  : np.array(normaliz, dtype=np.float64)
            }
        )
    
    is_complex      = np.issubdtype(hamil.dtype, np.complexfloating)
    if is_complex:
        logger.info("Using complex coefficients for the random mixtures", lvl=1, color='blue')
    else:
        logger.info("Using real coefficients for the random mixtures", lvl=1, color='blue')
    
    ########################
    ts_gamma        = [t_transform]
    results         = {}
    verbose_ev      = max(1, args.nreal_comb // 5)
    for gamma in gammas:
        if not isinstance(gamma, int) or gamma < 1:
            raise ValueError(f"Gamma must be a positive integer, got: {gamma}")
        
        logger.info(f"Calculating non-Gaussianity for gamma={gamma} with {args.nreal} realizations", lvl=1, color='green')
        
        # data structures
        num_total_realizations  = args.nreal_comb * (args.nreal if gamma > 1 else 1)
        ng_results              = np.zeros(num_total_realizations)
        gaussianity_results     = np.zeros(num_total_realizations)
        purity_results          = np.zeros(num_total_realizations)
        m4_results              = np.zeros(num_total_realizations)
        m6_results              = np.zeros(num_total_realizations)
        # many-body
        entropies_mb_stat       = np.zeros(num_total_realizations) if calculate_mb else None
        purity_mb_stat          = np.zeros(num_total_realizations) if calculate_mb else None
        particip_mb_stat        = np.zeros(num_total_realizations) if calculate_mb else None
        wick_diff               = np.zeros(num_total_realizations) if calculate_mb else None

        #! orbitals
        o_combs                 = orbitals_selection(orbitals_en, gamma, args.nreal_comb, rng)
        t_combs_in              = timer.now()
        logger.info(f"Generated unique combinations of orbitals for gamma={gamma} in t={timer.format_time(t_combs_in, ts_gamma[-1])}", lvl=2, color='yellow')

        #! allocate many coefficients to generate
        num_coeffs_to_gen       = args.nreal if gamma > 1 else 1
        all_coeffs              = generate_coeffs(is_complex, num_coeffs_to_gen, gamma, rng)
        t_coeffs_in             = timer.now()
        logger.info(f"Generated coefficients for gamma={gamma} in t={timer.format_time(t_coeffs_in, t_combs_in)}", lvl=2, color='yellow')
        
        #########################
        result_idx              = 0
        for ii, o_comb_indices in enumerate(o_combs):
            if ii >= args.nreal_comb:
                break

            if ii % verbose_ev == 0:
                logger.info(f"Combination {ii+1} of {args.nreal_comb}: {o_comb_indices if verbose  and len(o_comb_indices) < 10 else ''}. t={timer.format_time(timer.now(), t_combs_in)}", lvl=2, color='blue')

            #! Pre-fetch one-hot vectors for this combination
            # This is now just a fast lookup from a pre-computed list.
            occ_list_one_hot    = [orbitals_one_hot[k] for k in o_comb_indices]

            #! Pre-compute many-body states for this combination
            mb_basis_states     = []
            if calculate_mb:
                for comb_idx in o_comb_indices:
                    comb, en    = orbitals_en[comb_idx]
                    mb_basis_states.append(hamil.many_body_state(occupied_orbitals=comb))

            # Go through all realizations
            num_real = args.nreal if gamma > 1 else 1
            for jj in range(num_real):
                coefficients    = all_coeffs[jj]
                corr_sup        = Corr.corr_superposition(
                    W_A                 =   W,
                    occ_list            =   occ_list_one_hot, # Use the pre-fetched list
                    coeff               =   coefficients,
                    W_A_CT              =   W_CT,
                    raw                 =   True,
                    subtract_identity   =   True)[0]

                if calculate_mb:
                    # The expensive state creation is done. This is now just a fast linear combination.
                    mb_state                        = np.dot(coefficients, mb_basis_states)
                    # schmidt_values, ev, rho         = Dens.schmidt_numpy(state=mb_state, dimA=dimA, dimB=dimB, eig=False)
                    schmidt_values, ev, rho         = Dens.schmidt_numpy(state=mb_state, dimA=dimA, dimB=dimB, eig=True)
                    # calculate
                    entropies_mb_stat[result_idx]   = Entropy.vn_entropy(schmidt_values)
                    purity_mb_stat[result_idx]      = Entropy.purity(schmidt_values)
                    particip_mb_stat[result_idx]    = Entropy.information_entropy(mb_state)
                    # wick
                    n_for_wick                      = (corr_sup + np.eye(ns, dtype=hamil.dtype)) / 2.0
                    idx_j                           = rng.integers(0, ns)
                    idx_l                           = rng.integers(0, ns)
                    C_wick                          = Corr.corr_single2_slater_wick(corr=n_for_wick, ns=ns, j=idx_j, l=idx_l)
                    C_wick_mb                       = Corr.corr_from_statevector(mb_state, ns=ns, order=4, j=idx_j, l=idx_l)
                    # Frobenious radius
                    K_ki                            = C_wick - C_wick_mb
                    K2                              = np.sum(np.abs(K_ki)**2)
                    W2                              = np.sum(np.abs(C_wick)**2)
                    R_glob                          = float(K2 / (K2 + W2 + 1e-16)) # how to normalize?
                    wick_diff[result_idx]           = R_glob

                # measures for single particle
                t0                      = timer.now()
                corr_sup                = corr_sup[:la, :la] if args.ordered else corr_sup[la,la]
                corr_val, _             = np.linalg.eigh(corr_sup)
                NG, gaussianity, purity, m4, m6 = nongaussianity(eigvals=corr_val, return_stats=True)
                t1                      = timer.now()
                if ii % verbose_ev == 0 and jj == 0:
                    logger.info(f"Calculated non-Gaussianity for combination {ii+1}, realization {jj+1} in t={timer.format_time(t1, t0)}", lvl=3, color='blue')
                
                # Fill the pre-allocated arrays
                ng_results[result_idx]          = NG
                gaussianity_results[result_idx] = gaussianity
                purity_results[result_idx]      = purity
                m4_results[result_idx]          = m4
                m6_results[result_idx]          = m6
                result_idx                     += 1
    
        ng_results      = np.array(ng_results[:result_idx]) / normaliz
        new_data        = {
            f'correlation/gamma={gamma}/ng/v'        : ng_results,
            f'correlation/gamma={gamma}/ng/m1'       : np.mean(ng_results),
            f'correlation/gamma={gamma}/ng/m2'       : np.mean(ng_results**2),
            f'correlation/gamma={gamma}/ng/m4'       : np.mean(ng_results**4),
            # other statistics
            f'correlation/gamma={gamma}/gaussianity' : gaussianity_results,
            f'correlation/gamma={gamma}/purity'      : purity_results,
            f'correlation/gamma={gamma}/m4'          : m4_results,
            f'correlation/gamma={gamma}/m6'          : m6_results,
        }
        if calculate_mb:
            entropies_mb_stat = np.array(entropies_mb_stat) / normaliz
            new_data.update({
                    f'many_body/gamma={gamma}/ent/v'    : entropies_mb_stat,
                    f'many_body/gamma={gamma}/ent/m1'   : np.mean(entropies_mb_stat),
                    f'many_body/gamma={gamma}/ent/m2'   : np.mean(entropies_mb_stat**2),
                    f'many_body/gamma={gamma}/ent/m4'   : np.mean(entropies_mb_stat**4),
                    # other measures
                    f'many_body/gamma={gamma}/purity'   : purity_mb_stat,
                    f'many_body/gamma={gamma}/info_ent' : particip_mb_stat,
                    f'many_body/gamma={gamma}/wick'     : wick_diff,
                })

        HDF5Handler.append_hdf5(
            directory   =   savedir,
            filename    =   f'nongaussianity_{ran_file}.h5',
            new_data    =   new_data
        )
        
        if args.verbose:
            results[gamma] = {
                "CORRELATION/v"     : new_data[f'correlation/gamma={gamma}/ng/v'],
                "CORRELATION/m1"    : np.mean(new_data[f'correlation/gamma={gamma}/ng/v']),
                "CORRELATION/m2"    : np.mean(new_data[f'correlation/gamma={gamma}/ng/v']**2),
                "CORRELATION/m4"    : np.mean(new_data[f'correlation/gamma={gamma}/ng/v']**4),
            }
            if calculate_mb:
                results[gamma].update({
                    "MANY_BODY/v"      : new_data[f'many_body/gamma={gamma}/ent/v'],
                    "MANY_BODY/m1"     : np.mean(new_data[f'many_body/gamma={gamma}/ent/v']),
                    "MANY_BODY/m2"     : np.mean(new_data[f'many_body/gamma={gamma}/ent/v']**2),
                    "MANY_BODY/m4"     : np.mean(new_data[f'many_body/gamma={gamma}/ent/v']**4),
                    "MANY_BODY/wick"   : new_data[f'many_body/gamma={gamma}/wick'],
                })

        t_gamma = timer.now()
        logger.info(f"Finished gamma={gamma} calculations in t={timer.format_time(t_gamma, ts_gamma[-1])}", lvl=1, color='green')
        ts_gamma.append(t_gamma)
    
    if args.verbose:
        fig, ax = Plotter.get_subplots(nrows=2, ncols=1, figsize=(4, 4), sharex=True)

        for i, (gamma, res) in enumerate(results.items()):
            logger.info(f"Results for gamma={gamma}:", color='blue')
            # correlation entropies
            ent_correlation = [f'{x:.3e}' for x in res["CORRELATION/v"][:5]]
            logger.info(f"Correlation Entropies: {ent_correlation}...")
            logger.info(f"Mean: {res['CORRELATION/m1']:.3e}", lvl=1)
            logger.info(f"Variance: {res['CORRELATION/m2'] - res['CORRELATION/m1']**2:.3e}", lvl=1)  
            ax[0].scatter(gamma, res['CORRELATION/m1'], marker='o', color='blue', label='Correlation' if i == 0 else None)

            if calculate_mb:
                # many-body entropies
                ent_mb      = [f'{x:.3e}' for x in res["MANY_BODY/v"][:5]]
                logger.info(f"Many-Body Entropies: {ent_mb}...", color='red')
                logger.info(f"Mean: {res['MANY_BODY/m1']:.3e}", color='red', lvl=1)
                logger.info(f"Variance: {res['MANY_BODY/m2'] - res['MANY_BODY/m1']**2:.3e}", color='red', lvl=1)
                ax[0].scatter(gamma, res['MANY_BODY/m1'], marker='s', color='red', label=f'Many-body' if i == 0 else None)
                
                wick_mb     = [f'{x:.3e}' for x in res["MANY_BODY/wick"][:5]]
                logger.info(f"Wick Difference: {wick_mb}...", color='red')
                logger.info(f"Mean: {np.mean(res['MANY_BODY/wick']):.3e}", color='red', lvl=1)
                ax[1].scatter(gamma, np.mean(res['MANY_BODY/wick']), marker='s', color='red', label=f'Many-body' if i == 0 else None)
            logger.breakline(1)
            
        ax[0].set_ylabel(r"$\langle S \rangle / (L_A \log 2)$")
        ax[0].set_xlim(1, None)
        ax[0].set_ylim(0.5, 1.0)
        Plotter.set_tickparams(ax[0])
        Plotter.set_legend(ax[0])
        
        ax[1].set_xlabel(r"$\Gamma$")
        ax[1].set_ylabel(r"$|C_{\rm wick} - C_{\rm mb}|$")
        ax[1].set_xlim(1, None)
        ax[1].set_yscale('log')
        ax[1].set_ylim(1e-3, None)
        Plotter.set_tickparams(ax[1])
        Plotter.set_legend(ax[1])

        plt.show()

    t_end = timer.now()
    logger.info(f"Finished calculating non-Gaussianity after t={timer.format_time(t_end, t_start)}", color='red')

    return 0

# ------------------------

if __name__ == '__main__':
    sys.exit(main())

# ------------------------
#! EOF :|
# ------------------------