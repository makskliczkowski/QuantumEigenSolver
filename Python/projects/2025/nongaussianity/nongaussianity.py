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

# Import the external module for non-Gaussianity calculations
from nongaussianity_ext import *
from QES.general_python.common.timer import Timer
from QES.general_python.common.directories import Directories

def main():
    ap = argparse.ArgumentParser(description='Calculate non-Gaussianity in quantum states.')
    ap.add_argument('--lx',     type=int,   default=10,     help='Size of the system [X] (number of modes)')
    ap.add_argument('--ly',     type=int,   default=1,      help='Size of the system [Y] (number of modes)')
    ap.add_argument('--lz',     type=int,   default=1,      help='Size of the system [Z] (number of modes)')
    ap.add_argument('--la',     type=float, default=0.5,    help='Size of the system [A] (number of modes)')
    ap.add_argument('--seed',   type=int,   default=42,     help='Random seed for reproducibility')
    ap.add_argument('--occ',    type=float, default=0.5,    help='Occupation number. If > 1, treated as integer.')
    ap.add_argument('--ordered', action='store_true',       help='Whether to use ordered modes')
    # couplings
    ap.add_argument('--J',      type=float, default=1.0,    help='Coupling strength')
    ap.add_argument('--lmbd',   type=float, default=1.0,    help='Aubry-André model parameter')
    ap.add_argument('--beta',   type=float, default=1.0/np.sqrt(2), help='Beta parameter for AA model')
    ap.add_argument('--dtype',  type=str,   default='float64', help='Data type for calculations')
    # orbital selection and then random mixtures
    ap.add_argument('--norbitals',  type=int,   default=0,      help='Number of orbitals to select from the full spectrum.')
    ap.add_argument('--nreal',      type=int,   default=5,      help='Number of realizations to generate.')
    ap.add_argument('--gammas',     type=str,   default='1,2',  help='Gamma parameter for the random mixture')
    # other
    ap.add_argument('--e_window',   type=float, default=None,   help='Energy window for orbital selection')
    ap.add_argument('--savestates', type=bool,  default=False,  help='Whether to save the states')
    ap.add_argument('--savepath',   type=str,   default='./',   help='Path to save the data')

    ap.add_argument('--hamil',  type    = str,
                                default = 'SYK2',
                                choices = ModelHamiltonians,
                                help    = 'Type of Hamiltonian to use')

    args = ap.parse_args()

    # get the command line arguments
    seed        = args.seed
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
    
    # other
    ordered     = args.ordered
    rng         = np.random.default_rng(seed)
    e_window    = (-e_window, e_window) if args.e_window is not None else None
    savedir     = Directories(args.savepath, "data")
    savedir.mkdir()

    # mixture parameters
    if isinstance(args.gammas, str):
        # split the string by commas
        args.gammas = list(args.gammas.split(','))
        gammas      = []
        for g in args.gammas:
            if "^" in g:
                # calculate N^\xi gammas
                gamma = ns ** int(g.split('^')[1])
            elif int(g) >= 1:
                gamma = int(g)
            else:
                gamma = ns * float(g)
            gammas.append(gamma)
    else:
        raise ValueError(f"Invalid gamma value: {args.gammas}. Must be a comma-separated list of integers or floats.")

    timer       = Timer(name='nongaussianity', verbose=True, precision='ms')
    t_start     = Timer.now()
    logger.title(f'Calculating non-Gaussianity for {ns} modes with {nocc} occupied modes - particle conserving', 100, '#', lvl = 0, color = 'red')
    logger.info(f"Using gamma values: {gammas}", lvl=1, color='blue')
    logger.info(f"Using random seed: {seed}", lvl=1, color='blue')
    
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
        dtype       = np.dtype(args.dtype),
    )
    
    if model_type != ModelHamiltonians.FreeFermions:
        hamil.build(verbose=True, use_numpy=True)
        hamil.diagonalize(verbose=True)
    
    t_ham       = timer.now()
    logger.info(f"Hamiltonian {hamil}, dtype: {hamil.dtype}", lvl=2)
    logger.info(f"Finished building Hamiltonian after t={timer.format_time(t_ham, t_start)}", lvl=1, color='green')

    ##########################################
    #! 2) Choose orbitals for the later calculations
    ##########################################

    arrangement = range(ns)
    orbitals_en = choose_orbitals(
        arrangement = arrangement,
        filling     = nocc,
        hamil       = hamil,
        number      = args.norbitals if args.norbitals > 0 else None,
        e_window    = args.e_window if hasattr(args, 'e_window') else None,
        rng         = rng,
        constraints = [q_constraint(ns, 0.0, 1e-5)] if model_type == ModelHamiltonians.FreeFermions else [],    
    )
    t_orb       = timer.now()
    logger.info(f"Selected orbitals: {len(orbitals_en)} in t={timer.format_time(t_orb, t_ham)}", lvl=1, color='blue')

    ##########################################
    #! 3) Prepare transformation matrices
    ##########################################
    
    transform_pc    = prepare_trans_mat_orbitals(hamil, occ=la, ordered=ordered)
    W, W_A, W_A_CT  = transform_pc.W, transform_pc.W_A(), transform_pc.W_A_CT()
    t_transform     = timer.now()
    logger.info(f"Shape of W: {W.shape}, dtype: {W.dtype}", lvl=2)
    logger.info(f"Shape of W_A: {W_A.shape}", lvl=3)
    logger.info(f"Shape of W_A_CT: {W_A_CT.shape}", lvl=3)
    logger.info(f"Execution time: {timer.format_time(timer.now(), t_transform)}", lvl=2, color = 'yellow')

    ##########################################
    #! 4) Calculate the random mixtures
    ##########################################

    calculate_mb    = ns <= 14 # many-body states are feasable
    results         = {}
    
    # prepare the many-body state - memory efficient
    mb_state        = np.zeros(nh, dtype=hamil.dtype) if calculate_mb else None

    for gamma in gammas:
        
        if not isinstance(gamma, int) or gamma < 1:
            raise ValueError(f"Gamma must be a positive integer, got: {gamma}")
        logger.info(f"Calculating non-Gaussianity for gamma={gamma} with {args.nreal} realizations", lvl=1, color='green')
        
        # data structures
        entropies_sp_corr = []
        entropies_mb_stat = []
        
        # create the shuffle to take the orbitals from
        o_arrange   = np.arange(len(orbitals_en))
        rng.shuffle(o_arrange) # shuffle the arrangement to get different combinations
        o_combs     = itertools.combinations(o_arrange, gamma)

        # zero the mb state
        mb_state[:] = 0.0 if calculate_mb else None
        
        # go through the combinations
        for ii, o_comb in enumerate(itertools.islice(o_combs, args.nreal)):
            if ii >= args.nreal:
                break
            logger.info(f"Combination {ii+1} of {args.nreal}: {o_comb}", lvl=2, color='blue')

            # prepare the coefficients
            if hamil.dtype == np.complex128 or hamil.dtype == np.complex64:
                coefficients = rng.normal(size=gamma) + 1j * rng.normal(size=gamma)
            else:
                coefficients = rng.normal(size=gamma)
            coefficients = coefficients / np.linalg.norm(coefficients)
            
            # prepare the combinations
            if calculate_mb:
                for jj, comb_idx in enumerate(o_comb):
                    comb, en    = orbitals_en[comb_idx]
                    mb_state   += coefficients[jj] * hamil.many_body_state(occupied_orbitals=comb)

                #! density matrix - true entanglement entropy of the mixed state
                schmidt_values  = Dens.schmidt_numpy(state=mb_state, dimA=dimA, dimB=dimB, eig=False)[0]
                entropy_mb      = Entropy.vn_entropy(schmidt_values)
                entropies_mb_stat.append(entropy_mb)
            else:
                entropies_mb_stat.append(0.0) # placeholder for non-many-body states
            
            #! correlation matrix
            occ_list = [orbitals_en[o_comb[k]][0] for k in range(gamma)]
            occ_list = [to_one_hot(occ, ns) for occ in occ_list]
            corr_sup = Corr.corr_superposition(
                W_A                 = W_A,
                occ_list            = occ_list,
                coeff               = coefficients,
                W_A_CT              = W_A_CT,
                raw                 = True,
                subtract_identity   = True)[0]
            corr_val, _ = np.linalg.eigh(corr_sup)
            NG, gaussianity, purity, m4, m6 = nongaussianity(eigvals=corr_val, return_stats=True)
            entropies_sp_corr.append((NG, gaussianity, purity, m4, m6))
            
        # store the results
        results[gamma] = {
            "CORRELATION"   : entropies_sp_corr,
            "MANY_BODY"     : entropies_mb_stat
        }
        t_gamma = timer.now()
        logger.info(f"Finished gamma={gamma} calculations in t={timer.format_time(t_gamma, t_transform)}", lvl=1, color='green')
    print("Results:")
    for gamma, res in results.items():
        print(f"  Gamma={gamma}:")
        print(f"    CORRELATION:  {res['CORRELATION']}")
        print(f"    MANY_BODY:    {res['MANY_BODY']}")
    t_end = timer.now()
    logger.info(f"Finished calculating non-Gaussianity after t={timer.format_time(t_end)}")

    return 0

# ------------------------

if __name__ == '__main__':
    sys.exit(main())
    
# ------------------------
#! EOF