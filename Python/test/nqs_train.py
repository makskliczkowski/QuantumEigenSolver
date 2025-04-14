#!/usr/bin/env python3
"""
file        :   nqs_solver_profiled.py
author      :   Maksymilian Kliczkowski
Script to test and profile the NQS Solver using Monte Carlo methods.
It sets up a neural network, sampler, Hamiltonian, and profiles the training steps.
An argument parser is provided for configuration.
"""

import os
import sys
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange

# Add the parent directory to the system path. Use absolute path relative to this file's location
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from QES.general_python.algebra.utils import JAX_AVAILABLE
    from QES.general_python.common.flog import Logger, print_arguments, log_timing_summary
    from QES.general_python.common.plot import Plotter

    #! NQS
    import QES.NQS.nqs as nqsmodule

    #! Sampler
    import QES.Solver.MonteCarlo.sampler as Sampling

    #! Backend
    from QES.general_python.algebra.utils import get_backend

    #! Network
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM

    #! Hamiltonian
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.general_python.lattices.square import SquareLattice
    from QES.general_python.lattices.lattice import LatticeBC

    #! Solver
    import QES.general_python.algebra.solvers.stochastic_rcnfg as SR
    import QES.general_python.algebra.solvers as solvers
    import QES.general_python.algebra.preconditioners as preconditioners
except ImportError as e:
    print("Error importing modules. Please ensure the QES package is installed correctly.")
    print(e)
    sys.exit(1)

# Initialize logger
logger = Logger()
logger.title("Starting NQS Sampler Test Script...", 50, '#', lvl = 0)

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import jit
    jax.config.update('jax_platform_name', 'cpu')
else:
    jax     = None
    jnp     = None

# -----------------------------------------------------------------------
#! Helper Function: Single Training Step with Detailed Timing
# -----------------------------------------------------------------------

def single_step_train(i: int, lr: float, reset=False):
    #! Generate samples
    time_start       = time.time()
    (_, _), (configs, configs_ansatze), probabilities = nqs.sample(reset=reset)        
    time_sample      = time.time() - time_start

    #! Single step in the NQS (includes ansatz evaluation, local energy computation, and gradient calculation)
    time_start       = time.time()
    dpar, step_info  = nqs.single_step_jax(
                            params               = nqs.get_params(),
                            configs              = configs,
                            configs_ansatze      = configs_ansatze,
                            probabilities        = probabilities,
                            apply_fn             = ansatz,
                            local_energy_fun     = loc_energy,
                            flat_grad_fun        = flat_grad,
                            #! Stochastic reconfiguration options
                            use_sr               = True,
                            sr_options           = sr_options,
                            sr_precond_apply_fun = precond_fun,
                            sr_solve_linalg_fun  = solver_fun,
                        )
    time_single_step = time.time() - time_start

    if step_info.failed:
        logger.error(f"Step {i} failed. Check configurations and parameters.")
        return None, (None, time_sample, time_single_step)

    #! Update the parameters
    time_start       = time.time()
    nqs.update_parameters(-lr * dpar)
    time_update      = time.time() - time_start

    #! Return mean energy and timing details for update, sampling, and step
    return (step_info.mean_energy, step_info.std_energy), (time_update, time_sample, time_single_step)

# -----------------------------------------------------------------------
#! Training Function with Epoch-Level Profiling
# -----------------------------------------------------------------------

def adaptive_lr(epoch: int, initial_lr: float, decay_rate: float):
    return max(5.0e-3, initial_lr * (decay_rate ** epoch))

def train_function(n_epo: int, lr: float = 7e-2, reset: bool = False, decay_rate: float = 0.999):
    history      = np.zeros(n_epo, dtype=np.float64)
    history_std  = np.zeros(n_epo, dtype=np.float64)
    epoch_times  = np.zeros(n_epo, dtype=np.float64)
    pbar         = trange(n_epo, desc="Starting Training...", leave=True)

    for i in pbar:
        current_lr = adaptive_lr(i, lr, decay_rate)
        time_start  = time.time()
        (mean_energy, std_energy), (time_update, time_sample, time_single_step) = single_step_train(i, current_lr, reset)
        time_epoch  = time.time() - time_start

        if mean_energy is None:
            logger.warning(f"Training stopped early at epoch {i} due to failure in `single_step_train`.")
            break

        pbar.set_description(f"Epoch {i + 1}/{n_epo}")
        postfix_dict = {
            "E_mean"   : f"{mean_energy:.4e}",
            "E_std"    : f"{std_energy:.4e}",
            "lr"       : f"{current_lr:.3e}",
            "t_samp"   : f"{time_sample:.3e}s",
            "t_step"   : f"{time_single_step:.3e}s",
            "t_update" : f"{time_update:.3e}s",
            "t_epoch"  : f"{time_epoch:.3e}s"
        }
        pbar.set_postfix(postfix_dict, refresh=True)
        history[i]      = np.real(mean_energy)
        history_std[i]  = np.real(std_energy)
        epoch_times[i]  = time_epoch

    return history[:i+1], history_std[:i+1], epoch_times[:i+1]

#! Main function

if __name__ == "__main__":
# ----------------------- Argument Parser -------------------------------
    parser            = argparse.ArgumentParser(description='Train and profile the NQS Solver using Monte Carlo methods.')
    parser.add_argument('--backend',        type=str,   default='jax',  choices=['jax', 'numpy'], help='Backend to use (jax or numpy)')
    parser.add_argument('--seed',           type=int,   default=1234,   help='Random seed for reproducibility')
    parser.add_argument('--sites',          type=int,   default=8,      help='Number of sites')
    parser.add_argument('--n_epo',          type=int,   default=200,    help='Number of training epochs')
    parser.add_argument('--lr',             type=float, default=7e-2,   help='Learning rate')
    #! Optional arguments for network, sampler, solver
    parser.add_argument('--n_hidden',       type=int,   default=16,     help='Number of hidden units in the RBM')
    parser.add_argument('--numchains',      type=int,   default=100,    help='Number of chains for the sampler')
    parser.add_argument('--numsamples',     type=int,   default=100,    help='Number of samples for the sampler')
    parser.add_argument('--sweep_steps',    type=int,   default=8,      help='Number of sweep steps for the sampler')
    parser.add_argument('--therm_steps',    type=int,   default=25,     help='Number of thermalization steps for the sampler')
    parser.add_argument('--batch',          type=int,   default=1,      help='Batch size for the NQS solver')
    #! Optional arguments for SR
    parser.add_argument('--solver',         type=str,   default='scipy_cg', choices=['scipy_cg', 'cg'], help='Solver for the stochastic reconfiguration')
    parser.add_argument('--precond',        type=str,   default='jacobi',   choices=['jacobi', 'identity'], help='Preconditioner for the stochastic reconfiguration')
    parser.add_argument('--reg',            type=float, default=1,          help='Regularization parameter for the stochastic reconfiguration')
    parser.add_argument('--maxiter',        type=int,   default=500,        help='Maximum iterations for the stochastic reconfiguration')
    parser.add_argument('--tol',            type=float, default=1e-8,       help='Tolerance for the stochastic reconfiguration')
    parser.add_argument('--min_sr',         type=bool,  default=False,      help='Use minimum stochastic reconfiguration')
    #! Optional arguments for Profiling
    parser.add_argument('--profile', action='store_true',           help='Enable profiling with cProfile')
    parser.add_argument('--output', type=str, default='profile_output.txt', help='Output file for profiling results')
    
    # print the arguments
    print_arguments(parser, logger)
    
    # Parse the command line arguments
    args            = parser.parse_args()

    # ----------------------- Setup Backend and Seed ------------------------
    backend_str     = args.backend
    seed            = args.seed
    ns              = args.sites
    n_epo           = args.n_epo
    lr              = args.lr
    dtypex          = jnp.complex128
    seed            = args.seed
    
    #! Optional arguments for network, sampler, solver
    n_hidden        = args.n_hidden
    n_hidden        = args.n_hidden if n_hidden > 0 else ns * np.abs(n_hidden)
    numchains       = args.numchains
    numsamples      = args.numsamples
    sweep_steps     = args.sweep_steps
    therm_steps     = args.therm_steps
    batch_size      = args.batch
    #! Optional arguments for SR
    solver_id       = args.solver
    precond_id      = args.precond
    precond_id      = preconditioners.PreconditionersTypeSym.JACOBI if precond_id == 'jacobi' else preconditioners.PreconditionersTypeSym.IDENTITY
    reg             = args.reg
    maxiter         = args.maxiter
    tol             = args.tol
    min_sr          = args.min_sr
    #! Optional arguments for Profiling
    profile         = args.profile
    output          = args.output
    
    # ----------------------- Setup Backend ---------------------------------
    
    be_modules          = get_backend(backend_str, random=True, seed=seed, scipy=True)
    backend_np, (rng, rng_k), backend_sp = be_modules if isinstance(be_modules, tuple) else (be_modules, (None, None), None)
    
    # ----------------------- Setup Network ---------------------------------
    st_shape        = (ns, )
    net             = RBM(
                        input_shape    = st_shape, 
                        n_hidden       = n_hidden,
                        dtype          = dtypex,
                        seed           = seed,
                        visible_bias   = True,
                        bias           = True
                    )
    
    # ----------------------- Setup Sampler ---------------------------------
    sampler         = Sampling.MCSampler(
                        net          = net,
                        shape        = st_shape,
                        rng          = rng,
                        rng_k        = rng_k,
                        numchains    = numchains,
                        numsamples   = numsamples,
                        sweep_steps  = sweep_steps,
                        backend      = backend_np,
                        therm_steps  = therm_steps,
                        mu           = 2.0,
                        seed         = seed
                    )
    sampler_fun = sampler.get_sampler_jax(num_samples=numsamples, num_chains=numchains)

    # ----------------------- Setup Hamiltonian -----------------------------
    ham_dtype       = jnp.float64
    lattice         = SquareLattice(
                        dim          = 1, 
                        lx           = ns, 
                        ly           = 1, 
                        lz           = 1, 
                        bc           = LatticeBC.PBC
                    )
    
    j               = -1.0   
    g               = -0.5   
    h               = 0.5
    ham             = TransverseFieldIsing(
                        lattice        = lattice,
                        hilbert_space  = None,
                        j              = j,
                        hz             = h,
                        hx             = g,
                        dtype          = ham_dtype,
                        backend        = backend_np
                    )

    # ----------------------- Setup NQS Solver ------------------------------
    nqs             = nqsmodule.NQS(
                        net          = net,
                        sampler      = sampler,
                        hamiltonian  = ham,
                        lower_betas  = None,
                        lower_states = None,
                        seed         = seed,
                        beta         = 1.0,
                        mu           = sampler.get_mu(),
                        shape        = st_shape,
                        backend      = backend_np,
                        nthread      = 1,
                        batch_size   = batch_size,
                    )
    
    # ----------------------- Setup Stochastic Reconfiguration (SR) ---------
    sr_options      = SR.SRParams(
                        min_sr        = min_sr,
                        maxiter       = maxiter,
                        tol           = tol,
                        reg           = reg,
                        solver_form_s = False
                    )
    solver_linalg   = solvers.choose_solver(solver_id=solvers.SolverType.SCIPY_CG, sigma=reg)
    precond         = preconditioners.choose_precond(precond_id=precond_id, backend=backend_str)
    precond_apply   = precond.get_apply_gram()
    solver_linalg_func = solver_linalg.get_solver_func(
                            backend_module = backend_np,
                            use_matvec     = False,
                            use_matrix     = False,
                            use_fisher     = True,
                            sigma          = None
                        )
    solver_fun      = solver_linalg_func
    precond_fun     = precond_apply
    
    # ----------------------- Get Functions from NQS ------------------------
    ansatz          = nqs.ansatz
    loc_energy      = nqs.local_energy
    flat_grad       = nqs.flat_grad
    
    # ----------------------- Training and Profiling ------------------------
    if args.profile:
        import cProfile
        import pstats
        profiler    = cProfile.Profile()
        profiler.enable()
        energies, energies_std, times = train_function(n_epo, lr, reset=False)
        profiler.disable()
        profiler.dump_stats("profile_stats.prof")
        pstats.Stats(profiler).sort_stats("cumtime").print_stats(50)
        with open(output, 'w') as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats('cumtime').print_stats(50)
    else:
        energies, energies_std, times = train_function(n_epo, lr, reset=False)
    
    # ----------------------- Plotting --------------------------------------
    fig, ax = Plotter.get_subplots(nrows=2, ncols=1, sizex=5, sizey=5, dpi=100, sharex=True)
    Plotter.plot(ax[0], x=np.arange(len(energies)), y=np.real(energies), marker="o", markersize=0.5, lw=1)
    Plotter.set_ax_params(ax[0], title="Training Progress", ylabel=r"$E$")
    Plotter.set_tickparams(ax[0], maj_tick_l=2, min_tick_l=1)
    
    
    ax[1].plot(np.real(energies_std))
    Plotter.plot(ax[1], x=np.arange(len(energies_std)), y=np.real(energies_std),
                marker="o", markersize=0.5, lw=1)
    Plotter.set_ax_params(ax[1], xlabel="Epoch", ylabel=r"$\sigma_E$")
    Plotter.set_tickparams(ax[1], maj_tick_l=2, min_tick_l=1)
        
    plt.show()