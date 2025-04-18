'''
file        :   test/nqs_solver.py
author      :   Maksymilian Kliczkowski

Script to test the NQS Solver using Monte Carlo methods.
It sets up a simple neural network, configures a sampler,
generates samples, and prints information about the process.

In addition, it provides a Hamiltonian class to define the Hamiltonian
and a function to generate samples using the sampler.
'''

import argparse
import numpy as np
import os
import sys
import time

# Add the parent directory to the system path. Use absolute path relative to this file's location
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import QuantEvolveStudio methods
try:
    from QES.Solver.MonteCarlo.sampler import Sampler, MCSampler, get_backend, JAX_AVAILABLE
    from QES.general_python.ml.net_impl.net_simple import SimpleNet
    from QES.general_python.common.plot import MatrixPrinter
    from QES.general_python.common.flog import Logger, print_arguments, log_timing_summary
    from QES.Algebra.Model.dummy import DummyHamiltonian
    import QES.NQS.nqs as nqsmodule
    import QES.general_python.ml.net_impl.utils.net_utils as net_utils

except ImportError:
    print("Error: Could not import QES modules. Ensure the QES package is installed or the path is correct.")
    print(f"Attempted to add parent directory: {parent_dir}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Initialize logger
logger = Logger()
logger.title("Starting NQS Sampler Test Script...", 50, '#', lvl = 0)

import numpy as np
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jax     = None
    jnp     = None
    
# ------------------------------------------------------

def generate_samples(smplr, ntimes = 1):
    """
    Generate samples using the sampler.
    
    Parameters:
    smplr:
        The sampler object.
    ntimes:
        Number of times to sample.
    Returns:
        states:
            Generated states.
        logprobas:
            Logarithm of probabilities.
        configs:
            Generated configurations.
        configs_ansatze:
            Ansatz configurations.
        probabilities:
            Probabilities of the generated samples.
    """
    # Generate samples
    time_sample_s   = time.perf_counter()
    (states, logprobas), (configs, configs_ansatze), probabilities = smplr.sample()
    time_sample_f   = time.perf_counter() - time_sample_s
    logger.info(f"Time taken for first sampling: {time_sample_f:.4f} seconds", lvl=3)
    for _ in range(ntimes - 1):
        (states, logprobas), (configs, configs_ansatze), probabilities = smplr.sample()
    time_all        = time.perf_counter() - time_sample_s
    logger.info(f"Time taken for all sampling: {time_all:.4f} seconds.", lvl=3)
    logger.info(f"Time taken for each sampling: {(time_all - time_sample_f) / ntimes:.2e} seconds.", lvl=3)
    return (states, logprobas), (configs, configs_ansatze), probabilities

def test_network_quickness(net, backend, ns, ntimes = 1):
    '''
    Test the network quickness by generating random states and computing the ansatz.
    Parameters:
    net: 
        The neural network object.
    backend:
        The backend to use (numpy or jax).
    ns:
        Number of sites.
    ntimes:
        Number of times to test.
    '''

    states      = np.random.choice([-1,1], size=(10, ns), replace=True)
    params      = net.get_params()
    if backend == 'jax':
        states  = jnp.array(states)
    
    # first application
    time_s      = time.perf_counter()
    ansatz      = net(params, states)
    time_f      = time.perf_counter() - time_s
    logger.info(f"Time taken for first application: {time_f:.4f} seconds", lvl=3)
    for _ in range(ntimes - 1):
        ansatz  = net(params, states)
    time_all    = time.perf_counter() - time_s
    ansatz      = ansatz.reshape(-1, 1)
    logger.info(f"Time taken for all applications: {time_all:.4f} seconds.", lvl=3)
    logger.info(f"Time taken for each application: {(time_all - time_f) / ntimes:.2e} seconds.", lvl=3)
    logger.info(f"Shape of ansatz: {ansatz.shape}", lvl=3)
    return time_all + time_f

def _apply_hamiltonian(backend, ham, states, batch_size,
                    configs, configs_ansatze, probabilities,
                    net):
    """
    Apply the Hamiltonian to the states.
    
    Parameters:
    backend:
        The backend to use (numpy or jax).
    ham:
        The Hamiltonian object.
    states:
        The states to apply the Hamiltonian to.
    
    Returns:
        The result of applying the Hamiltonian to the states.
    """
    if backend == 'jax':
        states  = jnp.array(states)
        
        applied_energies, applied_energies_m, applied_energies_std = \
            net_utils.jaxpy.apply_callable_batched_jax(
                func            = ham.get_loc_energy_jax_fun(),
                states          = configs,
                sample_probas   = probabilities,
                logprobas_in    = configs_ansatze,
                logproba_fun    = net.get_apply(use_jax=True)[0],
                parameters      = net.get_params(),
                batch_size      = batch_size
            )
    else:
        applied_energies, applied_energies_m, applied_energies_std = \
            net_utils.numpy.apply_callable_batched_np(
                func            = ham.get_loc_energy_np_fun(),
                states          = configs,
                sample_probas   = probabilities,
                logprobas_in    = configs_ansatze,
                logproba_fun    = net.get_apply(use_jax=False)[0],
                parameters      = net.get_params(),
                batch_size      = batch_size
            )
    return applied_energies, applied_energies_m, applied_energies_std

def test_hamiltonian_quickness(backend, ham, states, batch_size,
                    configs, configs_ansatze, probabilities,
                    net, ntimes = 1):
    '''
    Test the Hamiltonian quickness by generating random states and computing the ansatz.
    Parameters:
    backend:
        The backend to use (numpy or jax).
    ham:
        The Hamiltonian object.
    states:
        The states to apply the Hamiltonian to.
    batch_size:
        The batch size to use.
    configs:
        The configurations to apply the Hamiltonian to.
    configs_ansatze:
        The ansatz configurations.
    probabilities:
        The probabilities associated with the configurations.
    net:
        The neural network object.
    ntimes:
        Number of times to test.
    '''
    
    time_s  = time.perf_counter()
    r, m, s = _apply_hamiltonian(backend, ham, states, batch_size, configs, configs_ansatze, probabilities, net)
    logger.info(f"Shape of applied energies: {r.shape}", lvl=3)
    logger.info(f"Mean of applied energies: {m}", lvl=3)
    logger.info(f"Standard deviation of applied energies: {s}", lvl=3)
    time_f  = time.perf_counter() - time_s
    logger.info(f"Time taken for first application: {time_f:.4f} seconds", lvl=3)
    for _ in range(ntimes - 1):
        r, m, s = _apply_hamiltonian(backend, ham, states.copy(), batch_size, configs, configs_ansatze, probabilities, net)
    time_all = time.perf_counter() - time_s
    logger.info(f"Time taken for all applications: {time_all:.4f} seconds.", lvl=3)
    logger.info(f"Time taken for each application: {(time_all - time_f) / ntimes:.2e} seconds.", lvl=3)
    return time_all + time_f

if __name__ == '__main__':
    
    start_time_script = time.perf_counter()     # Start total script timer
    
    # add the argument parser for jax or numpy
    parser = argparse.ArgumentParser(description='Test the NQS Sampler')
    parser.add_argument('--backend', type=str, default='jax', choices=['numpy', 'jax'], help='Backend to use (numpy or jax)')
    parser.add_argument('--seed', type=int, default=1725, help='Random seed for reproducibility')
    parser.add_argument('--sites', type=int, default=10, help='Number of sites')
    parser.add_argument('--dtype', type=str, default='complex128', choices=['float32', 'float64', 'complex64', 'complex128'], help='Data type for the backend')
    parser.add_argument('--nhidden', type=int, default=5, help='Number of hidden parameters')
    
    # sampler related arguments
    parser.add_argument('--nsamples', type = int, default = 10,   help = 'Number of samples to generate')
    parser.add_argument('--nchains',  type = int, default = 1,    help = 'Number of chains to use')
    parser.add_argument('--nsweeps',  type = int, default = 10,   help = 'Number of sweeps to use')
    parser.add_argument('--ntherm',   type = int, default = 10,   help = 'Number of thermal steps to use')
    parser.add_argument('--ntimes',   type = int, default = 5,    help = 'Number of times to generate samples')
    parser.add_argument('--nbatches', type = int, default = 1,    help = 'Number of batches to use')
    print_arguments(parser, logger)         # Print the arguments to the logger
    
    # Parse the arguments
    args                                    = parser.parse_args()
    backend_str                             = args.backend
    seed                                    = args.seed
    be_modules                              = get_backend(backend_str, random=True, seed=seed, scipy=True)
    bckm, (rng, rng_k), backend_sp          = be_modules if isinstance(be_modules, tuple) else (be_modules, (None, None), None)
    
    # printout
    logger.info(f"Using backend: {backend_str.upper()}")
    logger.info(f"Using data type: {args.dtype}")
    logger.info(f"Random seed: {seed}")
    
    ns                                      = args.sites
    st_shape                                = (ns, )
    activations                             = ('relu', 'relu')
    net                                     = SimpleNet(act_fun =   activations,
                                                input_shape     =   st_shape,
                                                output_shape    =   (1, ),
                                                layers          =   (args.nhidden, ),
                                                backend         =   bckm,
                                                dtype           =   args.dtype)

    # set the sampler
    numchains                               = args.nchains
    numsamples                              = args.nsamples
    numsweeps                               = args.nsweeps
    numtherm                                = args.ntherm
    numtotalsamples                         = args.ntimes
    
    sampler = MCSampler(        net         = net,              # set the variational model representation
                                shape       = st_shape,         # shape of the input state
                                rng         = rng,              # random number generator
                                rng_k       = rng_k,            # random number generator key (if applicable - jax)
                                numchains   = numchains,        # Number of chains to use
                                numsamples  = numsamples,       # Number of samples to generate for each chain
                                sweep_steps = numsweeps,        # Number of sweeps for the sampler
                                backend     = bckm,       # Backend for the sampler
                                therm_steps = numtherm,         # number of thermalization steps before storing samples
                                )
    start_time_sampling                     = time.perf_counter()
    setup_duration                          = start_time_sampling - start_time_script
    
    #! Create the hamiltonian
    ham_dtype           = float
    par                 = 0.5 + 1j * 0.5 if ham_dtype == complex else 0.5
    ham                 = DummyHamiltonian(hilbert_space=None, ns = ns, backend = bckm, dtype = ham_dtype)
    batch_size          = args.nbatches
    
    
    #! Print the information
    logger.info(f"Number of sites: {ns}", lvl = 1)
    logger.info(f"Number of samples: {numsamples}", lvl = 1)
    logger.info(f"Number of chains: {numchains}", lvl = 1)
    logger.info(f"Number of sweeps: {numsweeps}", lvl = 1)
    logger.info(f"Number of thermal steps: {numtherm}", lvl = 1)
    logger.info(f"Batch size: {batch_size}", lvl = 1)
    logger.info(f"Succesfully created the sampler {sampler}", lvl = 2)
    logger.info(f"Succesfully created the net {net}", lvl = 2)
    logger.info(f"Succesfully created the Hamiltonian {ham}", lvl = 2)
    
    
    #! Create the solver
    nqs                 = nqsmodule.NQS(
                            net         = net,
                            sampler     = sampler,
                            hamiltonian = ham,
                            lower_betas = None,
                            lower_states= None,
                            seed        = seed,
                            beta        = 1.0,
                            shape       = st_shape,
                            backend     = bckm,
                            nthread     = 1,
                            batch_size  = batch_size,
                        )
    
    #! Get the samples
    logger.info(f"Creating the samples for {args.ntimes} times...", lvl=0)
    (states, logprobas), (configs, configs_ansatze), probabilities = generate_samples(sampler, ntimes=args.ntimes)
    end_time_sampling                       = time.perf_counter()
    sampling_duration                       = (end_time_sampling - start_time_sampling) / args.ntimes
    # Typically (nchains * nsamples, N) or (nchains, nsamples, N)
    logger.info(f"Generated samples shape (configs): {configs.shape}", lvl=1)
    # Typically (nchains * nsamples, 1) or similar
    logger.info(f"Generated log-amplitudes shape (configs_ansatze): {configs_ansatze.shape}", lvl=1)
    # Typically (nchains * nsamples,) or similar
    logger.info(f"Generated log probabilities shape (logprobas): {logprobas.shape}", lvl=1)
    # Typically (nchains * nsamples,) or similar
    logger.info(f"Generated all probabilities shape: {probabilities.shape}", lvl=1)
    
    #! Test the network quickness
    logger.info(f"Testing the network quickness...", lvl=0)
    time_jax_f                      = test_network_quickness(net, backend_str, ns, ntimes=args.ntimes)
    time_elapsed                    = time_jax_f - end_time_sampling
    
    #! Apply the ansatz
    logger.info(f"Applying the ansatz...", lvl=0)
    time_ansatz_s                   = time.perf_counter()
    ansatz                          = nqs(configs, batch_size=batch_size)
    end_time_ansatz                 = time.perf_counter()
    ansatz_duration                 = end_time_ansatz - time_ansatz_s
    logger.info(f"Ansatz shape: {ansatz.shape}", lvl=1)
    logger.info(f"Ansatz mean: {np.mean(ansatz)}", lvl=1)
    logger.info(f"Ansatz std: {np.std(ansatz)}", lvl=1)
    logger.info(f"Ansatz max: {np.max(ansatz)}", lvl=1)
    logger.info(f"Ansatz min: {np.min(ansatz)}", lvl=1)
    logger.info(f"Ansatz duration: {ansatz_duration:.4f} seconds", lvl=1)
    
    #! Apply the Hamiltonian
    logger.info(f"Applying the Hamiltonian...", lvl=0)
    time_ham_s                     = time.perf_counter()
    end_time_ham = test_hamiltonian_quickness(backend_str, ham, states, batch_size, configs, configs_ansatze, probabilities, net, ntimes=args.ntimes)
    ham_duration                    = end_time_ham
    logger.info(f"Applied energies duration: {ham_duration:.4f} seconds", lvl=1)
    
    #! Evaluate through solver
    logger.info(f"Evaluating through solver...", lvl=0)
    time_solver_s                   = time.perf_counter()
    (configs, ansatze), probabilities, (v, means, stds) = nqs.evaluate_fun(
        states_and_psi  = (configs, configs_ansatze),
        probabilities   = probabilities,
        functions       = None,     # means that we evaluate energy
        batch_size      = batch_size
    )
    end_time_solver                 = time.perf_counter()
    logger.info(f"Solver evaluation duration: {end_time_solver - time_solver_s:.4f} seconds", lvl=1)
    logger.info(f"Mean of applied energies: {means}", lvl=1)
    logger.info(f"Standard deviation of applied energies: {stds}", lvl=1)
    
    #! Timing Summary
    end_time_script                 = time.perf_counter()
    total_duration                  = end_time_script - start_time_script
    post_sampling_duration          = total_duration - setup_duration - sampling_duration - time_jax_f - ansatz_duration - ham_duration
    logger.info(f"Total duration: {total_duration:.4f} seconds", lvl=0)
    logger.info(f"Setup duration: {setup_duration:.4f} seconds", lvl=0)

    phases                          = {
        "Setup"             : setup_duration,
        "Sampling"          : sampling_duration,
        "Post-processing"   : post_sampling_duration,
        "Network Quickness" : time_jax_f,
        "Ansatz Application": ansatz_duration,
        "Hamiltonian"       : ham_duration,
        "Solver Evaluation" : end_time_solver - time_solver_s
        }
    extra_timing_info = []
    if backend_str == 'jax':
        extra_timing_info.append("(Note: First JAX run may include compile time in 'Sampling')")

    perf_string                     = "Sampling Performance: N/A (duration or samples is zero)"
    if sampling_duration > 1e-6 and numtotalsamples > 0:
        samples_per_sec             = numtotalsamples / sampling_duration
        perf_string                 = f"Sampling Performance: {samples_per_sec:.2f} samples/sec"
    else:
        perf_string                 = "Sampling Performance: N/A (duration or samples is zero)"
    extra_timing_info.append(perf_string)

    log_timing_summary(
        logger          = logger,
        phase_durations = phases,
        total_duration  = total_duration,
        extra_info      = extra_timing_info,
        lvl             = 0
    )
    
#! End of the script