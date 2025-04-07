'''
file        :   test/nqs_sampler.py
author      :   Maksymilian Kliczkowski

Script to test the NQS Sampler using Monte Carlo methods.
It sets up a simple neural network, configures a sampler,
generates samples, and prints information about the process.
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
    from QES.Solver.MonteCarlo.sampler import Sampler, MCSampler, get_backend, _JAX_AVAILABLE
    from QES.general_python.ml.net_impl.net_simple import SimpleNet
    from QES.general_python.common.plot import MatrixPrinter
    from QES.general_python.common.flog import Logger, print_arguments, log_timing_summary
except ImportError:
    print("Error: Could not import QES modules. Ensure the QES package is installed or the path is correct.")
    print(f"Attempted to add parent directory: {parent_dir}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Initialize logger
logger = Logger()
logger.title("Starting NQS Sampler Test Script...", 50, '#', lvl = 0)

import numpy as np
if _JAX_AVAILABLE:
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
    activations                             = ('relu', 'sigmoid')
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
    
    #! Print the information
    logger.info(f"Number of sites: {ns}", lvl = 1)
    logger.info(f"Number of samples: {numsamples}", lvl = 1)
    logger.info(f"Number of chains: {numchains}", lvl = 1)
    logger.info(f"Number of sweeps: {numsweeps}", lvl = 1)
    logger.info(f"Number of thermal steps: {numtherm}", lvl = 1)
    logger.info(f"Succesfully created the sampler {sampler}", lvl = 2)
    logger.info(f"Succesfully created the net {net}", lvl = 2)
    
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
    
    #! Flow of samples
    logger.info("Detailed Flow of the Configurations:\n", lvl=0)
    logger.info(f"Total number of configurations: {len(configs)}", lvl=1)
    logger.info(f"Total number of states: {len(states)}", lvl=1)
    logger.info(f"Total number of probabilities: {len(probabilities)}\n", lvl=1)

    if ns < 6:
        num_to_print = min(3, len(configs))
        for i in range(num_to_print):
            logger.info(f"Configuration flow {i}", lvl=1)
            logger.info(f"Configuration \\(\\mathbf{{S}}_{{{i}}}\\):", lvl=2)
            MatrixPrinter.print_vector(configs[i])
            
            logger.info(f"Ansatz \\(f(\\mathbf{{S}}_{{{i}}}\\):", lvl=2)
            MatrixPrinter.print_vector(configs_ansatze[i].T)
            
            logger.info(f"Probability \\(P(\\mathbf{{S}}_{{{i}}}\\):", lvl=2)
            MatrixPrinter.print_vector(probabilities[i].T)
            logger.title("", 30, '-', lvl = 1)
        
    #! If is jax, get the sampler and check how fast it is as well
    time_jax_s = time.perf_counter()
    if backend_str == 'jax':
        sampler_fun = sampler.get_sampler_jax(num_samples=args.nsamples, num_chains=args.nchains)
        
        (states, logprobas), (configs, configs_ansatze), probabilities = sampler_fun(
            states_init = sampler.states,
            rng_k_init  = rng_k,
            params      = None,
        )
        time_jax_f  = time.perf_counter() - time_jax_s
        logger.info(f"Time taken for first JAX sampling: {time_jax_f:.4f} seconds", lvl=3)
        for i in range(args.ntimes - 1):
            (states, logprobas), (configs, configs_ansatze), probabilities = sampler_fun(
                states_init = sampler.states,
                rng_k_init  = rng_k,
                params      = None,
            )
        time_all            = time.perf_counter() - time_jax_s
        logger.info(f"Time taken for all JAX sampling: {time_all:.4f} seconds.", lvl=3)
        logger.info(f"Time taken for each JAX sampling: {(time_all - time_jax_f) / args.ntimes:.2e} seconds.", lvl=3)
        time_jax_f          = time.perf_counter() - time_jax_s
    else:
        time_jax_f  = 0.0
    
    #! Timing Summary
    end_time_script                 = time.perf_counter()
    total_duration                  = end_time_script - start_time_script
    post_sampling_duration          = total_duration - setup_duration - sampling_duration - time_jax_f

    phases                          = {
        "Setup"             : setup_duration,
        "Sampling"          : sampling_duration,
        "Post-processing"   : post_sampling_duration,
        "JAX-sampler"       : time_jax_f
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