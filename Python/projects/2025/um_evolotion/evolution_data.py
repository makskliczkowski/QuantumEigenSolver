import os, sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path

#! -------------------------------------------------------
os.environ['QES_BACKEND']   = 'numpy'   # Backend for numerical operations
os.environ['QES_LOGFILE']   = '0'       # Disable logging to file
#! -------------------------------------------------------

# --------------------------------

cwd         = Path.cwd()
mod_path    = Path(__file__).resolve()
file_path   = Path(__file__).parent
qes_path    = Path(__file__).parent.parent.parent.parent
lib_path    = qes_path / 'QES'
gen_python  = lib_path / 'general_python'
print("-> Current working directory:", cwd)
print("-> File path:", file_path)
print("-> Module path:", mod_path)
print("-> QES path:", qes_path)
print("-> General Python path:", gen_python)
print("-> Library path:", lib_path)
sys.path.insert(0, str(lib_path))
sys.path.insert(0, str(gen_python))

try:
    from QES.general_python.maths.statistics import HistogramAverage
    
    # MODEL
    from QES.Algebra.hamil import Hamiltonian
    from QES.Algebra.Model.Interacting.Spin.ultrametric import UltrametricModel
    from QES.Algebra.Model.Noninteracting.plrb import PowerLawRandomBanded
    
    # general
    from QES.general_python.common.directories import Directories
    from QES.general_python.common.hdf5_lib import HDF5Handler
    from QES.general_python.common.timer import Timer
    
except ImportError:
    print("Error importing QES modules. Please ensure QES is correctly installed and the PYTHONPATH is set.")
    sys.exit(1)

# --------------------------------

@dataclass
class EvolutionData:
    #! model identity and content
    
    ns                  : int
    alpha               : float
    hilbert_size        : int

    realizations        : int
    
    # evolution
    time_num            : int

    # other
    bandwidth_data      : float
    mean_lvl_space      : float
    
    # operators
    operators           : Dict[str, Any] = field(default_factory=dict)
    operators_mat       : Dict[str, np.ndarray] = field(default_factory=dict)

    additional_params   : Dict[str, Any] = field(default_factory=dict)
    modelstr            : str = field(init=False)
    
    # flags for simulation
    uniform             : bool = True
    dtype               : np.dtype = np.float64
    
    #! ------------------
    # arrays to allocate
    #! ------------------

    # energy related
    mean_energy         : float = field(init=False)
    energies            : np.ndarray = field(init=False)
    bandwidths          : np.ndarray = field(init=False)
    sigma_es            : np.ndarray = field(init=False)

    # states related
    ldos                : np.ndarray = field(init=False)
    iprs                : Dict[float, np.ndarray] = field(default_factory=dict)
    entros              : Dict[float, np.ndarray] = field(default_factory=dict)
    
    # operators
    diagonals_operators : Dict[str, np.ndarray] = field(default_factory=dict)
    diagonal_ensemble   : Dict[str, np.ndarray] = field(default_factory=dict)
    fidelity_susc_av    : Dict[str, np.ndarray] = field(init=False)
    fidelity_susc_typ   : Dict[str, np.ndarray] = field(init=False)

    # time evo
    time_vals           : Dict[str, np.ndarray] = field(default_factory=dict)
    vn_entropies        : np.ndarray = field(init=False)
    tsalis_entropies    : np.ndarray = field(init=False)
    schmidt_gaps        : np.ndarray = field(init=False)
    iprs_quench         : Dict[float, np.ndarray] = field(default_factory=dict)
    par_ent_quench      : Dict[float, np.ndarray] = field(default_factory=dict)
    quench_energies     : np.ndarray = field(init=False)

    # FFT data
    fft_omegas          : np.ndarray = field(init=False) #! NEW
    fft_results         : Dict[str, np.ndarray] = field(init=False) #! NEW
    fft_n               : Dict[str, np.ndarray] = field(init=False) #! NEW

    survival_proba      : np.ndarray = field(init=False) #! NEW

    # histograms
    indices_omegas      : np.ndarray = field(init=False)
    edges               : np.ndarray = field(init=False)
    h_av                : Dict[str, HistogramAverage] = field(default_factory=dict)
    h_typ               : Dict[str, HistogramAverage] = field(default_factory=dict)
    h_av_full           : Dict[str, HistogramAverage] = field(default_factory=dict) #! NEW
    time_steps          : Dict[str, HistogramAverage] = field(default_factory=dict)
    
    k_functions         : HistogramAverage = field(default_factory=dict) #! NEW
    k_functions_full    : HistogramAverage = field(default_factory=dict) #! NEW

    s_functions         : Dict[str, HistogramAverage] = field(default_factory=dict) #! NEW
    s_functions_full    : Dict[str, HistogramAverage] = field(default_factory=dict) #! NEW

#! ------------------

    def __post_init__(self):
        '''
        Initialize the core data structures for the evolution simulation.
        '''
        hs, nr, dt                  = self.hilbert_size, self.realizations, self.dtype

        # core arrays
        self.ldos                   = np.zeros((nr, hs), dtype=dt)
        self.energies               = np.zeros((nr, hs), dtype=dt)
        self.sigma_es               = np.zeros(nr, dtype=dt)
        self.bandwidths             = np.zeros(nr, dtype=dt)

        #! State properties
        self.iprs                   = { 2.0 : np.zeros((nr, hs), dtype=dt), 0.5 : np.zeros((nr, hs), dtype=dt)}
        self.entros                 = { 1.0 : np.zeros((nr, hs), dtype=dt), 2.0 : np.zeros((nr, hs), dtype=dt)}

        #! Operators
        self.diagonals_operators    = {k: np.zeros((nr, hs), dtype=dt) for k in self.operators.keys()}
        self.fidelity_susc_av       = {k: np.zeros((nr, hs), dtype=dt) for k in self.operators.keys()}
        self.fidelity_susc_typ      = {k: np.zeros((nr, hs), dtype=dt) for k in self.operators.keys()}
        
        #! Time evolution
        self.survival_proba         = np.zeros((nr, self.time_num), dtype=dt) #! NEW
        self.time_vals              = {k: np.zeros((nr, self.time_num), dtype=dt) for k in self.operators.keys()}
        self.diagonal_ensembles     = {k: np.zeros(nr, dtype=dt) for k in self.operators.keys()}

        #! Quench extras
        self.vn_entropies           = np.zeros(nr, dtype=dt)
        self.tsalis                 = np.zeros(nr, dtype=dt)
        self.schmidt_gaps           = np.zeros(nr, dtype=dt)
        self.iprs_quench            = {q: np.zeros(nr, dtype=dt) for q in (0.5, 1.0, 2.0)}
        self.par_ent_quench         = {q: np.zeros(nr, dtype=dt) for q in (0.5, 1.0, 2.0)}
        self.quench_energies        = np.zeros(nr, dtype=dt)

        #! histograms (shared edges across operators)
        self.indices_omegas         = np.zeros((0, 2), dtype=np.int64)
    
    def initialize_histograms_and_time(self) -> None:
        hs, dt                      = self.hilbert_size, self.dtype
        o_min, o_max                = 0.01/hs, 3.0*self.bandwidth_data
        n_bins                      = int(20*self.ns)
        self.edges                  = HistogramAverage.uniform_log(n_bins, v_max=o_max, v_min=o_min)
        self.h_av                   = {name: HistogramAverage(n_bins, edges=self.edges) for name in self.operators}
        self.h_typ                  = {name: HistogramAverage(n_bins, edges=self.edges) for name in self.operators}
        self.h_av_full              = {name: HistogramAverage(n_bins, edges=self.edges) for name in self.operators}

        self.k_functions            = HistogramAverage(n_bins, edges=self.edges)
        self.k_functions_full       = HistogramAverage(n_bins, edges=self.edges)

        self.s_functions            = {name: HistogramAverage(n_bins, edges=self.edges) for name in self.operators}
        self.s_functions_full       = {name: HistogramAverage(n_bins, edges=self.edges) for name in self.operators}

        #! time grid
        heisenberg_time = 2.0*np.pi / self.mean_lvl_space
        if self.uniform:
            dt_step = 2.0*np.pi / self.bandwidth_data
            t0      = heisenberg_time - self.time_num*dt_step/2.0
            if t0 < 0.0:
                t0  = heisenberg_time/100.0
            self.time_steps = np.linspace(t0, t0 + dt_step*self.time_num, num=self.time_num, dtype=dt)
        else:
            self.time_steps = np.logspace(-2, np.log10(hs*100), num=self.time_num, dtype=dt)
        
        #! FFT data
        try:
            # 1) calculate the time delta and number of time steps
            # 2) calculate the FFT frequencies and take only the positive frequencies
            # 3) calculate the FFT of the time evolution data, normalize it, and take the positive frequencies
            # 4) normalize the FFT by the number of samples
            if self.uniform:
                data_evo_time           = self.time_steps
                time_delta              = data_evo_time[1] - data_evo_time[0]
                time_num                = data_evo_time.shape[-1]
                self.fft_omegas         = np.fft.rfftfreq(time_num, d=time_delta) * 2 * np.pi
                self.fft_results        = {name: np.zeros((self.realizations, time_num//2 - 1), dtype=dt) for name in self.operators}
                self.fft_n              = {name: np.zeros(self.realizations, dtype=dt) for name in self.operators}
            else:
                self.fft_omegas         = np.empty((0,), dtype=dt)
                self.fft_results        = {name: np.zeros((self.realizations, self.fft_omegas.shape[0]), dtype=dt) for name in self.operators}
                self.fft_n              = {name: np.zeros(self.realizations, dtype=dt) for name in self.operators}
        except Exception as e:
            print(f"Error in FFT calculation: {e}")

    ##########################################################
    #! saving helpers
    ##########################################################
    
    def _build_stat_dict(self, completed: int) -> dict:
        data_stat = {
            'bandwidth'                       : self.bandwidths[:completed],
            'sigma_e'                         : self.sigma_es[:completed],
            'energies'                        : self.energies[:completed],
            'ldos'                            : self.ldos[:completed],
            'entropy/quench/von_neumann/1.0'  : self.vn_entropies[:completed],
            'entropy/quench/tsallis/2.0'      : self.tsalis[:completed],
            'entropy/quench/schmidt_gap'      : self.schmidt_gaps[:completed],
            'completed_realizations'          : completed,
        }
        # update the rest
        for q in self.iprs_quench:
            data_stat[f'iprs/quench/{q}'] = self.iprs_quench[q][:completed]
        for q in self.par_ent_quench:
            data_stat[f'participation/quench/{q}'] = self.par_ent_quench[q][:completed]
        for q in self.iprs:
            data_stat[f'iprs/{q}'] = self.iprs[q][:completed]
        for q in self.entros:
            data_stat[f'entropy/eigenbasis/participation/{q}'] = self.entros[q][:completed]
        return data_stat
    
    def _build_data_hist(self, completed: int) -> dict:
        #! build the histogram data
        any_hist = next(iter(self.h_av.values()))
        d = {
            'historgram/edges'          : self.edges,
            'historgram/counts'         : any_hist.counts(),
            'completed_realizations'    : completed,
        }
        
        #! update with standard f-functions
        for name in self.operators:
            d[f'historgram/{name}/average/values'] = self.h_av[name].averages_av()
            d[f'historgram/{name}/typical/values'] = self.h_typ[name].averages_av(is_typical=True)

        #! NEW, update with full f-functions
        for name in self.operators:
            d[f'historgram/{name}/full/average/values'] = self.h_av_full[name].averages_av()

        #! NEW, k_functions
        d[f'k_function/values'] = self.k_functions.averages_av()

        #! NEW, update with full k-functions
        d[f'k_function/full/values'] = self.k_functions_full.averages_av()

        #! NEW, s_functions
        for name in self.operators:
            d[f's_function/{name}/values'] = self.s_functions[name].averages_av()

        #! NEW, update with full s-functions
        for name in self.operators:
            d[f's_function/{name}/full/values'] = self.s_functions_full[name].averages_av()

        #! NEW, calculate s_times_f
        for name in self.operators:
            k_times_hist                            = d[f'k_function/values'] * d[f'historgram/{name}/average/values']
            k_times_hist[np.isnan(k_times_hist)]   /= np.trapz(k_times_hist, self.edges)
            d[f'k_times_f/{name}/values']           = k_times_hist
            # full
            k_times_hist_full                               = d[f'k_function/full/values'] * d[f'historgram/{name}/full/average/values']
            k_times_hist_full[np.isnan(k_times_hist_full)] /= np.trapz(k_times_hist_full, self.edges)
            d[f'k_times_f/{name}/full/values']              = k_times_hist_full
        
            # mix, use full k and normal f
            k_times_hist_mix                                = d[f'k_function/full/values'] * d[f'historgram/{name}/average/values']
            k_times_hist_mix[np.isnan(k_times_hist_mix)]   /= np.trapz(k_times_hist_mix, self.edges)
            d[f'k_times_f/{name}/mix/values']               = k_times_hist_mix

        return d

    def _build_diag_dict(self, completed: int) -> dict:
        d = { name: self.diagonals_operators[name][:completed] for name in self.diagonals_operators }
        d['completed_realizations'] = completed
        return d

    def _build_evo_dict(self, completed: int) -> dict:
        d = {
            'time'                              : self.time_steps,
            'time_evolution/quench/energy'      : self.quench_energies[:completed],
            'time_evolution/quench/survival'    : self.survival_proba[:completed],
            'completed_realizations'            : completed,
        }
        if self.uniform:
            d[f'fft/fft_omega']   = self.fft_omegas
            
        for name in self.operators:
            d[f'time_evolution/{name}/expectation'] = self.time_vals[name][:completed]
            d[f'time_evolution/{name}/diag_ens']    = self.diagonal_ensembles[name][:completed]
            # fft
            if self.uniform:
                for name in self.operators:
                    d[f'fft/{name}/fft']     = self.fft_results[name][:completed]
                    d[f'fft/{name}/fft_n']   = self.fft_n[name][:completed]

        return d
    
    def _build_fid_dict(self, completed: int) -> dict:
        d = {}
        for name in self.fidelity_susc_av:
            d[f'agp/fidelity/{name}/0.0']   = self.fidelity_susc_typ[name][:completed]
            d[f'agp/fidelity/{name}/bw_nh'] = self.fidelity_susc_av[name][:completed]
        return d
    
    ##########################################################
    
    def save_all(self, directory, rand_num: int, completed: int) -> None:
        HDF5Handler.save_hdf5(directory=directory, filename=f'stat_{rand_num}.h5', data=self._build_stat_dict(completed))
        HDF5Handler.save_hdf5(directory=directory, filename=f'hist_{rand_num}.h5', data=self._build_data_hist(completed))
        HDF5Handler.save_hdf5(directory=directory, filename=f'diag_{rand_num}.h5', data=self._build_diag_dict(completed))
        HDF5Handler.save_hdf5(directory=directory, filename=f'evo_{rand_num}.h5',  data=self._build_evo_dict(completed))
        HDF5Handler.save_hdf5(directory=directory, filename=f'fid_{rand_num}.h5',  data=self._build_fid_dict(completed))
        
#############################

def create_model(model: str, ns: int, param: float, seed: int, **kwargs):
    if model == 'um' or model == 'ultrametric':
        return UltrametricModel(ns=ns, n=kwargs.get('n', 2), alphas=param, backend=np, seed=seed)
    elif model == 'plrb' or model == 'powerlaw':
        return PowerLawRandomBanded(ns=ns, many_body=True, a=param, b=1.0, backend=np, seed=seed)
    else:
        raise ValueError(f"Unknown model: {model}")