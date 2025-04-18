#include "../../include/user_interface/user_interface.h"
#include "armadillo"
#include <memory>
#include <vector>

// ----------------------------------------------------------------
// File name prefixes
// ----------------------------------------------------------------
#define PREFIX(name, random_tag, extension) dir , name + random_tag + extension
#define STAT_PREFIX PREFIX("stat", random_tag, extension)
#define     KEY_ME                      "/ME"
#define 	KEY_EPS                     "eps_micro"						// for a given energy density
#define     KEY_MEAN_LVL                "mean_lvl_spacing"
#define     KEY_HEIS_TIME_GAMMA         "heis_time_gamma"
#define     KEY_HEIS_TIME_AROUND_MEAN   "heis_time_around_mean"
#define     KEY_HEIS_TIME_AROUND_MEAN_T "heis_time_around_mean_typ"
#define     KEY_ENERGIES                "energies"
#define LDOS_PREFIX PREFIX("ldos", random_tag, extension)
#define DIAG_PREFIX PREFIX("diag", random_tag, extension)
#define EDENS_PREFIX PREFIX("energydens", random_tag, extension)
#define     KEY_MEAN                    "mean"							// Tr<psi|H|psi> / N_h
#define     KEY_MEAN_STATE              "mean_state"					// <psi_0|H|psi_0>
#define     KEY_MEAN_STATE2             "mean_state2"					// <psi_0|H|psi_0>^2
#define EVO_PREFIX PREFIX("evo", random_tag, extension)
#define     KEY_TIME                    "time"							// time
#define 	KEY_TIME_SHORT				"time/short"					// time short
#define 	KEY_TIME_MEDIUM				"time/medium"					// time medium
#define 	KEY_TIME_MIDDLE				"time/middle"					// time middle
#define 	KEY_TIME_LONG				"time/long"						// time long
#define     KEY_ENT_ENTROPY_PREFIX      "entanglement_entropy/"			// entanglement entropy
#define     KEY_PARTICIPATION           "participation_entropy"			// participation entropy S_p = - sum_i (|psi_i|^4)
#define AVS_PREFIX PREFIX("avs", random_tag, extension)

#define SUFFIX_ME                       "/ME/"							// for mean energy state
#define SUFIX_BIPARTITE 			 	"/bipartite/"					// for bipartite entanglement
#define SUFFIX_EPS(x)                   "/eps=" + STR(x)	 			// for a given energy density		
#define SUFFIX_MICRO_ME                 "/micro/ME/1"					// for microcanonical ensemble
#define SUFFIX_MICRO_ME2                "/micro/ME/2"					// for microcanonical ensemble squared
#define SUFFIX_MICRO_ED(x)              "/micro/eps=" + STR(x) + "/1"	// for microcanonical ensemble
#define SUFFIX_MICRO2_ED(x)             "/micro/eps=" + STR(x) + "/2"	// for microcanonical ensemble squared
#define SUFFIX_DIAGONAL_ME              "/diagonal/ME"
#define SUFFIX_DIAGONAL_ED(x)           "/diagonal/eps=" + STR(x)

// ----------------------------------------------------------------

constexpr static bool check_saving_size(u64 _Nh, uint _r)
{
	return ((_Nh > ULLPOW(14)) ||(BETWEEN((size_t)std::log2(_Nh), 10, 15) && (_r % 10 == 0)) || ((_Nh <= 1024) && (_r % 50 == 0)));
}

constexpr static bool check_multithread_operator(u64 _Nh)
{
	return (_Nh <= ULLPOW(9));
}

// ----------------------------------------------------------------

template <typename T>
struct TimeEvoParams
{
    size_t _Ns;
    size_t _Nh;
    size_t _hs_fraction_diag        = 0;
    bool _isQuadratic               = true;
    bool _isManyBody                = true;

    long double _bw_est;
    long double _dtau_est;
    long double _heisenberg_time_est;
    long double _thouless_est;
    long long   _ntimes             = 100000;
    long long   _nrealiz            = 1;

    // stores the operators
    std::vector<std::shared_ptr<Operators::Operator<double>>> _ops;
    std::vector<std::string> _opsN;

    // stores the model info
    std::string modelInfo;
    std::string dir                 = "ETH_MAT_TIME_EVO";
    std::string randomStr;
    std::string extension;

    // stores the time evolution
    bool _uniform_time              = true;
    arma::Col<double> _timespace;

    // stores the check vectors
    v_1d<double> _to_check_microcanonical_eps;
    v_1d<double> _energy_densities;
	v_1d<int> _entropies_sites;
    TimeEvoParams()
    {
        _energy_densities               = { 0.1, 0.2, 0.3, 0.4                          };
        _entropies_sites                = { 1, int(_Ns / 2), (int)_Ns                   };
        _to_check_microcanonical_eps    = { 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4    };
    }

    // ------------------------------------------------------------
    //! Functions
    // ------------------------------------------------------------

    void set_timespace()
    {
		if (!_uniform_time)
		{
			_timespace = arma::logspace(-2, std::log10(_heisenberg_time_est * 1000), _ntimes);
		}
		else
		{	
			const long long start_short 	= 0;
			const long long start_medium 	= _thouless_est * 1e-2;
			const long long start_middle 	= _thouless_est * 0.5;
			const long long start_long 		= _heisenberg_time_est * 1e-2;
			const long long end_short 		= start_short + (_ntimes * _dtau_est);
			const long long end_medium 		= start_medium + (_ntimes * _dtau_est);
			const long long end_middle 		= start_middle + (_ntimes * _dtau_est);
			const long long end_long 		= start_long + (_ntimes * _dtau_est);

			// Create a uniform distribution of time points
			arma::Col<double> _short_times 	= arma::linspace(start_short, end_short, _ntimes);
			arma::Col<double> _medium_times = arma::linspace(start_medium, end_medium, _ntimes);
			arma::Col<double> _middle_times = arma::linspace(start_middle, end_middle, _ntimes);
			arma::Col<double> _long_times 	= arma::linspace(start_long, end_long, _ntimes);
			const long long _total_size		= (_short_times.size() + _medium_times.size() + _middle_times.size() + _long_times.size());
			
			// Combine all the time ranges
			_timespace.set_size(_total_size);
			_timespace.subvec(0, _short_times.size() - 1) = _short_times;
			_timespace.subvec(_short_times.size(), _short_times.size() + _medium_times.size() - 1) = _medium_times;
			_timespace.subvec(_short_times.size() + _medium_times.size(), _short_times.size() + _medium_times.size() + _middle_times.size() - 1) = _middle_times;
			_timespace.subvec(_short_times.size() + _medium_times.size() + _middle_times.size(), _total_size - 1) = _long_times;
		}
    }

    // ------------------------------------------------------------
	//! Set Hamiltonian parameters
	// ------------------------------------------------------------

    void set_hamil_params(MY_MODELS _type, std::shared_ptr<Hamiltonian<double>> _H, size_t n_tot, size_t n_real)
    {
        this->_Ns                       = n_tot;
        this->_Nh                       = _H->getHilbertSize();
        this->_hs_fraction_diag         = SystemProperties::hs_fraction_diagonal_cut(0.5, _Nh);
        this->_bw_est                   = HamiltonianHelpers::get_bandwidth_est(_type, _H->getMainParam(), _Nh) / TWOPI;
        this->_dtau_est                 = 1.0l / _bw_est;
        this->_heisenberg_time_est      = HamiltonianHelpers::get_heisenberg_time_est(_type, _H->getMainParam(), _Nh);
        this->_thouless_est             = HamiltonianHelpers::get_thouless_est(_type, _H->getMainParam(), _Nh);
        this->_ntimes                   = 100000;
        this->_nrealiz                  = n_real;
        // this->_isQuadratic              = _H->getIsQuadratic();
        // this->_isManyBody               = _H->getIsManyBody();
        this->_ops                      = {};
        this->_opsN                     = {};
    }   

	// ----------------------------------------------------------------
	//! Containers to hold all results
	// ----------------------------------------------------------------

	struct ResultContainers 
	{
		arma::Col<T> initial_state_me;                                  // [nh]
		arma::Mat<T> initial_state_ed;                                  // [nh x n_ops]

		// level statistics
		arma::Mat<double> mean_level_spacing;                           // [4 x n_real]
		arma::Mat<double> energies;                                     // [nh x n_real]

		// local density of states
		arma::Mat<double> ldos_me;                                      // [nh x n_real]
		std::vector<arma::Mat<double>> ldos_ed;                         // [n_edens][nh x n_real]

		// energy densities
		arma::Mat<double> energy_densities_me;                          // [3 x n_real] - mean energy, <state|H|state>, <state|H2|state>
		std::vector<arma::Mat<double>> energy_densities_ed;             // [n_edens][3 x n_real] - mean energy, <state|H|state>, <state|H2|state>

		// diagonal elements
		std::vector<arma::Mat<T>> diagonals;                            // [n_ops][nh x n_real]

		// time evolution
		arma::Col<double> times, times_s, times_m, times_mi, times_l;	// [n_times]
		std::vector<arma::Mat<T>> time_evo_me;                          // [n_ops][n_times x n_real]
		std::vector<std::vector<arma::Mat<T>>> time_evo_ed;             // [n_edens][n_ops][n_times x n_real]

		// entropies
		arma::Mat<double> time_participation_entropy;                   // [n_times x n_real]
		std::vector<arma::Mat<double> > time_ee;                        // [n_sites][n_times x n_real]
		arma::Mat<double> time_ee_bipartite;                            // [n_times x n_real]

		// microcanonical averages
		std::vector<arma::Mat<T>> microcanonical_me;                   	// [n_ops][n_eps x n_real]
		std::vector<arma::Mat<double>> microcanonical_me2;              // [n_ops][n_eps x n_real]
		std::vector<std::vector<arma::Mat<T>>> microcanonical_ed;      	// [n_edens][n_ops][n_eps x n_real]
		std::vector<std::vector<arma::Mat<double>>> microcanonical_ed2; // [n_edens][n_ops][n_eps x n_real]

		// diagonal ensembles
		arma::Mat<double> diagonal_me;                                  // [n_ops x n_real]
		std::vector<arma::Mat<double>> diagonal_ed;                     // [n_edens][n_ops x n_real]

		// finished realisations
		size_t realisation_count = 0;                                   // number of realizations finished

	} _r;

	// ----------------------------------------------------------------

	void save_for_realization(size_t r)
	{
		std::string random_tag = this->randomStr;

		//! level statistics
		saveAlgebraic(STAT_PREFIX, arma::vec(_r.mean_level_spacing.row(0).as_col()), KEY_MEAN_LVL, false);
		saveAlgebraic(STAT_PREFIX, arma::vec(_r.mean_level_spacing.row(1).as_col()), KEY_HEIS_TIME_GAMMA, true);
		saveAlgebraic(STAT_PREFIX, arma::vec(_r.mean_level_spacing.row(2).as_col()), KEY_HEIS_TIME_AROUND_MEAN, true);
		saveAlgebraic(STAT_PREFIX, arma::vec(_r.mean_level_spacing.row(3).as_col()), KEY_HEIS_TIME_AROUND_MEAN_T, true);
		saveAlgebraic(STAT_PREFIX, _r.energies, KEY_ENERGIES, true);

		//! LDOS
		saveAlgebraic(LDOS_PREFIX, _r.ldos_me, KEY_ME, false);
		for(size_t i = 0; i < _r.e_densities.size(); ++i)
			saveAlgebraic(LDOS_PREFIX, _r.ldos_ed[i], SUFFIX_EPS(_r.e_densities[i]), true);

		// energy densities obtained from the given initial state
		saveAlgebraic(EDENS_PREFIX, arma::vec(_r.energy_densities_me.row(0).t()), KEY_MEAN KEY_ME, false);			// mean energy
		saveAlgebraic(EDENS_PREFIX, arma::vec(_r.energy_densities_me.row(1).t()), KEY_MEAN_STATE KEY_ME, true);		// <state|H|state>
		saveAlgebraic(EDENS_PREFIX, arma::vec(_r.energy_densities_me.row(2).t()), KEY_MEAN_STATE2 KEY_ME, true);	// <state|H|state>^2
		for(size_t i = 0; i < _r._energy_densities_ed.size(); ++i) 
		{
			const std::string tag = SUFFIX_EPS(this->_energy_densities[i]);
			saveAlgebraic(EDENS_PREFIX, _r._energy_densities_ed[i], KEY_MEAN + tag, true);
			saveAlgebraic(EDENS_PREFIX, _r._energy_densities_ed[i], KEY_MEAN_STATE + tag, true);
			saveAlgebraic(EDENS_PREFIX, _r._energy_densities_ed[i], KEY_MEAN_STATE2 + tag, true);
		}

		//! time grid
		saveAlgebraic(EVO_PREFIX, _r.times, KEY_TIME, false);
		if (_uniform_time)
		{
			const long long start_short 	= 0;
			const long long start_medium 	= _thouless_est * 1e-2;
			const long long start_middle 	= _thouless_est * 0.5;
			const long long end_short 		= start_short + (_ntimes * _dtau_est);
			const long long end_medium 		= start_medium + (_ntimes * _dtau_est);
			const long long end_middle 		= start_middle + (_ntimes * _dtau_est);
			const long long _total_size		= (_timespace.size());
			const long long _short_size		= end_short - start_short;
			const long long _medium_size	= end_medium - start_medium;
			const long long _middle_size	= end_middle - start_middle;

			saveAlgebraic(dir, EVO_PREFIX, _timespace.subvec(0, _short_size - 1), KEY_TIME_SHORT, true);
			saveAlgebraic(dir, EVO_PREFIX, _timespace.subvec(_short_size, _short_size + _medium_size - 1), KEY_TIME_MEDIUM, true);
			saveAlgebraic(dir, EVO_PREFIX, _timespace.subvec(_short_size + _medium_size, _short_size + _medium_size + _middle_size - 1), KEY_TIME_MIDDLE, true);
			saveAlgebraic(dir, EVO_PREFIX, _timespace.subvec(_short_size + _medium_size + _middle_size, _total_size - 1), KEY_TIME_LONG, true);
		}

		//! time evolution of entropies
		saveAlgebraic(dir, EVO_PREFIX, _r.time_evo_me[0], "time_evo/sites", _r._entropies_sites, false);
		for(int i = 0; i < _entropies_sites.size(); ++i)
			saveAlgebraic(dir, EVO_PREFIX, _r.time_ee, KEY_ENT_ENTROPY_PREFIX SUFFIX_ME + STR(i), _r.time_ee[i], true);
		
		// bipartite entanglement entropy
		saveAlgebraic(EVO_PREFIX, _r.time_ee_bipartite, KEY_ENT_ENTROPY_PREFIX SUFFIX_ME SUFFIX_ME, true);
		saveAlgebraic(EVO_PREFIX, _r.time_participation_entropy, KEY_PARTICIPATION SUFFIX_ME, true);

		// microcanonical averages and diagonal ensembles
		saveAlgebraic(AVS_PREFIX, arma::vec(_to_check_microcanonical_eps), KEY_EPS, false);
		for(size_t i = 0; i < _r.microcanonical_me.size(); ++i) 
		{
			const std::string _name = this->_opsN[i];

			// time evolution
			saveAlgebraic(EVO_PREFIX, _r.time_evo_me[i], _name + SUFFIX_ME, true);
			saveAlgebraic(EVO_PREFIX, _r.time_evo_me2[i], _name + SUFFIX_ME, true);
			saveAlgebraic(AVS_PREFIX, _r.microcanonical_me[i], _name + SUFFIX_MICRO_ME, true);
			saveAlgebraic(AVS_PREFIX, _r.microcanonical_me2[i], _name + SUFFIX_MICRO_ME2, true);
			saveAlgebraic(AVS_PREFIX, _r.diagonal_me.row(i).t(), _name + SUFFIX_DIAGONAL_ME, true);
			
			// finite energy density
			for(size_t e = 0; e < this->_energy_densities.size(); ++e)
			{	
				const std::string tag = "/eps=" + std::to_string(this->_energy_densities[e]);
				saveAlgebraic(EVO_PREFIX, _r.time_evo_ed[e][i], _name + tag, true);
				saveAlgebraic(EVO_PREFIX, _r.time_evo_ed2[e][i], _name + tag, true);
				saveAlgebraic(AVS_PREFIX, _r.microcanonical_ed[e][i], _name + SUFFIX_MICRO_ED(this->_energy_densities[e]), true);
				saveAlgebraic(AVS_PREFIX, _r.microcanonical_ed2[e][i], _name + SUFFIX_MICRO2_ED(this->_energy_densities[e]), true);
				saveAlgebraic(AVS_PREFIX, _r.diagonal_ed[e].row(i).t(), _name +  , true);
	
			}
		}
	}


    // ------------------------------------------------------------
};


template<typename _T>
void UI::checkETH_time_evo(std::shared_ptr<Hamiltonian<_T>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

    TimeEvoParams _params;
    _params.set_hamil_params(_H->getTypeI(), _H, this->latP.Ntot_, this->modP.getRanReal());

    //! get info
	std::string modelInfo, dir = "ETH_MAT_TIME_EVO", randomStr, extension;
    this->get_inf_dir_ext_r(_H, _params.dir, _params.modelInfo, _params.randomStr, _params.extension);

	//! get the operators
	std::tie(_params._ops, _params._opsN) = this->ui_eth_getoperators(_params._Nh, _params._isQuadratic, _params._isManyBody);

	//! create the measurement class
	Simulation::DataContainer _container;
	Measurement<double> _measure(this->latP.Ntot_, _params.dir, _params._ops, _params._opsN, 1, _params._Nh);

	//! time evolution saved here
    _params.set_timespace();

	// -------------------------------- SAVER ------------------------------------

	auto _microcanonical_saver	= [&](	uint _r,								// realization
										uint _opi, 								// index of the operator
										VMAT<_T>& _microvals,					// microcanonical averages
										VMAT<double>& _microvals2,				// microcanonical averages squared
										arma::Mat<double>& _diagvals,			// diagonal ensamble values (for all realizations)
										const arma::Col<double>& _soverlaps,	// overlaps of the state with the eigenstates squared
										const v_1d<u64>& _mins,					// min indices for the energy window
										const v_1d<u64>& _maxs)					// max indices for the energy window
		{
			//! get the diagonal elements for a given realization and operator _opi
			const auto& _diagonal	= _diagonals[_opi].col(_r);
			//! long time average (but for all states times the overlaps) $\sum _n a_{nn} * <\psi_n|n>^2$
			_diagvals(_opi, _r)		= algebra::cast<double>(arma::dot(_diagonal, _soverlaps));

			const uint _size		= _to_check_microcanonical_eps.size();
			const u64 _Nh_minus_1	= _Nh - 1;
			
			for (int _ifrac = 0; _ifrac < _size; _ifrac++)
			{
				u64 _minin			= _mins[_ifrac];
				u64 _maxin			= _maxs[_ifrac];

				_minin				= std::max(_minin, 0ull);
				_maxin				= std::min(_maxin, _Nh_minus_1);

				if (_minin >= _maxin) 
				{
					_minin = _minin > 0 ? _minin - 1 : 0;
					_maxin = _maxin < _Nh_minus_1 ? _maxin + 1 : _Nh_minus_1;
				}

				// around the energy (for a given epsilon) (in the energy window - this is the microcanonical within the energy window)
				auto sub_diagonal					= _diagonal.subvec(_minin, _maxin);
				_microvals[_opi](_ifrac, _r)		= arma::mean(sub_diagonal);
				_microvals2[_opi](_ifrac, _r)		= arma::mean(arma::square(arma::abs(sub_diagonal)));
			}
		};

	// ----------------------------- EVOLVE STATE -------------------------------

	auto _evolveState = [&](uint _r, 											// realization
							const arma::Col<_T>& _initial_state,				// initial state to be evolved
							arma::Mat<double>* _ldoses,							// local density of states - save here (for a given realization)			
							arma::Mat<_T>* _energydensities,					// energy densities - save here (for a given realization)
							VMAT<_T>* _microvals,								// microcanonical averages - save here (for a given realization)
							VMAT<double>* _microvals2,	
							arma::Mat<double>* _diagvals,						// diagonal ensamble values - save here (for a given realization)
							VMAT<_T>* _timeEvolution,							// time evolution - save here (for a given realization) [all operators]
							v_1d<arma::Col<_T>>* _timeZero,						// time evolution at t=0 - save here (for a given realization) 
							const v_1d<GeneralizedMatrix<double>>& _matrices,	// matrices of the operators for this realization
							bool _entropyCalculate = true, 						// calculate the entropies?
							bool _append = false,								// append the values?
							bool _uselog = false,								// use the log of the values?
							bool _autocor= false)								// enable autocorrelation?
		{
			if (_timeEvolution == nullptr) {
				LOGINFO("Empty state evolution container... returning", LOG_TYPES::TRACE, 3);
				return;
			}

			// calculate the overlaps of the initial state with the eigenvectors 
			// (states are columns and vector is column as well, so we need to have the transpose)
			const auto& _eigvecs				= _H->getEigVec();
			const auto& _eigvals				= _H->getEigVal();
			const arma::Col<_T> _overlaps		= _eigvecs.t() * _initial_state;		// overlaps with the eigenstates
			const arma::Col<double> _soverlaps	= arma::square(arma::abs(_overlaps));	// overlaps squared

			// calculate the average energy index
			double _meanE						= _H->getEnAv();						// mean energy

			// save the local density of states - overlaps of the initial state with the eigenstates
			if (_ldoses)
				_ldoses->col(_r)				= SystemProperties::calculate_LDOS(_eigvals, _overlaps);

			if (_energydensities)
			{
				(*_energydensities)(0, _r)			= _meanE;									// mean energy
				// apply the Hamiltonian to the initial state
				const arma::Col<_T> _init_stat_H	= _H->getHamiltonian() * _initial_state;	// H|psi>
				const auto _E						= arma::cdot(_initial_state, _init_stat_H);	// energy expectation value
				const auto _E2						= arma::cdot(_init_stat_H, _init_stat_H);	// energy expectation value squared
				u64 _Eidx							= _H->calcEnIdx(algebra::cast<double>(_E));	// energy index closest to the <psi|H|psi> value
				LOGINFO(VEQ(_Eidx), LOG_TYPES::TRACE, 4);
				LOGINFO(VEQP(_E, 5), LOG_TYPES::TRACE, 4);
				LOGINFO(VEQP(_E2, 5), LOG_TYPES::TRACE, 4);
				(*_energydensities)(1, _r)			= _E;										// <psi|H|psi>
				(*_energydensities)(2, _r)			= _E2;										// <psi|H|H|psi>^2

				// calculate the bounds (min and max) for the energy window around the energy <psi|H|psi>
				std::vector<u64> _mins(_to_check_microcanonical_eps.size()), _maxs(_to_check_microcanonical_eps.size());
				for (int _ifrac = 0; _ifrac < _to_check_microcanonical_eps.size(); _ifrac++)
					std::tie(_mins[_ifrac], _maxs[_ifrac]) = _H->getEnArndEnEps(_Eidx, _to_check_microcanonical_eps[_ifrac]);
				
				// calculate the microcanonical averages
				if (_microvals && _microvals2 && _diagvals)
				{
		#pragma omp parallel for num_threads(this->threadNum)
					for (int _opi = 0; _opi < _ops.size(); ++_opi)
					{
						_microcanonical_saver(_r, _opi, *_microvals, *_microvals2, *_diagvals, _soverlaps, _mins, _maxs);
					}
				}
			}

			// -----------------------------------------------------------------------------
			v_1d<_T> _zerovalues(_ops.size());
#pragma omp parallel for num_threads(this->threadNum)
			for (uint _opi = 0; _opi < _ops.size(); ++_opi) {
				_zerovalues[_opi] = arma::as_scalar(arma::cdot(_initial_state, (_matrices[_opi] * _initial_state)));
				if (_timeZero)
					(*_timeZero)[_opi](_r) = _zerovalues[_opi];
			}

			// evolution
#pragma omp parallel for num_threads(this->threadNum) schedule(dynamic)
			for (int _ti = 0; _ti < _timespace.size(); _ti++)
			{
				const auto _time					= _timespace(_ti);
				const arma::Col<cpx> _st			= SystemProperties::TimeEvolution::time_evo(_eigvecs, _eigvals, _overlaps, _time);

				// for each operator we can now apply the expectation value		
				for (uint _opi = 0; _opi < _ops.size(); ++_opi)
				{
					const cpx _rt					= arma::as_scalar(arma::cdot(_st, (_matrices[_opi] * _st))) * (_autocor ? _zerovalues[_opi] : 1.0);
					auto& _time_evo 				= (*_timeEvolution)[_opi](_ti, _r);
					if (_append) {
						_time_evo += _uselog ? std::log(std::abs(algebra::cast<_T>(_rt))) : algebra::cast<_T>(_rt);
					} else {
						_time_evo = _uselog ? std::log(std::abs(algebra::cast<_T>(_rt))) : algebra::cast<_T>(_rt);
					}
				}

				// say the time
				if (_ti % (std::max(1, static_cast<int>(_timespace.size() / 3))) == 0)
					LOGINFO(VEQ(_ti) + "/" + STR(_timespace.size()), LOG_TYPES::TRACE, 3);
				

				// calculate the entanglement entropy for each site
				if (this->modP.eth_entro_ && _entropyCalculate)
				{
					//for (int i = 1; i <= _Ns; i++)
					auto _iter = 0;
					for (const auto i: _entropies_sites)
					{
						// calculate the entanglement entropy
						uint _maskA							= 1 << (i - 1);
						_time_entropy_me[_iter++](_ti, _r)	= Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_st, 1, _Ns, _maskA, DensityMatrix::RHO_METHODS::SCHMIDT, 2);
					}
					if(_Nh <= UI_LIMITS_MAXFULLED / 4)
						_time_entropy_bipartite_me(_ti, _r)	= Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_st, int(_Ns / 2), _Ns, (ULLPOW((int(_Ns / 2)))) - 1);
					// calculate the participation entropy
					{
						_time_part_entro(_ti, _r)			= SystemProperties::information_entropy(_st);
					}
				}
			}

		};


	// -------------------------------- SAVER ------------------------------------

	// create the saving function
	std::function<void(uint)> _saver = [&](uint _r)
		{
			// variance in th Hamiltonian
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(0).as_col()), "mean_lvl_spacing", false);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(1).as_col()), "heis_time_gamma", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(2).as_col()), "heis_time_around_mean", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(3).as_col()), "heis_time_around_mean_typ", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _energies, "energies", true);

			// save the ldos's
			saveAlgebraic(dir, "ldos" + randomStr + extension, _ldos_me, "ME", false);

			// save the energy densities
			{
				saveAlgebraic(dir, "energydens" + randomStr + extension, arma::Col<_T>(_energy_densities_me.row(0).as_col()), "mean", false);
				saveAlgebraic(dir, "energydens" + randomStr + extension, arma::Col<_T>(_energy_densities_me.row(1).as_col()), "mean_state", true);
				saveAlgebraic(dir, "energydens" + randomStr + extension, arma::Col<_T>(_energy_densities_me.row(2).as_col()), "mean_state2", true);
				
				for (int i = 0; i < _energy_densities_edens.size(); ++i)
				{
					saveAlgebraic(dir, "energydens" + randomStr + extension, _energy_densities_edens[i], "mean/" + STR(_energy_densities[i]), true);
					saveAlgebraic(dir, "energydens" + randomStr + extension, _energy_densities_edens[i], "mean_state/" + STR(_energy_densities[i]), true);
					saveAlgebraic(dir, "energydens" + randomStr + extension, _energy_densities_edens[i], "mean_state2/" + STR(_energy_densities[i]), true);
				}
			}

			// save the matrices for time evolution
			saveAlgebraic(dir, "evo" + randomStr + extension, _timespace, "time", false);
			if (_uniform_time)
			{
				const long long start_short 	= 0;
				const long long start_medium 	= _thouless_est * 1e-2;
				const long long start_middle 	= _thouless_est * 0.5;
				const long long start_long 		= _heisenberg_time_est * 1e-2;
				const long long end_short 		= start_short + (_ntimes * _dtau_est);
				const long long end_medium 		= start_medium + (_ntimes * _dtau_est);
				const long long end_middle 		= start_middle + (_ntimes * _dtau_est);
				const long long end_long 		= start_long + (_ntimes * _dtau_est);
				const long long _total_size		= (_timespace.size());
				const long long _short_size		= end_short - start_short;
				const long long _medium_size	= end_medium - start_medium;
				const long long _middle_size	= end_middle - start_middle;

				saveAlgebraic(dir, "evo" + randomStr + extension, _timespace.subvec(0, _short_size - 1), "time/short", false);
				saveAlgebraic(dir, "evo" + randomStr + extension, _timespace.subvec(_short_size, _short_size + _medium_size - 1), "time/medium", true);
				saveAlgebraic(dir, "evo" + randomStr + extension, _timespace.subvec(_short_size + _medium_size, _short_size + _medium_size + _middle_size - 1), "time/middle", true);
				saveAlgebraic(dir, "evo" + randomStr + extension, _timespace.subvec(_short_size + _medium_size + _middle_size, _total_size - 1), "time/long", true);
			}

			// save the time evolution of the entanglement entropy
			for(int i = 0; i < _entropies_sites.size(); ++i)
				saveAlgebraic(dir, "evo" + randomStr + extension, _time_entropy_me[i], "entanglement_entropy/ME/" + STR((_entropies_sites[i])), true);

			saveAlgebraic(dir, "evo" + randomStr + extension, _time_entropy_bipartite_me, "entanglement_entropy/ME/bipartite", true);
			saveAlgebraic(dir, "evo" + randomStr + extension, _time_part_entro, "participation_entropy/ME", true);

			// save the averages epsilon
			saveAlgebraic(dir, "avs" + randomStr + extension, arma::vec(_to_check_microcanonical_eps), "eps", false);
			saveAlgebraic(dir, "evo" + randomStr + extension, arma::vec(_energy_densities), "energy_densities", true);
			
			// go through the operators
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				
				saveAlgebraic(dir, "diag" + randomStr + extension, _diagonals[_opi], _name, _opi > 0);

				//! ME
				{
					saveAlgebraic(dir, "evo" + randomStr + extension, _time_evolution_me[_opi], _name + "/ME", true);
					saveAlgebraic(dir, "evo" + randomStr + extension, _time_evolution_0_me[_opi], _name + "/ME/zero", true);
					saveAlgebraic(dir, "avs" + randomStr + extension, _microcanonical_me[_opi], _name + "/micro/ME/1", true);
					saveAlgebraic(dir, "avs" + randomStr + extension, _microcanonical_me2[_opi], _name + "/micro/ME/2", true);
					// long time average
					saveAlgebraic(dir, "avs" + randomStr + extension, arma::vec(_diagonal_me.row(_opi).as_col()), _name + "/diagonal/ME", true);
				}
				//! energy density
				for (int i = 0; i < _energy_densities.size(); ++i)
				{
					std::string _eps_name = "/eps=" + STRP(_energy_densities[i], 3) + "/";
					saveAlgebraic(dir, "evo" + randomStr + extension, _time_evolution_edens[i][_opi], _name + _eps_name, true);
					saveAlgebraic(dir, "evo" + randomStr + extension, _time_evolution_0_edens[_opi].col(i), _name + _eps_name + "zero", true);
					saveAlgebraic(dir, "avs" + randomStr + extension, _microcanonical_edens[i][_opi], _name + "/micro" + _eps_name, true);
					saveAlgebraic(dir, "avs" + randomStr + extension, _microcanonical_edens2[i][_opi], _name + "/micro" + _eps_name, true);
					saveAlgebraic(dir, "avs" + randomStr + extension, arma::vec(_diagonal_edens[i].row(_opi).as_col()), _name + "/diagonal" + _eps_name, true);
				}
			}

			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
		};

	// -----------------------
	long _single_run_time 	= -1;
	long _remaining_time	= Slurm::get_remaining_time();
	LOGINFO("Remaining time: " + STR(_remaining_time) + " seconds", LOG_TYPES::INFO, 0);
	_timer.checkpoint("START");
	// -----------------------

	// go through realizations
	for (int _r = 0; _r < _nrealiz; ++_r)
	{
		arma::Col<_T> _diagonal;
		
		//! checkpoints etc
		{
			// -----------------------------------------------------------------------------
			LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
			LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
			_timer.checkpoint(STR(_r));

			this->ui_eth_randomize(_H, _r);
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);

			//! get the diagonal of the Hamiltonian
			_diagonal 			= _H->getDiag();
			_initial_state_me 	= SystemProperties::TimeEvolution::create_initial_quench_state<_T>(SystemProperties::TimeEvolution::QuenchTypes::SEEK, _Nh, _Ns, _H->getEnAv(), _diagonal);
			
			//! create states for energy densities
			for (int i = 0; i < _energy_densities.size(); ++i)
			{
				double _energy_state 		= _energy_densities[i] * _H->getBandwidth();
				_initial_state_edens.col(i) = SystemProperties::TimeEvolution::create_initial_quench_state<_T>(SystemProperties::TimeEvolution::QuenchTypes::SEEK, _Nh, _Ns, _energy_state, _diagonal);
			}	
		}

		// -----------------------------------------------------------------------------

		// calculator of the properties
		{
			// mean level spacing
			{
				long double _h_freq		= 1.0 / _Nh;
				_energies.col(_r)		= _H->getEigVal();
				_meanlvl(0, _r)			= _H->getMeanLevelSpacing();
				_meanlvl(1, _r)			= SystemProperties::mean_lvl_heis_time(_meanlvl(0, _r), _Ns);

				// get the average energy index and the points around it on the diagonal
				const auto [_min, _max] = _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
				auto _E					= _H->getEigVal().subvec(_min, _max);
				_meanlvl(2, _r)			= 1.0 / SystemProperties::mean_lvl_spacing(_E);
				// get the Heisenberg frequency from the mean level spacing
				_h_freq					= SystemProperties::mean_lvl_spacing_typ(_E);
				_meanlvl(3, _r)			= 1.0 / _h_freq;
				LOGINFO(StrParser::colorize(VEQP(_meanlvl(0, _r), 10) + ": mean level spacing", StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(1, _r)) + ": mean level Heisenberg time", StrParser::StrColors::blue), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(2, _r)) + ": mean level Heisenberg time around energy " + VEQP(_min, 3), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(3, _r)) + ": mean level Heisenberg time around energy - typical", StrParser::StrColors::yellow), LOG_TYPES::TRACE, 1);
			}

			// -----------------------------------------------------------------------------

			// other measures
			{
				// calculate the diagonals
				const auto& _matrices				= _measure.getOpG_mat();
				const auto& _eigvec					= _H->getEigVec();

#pragma omp parallel for num_threads(_Nh < ULLPOW(14) ? this->threadNum : 2)
				for (int _opi = 0; _opi < _ops.size(); ++_opi)
				{
					_diagonals[_opi].col(_r) = Operators::applyOverlapMat(_eigvec, _matrices[_opi]).diag();
				}

				_timer.checkpoint(STR(_r) + ": time evolution");

				// evolve the states
				_evolveState(_r, _initial_state_me, &_ldos_me, &_energy_densities_me,  
					&_microcanonical_me, &_microcanonical_me2, &_diagonal_me, &_time_evolution_me, &_time_evolution_0_me, _matrices, true, false, false, false);
				
				// do the same for the energy densities
				for (int i = 0; i < _energy_densities.size(); ++i)
				{
					_evolveState(_r, _initial_state_edens.col(i), &_ldos_edens[i], &_energy_densities_edens[i], 
						&_microcanonical_edens[i], &_microcanonical_edens2[i], &_diagonal_edens[i], &_time_evolution_edens[i], nullptr, _matrices, false, false, false, false);
				}

				LOGINFO(_timer.point(STR(_r) + ": time evolution"), "Time evolution: " + STR(_r), 3);
			}
		}
		
		// -----------------------------------------------------------------------------

		// get single runtime
		_single_run_time = _timer.elapsed<long>(STR(_r), Timer::TimePrecision::SECONDS);
		LOGINFO("Single run time: " + STR(_single_run_time) + " seconds", LOG_TYPES::TRACE, 1);

		// save the checkpoints
		if (check_saving_size(_Nh, _r) && symP.checkpoint_)
			_saver(_r);

		if (this->remainingSlurmTime(_r, &_timer, _single_run_time, _remaining_time))
			break;

		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '%', 1);
		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_saver(_nrealiz);

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

template void UI::checkETH_time_evo<double>(std::shared_ptr<Hamiltonian<double>> _H);
template void UI::checkETH_time_evo<cpx>(std::shared_ptr<Hamiltonian<cpx>> _H);