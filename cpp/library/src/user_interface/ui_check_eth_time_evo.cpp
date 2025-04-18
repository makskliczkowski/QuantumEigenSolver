#include "../../include/user_interface/user_interface.h"
#include "../../source/src/Include/linalg/generalized_matrix.h"
#include "armadillo"
#include <memory>
#include <vector>

// ----------------------------------------------------------------

constexpr static bool check_saving_size(u64 _Nh, uint _r)
{
	return ((_Nh > ULLPOW(14)) ||(BETWEEN((size_t)std::log2(_Nh), 10, 15) && (_r % 10 == 0)) || ((_Nh <= 1024) && (_r % 50 == 0)));
}

// --------------------------------------------------------------------

namespace TimeEvo
{

	template<typename T>
	void microcanonical_saver(
		size_t r,                                            // realization index
		size_t opi,                                          // operator index
		const std::vector<arma::Mat<T>>& diagonals,          // diagonals[opi] is [nh × n_real]
		const arma::Col<double>& soverlaps,                  // |<ψ|n>|² vector of length nh
		const std::vector<uint64_t>& mins,                   // window start indices per ε - given energy density
		const std::vector<uint64_t>& maxs,                   // window end indices per ε - given energy density
		arma::Mat<double>& diagvals,                         // [n_ops × n_real], to write long‑time avg here (the diagonal ensemble)
		std::vector<arma::Mat<T>>& microvals,                // [n_ops][n_eps × n_real] to save the microcanonical averages here
		std::vector<arma::Mat<double>>& microvals2           // [n_ops][n_eps × n_real] squares of the microcanonical averages
		) 
	{
		// grab the diagonal of operator opi for realization r
		const auto col 			= diagonals[opi].col(r);

		// long‑time (diagonal) average ∑_n a_nn |<ψ|n>|²
		diagvals(opi, r) 		= algebra::cast<double>(arma::dot(col, soverlaps));

		const size_t n_eps 		= mins.size();
		const uint64_t Nminus1 	= static_cast<uint64_t>(col.n_rows) - 1;

		for (size_t ei = 0; ei < n_eps; ++ei) 
		{
			// clamp window
			uint64_t mn = std::min<uint64_t>(std::max<uint64_t>(mins[ei], 0), Nminus1);
			uint64_t mx = std::min<uint64_t>(maxs[ei], Nminus1);
			if (mn >= mx) {
				mn = mins[ei] > 0 ? mins[ei] - 1 : 0;
				mx = maxs[ei] < Nminus1 ? maxs[ei] + 1 : Nminus1;
			}
			// compute microcanonical mean and mean‑square
			auto sub 				= col.subvec(mn, mx);
			microvals[opi](ei, r)  	= arma::mean(sub);
			microvals2[opi](ei, r) 	= arma::mean(arma::square(arma::abs(sub)));
		}
	}

	// explicit instantiations
	template void microcanonical_saver<double>(
		size_t, size_t,
		const std::vector<arma::Mat<double>>&,
		const arma::Col<double>&,
		const std::vector<uint64_t>&,
		const std::vector<uint64_t>&,
		arma::Mat<double>&,
		std::vector<arma::Mat<double>>&,
		std::vector<arma::Mat<double>>&);

	template void microcanonical_saver<std::complex<double>>(
		size_t, size_t,
		const std::vector<arma::Mat<std::complex<double>>>&,
		const arma::Col<double>&,
		const std::vector<uint64_t>&,
		const std::vector<uint64_t>&,
		arma::Mat<double>&,
		std::vector<arma::Mat<std::complex<double>>>&,
		std::vector<arma::Mat<double>>&);

	// --------------------------------------------------------------------

	template<typename T>
	void evolve_state(
		size_t                                   r,                      	// realization index
		const arma::Col<T>&                      init_state,             	// initial state vector [nh]
		const std::shared_ptr<Hamiltonian<T>>&   H,                     	// Hamiltonian object
		arma::Mat<double>*                       ldos,                   	// local density of states [nh x n_real] - optional
		arma::Mat<T>*                            energydensities,        	// energy densities [3 x n_real] - optional
		std::vector<arma::Mat<T>>*               micro,                  	// microcanonical averages [n_ops][n_eps x n_real] - optional
		std::vector<arma::Mat<double>>*          micro2,                 	// microcanonical averages squared [n_ops][n_eps x n_real] - optional
		arma::Mat<double>*                       diagvals,               	// diagonal ensemble [n_ops x n_real] - optional
		const std::vector<arma::Mat<T>>&         diagonals,              	// operator diagonals [n_ops][nh x n_real]
		std::vector<arma::Mat<T>>&               time_evo_me,            	// time evolution of operators [n_ops][n_times x n_real]
		std::vector<arma::Col<T>>*               time_zero_me,           	// zero time value of operators [n_ops][n_real] - optional
		std::vector<arma::Mat<double>>*          time_ee,                	// entanglement entropy [n_sites][n_times x n_real] - optional
		arma::Mat<double>*                       time_ee_bipartite,      	// entanglement entropy bipartite [n_times x n_real] - optional
		arma::Mat<double>*                       time_participation_entropy,// participation entropy [n_times x n_real] - optional
		const std::vector<GeneralizedMatrix<double>>& matrices,				// operators [n_ops][nh x nh]
		const arma::Col<double>&                 times,                  	// time grid [n_times]
		size_t                                   Ns,                		// total number of sites
		const std::vector<size_t>&               entropy_sites,     		// sites for entanglement entropy
		const std::vector<double>&               micro_epsilons,    		// epsilons for microcanonical windows
		bool                                     append,           			// append values?
		bool                                     use_log,         			// use log of values?
		bool                                     auto_cor,         			// enable autocorrelation?
		int                                      thread_num        			// number of threads
	) 
	{
		// Eigen decomposition info
		// const size_t Nh                  	= H->getHilbertSize();	
		const auto& eigvecs              	= H->getEigVec();
		const auto& eigvals              	= H->getEigVal();
		const arma::Col<T> overlaps      	= eigvecs.t() * init_state;
		const arma::Col<double> soverlaps	= arma::square(arma::abs(overlaps));
		
		// Local density of states
		if (ldos) {
			ldos->col(r) = SystemProperties::calculate_LDOS(eigvals, overlaps);
		}
		
		// Energy densities & microcanonical
		if (energydensities) 
		{
			double meanE			 	= H->getEnAv();
			(*energydensities)(0, r) 	= meanE;
			const arma::Col<T> Hpsi  	= H->getHamiltonian() * init_state;
			const auto E           		= arma::cdot(init_state, Hpsi);
			const auto E2          		= arma::cdot(Hpsi, Hpsi);
			(*energydensities)(1, r) 	= E;
			(*energydensities)(2, r) 	= E2;
			size_t Eidx              	= H->calcEnIdx(algebra::cast<double>(E));

			// microcanonical windows
			std::vector<uint64_t> mins(micro_epsilons.size()), maxs(micro_epsilons.size());
			for (size_t i = 0; i < micro_epsilons.size(); ++i) {
				std::tie(mins[i], maxs[i]) = H->getEnArndEnEps(Eidx, micro_epsilons[i]);
			}

			if (micro && micro2 && diagvals) {
				#pragma omp parallel for num_threads(thread_num)
				for (size_t opi = 0; opi < diagonals.size(); ++opi) {
					microcanonical_saver<T>(
						r, 
						opi,
						diagonals,
						soverlaps,
						mins, maxs,
						*diagvals,
						*micro,
						*micro2
					);
				}
			}
		}

		// Zero-time values for each of the operators
		std::vector<T> zerovals(matrices.size());
		#pragma omp parallel for num_threads(thread_num)
		for (size_t opi = 0; opi < matrices.size(); ++opi) {
			zerovals[opi] = arma::as_scalar(arma::cdot(init_state, matrices[opi] * init_state));
			if (time_zero_me) {
				(*time_zero_me)[opi](r) = zerovals[opi];
			}
		}

		const size_t n_times 	= times.n_elem;
		const size_t log_stride	= std::max<size_t>(1, n_times / 10);  // every 10%
		auto t_start   			= std::chrono::high_resolution_clock::now();
		auto t_last    			= t_start;
		std::atomic<size_t> completed{0};
#pragma omp parallel num_threads(thread_num)
		{
			// thread‑local workspace
			arma::Col<std::complex<double>> st_loc(eigvecs.n_rows);

#pragma omp for schedule(static)
			for (size_t ti = 0; ti < n_times; ++ti) 
			{
	
				// start current time for the performance
				const double t			= times(ti);
				st_loc 					= SystemProperties::TimeEvolution::time_evo(eigvecs, eigvals, overlaps, t, 1);

				for (size_t opi = 0; opi < matrices.size(); ++opi) 
				{
					auto val_c 			= arma::as_scalar(arma::cdot(st_loc, matrices[opi] * st_loc));
					if (auto_cor) 		val_c *= zerovals[opi];
					auto val_c_t 		= algebra::cast<T>(val_c);

					T val = 0.0;
					if (append) 
					{
						if (use_log)
							val = time_evo_me[opi](ti, r) + std::log(std::abs(val_c_t));
						else
							val = time_evo_me[opi](ti, r) + algebra::cast<T>(val_c_t);
					} 
					else 
					{
						if (use_log)
							val = std::log(std::abs(val_c_t));
						else
							val = algebra::cast<T>(val_c_t);
					}
					time_evo_me[opi](ti, r) = val;
				}

				// Entropy
				if (time_ee != nullptr) 
				{
					size_t iter = 0;
					for (auto site : entropy_sites) 
					{
						uint64_t maskA = 1ull << (site - 1);
						// Entanglement entropy per site
						(*time_ee)[iter++](ti, r) = Entropy::Entanglement::Bipartite::vonNeuman<cpx>(
							st_loc, 1, Ns, maskA, DensityMatrix::RHO_METHODS::SCHMIDT, 2);
					}
				}
				
				// Bipartite entanglement entropy
				if (time_ee_bipartite != nullptr) 
				{
					(*time_ee_bipartite)(ti, r) = Entropy::Entanglement::Bipartite::vonNeuman<cpx>(
						st_loc, int(Ns / 2), Ns, (ULLPOW(int(Ns / 2))) - 1);
				}

				// Participation entropy
				if (time_participation_entropy != nullptr) 
				{
					(*time_participation_entropy)(ti, r) = SystemProperties::information_entropy(st_loc);
				}

				size_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
				if (done % log_stride == 0) 
				{
#pragma omp critical
					{
						auto now     	= std::chrono::high_resolution_clock::now();
						auto total   	= std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count();
						auto partial 	= std::chrono::duration_cast<std::chrono::milliseconds>(now - t_last).count();
	
						int pct 		= static_cast<int>((100.0 * done) / n_times);
						LOGINFO("Time evolution progress: " 
								+ std::to_string(pct) + "% ("
								+ std::to_string(done) + "/"
								+ std::to_string(n_times) + ")",
								LOG_TYPES::TRACE, 4);
	
						LOGINFO("  since start: " + STRP(total,2) + " ms; "
								"since last log: " + STRP(partial,2) + " ms",
								LOG_TYPES::TRACE, 5);
	
						t_last = now;
					}
				}
			}
		}
	}

	// explicit instantiations
	template void evolve_state<double>(
		size_t, const arma::Col<double>&, const std::shared_ptr<Hamiltonian<double>>&, 
		arma::Mat<double>*, arma::Mat<double>*, std::vector<arma::Mat<double>>*, 
		std::vector<arma::Mat<double>>*, arma::Mat<double>*, const std::vector<arma::Mat<double>>&, 
		std::vector<arma::Mat<double>>&, std::vector<arma::Col<double>>*, 
		std::vector<arma::Mat<double>>*, arma::Mat<double>*, arma::Mat<double>*,
		const std::vector<GeneralizedMatrix<double>>&, const arma::Col<double>&, size_t, 
		const std::vector<size_t>&, const std::vector<double>&, bool, bool, bool, int);

	template void evolve_state<std::complex<double>>(
		size_t, const arma::Col<std::complex<double>>&, const std::shared_ptr<Hamiltonian<std::complex<double>>>&, 
		arma::Mat<double>*, arma::Mat<std::complex<double>>*, std::vector<arma::Mat<std::complex<double>>>*, 
		std::vector<arma::Mat<double>>*, arma::Mat<double>*, const std::vector<arma::Mat<std::complex<double>>>&, 
		std::vector<arma::Mat<std::complex<double>>>&, std::vector<arma::Col<std::complex<double>>>*, 
		std::vector<arma::Mat<double>>*, arma::Mat<double>*, arma::Mat<double>*,
		const std::vector<GeneralizedMatrix<double>>&, const arma::Col<double>&, size_t, 
		const std::vector<size_t>&, const std::vector<double>&, bool, bool, bool, int);
	
	// ---------------------------------------------------------------------

	template<typename T>
	void TimeEvoParams<T>::set_hamil_params(MY_MODELS _type, std::shared_ptr<Hamiltonian<T>> _H, size_t n_tot, size_t n_real)
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
		this->_uniform_time			 	= true;
		this->dir					    = this->_uniform_time ? "ETH_MAT_TIME_EVO_UNIFORM" : "ETH_MAT_TIME_EVO";
		// this->_isQuadratic              = _H->getIsQuadratic();
		// this->_isManyBody               = _H->getIsManyBody();
		this->_ops                      = {};
		this->_opsN                     = {};
		this->_entropies_sites 			= { 1, size_t(_Ns / 2), size_t(_Ns) 			};
		_energy_densities               = { 0.1, 0.2, 0.3                         		};
		_to_check_microcanonical_eps   	= { 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4    };
		this->time_tag					= std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

	}   

	// ---------------------------------------------------------------------

	template<typename T>
	void TimeEvoParams<T>::save_for_realization(size_t r)
	{
		std::string random_tag = this->randomStr;

		//! level statistics
		saveAlgebraic(dir, STAT_PREFIX(randomStr, extension, time_tag), arma::vec(_r.mean_level_spacing.row(0).as_col()), KEY_MEAN_LVL, false);
		saveAlgebraic(dir, STAT_PREFIX(randomStr, extension, time_tag), arma::vec(_r.mean_level_spacing.row(1).as_col()), KEY_HEIS_TIME_GAMMA, true);
		saveAlgebraic(dir, STAT_PREFIX(randomStr, extension, time_tag), arma::vec(_r.mean_level_spacing.row(2).as_col()), KEY_HEIS_TIME_AROUND_MEAN, true);
		saveAlgebraic(dir, STAT_PREFIX(randomStr, extension, time_tag), arma::vec(_r.mean_level_spacing.row(3).as_col()), KEY_HEIS_TIME_AROUND_MEAN_T, true);
		saveAlgebraic(dir, STAT_PREFIX(randomStr, extension, time_tag), _r.energies, KEY_ENERGIES, true);

		//! LDOS
		saveAlgebraic(dir, LDOS_PREFIX(randomStr, extension, time_tag), _r.ldos_me, KEY_ME, false);
		for(size_t i = 0; i < this->_energy_densities.size(); ++i)
			saveAlgebraic(dir, LDOS_PREFIX(randomStr, extension, time_tag), _r.ldos_ed[i], SUFFIX_EPS(this->_energy_densities[i]), true);

		//! energy densities obtained from the given initial state
		{
			saveAlgebraic(dir, EDENS_PREFIX(randomStr, extension, time_tag), arma::Col<T>(_r.energy_densities_me.row(0).as_col()), KEY_MEAN + KEY_ME, false);			// mean energy
			saveAlgebraic(dir, EDENS_PREFIX(randomStr, extension, time_tag), arma::Col<T>(_r.energy_densities_me.row(1).as_col()), KEY_MEAN_STATE + KEY_ME, true);	// <state|H|state>
			saveAlgebraic(dir, EDENS_PREFIX(randomStr, extension, time_tag), arma::Col<T>(_r.energy_densities_me.row(2).as_col()), KEY_MEAN_STATE2 + KEY_ME, true);	// <state|H^2|state>
			for(size_t i = 0; i < _r.energy_densities_ed.size(); ++i) 
			{
				const std::string tag = SUFFIX_EPS(this->_energy_densities[i]);
				saveAlgebraic(dir, EDENS_PREFIX(randomStr, extension, time_tag), arma::Col<T>(_r.energy_densities_ed[i].row(0).as_col()), KEY_MEAN + tag, true);
				saveAlgebraic(dir, EDENS_PREFIX(randomStr, extension, time_tag), arma::Col<T>(_r.energy_densities_ed[i].row(1).as_col()), KEY_MEAN_STATE + tag, true);
				saveAlgebraic(dir, EDENS_PREFIX(randomStr, extension, time_tag), arma::Col<T>(_r.energy_densities_ed[i].row(2).as_col()), KEY_MEAN_STATE2 + tag, true);
			}
		}

		//! time grid
		saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _timespace, KEY_TIME, false);
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

			saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _timespace.subvec(0, _short_size - 1), KEY_TIME_SHORT, true);
			saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _timespace.subvec(_short_size, _short_size + _medium_size - 1), KEY_TIME_MEDIUM, true);
			saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _timespace.subvec(_short_size + _medium_size, _short_size + _medium_size + _middle_size - 1), KEY_TIME_MIDDLE, true);
			saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _timespace.subvec(_short_size + _medium_size + _middle_size, _total_size - 1), KEY_TIME_LONG, true);
		}

		//! time evolution of entropies
		for(int i = 0; i < _entropies_sites.size(); ++i)
			saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _r.time_ee[i], KEY_ENT_ENTROPY_PREFIX + SUFFIX_ME + "/" + STR((_entropies_sites[i])), true);
		
		//! bipartite entanglement entropy
		saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _r.time_ee_bipartite, KEY_ENT_ENTROPY_PREFIX + SUFIX_BIPARTITE + SUFFIX_ME, true);
		saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _r.time_participation_entropy, KEY_PARTICIPATION + SUFFIX_ME, true);
		saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), arma::vec(this->_energy_densities), KEY_ENERGY_DENSITY, true);

		// microcanonical averages and diagonal ensembles
		saveAlgebraic(dir, AVS_PREFIX(randomStr, extension, time_tag), arma::vec(_to_check_microcanonical_eps), KEY_EPS, false);
		for(size_t i = 0; i < _r.microcanonical_me.size(); ++i) 
		{
			const std::string _name = this->_opsN[i];

			//! time evolution ME
			saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _r.time_evo_me[i], _name + SUFFIX_ME, true);
			saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _r.time_evo_zero_me[i], _name + SUFFIX_ZERO + SUFFIX_ME, true);
			saveAlgebraic(dir, AVS_PREFIX(randomStr, extension, time_tag), _r.microcanonical_me[i], _name + SUFFIX_MICRO_ME, true);
			saveAlgebraic(dir, AVS_PREFIX(randomStr, extension, time_tag), _r.microcanonical_me2[i], _name + SUFFIX_MICRO_ME2, true);
			saveAlgebraic(dir, AVS_PREFIX(randomStr, extension, time_tag), arma::vec(_r.diagonal_me.row(i).as_col()), _name + SUFFIX_DIAGONAL_ME, true);
			
			// diagonal matrix elements
			saveAlgebraic(dir, DIAG_PREFIX(randomStr, extension, time_tag), _r.diagonals[i], _name, i > 0);

			// finite energy density
			for(size_t e = 0; e < this->_energy_densities.size(); ++e)
			{	
				const std::string tag = "/eps=" + std::to_string(this->_energy_densities[e]);
				saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _r.time_evo_ed[e][i], _name + tag, true);
				saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _r.time_evo_zero_ed[e][i], _name + SUFFIX_ZERO + tag, true);
				saveAlgebraic(dir, AVS_PREFIX(randomStr, extension, time_tag), _r.microcanonical_ed[e][i], _name + SUFFIX_MICRO_ED(this->_energy_densities[e]), true);
				saveAlgebraic(dir, AVS_PREFIX(randomStr, extension, time_tag), _r.microcanonical_ed2[e][i], _name + SUFFIX_MICRO2_ED(this->_energy_densities[e]), true);
				saveAlgebraic(dir, AVS_PREFIX(randomStr, extension, time_tag), arma::vec(_r.diagonal_ed[e].row(i).as_col()), _name + SUFFIX_DIAGONAL_ED(this->_energy_densities[e]), true);
			}
		}
		
		// in every file, save the last realization number
		arma::vec _realization(1);
		_realization(0) = r;
		saveAlgebraic(dir, EVO_PREFIX(randomStr, extension, time_tag), _realization, KEY_REALIZATION, true);
		saveAlgebraic(dir, AVS_PREFIX(randomStr, extension, time_tag), _realization, KEY_REALIZATION, true);
		saveAlgebraic(dir, STAT_PREFIX(randomStr, extension, time_tag), _realization, KEY_REALIZATION, true);
		saveAlgebraic(dir, LDOS_PREFIX(randomStr, extension, time_tag), _realization, KEY_REALIZATION, true);
		saveAlgebraic(dir, EDENS_PREFIX(randomStr, extension, time_tag), _realization, KEY_REALIZATION, true);
		saveAlgebraic(dir, DIAG_PREFIX(randomStr, extension, time_tag), _realization, KEY_REALIZATION, true);


		LOGINFO("Checkpoint:" + STR(r), LOG_TYPES::TRACE, 4);
	}

	// explict instantiations
	template void TimeEvoParams<double>::save_for_realization(size_t);
	template void TimeEvoParams<std::complex<double>>::save_for_realization(size_t);

	// ---------------------------------------------------------------------
	
	
	// --------------------------------------------------------------------
	// Allocate and initialize all result containers for a new run
	// --------------------------------------------------------------------
	template<typename T>
	void TimeEvoParams<T>::allocate_result_containers()
	{
		// ----------- Logging and clearing old data -----------
		LOGINFO("Clearing old data...", LOG_TYPES::TRACE, 2);
		LOGINFO("The number of operators: " + STR(_ops.size()), LOG_TYPES::TRACE, 3);
		LOGINFO("The number of realizations: " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		LOGINFO("The number of energy densities: " + STR(_energy_densities.size()), LOG_TYPES::TRACE, 3);
		LOGINFO("The number of sites: " + STR(_Ns), LOG_TYPES::TRACE, 3);
		LOGINFO("The number of sites for entanglement entropy: " + STR(_entropies_sites.size()), LOG_TYPES::TRACE, 3);
		LOGINFO("The number of times: " + STR(_ntimes), LOG_TYPES::TRACE, 3);
		
		// Clear all containers
		_r.ldos_ed.clear();
		_r.energy_densities_ed.clear();
		_r.diagonals.clear();
		_r.time_evo_me.clear();
		_r.time_evo_zero_me.clear();
		_r.time_evo_ed.clear();
		_r.time_evo_zero_ed.clear();
		_r.microcanonical_ed.clear();
		_r.microcanonical_ed2.clear();
		_r.diagonal_ed.clear();
		_r.time_ee.clear();

		// ----------- State & spectrum containers -----------
		LOGINFO("Allocating state & spectrum containers...", LOG_TYPES::TRACE, 2);
		_r.initial_state_me.set_size(_Nh);
		_r.initial_state_ed.set_size(_Nh, _energy_densities.size());
		_r.mean_level_spacing.set_size(4, _nrealiz);
		_r.energies.set_size(_Nh, _nrealiz);
		LOGINFO("initial_state_me: " + STR(_Nh), LOG_TYPES::TRACE, 3);
		LOGINFO("initial_state_ed: " + STR(_Nh) + " x " + STR(_energy_densities.size()), LOG_TYPES::TRACE, 3);
		LOGINFO("mean_level_spacing: 4 x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		LOGINFO("energies: " + STR(_Nh) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);

		// ----------- Local density of states -----------
		LOGINFO("Allocating local density of states (ME + ED)...", LOG_TYPES::TRACE, 2);
		_r.ldos_me.set_size(_Nh, _nrealiz);
		LOGINFO("ldos_me: " + STR(_Nh) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		for (size_t e = 0; e < _energy_densities.size(); ++e) {
			_r.ldos_ed.emplace_back(_Nh, _nrealiz);
			LOGINFO("ldos_ed[" + STR(e) + "]: " + STR(_Nh) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		}

		// ----------- Energy densities -----------
		LOGINFO("Allocating energy densities (ME + ED)...", LOG_TYPES::TRACE, 2);
		_r.energy_densities_me.set_size(3, _nrealiz);
		LOGINFO("energy_densities_me: 3 x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		for (size_t e = 0; e < _energy_densities.size(); ++e) {
			_r.energy_densities_ed.emplace_back(3, _nrealiz);
			LOGINFO("energy_densities_ed[" + STR(e) + "]: 3 x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		}

		// ----------- Operator diagonals -----------
		LOGINFO("Allocating operator diagonals...", LOG_TYPES::TRACE, 2);
		for (size_t i = 0; i < _ops.size(); ++i) {
			_r.diagonals.emplace_back(_Nh, _nrealiz);
			LOGINFO("diagonals[" + STR(i) + "]: " + STR(_Nh) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		}

		// ----------- Time evolution (ME) -----------
		LOGINFO("Allocating time-evolution (ME)...", LOG_TYPES::TRACE, 2);
		this->set_timespace();
		if (this->_uniform_time)
		{
			this->_ntimes = 4 * this->_ntimes;
			LOGINFO("The true number of time steps: " + STR(_ntimes), LOG_TYPES::TRACE, 4);
		}
		for (size_t i = 0; i < _ops.size(); ++i)
		{
			_r.time_evo_me.emplace_back(arma::Mat<T>(_ntimes, _nrealiz, arma::fill::ones) * (-1e5));
			_r.time_evo_zero_me.emplace_back(arma::Col<T>(_nrealiz, arma::fill::ones) * (-1e5));
			LOGINFO("time_evo_me[" + STR(i) + "]: " + STR(_ntimes) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
			LOGINFO("time_evo_zero_me[" + STR(i) + "]: " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		}

		// ----------- Time evolution, microcanonical & diagonal (ED) -----------
		LOGINFO("Allocating time-evolution, microcanonical & diagonal (ED)...", LOG_TYPES::TRACE, 2);
		for (size_t e = 0; e < _energy_densities.size(); ++e)
		{
			LOGINFO("Energy density: " + STR(_energy_densities[e]), LOG_TYPES::TRACE, 3);
			std::vector<arma::Mat<T>> evo_ed;
			std::vector<arma::Col<T>> zero_ed;
			evo_ed.reserve(_ops.size());
			zero_ed.reserve(_ops.size());
			for (size_t i = 0; i < _ops.size(); ++i)
			{
				evo_ed.emplace_back(arma::Mat<T>(_ntimes, _nrealiz, arma::fill::ones) * (-1e5));
				zero_ed.emplace_back(arma::Col<T>(_nrealiz, arma::fill::ones) * (-1e5));
				LOGINFO("time_evo_ed[" + STR(e) + "][" + STR(i) + "]: " + STR(_ntimes) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 4);
				LOGINFO("time_evo_zero_ed[" + STR(e) + "][" + STR(i) + "]: " + STR(_nrealiz), LOG_TYPES::TRACE, 4);
			}
			_r.time_evo_ed.emplace_back(std::move(evo_ed));
			_r.time_evo_zero_ed.emplace_back(std::move(zero_ed));

			// microcanonical averages
			_r.microcanonical_ed.emplace_back(std::vector<arma::Mat<T>>(_ops.size(),
								arma::Mat<T>(_to_check_microcanonical_eps.size(), _nrealiz)));

			_r.microcanonical_ed2.emplace_back(
				std::vector<arma::Mat<double>>(
					_ops.size(),
					arma::Mat<double>(_to_check_microcanonical_eps.size(), _nrealiz)
				)
			);

			for (size_t i = 0; i < _ops.size(); ++i) {
				LOGINFO("microcanonical_ed[" + STR(e) + "][" + STR(i) + "]: " + STR(_to_check_microcanonical_eps.size()) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 4);
				LOGINFO("microcanonical_ed2[" + STR(e) + "][" + STR(i) + "]: " + STR(_to_check_microcanonical_eps.size()) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 4);
			}

			_r.diagonal_ed.emplace_back(_ops.size(), _nrealiz);
			LOGINFO("diagonal_ed[" + STR(e) + "]: " + STR(_ops.size()) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 4);
		}

		// ----------- Microcanonical & diagonal for ME -----------
		LOGINFO("Allocating microcanonical & diagonal for ME...", LOG_TYPES::TRACE, 2);
		_r.microcanonical_me .assign(
			_ops.size(),
			arma::Mat<T>(_to_check_microcanonical_eps.size(), _nrealiz)
		);
		_r.microcanonical_me2.assign(
			_ops.size(),
			arma::Mat<double>(_to_check_microcanonical_eps.size(), _nrealiz)
		);
		for (size_t i = 0; i < _ops.size(); ++i) {
			LOGINFO("microcanonical_me[" + STR(i) + "]: " + STR(_to_check_microcanonical_eps.size()) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
			LOGINFO("microcanonical_me2[" + STR(i) + "]: " + STR(_to_check_microcanonical_eps.size()) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		}
		_r.diagonal_me.set_size(_ops.size(), _nrealiz);
		LOGINFO("diagonal_me: " + STR(_ops.size()) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);

		// ----------- Participation & entanglement entropies -----------
		LOGINFO("Allocating participation & entanglement entropies...", LOG_TYPES::TRACE, 2);
		_r.time_participation_entropy.set_size(_ntimes, _nrealiz);
		LOGINFO("time_participation_entropy: " + STR(_ntimes) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);
		LOGINFO("entropies_sites: " + STR(_entropies_sites.size()), LOG_TYPES::TRACE, 3);
		for (size_t s = 0; s < _entropies_sites.size(); ++s) {
			_r.time_ee.emplace_back(_ntimes, _nrealiz);
			LOGINFO("time_ee[" + STR(s) + "]: " + STR(_ntimes) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 4);
		}
		_r.time_ee_bipartite.set_size(_ntimes, _nrealiz);
		LOGINFO("time_ee_bipartite: " + STR(_ntimes) + " x " + STR(_nrealiz), LOG_TYPES::TRACE, 3);

		// ----------- Finalize -----------
		LOGINFO("Result containers allocated. Resetting realization counter.", LOG_TYPES::TRACE, 2);
		_r.realisation_count = 0;
	}

	template void TimeEvoParams<double>::allocate_result_containers();
	template void TimeEvoParams<std::complex<double>>::allocate_result_containers();

}; // namespace TimeEvo

// --------------------------------------------------------------------

template<typename _T>
void UI::checkETH_time_evo(std::shared_ptr<Hamiltonian<_T>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

    TimeEvo::TimeEvoParams<_T> _p;
    _p.set_hamil_params((MY_MODELS)_H->getTypeI(), _H, this->latP.Ntot_, this->modP.getRanReal());

    //! get info
	std::string modelInfo, dir = "ETH_MAT_TIME_EVO", randomStr, extension;
    this->get_inf_dir_ext_r(_H, _p.dir, _p.modelInfo, _p.randomStr, _p.extension);

	//! get the operators
	std::tie(_p._ops, _p._opsN) = this->ui_eth_getoperators(_p._Nh, _p._isQuadratic, _p._isManyBody);

	//! create the measurement class
	// Simulation::DataContainer _container;
	Measurement<double> _measure(this->latP.Ntot_, _p.dir, _p._ops, _p._opsN, 1, _p._Nh);
	// -----------------------
	long _single_run_time 	= -1;
	long _remaining_time	= Slurm::get_remaining_time();
	LOGINFO("Remaining time: " + STR(_remaining_time) + " seconds", LOG_TYPES::INFO, 0);
	_timer.checkpoint("START");
	// -----------------------
	_p.allocate_result_containers();

#ifdef USE_MKL
    mkl_set_num_threads(1);
#endif

	//! go through realizations
	for (int _r = 0; _r < _p._nrealiz; ++_r)
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
			_diagonal 				= _H->getDiag();
			_p._r.initial_state_me 	= SystemProperties::TimeEvolution::create_initial_quench_state<_T>(SystemProperties::TimeEvolution::QuenchTypes::SEEK,
										_p._Nh, _p._Ns, _H->getEnAv(), _diagonal);
			
			//! create states for energy densities
			for (int i = 0; i < _p._energy_densities.size(); ++i)
			{
				double _energy_state 				= _H->getEigVal(0) + _p._energy_densities[i] * _H->getBandwidth();
				_p._r.initial_state_ed.col(i) 		= SystemProperties::TimeEvolution::create_initial_quench_state<_T>(SystemProperties::TimeEvolution::QuenchTypes::SEEK,
														_p._Nh, _p._Ns, _energy_state, _diagonal);
			}	
		}

		// -----------------------------------------------------------------------------

		//! calculator of the properties
		{
			// mean level spacing
			{
				_p._r.energies.col(_r)				= _H->getEigVal();
				_p._r.mean_level_spacing(0, _r)		= _H->getMeanLevelSpacing();
				// _p._r.mean_level_spacing(1, _r)		= SystemProperties::mean_lvl_heis_time(_p._r.energies(0, _r), _p._Ns);

				// get the average energy index and the points around it on the diagonal
				const auto [_min, _max] 			= _H->getEnArndAvIdx(_p._hs_fraction_diag / 2, _p._hs_fraction_diag / 2);
				auto _E								= _H->getEigVal().subvec(_min, _max);
				_p._r.mean_level_spacing(2, _r)		= 1.0 / SystemProperties::mean_lvl_spacing(_E);
				// get the Heisenberg frequency from the mean level spacing
				long double _h_freq					= SystemProperties::mean_lvl_spacing_typ(_E);
				_p._r.mean_level_spacing(3, _r)		= 1.0 / _h_freq;
				auto _meanlvl 						= _p._r.mean_level_spacing.col(_r);
				LOGINFO(StrParser::colorize(VEQ(_min), StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_max), StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQP(_meanlvl(0), 10) 	+ ": mean level spacing", StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(1)) 		+ ": mean level Heisenberg time", StrParser::StrColors::blue), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(2)) 		+ ": mean level Heisenberg time around energy ", StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(3)) 		+ ": mean level Heisenberg time around energy - typical", StrParser::StrColors::yellow), LOG_TYPES::TRACE, 1);
			}

			// -----------------------------------------------------------------------------

			// other measures
			{
				// calculate the diagonals
				const auto& _matrices 	= _measure.getOpG_mat();
				const auto& _eigvec 	= _H->getEigVec();
				
				// save the diagonals
#pragma omp parallel for num_threads(_p._Nh < ULLPOW(14) ? this->threadNum : 2)
				for (int _opi = 0; _opi < _p._ops.size(); ++_opi)
				{
					LOGINFO("Looking for diagonal of operator " + STR(_opi), LOG_TYPES::TRACE, 3);
					_p._r.diagonals[_opi].col(_r) = Operators::applyOverlapMat(_eigvec, _matrices[_opi]).diag();
				}
				_timer.checkpoint(STR(_r) + ": time evolution");

				//! evolve the states

				LOGINFO("Evolving the mean energy state", LOG_TYPES::TRACE, 3);
				TimeEvo::evolve_state(
					_r,
					_p._r.initial_state_me,
					_H,
					&_p._r.ldos_me,
					&_p._r.energy_densities_me,
					&_p._r.microcanonical_me,
					&_p._r.microcanonical_me2,
					&_p._r.diagonal_me,
					_p._r.diagonals,
					_p._r.time_evo_me,
					&_p._r.time_evo_zero_me,
					&_p._r.time_ee,
					// nullptr,
					&_p._r.time_ee_bipartite,
					&_p._r.time_participation_entropy,
					_matrices,
					_p._timespace,
					_p._Ns,
					_p._entropies_sites,
					_p._to_check_microcanonical_eps,
					false,
					false,
					false,
					this->threadNum
				);

				// do the same for the energy densities
				for (int i = 0; i < _p._energy_densities.size(); ++i)
				{
					LOGINFO("Evolving the energy density state eps=" + STR(_p._energy_densities[i]), LOG_TYPES::TRACE, 3);
					TimeEvo::evolve_state(
						_r,
						arma::Col<_T>(_p._r.initial_state_ed.col(i)),
						_H,
						&_p._r.ldos_ed[i],
						&_p._r.energy_densities_ed[i],
						&_p._r.microcanonical_ed[i],
						&_p._r.microcanonical_ed2[i],
						&_p._r.diagonal_ed[i],
						_p._r.diagonals,
						_p._r.time_evo_ed[i],
						&_p._r.time_evo_zero_ed[i],
						nullptr,
						nullptr,
						nullptr,
						_matrices,
						_p._timespace,
						_p._Ns,
						{},
						_p._to_check_microcanonical_eps,
						false,
						false,
						false,
						this->threadNum
					);
				}

				LOGINFO(_timer.point(STR(_r) + ": time evolution"), "Time evolution: " + STR(_r), 3);
			}
		}
		
		// -----------------------------------------------------------------------------

		// get single runtime
		_single_run_time = _timer.elapsed<long>(STR(_r), Timer::TimePrecision::SECONDS);
		LOGINFO("Single run time: " + STR(_single_run_time) + " seconds", LOG_TYPES::TRACE, 1);

		// save the checkpoints
		if (check_saving_size(_p._Nh, _r) && symP.checkpoint_)
			_p.save_for_realization(_r);

		if (this->remainingSlurmTime(_r, &_timer, _single_run_time, _remaining_time))
			break;

		_p._r.realisation_count++;

		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '%', 1);
		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_p.save_for_realization(_p._nrealiz);

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

template void UI::checkETH_time_evo<double>(std::shared_ptr<Hamiltonian<double>> _H);
template void UI::checkETH_time_evo<cpx>(std::shared_ptr<Hamiltonian<cpx>> _H);

// -------------------------------------------------------------------------------------