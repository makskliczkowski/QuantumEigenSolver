#pragma once
/***************************************
* Defines the methods for statistical 
* properties of the quantum systems
* FEB 2024, UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#include "../hamil.h"

#define SYSTEM_PROPERTIES_MIN_SPACING 1e-15
#define SYSTEM_PROPERTIES_THROW_DEGENERATE 1
#define SYSTEM_PROPERTIES_USE_OPENMP 0

// -------------------------------------------------------------------------------

namespace SystemProperties
{
	namespace TimeEvolution
	{
		/*
		* @brief Calculate the time evolution of the state. The 
		* time evolution is calculated based on the eigenstates and
		* eigenvalues of the system. The overlaps are also given. This specifies the 
		* coefficients of the state at time t = 0 with each of the eigenvectors of the system.
		* @param _eigenstates - the eigenstates of the system
		* @param _eigvals - the eigenvalues of the system
		* @param _overlaps - the overlaps of the state with the eigenstates
		* @returns the time evolved state
		*/
		template <typename _T, typename _S>
		[[nodiscard]]
		inline arma::Col<std::complex<double>> time_evo(const arma::Mat<_T>& _eigenstates, 
														const arma::Col<double>& _eigvals,
														const _S& _overlaps, 
														const double _time,
														size_t _threads = 1)
		{
			arma::Col<std::complex<double>> _ret(_eigenstates.n_cols, arma::fill::zeros);
			arma::Col<std::complex<double>> _exp = arma::exp(-I * _time * _eigvals);
		
			// go through the eigenstates
//#ifndef _DEBUG
//#pragma omp parallel for num_threads(_threads) reduction(+: _ret)
//#endif // _DEBUG
			for (auto i = 0; i < _eigenstates.n_cols; ++i)
			{
				_ret += _exp(i) * _overlaps(i) * _eigenstates.col(i);
			}
			return _ret;
		}

		// ---------------------------------------------------------------------------

		enum class QuenchTypes
		{
			// random
			RANDP,		
			RANDN,
			RANDU,
			// antiferromagnetic
			AF_UP,
			AF_DN,
			// ferromagnetic
			F_UP,
			F_DN,
			// domain walls
			DW_HALF_UP,
			DW_HALF_DN,
			DW_THIRD_UP,
			DW_THIRD_DN,
			// energy
			MEAN,
			SEEK
		};

		/*
		* @brief Create the initial state after the quench. The quench is defined by the initial state
		* and the type of the quench. The quench can be random, antiferromagnetic, ferromagnetic, or domain walls.
		* @param _type - the type of the quench
		* @param _Nh - the size of the Hilbert space
		* @param _Ns - the number of spins
		* @returns the initial state
		*/
		template <typename _T>
		[[nodiscard]]
		inline arma::Col<_T> create_initial_quench_state(QuenchTypes _type, u64 _Nh, uint _Ns)
		{
			arma::Col<_T> _ret(_Nh, arma::fill::zeros);
			switch (_type)
			{
			case SystemProperties::TimeEvolution::QuenchTypes::RANDP:
			{
				u64 _idx = 0;
				for (int i = 0; i < _Ns; ++i)
				{
					double _p = arma::randu<double>(arma::distr_param(0.0, 1.0));
					if (_p < 0.5)
						_idx |= 1 << i;
				}
				_ret(_idx) = 1.0;
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::RANDN:
			{
				_ret = arma::randn<arma::Col<_T>>(_ret.n_elem);
				_ret /= arma::norm(_ret);
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::RANDU:
			{
				_ret = arma::randu<arma::Col<_T>>(_ret.n_elem);
				_ret /= arma::norm(_ret);
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::AF_UP:
			{
				u64 _idx = _Ns % 2 == 0 ? _Nh / 3 : (_Nh + 1) / 3;
				_ret(_idx) = 1.0;
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::AF_DN:
			{
				u64 _idx = _Ns % 2 == 0 ? _Nh / 3 : (_Nh + 1) / 3;
				Binary::flipAll(_idx, _Ns);
				_ret(_idx) = 1.0;
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::F_UP:
			{
				_ret(_Nh - 1) = 1.0;
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::F_DN:
			{
				_ret(0) = 1.0;
				break;
			}
			// domain walls
			case SystemProperties::TimeEvolution::QuenchTypes::DW_HALF_UP:
			{
				_ret(ULLPOW((int(_Ns / 2) - 1))) = 1.0;
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::DW_HALF_DN:
			{

				u64 _idx = ULLPOW(((_Ns / 2) - 1));
				Binary::flipAll(_idx, _Ns);
				_ret(_idx) = 1.0;
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::DW_THIRD_UP:
			{
				u64 _idx = ULLPOW((int(_Ns / 3) - 1));
				_ret(_idx) = 1.0;
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::DW_THIRD_DN:
			{
				u64 _idx = ULLPOW((int(_Ns / 3) - 1));
				Binary::flipAll(_idx, _Ns);
				_ret(_idx) = 1.0;
				break;
			}
			default:
				_ret(0) = 1.0;
				break;
			}
			return _ret;
		}

		template <typename _T, typename _ET>
		[[nodiscard]]
		inline arma::Col<_T> create_initial_quench_state(QuenchTypes _type, u64 _Nh, uint _Ns, double _Eseek, const _ET& _energies)
		{
			arma::Col<_T> _ret(_Nh, arma::fill::zeros);
			
			switch (_type)
			{
			case SystemProperties::TimeEvolution::QuenchTypes::SEEK:
			{
				auto _idx	= arma::index_min(arma::abs(_energies - _Eseek));
				_ret(_idx)	= 1.0;
				break;
			}
			case SystemProperties::TimeEvolution::QuenchTypes::MEAN:
			{
				auto _mean	= arma::mean(_energies);
				auto _idx	= arma::index_min(arma::abs(_energies - _mean));
				_ret(_idx)	= 1.0;				
				break;
			}
			default:
				return create_initial_quench_state<_T>(_type, _Nh, _Ns);
				break;
			}
			return _ret;
		}


		// ---------------------------------------------------------------------------

		/*
		* @brief Calculate the mean energy after quench
		* @param _H - the Hamiltonian matrix
		* @param _state - the state
		* @returns the mean energy
		*/
		template <typename _T, typename _M>
		[[nodiscard]]
		inline typename std::common_type<_T,_M>::type calc_mean_energy_quench(const arma::SpMat<_M>& _H, const arma::Col<_T>& _state)
		{
			using _R	= typename std::common_type<_T, _M>::type;
			auto _out	= arma::cdot(_state, _H * _state);
			return algebra::cast<_R>(_out);
		}

	};

	// ---------------------------------------------------------------------------

	// -------------------------- P R O P E R T I E S ----------------------------

	// ---------------------------------------------------------------------------

	/*
	* @brief Caculate the Local density of states (LDOS) of the system. Otherwise known as the
	* strength function. The LDOS is calculated based on the energies and overlaps of the system.
	* @param _energies - the energies of the system
	* @param _ovelaps - the overlaps of the system
	* @param _degenerate - whether the system is degenerate
	* @returns the LDOS
	* @references PHYSICAL REVIEW B 99, 174313 (2019)
	*/
	template <typename _T>
	[[nodiscard]]
	inline arma::Col<double> calculate_LDOS(const arma::Col<double>& _energies, const arma::Col<_T>& _ovelaps, bool _degenerate = false)
	{
		if(!_degenerate)
			return arma::square(arma::abs(_ovelaps));
		else 
			{
			arma::Col<double> _ret(_energies.size(), arma::fill::zeros);
			for (u64 i = 0; i < _energies.size(); ++i)
			{
				auto _en = _energies(i);
				for (u64 j = 0; j < _energies.size(); ++j)
				{
					if (std::abs(_en - _energies(j)) < SYSTEM_PROPERTIES_MIN_SPACING)
						_ret(i) += std::norm(_ovelaps(j));
				}
			}
			return _ret;
		}
	}

	template <typename _T>
	[[nodiscard]]
	inline arma::uvec calculate_DOS(const _T& _energies, size_t _nbins = 100)
	{
		return arma::hist(_energies, _nbins);
	}

	// ---------------------------------------------------------------------------

	// -------------------- H I L B E R T   F R A C T I O N S --------------------

	// ---------------------------------------------------------------------------

	/*
	* @brief Get the Hilbert space fraction based on the parameter given by the user. It 
	* depends on the size of the Hilbert space - checks not to make the fraction too big
	* @param _fraction fraction of the Hilbert space to take
	* @param _hilbertSize size of the Hilbert space
	* @returns Hilbert space fraction
	*/
	inline u64 hs_fraction_diagonal_cut(long double _fraction, u64 _hilbertSize)
	{
		if (_fraction == 1.0)
			return _hilbertSize;

		u64 _states = (_fraction >= 1.0) ? std::min(u64(_fraction), _hilbertSize) : std::max(u64(4), u64(_fraction * _hilbertSize));
		if (_states >= _hilbertSize)
			return _hilbertSize;
		return _states;
	}

	// ---------------------------------------------------------------------------

	/*
	* @brief Get the specific indices in a range around a given index in the Hilber space. 
	* Checks for the boundaries as well.
	* @param _l number of elements to the left from the average energy
	* @param _r number of elements to the right from the average energy
	* @param _idx index around which to look for the boundaries
	* @param _hilbertSize size of the Hilbert space
	* @returns pair of indices of the energy spectrum around the average energy	*/
	inline std::pair<u64, u64> hs_fraction_around_idx(long long _l, long long _r, u64 _idx, u64 _hilbertSize)
	{
		u64 _min = std::max(0ll, (long long)_idx - _l);
		u64 _max = std::min(_hilbertSize - 1, _idx + _r);
		return std::make_pair(_min, _max);
	}

	// ---------------------------------------------------------------------------

	/*
	* @brief Targets the mean energy to be close to a target energy with some tolerance precision.
	* @param _l left energy
	* @param _r right energy
	* @param _target target energy
	* @param _tol tolerance of the proximity
	* @returns whether is close
	*/
	inline bool hs_fraction_close_mean(double _l, double _r, double _target = 0.0, double _tol = 0.0015)
	{
		return std::abs((_l +_r) / 2.0 - _target) < _tol;
	}

	// ---------------------------------------------------------------------------

	/*
	* @brief Targets the energy difference to be close to a target energy with some tolerance precision.
	* @param _l left energy
	* @param _r right energy
	* @param _target target energy
	* @param _tol tolerance of the proximity
	* @returns whether is close
	*/
	inline bool hs_fraction_diff_close_en(double _l, double _r, double _target = 0.0, double _tol = 0.0015)
	{
		return std::abs(std::abs(_l - _r) - _target) < _tol;
	}

	// ---------------------------------------------------------------------------

	/*
	* @brief Targets the energy difference to be close to a target energy with some tolerance precision.
	* @param _l left energy
	* @param _r right energy
	* @param _min minimum energy difference
	* @param _max maximum energy difference
	* @returns whether is close
	*/
	inline bool hs_fraction_diff_between(double _l, double _r, double _min, double _max)
	{
		auto _w = std::abs(_l - _r);
		return _w >= _min && _w <= _max;
	}

	// ---------------------------------------------------------------------------

	inline std::vector<std::tuple<double, u64, u64>> hs_fraction_offdiag(u64 _mn, u64 _max, u64 _hilbertSize, 
																	const arma::Col<double>& _energies,
																	double _targetEn	= 0.0, 
																	double _tol			= 0.0015,
																	bool _sort			= true)
	{
		v_1d<std::tuple<double, u64, u64>> _out;

		// go through the whole spectrum (do not save pairs, only one element as it's Hermitian.
		for (u64 i = _mn; i < _max; ++i)
		{
			auto _en_l = _energies(i);
			for (u64 j = i + 1; j < _max; ++j)
			{
				auto _en_r = _energies(j);
				// check the energy difference
				if (SystemProperties::hs_fraction_close_mean(_en_l, _en_r, _targetEn, _tol))
					_out.push_back(std::make_tuple(std::fabs(_en_r - _en_l), j, i));
			}
		}

		// sort the omegas, cause why not
		Containers::sort<0>(_out, [](const auto& a, const auto& b) { return a < b; });
		return _out;
	}

	// ---------------------------------------------------------------------------

	// ------------------- T R A N S F O R M   O P E R A T O R -------------------

	// ---------------------------------------------------------------------------

	/*
	* @brief Transforms the operator to the new basis
	* @param _op the operator
	* @param _basis the new basis
	* @returns the transformed operator
	*/
	template <typename _T>
	[[nodiscard]]
	inline arma::Mat<_T> transform_operator(const arma::Mat<_T>& _op, const arma::Mat<_T>& _basis)
	{
		arma::Mat<_T> _ret	= _basis * (_op * _basis.t());
		return _ret;
	}

	// ---------------------------------------------------------------------------

	// -------------------- M A T R I X   P R O P E R T I E S --------------------

	// ---------------------------------------------------------------------------

	/*
	* @brief Calculates the Hilbert Schmidt norm of the matrix.
	* The Hilbert Schmidt norm is defined as the trace of the square of the matrix
	* divided by the number of elements in the matrix.
	* @param _mat - the matrix
	* @returns the Hilbert Schmidt norm
	*/
	template<typename _T>
	[[nodiscard]]
	_T hilber_schmidt_norm(const arma::Mat<_T>& _mat)
	{
		return arma::trace(_mat * _mat) / (double)_mat.n_rows;
	}

	// ---------------------------------------------------------------------------
	
	// -------------------------- EIGENLEVEL STATISTICS -------------------------- 

	// ---------------------------------------------------------------------------

	/*
	* @brief Calculates the eigenlevel statistics of the system
	* aka the gap ratio. The gap ratio is a measure of the level spacing
	* repulsion. 
	* @param begin - iterator to the beginning of the eigenvalues
	* @param end - iterator to the end of the eigenvalues
	* @returns the gap ratio
	*/
	template <typename _iterator>
	[[nodiscard]]
	inline long double eigenlevel_statistics(_iterator begin, _iterator end)
	{
		// number of elements inside the range 
		// (move them, so that we can define
		// previous and next element)
		const _iterator first	= std::next(begin);
		const _iterator last	= std::prev(end);
		auto E_prev				= *begin;

		long double r			= 0.0;
		for (auto it = first; it != last; ++it) 
		{
			auto E_next			= *std::next(it);

			const double delta_n        = (*it) - E_prev;
			const double delta_n_next   = E_next - (*it);

			const double min = std::min(delta_n, delta_n_next);
			const double max = std::max(delta_n, delta_n_next);

			if (std::abs(delta_n) <= SYSTEM_PROPERTIES_MIN_SPACING)
			{ 
#ifdef _DEBUG
				LOGINFO(VEQ(*it), LOG_TYPES::ERROR, 1);
#endif
				LOGINFO("The level spacing is degenerate, the eigenlevel statistics might be wrong", LOG_TYPES::ERROR, 1);
#if SYSTEM_PROPERTIES_THROW_DEGENERATE
				throw std::invalid_argument("The level spacing is degenerate, the eigenlevel statistics might be wrong");
#endif			
			}
			r += min / max;

			E_prev = (*it);
		}
		return r / std::distance(first, last);
	}

	// ---------------------------------------------------------------------------

	template <typename _iterator, typename _Container>
	inline void eigenlevel_statistics(_iterator begin, _iterator end, _Container& _vec)
	{
		const size_t size       = std::distance(begin, end);
		// reset container
		_vec					= _Container(size - 2);

		for (auto it = begin; it != end; ++it) 
			_vec[std::distance(begin, it)] = eigenlevel_statistics(it, it + 3);
	}

	// ---------------------------------------------------------------------------

	template <typename _Container>
	[[nodiscard]]
	inline long double eigenlevel_statistics(const _Container& _energies)
	{
		return eigenlevel_statistics(_energies.begin(), _energies.end());
	}

	// ---------------------------------------------------------------------------

	template <typename _Container>
	inline void eigenlevel_statistics(const arma::Col<double>& _energies, _Container& _vec)
	{
		eigenlevel_statistics(_energies.begin(), _energies.end(), _vec);
	}

	// ---------------------------------------------------------------------------

	// -------------------------- PARTICIPATION RATIO ----------------------------

	// ---------------------------------------------------------------------------

	/*
	* @brief Calculates the participation ratio of the state
	* @param _state - the state
	* @param q - the exponent
	* @returns the participation ratio
	*/
	template <typename _C> 
	[[nodiscard]]
	inline long double participation_ratio(const _C& _state, double q = 1.0)
	{
		long double pr	= 0;
#if SYSTEM_PROPERTIES_USE_OPENMP
#	pragma omp parallel for reduction(+: pr)
#endif
		for (auto& _coeff: _state) 
			pr += std::pow(std::abs(algebra::conjugate(_coeff) * _coeff), q);
		return pr;
	}

	// ---------------------------------------------------------------------------

	/*
	* @brief Calculates the inverse participation ratio of the state
	* @param _state - the state
	* @param q - the exponent
	* @returns the inverse participation ratio
	*/
	template <typename _C> 
	[[nodiscard]]
	inline long double inverse_participation_ratio(const _C& _state)
	{
		long double ipr	= 0;
#if SYSTEM_PROPERTIES_USE_OPENMP
#	pragma omp parallel for reduction(+: ipr)
#endif
		for (auto& _coeff : _state)
		{
			auto _v = std::abs(algebra::conjugate(_coeff) * _coeff);
			ipr		+= _v * _v;
		}
		return ipr;
	}

	// ---------------------------------------------------------------------------

	// -------------------------- INFORMATION ENTROPY ----------------------------

	// ---------------------------------------------------------------------------

	/*
	* @brief Calculates the information entropy of the state
	* @param _state - the state
	* @returns the information entropy
	*/
	template <typename _C> 
	[[nodiscard]]
	inline long double information_entropy(const _C& _state) 
	{
		long double ent = 0;
#if SYSTEM_PROPERTIES_USE_OPENMP
#	pragma omp parallel for reduction(+: ent)
#endif
		for (auto& _coeff : _state)
		{
			auto _v = std::abs(algebra::conjugate(_coeff) * _coeff);
			ent += _v * std::log(_v);
		}
		return -ent; // std::log(0.48 * _state.size());
	}

	// ---------------------------------------------------------------------------

	/*
	* @brief Calculates the information entropy of the state.
	* The information entropy is calculated in the new basis
	* This corresponds to the situation when the system is slightly 
	* perturbed and the new basis is used to calculate the information entropy.
	* @param _state - the state
	* @param _newBasis - the new basis
	* @param _min - the minimum index inside new basis
	* @param _max - the maximum index inside new basis
	* @returns the information entropy
	*/
	template <typename _T> 
	[[nodiscard]]
	inline long double information_entropy(	const arma::Col<_T>& _state, 
											const arma::Mat<_T>& _newBasis,
											u64 _min = 0,                           
											u64 _max = -1) 
	{
		long double ent = 0;
#if SYSTEM_PROPERTIES_USE_OPENMP
#	pragma omp parallel for reduction(+: ent)
#endif
		for (long long _k = _min; _k < std::min(_max, _state.size()); ++_k)
		{
			auto _c		= arma::cdot(_newBasis.col(_k), _state);
			auto _v		= std::abs(algebra::conjugate(_c) * _c);
			ent			+= _v * std::log(_v);
		}
		return -ent / std::log(0.48 * _state.size());
	}

	// ---------------------------------------------------------------------------

	// ------------------------- O T H E R   H A M I L ---------------------------

	// ---------------------------------------------------------------------------
	
	/*
	* @brief Calculate the mean level spacing of the Hamiltonian matrix.
	* @param _E - the eigenvalues of the Hamiltonian matrix
	*/
	template <typename _T>
	[[nodiscard]]
	inline double mean_lvl_spacing(const _T& _E)
	{
		return arma::mean(arma::diff(_E));
	}

	template <typename _T>
	[[nodiscard]]
	inline double mean_lvl_spacing_typ(const _T& _E)
	{
		return std::exp(arma::mean(arma::log(arma::diff(_E))));
	}

	/*
	* @brief Calculates the mean level spacing of the Hamiltonian matrix. See the 
	* reference Suntajs, Vidmar, Ergodicity Breaking transition in zero dimensions, PRL 2022 for more details.
	* @param _H - the Hamiltonian matrix
	* @returns the mean level spacing
	*/
	template <typename _T>
	[[nodiscard]]
	inline long double mean_lvl_gamma(const _T& _H)
	{
		const auto _Nh		= _H.n_rows;
		const auto _trace	= arma::trace(_H) / (long double)_Nh;
		const auto _trace2	= arma::trace(arma::square(_H)) / (long double)_Nh;
		return _trace2 - _trace * _trace;
	}

	/*
	*/
	inline double mean_lvl_xi(double gamma2)
	{
		auto _gamma			= std::sqrt(gamma2);
		auto _NE			= 1000;
		arma::vec _energies = arma::linspace(0, _gamma, _NE);
		auto _dE			= _gamma / (double)_NE;
		arma::vec _values	= arma::exp(-arma::square(_energies) / 2.0 / gamma2) / std::sqrt(TWOPI * gamma2);
		return arma::sum(_values * _dE);
	}

	/*
	* @brief Calculates the Heisenberg time from the mean level spacing obtained from the Hamiltonian matrix.
	* @param gamma2 - the mean level spacing
	* @param L - the system size
	* @returns the Heisenberg time
	*/
	inline double mean_lvl_heis_time(double gamma2, int L)
	{
		const auto xi = mean_lvl_xi(gamma2);
		return xi * (long double)(ULLPOW(L)) / std::sqrt(gamma2);

	}

	/*
	* @brief Calculates the Heisenberg frequency from the mean level spacing obtained from the Hamiltonian matrix.
	* @param gamma2 - the mean level spacing
	* @param L - the system size
	* @returns the Heisenberg frequency
	*/
	inline double mean_lvl_heis_freq(double gamma2, int L)
	{
		return 1.0 / mean_lvl_heis_time(gamma2, L);
	}

	// ---------------------------------------------------------------------------
};