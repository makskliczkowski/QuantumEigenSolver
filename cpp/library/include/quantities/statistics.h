#pragma once
/***************************************
* Defines the methods for statistical 
* properties of the quantum systems
* FEB 2024, UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#include "armadillo"
#ifndef SYSTEM_PROPERTIES_H
#define SYSTEM_PROPERTIES_H

#ifndef SYMMETRIES_H
#	include "../hilbert_sym.h"
#endif // !SYMMETRIES_H

#ifndef OPERATORS_H
#	include "../algebra/operators.h"
#endif // !OPERATORS_H

constexpr auto SYSTEM_PROPERTIES_MIN_SPACING = 1e-15;
constexpr auto SYSTEM_PROPERTIES_THROW_DEGENERATE = 1;
constexpr auto SYSTEM_PROPERTIES_COEFF_THRESHOLD = 1e-11;
constexpr auto SYSTEM_PROPERTIES_USE_OPENMP = 0;

// -------------------------------------------------------------------------------

namespace SystemProperties
{
	namespace TimeEvolution
	{
		/**
		* @brief Computes the time evolution of a quantum state.
		*
		* This function calculates the time evolution of a quantum state given its eigenstates, 
		* eigenvalues, and overlaps. The result is a column vector of complex numbers representing 
		* the evolved state at a specified time.
		*
		* @tparam _T Type of the elements in the eigenstates matrix.
		* @tparam _S Type of the overlaps vector.
		* @param _eigenstates Matrix of eigenstates (each column is an eigenstate).
		* @param _eigvals Column vector of eigenvalues corresponding to the eigenstates.
		* @param _overlaps Vector of overlaps (coefficients) of the initial state with the eigenstates.
		* @param _time Time at which to evaluate the evolved state.
		* @param _threads Number of threads to use for parallel computation (default is 1).
		* @return Column vector of complex numbers representing the evolved state at the specified time.
		*/
		template <typename _T, typename _S = arma::Col<_T>>
		[[nodiscard]]
		inline arma::Col<std::complex<double>> time_evo(const arma::Mat<_T>& _eigenstates, 
														const arma::Col<double>& _eigvals,
														const _S& _overlaps, 
														const double _time,
														size_t _threads = 1)
		{
			return _eigenstates * (_overlaps % arma::exp(-cpx(0,_time) * _eigvals));
			// arma::Col<std::complex<double>> _exp = arma::exp(-I * _time * _eigvals);		// precompute the exponential
			// try armadillo multiplication
			// return _eigenstates * (_exp % _overlaps);
			// arma::Col<std::complex<double>> _ret(_eigenstates.n_cols, arma::fill::zeros);
			// for (auto i = 0; i < _eigenstates.n_cols; ++i)
				// _ret += (_exp(i) * _overlaps(i)) * _eigenstates.col(i);
			// return _ret;
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
				LOGINFO(VEQ(_mean) + ":" + VEQ(_idx), LOG_TYPES::TRACE, 3);
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

	}

	// ---------------------------------------------------------------------------

	// --------------------------- S P E C T R A L S -----------------------------

	// ---------------------------------------------------------------------------

	namespace AGP
	{
		/*
		* @brief Calculate the fidelity susceptability of the perturbation. The fidelity susceptability
		* is calculated based on the energies and overlaps of the system. The perturbation is given by the
		* matrix V (the perturbation matrix) and the cutoff mu.
		* @param _idx - the index of the energy
		* @param _energies - the energies of the system
		* @param _V - the perturbation matrix
		* @param _mu - the cutoff
		* @returns the fidelity susceptability for a given index (state)
		*/
		template <typename _T, typename _ET>
		[[nodiscard]] 
		inline _T fidelity_susceptability(const size_t _idx, const _ET& _energies, const GeneralizedMatrix<_T>& _V, double _mu)
		{
			// energy at _idx
			double _E 			= _energies(_idx);

			// go through other elements
			auto& _Velems 					= 	_V.row(_idx);
			const arma::Col<double> _omm  	= 	arma::square(_energies - _E);
			const double _mu2				= 	_mu * _mu;

			// calculate the sum
			_T _sum 		= 	0.0;
			for (size_t i = 0; i < _energies.size(); ++i)
			{
				if (i == _idx)
					continue;

				auto _nom = 	_Velems(i) * algebra::conjugate(_Velems(i)) * _omm(i);
				auto _den = 	(_omm(i) + _mu2);
				_sum 	  += 	_nom / _den / _den;
			}
			return _sum;
		}

		/**
		* @brief Computes the fidelity susceptibility for a given index.
		*
		* This function calculates the fidelity susceptibility for a given index `_idx` 
		* using the provided energy levels `_energies`, matrix `_V`, and parameter `_mu`.
		*
		* @tparam _T The type of the elements in the matrix `_V`.
		* @tparam _ET The type of the energy levels.
		* @param _idx The index for which the fidelity susceptibility is computed.
		* @param _energies A container holding the energy levels.
		* @param _V A matrix containing the eigenvectors.
		* @param _mu A parameter used in the computation.
		* @return The computed fidelity susceptibility.
		*/
		template <typename _T, typename _ET>
		[[nodiscard]]
		inline _T fidelity_susceptability(const size_t _idx, const _ET& _energies, const arma::Mat<_T>& _V, double _mu)
		{
			// energy at _idx
			double _E 						= 	_energies(_idx);

			// Precompute constants
			const arma::Row<_T>& _Velems 	= _V.row(_idx);
			const arma::Col<double> _omm 	= arma::square(_energies - _E);
			const double _mu2 				= _mu * _mu;

			// Accumulate the sum
			_T _sum = 0.0;
			for (size_t i = 0; i < _energies.size(); ++i)
			{
				if (i == _idx) continue;

				// Use std::norm to compute |V_nm|^2 efficiently
				double _V_sq = std::norm(_Velems(i));

				// Precompute the denominator once
				double _den = _omm(i) + _mu2;

				// Accumulate the result
				_sum += (_V_sq * _omm(i)) / (_den * _den);
			}

			return _sum;
		}

		template <typename _T, typename _ET >
		inline void fidelity_susceptability_tot(const _ET& _energies, const GeneralizedMatrix<_T>& _V, double _mu, arma::subview_col<_T>& _out)
		{
			// _out.set_size(_energies.size());
#pragma omp parallel for 
			for (size_t i = 0; i < _energies.size(); ++i)
				_out(i) = fidelity_susceptability<_T, _ET>(i, _energies, _V, _mu);
		}

		template <typename _T, typename _ET>
		inline void fidelity_susceptability_tot(const _ET& _energies, const arma::Mat<_T>& _V, double _mu, arma::subview_col<_T>& _out)
		{
			// _out.set_size(_energies.size());
#pragma omp parallel for
			for (size_t i = 0; i < _energies.size(); ++i)
				_out(i) = fidelity_susceptability<_T, _ET>(i, _energies, _V, _mu);
		}
		// ---------------------------------------------------------------------------
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
					// std::norm returns the |z|^2
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
	inline long double eigenlevel_statistics(const _iterator& begin, const _iterator& end)
	{
		// number of elements inside the range 
		// (move them, so that we can define
		// previous and next element)
		//const _iterator first	= std::next(begin);
		//const _iterator last	= std::prev(end);
		//auto E_prev				= *begin;
		auto it					= begin;
		auto E_prev				= *it;
		++it;
		auto E_cur				= *it;

		long double r			= 0.0;
		size_t _count			= 0;
		//for (auto it = first; it != last; ++it) 
		for(++it; it != end; ++it)
		{
			//auto E_next			= *std::next(it);
			auto E_next					= *it;

			//const double delta_n        = (*it) - E_prev;
			//const double delta_n_next   = E_next - (*it);
			const double delta_n        = E_cur - E_prev;
			const double delta_n_next   = E_next - (E_cur);

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

			//E_prev = (*it);
			E_prev = E_cur;
			E_cur  = E_next;
			++_count;
		}
		//return r / std::distance(first, last);
		return r / _count;
	}

	// ---------------------------------------------------------------------------

	//template <typename _iterator, typename _Container>
	//inline void eigenlevel_statistics(_iterator begin, _iterator end, _Container& _vec)
	//{
	//	const size_t size       = std::distance(begin, end);
	//	// reset container
	//	if(_vec.size() != size - 2)
	//		_vec				= _Container(size - 2);

	//	for (auto it = begin; it != end - 2; ++it)
	//	{
	//		auto _dist = std::distance(begin, it);
	//		_vec[_dist] = eigenlevel_statistics(it, it + 3);
	//	}
	//}

	template <typename _iterator, typename _Container>
	inline void eigenlevel_statistics(const _iterator& begin, const _iterator& end, _Container& _vec)
	{
		const size_t size       = std::distance(begin, end);
		// reset container
		if(_vec.size() != size - 2)
			_vec				= _Container(size - 2);

		for (auto it = begin; it != end - 2; ++it)
		{
			auto _dist = std::distance(begin, it);
			_vec[_dist] = eigenlevel_statistics(it, it + 3);
		}
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
	[[nodiscard]]
	inline long double eigenlevel_statistics(_Container& _energies)
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
			ipr		+= _v;
		}
		return ipr;
	}

	// ---------------------------------------------------------------------------

	/*
	* @brief Calculates the inverse participation ratio of the state
	* @param _state the state
	* @param q the exponent
	* @returns the participation ratio
	*/
	template <typename _C> 
	[[nodiscard]]
	inline long double inverse_participation_ratio(const _C& _state, double q)
	{
		long double ipr	= 0;
		if (q == 1.0)
			return inverse_participation_ratio(_state);

#if SYSTEM_PROPERTIES_USE_OPENMP
#	pragma omp parallel for reduction(+: pr)
#endif
		for (auto& _coeff : _state)
		{
			const auto abs_coeff = std::abs(algebra::conjugate(_coeff) * _coeff);
			if (abs_coeff > SYSTEM_PROPERTIES_COEFF_THRESHOLD)
				ipr += std::pow(abs_coeff, q);
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
			ent += EQP(_coeff, 0.0, SYSTEM_PROPERTIES_COEFF_THRESHOLD) ? 0.0 : _v * std::log(_v);
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
	template <typename _T, typename _T2> 
	[[nodiscard]]
	inline long double information_entropy(	const arma::Col<_T>& _state, 
											const arma::Mat<_T2>& _newBasis,
											u64 _min = 0,                           
											u64 _max = -1) 
	{
		using _typename 		= typename std::common_type<_T, _T2>::type;
		long double ent 		= 0;
		const u64 actual_max 	= std::min(_max, static_cast<u64>(_state.size()));

		// quicker, oy?
		if (_state.size() == actual_max)
		{
			const arma::Col<_typename> _c = _newBasis.t() * _state;
			return information_entropy(_c);
		}

#if SYSTEM_PROPERTIES_USE_OPENMP
#	pragma omp parallel for reduction(+: ent)
#endif
		for (long long _k = _min; _k < actual_max; ++_k)
		{
			_typename _c= 0.0;
			for (long long _j = 0; _j < _newBasis.n_cols; ++_j)
				_c += algebra::conjugate(_newBasis(_j, _k)) * _state(_j);			
			
			auto _v		= std::abs(algebra::conjugate(_c) * _c);
			if (_v > SYSTEM_PROPERTIES_COEFF_THRESHOLD)
				ent		+= _v * std::log(_v);
		}
		return -ent / std::log(0.48 * _state.size());
	}

	template <typename _T> 
	[[nodiscard]]
	inline long double information_entropy(const arma::Col<_T>& _state, const arma::Mat<_T>& _newBasis, const arma::Col<_T>& _energies, u64 _min = 0, u64 _max = -1)
	{
		long double ent 		= 0;
		const u64 actual_max 	= std::min(_max, static_cast<u64>(_state.size()));

		// quicker, oy?
		if (_state.size() == actual_max)
		{
			const arma::Col<_T> _c = _newBasis.t() * _state;
			return information_entropy(_c);
		}

#if SYSTEM_PROPERTIES_USE_OPENMP
#	pragma omp parallel for reduction(+: ent)
#endif
		for (long long _k = _min; _k < actual_max; ++_k)
		{
			auto _c		= arma::cdot(_newBasis.col(_k), _state);
			auto _v		= std::abs(algebra::conjugate(_c) * _c);
			if (_v > SYSTEM_PROPERTIES_COEFF_THRESHOLD)
				ent		+= _v * std::log(_v);
		}
		return -ent / std::log(0.48 * _state.size());
	}

	/*
	* @brief Calculates the participation ratio of the state
	* @param _state - the state
	* @param q - the exponent
	* @returns the participation ratio
	*/
	template <typename _C> 
	[[nodiscard]]
	inline long double participation_entropy(const _C& _state, double q = 1.0)
	{
		if (q == 1.0)
			return information_entropy(_state);
		return 1.0 / (1.0 - q) * std::log(inverse_participation_ratio(_state, q));
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
		using inner_type	= typename _T::elem_type;
		const auto _Nh		= _H.n_rows;
		const auto _trace	= algebra::cast<inner_type>(arma::trace(_H)) / (double)_Nh;
		const auto _trace2	= algebra::cast<inner_type>(arma::trace(arma::square(_H))) / (double)_Nh;
		return algebra::cast<double>(_trace2 - _trace * _trace);
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

	// -------------------- F O U R I E R   T R A N S F O R M --------------------

	// ---------------------------------------------------------------------------

	template <typename _Mt>
	[[nodiscard]]
	inline arma::Mat<cpx> function_fourier(const _Mt& _G)
	{			
		return arma::fft(_G);
	}

	template <typename _Mt>
	[[nodiscard]]
	inline arma::Mat<cpx> function_fourier(const _Mt& _G, const arma::Mat<cpx>& _DFT)
	{
		return _DFT * _G;
	}

	template <typename _Mt>
	[[nodiscard]]
	inline arma::Col<std::complex<double>> function_fourier_diag(const _Mt& _G, const arma::Mat<cpx>& _DFT)
	{
		return algebra::matDiagProduct(_DFT, _G);
	}

	template <typename _Mt>
	[[nodiscard]]
	inline arma::Col<std::complex<double>> function_fourier_diag_k(const _Mt& _G, const arma::Mat<cpx>& _exps, const arma::Mat<cpx>& _expst)
	{
		// Compute the result using matrix multiplication
		arma::Mat<cpx> temp		= _exps * _G;	// row times columns and sum
		// Compute the diagonal elements
		arma::Col<cpx> _ret(_G.n_rows, arma::fill::zeros);

		// rows are exponents for given k
		_ret = algebra::matDiagProduct(temp, _expst);
		//for (int k = 0; k < temp.n_rows; ++k)

		return _ret;
	}

	// ---------------------------------------------------------------------------

	template <typename _Mt>
	[[nodiscard]]
	inline cpx function_fourier(const _Mt& _G, Lattice* _lat, const int k)
	{
		auto _k = _lat->get_kVec(k);
		cpx _ret = 0.0;

		// go through lattice vectors
//#pragma omp parallel for reduction(+: _ret)
		for (int i = 0; i < _G.n_rows; i++)
		{
			auto _r1 = _lat->get_rVec(i);
			for (int j = 0; j < _G.n_cols; j++)
			{
				auto _r2 = _lat->get_rVec(j);
				_ret += _G(i, j) * std::exp(I * cpx(arma::dot(_k, (_r1 - _r2))));
			}
		}
		return _ret;
	}

	template<typename _Mt, typename _Vt>
	[[nodiscard]]
	inline cpx function_fourier(const _Mt& _G, Lattice* _lat, const _Vt& _k)
	{
		cpx _ret = 0.0;

		// go through lattice vectors
//#pragma omp parallel for reduction(+: _ret)
		for (int i = 0; i < _G.n_rows; i++)
		{
			auto _r1 = _lat->get_rVec(i);
			for (int j = 0; j < _G.n_cols; j++)
			{
				auto _r2 = _lat->get_rVec(j);
				_ret += _G(i, j) * std::exp(I * cpx(arma::dot(_k, (_r1 - _r2))));
			}
		}
		return _ret;
	}

	/*
	* @brief Calculate the Fourier transform of the function stored in the matrix G in the real space.
	* The first thing to do is to calculate the function as a diffrence of the positions in the matrix.
	* @param _G - the matrix with the function
	* @param _lat - the lattice
	* @param k1 - the first k vector
	* @param k2 - the second k vector
	* @returns the Fourier transformed function
	*/
	template <typename _Mt>
	[[nodiscard]]
	inline cpx function_fourier(const _Mt& _G, Lattice* _lat, const int k1, const int k2)
	{
		auto _k1 = _lat->get_kVec(k1);
		auto _k2 = _lat->get_kVec(k2);

		cpx _ret = 0.0;
		// go through lattice vectors
//#pragma omp parallel for reduction(+: _ret)
		for (int i = 0; i < _G.n_rows; i++)
		{
			const auto _r1 = _lat->get_rVec(i);
			for (int j = 0; j < _G.n_cols; j++)
			{
				const auto _r2 = _lat->get_rVec(j);
				_ret += _G(i, j) * std::exp(I * cpx(arma::dot(_k1, _r1) - arma::dot(_k2, _r2)));
			}
		}
		return _ret;
	}

	template <typename _Mt, typename _Vt>
	[[nodiscard]]
	inline cpx function_fourier(const _Mt& _G, Lattice* _lat, const _Vt& _k1, const _Vt& _k2)
	{
		cpx _ret = 0.0;

		// go through lattice vectors
//#pragma omp parallel for reduction(+: _ret)
		for (int i = 0; i < _G.n_rows; i++)
		{
			const auto _r1 = _lat->get_rVec(i);
			for (int j = 0; j < _G.n_cols; j++)
			{
				const auto _r2 = _lat->get_rVec(j);
				_ret += _G(i, j) * std::exp(I * (arma::dot(_k1, _r1) - arma::dot(_k2, _r2)));
			}
		}
		return _ret;
	}

	// ---------------------------------------------------------------------------

	template <typename _Mt>
	[[nodiscard]]
	inline arma::Mat<cpx> function_fourier(const _Mt& _G, Lattice* _lat)
	{
		const uint k_num = _lat->get_Ns();
		arma::Mat<cpx> _transform(k_num, k_num, arma::fill::zeros);

		// all the vectors in the inverse space
		for (int k1 = 0; k1 < k_num; k1++)
		{
			const auto _k1			= _lat->get_kVec(k1);
			for (int k2 = 0; k2 < k_num; k2++)
			{
				const auto _k2		= _lat->get_kVec(k2);
				_transform(k1, k2)	= function_fourier(_G, _lat, _k1, _k2);
			}
		}
		return _transform;
	}

	template<typename _Mt>
	[[nodiscard]]
	inline arma::Mat<cpx> function_fourier_translational_inv(const _Mt& _G, Lattice* _lat)
	{
		const uint k_num = _lat->get_Ns();
		arma::Mat<cpx> _transform(k_num, k_num, arma::fill::zeros);

		// all the vectors in the inverse space
		for (int k1 = 0; k1 < k_num; k1++)
		{
			const auto _k1		= _lat->get_kVec(k1);
			_transform(k1, k1)	= function_fourier(_G, _lat, _k1);
		}
		return _transform;
	}

	// ---------------------------------------------------------------------------

	/*
	* @brief Calculate the Fourier transform of the function stored in the matrix G in the real space.
	* The first thing to do is to calculate the function as a diffrence of the positions in the matrix.
	* @param _G - the matrix with the function
	* @param _DFT - the DFT matrix
	* @returns the Fourier transformed function
	*/
	template <typename _Mt>
	[[nodiscard]]
	inline arma::Col<cpx> function_fourier_t(const _Mt& _G, const arma::Mat<cpx>& _DFT)
	{
		arma::Col<cpx> _ret(_G.n_cols, arma::fill::zeros);

		// go through the matrix sites
		for (auto i = 0; i < _G.n_cols; ++i)
		{
			for (auto j = i; j < _G.n_rows; ++j)
			{
				auto _idx	= std::abs(i - j);
				_ret(_idx)	+= _G(i, j);
			}
		}
		// normalize
		_ret /= (1.0 * _G.n_cols);

		return _DFT * _ret;
	}

	// ---------------------------------------------------------------------------

	namespace Spectral
	{
		namespace Noninteracting
		{
			template <typename _Vt>
			[[nodiscard]]
			inline arma::uvec dos(const _Vt& _E, int bins = 100)
			{
				return arma::hist(arma::vec(_E), bins);
			}

			template <typename _Vt>
			[[nodiscard]]
			inline arma::uvec dos(const _Vt& _E, const arma::vec& bins)
			{
				return arma::hist(arma::vec(_E), bins);
			}

			/*
			* @brief Calculate the electronic Density of States. Use gaussian broadening to ensure smoothness of the SFs.
			* @param _E energie to be used for the eDOS
			* @param _Sig broadening of gaussians
			*/
			template <typename _Vt>
			[[nodiscard]]
			inline arma::mat dos_gauss(const _Vt& _E, double _sig = 1e-2)
			{
				auto _N		= int(_E.n_elem / 12);
				arma::mat _dos(_N, 2, arma::fill::zeros);
				_dos.col(0) = arma::linspace(_E(0), _E(_E.n_elem - 1), _N);
				for (const auto _Ein : _E)
				{
					_dos.col(1) += arma::normpdf(_dos.col(0), _Ein, _sig);
				}
				return _dos;
			}

			template <typename _Vt>
			[[nodiscard]]
			inline arma::mat dos_gauss(const arma::Col<double>& _edgs, const _Vt& _E, double _sig = 1e-2)
			{
				auto _N		= _edgs.n_elem;
				arma::Col<double> _dos(_N, arma::fill::zeros);
				for (const auto _Ein : _E)
				{
					_dos += arma::normpdf(_edgs, _Ein, _sig);
				}
				return _dos;
			}
			
			// ---------------------------------------------------------------------------

			/*
			* @brief Calculate the spectral function for noninteracting systems (time resolved Green's function).
			* The spectral function is calculated based on the Hamiltonian matrix and the energy.
			* This is calculated as $$ G(\omega) = \frac{1}{\omega - H + i\eta} $$.
			* @param _omega - the energy
			* @param _H		- the Hamiltonian matrix
			* @param _eta	- the broadening parameter
			* @returns the Green's function matrix
			*/
			template <typename _Mt>
			[[nodiscard]]
			inline arma::Mat<cpx> time_resolved_greens_function(const double _omega, const arma::Mat<_Mt>& _H, double _eta = 1e-1)
			{
				arma::Mat<cpx> _out(_H.n_rows, _H.n_cols, arma::fill::zeros);
				arma::Mat<cpx> _eye(_H.n_rows, _H.n_cols, arma::fill::eye);
				_out		=  _out - _H;
				_out.diag()	+= _omega + I * _eta;

				return arma::solve(_out, _eye, arma::solve_opts::likely_sympd);
				//return arma::inv_sympd(_out);
			}

			template <typename _T>
			[[nodiscard]]
			inline arma::Mat<cpx> time_resolved_greens_function(const double _omega, const arma::SpMat<_T>& _H, double _eta = 1e-1)
			{
				arma::SpMat<cpx> _out(_H.n_rows, _H.n_cols);
				_out		=  _out - _H;
				_out.diag()	+= cpx(_omega, _eta);
				return arma::spsolve(_out, arma::Mat<cpx>(_out.n_rows, _out.n_cols, arma::fill::eye));
			}

			template <typename _Mt>
			[[nodiscard]]
			inline arma::Mat<cpx> time_resolved_greens_function(const double _omega, const arma::Col<double>& _D, const arma::Mat<_Mt>& _U, double _eta = 1e-1)
			{
				arma::Col<cpx> _diaginv = 1.0 / (_omega - I * _eta - _D);
				return algebra::matTimesDiagMat(_U, _diaginv) * _U.t();
			}

			template <typename _T>
			[[nodiscard]]
			inline arma::Mat<cpx> time_resolved_greens_function(const double _omega, const arma::Col<double>& _D, const arma::SpMat<_T>& _U, double _eta = 1e-1)
			{
				arma::Col<cpx> _diaginv = 1.0 / (_omega + I * _eta - _D);
				return algebra::matTimesDiagMat(_U, _diaginv) * _U.t();
			}


			template <typename _T>
			[[nodiscard]]
			inline arma::Mat<cpx> time_resolved_greens_function(const arma::Col<double>& _omegas, const arma::Col<double>& _D, const arma::Mat<_T>& _U, double _eta = 1e-1)
			{
				arma::Mat<cpx> _diaginv(_D.n_elem, _omegas.n_elem, arma::fill::zeros);
				for(int i = 0; i < _omegas.n_elem; ++i)
					_diaginv.col(i) = 1.0 / (_D - _omegas(i) + I * _eta);
				return (_U * _diaginv.t()) * _U.t();
			}

			template <typename _T>
			[[nodiscard]]
			inline arma::Mat<cpx> time_resolved_greens_function(const arma::Col<double>& _omegas, const arma::Col<double>& _D, const arma::SpMat<_T>& _U, double _eta = 1e-1)
			{
				arma::Mat<cpx> _diaginv(_D.n_elem, _omegas.n_elem, arma::fill::zeros);
				for(int i = 0; i < _omegas.n_elem; ++i)
					_diaginv.col(i) = 1.0 / (_D - _omegas(i) + I * _eta);
				return (_U * _diaginv.t()) * _U.t();
			}

			// ---------------------------------------------------------------------------

			/*
			* @brief Calculate the spectral function for noninteracting systems (time resolved Green's function).
			* The spectral function is calculated based on the time resolved Green's function in the k-space.
			* @param _G - the time resolved Green's function
			* @returns the spectral function matrix
			*/
			template <typename _Mt>
			[[nodiscard]]
			inline arma::Mat<double> spectral_function(const _Mt& _G)
			{
				return -(1.0 / PI) * arma::imag(_G);
			}
		}
	}
}

#endif // __SYSTEM_PROPERTIES_H__