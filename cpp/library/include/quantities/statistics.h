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
														const long double _time,
														size_t _threads = 1)
		{
			arma::Col<std::complex<double>> _ret(_eigenstates.n_cols, arma::fill::zeros);

			// go through the eigenstates
#ifndef _DEBUG
#pragma omp parallel for num_threads(_threads) reduction(+: _ret)
#endif // _DEBUG
			for (auto i = 0; i < _eigenstates.n_cols; ++i)
			{
				const auto _exp = std::exp(-I * _time * _eigvals(i));
				_ret += _exp * _overlaps(i) * _eigenstates.col(i);
			}
			return _ret;
		}

	};

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
		for (u64 _k = _min; _k < std::min(_max, _state.size()); ++_k)
		{
			auto _c		= arma::cdot(_newBasis.col(_k), _state);
			auto _v		= std::abs(algebra::conjugate(_c) * _c);
			ent			+= _v * std::log(_v);
		}
		return -ent / std::log(0.48 * _state.size());
	}

};