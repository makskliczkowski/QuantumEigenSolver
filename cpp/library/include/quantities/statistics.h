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
	inline long double eigenlevel_statistics(_Container _energies)
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