#pragma once
/*********************************
* Contains the calculators for the
* density matrix of a quantum state.
*********************************/

#ifndef DENSITY_MATRIX_H
#define DENSITY_MATRIX_H

#ifndef HILBERT_H
#	include "../hilbert.h"
#endif // !HILBERT_H


/**
* @brief Enables one to construct density matrices of quantum states.
*/
namespace DensityMatrix
{
	// ###############################################################

	enum class RHO_METHODS 
	{
		STANDARD,
		STANDARD_CAST,
		SCHMIDT
	};

	// ###############################################################

	/**
	* @brief Calculates the bipartite reduced density matrix of the system via the state mixing
	* @param _s state to construct the density matrix from
	* @param _sizeA subsystem size
	* @param _Ns number of lattice sites
	* @param _Nint number of local fermionic modes
	* @returns the bipartite reduced density matrix
	*/
	template <typename _T>
	inline arma::Mat<_T> redDensMatStandard(const arma::Col<_T>& _s, uint _Ns, uint _sizeA, uint _Nint) 
	{
		// set subsystems size
		auto bitNum		= (uint)std::log2(_Nint);
		auto powA		= bitNum * _sizeA;
		auto powB		= bitNum * (_Ns - _sizeA);
		const u64 dimA	= ULLPOW(powA);
		const u64 dimB	= ULLPOW(powB);
		const u64 Nh	= dimA * dimB;
		// get the masks for the bit manipulation
		// const u64 maskA = (dimA - 1) << powB;
		const u64 maskB = dimB - 1;

		arma::Mat<_T> rho(dimA, dimA, arma::fill::zeros);

		//u64 _startingM	= 0;
		// loop over configurational basis
		for (u64 n = 0; n < Nh; n++) 
		{
			// u64 counter = 0;
			// find index of state in A, with same B-sides (by dividing the last bits are discarded). 
			// The same is done by shifting the bits to the right.
			// u64 idx	= n / dimB; bit shift should be faster
			u64 idx		= n >> powB;
			_T _sn_c    = algebra::conjugate(_s(n));
			// pick out state with same B side (last L-A_size bits) (should be faster than modulo)
			//for (u64 m = (n & maskA); m < Nh; m += dimB)
			////for (u64 m = n % dimB; m < Nh; m += dimB) 
			////for (u64 m = _startingM; m < Nh; m += dimB)
			//{
			//	rho(idx, counter) += _sn_c * _s(m);
			//	// increase counter to move along reduced basis
			//	counter++;
			//}
			rho.row(idx) += _sn_c * _s.elem(arma::regspace<arma::uvec>(n & maskB, dimB, Nh - 1));
		}
		return rho;
	};

	// ###############################################################

	/**
	* @brief Calculates the bipartite reduced density matrix of the system via the state mixing. Knowing the mapping with global symmetry.
	* @param _s state to construct the density matrix from
	* @param _sizeA subsystem size
	* @param _hilb used Hilbert space - contains the mapping
	* @returns the bipartite reduced density matrix
	*/
	template <typename _T>
	inline arma::Mat<_T> redDensMatStandard(const arma::Col<_T>& _s, uint _sizeA, const Hilbert::HilbertSpace<_T>& _hilb) {
		// set subsystems size
		uint Ns			= _hilb.getLatticeSize();
		uint Nint		= _hilb.getLocalHilbertSize() * _hilb.getNum();
		uint bitNum		= (uint)std::log2(Nint);
		const u64 dimA	= ULLPOW(bitNum * _sizeA);
		const u64 dimB	= ULLPOW((Ns - bitNum * _sizeA));
		const u64 Nh	= dimA * dimB;
		if (!_hilb.checkGSym())	return redDensMatStandard<_T>(_s, Ns, _sizeA, Nint);

		auto map		= _hilb.getFullMap();
		const u64 N		= map.size();
		// otherwise find in mapping
		auto find_index = [&](u64 _idx) 
			{ 
				return binarySearch(map, 0, Nh - 1, _idx); 
			};

		arma::Mat<_T> rho(dimA, dimA, arma::fill::zeros);
		for (u64 n = 0; n < N; n++) {
			// loop over configurational basis
			u64 ctr = 0;
			u64 true_n = map[n];
			// pick out state with same B side (last L-A_size bits)
			u64 idx = true_n / dimB;
			for (u64 m = true_n % dimB; m < Nh; m += dimB) {
				u64 j	= find_index(m);
				if (j >= 0)
					rho(idx, ctr) += algebra::conjugate(_s(n)) * _s(j);
				// increase counter to move along reduced basis
				ctr++;
			}
		}
		return rho;
	};

	// ###############################################################

	/**
	* @brief Using reshape method to calculate the reduced density matrix
	* @param _s state to construct the density matrix from
	* @param _sizeA subsystem size
	* @param _hilb used Hilbert space - contains the mapping
	* @returns the bipartite reduced density matrix
	*/
	template<typename _T>
	inline arma::Mat<_T> redDensMatSchmidt(const arma::Col<_T>& _s, uint _sizeA, const Hilbert::HilbertSpace<_T>& _hilb) 
	{
		// set subsystems size
		uint Ns			= _hilb.getLatticeSize();
		uint bitNum		= (uint)std::log2(_hilb.getLocalHilbertSize());
		const u64 dimA	= ULLPOW(bitNum * _sizeA);
		const u64 dimB	= ULLPOW(bitNum * (Ns - _sizeA));
		return arma::reshape(_s, dimA, dimB);
	}

	// ##############################################################################################################################

	// ##############################################################################################################################

	/**
	* @brief Using reshape method to calculate the reduced density matrix
	* @param _s state to construct the density matrix from
	* @param _sizeA subsystem size
	* @param _Ns number of lattice sites
	* @param _locHilbert local Hilbert space size
	* @returns the bipartite reduced density matrix
	*/
	template<typename _TV>
	inline arma::Mat<_TV> schmidt(const arma::Col<_TV>& _s, uint _sizeA, size_t _Ns, uint _locHilbert = 2) 
	{
		// set subsystems size
		uint bitNum		= (uint)std::log2(_locHilbert);
		const u64 dimA	= ULLPOW(bitNum * _sizeA);
		const u64 dimB	= ULLPOW(bitNum * (_Ns - _sizeA));
		return arma::reshape(_s, dimA, dimB);
	}

	// ###############################################################

	/**
	* @brief Using reshape method to calculate the reduced density matrix - Schmidt decomposition for the state
	* @param _s state to construct the density matrix from
	* @param _sizeA subsystem size
	* @param _size total size of the system
	* @param _mask mask for the reshaping
	* @param _locHilbertSize local Hilbert space size
	*/
	template<typename _TV, typename _T>
	inline typename std::enable_if<std::is_arithmetic<_T>::value, arma::Mat<_TV>>::type
	schmidt(const arma::Col<_TV>& _s, 
			size_t _sizeA, 
			size_t _size, 
			const _T& _maskA, 
			size_t _locHilbert = 2)
	{
		uint bitNum		= (uint)std::log2(_locHilbert);
		const u64 dA	= ULLPOW(bitNum * _sizeA);
		const u64 dB	= ULLPOW(bitNum * (_size - _sizeA));

		// create second part of the mask
		const _T _maskB = Binary::flipAll(_maskA, _size);

		// create the reduced density matrix
		arma::Mat<_TV> _psi(dA, dB, arma::fill::zeros);

		// loop over the state
		for(u64 _st = 0; _st < _s.size(); ++_st)
		{
			// get the index of the state in the reduced basis
			u64 _idxA	= Binary::extract(_st, _maskA);
			u64 _idxB	= Binary::extract(_st, _maskB);
			_psi(_idxA, _idxB) += _s(_st);
		}
		// return the new Schmidt decomposed matrix
		return _psi;
	}

	// ##############################################################################################################################
	
	// ##############################################################################################################################

	/**
	* @brief Calculates the reduced density matrix with one of the methods
	* @param _s state to construct the density matrix from
	* @param _sizeA subsystem size
	* @param _hilb used Hilbert space - contains the mapping
	* @param _ch method choice
	* @returns the bipartite reduced density matrix
	*/
	template <typename _T>
	inline arma::Mat<_T> redDensMat(const arma::Col<_T>& _s,
									uint _sizeA,
									Hilbert::HilbertSpace<_T>& _hilb,
									RHO_METHODS _ch = RHO_METHODS::SCHMIDT)
	{
		switch (_ch) {
		case RHO_METHODS::STANDARD:
			return redDensMatStandard<_T>(_s, _sizeA, _hilb);
			break;
		case RHO_METHODS::STANDARD_CAST:
			return redDensMatStandard<_T>(_hilb.castToFull(_s), _hilb.getLatticeSize(), _sizeA, _hilb.getNum());
			break;
		case RHO_METHODS::SCHMIDT:
			return redDensMatSchmidt<_T>(_hilb.castToFull(_s), _sizeA, _hilb);
			break;
		default:
			return redDensMatSchmidt<_T>(_hilb.castToFull(_s), _sizeA, _hilb);
			break;
		}
	}

	// ###############################################################

	template <typename _T>
	inline arma::Mat<_T> redDensMat(const arma::Col<_T>& _s, 
									uint _sizeA, size_t _Ns, 
									RHO_METHODS _ch = RHO_METHODS::SCHMIDT, 
									uint _locHilbert = 2)
	{
		switch (_ch) 
		{
		case RHO_METHODS::STANDARD:
			return arma::Mat<_T>();
			break;
		case RHO_METHODS::STANDARD_CAST:
			return arma::Mat<_T>();
			break;
		case RHO_METHODS::SCHMIDT:
			return schmidt(_s, _sizeA, _Ns, _locHilbert);
			break;
		default:
			return schmidt(_s, _sizeA, _Ns, _locHilbert);
			break;
		}
	}

	// ###############################################################

	template<typename _TV, typename _T>
	inline typename std::enable_if<std::is_arithmetic<_T>::value, arma::Mat<_TV>>::type
	redDensMat(const arma::Col<_TV>& _s, 
				size_t _sizeA, 
				size_t _size, 
				const _T& _maskA, 
				RHO_METHODS _ch = RHO_METHODS::SCHMIDT, 
				size_t _locHilbert = 2)
	{
		switch (_ch) 
		{
		case RHO_METHODS::STANDARD:
			return arma::Mat<_TV>();
			break;
		case RHO_METHODS::STANDARD_CAST:
			return arma::Mat<_TV>();
			break;
		case RHO_METHODS::SCHMIDT:
			return schmidt(_s, _sizeA, _size, _maskA, _locHilbert);
			break;
		default:
			return schmidt(_s, _sizeA, _size, _maskA, _locHilbert);
			break;
		}
	}
	
	// ##############################################################################################################################

	namespace Values
	{	
		template <typename _T>
		inline arma::vec redDensMat_v(const arma::Mat<_T>& _rho, 
									  RHO_METHODS _ch = RHO_METHODS::SCHMIDT)
		{
			if (_ch == RHO_METHODS::SCHMIDT)
				return arma::square(arma::svd(_rho));
			else
				return arma::eig_sym(_rho);
		}


		template <typename _T>
		inline arma::vec redDensMat_v(const arma::Col<_T>& _s, 
										uint _sizeA, size_t _Ns, 
										RHO_METHODS _ch = RHO_METHODS::SCHMIDT, 
										uint _locHilbert = 2)
		{
			auto _rho = redDensMat<_T>(_s, _sizeA, _Ns, _ch, _locHilbert);
			return redDensMat_v(_rho, _ch);
		}

		// ##############################################################################################################################

		template <typename _T>
		inline arma::vec redDensMat_v(const arma::Col<_T>& _s, 
									uint _sizeA,
									Hilbert::HilbertSpace<_T>& _hilb,
									RHO_METHODS _ch = RHO_METHODS::SCHMIDT)
		{
			auto _rho = redDensMat<_T>(_s, _sizeA, _hilb, _ch);
			return redDensMat_v(_rho, _ch);
		}

		// ##############################################################################################################################

		/**
		* @brief Calculates the reduced density matrix with one of the methods and returns the eigenvalues
		* @param _s state to construct the density matrix from
		* @param _sizeA subsystem size
		* @param _Ns number of lattice sites
		* @param _ch method choice
		* @param _locHilbert local Hilbert space size
		* @returns the eigenvalues of the reduced density matrix
		*/
		template<typename _TV, typename _T>
		inline typename std::enable_if<std::is_arithmetic<_T>::value, arma::vec>::type
		redDensMat_v(const arma::Col<_TV>& _s, 
				size_t _sizeA, 
				size_t _size, 
				const _T& _maskA, 
				RHO_METHODS _ch = RHO_METHODS::SCHMIDT, 
				size_t _locHilbert = 2)
		{
			auto _rho = redDensMat<_TV, _T>(_s, _sizeA, _size, _maskA, _ch, _locHilbert);
			return redDensMat_v(_rho, _ch);
		}
	};
};

#endif