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


/*
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

	/*
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
		const u64 dimA	= ULLPOW(bitNum * _sizeA);
		const u64 dimB	= ULLPOW((_Ns - bitNum * _sizeA));
		const u64 Nh	= dimA * dimB;

		arma::Mat<_T> rho(dimA, dimA, arma::fill::zeros);

		//u64 _startingM	= 0;
		// loop over configurational basis
		for (u64 n = 0; n < Nh; n++) 
		{
			u64 counter = 0;
			// find index of state with same B-side (by dividing the last bits are discarded)
			u64 idx		= n / dimB;
			// pick out state with same B side (last L-A_size bits)
			for (u64 m = n % dimB; m < Nh; m += dimB) 
			//for (u64 m = _startingM; m < Nh; m += dimB)
			{
				rho(idx, counter) += algebra::conjugate(_s(n)) * _s(m);
				// increase counter to move along reduced basis
				counter++;
			}
			//if(++_startingM >= dimB) _startingM = 0;
		}
		return rho;
	};

	// ###############################################################

	/*
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

	/*
	* @brief Using reshape method to calculate the reduced density matrix
	* @param _s state to construct the density matrix from
	* @param _sizeA subsystem size
	* @param _hilb used Hilbert space - contains the mapping
	* @returns the bipartite reduced density matrix
	*/
	template<typename _T>
	inline arma::Mat<_T> redDensMatSchmidt(const arma::Col<_T>& _s, uint _sizeA, const Hilbert::HilbertSpace<_T>& _hilb) {
		// set subsystems size
		uint Ns			= _hilb.getLatticeSize();
		uint bitNum		= (uint)std::log2(_hilb.getLocalHilbertSize());
		const u64 dimA	= ULLPOW(bitNum * _sizeA);
		const u64 dimB	= ULLPOW(bitNum * (Ns - _sizeA));
		return arma::reshape(_s, dimA, dimB);
	}
			
	// ###############################################################

	/*
	* @brief Calculates the reduced density matrix with one of the methods
	* @param _s state to construct the density matrix from
	* @param _sizeA subsystem size
	* @param _hilb used Hilbert space - contains the mapping
	* @param _ch method choice
	* @returns the bipartite reduced density matrix
	*/
	template <typename _T>
	inline arma::Mat<_T> redDensMat(const arma::Col<_T>& _s, uint _sizeA,
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
};

#endif