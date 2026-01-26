#pragma once
/*******************************
* Definitions for the thermal
* quantities for the models at 
* hand.
* Maksymilian Kliczkowski, 2023
*******************************/

#include "../../../source/src/Include/linalg/diagonalizers.h"

// #####################################################################################################################
// #####################################################################################################################
// ################################################### G E N E R A L ###################################################
// #####################################################################################################################
// #####################################################################################################################

template <typename _T>
class Thermal
{
public:
	// %%%%%%%%%%%%%% G E N E R A L   A V E R A G E %%%%%%%%%%%%%%

	template <template <class _TM = _T> class _MatType, HasMatrixType _Concept = _MatType<_T>, class _T2>
	static std::pair<long double, _T2> thermalAverage(	double _beta, 
														const arma::Mat<_T>& _eigVec, 
														const arma::Col<double>& _eigVal,
														const _MatType<_T2>& _operator);


	// %%%%%%%%%%%%%%%%%%% Q U A N T I T I E S %%%%%%%%%%%%%%%%%%%
	
	static inline double statisticalSum(const arma::Col<double>& _eigval, double _beta)				{ return arma::sum(arma::exp(-_beta * _eigval));	};
	static inline double statisticalSumVal(const arma::Col<double>& _exp, double _beta)				{ return arma::sum(_exp);							};
	static inline arma::Col<double> statisticalExp(const arma::Col<double>& _eigval, double _beta)	{ return arma::exp(-_beta * _eigval);				};
	
	static double heatCapacity(double _beta, const arma::Col<double>& _eigval);

	static inline double specificHeat(_T avH, _T avH2, double _beta)								{ return (avH2 - avH * avH) * _beta;				};
	static inline double magneticSusc(_T avMz, _T avMz2, double _beta)								{ return (avMz - avMz * avMz) * _beta;				};
};

// ######################################################################################################################

/**
* @brief Creates a thermal average in a specific symmetry sector of a given operator. We do not even need the Hilbert
* space. It is assumed that the matrix is correctly created in a given representative (REP) basis.
* @param _beta inverse temperature 1/T
* @param _eigVec vectors that transform to the Hamiltonian basis
* @param _eigVal eigenvalues to be used in the thermal average
* @param _operator operator matrix, already transformed to the correct basis using other methods
*/
template<typename _T>
template <template <class> class _MatType, HasMatrixType _Concept, class _T2>
inline std::pair<long double, _T2> Thermal<_T>::thermalAverage(	double _beta,
																const arma::Mat<_T>& _eigVec, 
																const arma::Col<double>& _eigVal, 
																const _MatType<_T2>& _operator)
{
	long double Z	= 0.0;
	long double A	= 0.0;
	auto _betaExp	= arma::exp(-beta * _eigVal);
	auto _Abase		= Diagonalizer<_T>::changeBase(_eigVec, _operator);
	return std::make_pair(arma::sum(_betaExp), arma::dot(_Abase.diag(), _betaExp);
}

// ######################################################################################################################

template<typename _T>
inline double Thermal<_T>::heatCapacity(double _beta, const arma::Col<double>& _eigval)
{
	auto _exponents = Thermal<_T>::statisticalExp(_eigval, _beta);
	auto _Z			= Thermal<_T>::statisticalSum(_eigval, _beta);
}