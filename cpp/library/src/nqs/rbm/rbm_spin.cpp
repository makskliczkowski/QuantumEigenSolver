#include "../../../include/NQS/rbm_final.hpp"

// ##########################################################################################################################################

// class instantiation for <spins, double, double, double>

template class RBM_S<2, double, double, double>;
// class instantiation for <spins, double, complex, double>
template class RBM_S<2, double, std::complex<double>, double>;
// class instantiation for <spins, complex, double, double>
template class RBM_S<2, std::complex<double>, double, double>;
// class instantiation for <spins, complex, complex, double>
template class RBM_S<2, std::complex<double>, std::complex<double>, double>;

// ##########################################################################################################################################

// !!!!!!!!!!!!!!!!!!! P R O B A B I L I T Y !!!!!!!!!!!!!!!!!!!

// %%%%%%%%%%%%%%%%%%% S I N G L E   F L I P %%%%%%%%%%%%%%%%%%%

/**
* @brief Computes the logarithm of the probability ratio for a given spin flip.
*
* This function calculates the logarithm of the probability ratio when a spin at position `fP` is flipped to the value `fV`.
* The calculation is performed differently based on whether `NQS_ANGLES_UPD` is defined or not.
*
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Type of the return value.
* @tparam _stateType Type of the state.
* @param fP Index of the spin to be flipped.
* @param fV New value of the spin after the flip.
* @return The logarithm of the probability ratio after the spin flip.
*/
template<typename _Ht, typename _T, class _stateType>
inline _T RBM_S<2, _Ht, _T, _stateType>::logPRatio(uint fP, float fV)
{
	// set the first value of b_visible
	_T val 			= 	RBM_SPIN_UPD(fV);

	// use value as the change already
#ifdef NQS_ANGLES_UPD
	//val				=	val * this->bV_(fP) + arma::sum(arma::log(arma::cosh(this->theta_ + val * this->W_.col(fP)) / this->thetaCOSH_));
	// val				=	std::exp(val * this->bV_(fP)) * arma::prod(arma::cosh(this->theta_ + val * this->W_.col(fP)) / this->thetaCOSH_);
	return val * this->bV_(fP) + RBM_ACCU_ARMA(arma::log(arma::cosh(this->theta_ + val * this->W_.col(fP))) - this->thetaCOSH_log_);
#else
	// flip the temporary vector
	this->tmpVec_	=	this->curVec_;
	flip(this->tmpVec_, fP, Operators::_SPIN_RBM);
	// calculate
	return val * this->bV_(fP) + arma::sum(arma::log(this->coshF(this->tmpVec_)) - this->thetaCOSH_log_);
#endif
}
// function instantiations 
template double RBM_S<2, double, double, double>::logPRatio(uint, float);
template std::complex<double> RBM_S<2, double, std::complex<double>, double>::logPRatio(uint, float);
template double RBM_S<2, std::complex<double>, double, double>::logPRatio(uint, float);
template std::complex<double> RBM_S<2, std::complex<double>, std::complex<double>, double>::logPRatio(uint, float);
// ##########################################################################################################################################

/**
* @brief Computes the logarithm of the probability ratio for a Restricted Boltzmann Machine (RBM) with spin configurations.
*
* This function calculates the logarithm of the probability ratio when flipping two spins in the RBM.
*
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Type of the return value.
* @tparam _stateType Type of the state.
* @param f1 Index of the first spin to flip.
* @param f2 Index of the second spin to flip.
* @param v1 Value associated with the first spin.
* @param v2 Value associated with the second spin.
* @return The logarithm of the probability ratio after flipping the specified spins.
*/
template<typename _Ht, typename _T, class _stateType>
inline _T RBM_S<2, _Ht, _T, _stateType>::logPRatio(uint f1, uint f2, float v1, float v2)
{
	// set the first value of b_visible
	_T val1 		= 	RBM_SPIN_UPD(v1);
	_T val2 		= 	RBM_SPIN_UPD(v2);

	// use value as the change already
#ifdef NQS_ANGLES_UPD
	//val				=	val * this->bV_(fP) + arma::sum(arma::log(arma::cosh(this->theta_ + val * this->W_.col(fP)) / this->thetaCOSH_));
	// val				=	std::exp(val * this->bV_(fP)) * arma::prod(arma::cosh(this->theta_ + val * this->W_.col(fP)) / this->thetaCOSH_);
	return val1 * this->bV_(f1) + val2 * this->bV_(f2) + RBM_ACCU_ARMA(arma::log(arma::cosh(this->theta_ + val1 * this->W_.col(f1) + val2 * this->W_.col(f2))) - this->thetaCOSH_log_);
#else
	// flip the temporary vector
	this->tmpVec_	=	this->curVec_;
	flip(this->tmpVec_, f1, Operators::_SPIN_RBM);
	flip(this->tmpVec_, f2, Operators::_SPIN_RBM);
	return val1 * this->bV_(f1) + val2 * this->bV_(f2) + arma::sum(arma::log(this->coshF(this->tmpVec_)) - this->thetaCOSH_log_);
#endif
}
// function instantiations
template double RBM_S<2, double, double, double>::logPRatio(uint, uint, float, float);
template std::complex<double> RBM_S<2, double, std::complex<double>, double>::logPRatio(uint, uint, float, float);
template double RBM_S<2, std::complex<double>, double, double>::logPRatio(uint, uint, float, float);
template std::complex<double> RBM_S<2, std::complex<double>, std::complex<double>, double>::logPRatio(uint, uint, float, float);

// ##########################################################################################################################################

// %%%%%%%%%%%%%%%% M U L T I P L E   F L I P S %%%%%%%%%%%%%%%%

/**
* @brief Computes the logarithm of the probability ratio after a given number of spin flips.
*
* This function calculates the logarithm of the probability ratio (logPRatio) after performing 
* a specified number of spin flips. The calculation is based on the current state of the 
* Restricted Boltzmann Machine (RBM) and the provided flip information.
*
* @tparam _Ht The type of the Hamiltonian.
* @tparam _T The type of the numerical values used in the computation.
* @tparam _stateType The type of the state representation.
* @param nFlips The number of spin flips to be performed.
* @return The logarithm of the probability ratio after the specified number of spin flips.
*/
template<typename _Ht, typename _T, class _stateType>
_T RBM_S<2, _Ht, _T, _stateType>::logPRatio(uint nFlips)
{
	if (nFlips == 1)											// you know what to do after one flip
		return RBM_S<2, _Ht, _T, _stateType>::logPRatio(this->flipPlaces_[0], this->flipVals_[0]);
	else if (nFlips == 2)										// you know what to do after two flips
		return RBM_S<2, _Ht, _T, _stateType>::logPRatio(this->flipPlaces_[0], this->flipPlaces_[1], this->flipVals_[0], this->flipVals_[1]);
	_T val				=	0;									// set the starting point
#if defined NQS_NOT_OMP_MT
	// const auto _thId	=	std::this_thread::get_id();
	arma::Col<_T>& thetaTMP	= this->thetaTMP_;					// thread local storage
#else
	arma::Col<_T>& thetaTMP	= this->thetaTmpCol_;
#endif
	thetaTMP			= 	this->theta_;						// set the temporary angles vector
	
	for (uint i = 0; i < nFlips; ++i)							// iterate through the flips
	{
		auto flipPlace	=	this->flipPlaces_[i];
		auto flipVal	=	this->flipVals_[i];
		_T currVal		=	RBM_SPIN_UPD(flipVal);
		thetaTMP		+=	currVal * this->W_.col(flipPlace);
		val				+=	currVal * this->bV_(flipPlace);
	}
	// use value as the change already
#ifdef NQS_ANGLES_UPD
#	ifdef NQS_NOT_OMP_MT
	return val + RBM_ACCU_ARMA(arma::log(arma::cosh(thetaTMP)) - this->thetaCOSH_log_);
#	else
	// val				=	std::exp(val) * arma::prod(arma::cosh(this->thetaTmp_) / this->thetaCOSH_);
	return val + RBM_ACCU_ARMA(arma::log(arma::cosh(this->thetaTmp_)) - this->thetaCOSH_log_);
#	endif
#else
	return val * this->bV_(fP) + RBM_ACCU_ARMA(arma::log(this->coshF(this->tmpVec_)) - arma::log(this->coshF(this->curVec_)));
#endif
}

// function instantiations
template double RBM_S<2, double, double, double>::logPRatio(uint);
template std::complex<double> RBM_S<2, double, std::complex<double>, double>::logPRatio(uint);
template double RBM_S<2, std::complex<double>, double, double>::logPRatio(uint);
template std::complex<double> RBM_S<2, std::complex<double>, std::complex<double>, double>::logPRatio(uint);

// ##########################################################################################################################################

// %%%%%%%%%%%%%%%%% U S I N G   V E C T O R S %%%%%%%%%%%%%%%%%

/**
* @brief Computes the logarithm of the probability ratio between two states.
*
* This function calculates the logarithm of the probability ratio between two 
* states, `_v1` and `_v2`, using the parameters of the Restricted Boltzmann Machine (RBM).
* This is the ratio of the ansatz probabilities of the two states:
* \f$ \frac{\psi(\mathbf{v}_2)}{\psi(\mathbf{v}_1)} \f$.
*
* @tparam _Ht Hamiltonian type.
* @tparam _T Numeric type for the return value.
* @tparam _stateType Type representing the state.
* @param _v1 The first state.
* @param _v2 The second state.
* @return The logarithm of the probability ratio between the two states.
*/
template<typename _Ht, typename _T, class _stateType>
inline _T RBM_S<2, _Ht, _T, _stateType>::logPRatio(Config_cr_t _v1, Config_cr_t _v2)
{
	return arma::dot(this->bV_, arma::Col<_stateType>(_v2 - _v1)) + RBM_ACCU_ARMA(arma::log(this->coshF(_v2)) - arma::log(this->coshF(_v1)));
}

// function instantiations
template double RBM_S<2, double, double, double>::logPRatio(Config_cr_t, Config_cr_t);
template std::complex<double> RBM_S<2, double, std::complex<double>, double>::logPRatio(Config_cr_t, Config_cr_t);
template double RBM_S<2, std::complex<double>, double, double>::logPRatio(Config_cr_t, Config_cr_t);
template std::complex<double> RBM_S<2, std::complex<double>, std::complex<double>, double>::logPRatio(Config_cr_t, Config_cr_t);

////////////////////////////////////////////////////////////////

/**
* @brief Computes the logarithm of the probability ratio for a given state.
*
* This function calculates the logarithm of the probability ratio between the current state
* and a new state represented by `_v1`. The calculation involves the dot product of the 
* bias vector `bV_` and the difference between `_v1` and the current vector `curVec_`.
* Additionally, it includes the logarithm of the hyperbolic cosine function applied to `_v1`
* and the current vector `curVec_`.
*
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Type of the return value.
* @tparam _stateType Type of the state.
* @param _v1 The new state for which the probability ratio is computed.
* @return The logarithm of the probability ratio.
*/
template<typename _Ht, typename _T, class _stateType>
inline _T RBM_S<2, _Ht, _T, _stateType>::logPRatio(Config_cr_t _v1)
{
#ifdef NQS_ANGLES_UPD
	return arma::dot(this->bV_, _v1 - this->curVec_) + RBM_ACCU_ARMA(arma::log(this->coshF(_v1)) - this->thetaCOSH_log_);
#else
	return arma::dot(this->bV_, _v1 - this->curVec_) + RBM_ACCU_ARMA(arma::log(this->coshF(_v1)) - arma::log(this->coshF(this->curVec_)));
#endif
}

// function instantiations
template double RBM_S<2, double, double, double>::logPRatio(Config_cr_t);
template std::complex<double> RBM_S<2, double, std::complex<double>, double>::logPRatio(Config_cr_t);
template double RBM_S<2, std::complex<double>, double, double>::logPRatio(Config_cr_t);
template std::complex<double> RBM_S<2, std::complex<double>, std::complex<double>, double>::logPRatio(Config_cr_t);

// %%%%%%%%%%%%% U S I N G   I N I T I A L I Z E R %%%%%%%%%%%%%

/**
* @brief Computes the logarithm of the probability ratio for a given set of flips and their corresponding values.
*
* This function calculates the logarithm of the probability ratio for a Restricted Boltzmann Machine (RBM) 
* with spin configurations. It handles multiple flips and updates the internal state accordingly.
*
* @tparam _Ht The type of the Hamiltonian.
* @tparam _T The type used for computations (e.g., double, float).
* @tparam _stateType The type representing the state.
* @param fP An initializer list of integers representing the positions of the flips.
* @param fV An initializer list of doubles representing the values of the flips.
* @return _T The logarithm of the probability ratio.
*
* @note If the number of flips is zero, the function returns 1.0. If there is only one flip, it calls the pRatio function.
* @note The function uses thread-local storage for temporary variables if NQS_NOT_OMP_MT is defined.
* @note The function updates the thetaTMP variable based on the flips and their values.
* @note The final value is computed using either the NQS_ANGLES_UPD or the default method.
*/
template<typename _Ht, typename _T, class _stateType>
_T RBM_S<2, _Ht, _T, _stateType>::logPRatio(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
	const size_t nFlips = fP.size();

	if (nFlips == 1) 
		return RBM_S<2, _Ht, _T, _stateType>::logPRatio(*fP.begin(), *fV.begin());
	else if (nFlips == 2)
		return RBM_S<2, _Ht, _T, _stateType>::logPRatio(*fP.begin(), *(fP.begin() + 1), *fV.begin(), *(fV.begin() + 1));	
	else if (nFlips == 0) 
		return 1.0;

	_T val			= 0.0;
	double currVal	= 0.0;
#if defined NQS_NOT_OMP_MT
	auto& thetaTMP	= this->thetaTMP_;							// thread local storage
#else
	arma::Col<_T>& thetaTMP	= this->thetaTmpCol_;
#endif
	thetaTMP		= this->theta_;
	for (uint i = 0; i < nFlips; ++i)							// iterate through the flips
	{
		auto flipPlace	= *(fP.begin() + i);
		auto flipVal	= *(fV.begin() + i);
		currVal			= RBM_SPIN_UPD(flipVal);				// set the current value
		thetaTMP		+= currVal * this->W_.col(flipPlace);	// !TODO speed this up by not creating thetaTMP
		val				+= currVal * this->bV_(flipPlace);		
	}
#ifdef NQS_ANGLES_UPD
	// val = std::exp(val) * arma::prod(arma::cosh(this->thetaTmp_[thId]) / this->thetaCOSH_);
	return val + RBM_ACCU_ARMA(arma::log(arma::cosh(thetaTMP)) - this->thetaCOSH_log_);
#else
	// val = val * arma::prod(this->coshF(this->tmpVec) / this->coshF(this->curVec));
	return val * this->bV_(fP) + RBM_ACCU_ARMA(arma::log(this->coshF(this->tmpVec_)) - arma::log(this->coshF(this->curVec_)));
#endif
}

// function instantiations
template double RBM_S<2, double, double, double>::logPRatio(std::initializer_list<int>, std::initializer_list<double>);
template std::complex<double> RBM_S<2, double, std::complex<double>, double>::logPRatio(std::initializer_list<int>, std::initializer_list<double>);
template double RBM_S<2, std::complex<double>, double, double>::logPRatio(std::initializer_list<int>, std::initializer_list<double>);
template std::complex<double> RBM_S<2, std::complex<double>, std::complex<double>, double>::logPRatio(std::initializer_list<int>, std::initializer_list<double>);

// ##########################################################################################################################################