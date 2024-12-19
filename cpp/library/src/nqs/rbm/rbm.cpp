#include "../../../include/NQS/rbm_final.hpp"

// ##########################################################################################################################################

// template instantiation of the RBM class for <spins, double, double, double>
// 1) all real types - double up to 4 spin modes
template class RBM<2u, double, double, double>;
template class RBM<3u, double, double, double>;
template class RBM<4u, double, double, double>;
// 2) all complex types - cpx up to 4 spin modes
template class RBM<2u, cpx, cpx, double>;
template class RBM<3u, cpx, cpx, double>;
template class RBM<4u, cpx, cpx, double>;
// 3) complex hamiltonian with real NQS - cpx up to 4 spin modes
template class RBM<2u, cpx, double, double>;
template class RBM<3u, cpx, double, double>;
template class RBM<4u, cpx, double, double>;
// 4) real hamiltonian with complex NQS - double up to 4 spin modes
template class RBM<2u, double, cpx, double>;
template class RBM<3u, double, cpx, double>;
template class RBM<4u, double, cpx, double>;

// ##########################################################################################################################################

// ############################################################## A N S A T Z ###############################################################

// ##########################################################################################################################################


/**
* @brief Computes the ansatz function for the given input state.
*
* This function calculates the ansatz value for a given input state using the 
* Restricted Boltzmann Machine (RBM) model. The ansatz is computed as the 
* exponential of the dot product between the bias vector and the input state, 
* multiplied by the product of the hyperbolic cosine function applied to the 
* input state.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for the return value.
* @tparam _stateType The type of the input state.
* @param _in The input state for which the ansatz is to be computed.
* @return The computed ansatz value.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
_T RBM<_spinModes, _Ht, _T, _stateType>::ansatz(Config_cr_t _in) const
{
	return algebra::cast<_T>(std::exp(arma::dot(this->bV_, _in)) * arma::prod(this->coshF(_in)));
};

// template instantiation of the function above
RBM_INST_CMB(double, double, ansatz, double, (const arma::Col<double>&), const);
RBM_INST_CMB(cpx, cpx, ansatz, cpx, (const arma::Col<double>&), const);
RBM_INST_CMB(double, cpx, ansatz, cpx, (const arma::Col<double>&), const);
RBM_INST_CMB(cpx, double, ansatz, double, (const arma::Col<double>&), const);

// ##########################################################################################################################################

/**
* @brief Computes the logarithm of the ansatz function for the given input state.
*
* This function calculates the logarithm of the ansatz function using the 
* Restricted Boltzmann Machine (RBM) model. It takes an input state and 
* computes the dot product of the visible bias vector with the input state, 
* and adds the sum of the logarithm of the hyperbolic cosine of the input state.
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the return value.
* @tparam _stateType Type of the input state.
* @param _in The input state for which the ansatz logarithm is computed.
* @return The logarithm of the ansatz function for the given input state.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T RBM<_spinModes, _Ht, _T, _stateType>::ansatzlog(Config_cr_t _in) const 
{
	return algebra::cast<_T>(arma::dot(this->bV_, _in) + arma::sum(arma::log(this->coshF(_in))));
}

// template instantiation of the function above
RBM_INST_CMB(double, double, ansatzlog, double, (const arma::Col<double>&), const);
RBM_INST_CMB(cpx, cpx, ansatzlog, cpx, (const arma::Col<double>&), const);
RBM_INST_CMB(double, cpx, ansatzlog, cpx, (const arma::Col<double>&), const);
RBM_INST_CMB(cpx, double, ansatzlog, double, (const arma::Col<double>&), const);

// ##########################################################################################################################################

/**
* @brief Calculates the logarithm of the ratio of two RBM states - used for calculating the excited states (_other->ansatz / this->ansatz).
* 
* This function computes the logarithm of the ratio of the ansatz values of two Restricted Boltzmann Machines (RBMs) for a given input state vector.
* It is typically used to evaluate the excited states by comparing the ansatz of the current RBM instance with another RBM instance.
* 
* @param _in The input state vector for which the ratio is to be calculated.
* @param _other Pointer to the other NQS (Neural Quantum State) instance to calculate the ratio with.
* @return The logarithm of the ratio of the two states (other / this) for the given state vector _in.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T RBM<_spinModes, _Ht, _T, _stateType>::ansatz_ratiolog(Config_cr_t _in, NQS<_spinModes, _Ht, _T, _stateType>* _other) const
{
	auto _rbm_other = dynamic_cast<RBM<_spinModes, _Ht, _T, _stateType>*>(_other);
	_T log_ratio = (arma::dot(_rbm_other->bV_ - this->bV_, _in)) + 
					arma::sum(arma::log(_rbm_other->coshF(_in))) - arma::sum(arma::log(this->coshF(_in)));
	return log_ratio;
}

// template instantiation of the function above
template double RBM<2u, double, double, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<2u, double, double, double>*) const;
template std::complex<double> RBM<2u, double, std::complex<double>, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<2u, double, std::complex<double>, double>*) const;
template double RBM<2u, std::complex<double>, double, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<2u, std::complex<double>, double, double>*) const;
template std::complex<double> RBM<2u, std::complex<double>, std::complex<double>, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<2u, std::complex<double>, std::complex<double>, double>*) const;
// 3 
template double RBM<3u, double, double, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<3u, double, double, double>*) const;
template std::complex<double> RBM<3u, double, std::complex<double>, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<3u, double, std::complex<double>, double>*) const;
template double RBM<3u, std::complex<double>, double, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<3u, std::complex<double>, double, double>*) const;
template std::complex<double> RBM<3u, std::complex<double>, std::complex<double>, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<3u, std::complex<double>, std::complex<double>, double>*) const;
// 4
template double RBM<4u, double, double, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<4u, double, double, double>*) const;
template std::complex<double> RBM<4u, double, std::complex<double>, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<4u, double, std::complex<double>, double>*) const;
template double RBM<4u, std::complex<double>, double, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<4u, std::complex<double>, double, double>*) const;
template std::complex<double> RBM<4u, std::complex<double>, std::complex<double>, double>::ansatz_ratiolog(const arma::Col<double>&, NQS<4u, std::complex<double>, std::complex<double>, double>*) const;

// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

// ##########################################################################################################################################

/**
* @brief Computes the gradient of the Restricted Boltzmann Machine (RBM) with respect to its parameters.
*
* This function calculates the gradient of the RBM parameters given the input state vector `_v` and the 
* index `_plc` which specifies the position in the derivatives matrix to update. The gradient is computed 
* by updating the visible layer, hidden layer, and weights derivatives.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type used for computations.
* @tparam _stateType The type of the state vector.
* @param _v The input state vector.
* @param _plc The index specifying the position in the derivatives matrix to update.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM<_spinModes, _Ht, _T, _stateType>::grad(Config_cr_t _v, uint _plc)
{
	// get the subviews
	auto _currDerivative	= this->derivatives_.row(_plc);
	auto _hiddDerivative	= _currDerivative.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1);
	auto _weightsDerivative = _currDerivative.subvec(this->info_p_.nVis_ + this->nHid_, this->rbmSize_ - 1);

#ifndef NQS_ANGLES_UPD
	this->setTheta(_v);                         // update the angles if it is necessary
#endif

	// calculate the flattened part
	_currDerivative.head(this->info_p_.nVis_) 	= arma::conv_to<arma::Row<_T>>::from(_v);
	_hiddDerivative								= arma::tanh(this->theta_).as_row();

#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
	for (int j = 0; j < this->info_p_.nVis_; ++j)
		_weightsDerivative.subvec(j * this->nHid_, (j + 1) * this->nHid_ - 1) = _v(j) * _hiddDerivative;
}
// template instantiation of the function above
RBM_INST_CMB(double, double, grad, void, (const arma::Col<double>&, uint), );
RBM_INST_CMB(cpx, cpx, grad, void, (const arma::Col<double>&, uint), );
RBM_INST_CMB(double, cpx, grad, void, (const arma::Col<double>&, uint), );
RBM_INST_CMB(cpx, double, grad, void, (const arma::Col<double>&, uint), );

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM<_spinModes, _Ht, _T, _stateType>::setWeights()
{
	this->bV_	= this->Weights_.subvec(0, this->info_p_.nVis_ - 1);
	this->bH_	= this->Weights_.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1);
	this->W_	= arma::reshape(this->Weights_.subvec(this->info_p_.nVis_ + this->nHid_, this->rbmSize_ - 1), this->W_.n_rows, this->W_.n_cols);
}
// template instantiation of the function above
RBM_INST_CMB_ALL(setWeights, void, (), );

/**
* @brief Updates the weights in the system according to a given gradient
* @warning uses forces vector (member of NQS : dF_) to update the gradients - preallocation for optimization
* @note the function is called after the gradient is calculated and inlined to the optimization process
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM<_spinModes, _Ht, _T, _stateType>::updateWeights()
{
	NQS_S<_spinModes, _Ht, _T, _stateType>::updateWeights();
	this->setWeights();
}
// template instantiation of the function above
RBM_INST_CMB_ALL(updateWeights, void, (), ); 

// ##########################################################################################################################################

// ############################################################ U P D A T E R S #############################################################

// ##########################################################################################################################################

#ifdef NQS_ANGLES_UPD

/**
* @brief Update angles with the flipped spin (spins)
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM<_spinModes, _Ht, _T, _stateType>::update(uint nFlips)
{
	for (uint i = 0; i < nFlips; ++i)
	{
#ifdef SPIN
		this->theta_	-=	(2.0 * this->flipVals_[i]) * this->W_.col(this->flipPlaces_[i]);
#else
		this->theta_	+=	(1.0 - 2.0 * this->flipVals_[i]) * this->W.col(flipPlaces_[i]);
#endif
	}
	this->thetaCOSH_	=	this->coshF();
	this->thetaCOSH_log_= 	arma::log(this->thetaCOSH_);

	NQS<_spinModes, _Ht, _T, _stateType>::update(nFlips);
}

///////////////////////////////////////////////////////////////////////

/**
* @brief Update angles with the flipped spin (spins)
* @param vector after the flips have been done
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM<_spinModes, _Ht, _T, _stateType>::update(Config_cr_t v, uint nFlips)
{
	for (uint i = 0; i < nFlips; ++i)
	{
		const auto fP = this->flipPlaces_[i];
#ifdef SPIN
		this->theta_ -= (2.0 * v(fP)) * this->W_.col(fP);
#else
		this->theta_ += (1.0 - 2.0 * v(fP)) * this->W_.col(fP);
#endif
	}
	this->thetaCOSH_ = this->coshF();
	this->thetaCOSH_log_ = arma::log(this->thetaCOSH_);
	NQS<_spinModes, _Ht, _T, _stateType>::update(v, nFlips);
}
#endif

// ##########################################################################################################################################

