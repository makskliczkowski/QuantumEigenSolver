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

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM<_spinModes, _Ht, _T, _stateType>::RBM(const RBM<_spinModes, _Ht, _T, _stateType>& _n)
	: NQS_S<_spinModes, _Ht, _T, _stateType>(_n)
{
	this->W_ 				= _n.W_;
	this->bV_ 				= _n.bV_;
	this->bH_ 				= _n.bH_;
	this->theta_ 			= _n.theta_;
	this->thetaCOSH_ 		= _n.thetaCOSH_;
	this->thetaTmpCol_ 		= _n.thetaTmpCol_;
	this->thetaCOSH_log_	= _n.thetaCOSH_log_;
	this->nHid_ 			= _n.nHid_;
	this->beta_ 			= _n.beta_;
	// sizes
	this->rbmSize_ 			= _n.rbmSize_;
	this->info_p_.fullSize_ = _n.info_p_.fullSize_;
}
// template instantiation of the function above
template RBM<2u, double, double, double>::RBM(const RBM<2u, double, double, double>&);
template RBM<3u, double, double, double>::RBM(const RBM<3u, double, double, double>&);
template RBM<4u, double, double, double>::RBM(const RBM<4u, double, double, double>&);
template RBM<2u, cpx, cpx, double>::RBM(const RBM<2u, cpx, cpx, double>&);
template RBM<3u, cpx, cpx, double>::RBM(const RBM<3u, cpx, cpx, double>&);
template RBM<4u, cpx, cpx, double>::RBM(const RBM<4u, cpx, cpx, double>&);
template RBM<2u, cpx, double, double>::RBM(const RBM<2u, cpx, double, double>&);
template RBM<3u, cpx, double, double>::RBM(const RBM<3u, cpx, double, double>&);
template RBM<4u, cpx, double, double>::RBM(const RBM<4u, cpx, double, double>&);
template RBM<2u, double, cpx, double>::RBM(const RBM<2u, double, cpx, double>&);
template RBM<3u, double, cpx, double>::RBM(const RBM<3u, double, cpx, double>&);
template RBM<4u, double, cpx, double>::RBM(const RBM<4u, double, cpx, double>&);

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM<_spinModes, _Ht, _T, _stateType>::RBM(RBM<_spinModes, _Ht, _T, _stateType>&& _n)
	: NQS_S<_spinModes, _Ht, _T, _stateType>(std::move(_n))
{
	this->W_ 				= std::move(_n.W_);
	this->bV_ 				= std::move(_n.bV_);
	this->bH_ 				= std::move(_n.bH_);
	this->theta_ 			= std::move(_n.theta_);
	this->thetaCOSH_ 		= std::move(_n.thetaCOSH_);
	this->thetaTmpCol_ 		= std::move(_n.thetaTmpCol_);
	this->thetaCOSH_log_	= std::move(_n.thetaCOSH_log_);
	this->nHid_ 			= _n.nHid_;
	this->beta_ 			= std::move(_n.beta_);
	// sizes
	this->rbmSize_ 			= _n.rbmSize_;
	this->info_p_.fullSize_ = _n.info_p_.fullSize_;
}
// template instantiation of the function above
template RBM<2u, double, double, double>::RBM(RBM<2u, double, double, double>&&);
template RBM<3u, double, double, double>::RBM(RBM<3u, double, double, double>&&);
template RBM<4u, double, double, double>::RBM(RBM<4u, double, double, double>&&);
template RBM<2u, cpx, cpx, double>::RBM(RBM<2u, cpx, cpx, double>&&);
template RBM<3u, cpx, cpx, double>::RBM(RBM<3u, cpx, cpx, double>&&);
template RBM<4u, cpx, cpx, double>::RBM(RBM<4u, cpx, cpx, double>&&);
template RBM<2u, cpx, double, double>::RBM(RBM<2u, cpx, double, double>&&);
template RBM<3u, cpx, double, double>::RBM(RBM<3u, cpx, double, double>&&);
template RBM<4u, cpx, double, double>::RBM(RBM<4u, cpx, double, double>&&);
template RBM<2u, double, cpx, double>::RBM(RBM<2u, double, cpx, double>&&);
template RBM<3u, double, cpx, double>::RBM(RBM<3u, double, cpx, double>&&);
template RBM<4u, double, cpx, double>::RBM(RBM<4u, double, cpx, double>&&);

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM<_spinModes, _Ht, _T, _stateType>& RBM<_spinModes, _Ht, _T, _stateType>::operator=(RBM<_spinModes, _Ht, _T, _stateType>& _other)
{
	if (this != &_other)
	{
		NQS_S<_spinModes, _Ht, _T, _stateType>::operator=(_other);
		this->W_ = _other.W_;
		this->bV_ = _other.bV_;
		this->bH_ = _other.bH_;
		this->theta_ = _other.theta_;
		this->thetaCOSH_ = _other.thetaCOSH_;
		this->thetaTmpCol_ = _other.thetaTmpCol_;
		this->thetaCOSH_log_ = _other.thetaCOSH_log_;
		this->nHid_ = _other.nHid_;
		this->beta_ = _other.beta_;
			// sizes
		this->rbmSize_ 			= _other.rbmSize_;
		this->info_p_.fullSize_ = _other.info_p_.fullSize_;
	}
	return *this;
}

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM<_spinModes, _Ht, _T, _stateType>& RBM<_spinModes, _Ht, _T, _stateType>::operator=(RBM<_spinModes, _Ht, _T, _stateType>&& _other)
{
	if (this != &_other)
	{
		NQS_S<_spinModes, _Ht, _T, _stateType>::operator=(std::move(_other));
		this->W_ = std::move(_other.W_);
		this->bV_ = std::move(_other.bV_);
		this->bH_ = std::move(_other.bH_);
		this->theta_ = std::move(_other.theta_);
		this->thetaCOSH_ = std::move(_other.thetaCOSH_);
		this->thetaTmpCol_ = std::move(_other.thetaTmpCol_);
		this->thetaCOSH_log_ = std::move(_other.thetaCOSH_log_);
		this->nHid_ = _other.nHid_;
		this->beta_ = std::move(_other.beta_);
		// sizes
		this->rbmSize_ 			= _other.rbmSize_;
		this->info_p_.fullSize_ = _other.info_p_.fullSize_;
	}
	return *this;
}

// template instantiation of the functions above
template RBM<2u, double, double, double>& RBM<2u, double, double, double>::operator=(RBM<2u, double, double, double>&);
template RBM<3u, double, double, double>& RBM<3u, double, double, double>::operator=(RBM<3u, double, double, double>&);
template RBM<4u, double, double, double>& RBM<4u, double, double, double>::operator=(RBM<4u, double, double, double>&);
template RBM<2u, cpx, cpx, double>& RBM<2u, cpx, cpx, double>::operator=(RBM<2u, cpx, cpx, double>&);
template RBM<3u, cpx, cpx, double>& RBM<3u, cpx, cpx, double>::operator=(RBM<3u, cpx, cpx, double>&);
template RBM<4u, cpx, cpx, double>& RBM<4u, cpx, cpx, double>::operator=(RBM<4u, cpx, cpx, double>&);
template RBM<2u, cpx, double, double>& RBM<2u, cpx, double, double>::operator=(RBM<2u, cpx, double, double>&);
template RBM<3u, cpx, double, double>& RBM<3u, cpx, double, double>::operator=(RBM<3u, cpx, double, double>&);
template RBM<4u, cpx, double, double>& RBM<4u, cpx, double, double>::operator=(RBM<4u, cpx, double, double>&);
template RBM<2u, double, cpx, double>& RBM<2u, double, cpx, double>::operator=(RBM<2u, double, cpx, double>&);
template RBM<3u, double, cpx, double>& RBM<3u, double, cpx, double>::operator=(RBM<3u, double, cpx, double>&);
template RBM<4u, double, cpx, double>& RBM<4u, double, cpx, double>::operator=(RBM<4u, double, cpx, double>&);

template RBM<2u, double, double, double>& RBM<2u, double, double, double>::operator=(RBM<2u, double, double, double>&&);
template RBM<3u, double, double, double>& RBM<3u, double, double, double>::operator=(RBM<3u, double, double, double>&&);
template RBM<4u, double, double, double>& RBM<4u, double, double, double>::operator=(RBM<4u, double, double, double>&&);
template RBM<2u, cpx, cpx, double>& RBM<2u, cpx, cpx, double>::operator=(RBM<2u, cpx, cpx, double>&&);
template RBM<3u, cpx, cpx, double>& RBM<3u, cpx, cpx, double>::operator=(RBM<3u, cpx, cpx, double>&&);
template RBM<4u, cpx, cpx, double>& RBM<4u, cpx, cpx, double>::operator=(RBM<4u, cpx, cpx, double>&&);
template RBM<2u, cpx, double, double>& RBM<2u, cpx, double, double>::operator=(RBM<2u, cpx, double, double>&&);
template RBM<3u, cpx, double, double>& RBM<3u, cpx, double, double>::operator=(RBM<3u, cpx, double, double>&&);
template RBM<4u, cpx, double, double>& RBM<4u, cpx, double, double>::operator=(RBM<4u, cpx, double, double>&&);
template RBM<2u, double, cpx, double>& RBM<2u, double, cpx, double>::operator=(RBM<2u, double, cpx, double>&&);
template RBM<3u, double, cpx, double>& RBM<3u, double, cpx, double>::operator=(RBM<3u, double, cpx, double>&&);
template RBM<4u, double, cpx, double>& RBM<4u, double, cpx, double>::operator=(RBM<4u, double, cpx, double>&&);

// ##########################################################################################################################################

/**
* @brief Clones the properties of another RBM object into this one.
*
* This function attempts to dynamically cast the provided object to an RBM object
* with the same template parameters. If successful, it copies the internal properties
* from the other RBM object to this one.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The type of the parameters.
* @tparam _stateType The type of the state.
* @param _other A shared pointer to another Monte Carlo object to clone from.
*
* @throws std::exception If there is an error during the cloning process.
* @throws std::bad_cast If the dynamic cast fails.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM<_spinModes, _Ht, _T, _stateType>::clone(MC_t_p _other)
{
	try
	{
		auto _n = std::dynamic_pointer_cast<RBM<_spinModes, _Ht, _T, _stateType>>(_other);
		if (_n)
		{
			this->W_ 			= _n->W_;
			this->bV_ 			= _n->bV_;
			this->bH_ 			= _n->bH_;
			this->theta_ 		= _n->theta_;
			this->thetaCOSH_ 	= _n->thetaCOSH_;
			this->thetaTmpCol_ 	= _n->thetaTmpCol_;
			this->thetaCOSH_log_= _n->thetaCOSH_log_;
			this->nHid_			= _n->nHid_;
		}
	}
	catch (std::bad_cast & e)
	{
		std::cerr << "Error in cloning the RBM object: " << e.what() << std::endl;
	}
	catch (std::exception& e)
	{
		std::cerr << "Error in cloning the RBM object: " << e.what() << std::endl;
	}

	// clone the base class
	NQS<_spinModes, _Ht, _T, _stateType>::clone(_other);
}
// template instantiation of the function above
RBM_INST_CMB_ALL(clone, void, (MC_t_p), );

// ##########################################################################################################################################

/**
* @brief Initializes the RBM (Restricted Boltzmann Machine) object.
* 
* This function is responsible for setting up the initial state of the RBM object.
* It is a template function with the following parameters:
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The type representing the Hamiltonian.
* @tparam _T The type representing the parameters of the RBM.
* @tparam _stateType The type representing the state of the system.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::init()
{
	// initialize biases visible
// #ifndef _DEBUG
// #pragma omp parallel for num_threads(this->threads_.threadNum_)
// #endif
	NQS_S<_spinModes, _Ht, _T, _stateType>::init();
	const double stddev = std::max(std::sqrt(2.0 / (this->info_p_.nVis_ + this->nHid_)), 0.01);

	// Initialize visible biases
	for (int i = 0; i < this->info_p_.nVis_; i++) {
		this->bV_(i) = algebra::cast<_T>(this->ran_->template randomNormal<double>(0.0, stddev) + I * this->ran_->template randomNormal<double>(0.0, stddev));
	}

	// Initialize hidden biases
	for (int i = 0; i < this->nHid_; i++) {
		this->bH_(i) = algebra::cast<_T>(this->ran_->template randomNormal<double>(0.0, stddev) + I * this->ran_->template randomNormal<double>(0.0, stddev));
	}

	// Initialize weights matrix using Xavier Initialization
	for (int i = 0; i < this->W_.n_rows; i++) {
		for (uint j = 0; j < this->W_.n_cols; j++) {
			// this->W_(i, j) = algebra::cast<_T>(this->ran_->template randomNormal<double>(0.0, stddev) + I * this->ran_->template randomNormal<double>(0.0, stddev));
			this->W_(i, j) = algebra::cast<_T>(this->ran_->template random<double>(-0.01, 0.01) + I * this->ran_->template random<double>(-0.01, 0.01));
		}
	}
	this->Weights_.subvec(0, this->info_p_.nVis_ - 1) 									= this->bV_;
	this->Weights_.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1) 	= this->bH_;
	this->Weights_.subvec(this->info_p_.nVis_ + this->nHid_, this->rbmSize_ - 1) 		= this->W_.as_col();
}
// template instantiation of the function above
RBM_INST_CMB_ALL(init, void, (), );

// ##########################################################################################################################################

/**
* @brief sets the current angles vector according to arXiv:1606.02318v1
* @param v replaces current vector
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::setTheta(Config_cr_t v)
{
	this->theta_		= this->bH_ + this->W_ * v;
	this->thetaCOSH_	= this->coshF();
	this->thetaCOSH_log_= arma::log(this->thetaCOSH_);
}
// template instantiation of the function above
RBM_INST_CMB_ALL(setTheta, void, (Config_cr_t), );
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

