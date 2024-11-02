#include "./nqs_sr.tpp"

// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

// ##########################################################################################################################################

/*
* @brief Calculate the final step for the weights to be updated.
* First of all, it calculates the generalized forces for the weights.
* The forces are calculated as the covariance of the derivatives and the local energies.
* The forces are then used to update the weights.
* 
* @param _energies vector of the local energies for the current state
* @param _step current step of the training
* @param _currLoss current loss of the system - here is the energy
* @warning Uses the forces vector (member F_) to store the forces that update the weights
* 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradFinal(const NQSB& _energies, int _step, _T _currLoss)
{
	// calculate current learning rate based on the scheduler
	this->info_p_.lr_ = this->info_p_.lr(_step, algebra::real(_currLoss));
	const _T _samples = static_cast<_T>(_energies.n_elem);
	// calculate the derivatives 
	{
		// calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}> 
		// [+ sum_i ^{n-1} \beta _i <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> <Psi _W/Psi_W(i)>] 
		// - for the excited states, the derivatives are appe	
		// calculate the centered derivatives
		this->derivativesMean_ 			= arma::mean(this->derivatives_, 0);									// calculate the mean of the derivatives
		this->derivativesCentered_ 		= this->derivatives_.each_row() - this->derivativesMean_;				// calculate the centered derivatives
		this->derivativesCenteredH_		= this->derivativesCentered_.t();										// calculate the transposed centered derivatives
		this->F_						= this->derivativesCenteredH_ * ((_energies - _currLoss) / _samples);	// calculate the covariance vector for the gradient 

	// #pragma omp parallel for num_threads(this->threads_.threadNum_)
		for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++)			// append with the lower states derivatives - if the lower states are used
		{
			// Calculate <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> 
			const auto& ratios_excited	= this->lower_states_.ratios_excited_[_low];	// <Psi_W_j / Psi_W> evaluated at W - column vector
			const auto& ratios_lower 	= this->lower_states_.ratios_lower_[_low];		// <Psi_W / Psi_W_j> evaluated at W_j - column vector
			const auto& f_lower_b 		= this->lower_states_.f_lower_b_[_low];			// penalty for the lower states 
			const _T _meanLower 		= arma::mean(ratios_lower);						// mean of the ratios in the lower state
			const _T _meanExcited 		= arma::mean(ratios_excited);					// mean of the ratios in the excited state
			this->F_ 					+= this->derivativesCenteredH_ * ((ratios_excited - _meanExcited) * (f_lower_b * _meanLower / _samples));
		}
	}
	// fix the NANs
	// if (!arma::is_finite(this->F_)) {
	// 	stoutd("Non-finite values in the gradient: ");
	// 	this->F_.replace(arma::datum::nan, 0.0);	// replace NaNs with zeros
	// }
	
	// ---- STOCHASTIC RECONFIGURATION WITH MATRIX CALCULATION ----
#ifdef NQS_USESR_MAT_USED
	// update model by recalculating the gradient (applying the stochastic reconfiguration)
	// this->S_ = arma::cov(this->derivativesC_, this->derivatives_, 1);
	this->S_ = this->derivativesCenteredH_ * this->derivativesCentered_ / _samples;
	
	// check the norm of the gradient and normalize it if needed
	// if (auto gradNorm = arma::norm(this->F_); gradNorm > NQS_SREG_GRAD_NORM_THRESHOLD)
	// {
	// 	std::cerr << "Gradient norm is too large: " << gradNorm << std::endl;
	// 	this->F_ *= NQS_SREG_GRAD_NORM_THRESHOLD / gradNorm;
	// }
#endif

	// ---- STOCHASTIC RECONFIGURATION POSSIBLY WITHOUT MATRIX CALCULATION ----
#if defined NQS_USESR
	return this->gradSR(_step, _currLoss);
#else
	// standard updater with the gradient only!
	this->updateWeights_ = true;
	this->dF_ = this->info_p_.lr_ * this->F_;
#endif
}

// ##########################################################################################################################################