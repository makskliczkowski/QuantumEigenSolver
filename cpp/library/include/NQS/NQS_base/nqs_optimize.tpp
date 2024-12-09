#include "./nqs_sr.tpp"

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setPreconditioner(int _pre) 									
{ 
	if (_pre != 0) 
	{ 
		this->precond_ = algebra::Solvers::Preconditioners::choose<_T>(_pre);
		LOGINFO("Using preconditioner: " + algebra::Solvers::Preconditioners::name(_pre), LOG_TYPES::CHOICE, 3);
	}
};

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setSolver(int _sol, double _tol, int _maxiter, double _reg)						
{ 
	this->info_p_.setSolver(_sol, _maxiter, _tol); 
	this->solver_ = algebra::Solvers::General::choose<_T, true>(_sol, this->info_p_.fullSize_, _tol, _maxiter, _reg);
	LOGINFO("Using solver: " + algebra::Solvers::General::name(_sol) + " with tolerance: " + VEQPS(_tol, 3) + " and iterations: " + STR(_maxiter), LOG_TYPES::CHOICE, 3); 
};

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setPinv(double _pinv)											
{ 
	this->info_p_.pinv_ = _pinv; 
	if (_pinv > 0) 
		LOGINFO("Using pseudoinverse: " + VEQPS(_pinv, 3), LOG_TYPES::CHOICE, 3); 
	else 
	LOGINFO("Using ARMA solver", LOG_TYPES::CHOICE, 3); 
};


// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

// ##########################################################################################################################################


/**
* @brief Computes the final gradient for the Neural Quantum State (NQS) optimization.
*
* This function calculates the gradient for the NQS optimization process, taking into account
* the current learning rate, the centered derivatives, and the covariance vector. It also
* includes the derivatives for the lower states if they are used. The function can optionally
* apply stochastic reconfiguration with or without matrix calculation.
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the calculations.
* @tparam _stateType State type.
* @param _energies Energies of the NQS.
* @param _step Current optimization step.
* @param _currLoss Current loss value.
 */
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradFinal(const NQSB& _energies, int _step, _T _currLoss)
{
	// calculate current learning rate based on the scheduler
	this->info_p_.lr_ = this->info_p_.lr(_step, algebra::real(_currLoss));
	const _T _samples = static_cast<_T>(_energies.n_elem);

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

/**
* @brief Perform the final gradient evolution step for the Neural Quantum State (NQS) optimization.
* 
* This function calculates the covariance derivatives and centered derivatives for the gradient evolution.
* Performs the time evolution of the quantum state using the stochastic reconfiguration method - minimizing the Fubini-Study metric.
* It also handles the inclusion of lower state derivatives if they are used.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for calculations.
* @tparam _stateType State type.
* @param _energies Vector of local energies.
* @param _step Current optimization step.
* @param _dt Time step for the gradient evolution.
* @param _currLoss Current loss value.
* 
* @throws std::runtime_error If the function is not implemented - this is a placeholder for future implementation when no S matrix is used. !TODO: (HOW?)
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradEvoFinal(const NQSB& _energies, int _step, double _dt, _T _currLoss, bool _useRungeKutta)
{
	const _T _samples = static_cast<_T>(_energies.n_elem);
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

	// ---- STOCHASTIC RECONFIGURATION WITH MATRIX CALCULATION ----
#ifdef NQS_USESR_MAT_USED
	this->S_ = this->derivativesCenteredH_ * this->derivativesCentered_ / _samples;
#endif

	// ---- STOCHASTIC RECONFIGURATION POSSIBLY WITHOUT MATRIX CALCULATION ----
#if defined NQS_USESR
	return this->gradTime(_step, _dt, _currLoss, _useRungeKutta);
#else
	throw std::runtime_error("The function is not implemented yet.");
#endif
}
