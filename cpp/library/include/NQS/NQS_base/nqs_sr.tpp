// -----------------------------------------------------------------------------------------------------------------------------------
#include "./nqs_fermions.tpp"
// -----------------------------------------------------------------------------------------------------------------------------------

#ifdef NQS_USESR

// ###################################################################################################################################

/**
* @brief Sets the training parameters for the lower states.
* 
* @param _par Training parameters to be set.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::setTrainParExc(const NQS_train_t& _par)
{
	this->lower_states_.train_lower_ = _par;
	LOGINFO("Training parameters for lower states set.", LOG_TYPES::INFO, 2);
}

/**
* @brief Sets the learning rate scheduler.
* 
* @param _sch Scheduler type.
* @param _lr Initial learning rate.
* @param _lrd Learning rate decay.
* @param _epo Number of epochs.
* @param _pat Patience for early stopping.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::setScheduler(int _sch, double _lr, double _lrd, size_t _epo, size_t _pat)
{
	this->info_p_.p_ = MachineLearning::Schedulers::get_scheduler(_sch, _lr, _epo, _lrd, _pat); 
	LOGINFO("Scheduler set with type: " + STR(_sch) + ", initial learning rate: " + VEQPS(_lr, 3) + 
			", learning rate decay: " + VEQPS(_lrd, 3) + ", epochs: " + STR(_epo) + ", patience: " + STR(_pat), LOG_TYPES::INFO, 2);
}

/**
* @brief Sets the early stopping parameters.
* 
* @param _pat Patience for early stopping.
* @param _minDlt Minimum delta for improvement.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::setEarlyStopping(size_t _pat, double _minDlt)
{
	if (_pat != 0)
	{ 
		this->info_p_.setEarlyStopping(_pat, _minDlt); 
		LOGINFO("Early stopping set with patience: " + STR(_pat) + ", minimum delta: " + VEQPS(_minDlt, 3), LOG_TYPES::CHOICE, 3);
	}
}

/**
* @brief Sets the regularization scheduler for the covariance matrix.
* 
* @param _sch Scheduler type.
* @param _sreg Initial regularization factor.
* @param _sregd Regularization factor decay.
* @param _epo Number of epochs.
* @param _pat Patience for early stopping.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::setSregScheduler(int _sch, double _sreg, double _sregd, size_t _epo, size_t _pat)
{
	this->info_p_.sreg_ = _sreg; 
	if (_sreg > 0) 
	{ 
		LOGINFO("Regularization set with initial factor: " + VEQPS(_sreg, 3) + ", scheduler type: " + STR(_sch) + 
				", decay: " + VEQPS(_sregd, 3) + ", epochs: " + STR(_epo) + ", patience: " + STR(_pat), LOG_TYPES::CHOICE, 3); 
		this->info_p_.s_ = MachineLearning::Schedulers::get_scheduler(_sch, _sreg, _epo, _sregd, _pat); 
	}
}

// ###################################################################################################################################


/**
* @brief Calculates the update parameters for weights using the stochastic reconfiguration (SR) method.
* 
* This function updates the weights using a second-order optimization method, where the update rule is:
* weights[new] <- weights[old] - lr * S^{-1} * F
* Here, S is the geometric tensor, calculated as S = <\Delta _k* \Delta _k> - <\Delta _k*><\Delta _k>.
* The geometric tensor acts as a regularizer for the gradient descent method.
* 
* The method can use either the pseudoinverse or direct inversion of the matrix S. If the stochastic reconfiguration
* is skipped, the standard gradient descent is used instead. The function can also run without explicitly calculating
* the geometric tensor S.
* 
* @param step Current step of updating, used for regularization purposes.
* @param _currLoss Current loss value, used for regularization purposes.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradSR(uint step, _T _currLoss)
{
	bool _inversionSuccess 		= false;
	if (this->info_p_.sreg_ > 0) 
		this->covMatrixReg(step);

#ifdef NQS_USESR_MAT_USED
	{
		// regularize the covariance matrix before inverting it (if needed and set)

		// calculate the pseudoinverse
		int _attempts 			= 0;
		double _regFactor		= this->info_p_.sreg_ > 0 ? this->info_p_.sreg_ : 1e-5;

		while (!_inversionSuccess && _attempts < NQS_SREG_ATTEMPTS)
		{
			try {
				if (this->info_p_.pinv_ > 0.0)
					this->dF_ = this->info_p_.lr_ * (arma::pinv(this->S_, this->info_p_.pinv_) * this->F_);
				else if (this->info_p_.pinv_ == 0.0)
					this->dF_ = this->info_p_.lr_ * (arma::pinv(this->S_) * this->F_);
				else 
					this->dF_ = this->info_p_.lr_ * arma::solve(this->S_, this->F_, arma::solve_opts::likely_sympd);
				
				// if the inversion was successful, set the flag
				_inversionSuccess = true;
			} catch (std::exception& e) {
                // Increase regularization factor and apply to diagonal
				LOGINFO("Inverse calculation failed, attempt " + STR(_attempts + 1) + ". E: " + e.what(), LOG_TYPES::ERROR, 1);
				if (!arma::is_finite(this->S_)) {
					LOGINFO("Non-finite values in the diagonal of the covariance matrix. Stopping the training.", LOG_TYPES::ERROR, 1);
					break;
				}

                this->S_.diag() += _regFactor;  	// Apply regularization to diagonal
                _regFactor 		*= 10;  			// Increase regularization factor for next attempt
                ++_attempts;
			}
		}
	}
#else
	if (this->precond_ != nullptr)
		this->precond_->set(this->derivativesCenteredH_, this->derivativesCentered_, -1.0);
	
	if (this->solver_ != nullptr) 
	{
		this->solver_->setReg(this->info_p_.sreg_);											// set the regularization						
		this->solver_->solve(this->derivativesCentered_, this->derivativesCenteredH_, 		// S and S+ matrices
							this->F_, 														// b
							nullptr, //step <= 1 ? nullptr : &this->dF_, 								// x0
							this->precond_);												// preconditioner
		_inversionSuccess = this->solver_->isConverged();
		this->dF_ = this->info_p_.lr_ * this->solver_->solution();							// get the solution
	} 
	else {
		// !DEPRECATED - use solver class instead
		this->dF_ = this->info_p_.lr_ * algebra::Solvers::FisherMatrix::solve<_T>(
										this->info_p_.solver_,													// choose the solver type 
										this->derivativesCentered_,     										// Ensure this matches the type expected by _gramMatrix
										this->derivativesCenteredH_,											// This should also match arma::Col<_T>
										this->F_,               												// This should be of type arma::Col<_T>
										nullptr, //step <= 1 ? nullptr : &this->dF_,										// This should also match arma::Col<_T>
										this->precond_ ? this->precond_ : nullptr, 								// Preconditioner
										this->info_p_.tol_,                  		 							// Tolerance
										std::min(size_t(5 * this->F_.n_elem), size_t(this->info_p_.maxIter_)),	// Max iterations,
										&_inversionSuccess,														// Convergence flag						
										this->info_p_.sreg_ //this->precond_ ? -1.0 : this->info_p_.sreg_								// Set the regularization only if no preconditioner is used 
										);
	}

#endif
    this->updateWeights_ = _inversionSuccess;
}
// -----------------------------------------------------------------------------------------------------------------------------------

/*
! TODO 
* @brief The inverse of the covariance matrix is poorly defined at the begining of training. 
* Use regularization to fix that issue.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::covMatrixReg(int step, _T _currLoss)
{
	this->info_p_.sreg_ = this->info_p_.sreg(step, algebra::real(_currLoss));
#ifndef NQS_USESR_NOMAT_USED
	this->S_.diag() += this->info_p_.sreg_ / (step + 1);
#endif
}

#endif

// -----------------------------------------------------------------------------------------------------------------------------------

