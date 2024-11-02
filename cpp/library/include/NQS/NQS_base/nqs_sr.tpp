// -----------------------------------------------------------------------------------------------------------------------------------
#include "./nqs_sampling.tpp"
// -----------------------------------------------------------------------------------------------------------------------------------

#ifdef NQS_USESR

/*
* @brief Calculates the update parameters for weights to be updated in a form:
* weights[new] <- weights[old] - lr * S^{-1} * F --> second order optimization method (stochastic reconfiguration)
* It uses the matrix S = <\Delta _k* \Delta _k> - <\Delta _k*><\Delta _k> to calculate the update. This matrix is 
* called the geometric tensor. It is used as a regularizer for the gradient descent method.
* This method can be either calculated with the pseudoinverse or with the direct inversion of the matrix.
* Otherwise, one can skip the stochastic reconfiguration and use the standard gradient descent.
* @note The method can be run without calculating the geometric tensor S explicitly.
* @param step current step of updating - for the regularization purpose
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradSR(uint step, _T _currLoss)
{
	bool _inversionSuccess 		= false;

#ifdef NQS_USESR_MAT_USED
	{
		// regularize the covariance matrix before inverting it (if needed and set)
		// if (this->info_p_.sreg_ > 0) 
			// this->covMatrixReg(step);

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

    // calculate the matrix delta \Delta O = O - <O> as the covariance matrix S = <\Delta O * \Delta O^*>

    // Ensure this->F_ is properly initialized before use
#ifndef NQS_USESR_NOMAT_USEPRECONDITIONING
    this->dF_ = this->info_p_.lr_ * algebra::Solvers::ConjugateGradient::FisherMatrix::conjugate_gradient<_T>(
                                    this->derivativesCentered_,     	// Ensure this matches the type expected by _gramMatrix
									this->derivativesCenteredH_,		// This should also match arma::Col<_T>
                                    this->F_,               			// This should be of type arma::Col<_T>
                                    step <= 1 ? nullptr : &this->dF_,	// This should also match arma::Col<_T>
                                    1e-6,                  		 		// Tolerance
                                    1000,                    			// Max iterations,
									&_inversionSuccess
                                );
#else 
    this->dF_ = this->info_p_.lr_ * algebra::Solvers::ConjugateGradient::FisherMatrix::conjugate_gradient<_T>(
									this->derivativesCentered_,     	// Ensure this matches the type expected by _gramMatrix
									this->derivativesCenteredH_,		// This should also match arma::Col<_T>
									this->F_,               			// This should be of type arma::Col<_T>
									step <= 1 ? nullptr : &this->F_,	// This should also match arma::Col<_T>
									1.0 / this->F_, 					// Preconditioner
									1e-6,                  		 		// Tolerance
									1000,                    			// Max iterations,
									&_inversionSuccess
                                );
#endif
	// arma::Col<_T> _F = this->F_;

	// solve the system manually
	// algebra::Solvers::solve(this->F_, _Fun, _F, algebra::Solvers::SolverType::MY_CONJ_GRAD, 1.0e-5);
	// algebra::Solvers::ConjugateGradient::solve_my_conj_grad<_T>(this->F_, _Fun, this->x_, this->r_, this->p_, this->Ap_, 1e-6);

	// _F *= this->info_p_.lr_;
	// exchange the vectors
	// this->F_ = std::move(_F);
	_inversionSuccess = true;
	// this->F_ = this->info_p_.lr_ * this->x_;
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
inline void NQS<_spinModes, _Ht, _T, _stateType>::covMatrixReg(int step)
{
#ifndef NQS_USESR_NOMAT_USED
	this->S_.diag() += this->info_p_.sreg_ / (step + 1);
#else

#endif
}

#endif
