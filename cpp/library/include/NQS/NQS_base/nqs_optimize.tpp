// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

// ##########################################################################################################################################

// -----------------------------------------------------------------------------------------------------------------------------------
#include "./nqs_sampling.tpp"
#include "armadillo"
#include <iostream>


#ifdef NQS_USESR
#ifdef NQS_USESR_NOMAT
// -----------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Calculates the matrix element of the geometric tensor S_ij = <\Delta _i* \Delta _j> - <\Delta _i*><\Delta _j> 
* without constructing the matrix explicitly. This is done by calculating the matrix element directly. 
* @param i row index - the matrix is symmetric
* @param j column index - the matrix is symmetric
* @returns the matrix element of the geometric tensor S_ij
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T NQS<_spinModes, _Ht, _T, _stateType>::getSRMatrixElement(size_t i, size_t j)
{
	// get the subviews of the columns (each column has multiple observations now)
	auto _colO_i 		= this->derivativesC_.col(i);
	auto _colO_j 		= this->derivatives_.col(j);

	// compute <\Delta _i* \Delta _j>
	_T avg_Oi_Oj 		= arma::dot(_colO_i, _colO_j) / static_cast<double>(_colO_i.n_elem);

	// compute <\Delta _i*><\Delta _j>
	_T avg_Oi_avg_Oj 	= algebra::conjugate(this->derivativesM_(i)) * this->derivativesM_(j);

		#ifdef NQS_SREG
		{
		// regularize the covariance matrix before inverting it (if needed and set) by adding the regularization term
		if (i == j)
			avg_Oi_Oj += this->covMatrixRegStart * avg_Oi_avg_Oj;
		}
		#endif
	return avg_Oi_Oj - avg_Oi_avg_Oj;
}

// -----------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Calculates the matrix-vector multiplication using the elements obtained from the getSRMatrixElement. 
* This is done without constructing the matrix explicitly. Once the matrix is known, the multiplication is done
* by calculating the matrix-vector multiplication directly.
* @param x vector to be multiplied with the matrix
* @param y resulting vector from the multiplication
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::getSRMatVec(const arma::Col<_T>& x, arma::Col<_T>& y, size_t n)
{	
	// implement the matrix-vector multiplication using the elements obtained from the getSRMatrixElement
	// #pragma omp parallel for num_threads(this->threadNum_)
	for (size_t i = 0; i < n; ++i)
	{
		// initialize the y vector
		y(i) = 0.0;

		// y_i = \sum _j S_ij x_j
		// Compute the matrix-vector multiplication using only upper triangular matrix and reflection
		// as the matrix is square and symmetric
		for (size_t j = 0; j < n; j++)
			y(i) += this->getSRMatrixElement(i, j) * x(j);
// 		for (size_t j = i; j < n; j++) {
//             const _T S_ij = this->getSRMatrixElement(i, j);
//             y(i) += S_ij * x(j);

//             // Since S is symmetric, reflect the result to y(j)
//             if (i != j) {
// // #pragma omp atomic
//                 y(j) += S_ij * x(i);
//             }
//         }
	}

}
// ----------------------------------------------------------------------------------------------------------------------------------- 
#endif 

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

#ifndef NQS_USESR_NOMAT
	{
		// regularize the covariance matrix before inverting it (if needed and set)
		// if (this->info_p_.sreg_ > 0) 
			// this->covMatrixReg(step);

#ifdef NQS_PINV
		// calculate the pseudoinverse
		int _attempts 			= 0;
		double _regFactor		= this->info_p_.sreg_ > 0 ? this->info_p_.sreg_ : 1e-5;

		while (!_inversionSuccess && _attempts < NQS_SREG_ATTEMPTS)
		{
			try {
				if (this->info_p_.pinv_ > 0.0)
					this->F_ = this->info_p_.lr_ * (arma::pinv(this->S_, this->info_p_.pinv_) * this->F_);
				else if (this->info_p_.pinv_ == 0.0)
					this->F_ = this->info_p_.lr_ * (arma::pinv(this->S_) * this->F_);
				else 
					this->F_ = this->info_p_.lr_ * arma::solve(this->S_, this->F_, arma::solve_opts::likely_sympd);
				
				// if the inversion was successful, set the flag
				_inversionSuccess = true;
			} catch(std::exception& e) {
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
#else 
		// solve normally
		//this->F_ = this->info_p_.lr_ * (arma::inv(this->S_) * this->F_);
		try {
			this->F_ = this->info_p_.lr_ * arma::solve(this->S_, this->F_, arma::solve_opts::likely_sympd);
			_inversionSuccess = true;
		} catch(std::exception& e) {
			_inversionSuccess = false;
		}
#endif 
	}
#else
	std::function<void(const arma::Col<_T>&, arma::Col<_T>&, size_t)> _Fun = std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::getSRMatVec, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

	arma::Col<_T> _F = this->F_;

	// solve the system manually
	algebra::Solvers::solve(this->F_, _Fun, _F, algebra::Solvers::SolverType::MY_CONJ_GRAD, 1.0e-5);
	// algebra::Solvers::ConjugateGradient::solve_my_conj_grad<_T>(this->F_, _Fun, this->x_, this->r_, this->p_, this->Ap_, 1e-6);

	_F *= this->info_p_.lr_;
	// exchange the vectors
	this->F_ = std::move(_F);
	_inversionSuccess = true;
	// this->F_ = this->info_p_.lr_ * this->x_;
#endif
    this->updateWeights_ = _inversionSuccess;
}
// -----------------------------------------------------------------------------------------------------------------------------------
#endif

///////////////////////////////////////////////////////////////////////

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
	this->info_p_.lr_ 		= this->info_p_.lr(_step, algebra::real(_currLoss));

	// calculate the derivatives 
	{
		// calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}> 
		// [+ sum_i ^{n-1} \beta _i <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> <Psi _W/Psi_W(i)>] 
		// - for the excited states, the derivatives are appe	
		this->derivativesC_ = arma::conj(this->derivatives_);						// precalculate the conjugate of the derivatives
		this->F_			= arma::cov(this->derivativesC_, _energies, 1);			// calculate the covariance vector for the gradient 
		// by hand 
		// this->F_ 			= arma::mean(this->derivativesC_.each_col() % _energies, 0).t() - arma::mean(this->derivativesC_, 0).t() * _currLoss;
	
		// append with the lower states derivatives - if the lower states are used
	// #pragma omp parallel for num_threads(this->threads_.threadNum_)
		for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++)
		{
			// Calculate <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> 
			const auto& ratios_excited	= this->lower_states_.ratios_excited_[_low];// <Psi_W_j / Psi_W> evaluated at W - column vector
			const auto& ratios_lower 	= this->lower_states_.ratios_lower_[_low];	// <Psi_W / Psi_W_j> evaluated at W_j - column vector
			const auto& f_lower_b 		= this->lower_states_.f_lower_b_[_low];		// penalty for the lower states 
			const auto _meanLower 		= arma::mean(ratios_lower);					// mean of the ratios in the lower state

			auto _cov 					= arma::cov(this->derivativesC_, ratios_excited, 1);
			this->F_ 					+= _meanLower * f_lower_b * _cov;
			// continue;

			// manually
			// const auto _meanExcited 	= arma::mean(ratios_excited);					// mean of the ratios in the excited state
			// auto _diff 					= ratios_excited - _meanExcited;				// difference between the ratios and the mean Eq. (14) in the paper

			// multiply each row of \Delta _k* with the difference at each realization (each element of the row)
			// and then multiply with the mean of the lower states
			// this->F_	 				+= (f_lower_b * _meanLower) * arma::mean((this->derivativesC_.each_col() % _diff), 0).t();
		}
	}
	// fix the NANs
	// if (!arma::is_finite(this->F_)) {
	// 	stoutd("Non-finite values in the gradient: ");
	// 	this->F_.replace(arma::datum::nan, 0.0);	// replace NaNs with zeros
	// }
	
#ifdef NQS_USESR
	// update model by recalculating the gradient (applying the stochastic reconfiguration)
	this->S_ = arma::cov(this->derivativesC_, this->derivatives_, 1);
	// this->S_ 	= arma::mean(arma::square(arma::abs(this->derivatives_)), 0) - arma::real(arma::mean(this->derivatives_, 0) * arma::mean(this->derivatives_, 1));
	
	// check the norm of the gradient and normalize it if needed
	// if (auto gradNorm = arma::norm(this->F_); gradNorm > NQS_SREG_GRAD_NORM_THRESHOLD)
	// {
	// 	std::cerr << "Gradient norm is too large: " << gradNorm << std::endl;
	// 	this->F_ *= NQS_SREG_GRAD_NORM_THRESHOLD / gradNorm;
	// }

	return this->gradSR(_step, _currLoss);
#else
	// standard updater with the gradient only!
	this->F_ *= this->info_p_.lr_;
#endif
	this->updateWeights_ = true;
}

// ##########################################################################################################################################

///////////////////////////////////////////////////////////////////////
/*
! TODO 
* @brief The inverse of the covariance matrix is poorly defined at the begining of training. 
* Use regularization to fix that issue.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::covMatrixReg(int step)
{
	this->S_.diag() += this->info_p_.sreg_ / (step + 1);
}
