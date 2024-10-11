// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

// ##########################################################################################################################################

// -----------------------------------------------------------------------------------------------------------------------------------
#include "armadillo"
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
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradSR(uint step)
{
	#ifndef NQS_USESR_NOMAT
	{
		// regularize the covariance matrix before inverting it (if needed and set)
		{
		#ifdef NQS_SREG
		this->covMatrixReg();
		#endif
		}
		
		#ifdef NQS_PINV
		{
			// calculate the pseudoinverse
			if (NQS_PINV > 0)
				this->F_ = this->info_p_.lr_ * (arma::pinv(this->S_, NQS_PINV) * this->F_);
			else
				this->F_ = this->info_p_.lr_ * (arma::pinv(this->S_) * this->F_);
			return;
		}
		#else 
		// solve normally
		//this->F_ = this->info_p_.lr_ * (arma::inv(this->S_) * this->F_);
		this->F_ = this->info_p_.lr_ * arma::solve(this->S_, this->F_, arma::solve_opts::likely_sympd);
		#endif 
	}
	#else
	{
		std::function<void(const arma::Col<_T>&, arma::Col<_T>&, size_t)> _Fun = std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::getSRMatVec, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

		arma::Col<_T> _F = this->F_;

		// solve the system manually
		algebra::Solvers::solve(this->F_, _Fun, _F, algebra::Solvers::SolverType::MY_CONJ_GRAD, 1.0e-5);
		// algebra::Solvers::ConjugateGradient::solve_my_conj_grad<_T>(this->F_, _Fun, this->x_, this->r_, this->p_, this->Ap_, 1e-6);

		_F *= this->info_p_.lr_;
		// exchange the vectors
		this->F_ = std::move(_F);
		// this->F_ = this->info_p_.lr_ * this->x_;
	}
	#endif
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
* @warning Uses the forces vector (member F_) to store the forces that update the weights
* 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradFinal(const NQSB& _energies)
{
	// calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}> 
	// [+ sum_i ^{n-1} \beta _i <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> <Psi _W/Psi_W(i)>] 
	// - for the excited states, the derivatives are appe

	// calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}>
	this->derivativesC_ = arma::conj(this->derivatives_);				// precalculate the conjugate of the derivatives
	this->F_			= arma::cov(this->derivativesC_, _energies, 1);	// calculate the covariance vector for the gradient 

	// !TODO modify this for excited states! 
	// append with the lower states derivatives - if the lower states are used
	for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++)
	{
		// Calculate <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> 
		auto _meanExcited 	= arma::mean(this->lower_states_.ratios_excited_[_low]);
		auto _meanLower 	= arma::mean(this->lower_states_.ratios_lower_[_low]);
		auto _diff 			= this->lower_states_.ratios_excited_[_low] - _meanExcited;
		// multiply each row of \Delta _k* with the difference at each realization (each element of the row)
		// and then multiply with the mean of the lower states
		this->F_ += (this->lower_states_.f_lower_b_[_low] * _meanLower) * arma::mean(this->derivativesC_.each_col() % _diff, 0).t();
	}
	
#ifdef NQS_USESR
	{
	#ifndef NQS_USESR_NOMAT
		// form the covariance matrix explicitly
		this->S_		= arma::cov(this->derivativesC_, this->derivatives_, 1);
	#else 
		{
		arma::Mat<_T> S1 	= arma::cov(this->derivativesC_, this->derivatives_, 0);
		std::cout << "Covariance matrix: " << S1.n_rows << ", " << S1.n_cols << std::endl;


		// calculate the matrix without forming it explicitly and use the mean of the derivatives
		this->derivativesM_ = arma::mean(this->derivatives_, 0).as_col();
		std::cout << "Mean of the derivatives: " << this->derivativesM_.n_elem << std::endl;

		arma::Mat<_T> S2	= S1;
		S2.zeros();

		for (size_t i = 0; i < this->derivativesC_.n_cols; ++i)
			for (size_t j = 0; j < this->derivativesC_.n_cols; ++j)
				S2(i, j) = this->getSRMatrixElement(i, j);
		// check the difference
		arma::Mat<double> diff = arma::abs(S1 - S2);
		diff.print("Difference: ");
		}
	#endif
		// update model by recalculating the gradient (applying the stochastic reconfiguration)
		this->gradSR(0);
	}
#else
	{
		// standard updater with the gradient only!
		this->F_ *= this->info_p_.lr_;
	}
#endif
}

// ##########################################################################################################################################

///////////////////////////////////////////////////////////////////////
#ifdef NQS_SREG
/*
! TODO 
* @brief The inverse of the covariance matrix is poorly defined at the begining of training. 
* Use regularization to fix that issue.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::covMatrixReg()
{
	if (this->covMatrixRegStart != 0)
	{
		this->S_.diag() *= (1.0 + this->covMatrixRegStart);
	}
	if (this->covMatrixRegStart2 != 0)
	{
		auto maximal_re	=	arma::max(arma::real(this->S_.diag()));
		this->S_.diag()	+= this->covMatrixRegStart2 * maximal_re;
	}
}
#endif