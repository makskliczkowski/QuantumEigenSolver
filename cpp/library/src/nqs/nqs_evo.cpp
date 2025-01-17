#include "../../include/NQS/nqs_final.hpp"


// ##########################################################################################################################################

/**
* @brief Computes the gradient over time for the Neural Quantum State (NQS) using either a standard method or Runge-Kutta method.
* This means that the time evolution of the state is computed by solving the Fubini Study equation.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the computation.
* @tparam _stateType State type.
* 
* @throws std::runtime_error If the covariance matrix S and gradient vector F are not set before the inversion.
* @throws std::runtime_error If the solver is not set.
* 
* This function performs the following steps:
* 1. If regularization is enabled, it updates the covariance matrix.
* 2. If the NQS_USESR_MAT_USED flag is defined, it throws a runtime error indicating that the function is not implemented.
* 3. If the NQS_USESR_MAT_USED flag is not defined, it checks that the centered derivatives are set.
* 4. If a preconditioner is set, it updates the preconditioner.
* 5. If a solver is set, it performs the following:
*    - Sets the regularization for the solver.
*    - Solves the system using the solver.
*    - If Runge-Kutta method is not used, it checks for convergence and updates the solution.
*    - If Runge-Kutta method is used, it performs multiple steps to compute the final solution.
* 6. If the solver is not set, it logs an error and throws a runtime error.
* 7. Updates the weights based on the success of the inversion.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::gradTime(size_t _step, NQSB* _dF)
{
	const _T _multiplier 		= algebra::cast<_T>(I);
	if (this->info_p_.sreg_ > 0) 
		this->covMatrixReg(_step);

#ifdef NQS_USESR_MAT_USED
	assert(this->S_.empty() == false && this->F_.empty() == false && "Covariance matrix S and gradient vector F must be set before the inversion.");
	{
		throw std::runtime_error("Function not implemented.");
		// !TODO: Implement the inversion of the matrix S using the Cholesky decomposition
	}
#else
	assert(this->derivativesCentered_.empty() == false && this->derivativesCenteredH_.empty() == false && "Centered derivatives F_k and F_k^T must be set before the inversion.");
	if (this->precond_ != nullptr)
		this->precond_->set(this->derivativesCenteredH_, this->derivativesCentered_, -1.0);
	
	if (this->solver_ != nullptr) 
	{
		{
			// !TODO check the SVD, whether it correctly works for regularization
			// arma::Mat<_T> _U, _V;
			// arma::vec _sigma;
			// arma::svd(_U, _sigma, _V, this->derivativesCentered_);
			// clamp the singular values
			// _sigma = arma::clamp(_sigma, 1e-6, arma::datum::inf);
			// this->derivativesCentered_ = (_U * arma::diagmat(_sigma)) * _V.t();
			// this->derivativesCenteredH_ = (_V * arma::diagmat(_sigma)) * _U.t();
		}

		this->solver_->setReg(this->info_p_.sreg_);											// set the regularization						
		this->solver_->solve(this->derivativesCentered_, this->derivativesCenteredH_, 		// S and S+ matrices
							_multiplier * this->F_, 										// b
							nullptr, //step <= 1 ? nullptr : &this->dF_, 					// x0
							this->precond_.get());											// preconditioner
        if (!this->solver_->isConverged())
            throw std::runtime_error("Solver did not converge.");                           // check if the solver converged
		if (_dF != nullptr)
			*_dF = this->solver_->moveSolution();											// get the solution
		else
			this->dF_ = this->solver_->moveSolution();										// get the solution
	} else {
		LOGINFO("Solver is not set. Cannot perform the inversion.", LOG_TYPES::ERROR, 1);
		throw std::runtime_error("Solver is not set. Cannot perform the inversion.");
	}
#endif
}
// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, gradTime, void, (size_t, arma::Col<double>*));
NQS_INST_CMB(double, std::complex<double>, gradTime, void, (size_t, arma::Col<std::complex<double>>*));
NQS_INST_CMB(std::complex<double>, double, gradTime, void, (size_t, arma::Col<double>*));
NQS_INST_CMB(std::complex<double>, std::complex<double>, gradTime, void, (size_t, arma::Col<std::complex<double>>*));

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
* @param _currLoss Current loss value.
* 
* @throws std::runtime_error If the function is not implemented - this is a placeholder for future implementation when no S matrix is used. !TODO: (HOW?)
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::gradEvoFinal(const Container_t& _energies, int _step, _T _currLoss, NQSB* _dF)
{
	this->gradF(_energies, _step, _currLoss);						// calculate the gradient of the loss function
#ifdef NQS_USESR_MAT_USED											// ---- STOCHASTIC RECONFIGURATION WITH MATRIX CALCULATION ----
	this->S_ = this->derivativesCenteredH_ * this->derivativesCentered_ / _samples;
#endif

#if defined NQS_USESR												
	return this->gradTime(_step, _dF);								// ---- STOCHASTIC RECONFIGURATION POSSIBLY WITHOUT MATRIX CALCULATION ----
#else
	throw std::runtime_error("The function is not implemented yet.");
#endif
}
// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, gradEvoFinal, void, (const arma::Col<double>&, int, double, arma::Col<double>*));
NQS_INST_CMB(double, std::complex<double>, gradEvoFinal, void, (const arma::Col<std::complex<double>>&, int, std::complex<double>, arma::Col<std::complex<double>>*));
NQS_INST_CMB(std::complex<double>, double, gradEvoFinal, void, (const arma::Col<double>&, int, double, arma::Col<double>*));
NQS_INST_CMB(std::complex<double>, std::complex<double>, gradEvoFinal, void, (const NQSB&, int, std::complex<double>, arma::Col<std::complex<double>>*));

// ##########################################################################################################################################

// ########################################################## E V O L U T I O N ############################################################

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::evolveStep(size_t _step, 
									double dt, 
									const MonteCarlo::MCS_train_t& _par, 
									const bool quiet, 
									const bool randomStart, 
                                    const bool updateWeights,
									NQSB* _dF)
									
{
	if (this->E_ .n_elem != _par.nblck_)
		this->E_ .resize(_par.nblck_);

	this->total_ 	= 0;										// reset the total number of flips
	this->accepted_ = 0;										// reset the number of accepted flips
	if (randomStart && _par.MC_th_ > 0) {
		this->setRandomState();									// set the random state at the begining
		this->blockSample<false>(_par.MC_th_, NQS_STATE);		// thermalize the system - burn-in
	} else {
		this->blockSample<true>(_par.MC_th_, NQS_STATE);		// thermalize the system - burn-in
	}

	_T _meanEn = 0.0, _stdEn = 0.0;								// mean and standard deviation of the energy
	for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 		// iterate blocks - this ensures the calculation of a stochastic gradient constructed within the block
	{		
		this->blockSample<false>(_par.bsize_, NQS_STATE);		// sample them using the local Metropolis sampling
		this->grad(NQS_STATE, _taken);							// calculate the gradient at each point of the iteration! - this is implementation specific!!!
		this->E_ (_taken) = this->locEnKernel();				// local energy - stored at each point within the estimation of the gradient (stochastic)
	}

	MonteCarlo::blockmean(this->E_ , std::max((size_t)_par.bsize_, (size_t)8), &_meanEn, &_stdEn); 	// save the mean energy
	this->gradEvoFinal(this->E_ , _step, _meanEn, _dF);	        // calculate the final update vector - either use the stochastic reconfiguration or the standard gradient descent

	if (updateWeights)
		this->updateWeights(dt); 								// finally, update the weights with the calculated gradient (force) [can be done with the stochastic reconfiguration or the standard gradient descent] - implementation specific!!!
}
// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, evolveStep, void, (size_t, double, const MonteCarlo::MCS_train_t&, const bool, const bool, const bool, arma::Col<double>*));
NQS_INST_CMB(double, std::complex<double>, evolveStep, void, (size_t, double, const MonteCarlo::MCS_train_t&, const bool, const bool, const bool, arma::Col<std::complex<double>>*));
NQS_INST_CMB(std::complex<double>, double, evolveStep, void, (size_t, double, const MonteCarlo::MCS_train_t&, const bool, const bool, const bool, arma::Col<double>*));
NQS_INST_CMB(std::complex<double>, std::complex<double>, evolveStep, void, (size_t, double, const MonteCarlo::MCS_train_t&, const bool, const bool, const bool, arma::Col<std::complex<double>>*));

// ##########################################################################################################################################

/**
* @brief Evolves the state of the system using the specified parameters.
* 
* This function sets up the evolution functions for the system, both with and without the final update.
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The type of the time parameter.
* @tparam _stateType The type of the state.
* @param _par The Monte Carlo training parameters.
* @param quiet If true, suppresses output during evolution.
* @param randomStart If true, starts the evolution from a random state.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::evolveSet(const MonteCarlo::MCS_train_t& _par, const bool quiet, const bool randomStart)
{
	// without the final update
	this->evolveFunc_ = [this, &_par, quiet, randomStart](double t, double dt, const NQSB& y) -> NQSB
    {   
		this->setWeights(y);
		this->evolveStep(t / dt, dt, _par, quiet, randomStart, false);
		return this->dF_;
	};
	// with the final update
	this->evolveFuncR_ = [this, &_par, quiet, randomStart](double t, double dt, const NQSB& y, NQSB* _dF) -> void
	{   
		this->setWeights(y);
		this->evolveStep(t / dt, dt, _par, quiet, randomStart, false, _dF);
	};
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB_ALL(evolveSet, void, (const MonteCarlo::MCS_train_t&, const bool, const bool));
// ##########################################################################################################################################

/**
* @brief Performs a single evolution step in the Neural Quantum State (NQS) system
* 
* @details This method advances the quantum state evolution by one time step using
* the provided Initial Value Problem (IVP) solver. It updates the weights of the
* neural network according to the evolution equations.
* 
* @param _step Current step number in the evolution process
* @param dt Time step size for the evolution
* @param _ivp Pointer to the Initial Value Problem solver object. Must not be nullptr
* 
* @throws std::runtime_error if _ivp is nullptr
* 
* @tparam _spinModes Number of spin modes in the system
* @tparam _Ht Hamiltonian type
* @tparam _T Data type for calculations (typically complex or real)
* @tparam _stateType Type of quantum state representation
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::evolveStepSet(size_t _step, double dt, algebra::ODE::IVP<_T, arma::Col<_T>>* _ivp)
{
	if (_ivp == nullptr)
	{
		LOGINFO("The IVP object is not set. Cannot perform the evolution step.", LOG_TYPES::ERROR, 1);
		throw std::runtime_error("The IVP object is not set. Cannot perform the evolution step.");
	}
	// this->covMatrixReg(_step);
#ifdef _DEBUG
	const auto _tp = NOW;
#endif
	this->setWeights(_ivp->step(this->evolveFunc_, _step, dt, this->Weights_));
#ifdef _DEBUG
	LOGINFO(_tp, "Evolution step time: ", 3);
#endif
}
// template instantiation of function above for <spins, double and complex, double and complex, double>
template void NQS<2u, double, double, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<double, arma::Col<double>>*);
template void NQS<3u, double, double, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<double, arma::Col<double>>*);
template void NQS<4u, double, double, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<double, arma::Col<double>>*);
template void NQS<2u, cpx, cpx, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<cpx, arma::Col<cpx>>*);
template void NQS<3u, cpx, cpx, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<cpx, arma::Col<cpx>>*);
template void NQS<4u, cpx, cpx, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<cpx, arma::Col<cpx>>*);
template void NQS<2u, cpx, double, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<double, arma::Col<double>>*);
template void NQS<3u, cpx, double, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<double, arma::Col<double>>*);
template void NQS<4u, cpx, double, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<double, arma::Col<double>>*);
template void NQS<2u, double, cpx, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<cpx, arma::Col<cpx>>*);
template void NQS<3u, double, cpx, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<cpx, arma::Col<cpx>>*);
template void NQS<4u, double, cpx, double>::evolveStepSet(size_t, double, algebra::ODE::IVP<cpx, arma::Col<cpx>>*);
// ##########################################################################################################################################