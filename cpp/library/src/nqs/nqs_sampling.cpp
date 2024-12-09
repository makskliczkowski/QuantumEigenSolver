#include "../../include/NQS/nqs_final.hpp"
#include "../../source/src/Include/random.h"

// ##########################################################################################################################################

// ############################################################ S A M P L I N G #############################################################

// ##########################################################################################################################################

/**
* @brief Perform block sampling for the Neural Quantum State (NQS).
*
* This function performs block sampling for the NQS, which involves flipping spins and 
* updating the state based on the Metropolis-Hastings algorithm. The function can also 
* handle thermalization if required.
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the NQS.
* @tparam _stateType Type of the state representation.
* @param _bSize Number of block steps to perform.
* @param _start Initial state to start sampling from.
* @param _therm Boolean flag indicating whether thermalization is required.
* @note For the probability calculation, here the temperature is used as the inverse of the beta parameter. As for default, beta is set to 1. and 
*	   	the temperature is set to 1.0 - we recall the original implementation of the Metropolis-Hastings algorithm for NQS. Otherwise the beta parameter
*		should be set to the desired value (0 - infinite) and the temperature should be set to 1.0 / beta - used for the parallel tempering algorithm.
* 		!TODO: Implement the parallel tempering algorithm and check whether the temperature is set correctly.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::blockSample(uint _bSize, NQS_STATE_T _start, bool _therm)
{
	// Set state based on whether thermalization is required or _start differs from current state
	if (_therm 
#ifndef NQS_USE_VEC_ONLY
		|| _start != this->curState_
#endif
		) this->setState(_start, _therm);

	this->tmpVec_ = this->curVec_; 													// set the temporary state - the vectors are useful so set them accordingly
	for (uint bStep = 0; bStep < _bSize; ++bStep) 									// go through each block step
	{
		this->chooseRandomFlips(); 													// set the random flip sites - it depends on a given implementation of the NQS
		this->applyFlipsT(); 														// flip the vector - use temporary vector tmpVec to store the flipped vector

#ifndef NQS_ANGLES_UPD
		// double proba = std::abs(this->pRatio(this->curVec_, this->tmpVec_));
		double proba = std::abs(std::exp(this->beta_ * this->logPRatio(this->curVec_, this->tmpVec_)));
#else
		// double proba = std::abs(this->pRatio(this->nFlip_)); 					// check the probability (choose to use the iterative update of presaved weights [the angles previously updated] or calculate ratio from scratch)
		double proba = std::abs(std::exp(this->beta_ * this->logPRatio(this->nFlip_)));
#endif
		proba = proba * proba;
		if (this->ran_->template random<double>() < proba) 							// we need to take into account the probability coming from the ratio of states (after and before the flip)
		{
			this->accepted_++; 														// increase the number of accepted flips
			this->applyFlipsC(); 													// update current state and vector when the flip has been accepted (the probability is higher than the random number)
			this->update(this->nFlip_); 											// update angles if needed
		}
		else
		{
			this->unupdate(); 														// set the vector back to normal (unflip)
			this->unapplyFlipsT();
		}
		this->total_++; 															// increase the total number of flips
	}

#ifndef NQS_USE_VEC_ONLY
	this->curState_ = BASE_TO_INT<u64>(this->curVec_, discVal_); 					// set the integer state			
#endif
#ifndef NQS_ANGLES_UPD
	this->setState(NQS_STATE, true); 												// Set the state again if angles update is disabled
#endif
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, blockSample, void, (uint, NQS_STATE_T, bool));
NQS_INST_CMB(double, std::complex<double>, blockSample, void, (uint, NQS_STATE_T, bool));
NQS_INST_CMB(std::complex<double>, double, blockSample, void, (uint, NQS_STATE_T, bool));
NQS_INST_CMB(std::complex<double>, std::complex<double>, blockSample, void, (uint, NQS_STATE_T, bool));

// ##########################################################################################################################################

// ########################################################## L O C   E N E R G Y ###########################################################

// ##########################################################################################################################################

/**
* @brief Computes the local energy kernel for the Neural Quantum State (NQS).
*
* This function calculates the local energy of the NQS using either OpenMP for parallelization
* or a custom threading mechanism, depending on the compilation flags. The local energy is 
* computed by iterating over all sites and summing up the contributions from each site (quantum state mode within a Hamiltonian).
* The function also handles the computation of the local energy for lower states if they are used as:
* E_loc += \sum _{s'} <s|psi_wl><psi_wl|s'> \times \frac{\psi _w(s')}{\psi _w(s)} \times \frac{\psi _wl(s)}{\psi _wl(s')} \times proba_wl(s', s)
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the energy.
* @tparam _stateType State type.
* @return The computed local energy of type _T - this is always casted to the correct type (double or complex).
*
* @note If NQS_USE_OMP is defined, OpenMP is used for parallelization. Otherwise, a custom 
*       threading mechanism is employed.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::locEnKernel()
{	
#ifdef NQS_USE_OMP
	{
		double energyR	= 0.0;
		double energyI	= 0.0;

#ifndef _DEBUG
		#pragma omp parallel for reduction(+ : energyR, energyI) num_threads(this->threadNum_)
#endif
		for (int site = 0; site < this->info_p_.nSites_; site++) 
		{
			auto energy	=	this->H_->locEnergy(NQS_STATE, 
												site, 
												std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
												this,
												std::placeholders::_1,
												std::placeholders::_2));
			energyR		+=	algebra::real(energy);
			energyI		+=	algebra::imag(energy);
		}
		return toType<_T>(energyR, energyI);
	}
#else
	{
		_T energy = 0.0;
		for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++) // run all threads
		{
			std::lock_guard<std::mutex> lock(this->threads_.kernels_[_thread].mutex);
			this->threads_.kernels_[_thread].flagThreadRun_	= true;
			this->threads_.kernels_[_thread].end_			= false;
			this->threads_.kernels_[_thread].cv.notify_one();
		}

		for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++) // wait for all threads
		{
			{
				std::unique_lock<std::mutex> lock(this->threads_.kernels_[_thread].mutex);
				this->threads_.kernels_[_thread].cv.wait(lock, [this, _thread] { return !this->threads_.kernels_[_thread].flagThreadRun_; });
			}
			energy += this->threads_.kernels_[_thread].kernelValue_;
		}

		// for the lower states - only if the lower states are used
		if (this->lower_states_.f_lower_size_ != 0) 
		{
			// set new projector (\sum _{s'} <s|psi_wl><psi_wl|s'>) = \sum _{s'} \frac{\psi _w(s')}{\psi _w(s)} \times \frac{\psi _wl(s)}{\psi _wl(s')} \times proba_wl(s', s)
			this->lower_states_.setProjector(NQS_STATE);

			_T _elower = 0.0;
#ifndef _DEBUG 
# 	pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
			for (int _low = 0; _low < this->lower_states_.f_lower.size(); _low++)
				_elower += this->lower_states_.collectLowerEnergy(_low);
			energy += _elower;													// add the lower states energy contribution (at the end)
		}

		return energy;
	}
#endif
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, locEnKernel, double, ());
NQS_INST_CMB(double, std::complex<double>, locEnKernel, std::complex<double>, ());
NQS_INST_CMB(std::complex<double>, double, locEnKernel, double, ());
NQS_INST_CMB(std::complex<double>, std::complex<double>, locEnKernel, std::complex<double>, ());

///////////////////////////////////////////////////////////////////////

#ifdef NQS_NOT_OMP_MT

/**
* @brief Kernel function to compute the local energy for a range of sites.
*
* This function is executed by a thread to compute the local energy for a specified range of sites (quantum state modes).
* The thread will wait until it is signaled to run, and will terminate if the kill flag is set.
* The computed local energy is stored in the thread's kernel value and the flag is set to indicate that the thread has finished.
* The flag is given in the form of a condition variable to notify other threads if needed.
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the local energy.
* @tparam _stateType State type.
* @param _start The starting index of the site range.
* @param _end The ending index of the site range.
* @param _threadNum The index of the thread executing this function.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::locEnKernel(uint _start, uint _end, uint _threadNum)
{
	while (!this->threads_.kernels_[_threadNum].flagThreadKill_)	// does not go in if the simulation is finished
	{
		{
			// acquire mutex lock as required by condition variable
			std::unique_lock<std::mutex> lock(this->threads_.kernels_[_threadNum].mutex);	
			// thread will suspend here and release the lock if the expression does not return true
			this->threads_.kernels_[_threadNum].cv.wait(lock, [this, _threadNum] { return this->threads_.kernels_[_threadNum].flagThreadRun_; });	

			if (this->threads_.kernels_[_threadNum].flagThreadKill_)
			{
				this->threads_.kernels_[_threadNum].end_ = true;
				break;
			}
		}

		// Process the work
		this->threads_.kernels_[_threadNum].kernelValue_ = 0.0;
		for (auto site = _start; site < _end; ++site)
			this->threads_.kernels_[_threadNum].kernelValue_ += algebra::cast<_T>(this->H_->locEnergy(NQS_STATE, site, this->pKernelFunc_));

		{
			std::lock_guard<std::mutex> lock(this->threads_.kernels_[_threadNum].mutex);
			this->threads_.kernels_[_threadNum].flagThreadRun_	= false;
			this->threads_.kernels_[_threadNum].end_			= true; 
		}
		this->threads_.kernels_[_threadNum].cv.notify_one(); 		// Notify waiting threads if needed
	}
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, locEnKernel, void, (uint, uint, uint));
NQS_INST_CMB(double, std::complex<double>, locEnKernel, void, (uint, uint, uint));
NQS_INST_CMB(std::complex<double>, double, locEnKernel, void, (uint, uint, uint));
NQS_INST_CMB(std::complex<double>, std::complex<double>, locEnKernel, void, (uint, uint, uint));

#endif

// ##########################################################################################################################################

// time evolution

/**
* @brief Computes the gradient over time for the Neural Quantum State (NQS) using either a standard method or Runge-Kutta method.
* This means that the time evolution of the state is computed by solving the Fubini Study equation.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the computation.
* @tparam _stateType State type.
* 
* @param _step Current step in the gradient computation.
* @param _dt Time step for the gradient computation.
* @param _currLoss Current loss value.
* @param _useRungeKutta Flag indicating whether to use the Runge-Kutta method.
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
void NQS<_spinModes, _Ht, _T, _stateType>::gradTime(size_t _step, double _dt, _T _currLoss, const bool _useRungeKutta)
{
	bool _inversionSuccess 		= false;
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
		this->solver_->setReg(this->info_p_.sreg_);											// set the regularization						
		this->solver_->solve(this->derivativesCentered_, this->derivativesCenteredH_, 		// S and S+ matrices
							_multiplier * this->F_, 										// b
							nullptr, //step <= 1 ? nullptr : &this->dF_, 					// x0
							this->precond_);												// preconditioner
		if (!_useRungeKutta)
		{
			_inversionSuccess = this->solver_->isConverged();
			this->dF_ = _dt * this->solver_->solution();									// get the solution
		} else {
			this->RK_.k1 = this->solver_->solution();										// get the solution for the first step
			// solve for next
			this->solver_->solve(this->derivativesCentered_, this->derivativesCenteredH_, 
								this->F_ + (_dt / 2.0 * this->RK_.k1), nullptr, this->precond_);
			this->RK_.k2 = this->solver_->solution();										// get the solution for the second step
			// solve for next
			this->solver_->solve(this->derivativesCentered_, this->derivativesCenteredH_, 
								this->F_ + (_dt / 2.0 * this->RK_.k2), nullptr, this->precond_);
			this->RK_.k3 = this->solver_->solution();										// get the solution for the third step	
			// solve for next
			this->solver_->solve(this->derivativesCentered_, this->derivativesCenteredH_, 
								this->F_ + (_dt * this->RK_.k3), nullptr, this->precond_);
			this->RK_.k4 = this->solver_->solution();										// get the solution for the fourth step
			// calculate the final solution
			this->dF_ = (_dt / 6.0) * (this->RK_.k1 + 2.0 * this->RK_.k2 + 2.0 * this->RK_.k3 + this->RK_.k4);	
		}
	} else {
		LOGINFO("Solver is not set. Cannot perform the inversion.", LOG_TYPES::ERROR, 1);
		throw std::runtime_error("Solver is not set. Cannot perform the inversion.");
	}

#endif
    this->updateWeights_ = _inversionSuccess;
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, gradTime, void, (size_t, double, double, const bool));
NQS_INST_CMB(double, std::complex<double>, gradTime, void, (size_t, double, std::complex<double>, const bool));
NQS_INST_CMB(std::complex<double>, double, gradTime, void, (size_t, double, double, const bool));
NQS_INST_CMB(std::complex<double>, std::complex<double>, gradTime, void, (size_t, double, std::complex<double>, const bool));

// ##########################################################################################################################################