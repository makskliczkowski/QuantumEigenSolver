#include "../../include/NQS/nqs_final.hpp"

// ##########################################################################################################################################

// ############################################################ S A M P L I N G #############################################################

// ##########################################################################################################################################

/** 
* @brief Block updates the current state according to Metropolis-Hastings algorithm. The block size is chosen so that
* it gives the given statistics of the averages found.
* @param _bSize the size of the correlation block; how many flips one needs to do in order to get rid of the MC correlations - bin size 
* @param _start the state to start from
* @param _nFlip number of flips in a single step
* @param _therm whether the thermalization is needed
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
		double proba = std::abs(this->pRatio(this->curVec_, this->tmpVec_));
#else
		double proba = std::abs(this->pRatio(this->nFlip_)); 						// check the probability (choose to use the iterative update of presaved weights [the angles previously updated] or calculate ratio from scratch)
#endif
		proba = proba * proba;
		if (this->ran_.template random<double>() < proba) 							// we need to take into account the probability coming from the ratio of states (after and before the flip)
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
	this->curState_ = BASE_TO_INT<u64>(this->curVec_, discVal_); // set the integer state			
#endif

	// Set the state again if angles update is disabled
#ifndef NQS_ANGLES_UPD
	this->setState(NQS_STATE, true); // set the state again									
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
* @brief Calculate the local energy depending on the given Hamiltonian - kernel with OpenMP is used
* when the omp pragma NQS_USE_OMP is set or multithreading is not used, otherwise threadpool is used
* to calculate the local energies.
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
			energy += _elower;
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
* @brief Allows to run a thread pool based on the condition that all threads wait for a mutex to further run the program.
* The threads are killed when the program is finished. The function calculates the local energy for a given thread.
* @param _start starting site for a given thread
* @param _end ending site for a given thread
* @param _threadNum number of the thread currently run
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

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradTime(size_t _step, double _dt, _T _currLoss)
{
	bool _inversionSuccess 		= false;
	const _T _multiplier 		= algebra::cast<_T>(I) * _dt;
	if (this->info_p_.sreg_ > 0) 
		this->covMatrixReg(_step);

#ifdef NQS_USESR_MAT_USED
	assert(this->S_.empty() == false && this->F_.empty() == false && "Covariance matrix S and gradient vector F must be set before the inversion.");

	{
		// regularize the covariance matrix before inverting it (if needed and set)

		// calculate the pseudoinverse
		int _attempts 			= 0;
		double _regFactor		= this->info_p_.sreg_ > 0 ? this->info_p_.sreg_ : 1e-5;

		while (!_inversionSuccess && _attempts < NQS_SREG_ATTEMPTS)
		{
			try {
				if (this->info_p_.pinv_ > 0.0)
					this->dF_ = _multiplier * (arma::pinv(this->S_, this->info_p_.pinv_) * this->F_);
				else if (this->info_p_.pinv_ == 0.0)
					this->dF_ = _multiplier * (arma::pinv(this->S_) * this->F_);
				else 
					this->dF_ = _multiplier * arma::solve(this->S_, this->F_, arma::solve_opts::likely_sympd);
				
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
	assert(this->derivativesCentered_.empty() == false && this->derivativesCenteredH_.empty() == false && "Centered derivatives F_k and F_k^T must be set before the inversion.");
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
		this->dF_ = _multiplier * this->solver_->solution();								// get the solution
	} 
	else {
		// !DEPRECATED - use solver class instead
		this->dF_ = _multiplier * algebra::Solvers::FisherMatrix::solve<_T>(
										this->info_p_.solver_,													// choose the solver type 
										this->derivativesCentered_,     										// Ensure this matches the type expected by _gramMatrix
										this->derivativesCenteredH_,											// This should also match arma::Col<_T>
										this->F_,               												// This should be of type arma::Col<_T>
										nullptr, //step <= 1 ? nullptr : &this->dF_,							// This should also match arma::Col<_T>
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

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, gradTime, void, (size_t, double, double));
NQS_INST_CMB(double, std::complex<double>, gradTime, void, (size_t, double, std::complex<double>));
NQS_INST_CMB(std::complex<double>, double, gradTime, void, (size_t, double, double));
NQS_INST_CMB(std::complex<double>, std::complex<double>, gradTime, void, (size_t, double, std::complex<double>));