#include "./nqs_fermions.tpp"

// ##########################################################################################################################################

// ############################################################ S A M P L I N G #############################################################

// ##########################################################################################################################################

/*
* @brief Block updates the current state according to Metropolis-Hastings algorithm. The block size is chosen so that
* it gives the given statistics of the averages found.
* @param _bSize the size of the correlation block; how many flips one needs to do in order to get rid of the MC correlations - bin size 
* @param _start the state to start from
* @param _nFlip number of flips in a single step
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::blockSample(uint _bSize, NQS_STATE_T _start, bool _therm)
{
	// check whether we should set a state again or thermalize the whole process (applies on integer state)
	// Set state based on whether thermalization is required or _start differs from current state
	if (_therm 
#ifndef NQS_USE_VEC_ONLY
		|| _start != this->curState_
#endif
		) this->setState(_start, _therm);

	// set the temporary state - the vectors are usefull so set them accordingly
	this->tmpVec_ = this->curVec_;

	for (uint bStep = 0; bStep < _bSize; ++bStep) // go through each block step
	{
		this->chooseRandomFlips(); 	// set the random flip sites - it depends on a given implementation of the NQS
		this->applyFlipsT();		// flip the vector - use temporary vector tmpVec to store the flipped vector

		// check the probability (choose to use the iterative update of presaved weights [the angles previously updated] or calculate ratio from scratch)
#ifndef NQS_ANGLES_UPD
		double proba = std::abs(this->pRatio(this->curVec_, this->tmpVec_));
#else
		double proba = std::norm(this->pRatio(this->nFlip_));
#endif
		// we need to take into account the probability comming from the ratio of states (after and before the flip)
		if (this->ran_.template random<float>() < proba)
		{
			// update current state and vector when the flip has been accepted (the probability is higher than the random number)
			this->applyFlipsC();
			// update angles if needed
			this->update(this->nFlip_);
		}
		else
		{
			// set the vector back to normal (unflip)
			this->unupdate();
			this->unapplyFlipsT();
		}
	}

	// set the current integer state
#ifndef NQS_USE_VEC_ONLY
	this->curState_ = BASE_TO_INT<u64>(this->curVec_, discVal_);
#endif

	// Set the state again if angles update is disabled
#ifndef NQS_ANGLES_UPD
	this->setState(NQS_STATE, true);
#endif
}

// ##########################################################################################################################################

// ########################################################## L O C   E N E R G Y ###########################################################

// ##########################################################################################################################################

/*
* @brief Calculate the local energy depending on the given Hamiltonian - kernel with OpenMP is used
* when the omp pragma NQS_USE_OMP is set or multithreading is not used, otherwise threadpool is used
* to calculate the local energies.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T NQS<_spinModes, _Ht, _T, _stateType>::locEnKernel()
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
			std::function<_T(const NQSS&)> _pratio 	= [&](const NQSS& _v) { return this->pRatio(_v); };
			this->lower_states_.setProjector(this->info_p_.nVis_, NQS_STATE, _pratio);

			_T _elower = 0.0;
			for (int _low = 0; _low < this->lower_states_.f_lower.size(); _low++)
				_elower += this->lower_states_.collectLowerEnergy(_low);
			// std::cout << "Lower energy: " << algebra::real(_elower) << ", " << algebra::imag(_elower) << std::endl;
			// std::cout << "Energy: " << algebra::real(energy) << ", " << algebra::imag(energy) << std::endl;
			energy += _elower;
		}

		return energy;
	}
#endif
}

///////////////////////////////////////////////////////////////////////

#ifdef NQS_NOT_OMP_MT
/*
* @brief Allows to run a thread pool based on the condition that all threads wait for a mutex to further run the program.
* The threads are killed when the program is finished. The function calculates the local energy for a given thread.
* @param _start starting site for a given thread
* @param _end ending site for a given thread
* @param _threadNum number of the thread currently run
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::locEnKernel(uint _start, uint _end, uint _threadNum)
{
	while (!this->threads_.kernels_[_threadNum].flagThreadKill_)	// does not go in if the simulation is finished
	{
		// wait for the lock to end
		{
			// aquire mutex lock as required by condition variable
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
		{
			this->threads_.kernels_[_threadNum].kernelValue_ += algebra::cast<_T>(this->H_->locEnergy(NQS_STATE,
																							site, 
																							std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
																							this,
																							std::placeholders::_1,
																							std::placeholders::_2)));
		}

		// lock again
		{
			std::lock_guard<std::mutex> lock(this->threads_.kernels_[_threadNum].mutex);
			this->threads_.kernels_[_threadNum].flagThreadRun_	= false;
			this->threads_.kernels_[_threadNum].end_				= true; 
		}
		this->threads_.kernels_[_threadNum].cv.notify_one(); // Notify waiting threads if needed
	}
}
#endif