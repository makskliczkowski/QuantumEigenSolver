#include "./nqs_optimize.tpp"

// ##########################################################################################################################################

// ############################################################ T R A I N I N G #############################################################

// ##########################################################################################################################################

/*
* @brief Perform single training of the NQS.
* @param mcSteps Monte Carlo steps to be used in the training - this is an outer loop for the training (number of iterations)
* @param nThrm Monte Carlo steps spent for thermalizing the system - burn-in
* @param nBlck	number of blocks to be used for calculating the covariance matrix - number of iterations for the gradient calculation (single iteration step)
* @param bSize number of update steps within a single block - size of the block (number of flip trials (global))
* @param nFlip number of flips for a single update of a state (default is 1)
* @param quiet wanna talk? (default is false)
* @param _t timepoint for timestamping the training
* @param progPrc progress percentage to be displayed in the progress bar
*/
#include <functional>
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline arma::Col<_T> NQS<_spinModes, _Ht, _T, _stateType>::train(const NQS_train_t& _par,
																bool quiet,
																clk::time_point _t,
																uint progPrc)
{
	{
		// make the pbar!
		this->pBar_ = pBar(progPrc, _par.MC_sam_);

		// set the info about training
		_par.hi();

		// set the derivatives to fullsize
		this->derivativesReset(_par.nblck_);

		// set the size of the containers for the lower states
		this->lower_states_.setDerivContSize(_par.nblck_, _par.nblck_);
	}
	TIMER_CREATE(_timer);

	// save all average weights for covariance matrix
	arma::Col<_T> meanEn(_par.MC_sam_, arma::fill::zeros);
	// history of energies (for given weights) - here we save the local energies at each block
	arma::Col<_T> En(_par.nblck_, arma::fill::zeros);
	// set the random state at the begining and the number of flips
	{
		this->setRandomState();
		this->setRandomFlipNum(_par.nFlip);
	}

	// go through the Monte Carlo steps
	for (uint i = 1; i <= _par.MC_sam_; ++i)
	{
		// set the random state at the begining
		if (_par.MC_th_ > 0)
			this->setRandomState();

		// thermalize
		this->blockSample(_par.MC_th_, NQS_STATE, false);

		// iterate blocks - this ensures the calculation of a stochastic gradient constructed within the block
		for (uint _taken = 0; _taken < _par.nblck_; ++_taken) {

			// sample them!
			this->blockSample(_par.bsize_, NQS_STATE, false);

			// calculate the gradient at each point of the iteration! - this is implementation specific!!!
			this->grad(this->curVec_, _taken);

			// local energy - stored at each point within the estimation of the gradient (stochastic)
			En(_taken) = this->locEnKernel();

			// calculate the excited states overlaps for the gradient - if used
			for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++)
				this->lower_states_.ratios_excited_[_low](_taken) = std::exp(this->lower_states_.ansatzlog(this->curVec_, _low) - this->ansatzlog(this->curVec_));
		}

		// collect the average for the lower states and collect the same for the lower states with this ansatz - for the gradient calculation
		for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++)
			this->lower_states_.collectRatiosLower(_low, [&](Operators::_OP_V_T_CR _v) { return this->ansatzlog(_v); });
		
		// calculate the final update vector - either use the stochastic reconfiguration or the standard gradient descent !TODO: implement optimizers
		TIMER_START_MEASURE(this->gradFinal(En), (i % this->pBar_.percentageSteps == 0), _timer, STR(i));

		// finally, update the weights with the calculated gradient (force) [can be done with the stochastic reconfiguration or the standard gradient descent] - implementation specific!!!=
		this->updateWeights();
		
		// save the mean energy
		meanEn(i - 1) = arma::mean(En);
		LOGINFO(VEQ(meanEn(i - 1)), LOG_TYPES::TRACE, 1);


		// update the progress bar
		PROGRESS_UPD_Q(i, this->pBar_, "PROGRESS NQS", !quiet);
		
#ifdef NQS_SAVE_WEIGHTS
		{
		if (i % this->pBar_.percentageSteps == 0)  
			this->saveWeights(_par.dir + NQS_SAVE_DIR, "weights.h5");
		}
#endif
	}
	LOGINFO(_t, "NQS_EQ", 1);
	return meanEn;
}

// ##########################################################################################################################################

// ########################################################## C O L L E C T I O N ###########################################################

// ##########################################################################################################################################

/*
* @brief Collects the samples for the given number of Monte Carlo steps. The samples are collected in the form of the measurements
* of the operator. The operator is given as a function that calculates the probability of the operator.
* @param _par parameters for the training
* @param _meas measurement object to store the measurements
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>:: collect(const NQS_train_t& _par, NQSAv::MeasurementNQS<_T>& _meas)
{
	// set the random state at the begining
	this->setRandomFlipNum(_par.nFlip);

	// allows to calculate the probability of the operator (for operator measurements)
	std::function<_T(const NQSS& _v)> opFun = [&](const NQSS& v) { return this->pRatio(v); };
	
	// go through the number of samples to be collected
	for (uint i = 1; i <= _par.MC_sam_; ++i)
	{	
		// random flip
		if (_par.MC_th_ > 0)
			this->setRandomState();

		// remove autocorrelations and thermalize
		this->blockSample(_par.MC_th_, NQS_STATE, false);

		// iterate blocks - allows to collect samples outside of the block
		for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 
		{
			// sample them!
			this->blockSample(_par.bsize_, NQS_STATE, false);
			
			// measure 
			_meas.measure(this->curVec_, opFun);
		}
		// normalize the measurements - this also creates a new block of measurements
		_meas.normalize(_par.nblck_);												
	}
}

/*
* @brief Collects the samples for the given number of Monte Carlo steps. The samples are collected in the form of the measurements
* of the operator. The operator is given as a function that 
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::collect(const NQS_train_t& _par, 
														const Operators::OperatorNQS<_T>& _opG,
														Operators::Containers::OperatorContainer<_T>& _cont)
{
	// set the random state at the begining
	this->setRandomFlipNum(_par.nFlip);

	// allows to calculate the probability of the operator (for operator measurements)
	std::function<_T(const NQSS& _v)> opFun = [&](const NQSS& v) { return this->pRatio(v); };
	
	// go through the number of samples to be collected
	for (uint i = 1; i <= _par.MC_sam_; ++i)
	{	
		// random flip
		if (_par.MC_th_ > 0)
			this->setRandomState();

		// remove autocorrelations and thermalizes
		this->blockSample(_par.MC_th_, NQS_STATE, false);

		// iterate blocks - allows to collect samples outside of the block
		for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 
		{
			// sample them!
			this->blockSample(_par.bsize_, NQS_STATE, false);
			
			// measure 
			NQSAv::MeasurementNQS<_T>::measure(NQS_STATE, _opG, opFun, _cont);
		}
		// normalize the measurements - this also creates a new block of measurements
		NQSAv::MeasurementNQS<_T>::normalize(_par.nblck_, _cont);												
	}
}

/*
* @brief Collects the samples for the given number of Monte Carlo steps. The samples are collected in the form of the measurements
* of the operator. The operator is given as a function that calculates the probability of the operator.
* @param _par parameters for the training
* @param _meas measurement object to store the measurements
* @param _collectEn collect the energy as well
* @param _t timepoint for timestamping the training
* @param quiet wanna talk? (default is false)
* @returns the mean energy of the system
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline arma::Col<_T> NQS<_spinModes, _Ht, _T, _stateType>::collect(	const NQS_train_t& _par,
																	bool quiet,
																	clk::time_point _t,
																	NQSAv::MeasurementNQS<_T>& _meas,
																	bool _collectEn)
{																							
	{
		this->pBar_	= pBar(20, _par.MC_sam_);
		_par.hi("Collect: ");
	}
	TIMER_CREATE(_timer);

	arma::Col<_T> meanEn, En;
	if (_collectEn) {
		meanEn 	= arma::Col<_T>(_par.MC_sam_, arma::fill::zeros);
		En 		= arma::Col<_T>(_par.nblck_, arma::fill::zeros);
	}

	// set the random state at the begining
	this->setRandomFlipNum(_par.nFlip);

	// allows to calculate the probability of the operator (for operator measurements)
	std::function<_T(const NQSS& _v)> opFun = [&](const NQSS& v) { return this->pRatio(v); };
	
	// go through the number of samples to be collected
	for (uint i = 1; i <= _par.MC_sam_; ++i)
	{
		// random flip
		this->setRandomState();

		// remove autocorrelations
		this->blockSample(_par.MC_th_, NQS_STATE, false);

		// iterate blocks - allows to collect samples outside of the block
		for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 
		{
			// sample them!
			this->blockSample(_par.bsize_, NQS_STATE, false);

			if (_collectEn) En(_taken) = this->locEnKernel();
			
			// one can calculate the local energy here (either of the ground state or the excited state)
			// collect other										
			TIMER_START_MEASURE(_meas.measure(this->curVec_, opFun), (i % this->pBar_.percentageSteps == 0 && _taken == 0), _timer, STR(i)); 	
			//TIMER_START_MEASURE(_meas.measure(BASE_TO_INT<u64>(this->curVec_, this->discVal_), opFun), (i % this->pBar_.percentageSteps == 0 && _taken == 0), _timer, STR(i)); 	
		}

		_meas.normalize(_par.nblck_);												// normalize the measurements
		meanEn(i - 1) = arma::mean(En); 											// save the mean energy
		PROGRESS_UPD_Q(i, this->pBar_, "PROGRESS NQS", !quiet); 					// update the progress bar

	}
	LOGINFO(_t, "NQS_COLLECTION", 1);
	return meanEn;
}

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::collect_ratio(const NQS_train_t& _par, std::function<_T(const NQSS&)> _f,
								arma::Col<_T>& _container)
{
	// set the random state at the begining
	this->setRandomFlipNum(_par.nFlip);

	// go through the number of samples to be collected
	_T _values = 0.0;
	for (uint i = 1; i <= _par.MC_sam_; ++i)
	{	
		// random flip
		if (_par.MC_th_ > 0)
			this->setRandomState();

		// remove autocorrelations and thermalizes
		this->blockSample(_par.MC_th_, NQS_STATE, false);

		// iterate blocks - allows to collect samples outside of the block
		for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 
		{
			// sample them!
			this->blockSample(_par.bsize_, NQS_STATE, false);
			
			// calculate f(s) / \psi(s)
			_values += std::exp(_f(this->curVec_) - this->ansatz(this->curVec_));
		}
		_container(i - 1) = _values / (double)_par.nblck_;												
	}
}