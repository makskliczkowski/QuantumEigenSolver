#include "./nqs_optimize.tpp"
#include "nqs_definitions_base.h"

// ##########################################################################################################################################

// ############################################################ T R A I N I N G #############################################################

// ##########################################################################################################################################

#include <cmath>
#include <functional>
#include <utility>

/*
* @brief Checks the stopping condition for the training and eventually stops the training. Also, saves the weights if needed.
* @note This function is called at the end of each iteration in the training loop.
* @param i current iteration
* @param _par parameters for the training
* @param _currLoss current loss value
* @param _quiet quiet mode
* @returns whether the training should be stopped
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool NQS<_spinModes, _Ht, _T, _stateType>::trainStop(size_t i, const NQS_train_t& _par, _T _currLoss, bool _quiet)
{
	auto best 				= this->info_p_.best();					// update the best value
	const std::string _prog = "PROGRESS NQS: E(" 	+ STR(i - 1) 	+ "/" + 
								STR(_par.MC_sam_) 	+ ")=" 			+ STRPS(_currLoss, 4) 			+ ". " + 
								VEQPS(best, 4) 		+ ". eta=" 		+ STRPS(this->info_p_.lr_, 4) 	+ 
								". reg=" 			+ STRPS(this->info_p_.sreg_, 4);
	PROGRESS_UPD_Q(i, this->pBar_, _prog, !_quiet);					// update the progress bar
	
	this->updateWeights_ 	= !this->info_p_.stop(i, _currLoss) && this->updateWeights_;
#ifdef NQS_SAVE_WEIGHTS
	if (i % this->pBar_.percentageSteps == 0 || !this->updateWeights_)  
		this->saveWeights(_par.dir + NQS_SAVE_DIR, "weights_" + STR(this->lower_states_.f_lower_size_) + ".h5");
#endif

	if (!this->updateWeights_) {									// if the training should be stopped
		LOGINFO("Stopping at " + STR(i) + " iteration with last value: " + STRPS(_currLoss, 4), LOG_TYPES::WARNING, 1);
		return true;
	}
	return false;
}

// ##########################################################################################################################################

/**
* @brief Perform single training step of the NQS. This function is called in the training loop. It samples the states, calculates the gradient, and updates the weights.
* @param i current iteration 
* @param En vector of local energies
* @param meanEn vector of mean energies
* @param stdEn vector of standard deviations
* @param _par parameters for the training
* @param quiet quiet mode
* @param _timer timer for the training
* @returns whether the training should be stopped
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool NQS<_spinModes, _Ht, _T, _stateType>::trainStep(size_t i, 
															arma::Col<_T>& En,
															arma::Col<_T>& meanEn, 
															arma::Col<_T>& stdEn, 
															const NQS_train_t& _par, const bool quiet, Timer& _timer)
{
	if (_par.MC_th_ > 0) 
		this->setRandomState();									// set the random state at the begining
	this->blockSample(_par.MC_th_, NQS_STATE, false);			// thermalize

	for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 		// iterate blocks - this ensures the calculation of a stochastic gradient constructed within the block
	{		
		this->blockSample(_par.bsize_, NQS_STATE, false);		// sample them using the local Metropolis sampling
		this->grad(this->curVec_, _taken);						// calculate the gradient at each point of the iteration! - this is implementation specific!!!
		En(_taken) = this->locEnKernel();						// local energy - stored at each point within the estimation of the gradient (stochastic)

		if (this->lower_states_.f_lower_size_ > 0)				// calculate the excited states overlaps for the gradient - if used
		{
#ifndef _DEBUG 
# 	pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
		for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++) // calculate the excited states overlaps for the gradient - if used
			this->lower_states_.ratios_excited_[_low](_taken) = this->lower_states_.collectExcitedRatios(_low, NQS_STATE);
		}
	}

	if (this->lower_states_.f_lower_size_ > 0)					// collect the average for the lower states and collect the same for the lower states with this ansatz - for the gradient calculation
	{
#ifndef _DEBUG 
# 	pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
		for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++)  		// collect the average for the lower states and collect the same for the lower states with this ansatz - for the gradient calculation
			this->lower_states_.collectLowerRatios(_low);
	}
	MonteCarlo::blockmean(En, _par.bsize_, &meanEn(i - 1), &stdEn(i - 1)); 	// save the mean energy
	TIMER_START_MEASURE(this->gradFinal(En, i, meanEn(i - 1)), (i % this->pBar_.percentageSteps == 0), _timer, STR(i)); // calculate the final update vector - either use the stochastic reconfiguration or the standard gradient descent

	// LOGINFO(_t, VEQ(i-1) + " --- " + VEQP(meanEn(i-1), 4), 3);

	if (this->updateWeights_)
		this->updateWeights(); 									// finally, update the weights with the calculated gradient (force) [can be done with the stochastic reconfiguration or the standard gradient descent] - implementation specific!!!

	if (this->trainStop(i, _par, meanEn(i - 1), quiet))
		return true;
	return false;
}


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
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline std::pair<arma::Col<_T>, arma::Col<_T>> NQS<_spinModes, _Ht, _T, _stateType>::train(const NQS_train_t& _par,
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
		this->lower_states_.setDerivContSize(_par.nblck_);
	}

	Timer _timer;															// timer for the training
	arma::Col<_T> meanEn(_par.MC_sam_, arma::fill::zeros);					// here we save the mean energy
	arma::Col<_T> stdEn(_par.MC_sam_, arma::fill::zeros);					// here we save the standard deviation of the energy
	arma::Col<_T> En(_par.nblck_, arma::fill::zeros);						// history of energies (for given weights) - here we save the local energies at each block
	this->setRandomState();													// set the random state at the begining and the number of flips
	this->setRandomFlipNum(_par.nFlip);										// set the random state at the begining and the number of flips

	// go through the Monte Carlo steps
	uint i = 1;
	for (i = 1; i <= _par.MC_sam_; ++i) {
		if (this->trainStep(i, En, meanEn, stdEn, _par, quiet, _timer)) 	// perform the training step
			break;
	}

	LOGINFO(_t, "NQS_EQ_" + STR(this->lower_states_.f_lower_size_), 1);
	if (i > 2) 
		return std::make_pair(meanEn.subvec(0, i - 2), stdEn.subvec(0, i - 2));
	return std::make_pair(meanEn, stdEn);
}

// ##########################################################################################################################################

// ########################################################## C O L L E C T I O N ###########################################################

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T NQS<_spinModes, _Ht, _T, _stateType>::collect(const NQS_train_t& _par, 
														Operators::OperatorNQS<_T>& _opG, v_1d<_T>* _opvals, v_1d<_T>* _energies, bool quiet, clk::time_point _t)
{
	if (!_opvals)
		throw std::runtime_error("NQS::collect: _opvals is not initialized!");

	for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 
	{
		this->blockSample(_par.bsize_, NQS_STATE, false);			// sample them to calculate the operator and the energy
		
		if (_energies)
			_energies->push_back(this->locEnKernel());				// local energy - stored at each point within the estimation of the gradient (stochastic)

		const auto _val = _opG(NQS_STATE, this->pRatioFunc_);	// calculate the operator value
		_opvals->push_back(_val);						// calculate the operator value
	}
	if (_energies)
		return std::accumulate(_energies->begin(), _energies->end(), (_T)0.0) / static_cast<_T>(_energies->size());
	return 0.0;
}

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
			_meas.measure(this->curVec_, this->pRatioFunc_);
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
	// this->setRandomFlipNum(_par.nFlip);

	// allows to calculate the probability of the operator (for operator measurements)
	// std::function<_T(const NQSS& _v)> opFun = [&](const NQSS& v) { return this->pRatio(v); };
	
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
			// auto [_, _val] = 
			NQSAv::MeasurementNQS<_T>::measure(NQS_STATE, _opG, this->pRatioFunc_, _cont);
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
																	bool _collectEn,
																	uint progPrc)
{																							
	arma::Col<_T> meanEn, En;
	{
		this->pBar_	= pBar(progPrc, _par.MC_sam_);
		_par.hi("Collect: ");

		if (_collectEn) {
			meanEn 	= arma::Col<_T>(_par.MC_sam_, arma::fill::zeros);
			En 		= arma::Col<_T>(_par.nblck_, arma::fill::zeros);
		}
	}
	TIMER_CREATE(_timer);

	// set the random state at the begining
	this->setRandomFlipNum(_par.nFlip);

	// go through the number of samples to be collected
	for (uint i = 1; i <= _par.MC_sam_; ++i)
	{
		// random flip
		if (_par.MC_th_ > 0)
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
			_meas.measure(NQS_STATE, this->pRatioFunc_);
			// TIMER_START_MEASURE(_meas.measure(NQS_STATE, this->pRatioFunc_), (i % this->pBar_.percentageSteps == 0 && _taken == 0), _timer, STR(i));
			//TIMER_START_MEASURE(_meas.measure(BASE_TO_INT<u64>(this->curVec_, this->discVal_), opFun), (i % this->pBar_.percentageSteps == 0 && _taken == 0), _timer, STR(i)); 	
		}

		_meas.normalize(_par.nblck_);												// normalize the measurements
		MonteCarlo::blockmean(En, _par.nblck_, &meanEn(i-1)); 						// save the mean energy
		PROGRESS_UPD_Q(i, this->pBar_, "PROGRESS NQS", !quiet); 					// update the progress bar

	}
	LOGINFO(_t, "NQS_COLLECTION", 1);
	return meanEn;
}

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::collect_ratio(const NQS_train_t& _par, std::function<_T(const NQSS&)> _f, arma::Col<_T>& _container)
{
	// set the random state at the begining
	// this->setRandomFlipNum(_par.nFlip);
	_container.zeros();

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
			
			// calculate f(s) / \psi(s)
			const _T _top 		= _f(NQS_STATE);
#ifdef NQS_LOWER_RATIO_LOGDIFF
			const _T _bottom 	= this->ansatzlog(NQS_STATE);
			_container(_taken) 	+= std::exp(_top - _bottom);
#else 
			const _T _bottom 	= this->ansatz(NQS_STATE);
			_container(_taken) 	+= _top / _bottom;
#endif
		}									
	}
	// normalize the measurements - this also creates a new block of measurements - divide by the number of samples
	if (_par.MC_sam_ > 1)
		_container /= _par.MC_sam_;
}

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::collect_ratio(const NQS_train_t& _par, NQS<_spinModes, _Ht, _T, _stateType>* other, arma::Col<_T>& _container)
{
	// set the random state at the begining
	// this->setRandomFlipNum(_par.nFlip);
	_container.zeros();

	if (!other)
		return;

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

			auto _val 			= this->ansatz_ratio(NQS_STATE, other);
			_container(_taken) += _val;
		}									
	}
	// normalize the measurements - this also creates a new block of measurements - divide by the number of samples
	if (_par.MC_sam_ > 1)
		_container /= _par.MC_sam_;
}