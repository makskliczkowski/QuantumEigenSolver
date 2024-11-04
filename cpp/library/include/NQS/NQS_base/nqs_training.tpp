#include "./nqs_optimize.tpp"
#include "nqs_definitions_base.h"

// ##########################################################################################################################################

// ############################################################ T R A I N I N G #############################################################

// ##########################################################################################################################################

#include <cmath>
#include <functional>
#include <utility>

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
	TIMER_CREATE(_timer);

	// save all average weights for covariance matrix
	arma::Col<_T> meanEn(_par.MC_sam_, arma::fill::zeros);
	arma::Col<_T> stdEn(_par.MC_sam_, arma::fill::zeros);
	// history of energies (for given weights) - here we save the local energies at each block
	arma::Col<_T> En(_par.nblck_, arma::fill::zeros);
	// set the random state at the begining and the number of flips
	{
		this->setRandomState();
		this->setRandomFlipNum(_par.nFlip);
	}

	// go through the Monte Carlo steps
	uint i = 1;
	for (i = 1; i <= _par.MC_sam_; ++i)
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
#ifndef _DEBUG 
# 	pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
			for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++)
				this->lower_states_.ratios_excited_[_low](_taken) = this->lower_states_.collectExcitedRatios(_low, NQS_STATE);
		}

		// collect the average for the lower states and collect the same for the lower states with this ansatz - for the gradient calculation
#ifndef _DEBUG 
# 	pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
		for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++) 
			this->lower_states_.collectLowerRatios(_low);
		
		MonteCarlo::blockmean(En, _par.bsize_, &meanEn(i - 1), &stdEn(i - 1));			// save the mean energy

		// calculate the final update vector - either use the stochastic reconfiguration or the standard gradient descent !TODO: implement optimizers
		TIMER_START_MEASURE(this->gradFinal(En, i, meanEn(i - 1)), (i % this->pBar_.percentageSteps == 0), _timer, STR(i));
		// LOGINFO(_t, VEQ(i-1) + " --- " + VEQP(meanEn(i-1), 4), 3);

		if (this->updateWeights_)
			this->updateWeights(); // finally, update the weights with the calculated gradient (force) [can be done with the stochastic reconfiguration or the standard gradient descent] - implementation specific!!!

		{
			// update the progress bar
			auto best 				= this->info_p_.best();
			const std::string _prog = "PROGRESS NQS: E(" + STR(i - 1) + "/" + STR(_par.MC_sam_) + ")=" + STRPS(meanEn(i - 1), 4) + ". " + VEQPS(best, 4) + ". eta=" + STRPS(this->info_p_.lr_, 4) + ". reg=" + STRPS(this->info_p_.sreg_, 4);
			PROGRESS_UPD_Q(i, this->pBar_, _prog, !quiet);
			
			this->updateWeights_ = !this->info_p_.stop(i, meanEn(i - 1)) && this->updateWeights_;
#ifdef NQS_SAVE_WEIGHTS
			if (i % this->pBar_.percentageSteps == 0 || !this->updateWeights_)  
				this->saveWeights(_par.dir + NQS_SAVE_DIR, "weights_" + STR(this->lower_states_.f_lower_size_) + ".h5");
#endif	
			if (!this->updateWeights_) {
				LOGINFO("Stopping at " + STR(i) + " iteration with last value: " + STRPS(meanEn(i - 1), 4), LOG_TYPES::WARNING, 1);
				break;
			}
		}
	}
	LOGINFO(_t, "NQS_EQ_" + STR(this->lower_states_.f_lower_size_), 1);
	return std::make_pair(meanEn.subvec(0, i - 2), stdEn.subvec(0, i - 2));
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
	// this->setRandomFlipNum(_par.nFlip);

	// allows to calculate the probability of the operator (for operator measurements)
	// std::function<_T(const NQSS& _v)> opFun = [&](const NQSS& v) { return this->pRatio(v); };
	
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