#include "./nqs_optimize.tpp"
#include "nqs_definitions_base.h"

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