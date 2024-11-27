#include "./nqs_optimize.tpp"
#include "nqs_definitions_base.h"

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