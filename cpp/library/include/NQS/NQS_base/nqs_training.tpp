#include "./nqs_optimize.tpp"

// ##########################################################################################################################################
namespace NQS_NS
{
	/**
	* @brief Collects the samples for the given number of Monte Carlo steps. The samples are collected in the form of the measurements
	* of the operator. The operator is given as a function that calculates the probability of the operator.
	* @param _par parameters for the training
	* @param _meas measurement object to store the measurements
	*/
	template <uint _spinModes, typename _Ht, typename _T, class _stateType>
	inline void NQS<_spinModes, _Ht, _T, _stateType>:: collect(const MonteCarlo::MCS_train_t& _par, NQSAv::MeasurementNQS<_T>& _meas)
	{
		for (uint i = 1; i <= _par.MC_sam_; ++i)
		{	
			// random flip
			if (_par.MC_th_ > 0)
				this->setRandomState();

			// remove autocorrelations and thermalize
			this->blockSample<false>(_par.MC_th_, NQS_STATE);

			// iterate blocks - allows to collect samples outside of the block
			for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 
			{
				// sample them!
				this->blockSample<false>(_par.bsize_, NQS_STATE);
				
				// measure 
				_meas.measure(this->curVec_, this->pRatioFunc_);
			}
			// normalize the measurements - this also creates a new block of measurements
			_meas.normalize(_par.nblck_);												
		}
	}

	// ######################################################################################################################################
	
}

// ##########################################################################################################################################

