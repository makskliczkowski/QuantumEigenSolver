// -----------------------------------------------------------------------------------------------------------------------------------
#include "./nqs_fermions.tpp"
#include <memory>
// -----------------------------------------------------------------------------------------------------------------------------------

#ifdef NQS_USESR

// ###################################################################################################################################

/**
* @brief Sets the training parameters for the lower states.
* 
* @param _par Training parameters to be set.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setTrainParExc(const MonteCarlo::MCS_train_t& _par)
{
	this->lower_states_.train_lower_ = _par;
	LOGINFO("Training parameters for lower states set.", LOG_TYPES::INFO, 2);
	LOGINFO("", LOG_TYPES::TRACE, 40, 2);
}

/**
* @brief Sets the learning rate scheduler.
* 
* @param _sch Scheduler type.
* @param _lr Initial learning rate.
* @param _lrd Learning rate decay.
* @param _epo Number of epochs.
* @param _pat Patience for early stopping.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setScheduler(int _sch, double _lr, double _lrd, size_t _epo, size_t _pat)
{
	if (_sch >= 0)
	{
		auto _raw			= MachineLearning::Schedulers::get_scheduler(_sch, _lr, _epo, _lrd, _pat);
		this->info_p_.p_ 	= _raw->move(); 	
		LOGINFO(std::format("Scheduler set: type={}, lr={}, decay={}, epochs={}, patience={}", 
				_sch, _lr, _lrd, _epo, _pat), LOG_TYPES::INFO, 2);
	}
}

/**
* @brief Sets the early stopping parameters.
* 
* @param _pat Patience for early stopping.
* @param _minDlt Minimum delta for improvement.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setEarlyStopping(size_t _pat, double _minDlt)
{
	if (_pat > 0)
	{ 
		this->info_p_.setEarlyStopping(_pat, _minDlt); 
		LOGINFO("Early stopping set with patience: " + STR(_pat) + ", minimum delta: " + VEQPS(_minDlt, 3), LOG_TYPES::CHOICE, 3);
	}
}

/**
* @brief Sets the regularization scheduler for the covariance matrix.
* 
* @param _sch Scheduler type.
* @param _sreg Initial regularization factor.
* @param _sregd Regularization factor decay.
* @param _epo Number of epochs.
* @param _pat Patience for early stopping.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setSregScheduler(int _sch, double _sreg, double _sregd, size_t _epo, size_t _pat)
{
	this->info_p_.sreg_ 	= _sreg; 
	if (_sreg > 0) 
	{ 
		LOGINFO("Regularization set with initial factor: " + VEQPS(_sreg, 3) + ", scheduler type: " + STR(_sch) + 
				", decay: " + VEQPS(_sregd, 3) + ", epochs: " + STR(_epo) + ", patience: " + STR(_pat), LOG_TYPES::CHOICE, 3); 
		
		// set the new scheduler
		auto _raw 			= MachineLearning::Schedulers::get_scheduler(_sch, _sreg, _epo, _sregd, _pat);
		this->info_p_.s_ 	= _raw->move();
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

/**
! TODO 
* @brief The inverse of the covariance matrix is poorly defined at the begining of training. 
* Use regularization to fix that issue.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::covMatrixReg(int step, _T _currLoss)
{
	this->info_p_.sreg_ = this->info_p_.sreg(step, algebra::real(_currLoss));
#ifndef NQS_USESR_NOMAT_USED
	this->S_.diag() += this->info_p_.sreg_ / (step + 1);
#endif
}

#endif

// -----------------------------------------------------------------------------------------------------------------------------------

