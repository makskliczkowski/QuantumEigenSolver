#include "rbm_pp_pratio.tpp"

// ##########################################################################################################################################

/*
* @brief Updates the weights in the system according to a given gradient
* @warning uses forces vector (member of NQS : F_) to update the gradients - preallocation for optimization
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::updateWeights()
{
	// update RBM weights
	RBM_S<_spinModes, _Ht, _T, _stateType>::updateWeights();
	// Fmat is a vector
	this->Fpp_ += this->dF_.subvec(this->rbmSize_, this->rbmPPSize_ - 1);
}
