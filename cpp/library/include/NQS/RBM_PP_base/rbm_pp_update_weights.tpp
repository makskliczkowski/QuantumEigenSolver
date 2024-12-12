#include "rbm_pp_pratio.tpp"

// ##########################################################################################################################################

/**
* @brief Updates the weights in the system according to a given gradient
* @warning uses forces vector (member of NQS : dF_) to update the gradients - preallocation for optimization
* @note the function is called after the gradient is calculated and inlined to the optimization process
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::updateWeights()
{
	RBM_S<_spinModes, _Ht, _T, _stateType>::updateWeights();
	this->setWeights();
}

// ##########################################################################################################################################

/**
* @brief Sets up the post-processing weights (Fpp) from the full weight vector.
* 
* This function extracts a subvector from the full weight vector (Weights_) starting from
* rbmSize_ up to rbmPPSize_ - 1 and assigns it to the post-processing weights (Fpp_).
* The post-processing weights are used for additional transformations after the main RBM
* computation.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::setWeights()
{
	this->Fpp_ = this->Weights_.subvec(this->rbmSize_, this->rbmPPSize_ - 1);
}

// ##########################################################################################################################################