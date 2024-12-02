#include "rbm_pp_pfaffian.tpp"

// ##########################################################################################################################################

// ######################################################### P R O B A B I L I T Y ##########################################################

// ##########################################################################################################################################

// %%%%%%%%%%%%%%%%%%% S I N G L E   F L I P %%%%%%%%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use only one flip
* @param fP place where the flip has been applied
* @param fV value before the flip!
* @returns probability ratio for a given ansatz based on the current state
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::logPRatio(uint fP, float fV)
{
	// update pfaffian candidate matrix and its corresponding value
	this->updFPP_C(fP, fV);
#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	this->updatePfaffian_C(fP);
#else
	this->setPfaffian_C();
#endif
	// return RBM_S<_spinModes, _Ht, _T, _stateType>::pRatio(fP, fV) * (this->pfaffianNew_ / this->pfaffian_);
	return RBM_S<_spinModes, _Ht, _T, _stateType>::logPRatio(fP, fV) + std::log(this->pfaffianNew_) - std::log(this->pfaffian_);

}


// %%%%%%%%%%%%%%%% M U L T I P L E   F L I P S %%%%%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use multiple flips.
* Uses the flips stored within the NQS class (flipPlaces_, flipVals_)
* If multiple flips are used, one should use calculate the Xinv and pfaffian from scratch
* @param nFlips number of flips to be used
* @returns probability ratio for a given ansatz based on the current state
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
[[deprecated("Use pRatio(uint fP, float fV) instead")]]
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::logPRatio(uint nFlips)
{
	// update pfaffian candidate matrix and its corresponding value
	for (auto i = 0; i < nFlips; ++i)
		this->updFPP_C(this->flipPlaces_[i], this->flipVals_[i]);
#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	for (auto i = 0; i < nFlips; ++i)
		this->updatePfaffian_C(this->flipPlaces_[i]);
	//LOG_DEBUG(VEQP(this->pfaffianNew_, 10), "Pfaffian from update");
	//this->setPfaffian_C();
	//LOG_DEBUG(VEQP(this->pfaffianNew_, 10), "Pfaffian from scratch");
#else
	this->setPfaffian_C();
#endif
	// return RBM_S<_spinModes, _Ht, _T, _stateType>::pRatio(nFlips) * (this->pfaffianNew_ / this->pfaffian_);
	return RBM_S<_spinModes, _Ht, _T, _stateType>::logPRatio(nFlips) + std::log(this->pfaffianNew_) - std::log(this->pfaffian_);
}

// %%%%%%%%%%%%%%%%% U S I N G   V E C T O R S %%%%%%%%%%%%%%%%%

/*
* @brief computes (Psi'/Psi), where (Psi') is the state with certain positions flipped. 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::logPRatio(const NQSS& _v1, const NQSS& _v2)
{
	// return RBM_S<_spinModes, _Ht, _T, _stateType>::pRatio(_v1, _v2) * this->getPfaffian(_v2) / this->getPfaffian(_v1);
	return RBM_S<_spinModes, _Ht, _T, _stateType>::logPRatio(_v1, _v2) + std::log(this->getPfaffian(_v2)) - std::log(this->getPfaffian(_v1));
}

// %%%%%%%%%%%%% U S I N G   I N I T I A L I Z E R %%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use multiple flips ariving from the external vectors.
* Uses the flips stored within the NQS class (fP, fV)
* @param fP flip places to be used
* @param fV flip values to be used
* @returns probability ratio for a given ansatz based on the current state
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::logPRatio(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
#if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP
	auto thId			= std::this_thread::get_id();
	this->XTmp_[thId]	= this->X_;

	// update
	this->updFPP_F(fP, fV, this->XTmp_[thId]);
#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	auto _pfaffian		= this->pfaffian_;
	if (fP.size() == 1)
		for(const auto& _row: fP)
			this->updatePfaffian(_row, _pfaffian, this->XTmp_[thId]);
	else
		_pfaffian		= this->getPfaffian(this->XTmp_[thId]);
#else
	auto _pfaffian		= this->getPfaffian(this->XTmp_[thId]);
#endif
#else
	this->XTmp_			= this->X_;
	// update
	this->updFPP_F(fP, fV, this->XTmp_);
	auto _pfaffian		= this->getPfaffian(this->XTmp_);
#endif
	// return RBM_S<_spinModes, _Ht, _T, _stateType>::pRatio(fP, fV) * _pfaffian / this->pfaffian_;
	return RBM_S<_spinModes, _Ht, _T, _stateType>::logPRatio(fP, fV) + std::log(_pfaffian) - std::log(this->pfaffian_);
}

// ##########################################################################################################################################