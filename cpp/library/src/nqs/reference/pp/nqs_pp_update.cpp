#include "../../../../include/NQS/NQS_ref_base/PP/nqs_pp.h"
#include "armadillo"

// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

// ##########################################################################################################################################

/**
* @brief Computes the gradient of the neural quantum state with respect to the parameters.
* 
* This function calculates the derivative of the neural quantum state (NQS) with respect to 
* the parameters, including the correlated part and the Pfaffian part. The gradient is used 
* in the optimization process to update the parameters of the NQS.
* 
* @tparam _spinModes The spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The type of the parameters.
* @tparam _stateType The type of the state.
* @tparam _CorrState The type of the correlated state.
* @param _v The configuration.
* @param _plc The index of the parameter to be updated.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::grad(Config_cr_t _v, uint _plc)
{
	NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::grad(_v, _plc);       // calculate the derivative of the previous part (correlated part)
	
    // calculate the derivative of the Pffafian 
	// as 1/Pf(X) * \partial Pf(X) / \partial x = 1/2 Tr(X^{-1} \partial X / \partial x)
#ifndef NQS_REF_PP_USE_PFAFFIAN_UPDATE
	this->pp_weights_.X_inv = arma::inv(this->pp_weights_.X_, arma::solve_opts::likely_sympd);
#endif

	// not multiplying by pfaffian, as it's dividied by it later in the definition of the derivative (log derivative is calculated)
	this->pp_weights_.Xinv_skew_    = (this->pp_weights_.X_inv.st() - this->pp_weights_.X_inv);
    const auto _prevsize            = NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::size();
	auto _currDerivative	        = this->derivatives_.derivatives.row(_plc).subvec(_prevsize, this->PPsize_ - 1);
	_currDerivative.zeros();

//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threads_.threadNum_)
//#endif
	int _spinIter [[maybe_unused]] = 0;
    // go through the spin sectors
	for (const auto& s: this->spinSectors_)
	{
        // go through the particles and calculate the derivatives for the Pfaffian
		for (uint i = 0; i < this->info_p_.nParticles_; ++i)
		{
			// if left spin differs from the current left spin, just continue
            if (Binary::check(NQS_STATE, i) != s[0])
				continue;

			for (uint j = 0; j < this->info_p_.nParticles_; ++j)
			{
				// skip the diagonal part
				if (j == i)
					continue;

				// if right spin differs from the current right spin, just continue
                if (Binary::check(NQS_STATE, j) != s[1])
                    continue;
                auto _idx = this->getFPPIndex(s[0], s[1], i, j);
				_currDerivative(_idx)           = this->pp_weights_.Xinv_skew_(i, j);
			}
		}
		_spinIter += this->info_p_.nParticlesSquared_;
	}
}
// template instantiation of the function above
NQS_PP_INST_CMB_ALL(grad, void, (Config_cr_t, uint), );

// ##########################################################################################################################################

// ############################################################ U P D A T E R S #############################################################

// ##########################################################################################################################################

////////////////////////////////////////////////////////////////////////
#ifdef NQS_ANGLES_UPD

/**
* @brief Update angles with the flipped spin (spins)
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::update(uint nFlips)
{
	// update the angles in the RBM
	NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::update(nFlips);
	// as the candidate pfaffian shall be already updated, use it instead of calculating everything all the time (probably not as efficient)
	// replace updating the pfaffian back
#ifdef NQS_REF_PP_USE_PFAFFIAN_UPDATE
	for(uint i = 0; i < nFlips; i++)
        this->pp_weights_.X_inv     = algebra::scherman_morrison_skew(this->pp_weights_.X_inv, this->flipPlaces_[i], this->pp_weights_.X_.row(this->flipPlaces_[i]));
#endif
    this->pp_weights_.X_		    = this->pp_weights_.X_;
    this->pp_weights_.pfaffian_     = this->pp_weights_.pfaffianNew_;
    this->pp_weights_.pfaffianLog_  = this->pp_weights_.pfaffianNewLog_;
}
// template instantiation of the function above
NQS_PP_INST_CMB_ALL(update, void, (uint), );
// ##########################################################################################################################################

/**
* @brief If a new pfaffian state has not been accepted, unupdate the values
* @param nFlips number of flips to be used
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::unupdate(uint nFlips)
{
    NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::unupdate(nFlips);
	// unupdate the Pffafian matrix
	// as the candidate pfaffian shall be already updated, use it instead of calculating everything all the time (probably not as efficient)
	// replace updating the pfaffian back
	// this->Xnew_			= this->X_;
	// this->pfaffianNew_	= this->pfaffian_;
}
// template instantiation of the function above
NQS_PP_INST_CMB_ALL(unupdate, void, (uint), );
// ##########################################################################################################################################

/**
* @brief Update angles with the flipped spin (spins)
* @param vector after the flips has been done
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::update(Config_cr_t v, uint nFlips)
{
    NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::update(v, nFlips);
#ifdef NQS_REF_PP_USE_PFAFFIAN_UPDATE
	for(uint i = 0; i < nFlips; i++)
        this->pp_weights_.X_inv     = algebra::scherman_morrison_skew(this->pp_weights_.X_inv, this->flipPlaces_[i], this->pp_weights_.X_.row(this->flipPlaces_[i]));
#endif
    this->pp_weights_.X_		    = this->pp_weights_.X_;
    this->pp_weights_.pfaffian_     = this->pp_weights_.pfaffianNew_;
    this->pp_weights_.pfaffianLog_  = this->pp_weights_.pfaffianNewLog_;
}
// template instantiation of the function above
NQS_PP_INST_CMB_ALL(update, void, (Config_cr_t, uint), );
// ##########################################################################################################################################
#endif

////////////////////////////////////////////////////////////////////////////////////