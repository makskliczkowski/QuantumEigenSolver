#include "../../../../include/NQS/NQS_ref_base/PP/nqs_pp_spins.hpp"

// ##########################################################################################################################################

/**
* @brief Calculates the new Pfaffian matrix to determine the overlap <x|\\phi_ref> in the PP wave function.
* 
* The overlap is computed as (N/2)! Pf[X] for the standard wave function:
* |\\phi _ref> = (\\sum _{i,j} \\sum _{\\sigma, \\sigma'} F_{ij}^{\\sigma,\\sigma'} c_{i\\sigma}^\\dag c_{j\\sigma}^\\dag)
* 
* This function seeks the positions of spins and computes the matrix elements:
* X_ij = F_{ri,rj}^{\\sigma_i, \\sigma_j} - F_{rj,ri}^{\\sigma_j, \\sigma_i}
* 
* The current implementation uses a double loop to iterate over the spins and positions.
* Note: This implementation assumes half-filling and checks the spin ad hoc. [TODO: Optimize this process].
* 
* @param _n Configuration state defined as a column vector of ones and zeros:
* - For fermions: A vector of #Nsite spin ups followed by #Nsite spin downs.
* - For spins: A vector of #Nsite spins that can be transformed to Abrikosov fermions.
* 
* @returns The Pfaffian matrix.
*/
template <typename _Ht, typename _T, class _stateType, class _CorrState>
typename NQS_PP_S<2, _Ht, _T, _stateType, _CorrState>::NQSW
NQS_PP_S<2, _Ht, _T, _stateType, _CorrState>::calculateX(Config_cr_t _n) const
{
	// remember that FF is such that is starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN})
	NQSW _out(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);

    // !TODO implement this as real positions of the particles...
    // !currently it's just a loop over the spins and positions iterate over the spins

	// go through ri, aka, the ith particle site
	for (auto i = 0; i < this->info_p_.nSites_; ++i)
	{
		const int spin_ri = Binary::check(_n, i);

		// go through rj, aka, the jth particle site
		for (uint j = i + 1; j < this->info_p_.nSites_; ++j)
		{
			// check the spin at a given position
            const int spin_rj = Binary::check(_n, j);
			{
                const auto _leftidx     = this->getFPPIndex(spin_ri, spin_rj, i, j);
                const auto _rightidx    = this->getFPPIndex(spin_rj, spin_ri, j, i);
                const auto _left        = this->pp_weights_.F_r1r2_s1s2_(static_cast<u64>(_leftidx));
                const auto _right       = this->pp_weights_.F_r1r2_s1s2_(static_cast<u64>(_rightidx));
				// F_{ri,rj}^{\\sigma_i, \\sigma_j} - find the index corresponding to those particles in F
				// F_{rj,ri}^{\\sigma_j, \\sigma_i} - find the index corresponding to those particles in F
				// get the Pffafian please X_ij = F_{ri,rj}^{\\sigma_i, \\sigma_j} - F_{rj,ri}^{\\sigma_j, \\sigma_i}
				_out(i, j)              = _left - _right;
				_out(j, i)              = -_out(i, j);
			}
		}
	}
	return _out;
}
// template instantiation
template class NQS_PP_S<2, double, double, double>;
template class NQS_PP_S<2, std::complex<double>, std::complex<double>, double>;
template class NQS_PP_S<2, double, std::complex<double>, double>;
template class NQS_PP_S<2, std::complex<double>, double, double>;
// ##########################################################################################################################################

#ifndef NQS_USE_VEC_ONLY
// template <typename _Ht, typename _T, class _stateType>
// typename RBM_PP_S<2, _Ht, _T, _stateType>::NQSW RBM_PP_S<2, _Ht, _T, _stateType>::getPPMat(u64 _n) const
// {
// 	NQSW _out(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);
// 	// go through ri
// 	for (uint i = 0; i < this->info_p_.nSites_; ++i)
// 	{
// 		// check the spin at a given position
// 		bool spinI = checkBit(_n, i);
// 		for (uint j = 0; j < this->info_p_.nSites_; ++j)
// 		{
// 			// check the spin at a given position
// 			bool spinJ		=	checkBit(_n, j);
// 			{
// 				// F_{ri,rj}^{\\sigma_i, \\sigma_j}
// 				uint positionLeft	=	(spinI ? 0 : 2 * this->info_p_.nSites_ * this->info_p_.nSites_) + 
// 										(spinJ ? 0 : this->info_p_.nSites_ * this->info_p_.nSites_)		+ 
// 										i * this->info_p_.nSites_ + j;
// 				// F_{rj,ri}^{\\sigma_j, \\sigma_i}
// 				uint positionRight	=	(spinI ? 2 * this->info_p_.nSites_ * this->info_p_.nSites_ : 0) + 
// 										(spinJ ? this->info_p_.nSites_ * this->info_p_.nSites_ : 0)		+ 
// 										j * this->info_p_.nSites_ + i;
// 				// get the Pffafian please
// 				_out(i, j) = this->Fpp_(positionLeft) - this->Fpp_(positionRight);
// 			}
// 		}
// 	}
// 	return _out;
// }
#endif

// ##########################################################################################################################################

/**
* @brief Updates the X matrix when a spin flip occurs at a given position.
*
* This function updates the X matrix when a spin flip occurs at a given position (fP) with a given value (fV).
* 
* Remember that FF is such that it starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN} = {I, II}, {III, IV}).
* If the fP is known, only the column and row containing it shall be updated.
* We will use the current state as well to see where the other places are.
* Remember, fV is the other way around (so it's before the flip).
* New k, when particle on fP has been set for F_{ki}^{\sigma_k', \sigma_i}.
* Run over the columns.
*
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Type of the parameter.
* @tparam _stateType Type of the state.
* @tparam _CorrState Type of the correlated state.
* @param fP The position of the flip.
* @param fV The value before the flip.
*/
template <typename _Ht, typename _T, class _stateType, class _CorrState>
void NQS_PP_S<2, _Ht, _T, _stateType, _CorrState>::setX_changed(int fP, float fV)
{
	// remember that FF is such that it starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN} = {I, II} , {III, IV})
	// if the fP is known, only the column and row containing it shall be updated
	// we will use current state as well to see where the other places are
	// remember, fV is the other way around (so it's before the flip)
	// new k, when particle on fP has been set for F_{ki}^{\sigma_k', \sigma _i}
	// run over the columns

	// check if the column is already set to a given size and resize if not
	if (this->pp_weights_.X_upd_.empty())
		this->pp_weights_.X_upd_.emplace_back(NQSB(this->info_p_.nParticles_));
	else if (this->pp_weights_.X_upd_[0].n_rows != this->info_p_.nParticles_)
		this->pp_weights_.X_upd_[0].resize(this->info_p_.nParticles_);

	// check the spin at a given position (fP) -> this corresponds to the value before the flip
	const int flipped_val = !(fV > 0); // take the opposite value - the value after the flip

	// go through the elements of the matrix and update the values analogously to the calculation
	// remember, not only we update the column but also the row - skew-symmetric matrix
	for (auto i = 0; i < this->info_p_.nParticles_; ++i)
	{
		if (i == fP) // the same is not allowed (no double occupations)
			continue;

		int spin_next 	= Binary::check(NQS_STATE, i); // check the bit on the i'th place to know in which place you'll end up

		// F_{ri,rj}^{\\sigma_i, \\sigma_j} - find the index corresponding to those particles in F
		auto posLeft 	= this->getFPPIndex(flipped_val, spin_next, fP, i);
		// F_{rj,ri}^{\\sigma_j, \\sigma_i} - find the index corresponding to those particles in F
		auto posRight 	= this->getFPPIndex(spin_next, flipped_val, i, fP);
		// update the matrix column and row (remember, the column is then - the row)
		auto value = this->pp_weights_.F_r1r2_s1s2_(posLeft) - this->pp_weights_.F_r1r2_s1s2_(posRight);
		this->pp_weights_.X_upd_[0](fP, i) = value;
		this->pp_weights_.X_upd_[0](i, fP) = -value;
	}
}
// template instantiation
template void NQS_PP_S<2, double, double, double>::setX_changed(int, float);
template void NQS_PP_S<2, std::complex<double>, std::complex<double>, double>::setX_changed(int, float);
template void NQS_PP_S<2, double, std::complex<double>, double>::setX_changed(int, float);
template void NQS_PP_S<2, std::complex<double>, double, double>::setX_changed(int, float);
// ##########################################################################################################################################

/**
* @brief Updates the matrix X_t based on the changes in spins.
*
* This function updates the matrix X_t efficiently by leveraging thread-local storage for states and minimizing redundant operations.
*
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Type of the parameter.
* @tparam _stateType Type of the state.
* @tparam _CorrState Type of the correlated state.
* @param _fP Vector of initial positions.
* @param _fV Vector of initial values.
* @param X_t Matrix to be updated.
*/
template <typename _Ht, typename _T, class _stateType, class _CorrState>
void NQS_PP_S<2, _Ht, _T, _stateType, _CorrState>::setX_changed(int_ini_t _fP, dbl_ini_t _fV, NQSW& X_t) 
{
    const auto nParticles = this->info_p_.nParticles_;
    const auto nVis = this->info_p_.nVis_;

    // Ensure `states_upd_` is resized appropriately
    if (this->pp_weights_.states_upd_.size() != nVis) {
        this->pp_weights_.states_upd_.resize(nVis, false);
    }

    // Update thread-local states based on the initial spins
    for (auto i = 0; i < nParticles; ++i) {
        this->pp_weights_.states_upd_[i] = Binary::check(NQS_STATE, i);
    }

    // Flip the states based on the input `_fP` and `_fV`
    for (size_t fPi = 0; fPi < _fP.size(); ++fPi) {
		const auto _fPin = *(std::next(_fP.begin(), fPi));
		const auto _fVin = *(std::next(_fV.begin(), fPi));
        this->pp_weights_.states_upd_[_fPin] = !(_fVin > 0);
    }

    // Update the matrix `X_t`
    for (size_t fPi = 0; fPi < _fP.size(); ++fPi) 
	{
        const auto _fPin = *(std::next(_fP.begin(), fPi));
        const auto _fVin = this->pp_weights_.states_upd_[_fPin];

        // Iterate over all particles to update matrix values
        for (size_t i = 0; i < nParticles; ++i) 
		{
            if (i == _fPin)
                continue; // Skip self-interactions

            const int spin_next		= this->pp_weights_.states_upd_[i];

            // Compute indices for the correlated function F
            const auto posLeft 		= this->getFPPIndex(_fVin, spin_next, _fPin, i);
            const auto posRight 	= this->getFPPIndex(spin_next, _fVin, i, _fPin);

            // Update matrix values using symmetry
            const auto value = this->pp_weights_.F_r1r2_s1s2_(posLeft) - this->pp_weights_.F_r1r2_s1s2_(posRight);
            X_t(_fPin, i) = value;
            X_t(i, _fPin) = -value; // Enforce skew-symmetric property
        }
    }
}
// template instantiation
template void NQS_PP_S<2, double, double, double>::setX_changed(int_ini_t, dbl_ini_t, NQSW&);
template void NQS_PP_S<2, std::complex<double>, std::complex<double>, double>::setX_changed(int_ini_t, dbl_ini_t, NQSW&);
template void NQS_PP_S<2, double, std::complex<double>, double>::setX_changed(int_ini_t, dbl_ini_t, NQSW&);
template void NQS_PP_S<2, std::complex<double>, double, double>::setX_changed(int_ini_t, dbl_ini_t, NQSW&);
// ##########################################################################################################################################

/**
* @brief Updates the matrix `X_t` based on the changes in particle states.
*
* This function updates the matrix `X_t` by flipping the states of particles
* according to the input vectors `_fP` and `_fV`. It ensures that the size of
* `X_t` and `states_upd_` are appropriate and then updates the matrix values
* using the correlated function `F_r1r2_s1s2_`.
*
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Type of the temperature.
* @tparam _stateType Type of the state.
* @tparam _CorrState Type of the correlated state.
* @param _fP Vector of particle indices to be flipped.
* @param _fV Vector of values indicating the new state of the particles.
* @param X_t Matrix to be updated based on the new particle states.
*/
template <typename _Ht, typename _T, class _stateType, class _CorrState>
void NQS_PP_S<2, _Ht, _T, _stateType, _CorrState>::setX_changed(int_ini_t _fP, dbl_ini_t _fV, v_1d<NQSB>& X_t)
{
	const auto nParticles 	= this->info_p_.nParticles_;
	const auto nVis	 		= this->info_p_.nVis_;

	// check the size of X_t and resize if necessary
	if (X_t.size() != _fP.size())
		X_t.resize(_fP.size(), NQSB(nParticles, nParticles));

	// Ensure `states_upd_` is resized appropriately
	if (this->pp_weights_.states_upd_.size() != nVis) 
		this->pp_weights_.states_upd_.resize(nVis, false);

	// Update thread-local states based on the initial spins
	for (auto i = 0; i < nParticles; ++i) 
		this->pp_weights_.states_upd_[i] = Binary::check(NQS_STATE, i);

	// Flip the states based on the input `_fP` and `_fV`
	for (size_t fPi = 0; fPi < _fP.size(); ++fPi) 
	{
		const auto _fPin = *(std::next(_fP.begin(), fPi));
		const auto _fVin = *(std::next(_fV.begin(), fPi));
        this->pp_weights_.states_upd_[_fPin] = !(_fVin > 0);
    }

	// Update the matrix `X_t`
	for (size_t fPi = 0; fPi < _fP.size(); ++fPi) {
		const auto _fPin = *(std::next(_fP.begin(), fPi));
		const auto _fVin = this->pp_weights_.states_upd_[_fPin];

		// Iterate over all particles to update matrix values
		for (size_t i = 0; i < nParticles; ++i) 
		{
			if (i == _fPin)
				continue;

			// Determine the spin state of the other particle
			const int spin_next 	= this->pp_weights_.states_upd_[i];

			// Compute indices for the correlated function F
			const auto posLeft 		= this->getFPPIndex(_fVin, spin_next, _fPin, i);
			const auto posRight 	= this->getFPPIndex(spin_next, _fVin, i, _fPin);

			// Update matrix values using symmetry
			const auto value 		= this->pp_weights_.F_r1r2_s1s2_(posLeft) - this->pp_weights_.F_r1r2_s1s2_(posRight);
			X_t[fPi](_fPin, i) 		= value;
			X_t[fPi](i, _fPin) 		= -value; // Enforce skew-symmetric property
		}
	}
}
// template instantiation
template void NQS_PP_S<2, double, double, double>::setX_changed(int_ini_t, dbl_ini_t, v_1d<NQSB>&);
template void NQS_PP_S<2, std::complex<double>, std::complex<double>, double>::setX_changed(int_ini_t, dbl_ini_t, v_1d<NQSB>&);
template void NQS_PP_S<2, double, std::complex<double>, double>::setX_changed(int_ini_t, dbl_ini_t, v_1d<NQSB>&);
template void NQS_PP_S<2, std::complex<double>, double, double>::setX_changed(int_ini_t, dbl_ini_t, v_1d<NQSB>&);
// ##########################################################################################################################################

/**
* @brief Sets the X_changed parameter with the given initial values.
*
* This function updates the X_changed parameter using the provided initial 
* values for the parameter and value. It also utilizes the pp_weights_ 
* member's X_upd_ attribute for the update process.
*
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Type of the parameter.
* @tparam _stateType Type of the state.
* @tparam _CorrState Type of the correlated state.
* @param _fP Initial value for the parameter.
* @param _fV Initial value for the value.
*/
template <typename _Ht, typename _T, class _stateType, class _CorrState>
inline void NQS_PP_S<2, _Ht, _T, _stateType, _CorrState>::setX_changed(int_ini_t _fP, dbl_ini_t _fV)
{
	this->setX_changed(_fP, _fV, this->pp_weights_.X_upd_);
}
// template instantiation
template void NQS_PP_S<2, double, double, double>::setX_changed(int_ini_t, dbl_ini_t);
template void NQS_PP_S<2, std::complex<double>, std::complex<double>, double>::setX_changed(int_ini_t, dbl_ini_t);
template void NQS_PP_S<2, double, std::complex<double>, double>::setX_changed(int_ini_t, dbl_ini_t);
template void NQS_PP_S<2, std::complex<double>, double, double>::setX_changed(int_ini_t, dbl_ini_t);
// ##########################################################################################################################################

/**
* @brief Updates the X_changed parameter for the NQS_PP_S class.
*
* This function updates the X_changed parameter using the provided function pointer
* and value, and applies the update to the pp_weights_ member's X_upd_ attribute.
*
* @tparam _Ht The type of the Hamiltonian.
* @tparam _T The type of the parameter.
* @tparam _stateType The type of the state.
* @tparam _CorrState The type of the correlated state.
* @param _fP The function pointer used to update X_changed.
* @param _fV The value used to update X_changed.
*/
template <typename _Ht, typename _T, class _stateType, class _CorrState>
inline void NQS_PP_S<2, _Ht, _T, _stateType, _CorrState>::updX_changed(int_ini_t _fP, dbl_ini_t _fV)
{
	// check if the _fP.size() is the same as the _X_upd_.size() and resize if necessary by updating the X matrix
	if (this->pp_weights_.X_upd_.size() != _fP.size())
		this->setX_changed(_fP, _fV);

	for (size_t i = 0; i < _fP.size(); ++i) 
	{
		const auto _fPin 				= *(std::next(_fP.begin(), i));
		this->pp_weights_.X_.row(_fPin) = this->pp_weights_.X_upd_[i];
		this->pp_weights_.X_.col(_fPin) = -this->pp_weights_.X_upd_[i];
	}
}
// template instantiation
template void NQS_PP_S<2, double, double, double>::updX_changed(int_ini_t, dbl_ini_t);
template void NQS_PP_S<2, std::complex<double>, std::complex<double>, double>::updX_changed(int_ini_t, dbl_ini_t);
template void NQS_PP_S<2, double, std::complex<double>, double>::updX_changed(int_ini_t, dbl_ini_t);
template void NQS_PP_S<2, std::complex<double>, double, double>::updX_changed(int_ini_t, dbl_ini_t);

// ##########################################################################################################################################

/**
* @brief Updates the X matrix for the given index.
*
* This function updates the row and column of the X matrix at the specified index `fP`
* with the values from the `X_upd_` array. The row is set to the first element of `X_upd_`
* and the column is set to the negation of the first element of `X_upd_`.
*
* @tparam _Ht The type of the Hamiltonian.
* @tparam _T The type of the parameters.
* @tparam _stateType The type of the state.
* @tparam _CorrState The type of the correlated state.
* @param fP The index of the row and column to be updated.
* @see NQS_PP_S::setX_changed for the calculation of the updated matrix !
*/
template <typename _Ht, typename _T, class _stateType, class _CorrState>
inline void NQS_PP_S<2, _Ht, _T, _stateType, _CorrState>::updX_changed(int fP)
{
	this->pp_weights_.X_.row(fP) = 	this->pp_weights_.X_upd_[0];
	this->pp_weights_.X_.col(fP) = -this->pp_weights_.X_upd_[0];
}
// template instantiation
template void NQS_PP_S<2, double, double, double>::updX_changed(int);
template void NQS_PP_S<2, std::complex<double>, std::complex<double>, double>::updX_changed(int);
template void NQS_PP_S<2, double, std::complex<double>, double>::updX_changed(int);
template void NQS_PP_S<2, std::complex<double>, double, double>::updX_changed(int);
// ##########################################################################################################################################







