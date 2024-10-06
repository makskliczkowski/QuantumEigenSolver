//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FERMIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////
#	ifndef RBM_PP_4_H
#		define RBM_PP_4_H

///*
//* @brief Allows one to find the new Pfaffian matrix to calculate the overlap <x|\\phi_ref> in the PP wave function.
//* This is done as (N/2)! Pf[X] for the standard
//* |\\phi _ref> = (\\sum _{i,j} \\sum _{\\sigma, \\sigma'} F_{ij}^{\\sigma,\\sigma'} c_{i\\sigma}^\\dag c_{j\\sigma}^\\dag
//* One seeks for the position of spins and obtains X_ij = F_{ri,rj}^{\\sigma_i, \\sigma_j} - F_{rj,ri}^{\\sigma_j, \\sigma_i}.
//* For now we calculate this as a double loop
//* This is for the Fermions and in this loop it just checks the spin ad hoc, as half filling is imposed.
//[!TODO optimize].
//*
//* @param state defined as a column vector of ones and zeros:
//*	for fermions one has vector of #Nsite spin ups and then #Nsite spin downs
//*	for spins one has only vector of #Nsite spins that can be transformed to Abrikosov fermions
//*/
//template <typename _Ht, typename _T, class _stateType>
//void RBM_PP_S<4, _Ht, _T, _stateType>::ansatzPP(const _stateType& _n)
//{
//	// !TODO
//	LOGINFO("Function " + __FUNCTION__ + " not implemented yet!", LOG_TYPES::ERROR, 0);
//	throw (std::runtime_error(__FUNCTION__));
//}
#	endif
//////////////////////////////////////////////////////////////////////////////////////////