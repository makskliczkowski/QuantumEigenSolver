#include "rbm_pp_general.tpp"

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SPINS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////
#	ifndef RBM_PP_2_H
#		define RBM_PP_2_H

/*
* @brief Restricted Boltzmann Machines ansatz with Pair Product reference state for NQS
*/
template <typename _Ht, typename _T, class _stateType>
class RBM_PP_S<2, _Ht, _T, _stateType> : public RBM_PP<2, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
	using NQSLS_p = typename RBM_PP<2, _Ht, _T, _stateType>::NQSLS_p;
	/* ------------------------------------------------------- */
public:
	RBM_PP_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum = 1, 
				int _nParticles = -1, const NQSLS_p& _lower = {}, const std::vector<double>& _beta = {})
		: RBM_PP<2, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles, _lower, _beta) 
	{
		this->init();
		// sets the thetas and Pfaffian as well
		this->setRandomState(true);
	};

	/* ------------------------------------------------------- */
	// --------------------- G E T T E R S ---------------------
	virtual auto getPPMat(const NQSS& _n)	const -> NQSW	override;
#ifndef NQS_USE_VEC_ONLY
	virtual auto getPPMat(u64 _n)			const -> NQSW	override;
#endif
protected:
	// --------------------------- A N S A T Z -----------------
	void updFPP_C(uint fP, float fV)						override;
	void updFPP_C(std::initializer_list<int> fP,
					 std::initializer_list<double> fV)		override;
	void updFPP(uint fP, float fV)							override;
	void updFPP(std::initializer_list<int> fP,
				std::initializer_list<double> fV)			override;
	void updFPP_F(std::initializer_list<int> fP,
				std::initializer_list<double> fV,
				arma::Mat<_T>& _Xtmp)						override;
};

// !!!!!!!!!!!!!!!! P F F A F I A N   S T A T E !!!!!!!!!!!!!!!!

/*
* @brief Allows one to find the new Pfaffian matrix to calculate the overlap <x|\\phi_ref> in the PP wave function.
* This is done as (N/2)! Pf[X] for the standard
* |\\phi _ref> = (\\sum _{i,j} \\sum _{\\sigma, \\sigma'} F_{ij}^{\\sigma,\\sigma'} c_{i\\sigma}^\\dag c_{j\\sigma}^\\dag
* One seeks for the position of spins and obtains X_ij = F_{ri,rj}^{\\sigma_i, \\sigma_j} - F_{rj,ri}^{\\sigma_j, \\sigma_i}.
* For now we calculate this as a double loop
* This is for the spins and in this loop it just checks the spin ad hoc, as half filling is imposed.
[!TODO optimize].
*
* @param state defined as a column vector of ones and zeros:
*	for fermions one has vector of #Nsite spin ups and then #Nsite spin downs
*	for spins one has only vector of #Nsite spins that can be transformed to Abrikosov fermions
* @returns the Pfaffian matrix
*/
template <typename _Ht, typename _T, class _stateType>
typename RBM_PP_S<2, _Ht, _T, _stateType>::NQSW RBM_PP_S<2, _Ht, _T, _stateType>::getPPMat(const NQSS& _n) const
{
	// remember that FF is such that is starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN})
	NQSW _out(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);

	// go through ri, aka, the ith particle site
	for (uint i = 0; i < this->info_p_.nSites_; ++i)
	{
		// check the spin at a given position i
		bool spin_ri = checkBit(_n, i);

		// go through rj, aka, the jth particle site
		for (uint j = i + 1; j < this->info_p_.nSites_; ++j)
		{
			// check the spin at a given position
			bool spin_rj = checkBit(_n, j);
			{
				// F_{ri,rj}^{\\sigma_i, \\sigma_j} - find the index corresponding to those particles in F
				// F_{rj,ri}^{\\sigma_j, \\sigma_i} - find the index corresponding to those particles in F
				// get the Pffafian please X_ij = F_{ri,rj}^{\\sigma_i, \\sigma_j} - F_{rj,ri}^{\\sigma_j, \\sigma_i}
				_out(i, j) = this->Fpp_(this->getFPPIndex(spin_ri, spin_rj, i, j)) - this->Fpp_(this->getFPPIndex(spin_rj, spin_ri, j, i));
				_out(j, i) = -_out(i, j);
			}
		}
	}
	return _out;
}

////////////////////////////////////////////////////////////////

#ifndef NQS_USE_VEC_ONLY
template <typename _Ht, typename _T, class _stateType>
typename RBM_PP_S<2, _Ht, _T, _stateType>::NQSW RBM_PP_S<2, _Ht, _T, _stateType>::getPPMat(u64 _n) const
{
	NQSW _out(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);
	// go through ri
	for (uint i = 0; i < this->info_p_.nSites_; ++i)
	{
		// check the spin at a given position
		bool spinI = checkBit(_n, i);
		for (uint j = 0; j < this->info_p_.nSites_; ++j)
		{
			// check the spin at a given position
			bool spinJ		=	checkBit(_n, j);
			{
				// F_{ri,rj}^{\\sigma_i, \\sigma_j}
				uint positionLeft	=	(spinI ? 0 : 2 * this->info_p_.nSites_ * this->info_p_.nSites_) + 
										(spinJ ? 0 : this->info_p_.nSites_ * this->info_p_.nSites_)		+ 
										i * this->info_p_.nSites_ + j;
				// F_{rj,ri}^{\\sigma_j, \\sigma_i}
				uint positionRight	=	(spinI ? 2 * this->info_p_.nSites_ * this->info_p_.nSites_ : 0) + 
										(spinJ ? this->info_p_.nSites_ * this->info_p_.nSites_ : 0)		+ 
										j * this->info_p_.nSites_ + i;
				// get the Pffafian please
				_out(i, j) = this->Fpp_(positionLeft) - this->Fpp_(positionRight);
			}
		}
	}
	return _out;
}
#endif

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// !!!!!!!!!!!! P F F A F I A N   C A N D I D A T E !!!!!!!!!!!!

/*
* @brief Updates the pfaffian candidate according to a single flip
* @param fP place of the flip in the current vector
* @param fV value before the flip at fP
*/
template <typename _Ht, typename _T, class _stateType>
void RBM_PP_S<2, _Ht, _T, _stateType>::updFPP_C(uint fP, float fV)
{
	// remember that FF is such that is starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN} = {I, II} , {III, IV})
	// if the fP is known, only the column and row containing it shall be updated
	// we will use current state as well to see where the other places are
	// remember, fV is the other way around (so it's before the flip)
	// new k, when particle on fP has been set for F_{ki}^{\sigma_k', \sigma _i}
	// run over the columns
	auto fVV = fV < 0;
	for (uint i = 0; i < this->info_p_.nParticles_; ++i)
	{
		// the same is not allowed (no double occupations)
		if (i == fP)
			continue;

		// check the bit on the i'th place to know in which place you'll end up
		bool spin_next	= checkBit(this->curVec_, i);

		// F_{ri,rj}^{\\sigma_i, \\sigma_j} - find the index corresponding to those particles in F
		auto posLeft	= this->getFPPIndex(fVV, spin_next, fP, i);
		// F_{rj,ri}^{\\sigma_j, \\sigma_i} - find the index corresponding to those particles in F
		auto posRight	= this->getFPPIndex(spin_next, fVV, i, fP);

		// update 
		this->Xnew_(fP, i) = this->Fpp_(posLeft) - this->Fpp_(posRight);
		this->Xnew_(i, fP) = -this->Xnew_(fP, i);
	}
}

////////////////////////////////////////////////////////////////

/*
* @brief Updates the pfaffian according to a single flip
* @param fP place of the flip in the current vector
* @param fV value before the flip at fP
*/
template <typename _Ht, typename _T, class _stateType>
void RBM_PP_S<2, _Ht, _T, _stateType>::updFPP(uint fP, float fV)
{
	// remember that FF is such that is starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN} = {I, II} , {III, IV})
	// if the fP is known, only the column and row containing it shall be updated
	// we will use current state as well to see where the other places are
	// remember, fV is the other way around (so it's before the flip)
	// new k, when particle on fP has been set for F_{ki}^{\sigma_k', \sigma _i}
	// run over the columns
	for (uint i = 0; i < this->info_p_.nParticles_; ++i)
	{
		// the same is not allowed (no double occupations)
		if (i == fP)
			continue;

		// check the bit on the i'th place to know in which place you'll end up
		bool spin_next	= checkBit(this->curVec_, i);

		// F_{ri,rj}^{\\sigma_i, \\sigma_j} - find the index corresponding to those particles in F
		auto posLeft	= this->getFPPIndex(fV < 0, spin_next, fP, i);
		// F_{rj,ri}^{\\sigma_j, \\sigma_i} - find the index corresponding to those particles in F
		auto posRight	= this->getFPPIndex(spin_next, fV < 0, i, fP);

		// update 
		this->X_(fP, i) = this->Fpp_(posLeft) - this->Fpp_(posRight);
		this->X_(i, fP) = -this->X_(fP, i);
	}
}

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

template <typename _Ht, typename _T, class _stateType>
void RBM_PP_S<2, _Ht, _T, _stateType>::updFPP_F(std::initializer_list<int> fP, std::initializer_list<double> fV, arma::Mat<_T>& _XTmp)
{
	// remember that FF is such that is starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN} = {I, II} , {III, IV})
	// if the fP is known, only the column and row containing it shall be updated
	// we will use current state as well to see where the other places are
	// remember, fV is the other way around (so it's before the flip)
	// new k, when particle on fP has been set for F_{ki}^{\sigma_k', \sigma _i}
	// run over the columns

	// !TODO Optimize!
	v_1d<bool> _state(this->curVec_.size());

	for (auto i = 0; i < this->info_p_.nParticles_; ++i)
		_state[i] = checkBit(this->curVec_, i);

	for (auto fPi = 0; fPi < fP.size(); ++fPi)
		_state[*(fP.begin() + fPi)] = (*(fV.begin() + fPi) < 0);

	for (auto fPi = 0; fPi < fP.size(); ++fPi)
	{
		auto fPP = *(fP.begin() + fPi);
		auto fVV = _state[fPP];

		for (auto i = 0; i < this->info_p_.nParticles_; ++i)
		{
			// the same is not allowed (no double occupations)
			if (i == fPP)
				continue;

			// check the bit on the i'th place to know in which place you'll end up
			bool spin_next	= _state[i];

			// F_{ri,rj}^{\\sigma_i, \\sigma_j} - find the index corresponding to those particles in F
			auto posLeft	= this->getFPPIndex(fVV, spin_next, fPP, i);
			// F_{rj,ri}^{\\sigma_j, \\sigma_i} - find the index corresponding to those particles in F
			auto posRight	= this->getFPPIndex(spin_next, fVV, i, fPP);

			// update 
			_XTmp(fPP, i)	= this->Fpp_(posLeft) - this->Fpp_(posRight);
			_XTmp(i, fPP)	= -_XTmp(fPP, i);
		}
	}
}

////////////////////////////////////////////////////////////////

template <typename _Ht, typename _T, class _stateType>
void RBM_PP_S<2, _Ht, _T, _stateType>::updFPP_C(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
	this->updFPP_F(fP, fV, this->Xnew_);
}

////////////////////////////////////////////////////////////////

template <typename _Ht, typename _T, class _stateType>
void RBM_PP_S<2, _Ht, _T, _stateType>::updFPP(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
	this->updFPP_F(fP, fV, this->X_);
}

////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#	endif