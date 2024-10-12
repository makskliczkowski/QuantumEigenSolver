#include "nqs_general.inl"

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SPINS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

#ifndef NQS_2_H
#	define NQS_2_H

/*
* @brief Template NQS for Spins (hardcore bosons)
*/
template <typename _Ht, typename _T, class _stateType>
class NQS_S<2, _Ht, _T, _stateType> : public NQS<2, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
	using NQSLS_p =	NQS<2, _Ht, _T, _stateType>::NQSLS_p;
public:
	NQS_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, double _lr, uint _threadNum = 1, int _nParticles = 1, const NQSLS_p& _lower = {}, const std::vector<double>& _beta = {})
		: NQS<2, _Ht, _T, _stateType>(_H, _lr, _threadNum, _H->getNs(), _lower, _beta) 
	{	};

protected:
	// -------------------------- F L I P S --------------------------
	virtual void chooseRandomFlips()			override;

	// apply flips to the temporary vector or the current vector according the template
	virtual void applyFlipsT()					override { for (auto& i : this->flipPlaces_) flip(this->tmpVec_, i, 0, this->discVal_);	};
	virtual void applyFlipsC()					override { for (auto& i : this->flipPlaces_) flip(this->curVec_, i, 0, this->discVal_);	};
	virtual void setRandomFlipNum(uint _nFlips) override;
};

// !!!!!!!!!!!!!!!!!! F L I P S !!!!!!!!!!!!!!!!!!

/*
* @brief Randomly flip the discrete variables at chosen flip places. Sets the random flips to the vector already saved.
*/
template<typename _Ht, typename _T, class _stateType>
inline void NQS_S<2, _Ht, _T, _stateType>::chooseRandomFlips()
{
	// go through the vector elements
	for (auto i = 0; i < this->flipPlaces_.size(); ++i)
	{
		auto fP					= this->ran_.template randomInt<uint>(0, this->info_p_.nVis_);
		// choose the flip place of the vector
		this->flipPlaces_[i]	= fP;
		// save the element of a vector before the flip
		this->flipVals_[i]		= this->tmpVec_(fP);
	}
}

//////////////////////////////////////////////////

/*
* @brief Set the number of random flips.
* @param _nFlips number of flips to be used
*/
template<typename _Ht, typename _T, class _stateType>
inline void NQS_S<2, _Ht, _T, _stateType>::setRandomFlipNum(uint _nFlips)
{
	this->nFlip_ = _nFlips;
	this->flipPlaces_.resize(this->nFlip_);
	this->flipVals_.resize(this->nFlip_);
}

#endif