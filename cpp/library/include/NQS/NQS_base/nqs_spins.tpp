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
	// #################################################################
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	NQS_PUBLIC_TYPES(_T, _stateType);
	NQS_HAMIL_TYPES(_Ht, 2);
	using NQSLS_p =	typename NQS<2, _Ht, _T, _stateType>::NQSLS_p;
	// #################################################################
public:
	NQS_S(const NQS_Const_par_t<2, _Ht, _T, _stateType>& _p)
		: NQS<2, _Ht, _T, _stateType>(_p) 
	{ 

	};
	NQS_S(const NQS_Const_par_t<2, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
		: NQS<2, _Ht, _T, _stateType>(_p, _lower, _beta) 
	{ 

	};
	// #################################################################
protected:
	// --------------------------- F L I P S ---------------------------
	virtual void chooseRandomFlips()			override;

	// apply flips to the temporary vector or the current vector according the template
	virtual void applyFlipsT()					override { for (auto& i : this->flipPlaces_) flip(this->tmpVec_, i, 0, this->discVal_);	};
	virtual void applyFlipsC()					override { for (auto& i : this->flipPlaces_) flip(this->curVec_, i, 0, this->discVal_);	};
	virtual void setRandomFlipNum(uint _nFlips) override;

	// **********************************************************************************************************************
	virtual auto clone() 						const -> MC_t_p override = 0;
	// **********************************************************************************************************************
};

// !!!!!!!!!!!!!!!!!! F L I P S !!!!!!!!!!!!!!!!!!

#include <unordered_set>

/**
* @brief Chooses random positions in a vector to flip and stores the positions and their values.
*
* This function selects a specified number of unique random positions (flips) within a vector.
* It ensures that each chosen position is unique by using an unordered set to track the positions
* that have already been selected. For each chosen position, it stores the position and the value
* of the vector at that position before the flip.
*
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Type of the elements in the vector.
* @tparam _stateType Type of the state.
*/
template<typename _Ht, typename _T, class _stateType>
inline void NQS_S<2, _Ht, _T, _stateType>::chooseRandomFlips()
{
	std::unordered_set<uint> chosenPlaces;
	chosenPlaces.reserve(this->nFlip_);

	for (auto i = 0; i < this->nFlip_; ++i)
	{
		uint fP;
		do {
			fP = this->ran_->template randomInt<uint>(0, this->info_p_.nVis_);
		} while (chosenPlaces.find(fP) != chosenPlaces.end());
		chosenPlaces.insert(fP);

		this->flipPlaces_[i] = fP;						// choose the flip place of the vector
		this->flipVals_[i] = this->tmpVec_(fP);			// save the element of a vector before the flip
	}
}

// ##########################################################################################################################################

/**
* @brief Sets the number of random flips and resizes the flipPlaces_ and flipVals_ vectors accordingly.
* 
* @tparam _Ht Template parameter for Hamiltonian type.
* @tparam _T Template parameter for data type.
* @tparam _stateType Template parameter for state type.
* @param _nFlips The number of random flips to set.
*/
template<typename _Ht, typename _T, class _stateType>
inline void NQS_S<2, _Ht, _T, _stateType>::setRandomFlipNum(uint _nFlips)
{
	this->nFlip_ = _nFlips;
	if (this->flipPlaces_.size() != this->nFlip_)
		this->flipPlaces_.resize(this->nFlip_);
	if (this->flipVals_.size() != this->nFlip_)
		this->flipVals_.resize(this->nFlip_);
}

// ##########################################################################################################################################

#endif