#pragma once

/***************************************
* Defines the quadratic lattice Hamiltonian
* class. Allows for later inhertiance
* for a fine model specialization.
* JULY 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#ifndef HAMIL_H
	#include "hamil.h"
#endif

#ifndef HAMIL_QUADRATIC_H
#define HAMIL_QUADRATIC_H

// ######################### EXISTING MODELS ############################
enum MY_MODELS_Q {													 // #
	NONE, FREE_FERMIONS_M, AUBRY_ANDRE_M, SYK2_M, ANDERSON_M		 // #
};																	 // #
BEGIN_ENUM(MY_MODELS)												 // #
{																	 // #
	DECL_ENUM_ELEMENT(NONE),										 // #
	DECL_ENUM_ELEMENT(FREE_FERMIONS_M),								 // #
	DECL_ENUM_ELEMENT(AUBRY_ANDRE_M),								 // #
	DECL_ENUM_ELEMENT(SYK2_M),										 // #
	DECL_ENUM_ELEMENT(ANDERSON_M)									 // #
}																	 // #
END_ENUM(MY_MODELS)                                                  // #	
// ######################################################################

/*
* @brief Allows one to construct a non-interacting Hamiltonian
*/
template <typename _T>
class QuadraticHamiltonian : public Hamiltonian<_T>
{
protected:
	MY_MODELS_Q type_				= MY_MODELS_Q::NONE;
	uint		size_				= 1;
	bool		particleConverving_ = true;
	_T			constant_			= 0.0;
	
	// ------------------ M A N Y   B O D Y ------------------
	double getManyBodyEnergy(const std::vector<uint_fast16_t>& _state);
	double getManyBodyEnergy(u64 _state);

public:
	QuadraticHamiltonian()			= default;
	QuadraticHamiltonian(std::shared_ptr<Lattice> _lat, _T _constant, bool _partCons = true)
		: constant_(_constant), particleConverving_(_partCons)
	{
		this->ran_					= randomGen();
		this->lat_					= _lat;
		this->Ns					= _lat->get_Ns();
		this->size_					= this->particleConverving_ ? this->Ns : 2 * this->Ns;
		this->Nh					= this->size_;
		this->init();
	}
};

// ################################################################################################################################################

/*
* @brief Given a set of single particle indices it creates a many body energy from single particle states
*/
template<typename _T>
inline double QuadraticHamiltonian<_T>::getManyBodyEnergy(const std::vector<uint_fast16_t>& _state)
{
	// if Hamiltonian conserves the number of particles the excitations are easy
	if(this->particleConverving_)
	{
		double _energy	=		0.0;
		for(auto& i: _state)
			_energy		+=		this->eigVal_(i);
		return _energy;
	}
	// otherwise one has to create pairs of states
	else
	{
		double _energy = 0.0;
		// the spectrum is symmetric (2V states)
		for (auto& i : _state)
		{
			auto _idx	=	this->Ns - 1; 
			_energy		+=	this->eigVal_(_idx + i) + this->eigVal_(_idx - i);
		}
		return _energy;
	}
}

/*
* @brief Given a lattice like state of zeros and ones it creates a many body energy from single particle states
*/
template<typename _T>
inline double QuadraticHamiltonian<_T>::getManyBodyEnergy(u64 _state)
{
	std::vector<uint_fast16_t> _idxs = {};
	for (int i = 0; i < this->Ns; ++i)
		if (checkBit(_state, this->Ns))
			_idxs.push_back(i);
	return getManyBodyEnergy(_idxs);
}

// ################################################################################################################################################


#endif