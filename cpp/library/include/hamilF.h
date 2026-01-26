#pragma once
/***************************************
* Defines the fermionic lattice Hamiltonian
* class. Allows for later inhertiance
* for a fine model specialization.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#ifndef HAMIL_H
#	include "hamil.h"
#endif

#ifndef HAMIL_FERMIONIC_H
#define HAMIL_FERMIONIC_H

// ########################### EXISTING MODELS #############################
enum class MY_MODELS_F													// #
{																		// #
	HUBBARD_M, NONE														// #
};																		// #
BEGIN_ENUMC(MY_MODELS_F)												// #
{																		// #
		DECL_ENUM_ELEMENT(HUBBARD_M),									// #
		DECL_ENUM_ELEMENT(NONE)											// #
}																		// #
END_ENUMC(MY_MODELS_F)													// #	
// #########################################################################

/**
* @brief Allows one to construct a interacting Hamiltonian
*/
template <typename _T>
class FermionicHamiltonian : public Hamiltonian<_T, 4>
{
	// ------------------------------------------- CLASS TYPES ----------------------------------------------
public:
	using		NQSFun			= std::function<cpx(std::initializer_list<int>, std::initializer_list<double>)>;
	using		manyBodyTuple	= std::tuple<v_1d<double>, v_1d<arma::uvec>>;
	// ------------------------------------------- CLASS FIELDS ---------------------------------------------
protected:
	MY_MODELS_F type_			= MY_MODELS_F::NONE;

public:
	virtual ~FermionicHamiltonian() { LOGINFO("Fermionic Hamiltonian destructor called.", LOG_TYPES::INFO, 4);		};
	FermionicHamiltonian()		= default;


public:
	auto getTypeI()				const -> uint				{ return (uint)this->type_;								};	// get type integer
	auto getType()				const -> std::string		{ return getSTR_MY_MODELS_F((uint)this->type_);			};	// get type string

	// ------------------- O V E R R I D E -------------------
	void locEnergy(u64 _elemId, u64 _elem, uint _site)		override;
	cpx locEnergy(u64 _id, uint s, NQSFun f1)				override;
	cpx locEnergy(const DCOL& v, uint site, 
				  NQSFun f1, DCOL& tmp)						override;
};

#endif