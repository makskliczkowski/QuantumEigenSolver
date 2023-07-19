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
enum class MY_MODELS_Q{												 // #
	FREE_FERMIONS_M, AUBRY_ANDRE_M, SYK2_M, ANDERSON_M, NONE		 // #
};																	 // #
BEGIN_ENUMC(MY_MODELS_Q)										     // #
{																	 // #
	DECL_ENUM_ELEMENT(FREE_FERMIONS_M),								 // #
	DECL_ENUM_ELEMENT(AUBRY_ANDRE_M),								 // #
	DECL_ENUM_ELEMENT(SYK2_M),										 // #
	DECL_ENUM_ELEMENT(ANDERSON_M),									 // #
	DECL_ENUM_ELEMENT(NONE)											 // #
}																	 // #
END_ENUMC(MY_MODELS_Q)												 // #	
template <>															 // #
inline std::string str_p(const MY_MODELS_Q v, 						 // #
						 const int n, 								 // #
						 bool scientific)							 // #
{																	 // #
	return str_p(static_cast<std::underlying_type_t<MY_MODELS_Q>>(v),// # 
				 n, scientific);									 // #
}																	 // #
// ######################################################################

/*
* @brief Allows one to construct a non-interacting Hamiltonian
*/
template <typename _T>
class QuadraticHamiltonian : public Hamiltonian<_T>
{
public:
	using		NQSFun					= std::function<cpx(std::initializer_list<int>, std::initializer_list<double>)>;
	using		manyBodyTuple			= std::tuple<std::vector<double>, std::vector<arma::uvec>>;
protected:
	MY_MODELS_Q type_					= MY_MODELS_Q::NONE;
	uint		size_					= 1;
	bool		particleConverving_		= true;
	_T			constant_				= 0.0;
	
	// ------------------ M A N Y   B O D Y ------------------
public:
	void getManyBodyEnergies(uint N, std::vector<double>& manyBodySpectrum, std::vector<arma::uvec>& manyBodyOrbitals, int _num = -1);
	void getManyBodyEnergiesZero(uint N, std::vector<double>& manyBodySpectrum, std::vector<arma::uvec>& manyBodyOrbitals, int _num = -1);

	double getManyBodyEnergy(const std::vector<uint_fast16_t>& _state);
	double getManyBodyEnergy(u64 _state);

	~QuadraticHamiltonian()			= default;
	QuadraticHamiltonian()			= default;
	QuadraticHamiltonian(std::shared_ptr<Lattice> _lat, _T _constant, bool _partCons = true)
		: particleConverving_(_partCons), constant_(_constant)
	{
		LOGINFO("Creating quadratic model: ", LOG_TYPES::CHOICE, 1);
		this->ran_					= randomGen();
		this->lat_					= _lat;
		this->Ns					= _lat->get_Ns();
		this->size_					= this->particleConverving_ ? this->Ns : 2 * this->Ns;
		this->Nh					= this->size_;
		this->init();
	}

	// ########### GETTERS ###########
	virtual auto getTransMat()				-> arma::Mat<_T>		{ return this->eigVec_; };
	virtual auto getSPEnMat()				-> arma::Col<double>	{ return this->eigVal_; };
	auto getTypeI()							const -> uint			{ return (uint)this->type_; };
	auto getType()							const -> std::string	{ return getSTR_MY_MODELS_Q(this->type_); };

	// ########### OVERRIDE ##########
	void locEnergy(u64 _elemId, u64 _elem, uint _site)	override {};
	cpx locEnergy(u64 _id, uint s, NQSFun f1)			override { return 0.0;  };
	cpx locEnergy(const arma::Col<double>& v, uint site, NQSFun f1, arma::Col<double>& tmp) override { return 0.0; };

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

/*
* @brief Create combination of quasiparticle orbitals to obtain the many body product states... 
* @warning Using zero energy and zero total momemtum (based on the knowledge of model)
* @param N number of particles
* @param _num number of combinations
*/
template<typename _T>
inline void QuadraticHamiltonian<_T>::getManyBodyEnergiesZero(uint N, std::vector<double>& manyBodySpectrum, std::vector<arma::uvec>& manyBodyOrbitals, int _num)
{
	if (this->Ns % 8 != 0)
		throw std::runtime_error(std::string("Method is not suitable for such system sizes..."));
	
	// create orbitals
	std::vector<uint> orbitals;
	for (int i = 1; i < (int(this->Ns / 4) - 1); ++i)
		orbitals.push_back(i);

	// resize
	manyBodySpectrum = {};
	manyBodyOrbitals = {};

	// get through combinations!
	for (int i = 0; i < _num; ++i)
		{
			// create combination
			auto _combinationTmp	=	this->ran_.choice(orbitals, int(N / 4));

			// if we cannot create more combinations...
			if (_combinationTmp.size() < int(N / 4))
				break;

			auto _combination		=	_combinationTmp;

			// push the rest...
			for (auto _comb : _combinationTmp)
			{
				_combination.push_back(_comb + int(this->Ns / 2));
				_combination.push_back(this->Ns - _comb);
				_combination.push_back(this->Ns - _comb - int(this->Ns / 2));
			}

			// transform to uvec
			arma::uvec _combinationV(N);
			for (int j = 0; j < _combination.size(); j++)
				_combinationV(j)	=	_combination[j];
			// append
			manyBodyOrbitals.push_back(_combinationV);
			// get energy
			double _manyBodyEn		=	this->getManyBodyEnergy(_combination);
			manyBodySpectrum.push_back(_manyBodyEn);
		}
}

/*
* @brief Create combination of quasiparticle orbitals to obtain the many body product states...
* @param N number of particles
* @param _num number of combinations
*/
template<typename _T>
inline void QuadraticHamiltonian<_T>::getManyBodyEnergies(uint N, std::vector<double>& manyBodySpectrum, std::vector<arma::uvec>& manyBodyOrbitals, int _num)
{
	// resize
	manyBodySpectrum = {};
	manyBodyOrbitals = {};

	// create orbitals
	std::vector<uint> orbitals(this->Ns);
	std::iota(orbitals.begin(), orbitals.end(), 0);

	// get through combinations!
	for (int i = 0; i < _num; ++i)
	{
		// create combination
		auto _combination		=	this->ran_.choice(orbitals, N);
		// transform to uvec
		arma::uvec _combinationV(_combination.size());
		for (int j = 0; j < _combination.size(); j++)
			_combinationV(j)	=	_combination[j];
		// append
		manyBodyOrbitals.push_back(_combinationV);
		// get energy
		double _manyBodyEn		=	this->getManyBodyEnergy(_combination);
		manyBodySpectrum.push_back(_manyBodyEn);
	}
}

// ################################################################################################################################################

#endif