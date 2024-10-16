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

// ########################### EXISTING MODELS #############################
enum class MY_MODELS_Q													// #
{																		// #
	FREE_FERMIONS_M, AUBRY_ANDRE_M, SYK2_M, ANDERSON_M, NONE			// #
};																		// #
BEGIN_ENUMC(MY_MODELS_Q)												// #
{																		// #
		DECL_ENUM_ELEMENT(FREE_FERMIONS_M),								// #
		DECL_ENUM_ELEMENT(AUBRY_ANDRE_M),								// #
		DECL_ENUM_ELEMENT(SYK2_M),										// #
		DECL_ENUM_ELEMENT(ANDERSON_M),									// #
		DECL_ENUM_ELEMENT(NONE)											// #
}																		// #
END_ENUMC(MY_MODELS_Q)													// #	
////////////////////////////////////////////////////////////////////////////
template <>																// #
inline std::string str_p(const MY_MODELS_Q v, 							// #
	const int n, 														// #
	bool scientific)													// #
{																		// #
	return str_p(static_cast<std::underlying_type_t<MY_MODELS_Q>>(v),	// # 
		n, scientific);													// #
}																		// #
////////////////////////////////////////////////////////////////////////////
inline bool isQuadraticRandom(uint _type)								// #
{																		// #
	return	_type == (uint)MY_MODELS_Q::ANDERSON_M ||					// #
		_type == (uint)MY_MODELS_Q::SYK2_M;								// #
}																		// #
inline std::string filenameQuadraticRandom(std::string _f,				// #
	uint _type,															// #
	randomGen& ran)														// #
{																		// #
	if (isQuadraticRandom(_type))										// #
		return _f + "_R=" + STR(ran.randomInt(0, 1000));				// #
	return _f;
}																		// #
// #########################################################################

/*
* @brief Allows one to construct a non-interacting Hamiltonian
*/
template <typename _T>
class QuadraticHamiltonian : public Hamiltonian<_T>
{
public:
	using		NQSFun				= std::function<cpx(std::initializer_list<int>, std::initializer_list<double>)>;
	using		manyBodyTuple		= std::tuple<v_1d<double>, v_1d<arma::uvec>>;

protected:
	MY_MODELS_Q type_				= MY_MODELS_Q::NONE;
	uint		size_				= 1;
	// check the particle conservation
	bool		particleConverving_ = true;
	_T			constant_		= 0.0;

public:
	// ------------------ M A N Y   B O D Y ------------------
	void getManyBodyEnergies(uint N, v_1d<double>& manyBodySpectrum, v_1d<arma::uvec>& manyBodyOrbitals, int _num = -1);
	void getManyBodyEnergiesZero(uint N, v_1d<double>& manyBodySpectrum, v_1d<arma::uvec>& manyBodyOrbitals, int _num = -1);

	// energy
	template<typename _T2, typename = typename std::enable_if<std::is_arithmetic<_T2>::value, _T2>::type>
	double getManyBodyEnergy(const v_1d<_T2>& _state);
	double getManyBodyEnergy(u64 _state);

	// state
	template<typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	arma::Mat<_T> getSlater(const v_1d<_T1>& _singlePartOrbs, u64 _realSpaceOccupation);

	template<typename _T1, typename _T2, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	arma::Col<_T> getManyBodyState(const v_1d<_T1>& _singlePartOrbs, const Hilbert::HilbertSpace<_T2>& _hibert);

	// --------------- C O N S T R U C T O R S ---------------
	~QuadraticHamiltonian()			= default;
	QuadraticHamiltonian()			= default;
	QuadraticHamiltonian(std::shared_ptr<Lattice> _lat, _T _constant, bool _partCons = true)
		: particleConverving_(_partCons), constant_(_constant)
	{
		LOGINFO("Creating quadratic model: ", LOG_TYPES::CHOICE, 1);
		this->ran_	= randomGen();
		this->lat_	= _lat;
		this->Ns	= _lat->get_Ns();
		this->size_ = this->particleConverving_ ? this->Ns : 2 * this->Ns;
		this->Nh	= this->size_;
		this->init();
	}

	// ########### GETTERS ###########
	virtual auto getTransMat()		-> arma::Mat<_T>		{ return this->eigVec_;							};	// returns the unitary base transformation {<x|q>}
	virtual auto getSPEnMat()		-> arma::Col<double>	{ return this->eigVal_;							};	// returns the single particle energies
	auto getTypeI()					const -> uint			{ return (uint)this->type_;						};	// get type integer
	auto getType()					const -> std::string	{ return getSTR_MY_MODELS_Q((uint)this->type_); };	// get type string

	// ########### OVERRIDE ##########
	void locEnergy(u64 _elemId, u64 _elem, uint _site)		override {										};
	cpx locEnergy(u64 _id, uint s, NQSFun f1)				override { return 0.0;							};
	cpx locEnergy(const DCOL& v,
		uint site, NQSFun f1,
		DCOL& tmp)											override { return 0.0;							};

};

// ##################################################################################################################################
// ##################################################################################################################################
// ####################################################### E N E R G Y ##############################################################
// ##################################################################################################################################
// ##################################################################################################################################

/*
* @brief Given a set of indices, it creates a many body energy from single particle states. For example, having [0, 1, 1, 0, 1],
* the procedure will use orbitals N_1, N_2, N_4 to calculate the many body energy as a sum of single particle orbital energies.
* @param getManyBodyEnergy _state set of indices that indicate single particle states occupied
* @returns many body energy
*/
template<typename _T>
template<typename _T2, typename>
inline double QuadraticHamiltonian<_T>::getManyBodyEnergy(const v_1d<_T2>& _state)
{
	// if Hamiltonian conserves the number of particles the excitations are easy
	if (this->particleConverving_)
	{
		double _energy = 0.0;
		for (auto& i : _state)
			_energy += this->eigVal_(i);
		return _energy;
	}
	// otherwise one has to create pairs of states
	else
	{
		double _energy = 0.0;
		// the spectrum is symmetric (2V states)
		for (auto& i : _state)
		{
			auto _idx = this->Ns - 1;
			_energy += this->eigVal_(_idx + i) + this->eigVal_(_idx - i);
		}
		return _energy;
	}
}

/*
* @brief Given an integer, it transforms it to a set of indices. For the people that like binary!
* It creates a many body energy from single particle states. For example, having [0, 1, 1, 0, 1],
* the procedure will use orbitals N_1, N_2, N_4 to calculate the many body energy as a sum of single particle orbital energies.
* @param getManyBodyEnergy _state set of indices that indicate single particle states occupied
* @returns many body energy
*/
template<typename _T>
inline double QuadraticHamiltonian<_T>::getManyBodyEnergy(u64 _state)
{
	v_1d<uint_fast16_t> _idxs = {};
	for (int i = 0; i < this->Ns; ++i)
		if (checkBit(_state, this->Ns))
			_idxs.push_back(i);
	return this->getManyBodyEnergy(_idxs);
}

// ##################################################################################################################################
// ##################################################################################################################################
// ##################################################### E N E R G I E S ############################################################
// ##################################################################################################################################
// ##################################################################################################################################

/*
* @brief Create combination of quasiparticle orbitals to obtain the many body product states...
* @warning Using zero energy (based on the knowledge of model)
* @param N number of particles
* @param _num number of combinations
*/
template<typename _T>
inline void QuadraticHamiltonian<_T>::getManyBodyEnergiesZero(uint N, v_1d<double>& manyBodySpectrum, v_1d<arma::uvec>& manyBodyOrbitals, int _num)
{
	LOGINFO("Looking for degenerate zero energy many body states only!", LOG_TYPES::CHOICE, 2);
	if (this->Ns % 4 != 0)
		throw std::runtime_error(std::string("Method is not suitable for such system sizes..."));

	// create orbitals
	v_1d<uint> orbitals;
	for (int i = 1; i < this->Ns / 2 - 1; ++i)
		orbitals.push_back(i);

	// resize
	manyBodySpectrum			= {};
	manyBodyOrbitals			= {};

	// get through combinations!
	for (int i = 0; i < _num; ++i)
	{
		// create combination
		auto _combinationTmp	= this->ran_.choice(orbitals, int(N / 2));

		// if we cannot create more combinations...
		if (_combinationTmp.size() < int(N / 2))
			break;

		auto _combination		= _combinationTmp;

		// push the rest...
		for (auto _comb : _combinationTmp)
		{
			// #1 to make the energy zero
			_combination.push_back(_comb + int(this->Ns / 2));
			//// #2 to make the momentum zero 
			//_combination.push_back(this->Ns - _comb);
			//// #3 to make the energy zero again and momentum with #1
			//_combination.push_back(this->Ns - _comb - int(this->Ns / 2));
		}
		//std::sort(_combination.begin(), _combination.end());

		// transform to uvec
		arma::uvec _combinationV(N);
		for (int j = 0; j < _combination.size(); j++)
			_combinationV(j) = _combination[j];

		// append
		manyBodyOrbitals.push_back(_combinationV);

		// get energy
		double _manyBodyEn = this->getManyBodyEnergy(_combination);
		manyBodySpectrum.push_back(_manyBodyEn);
	}
}

/*
* @brief Create combination of quasiparticle orbitals to obtain the many body product states...
* @param N number of particles out of the single particle sectors
* @param manyBodySpectrum save the many body energies here!
* @param manyBodyOrbitals save the many body orbitals there - choose N ones out of a vector of length V.
* @param _num number of combinations
*/
template<typename _T>
inline void QuadraticHamiltonian<_T>::getManyBodyEnergies(uint N, v_1d<double>& manyBodySpectrum, v_1d<arma::uvec>& manyBodyOrbitals, int _num)
{
	// resize, those will save the eigenspectrum of many body states
	manyBodySpectrum = {};
	manyBodyOrbitals = {};

	// create orbitals and set them to zero
	v_1d<uint> orbitals(this->Ns);
	std::iota(orbitals.begin(), orbitals.end(), 0);

	// get through combinations!
#pragma omp parallel for
	for (int i = 0; i < _num; ++i)
	{
		// create combination (choose N elements and set them to 1)
		auto _combination = this->ran_.choice(orbitals, N);

		// transform to uvec
		arma::uvec _combinationV(_combination.size());

		for (int j = 0; j < _combination.size(); j++)
			_combinationV(j) = _combination[j];

		// append
#pragma omp critical
		manyBodyOrbitals.push_back(_combinationV);

		// get energy
		double _manyBodyEn = this->getManyBodyEnergy(_combination);
#pragma omp critical
		manyBodySpectrum.push_back(_manyBodyEn);
	}
}

// ################################################################################################################################################

/*
* @brief Sets the slater determinant in order to create a single coefficient of a state.
* @param _singlePartOrbs single particle orbitals that are occupied
* @param _realSpaceOccupations state that represents real space occupations
*/
template<typename _T>
template<typename _T1, typename>
inline arma::Mat<_T> QuadraticHamiltonian<_T>::getSlater(const v_1d<_T1>& _singlePartOrbs, u64 _realSpaceOccupations)
{
	// get the number of particles to set the Slater matrix
	int _particleNumber = _singlePartOrbs.size();

	// create Slater matrix
	arma::Mat<_T> _slater(_particleNumber, _particleNumber, arma::fill::zeros);

	//// if wrong number of particles, skip calculations
	//if (std::accumulate(_realSpaceOccupations.begin(), _realSpaceOccupations.end()) != _particleNumber)
	//	return {};

	// go through the basis states with a given particle number (rows - real space vectors)
	int iterator = 0;
	for (auto i = 0; i < this->Ns; ++i)
	{
		if (!checkBit(_realSpaceOccupations, i))
			continue;
		// go through the orbitals
		for (auto j = 0; j < _particleNumber; j++)
			_slater(iterator, j) = this->eigVec_(_singlePartOrbs[j], i);
		iterator++;
	}
	return _slater;
}

/*
* @brief For a given single particle state (product of single particle orbitals),
* creates a many body state in a full Hilbert space by calculating the Slater determinants.
* @param _singlePartOrbs set of indices indicating taken single particle orbitals
* @param _hilbert class containing the information about the full Hilbert space
*/
template<typename _T>
template<typename _T1, typename _T2, typename>
inline arma::Col<_T> QuadraticHamiltonian<_T>::getManyBodyState(const v_1d<_T1>& _singlePartOrbs, const Hilbert::HilbertSpace<_T2>& _hilbert)
{
	// check the number of single particle orbitals
	auto _singlePartOrb = _singlePartOrbs.size();

	if (!this->particleConverving_)
		throw std::runtime_error("This Hamiltonian does not have the eigenstates in Slater determinant form!");

	// save the state 
	arma::Col<_T> _stateOut(_hilbert.getHilbertSize(), arma::fill::zeros);

	// go through the Hilbert space basis
	for (u64 i = 0; i < _hilbert.getHilbertSize(); ++i)
	{
		auto _state = _hilbert.getMapping(i);
		if (std::popcount(_state) != _singlePartOrb)
			continue;
		// set the element of the vector to slater determinant
		auto _eigSlater		= arma::eig_gen(this->getSlater(_singlePartOrbs, _state));
		auto _prodSlatter	= arma::prod(_eigSlater);
		_stateOut(i)		= toType<_T>(std::real(_prodSlatter), std::imag(_prodSlatter));
	}
	return _stateOut;
}

#endif