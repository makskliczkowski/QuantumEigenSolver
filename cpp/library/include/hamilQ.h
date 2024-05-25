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
////////////////////////////////////////////////////////////////////////////
template <>																// #
inline std::string str_p(const MY_MODELS v, 							// #
	const int n, 														// #
	bool scientific)													// #
{																		// #
	return str_p(static_cast<std::underlying_type_t<MY_MODELS>>(v),		// # 
		n, scientific);													// #
}																		// #
////////////////////////////////////////////////////////////////////////////
inline bool isQuadraticRandom(uint _type)								// #
{																		// #
	return	_type == (uint)MY_MODELS::ANDERSON_M ||						// #
			_type == (uint)MY_MODELS::SYK2_M	 ||						// #
			_type == (uint)MY_MODELS::POWER_LAW_RANDOM_BANDED_M;		// #
}																		// #
inline std::string filenameQuadraticRandom(std::string _f,				// #
	uint _type,															// #
	randomGen& ran)														// #
{																		// #
	if (isQuadraticRandom(_type))										// #
		return _f + "_R=" + STR(ran.randomInt<int>(0, 1000));			// #
	return _f;															// #
}																		// #
// #########################################################################

/*
* @brief Allows one to construct a non-interacting Hamiltonian - can be transformed to spinless fermions 
*/
template <typename _T>
class QuadraticHamiltonian : public Hamiltonian<_T, 2>
{
	// ------------------------------------------- CLASS TYPES ----------------------------------------------
public:
	using		NQSFun				= std::function<cpx(std::initializer_list<int>, std::initializer_list<double>)>;
	using		manyBodyTuple		= std::tuple<v_1d<double>, v_1d<arma::uvec>>;

protected:
	MY_MODELS  type_				= MY_MODELS::NONE;
	uint		size_				= 1;
	// check the particle conservation
	bool		particleConverving_ = true;
	// constant to be added to the energy
	_T			constant_			= 0.0;

	// ------------------ M A N Y   B O D Y ------------------
public:
	void getManyBodyOrbitals(uint N, v_1d<uint> _orbitals, v_1d<v_1d<uint>>& manyBodyOrbitals);
	void getManyBodyOrbitals(uint N, v_1d<uint> _orbitals, v_1d<v_1d<uint>>& manyBodyOrbitals, uint _num, uint _threadNum = 1);
	void getManyBodyEnergies(v_1d<double>& manyBodySpectrum, const v_1d<v_1d<uint>>& manyBodyOrbitals, uint _threadNum = 1);

	// ------------------ energy
	template<typename _T2, typename _A, typename = typename std::enable_if<std::is_arithmetic<_T2>::value, _T2>::type>
	double getManyBodyEnergy(const std::vector<_T2, _A>& _state);
	double getManyBodyEnergy(const v_1d<std::seed_seq::result_type>& _state);
	double getManyBodyEnergy(u64 _state);

	// ----------------- Particle Conserving -----------------

	// ------------------ slater
	template<typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	arma::Mat<_T> getSlater(const v_1d<_T1>& _singlePartOrbs, u64 _realSpaceOccupation);

	template<typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	arma::Mat<_T> getSlater(const arma::Col<_T1>& _singlePartOrbs, u64 _realSpaceOccupation);

	template<typename _T1>
	arma::Mat<_T> getSlater(const _T1& _singlePartOrbs, u64 _realSpaceOccupation, arma::Mat<_T>& _slater);

	// ------------------ state from orbitals
	template<typename _T1, typename _T2, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	arma::Col<_T> getManyBodyState(const v_1d<_T1>& _singlePartOrbs, const Hilbert::HilbertSpace<_T2>& _hilbert);

	template<typename _T1, typename _T2, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	arma::Col<_T> getManyBodyState(const arma::Col<_T1>& _singlePartOrbs, const Hilbert::HilbertSpace<_T2>& _hilbert);

	template<typename _T1, typename _T2>
	arma::Col<_T> getManyBodyState(const _T1& _singlePartOrbs, const Hilbert::HilbertSpace<_T2>& _hilbert, arma::Mat<_T2>& _slater);

	// --------------- C O N S T R U C T O R S ---------------
	virtual ~QuadraticHamiltonian()	= default;
	QuadraticHamiltonian()			= default;
	QuadraticHamiltonian(size_t _Ns, _T _constant, bool _partCons = true, bool _isSparse = true)
	{
		LOGINFO("Creating quadratic model: ", LOG_TYPES::CHOICE, 1);
		this->ran_	= randomGen();
		this->Ns	= _Ns;
		this->Ns_	= _Ns;
		this->size_ = _partCons ? _Ns : 2 * _Ns;
		this->Nh	= this->size_;
		this->Nh_	= this->size_;
		this->isManyBody_	= false;
		this->isQuadratic_	= true;
		this->isSparse_		= _isSparse;
		this->init();

	}
	QuadraticHamiltonian(std::shared_ptr<Lattice> _lat, _T _constant, bool _partCons = true, bool _isSparse = true)
		: particleConverving_(_partCons), constant_(_constant)
	{
		LOGINFO("Creating quadratic model: ", LOG_TYPES::CHOICE, 1);
		this->ran_	= randomGen();
		this->lat_	= _lat;
		this->Ns	= _lat->get_Ns();
		this->Ns_	= _lat->get_Ns();
		this->size_ = this->particleConverving_ ? this->Ns : 2 * this->Ns;
		this->Nh	= this->size_;
		this->Nh_	= this->size_;
		this->isManyBody_	= false;
		this->isQuadratic_	= true;
		this->isSparse_		= _isSparse;
		this->init();
	}

	// -------------------- G E T T E R S --------------------
	virtual auto getTransMat()		-> arma::Mat<_T>		{ return this->eigVec_;							};	// returns the unitary base transformation {<x|q>}
	virtual auto getSPEnMat()		-> arma::Col<double>	{ return this->eigVal_;							};	// returns the single particle energies
	auto getTypeI()					const -> uint			{ return (uint)this->type_;						};	// get type integer
	auto getType()					const -> std::string	{ return getSTR_MY_MODELS(this->type_);	};	// get type string

	// ------------------- O V E R R I D E -------------------
	void locEnergy(u64 _elemId, u64 _elem, uint _site)		override {										};
	cpx locEnergy(u64 _id, uint s, NQSFun f1)				override { return 0.0;							};
	cpx locEnergy(const DCOL& v,
					uint site, NQSFun f1)					override { return 0.0;							};

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
template<typename _T2, typename _A, typename>
inline double QuadraticHamiltonian<_T>::getManyBodyEnergy(const std::vector<_T2, _A>& _state)
{
	double _energy = 0.0;
	// if Hamiltonian conserves the number of particles the excitations are easy
	if (this->particleConverving_)
	{
		for (auto& i : _state)
			_energy += this->eigVal_(i);
	}
	// otherwise one has to create pairs of Fermions (the Bogolubov quasiparticles)
	else
	{
		// the spectrum is symmetric (2V states)
		for (auto& i : _state)
		{
			auto _idx = this->Ns - 1;
			_energy += this->eigVal_(_idx + i) + this->eigVal_(_idx - i);
		}
	}
	return _energy;
}

// -----------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Given a set of indices, it creates a many body energy from single particle states. For example, having [1, 3, 5],
* the procedure will use orbitals N_1, N_3, N_5 to calculate the many body energy as a sum of single particle orbital energies.
* @param _state set of indices that indicate single particle states
* */
template<typename _T>
inline double QuadraticHamiltonian<_T>::getManyBodyEnergy(const v_1d<std::seed_seq::result_type>& _state)
{
	double _energy = 0.0;
	// if Hamiltonian conserves the number of particles the excitations are easy
	if (this->particleConverving_)
	{
		for (int j = 0; j < _state.size(); j++)
		{
			auto _idx	= _state[j];
			_energy		+= this->eigVal_(_idx);
		}
	}
	// otherwise one has to create pairs of Fermions (the Bogolubov quasiparticles)
	else
	{
		// the spectrum is symmetric (2V states)
		for (int j = 0; j < _state.size(); j++)
		{
			auto _idx	= this->Ns - 1;
			_energy		+= this->eigVal_(_idx + _state[j]) + this->eigVal_(_idx - _state[j]);
		}
	}
	return _energy;
}

// ----------------------------------------------------------------------------------------------------------------------------------

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
* @param N number of particles out of the single particle sectors
* @param _orbitals set of many single particle states selection
* @param manyBodyOrbitals save the many body orbitals there - choose N ones out of a vector of length V.
*/
template<typename _T>
inline void QuadraticHamiltonian<_T>::getManyBodyOrbitals(uint N, v_1d<uint> _orbitals, v_1d<v_1d<uint>>& manyBodyOrbitals)
{
	// only correct orbitals are to be given
	if (this->Ns != _orbitals.size())
		throw std::runtime_error(std::string("Wrong number of orbitals given!"));

	// clear me!
	manyBodyOrbitals.clear();
	manyBodyOrbitals = Vectors::combinations(_orbitals, N);
}

// ##################################################################################################################################

/*
* @brief Create combination of quasiparticle orbitals to obtain the many body product states... Does so by random selection.
* @param N number of particles out of the single particle sectors
* @param _orbitals set of many single particle states selection
* @param manyBodyOrbitals save the many body orbitals there - choose N ones out of a vector of length V.
* @param _num number of many body states
* @param _threadNum number of threads
*/
template<typename _T>
inline void QuadraticHamiltonian<_T>::getManyBodyOrbitals(uint N, v_1d<uint> _orbitals, v_1d<v_1d<uint>>& manyBodyOrbitals, uint _num, uint _threadNum)
{
	// only correct orbitals are to be given
	if (this->Ns != _orbitals.size())
		throw std::runtime_error(std::string("Wrong number of orbitals given!"));

	// clear me!
	manyBodyOrbitals.clear();
	manyBodyOrbitals.resize(_num);

	// go through random iterations
#pragma omp parallel for num_threads(_threadNum)
	for (int i = 0; i < _num; ++i)
		manyBodyOrbitals[i] = this->ran_.choice(_orbitals, N);
}

// ##################################################################################################################################

/*
* @brief Create combination of quasiparticle orbitals to obtain the many body product states...
* @param manyBodySpectrum save the many body energies here!
* @param manyBodyOrbitals set of many single particle states selection 
* @param _threadNum number of threads
*/
template<typename _T>
inline void QuadraticHamiltonian<_T>::getManyBodyEnergies(v_1d<double>& manyBodySpectrum, const v_1d<v_1d<uint>>& manyBodyOrbitals, uint _threadNum)
{
	// resize, those will save the eigenspectrum of many body states
	manyBodySpectrum.clear();
	manyBodySpectrum.resize(manyBodyOrbitals.size());

	// get through combinations!
#ifndef _DEBUG
#	pragma omp parallel for num_threads(_threadNum)
#endif
	for (auto i = 0; i < manyBodyOrbitals.size(); ++i)
		manyBodySpectrum[i] = (this->getManyBodyEnergy(manyBodyOrbitals[i]));
}

// ##################################################################################################################################

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
	const int _particleNumber = _singlePartOrbs.size();

	// create Slater matrix
	arma::Mat<_T> _slater(_particleNumber, _particleNumber, arma::fill::zeros);

	// go through the basis states with a given particle number (rows - real space vectors)
	int iterator = 0;
	for (int i = 0; i < this->Ns; ++i)
	{
		if (!checkBit(_realSpaceOccupations, i))
			continue;

		// check if we can do this via Armadillo
		const auto eigVecRow	= this->eigVec_.col(i).subvec(_singlePartOrbs.data(), _particleNumber);
		_slater.row(iterator++) = arma::conv_to<arma::Row<_T>>::from(eigVecRow);

		// go through the orbitals
		//for (auto j = 0; j < _particleNumber; j++)
			//_slater(iterator, j) = this->eigVec_(_singlePartOrbs[j], i);
		//iterator++;
	}
	return _slater;
}

// -----------------------------------------------------------------------------------------------------------------------------------

template<typename _T>
template<typename _T1, typename>
inline arma::Mat<_T> QuadraticHamiltonian<_T>::getSlater(const arma::Col<_T1>& _singlePartOrbs, u64 _realSpaceOccupations)
{
	// get the number of particles to set the Slater matrix
	const int _particleNumber = _singlePartOrbs.size();

	// create Slater matrix
	arma::Mat<_T> _slater(_particleNumber, _particleNumber, arma::fill::zeros);

	// go through the basis states with a given particle number (rows - real space vectors)
	int iterator = 0;
	for (auto i = 0; i < this->Ns; ++i)
	{
		if (!checkBit(_realSpaceOccupations, i))
			continue;

		// Extract the required elements from eigVec_ using matrix slicing
		_slater.row(iterator++) = this->eigVec_.submat(_singlePartOrbs, arma::span(i, i));

		//// go through the orbitals
		//for (auto j = 0; j < _particleNumber; j++)
		//	_slater(iterator, j) = this->eigVec_(_singlePartOrbs(j), i);
		//iterator++;
	}
	return _slater;
}

// -----------------------------------------------------------------------------------------------------------------------------------

template<typename _T>
template<typename _T1>
inline arma::Mat<_T> QuadraticHamiltonian<_T>::getSlater(const _T1& _singlePartOrbs, u64 _realSpaceOccupations, arma::Mat<_T>& _slater)
{
	// get the number of particles to set the Slater matrix
	int _particleNumber = _singlePartOrbs.size();

	// create Slater matrix
	if(_slater.n_rows != _particleNumber && _slater.n_rows != _slater.n_cols)
		_slater = arma::Mat<_T>(_particleNumber, _particleNumber, arma::fill::zeros);

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

// ################################################################################################################################################

/*
* @brief For a given single particle state (product of single particle orbitals),
* creates a many body state in a full Hilbert space by calculating the Slater determinants.
* @param _singlePartOrbs set of indices indicating taken single particle orbitals
* @param _hilbert class containing the information about the full Hilbert space
* !TODO Implement Bogolubov quasiparticles!
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
		// get mapping to the Hilbert space
		auto _state			= _hilbert.getMapping(i);
		if (std::popcount(_state) != _singlePartOrb)
			continue;
		// set the element of the vector to slater determinant
		auto _prodSlatter	= arma::prod(arma::eig_gen(this->getSlater(_singlePartOrbs, _state)));
		_stateOut(i)		= toType<_T>(std::real(_prodSlatter), std::imag(_prodSlatter));
	}
	return _stateOut;
}


// -----------------------------------------------------------------------------------------------------------------------------------

/*
* @brief For a given single particle state (product of single particle orbitals),
* creates a many body state in a full Hilbert space by calculating the Slater determinants.
* @param _singlePartOrbs set of indices indicating taken single particle orbitals
* @param _hilbert class containing the information about the full Hilbert space
* !TODO Implement Bogolubov quasiparticles!
*/
template<typename _T>
template<typename _T1, typename _T2, typename>
inline arma::Col<_T> QuadraticHamiltonian<_T>::getManyBodyState(const arma::Col<_T1>& _singlePartOrbs, const Hilbert::HilbertSpace<_T2>& _hilbert)
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
		auto _prodSlatter	= arma::prod(arma::eig_gen(this->getSlater(_singlePartOrbs, _state)));
		_stateOut(i)		= toType<_T>(std::real(_prodSlatter), std::imag(_prodSlatter));
	}
	return _stateOut;
}

// -----------------------------------------------------------------------------------------------------------------------------------

/*
* @brief For a given single particle state (product of single particle orbitals),
* creates a many body state in a full Hilbert space by calculating the Slater determinants.
* @param _singlePartOrbs set of indices indicating taken single particle orbitals
* @param _hilbert class containing the information about the full Hilbert space
* @param _slater matrix that contains the Slater determinants (to be saved onto)
* !TODO Implement Bogolubov quasiparticles!
*/
template<typename _T>
template<typename _T1, typename _T2>
inline arma::Col<_T> QuadraticHamiltonian<_T>::getManyBodyState(const _T1& _singlePartOrbs, const Hilbert::HilbertSpace<_T2>& _hilbert, arma::Mat<_T2>& _slater)
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
		auto _prodSlatter	= arma::prod(arma::eig_gen(this->getSlater(_singlePartOrbs, _state, _slater)));
		_stateOut(i)		= toType<_T>(std::real(_prodSlatter), std::imag(_prodSlatter));
	}
	return _stateOut;
}

#endif