/**
* @file rbm_pp.h
* @brief Header file for the RBM_PP class, which implements the Restricted Boltzmann Machines (RBM) ansatz with Pair Product (PP) reference state for Neural Quantum States (NQS).
* 
* This class provides the implementation of the RBM ansatz with an additional Pair Product reference state, which is used to model quantum states. The class includes methods for setting and updating the state, calculating probabilities, managing weights, and computing the Pfaffian for the PP matrix.
* 
* References:
* - Journal of the Physical Society of Japan Vol. 77, No. 11, November, 2008, 114701
* - https://doi.org/10.1103/PhysRevB.96.205152
* - https://doi.org/10.1103/PhysRevX.11.031034
* 
* @copyright
* Maksymilian Kliczkowski, Wroclaw University of Science and Technology
* 2024
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the wave function (default is the same as Hamiltonian type).
* @tparam _stateType Data type for the state (default is double).
*/
#pragma once
#ifndef RBMPP_H
#	define RBMPP_H
#	ifndef RBM_H
#		include "../rbm_final.hpp"
#	endif // !NQS_H
#	define NQS_RBM_PP_USE_PFAFFIAN_UPDATE

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! B A S E !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief Restricted Boltzmann Machine with Pair Product (RBM-PP) State Ansatz
* 
* @details This class implements a neural-network quantum state ansatz that combines 
* a Restricted Boltzmann Machine (RBM) with a pair-product (PP) state. The PP state 
* introduces additional correlations between pairs of particles through antisymmetric 
* variational parameters, making it particularly suitable for fermionic systems.
* The total wavefunction is represented as a product of the RBM and PP parts:
* Ψ(σ) = Ψ_RBM(σ) × Pf(F_PP(σ))
* where Pf denotes the Pfaffian and F_PP is an antisymmetric matrix of variational parameters.
*
* @tparam _spinModes Number of possible spin states per site
* @tparam _Ht Data type for Hamiltonian elements
* @tparam _T Data type for calculations (defaults to _Ht)
* @tparam _stateType Data type for quantum states (defaults to double)
*
* Key features:
* - Combines RBM and PP state representations
* - Handles antisymmetric correlations through Pfaffian calculations
* - Supports multithreading for performance optimization
* - Provides methods for state manipulation, weight updates, and gradient calculations
* - Implements efficient Pfaffian updates for Monte Carlo sampling
*/
template < 	uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class RBM_PP : public RBM_S<_spinModes, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	
	using NQSLS_p = typename RBM_S<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
protected:
	// architecture parameters
	uint nPP_					= 1;
	uint nSites2_				= 1;
	uint nParticles2_			= 1;
	u64 rbmPPSize_				= 1;

	/* ------------------------------------------------------------ */
	// ------------------------ W E I G H T S ------------------------
	NQSB Fpp_;													// for storing the additional variational parameters from the PP
	v_2d<bool> spinSectors_;									// go through the quarters (spin sectors)

	_T pfaffian_				= 0.0;							// store the last pfaffian value
	_T pfaffianNew_				= 0.0;							// store the last pfaffian value for the candidate
	NQSW X_;													// for stroing the matrix for Pfaffian calculation at each step
	NQSW Xinv_;													// for stroing the matrix inverse for Pfaffian calculation at each step
	NQSW XinvSkew_;												// for stroing the matrix inverse for Pfaffian calculation at each step
	NQSW Xnew_;													// for stroing the matrix for Pfaffian calculation at each step - new candidate
	
// for calculating the Pfaffian probabilities from the Hamiltonian
#if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP 
	// create the map for thetas for a given thread
	std::map<std::thread::id, NQSW> XTmp_;
#else
	NQSW XTmp_;
#endif

	/* ------------------------------------------------------------ */
	
	u64 getFPPIndex(bool _spini, bool _spinj, uint ri, uint rj)	const;

	/* ------------------------------------------------------------ */
protected:
	// ----------------------- S T A R T E R S -----------------------
	void setInfo()												override final;
	void allocate()												override final;

	// ------------------------ S E T T E R S ------------------------
	virtual void setState(const NQSS& _st, bool _set)			override final;
	virtual void setState(u64 _st, bool _set)					override final;

	/* ------------------------------------------------------------ */
	// -------------------- P R O B A B I L I T Y --------------------
	auto logPRatio(uint fP, float fV)							-> _T override final;
	auto logPRatio(uint nFlips)									-> _T override final;
	auto logPRatio(const NQSS& _v1,
					const NQSS& _v2)							-> _T override final;
	auto logPRatio(std::initializer_list<int> fP,
				std::initializer_list<double> fV)				-> _T override final;

	// ------------------------ W E I G H T S ------------------------
public:
	bool setWeights(std::string _path, std::string _file)		override final;
	bool saveWeights(std::string _path, std::string _file)		override final;

protected:
	void updateWeights()										override final;
	// updates
#ifdef NQS_ANGLES_UPD
	void update(uint nFlips)									override final;
	void update(const NQSS& v, uint nFlips)						override final;
	void unupdate(uint nFlips)									override final;
#endif

	// ---------------------------- T R A I N ----------------------------	
	void grad(const NQSS& _v, uint _plc)						override final;

	// --------------------------- A N S A T Z ---------------------------
	virtual void updFPP_C(uint fP, float fV)					= 0;
	virtual void updFPP_C(std::initializer_list<int> fP,
						std::initializer_list<double> fV)		= 0;
	virtual void updFPP(uint fP, float fV)						= 0;
	virtual void updFPP(std::initializer_list<int> fP,
						std::initializer_list<double> fV)		= 0;
	virtual void updFPP_F(std::initializer_list<int> fP, 
						std::initializer_list<double> fV,
						arma::Mat<_T>& _Xtmp)					= 0;
public: 				  
	~RBM_PP() override											{ DESTRUCTOR_CALL;												};
	RBM_PP(std::shared_ptr<Hamiltonian<_Ht, _spinModes>> _H, uint _nHid, double _lr, uint _threadNum = 1, 
													int _nPart = -1, 
													const NQSLS_p& _lower = {}, 
													std::vector<double> _beta = {});

	// --------------------- S E T T E R S ---------------------
	void init()													override final;
	// for the PP matrix
	void setFPP()												{ this->X_ = this->getPPMat();									};
	void setFPP(const NQSS& _n)									{ this->X_ = this->getPPMat(_n);								};
	// void setFPP(u64 _n)											{ this->X_ = this->getPPMat(_n);								};
	// for the new PP matrix
	void setFPP_C()												{ this->Xnew_ = this->getPPMat();								};
	void setFPP_C(const NQSS& _n)								{ this->Xnew_ = this->getPPMat(_n);								};
	// void setFPP_C(u64 _n)										{ this->Xnew_ = this->getPPMat(_n);								};
	// for the pfaffian value
	void setPfaffian()											{ this->pfaffian_ = this->getPfaffian();						};
	void setPfaffian(const NQSS& _n)							{ this->setFPP(_n); this->setPfaffian();						};
	void setPfaffian(const NQSW& _M)							{ this->pfaffian_ = this->getPfaffian(_M);						};
	// for the candidate pfaffian value 
	void setPfaffian_C()										{ this->pfaffianNew_ = this->getPfaffian_C();					};
	void setPfaffian_C(const NQSS& _n)							{ this->setFPP_C(_n); this->setPfaffian_C();					};
	void setPfaffian_C(const NQSW& _M)							{ this->pfaffianNew_ = this->getPfaffian(_M);					};

	// -------------------- U P D A T E R S --------------------

#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	void updateXInv(uint _row);
	void updateXInv_C(uint _row);
	void updatePfaffian(uint _row);
	void updatePfaffian(uint _row, _T& _pfaffian);
	void updatePfaffian(uint _row, _T& _pfaffian, const arma::Mat<_T>& _X);
	void updatePfaffian_C(uint _row);
#endif

	// --------------------- G E T T E R S ---------------------
	
	virtual auto getPPMat(const NQSS& _n)	const -> NQSW		= 0;
#ifndef NQS_USE_VEC_ONLY
	virtual auto getPPMat(u64 _n)			const -> NQSW		= 0;
#endif
	// based on the current vector, obtain a matrix for Pffafian calcuation
	virtual auto getPPMat()					const -> NQSW		{ return this->getPPMat(this->curVec_);										};
	// based on the current matrix, obtain the Pffafian
	auto getPfaffian()						const -> _T			{ return algebra::Pfaffian::pfaffian<_T>(this->X_, this->info_p_.nParticles_);		};
	// based on the current candidate matrix, obtain the Pffafian
	auto getPfaffian_C()					const -> _T			{ return algebra::Pfaffian::pfaffian<_T>(this->Xnew_, this->info_p_.nParticles_);	};
	// get the Pffafian value for a given vector
	auto getPfaffian(const NQSS& _in)		const -> _T			{ return this->getPfaffian(this->getPPMat(_in));							};
	// based on given matrix, obtain the Pffafian
	static auto getPfaffian(const NQSW& _M)	-> _T				{ return algebra::Pfaffian::pfaffian<_T>(_M, _M.n_rows);					};
	
	// --------------------------------------------------------

	// get the current Pffafian value
	auto getCurrentPfaffian()				const -> _T			{ return this->pfaffian_;													};
	// get the current Pffafian candidate value
	auto getCurrentPfaffian_C()				const -> _T			{ return this->pfaffianNew_;												};
	// get size of the PP function matrix
	auto getNPP()							const -> uint		{ return this->nPP_;														};
	// --------------------- F I N A L E -----------------------
	auto ansatz(const NQSS& _in)			const -> _T			override final;
	auto ansatzlog(const NQSS& _in)			const -> _T			override final;
	auto ansatz_ratiolog(const NQSS& _in, 
		NQS<_spinModes, _Ht, _T, _stateType>* _other) 			const -> _T override final;

	// ---------------------------------------------------------
public:
	virtual auto clone() 					const -> MC_t_p		override = 0;

};

/**
* @brief Constructor for the RBM_PP class.
* 
* This constructor initializes an instance of the RBM_PP class, which is a derived class of RBM_S. 
* It sets up the spin sectors, calculates various sizes related to the problem, and allocates necessary resources.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Data type used in the calculations.
* @tparam _stateType Type of the state.
* 
* @param _H Shared pointer to the Hamiltonian object.
* @param _nHid Number of hidden units.
* @param _lr Learning rate.
* @param _threadNum Number of threads to be used.
* @param _nPart Number of particles.
* @param _lower Lower bound for the NQSLS_p object.
* @param _beta Vector of beta values.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline RBM_PP<_spinModes, _Ht, _T, _stateType>::RBM_PP(std::shared_ptr<Hamiltonian<_Ht, _spinModes>> _H, uint _nHid, 
							double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta)
	: RBM_S<_spinModes, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nPart, _lower, _beta)
{
	// create the spin sectors
	spinSectors_.push_back({ 1, 1 });
	spinSectors_.push_back({ 1, 0 });
	spinSectors_.push_back({ 0, 1 });
	spinSectors_.push_back({ 0, 0 });

	// !TODO make this changable
	this->nSites2_		= this->info_p_.nSites_ * this->info_p_.nSites_;
	this->nParticles2_	= this->info_p_.nParticles_ * this->info_p_.nParticles_;
	this->nPP_			= this->spinSectors_.size() * this->nSites2_; // for both spin channels
	this->rbmPPSize_	= this->rbmSize_ + this->nPP_;
	this->info_p_.fullSize_		= this->rbmPPSize_;
	this->allocate();
	this->setInfo();
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#define RBM_PP_INST_CMB(_Ht, _T, FUN, FUNRET, ARGS, ADD) 							\
					template FUNRET  RBM_PP<2u, _Ht, _T, double>::FUN ARGS ADD; 	\
					template FUNRET  RBM_PP<3u, _Ht, _T, double>::FUN ARGS ADD; 	\
					template FUNRET  RBM_PP<4u, _Ht, _T, double>::FUN ARGS ADD;

#define RBM_PP_INST_CMB_ALL(FUN, FUNRET, ARGS, ADD) 								\
					RBM_PP_INST_CMB(double, double, FUN, FUNRET, ARGS, ADD) 		\
					RBM_PP_INST_CMB(double, std::complex<double>, FUN, FUNRET, ARGS, ADD) \
					RBM_PP_INST_CMB(std::complex<double>, double, FUN, FUNRET, ARGS, ADD) \
					RBM_PP_INST_CMB(std::complex<double>, std::complex<double>, FUN, FUNRET, ARGS, ADD)

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#endif