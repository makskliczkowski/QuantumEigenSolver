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
#ifndef RBMPP_H
#define RBMPP_H
#ifndef RBM_H
#	include "../nqs_ref.hpp"
#endif
// ***************************************************************************************
#define NQS_REF_PP_USE_PFAFFIAN_UPDATE
// ***************************************************************************************

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! B A S E !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

template <typename _Ht, typename _T, class _stateType>
struct NQS_PP_weights
{
	// **********************************************************************************************************************
	NQS_PUBLIC_TYPES(_T, _stateType);
	MCS_PUBLIC_TYPES(_T, _stateType, NQS_STATE_R_T); 			// type definitions for the Monte Carlo solver
	// **********************************************************************************************************************
	NQSB F_r1r2_s1s2_;											// for storing the additional variational parameters from the PP
	NQSW X_, X_inv;												// for stroing the X matrix of the variational parameter X = X(x), where X is the configuration. X is skew-symmetric
#ifndef NQS_REF_PP_USE_PFAFFIAN_UPDATE
	static inline thread_local NQSW XTmp_;						// for calculating the Pfaffian probabilities from the Hamiltonian
	static inline thread_local NQSW XTmpInv_;					// for calculating the Pfaffian probabilities from the Hamiltonian
#endif
	// **********************************************************************************************************************
	_T pfaffian_				= 0.0;							// store the last pfaffian value
	_T pfaffianNew_				= 0.0;							// store the last pfaffian value for the candidate - can be updated with the Cayley's idendity
	// **********************************************************************************************************************
};

// ##########################################################################################################################

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
* @important The class will be designed as for the reference state |\psi > = \sum _x F(x) <x|phi>_{ref} |x>,
* where F(x) is the variational part taking into account the correlations between the particles and <x|phi>_{ref} is 
* overlap with a reference state. For simplicity, the reference state will be taken as the product of the single-particle states.
*/
template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double, class _CorrState = NQS_S<_spinModes, _Ht, _T, _stateType>>
class NQS_PP : public NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>
{
	// **********************************************************************************************************************
	MCS_PUBLIC_TYPES(_T, _stateType, NQS_STATE_R_T); 			// type definitions for the Monte Carlo solver
	NQS_PUBLIC_TYPES(_T, _stateType);							// type definitions for the NQS
	NQS_HAMIL_TYPES(_Ht, _spinModes);							// type definitions for the Hamiltonian
	using NQSLS_p = typename NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::NQSLS_p;
	// **********************************************************************************************************************
protected:
	uint nPP_					= 1;							// number of PP variational parameters (connected to the number of particles)
	u64 PPsize_					= 1;							// size of the PP part of the wave function
	// **********************************************************************************************************************	
	NQS_PP_weights<_Ht, _T, _stateType> pp_weights_;			// for storing the weights and the Pfaffian
	v_2d<int> spinSectors_		= {{1,1},{1,0},{0,1},{0,0}};	// go through the quarters (spin sectors) of the matrix F to interpret it !TODO: implement the spin sectors 
	// ***************************************************************************************************************************
	u64 getFPPIndex(bool _spini, bool _spinj, uint ri, uint rj)	const;
protected:														// ----------------------- S T A R T E R S -----------------------
	void setInfo()												override final;
	void allocate()												override final;
public:															// ------------------------ S E T T E R S ------------------------
	virtual void setState(const Config_t& _st, bool _set)		override final;
	virtual void setState(u64 _st, bool _set)					override final;
protected:														// -------------------- P R O B A B I L I T Y --------------------
	auto logPRatio(uint fP, float fV)							-> _T override final;
	auto logPRatio(uint nFlips)									-> _T override final;
	auto logPRatio(Config_cr_t _v1, Config_cr_t _v2)			-> _T override final;
	auto logPRatio(int_ini_t fP, dbl_ini_t fV)					-> _T override final;
	// ***************************************************************************************************************************
public:															// ------------------------ W E I G H T S ------------------------
	bool setWeights(std::string _path, std::string _file)		override final;
	bool saveWeights(std::string _path, std::string _file)		override final;
	void setWeights()											override final;
protected:
	void updateWeights()										override final;
	// ***************************************************************************************************************************
#ifdef NQS_ANGLES_UPD
	void update(uint nFlips)									override final;
	void update(Config_cr_t v, uint nFlips)						override final;
	void unupdate(uint nFlips)									override final;
#endif
	// ***************************************************************************************************************************
protected:														// -------------------------- T R A I N --------------------------
	void grad(Config_cr_t _v, uint _plc)						override final;
protected:														// ------------------------- A N S A T Z -------------------------
	virtual void updFPP_C(uint fP, float fV)					= 0;
	virtual void updFPP_C(int_ini_t fP, dbl_ini_t fV)			= 0;
	virtual void updFPP(uint fP, float fV)						= 0;
	virtual void updFPP(int_ini_t fP, dbl_ini_t fV)				= 0;
	virtual void updFPP_F(int_ini_t, dbl_ini_t, NQSW&) 			= 0;
	// ***************************************************************************************************************************
public: 				  
	~NQS_PP() override											{ DESTRUCTOR_CALL;												};
	NQS_PP(std::shared_ptr<Hamiltonian<_Ht, _spinModes>> _H, uint _nHid, double _lr, uint _threadNum = 1, int _nPart = -1, 
													const NQSLS_p& _lower = {}, std::vector<double> _beta = {});
	NQS_PP(const NQS_PP<_spinModes, _Ht, _T, _stateType>& _other);
	NQS_PP(NQS_PP<_spinModes, _Ht, _T, _stateType>&& _other);
public:															// --------------------- S E T T E R S ---------------------
	void init()													override final;
	// for the PP matrix
	void setFPP()												{ this->X_ = this->getPPMat();									};
	void setFPP(Config_cr_t _n)									{ this->X_ = this->getPPMat(_n);								};
	// for the new PP matrix
	void setFPP_C()												{ this->Xnew_ = this->getPPMat();								};
	void setFPP_C(Config_cr_t _n)								{ this->Xnew_ = this->getPPMat(_n);								};
	// for the pfaffian value
	void setPfaffian()											{ this->pfaffian_ = this->getPfaffian();						};
	void setPfaffian(Config_cr_t _n)							{ this->setFPP(_n); this->setPfaffian();						};
	void setPfaffian(const NQSW& _M)							{ this->pfaffian_ = this->getPfaffian(_M);						};
	// for the candidate pfaffian value 
	void setPfaffian_C()										{ this->pfaffianNew_ = this->getPfaffian_C();					};
	void setPfaffian_C(Config_cr_t _n)							{ this->setFPP_C(_n); this->setPfaffian_C();					};
	void setPfaffian_C(const NQSW& _M)							{ this->pfaffianNew_ = this->getPfaffian(_M);					};
public:															// -------------------- U P D A T E R S --------------------
	// ***************************************************************************************************************************
#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	void updateXInv(uint _row);
	void updateXInv_C(uint _row);
	void updatePfaffian(uint _row);
	void updatePfaffian(uint _row, _T& _pfaffian);
	void updatePfaffian(uint _row, _T& _pfaffian, const arma::Mat<_T>& _X);
	void updatePfaffian_C(uint _row);
#endif
	// ***************************************************************************************************************************
public:															// --------------------- G E T T E R S ---------------------
	virtual auto size()						const -> size_t		override final	{ return this->rbmSize_ + this->PPsize_;					};
	virtual auto getPPMat(Config_cr_t _n)	const -> NQSW		= 0;
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
	auto getPfaffian(Config_cr_t _in)		const -> _T			{ return this->getPfaffian(this->getPPMat(_in));							};
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
	auto ansatz(Config_cr_t _in)			const -> _T			override final;
	auto ansatzlog(Config_cr_t _in)			const -> _T			override final;
	auto ansatz_ratiolog(Config_cr_t _in, 
		NQS<_spinModes, _Ht, _T, _stateType>* _other) 			const -> _T override final;

	// ---------------------------------------------------------
public:
	// **********************************************************************************************************************
	virtual auto clone() 					const -> MC_t_p		override = 0;
	virtual auto clone(MC_t_p _n) 			-> void				override;
	// **********************************************************************************************************************
};

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

// ##########################################################################################################################

#ifndef NQS_PP_GEN_H
#define NQS_PP_GEN_H
template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double, class _CorrState = NQS_S<_spinModes, _Ht, _T, _stateType>>
class NQS_PP_S : public NQS_PP<_spinModes, _Ht, _T, _stateType>
{
	// **********************************************************************************************************************
	NQS_PUBLIC_TYPES(_T, _stateType);
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	using NQSLS_p =	typename NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::NQSLS_p;
	// **********************************************************************************************************************
	NQS_PP_S(std::shared_ptr<Hamiltonian<_Ht, _spinModes>>& _H, 
			uint _nHid, 
			double _lr, 
			uint _threadNum = 1, 
			int _nParticles = -1,
			const NQSLS_p& _lower = {}, 
			const std::vector<double>& _beta = {})
		: NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>(_H, _nHid, _lr, _threadNum, _nParticles, _lower, _beta)
			{ NQS_LOG_ERROR_SPIN_MODES; };
	// **********************************************************************************************************************
	// --------------------- G E T T E R S ---------------------
	virtual auto getPPMat(Config_cr_t _n)	const -> NQSW	override { NQS_LOG_ERROR_SPIN_MODES; return NQSW(); };
#ifndef NQS_USE_VEC_ONLY
	virtual auto getPPMat(u64 _n)			const -> NQSW	override { NQS_LOG_ERROR_SPIN_MODES; return NQSW(); };
#endif
protected:
	// --------------------------- A N S A T Z ---------------------------
	void updFPP_C(uint fP, float fV)						override { NQS_LOG_ERROR_SPIN_MODES; };
	void updFPP_C(std::initializer_list<int> fP,
				std::initializer_list<double> fV)			override { NQS_LOG_ERROR_SPIN_MODES; };
	void updFPP(uint fP, float fV)							override { NQS_LOG_ERROR_SPIN_MODES; };
	void updFPP(std::initializer_list<int> fP,
				std::initializer_list<double> fV)			override { NQS_LOG_ERROR_SPIN_MODES; };
	void updFPP_F(std::initializer_list<int> fP,
				std::initializer_list<double> fV,
				arma::Mat<_T>& _Xtmp)						override { NQS_LOG_ERROR_SPIN_MODES; };

	// -------------------------------------------------------------------
public:
	virtual auto clone() 									const -> MC_t_p override
	{
		return std::make_shared<MC_t>(*this);
	}

	// -------------------------------------------------------------------
};
#endif //! NQS_PP_GEN_H
// ##########################################################################################################################