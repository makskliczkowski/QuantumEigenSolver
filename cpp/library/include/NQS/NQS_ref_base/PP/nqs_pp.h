/**
* @file nqs_pp.h
* @brief Header file for the NQS_PP class, which implements the Pair Product (PP) reference state for Neural Quantum States (NQS).
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
#ifndef NQSPP_H
#define NQSPP_H
#ifndef NQS_REF_STATE_H
#	include "../nqs_ref.hpp"
#endif
// ***************************************************************************************
#define NQS_REF_PP_USE_PFAFFIAN_UPDATE
// ***************************************************************************************`w

template <typename _Ht, typename _T, class _stateType>
struct NQS_PP_weights
{
	// **********************************************************************************************************************
	NQS_PUBLIC_TYPES(_T, _stateType);
	MCS_PUBLIC_TYPES(_T, _stateType, NQS_STATE_R_T); 			// type definitions for the Monte Carlo solver
	// **********************************************************************************************************************
	NQSB F_r1r2_s1s2_;											// for storing the additional variational parameters from the PP
																// this is the matrix of weights stored as a vector (this->nPP_ x 1)
	NQSW X_, X_inv;												// for stroing the X matrix of the variational parameter X = X(x), where X is the configuration. X is skew-symmetric
																// the X matrix is defined as of size (N_particles x N_particles) and is skew-symmetric (X = -X^T)
	NQSW Xinv_skew_;											// for storing the value of X^{-1}^T - X^{-1}
	// **********************************************************************************************************************
	static inline thread_local v_1d<NQSB> X_upd_;				// for storing the updated X matrix elements (one updates the matrix 
																// by changing a single row and column, so we need to store the updated values)
																// however, we can change multiple rows and columns at once (e.g., for the Monte Carlo updates)
																// then we need to store the updated values for all the rows and columns - hence the vector
	static inline thread_local v_1d<bool> states_upd_;			// for storing the updated states for the X matrix - if the number of flips is larger than 2
	// **********************************************************************************************************************	
#ifndef NQS_REF_PP_USE_PFAFFIAN_UPDATE
	static inline thread_local NQSW XTmp_;						// for calculating the Pfaffian probabilities from the Hamiltonian
	static inline thread_local NQSW XTmpInv_;					// for calculating the Pfaffian probabilities from the Hamiltonian
#endif
	// **********************************************************************************************************************
	_T pfaffianLog_				= 0.0;							// store the last pfaffian value
	_T pfaffian_				= 0.0;							// store the last pfaffian value
	inline static thread_local _T pfaffianNewLog_	= 0.0;		// store the last pfaffian value for the candidate - can be updated with the Cayley's idendity
	inline static thread_local _T pfaffianNew_		= 0.0;		// store the last pfaffian value for the candidate - can be updated with the Cayley's idendity
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
template <uint _spinModes, typename _Ht, typename _type = _Ht, class _stateType = double, class _CorrState = NQS_S<_spinModes, _Ht, _type, _stateType>>
requires NQS_S_concept<_spinModes, _Ht, _type, _stateType, _CorrState>
class NQS_PP : public NQS_ref<_spinModes, _Ht, _type, _stateType, _CorrState>
{
	// **********************************************************************************************************************
	MCS_PUBLIC_TYPES(_type, _stateType, NQS_STATE_R_T); 		// type definitions for the Monte Carlo solver
	NQS_PUBLIC_TYPES(_type, _stateType);						// type definitions for the NQS
	NQS_HAMIL_TYPES(_Ht, _spinModes);							// type definitions for the Hamiltonian
	using NQSLS_p 	= typename NQS_ref<_spinModes, _Ht, _type, _stateType, _CorrState>::NQSLS_p;
	using NQS_t 	= NQS_ref<_spinModes, _Ht, _type, _stateType, _CorrState>::NQS_t;
	using NQS_t_p	= typename std::shared_ptr<NQS_t>;
	// **********************************************************************************************************************
protected:
	uint nPP_					= 1;							// number of PP variational parameters (connected to the number of particles)
	u64 PPsize_					= 1;							// size of the PP part of the wave function
	// **********************************************************************************************************************	
	NQS_PP_weights<_Ht, _type, _stateType> pp_weights_;			// for storing the weights and the Pfaffian
	v_2d<int> spinSectors_		= {{1,1},{1,0},{0,1},{0,0}};	// go through the quarters (spin sectors) of the matrix F to interpret it !TODO: implement the spin sectors 
	// ***************************************************************************************************************************
	u64 getFPPIndex(int _spini, int _spinj, uint ri, uint rj)	const;
protected:
	// ***************************************************************************************************************************
	_type getPfaffian(Config_cr_t _v)							const;
	_type getPfaffian()											const;
	_type getPfaffianLog(Config_cr_t _v)						const { return std::log(this->getPfaffian(_v));					};
	_type getPfaffianLog()										const { return std::log(this->getPfaffian());					};
	// sets the pfaffian based on the configuration of the system
	void setPfaffian();
	void setPfaffian(Config_cr_t _v)							{ this->setState(NQS_STATE, true); return this->setPfaffian();	};
	// ***************************************************************************************************************************
	_type getUpdatedPfaffian(uint row)							const;
	_type getUpdatedPfaffian(uint row, _type _pfaffian)			const;
	_type getUpdatedPfaffian(uint row, _type, const NQSB&)		const;
	_type getUpdatedPfaffian(uint row, _type, const NQSW&)		const;
	void updatePfaffian(uint row, _type& _pfaffian)						{ this->pp_weights_.pfaffian_ = this->getUpdatedPfaffian(row, _pfaffian);			};
	void updatePfaffian(uint row)										{ this->updatePfaffian(row, this->pp_weights_.pfaffian_);							};
	void updatePfaffian(uint row, _type& _pfaffian, const NQSB& _Xrow) 	{ this->pp_weights_.pfaffian_ = this->getUpdatedPfaffian(row, _pfaffian, _Xrow); 	};
	void updatePfaffian(uint row, _type& _pfaffian, const NQSW& _X)		{ this->pp_weights_.pfaffian_ = this->getUpdatedPfaffian(row, _pfaffian, _X);		};
	// ***************************************************************************************************************************
public:
	virtual NQSW calculateX(Config_cr_t _v)						const = 0;	// calculate the X matrix for the given configuration NQS_STATE (or any other configuration)
protected:
	void setX(Config_cr_t _n);									// set the X matrix for the given configuration NQS_STATE (or any other configuration)
	void setX()													{ this->setX(NQS_STATE);										};			
public:
	const NQSW& getXinv()										const { return this->pp_weights_.X_inv;							};
	const NQSW& getX()											const { return this->pp_weights_.X_;							};
	// ***************************************************************************************************************************
	virtual void setX_changed(int _fP, float _fV)				= 0;		// implements the changed X matrix after the update of a single row and column (fP, fV)
	virtual void setX_changed(int_ini_t, dbl_ini_t, NQSW& X_t)	= 0;		// implements the changed X matrix after the update of a single row and column (fP, fV)
	virtual void setX_changed(int_ini_t, dbl_ini_t, v_1d<NQSB>&)= 0;		// implements the changed X matrix after the update of a single row and column (fP, fV)
	virtual void setX_changed(int_ini_t, dbl_ini_t)				= 0;		// implements the changed X matrix after the update of a single row and column (fP, fV)
	virtual void updX_changed(int _fP)							= 0;		// implements the changed X matrix after the update of a single row and column (fP, fV)
	virtual void updX_changed(int_ini_t, dbl_ini_t)				= 0;		// implements the changed X matrix after the update of a single row and column (fP, fV)
	// ***************************************************************************************************************************
	virtual void updateXInv(uint row);										// update the X matrix after the update of a single row and column (fP, fV)
	// ***************************************************************************************************************************
protected:														// ----------------------- S T A R T E R S -----------------------
	void allocate()												override final;
	void setInfo()												override final;
public:															// ------------------------ S E T T E R S ------------------------
	virtual void setState(const Config_t& _st, bool _set = true)override final;
	virtual void setState(u64 _st, bool _set = true)			override final;
protected:														// -------------------- P R O B A B I L I T Y --------------------
	auto logPRatio(uint fP, float fV)							-> _type override final;
	auto logPRatio(uint, uint, float, float)					-> _type override final;
	auto logPRatio(uint nFlips)									-> _type override final;
	auto logPRatio(Config_cr_t _v1, Config_cr_t _v2)			-> _type override final;
	auto logPRatio(int_ini_t fP, dbl_ini_t fV)					-> _type override final;
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
	// ***************************************************************************************************************************
public: 				  
	~NQS_PP() override											{ DESTRUCTOR_CALL;												};
	NQS_PP(const NQS_Const_par_t<_spinModes, _Ht, _type, _stateType>& _p)
		: NQS_ref<_spinModes, _Ht, _type, _stateType, _CorrState>(_p)	
	{
		this->init();
		this->nPP_					= this->spinSectors_.size() * this->info_p_.nSitesSquared_;
		this->PPsize_			    = this->nPP_;
		this->info_p_.fullSize_		= NQS_ref<_spinModes, _Ht, _type, _stateType, _CorrState>::size() + this->PPsize_;
		this->allocate();
		this->setInfo();
	}
	NQS_PP(const NQS_Const_par_t<_spinModes, _Ht, _type, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
		: NQS_ref<_spinModes, _Ht, _type, _stateType, _CorrState>(_p, _lower, _beta)	
	{ 
		this->init();
		this->nPP_					= this->spinSectors_.size() * this->info_p_.nSitesSquared_;
		this->PPsize_			    = this->nPP_;
		this->info_p_.fullSize_		= NQS_ref<_spinModes, _Ht, _type, _stateType, _CorrState>::size() + this->PPsize_;
		this->allocate();
		this->setInfo();
	};
	NQS_PP(const NQS_PP<_spinModes, _Ht, _type, _stateType, _CorrState>& _other);
	NQS_PP(NQS_PP<_spinModes, _Ht, _type, _stateType, _CorrState>&& _other);
	// **********************************************************************************************************************
public:															// --------------------- S E T T E R S ---------------------
	void init()													override final;
	// ***************************************************************************************************************************
public:															// --------------------- G E T T E R S ---------------------
	virtual auto size()						const -> size_t		override final	{ return this->PPsize_;							};
	// ***************************************************************************************************************************
	auto getCurrentPfaffian()				const -> _type			{ return this->pp_weights_.pfaffian_;						};
	auto getNPP()							const -> uint			{ return this->nPP_;										};
	// ***************************************************************************************************************************
	// --------------------- F I N A L E -----------------------
	auto ansatz(Config_cr_t _in)			const -> _type			override final;
	auto ansatzlog(Config_cr_t _in)			const -> _type			override final;
	auto ansatz_ratiolog(Config_cr_t _in, NQS<_spinModes, _Ht, _type, _stateType>* _other) const -> _type override final;
public:
	// **********************************************************************************************************************
	virtual auto clone() 					const -> MC_t_p		override = 0;
	virtual auto clone(MC_t_p _n) 			-> void				override;
	// **********************************************************************************************************************
};

// ##########################################################################################################################

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#define NQS_PP_INST_CMB(_Ht, _T, FUN, FUNRET, ARGS, ADD) 									\
					template FUNRET  NQS_PP<2u, _Ht, _T, double>::FUN ARGS ADD; 			\
					template FUNRET  NQS_PP<4u, _Ht, _T, double>::FUN ARGS ADD;

#define NQS_PP_INST_CMB_ALL(FUN, FUNRET, ARGS, ADD) 										\
			NQS_PP_INST_CMB(double, double, FUN, FUNRET, ARGS, ADD) 						\
			NQS_PP_INST_CMB(double, std::complex<double>, FUN, FUNRET, ARGS, ADD) 			\
			NQS_PP_INST_CMB(std::complex<double>, double, FUN, FUNRET, ARGS, ADD) 			\
			NQS_PP_INST_CMB(std::complex<double>, std::complex<double>, FUN, FUNRET, ARGS, ADD)
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#endif	// RBMPP_H
// ##########################################################################################################################

// how to add specializations here?