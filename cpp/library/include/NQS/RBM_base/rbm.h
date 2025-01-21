#pragma once
#include <cstddef>
#ifndef RBM_H
#define RBM_H

#ifndef NQS_H
	#include "../nqs_final.hpp"
#endif // !NQS_H

//////////////////////////////////////////////////////////////////////////////////////////

template < 	uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class RBM : public NQS_S<_spinModes, _Ht, _T, _stateType>
{
public:
	// **********************************************************************************************************************
	NQS_PUBLIC_TYPES(_T, _stateType);
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 			// type definitions for the Monte Carlo solver
	using NQSLS_p =	typename NQS_S<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
	// **********************************************************************************************************************
protected:
	uint nHid_						=						1;
	u64 rbmSize_					=						1;
protected:													// ------------------------ W E I G H T S -----------------------
	NQSW W_;												// weight matrix
	NQSB bV_;												// visible bias
	NQSB bH_;												// hidden bias
	// **********************************************************************************************************************
protected:													// ------------------------- A N G L E S ------------------------
	NQSB theta_, thetaCOSH_, thetaCOSH_log_;				// for storing the angles for the RBM
	// **********************************************************************************************************************
public:														// COSH of the angles
	auto coshF(Config_cr_t _v)		const -> NQSB			{ return arma::cosh(this->bH_ + this->W_ * _v);		};
	auto coshF()					const -> NQSB			{ return arma::cosh(this->theta_);					};
	NQSB thetaTmpCol_;
	// **********************************************************************************************************************
protected:													// ---------------------- T H R E A D I N G ---------------------
	// !TODO - implement the threading for the RBM with better handling of the threads
#ifdef NQS_NOT_OMP_MT
    thread_local static inline NQSB thetaTMP_;				// Thread-local storage for thetaTMP
#endif
	// **********************************************************************************************************************
protected:													// ----------------------- S T A R T E R S -----------------------
	virtual void allocate()									override;
	virtual void setInfo()									override;
	// **********************************************************************************************************************
	virtual void setState(const Config_t& _st, bool _set)	override;
	virtual void setState(u64 _st, bool _set)				override;
	// **********************************************************************************************************************
public:														// -------------------- P R O B A B I L I T Y --------------------
	virtual auto logPRatio(uint fP, float fV)				-> _T override = 0;
	virtual auto logPRatio(uint nFlips)						-> _T override = 0;
	virtual auto logPRatio(Config_cr_t _v1, Config_cr_t _v2)-> _T override = 0;
	virtual auto logPRatio(Config_cr_t _v1)					-> _T override = 0;
	virtual auto logPRatio(int_ini_t fP, dbl_ini_t fV) 		-> _T override = 0;
public:														// ------------------------ W E I G H T S ------------------------
	// **********************************************************************************************************************
	virtual bool saveWeights(std::string, std::string)		override;
	virtual bool setWeights(std::string, std::string)		override;
	virtual void setWeights()								override;
protected:
	virtual void updateWeights()							override;
	// **********************************************************************************************************************
protected:													// set the angles for the RBM to be updated
	void setTheta()											{ this->setTheta(this->curVec_); };
	void setTheta(Config_cr_t v);
#ifdef NQS_ANGLES_UPD
	virtual void update(uint nFlips)						override;
	virtual void update(Config_cr_t v, uint nFlips)			override;
#endif
	// **************************************************************************************************************************
protected:													// ------------------------- T R A I N --------------------------
	virtual void grad(Config_cr_t _v, uint _plc)			override;
public:
	virtual ~RBM()											{ DESTRUCTOR_CALL; };
	RBM(const RBM<_spinModes, _Ht, _T, _stateType>& _other);
	RBM(RBM<_spinModes, _Ht, _T, _stateType>&& _other);
	RBM(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p);
	RBM(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta);

	// ------------------- O P E R A T O R S -------------------
	auto operator=(RBM<_spinModes, _Ht, _T, _stateType>& _other) 	-> RBM<_spinModes, _Ht, _T, _stateType>&;
	auto operator=(RBM<_spinModes, _Ht, _T, _stateType>&& _other) 	-> RBM<_spinModes, _Ht, _T, _stateType>&;

	// --------------------- S E T T E R S ---------------------
	virtual void init()										override;
	
	// --------------------- G E T T E R S ---------------------
	virtual auto size()				const -> size_t override		{ return this->rbmSize_; 	};
	auto getNhid()					const -> uint					{ return this->nHid_; 		};
	auto getWeightsMat()			const -> NQSW					{ return this->W_; 			};
	auto getVisibleBias()			const -> NQSB					{ return this->bV_; 		};
	auto getHiddenBias()			const -> NQSB					{ return this->bH_; 		};
	// --------------------- F I N A L E -----------------------
	virtual auto ansatz(Config_cr_t _in) 					const -> _T override;
	virtual auto ansatzlog(Config_cr_t _in) 				const -> _T override;
	virtual auto ansatz_ratiolog(Config_cr_t _in, 
		NQS<_spinModes, _Ht, _T, _stateType>* _other) 		const -> _T override;
	// **********************************************************************************************************************
	virtual auto clone() 			const -> MC_t_p 		override = 0;
	virtual auto clone(MC_t_p _n)   -> void 				override;
	// **********************************************************************************************************************
};	

// ##########################################################################################################################################

#define RBM_INST_CMB(_Ht, _T, FUN, FUNRET, ARGS, ADD) 							\
					template FUNRET  RBM<2u, _Ht, _T, double>::FUN ARGS ADD; 	\
					template FUNRET  RBM<3u, _Ht, _T, double>::FUN ARGS ADD; 	\
					template FUNRET  RBM<4u, _Ht, _T, double>::FUN ARGS ADD; 	

#define RBM_INST_CMB_ALL(FUN, FUNRET, ARGS, ADD) 								\
					RBM_INST_CMB(double, double, FUN, FUNRET, ARGS, ADD) 		\
					RBM_INST_CMB(double, std::complex<double>, FUN, FUNRET, ARGS, ADD) \
					RBM_INST_CMB(std::complex<double>, double, FUN, FUNRET, ARGS, ADD) \
					RBM_INST_CMB(std::complex<double>, std::complex<double>, FUN, FUNRET, ARGS, ADD)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif