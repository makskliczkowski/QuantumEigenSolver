#include "rbm_setup.hpp"
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GENERAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ifndef RBM_GEN_H
#define RBM_GEN_H

template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class RBM_S : public RBM<_spinModes, _Ht, _T, _stateType>
{
public:
	// **********************************************************************************************************************
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	NQS_PUBLIC_TYPES(_T, _stateType);
	NQS_HAMIL_TYPES(_Ht, _spinModes);
	using NQSLS_p =	typename RBM<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
	// **********************************************************************************************************************
public:
	// **********************************************************************************************************************
	RBM_S()																= default;
	RBM_S(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p)
		: RBM<_spinModes, _Ht, _T, _stateType>(_p) 
	{ 
		NQS_LOG_ERROR_SPIN_MODES;
	};

	RBM_S(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
		: RBM<_spinModes, _Ht, _T, _stateType>(_p, _lower, _beta) 
	{ 
		NQS_LOG_ERROR_SPIN_MODES;
	};
	
	/* ------------------------------------------------------------ */
protected:
	// -------------------- P R O B A B I L I T Y --------------------
	virtual auto logPRatio(uint fP, float fV)		-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual auto logPRatio(uint nFlips)				-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual auto logPRatio(Config_cr_t _v1,					
					Config_cr_t _v2)				-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual auto logPRatio(Config_cr_t _v1)			-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual auto logPRatio(std::initializer_list<int> fP,		
				std::initializer_list<double> fV)	-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
	// **********************************************************************************************************************
public:
	virtual auto clone() const -> MC_t_p override {
		return std::make_shared<RBM_S>(*this); // Ensure MC_t resolves to RBM_S
	}
	//////////////////////////////////////////////////////////////////
};

#endif