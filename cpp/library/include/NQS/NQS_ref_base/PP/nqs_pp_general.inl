#include "./nqs_pp.h"

// ##########################################################################################################################
#ifndef NQS_PP_GEN_H
#define NQS_PP_GEN_H

template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double, class _CorrState = NQS_S<_spinModes, _Ht, _T, _stateType>>
class NQS_PP_S : public NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>
{
	// **********************************************************************************************************************
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	NQS_PUBLIC_TYPES(_T, _stateType);
	NQS_HAMIL_TYPES(_Ht, _spinModes);
	using NQSLS_p =	typename NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::NQSLS_p;
	// **********************************************************************************************************************
public:
	NQS_PP_S(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p)
		: NQS_PP<_spinModes, _Ht, _T, _stateType>(_p)					{ NQS_LOG_ERROR_SPIN_MODES; };
	NQS_PP_S(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
		: NQS_PP<_spinModes, _Ht, _T, _stateType>(_p, _lower, _beta)	{ NQS_LOG_ERROR_SPIN_MODES; };
public:
	// **********************************************************************************************************************
	virtual NQSW calculateX(Config_cr_t _n)	const override
    { 
        NQS_LOG_ERROR_SPIN_MODES; 
        return NQSW(); 
    };
    // **********************************************************************************************************************
public:
	virtual void setX_changed(int _fP, float _fV) override 					{ NQS_LOG_ERROR_SPIN_MODES; };
	virtual void setX_changed(int_ini_t, dbl_ini_t, NQSW& X_t) override 	{ NQS_LOG_ERROR_SPIN_MODES; };
	virtual void setX_changed(int_ini_t, dbl_ini_t, v_1d<NQSB>&) override	{ NQS_LOG_ERROR_SPIN_MODES; };
	virtual void setX_changed(int_ini_t, dbl_ini_t) override 				{ NQS_LOG_ERROR_SPIN_MODES; };
	virtual void updX_changed(int _fP) override 							{ NQS_LOG_ERROR_SPIN_MODES; };
	virtual void updX_changed(int_ini_t, dbl_ini_t) override 				{ NQS_LOG_ERROR_SPIN_MODES; };
	// **********************************************************************************************************************
// #ifndef NQS_USE_VEC_ONLY
	// virtual auto getPPMat(u64 _n)			const -> NQSW	override { NQS_LOG_ERROR_SPIN_MODES; return NQSW(); };
// #endif
protected:
	// --------------------------- A N S A T Z ---------------------------
	// void updFPP_C(uint fP, float fV)						override { NQS_LOG_ERROR_SPIN_MODES; };
	// void updFPP_C(std::initializer_list<int> fP,
	// 			std::initializer_list<double> fV)			override { NQS_LOG_ERROR_SPIN_MODES; };
	// void updFPP(uint fP, float fV)							override { NQS_LOG_ERROR_SPIN_MODES; };
	// void updFPP(std::initializer_list<int> fP,
	// 			std::initializer_list<double> fV)			override { NQS_LOG_ERROR_SPIN_MODES; };
	// void updFPP_F(std::initializer_list<int> fP,
	// 			std::initializer_list<double> fV,
	// 			arma::Mat<_T>& _Xtmp)						override { NQS_LOG_ERROR_SPIN_MODES; };

    // **********************************************************************************************************************
public:
	virtual auto clone() 									const -> MC_t_p override
	{
		return std::make_shared<NQS_PP_S>(*this);
	}
    // **********************************************************************************************************************
};
#endif //! NQS_PP_GEN_H
// ##########################################################################################################################