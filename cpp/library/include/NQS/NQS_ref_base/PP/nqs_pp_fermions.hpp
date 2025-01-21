#include "./nqs_pp_spins.hpp"
// ##########################################################################################################################################

template <typename _Ht, typename _T, class _stateType, class _CorrState>
class NQS_PP_S<4u, _Ht, _T, _stateType, _CorrState> : public NQS_PP<4, _Ht, _T, _stateType, _CorrState>
{
    // **********************************************************************************************************************
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	NQS_PUBLIC_TYPES(_T, _stateType);
    NQS_HAMIL_TYPES(_Ht, 4);
    using NQSLS_p = typename NQS_PP<4, _Ht, _T, _stateType, _CorrState>::NQSLS_p;
    // **********************************************************************************************************************
public:
    NQS_PP_S(const NQS_Const_par_t<4, _Ht, _T, _stateType>& _p)
        : NQS_PP<4, _Ht, _T, _stateType, _CorrState>(_p)
	{
		this->init();
	};
    NQS_PP_S(const NQS_Const_par_t<4, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
        : NQS_PP<4, _Ht, _T, _stateType, _CorrState>(_p, _lower, _beta)
    {
        this->init();
    };
    // **********************************************************************************************************************
    virtual NQSW calculateX(Config_cr_t _n)							const override { NQS_LOG_ERROR_SPIN_MODES; return NQSW(); };
    // **********************************************************************************************************************
public:
	virtual void setX_changed(int _fP, float _fV)					override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual void setX_changed(int_ini_t, dbl_ini_t, NQSW& X_t)		override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual void setX_changed(int_ini_t, dbl_ini_t, v_1d<NQSB>&)	override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual void setX_changed(int_ini_t, dbl_ini_t)					override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual void updX_changed(int _fP)								override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual void updX_changed(int_ini_t, dbl_ini_t)					override { NQS_LOG_ERROR_SPIN_MODES; };
	// **********************************************************************************************************************
public:
	// **********************************************************************************************************************
	virtual MC_t_p clone() const override
	{
		return std::make_shared<NQS_PP_S>(*this);
	}
	// **********************************************************************************************************************
};

// ##########################################################################################################################################