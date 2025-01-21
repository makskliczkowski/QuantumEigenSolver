#include "./nqs_pp_general.inl"
// ##########################################################################################################################

#ifndef RBM_PP_2_H
#define RBM_PP_2_H

template <typename _Ht, typename _T, class _stateType, class _CorrState>
class NQS_PP_S<2u, _Ht, _T, _stateType, _CorrState> : public NQS_PP<2, _Ht, _T, _stateType, _CorrState>
{
    // **********************************************************************************************************************
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	NQS_PUBLIC_TYPES(_T, _stateType);
    NQS_HAMIL_TYPES(_Ht, 2);
    using NQSLS_p = typename NQS_PP<2, _Ht, _T, _stateType, _CorrState>::NQSLS_p;
    // **********************************************************************************************************************
public:
    NQS_PP_S(const NQS_Const_par_t<2, _Ht, _T, _stateType>& _p)
        : NQS_PP<2, _Ht, _T, _stateType, _CorrState>(_p)
	{
		this->init();
	};
    NQS_PP_S(const NQS_Const_par_t<2, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
        : NQS_PP<2, _Ht, _T, _stateType, _CorrState>(_p, _lower, _beta)
    {
        this->init();
    };
    // **********************************************************************************************************************
    virtual NQSW calculateX(Config_cr_t _n)							const override;
    // **********************************************************************************************************************
public:
	virtual void setX_changed(int _fP, float _fV)					override;
	virtual void setX_changed(int_ini_t, dbl_ini_t, NQSW& X_t)		override;
	virtual void setX_changed(int_ini_t, dbl_ini_t, v_1d<NQSB>&)	override;
	virtual void setX_changed(int_ini_t, dbl_ini_t)					override;
	virtual void updX_changed(int _fP)								override;
	virtual void updX_changed(int_ini_t, dbl_ini_t)					override;
	// **********************************************************************************************************************
protected:
	// --------------------------- A N S A T Z -----------------
	// void updFPP_C(uint fP, float fV)						override;
	// void updFPP_C(std::initializer_list<int> fP,
	// 				 std::initializer_list<double> fV)		override;
	// void updFPP(uint fP, float fV)							override;
	// void updFPP(std::initializer_list<int> fP,
	// 			std::initializer_list<double> fV)			override;
	// void updFPP_F(std::initializer_list<int> fP,
	// 			std::initializer_list<double> fV,
	// 			arma::Mat<_T>& _Xtmp)						override;

	// ---------------------------------------------------------

	// **********************************************************************************************************************
	virtual MC_t_p clone() const override
	{
		return std::make_shared<NQS_PP_S>(*this);
	}
	// **********************************************************************************************************************
};

#endif //! RBM_PP_2_H