#include "rbm_general.inl"
// ##########################################################################################################################################
#ifndef RBM_2_H
#define RBM_2_H

#ifdef SPIN
#	define RBM_SPIN_UPD(_val) -2.0 * _val
#else
#	define RBM_SPIN_UPD(_val) 1.0 - 2.0 * _val
#endif

#define RBM_ACCU_ARMA arma::accu
// #define RBM_ACCU_ARMA arma::sum

/*
* @brief Template NQS for Spins (hardcore bosons)
*/
template <typename _Ht, typename _T, class _stateType>
class RBM_S<2, _Ht, _T, _stateType> : public RBM<2, _Ht, _T, _stateType>
{
	// **********************************************************************************************************************
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	NQS_PUBLIC_TYPES(_T, _stateType);
	NQS_HAMIL_TYPES(_Ht, 2);
	using NQSLS_p =	typename RBM<2, _Ht, _T, _stateType>::NQSLS_p;
	// **********************************************************************************************************************
public:
	RBM_S(const NQS_Const_par_t<2, _Ht, _T, _stateType>& _p)
		: RBM<2, _Ht, _T, _stateType>(_p) 
	{ 
		this->init();
	};

	RBM_S(const NQS_Const_par_t<2, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
		: RBM<2, _Ht, _T, _stateType>(_p, _lower, _beta) 
	{ 
		this->init(); 
	};
	// **********************************************************************************************************************
protected:
	// -------------------- P R O B A B I L I T Y -------------------
	virtual auto logPRatio(uint fP, float fV)						-> _T	override;
	virtual auto logPRatio(uint f1, uint f2, float v1, float v2) 	-> _T	override;
	virtual auto logPRatio(uint nFlips)								-> _T	override;
	virtual auto logPRatio(Config_cr_t _v1,					
					Config_cr_t _v2)								-> _T	override;
	virtual auto logPRatio(Config_cr_t _v1)							-> _T	override;
	virtual auto logPRatio(std::initializer_list<int> fP,		
				std::initializer_list<double> fV)					-> _T	override;

	/////////////////////////////////////////////////////////////////

	virtual MC_t_p clone() const override 
	{
		return std::make_shared<RBM_S>(*this); 
	}

	/////////////////////////////////////////////////////////////////
};

// ##########################################################################################################################################

#endif