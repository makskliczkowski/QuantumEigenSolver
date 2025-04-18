#include "rbm_general.inl"
// ##########################################################################################################################################

#ifndef RBM_2_H
#define RBM_2_H

#ifdef SPIN
#	define RBM_SPIN_UPD(_val) -2.0 * _val
#else
#	define RBM_SPIN_UPD(_val) 1.0 - 2.0 * _val
#endif

#define RBM_ACCU arma::accu
// #define RBM_ACCU arma::sum
namespace NQS_NS
{
	/**
	* @brief Template NQS for Spins (hardcore bosons)
	*/
	template <typename _Ht, typename _T, class _stateType>
	class RBM_S<2u, _Ht, _T, _stateType> : public RBM<2u, _Ht, _T, _stateType>
	{
		// **********************************************************************************************************************
		MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
		NQS_PUBLIC_TYPES(_T, _stateType);
		NQS_HAMIL_TYPES(_Ht, 2);
		using NQSLS_p =	typename RBM<2, _Ht, _T, _stateType>::NQSLS_p;
		// **********************************************************************************************************************
	public:
		// **********************************************************************************************************************
		RBM_S()															= default;
		// **********************************************************************************************************************
		RBM_S(const NQS_Const_par_t<2u, _Ht, _T, _stateType>& _p)
			: RBM<2u, _Ht, _T, _stateType>(_p) 
		{ 
			this->init();
		};
		// **********************************************************************************************************************
		RBM_S(const NQS_Const_par_t<2u, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
			: RBM<2u, _Ht, _T, _stateType>(_p, _lower, _beta) 
		{ 
			this->init(); 
		};
		// **********************************************************************************************************************
		RBM_S(const RBM_S<2u, _Ht, _T, _stateType>& _other)
			: RBM<2u, _Ht, _T, _stateType>(_other)
		{
		};
		// **********************************************************************************************************************
		RBM_S(RBM_S<2u, _Ht, _T, _stateType>&& _other)
			: RBM<2u, _Ht, _T, _stateType>(std::move(_other))
		{
		};
		// **********************************************************************************************************************
	protected:
		// -------------------- P R O B A B I L I T Y -------------------
		virtual auto logPRatio(uint fP, float fV)						-> _T	override;
		virtual auto logPRatio(uint f1, uint f2, float v1, float v2) 	-> _T	override;
		virtual auto logPRatio(uint nFlips)								-> _T	override;
		virtual auto logPRatio(Config_cr_t _v1, Config_cr_t _v2)		-> _T	override;
		virtual auto logPRatio(Config_cr_t _v1)							-> _T	override;
		virtual auto logPRatio(int_ini_t fP, dbl_ini_t fV)				-> _T	override;

		/////////////////////////////////////////////////////////////////
		virtual MC_t_p clone() 											const override { return std::make_shared<RBM_S>(*this); }
		virtual void clone(MC_t_p _n) 									override { RBM<2, _Ht, _T, _stateType>::clone(_n); }
		/////////////////////////////////////////////////////////////////
	};
};
// ##########################################################################################################################################

#endif