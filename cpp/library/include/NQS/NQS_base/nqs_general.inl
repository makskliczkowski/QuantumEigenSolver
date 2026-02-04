#include "nqs.h"
// ##########################################################################################################################################

namespace NQS_NS
{
	template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
	class NQS_S : public NQS<_spinModes, _Ht, _T, _stateType>
	{
		// ***********************************************************************************
		MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
		NQS_PUBLIC_TYPES(_T, _stateType);
		NQS_HAMIL_TYPES(_Ht, _spinModes);									// type definitions for the Hamiltonian
		using NQSLS_p =	typename NQS<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
		// ***********************************************************************************
	public:
		NQS_S() = default;
		NQS_S(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p)
			: NQS<_spinModes, _Ht, _T, _stateType>(_p)						{ NQS_LOG_ERROR_SPIN_MODES; };
		NQS_S(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
			: NQS<_spinModes, _Ht, _T, _stateType>(_p, _lower, _beta)		{ NQS_LOG_ERROR_SPIN_MODES; };	
		// ***********************************************************************************
	protected:
		// -------------------------- F L I P S --------------------------
		virtual void chooseRandomFlips()			override { NQS_LOG_ERROR_SPIN_MODES; };

		// apply flips to the temporary vector or the current vector according the template
		virtual void applyFlipsT()					override { NQS_LOG_ERROR_SPIN_MODES; };
		virtual void applyFlipsC()					override { NQS_LOG_ERROR_SPIN_MODES; };
		virtual void setRandomFlipNum(uint _nFlips)	override { NQS_LOG_ERROR_SPIN_MODES; };

		// ***********************************************************************************
		virtual auto clone() 						const -> MC_t_p override = 0;
		virtual auto clone(MC_t_p _n) 				-> void override { NQS<_spinModes, _Ht, _T, _stateType>::clone(_n); };
		// ***********************************************************************************
	};

	// #######################################################################################################################################

	// create a concept that it needs to derive from the NQS_S

	// #######################################################################################################################################

	template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double, class _CorrState = NQS_S<_spinModes, _Ht, _T, _stateType>>
	// concept NQS_S_concept = std::is_base_of<NQS<_spinModes, _Ht, _T, _stateType>, _CorrState>::value;
	concept NQS_S_concept = std::is_convertible<_CorrState*, NQS_S<_spinModes, _Ht, _T, _stateType>*>::value && 
							std::is_constructible_v<_CorrState>;


	// #######################################################################################################################################

};	// namespace NQS_NS
