#include "nqs.h"

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GENERAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class NQS_S : public NQS<_spinModes, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver

	using NQSLS_p =	typename NQS<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
public:
	NQS_S(std::shared_ptr<Hamiltonian<_Ht, _spinModes>>& _H, double _lr, uint _threadNum, int _nParticles, 
										const NQSLS_p& _lower, const std::vector<double>& _beta)
		: NQS<_spinModes, _Ht, _T, _stateType>(_H, _lr, _threadNum, _nParticles, _lower, _beta)
															{ NQS_LOG_ERROR_SPIN_MODES; };
protected:
	// -------------------------- F L I P S --------------------------
	virtual void chooseRandomFlips()			override { NQS_LOG_ERROR_SPIN_MODES; };

	// apply flips to the temporary vector or the current vector according the template
	virtual void applyFlipsT()					override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual void applyFlipsC()					override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual void setRandomFlipNum(uint _nFlips)	override { NQS_LOG_ERROR_SPIN_MODES; };

	///////////////////////////////////////////////////////////////////////////////////

	virtual auto clone() 						const -> MC_t_p override = 0;

	///////////////////////////////////////////////////////////////////////////////////
};