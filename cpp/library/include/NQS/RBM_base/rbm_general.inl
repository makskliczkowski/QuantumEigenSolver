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
	NQS_PUBLIC_TYPES(_T, _stateType);
	using NQSLS_p =	RBM<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
public:
	RBM_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr,
					 uint _threadNum = 1, int _nParticles = -1, const NQSLS_p& _lower = {}, const std::vector<double>& _beta = {})
		: RBM<_spinModes, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles, _lower, _beta)
																	 { NQS_LOG_ERROR_SPIN_MODES; };

	/* ------------------------------------------------------------ */
	protected:
	// -------------------- P R O B A B I L I T Y --------------------
	virtual auto pRatio(uint fP, float fV)			-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual auto pRatio(uint nFlips)				-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual auto pRatio(const NQSS& _v1,					
					const NQSS& _v2)				-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual auto pRatio(const NQSS& _v1)			-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual auto pRatio(std::initializer_list<int> fP,		
				std::initializer_list<double> fV)	-> _T	override { NQS_LOG_ERROR_SPIN_MODES; };
};

#endif