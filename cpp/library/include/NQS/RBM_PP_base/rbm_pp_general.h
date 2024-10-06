//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GENERAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////
#ifndef RBM_PP_GEN_H
#define RBM_PP_GEN_H

template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class RBM_PP_S : public RBM_PP<_spinModes, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);

	RBM_PP_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum = 1, int _nParticles = -1)
		: RBM_PP<_spinModes, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles)
																	 { NQS_LOG_ERROR_SPIN_MODES; };

	/* ------------------------------------------------------- */
	// --------------------- G E T T E R S ---------------------
	virtual auto getPPMat(const NQSS& _n)	const -> NQSW	override { NQS_LOG_ERROR_SPIN_MODES; return NQSW(); };
#ifndef NQS_USE_VEC_ONLY
	virtual auto getPPMat(u64 _n)			const -> NQSW	override { NQS_LOG_ERROR_SPIN_MODES; return NQSW(); };
#endif
protected:
	// --------------------------- A N S A T Z ---------------------------
	void updFPP_C(uint fP, float fV)						override { NQS_LOG_ERROR_SPIN_MODES; };
	void updFPP_C(std::initializer_list<int> fP,
				std::initializer_list<double> fV)			override { NQS_LOG_ERROR_SPIN_MODES; };
	void updFPP(uint fP, float fV)							override { NQS_LOG_ERROR_SPIN_MODES; };
	void updFPP(std::initializer_list<int> fP,
				std::initializer_list<double> fV)			override { NQS_LOG_ERROR_SPIN_MODES; };
	void updFPP_F(std::initializer_list<int> fP,
				std::initializer_list<double> fV,
				arma::Mat<_T>& _Xtmp)						override { NQS_LOG_ERROR_SPIN_MODES; };
};
#endif