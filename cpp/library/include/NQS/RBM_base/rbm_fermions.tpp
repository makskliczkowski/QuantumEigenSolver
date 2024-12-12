#include "rbm_spins.tpp"
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FERMIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ifndef RBM_4_H
#define RBM_4_H

/*
* @brief Template NQS for Spins (hardcore bosons)
*/
template <typename _Ht, typename _T, class _stateType>
class RBM_S<4, _Ht, _T, _stateType> : public RBM<4, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	using NQSLS_p =	typename RBM<2, _Ht, _T, _stateType>::NQSLS_p;
public:
	RBM_S(std::shared_ptr<Hamiltonian<_Ht, 4>>& _H, uint _nHid, double _lr,
		uint _threadNum = 1, int _nParticles = -1, const NQSLS_p& _lower = {}, const std::vector<double>& _beta = {})
		: RBM<4, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles, _lower, _beta) 
	{ 
		this->init();
		this->setRandomState(true);
	};

	/* ----------------------------------------------------------- */
protected:
	// -------------------- P R O B A B I L I T Y -------------------
	virtual auto logPRatio(uint fP, float fV)		-> _T	override { return _T(1.0); }
	virtual auto logPRatio(uint nFlips)				-> _T	override { return _T(1.0); }
	virtual auto logPRatio(Config_cr_t _v1,					
					Config_cr_t _v2)				-> _T	override { return _T(1.0); }
	virtual auto logPRatio(Config_cr_t _v1)			-> _T	override { return _T(1.0); }
	virtual auto logPRatio(std::initializer_list<int> fP,		
				std::initializer_list<double> fV)	-> _T	override { return _T(1.0); }

	/////////////////////////////////////////////////////////////////

	virtual MC_t_p clone() const override {
		return std::make_shared<RBM_S>(*this); 
	}
};

// ##########################################################################################################################################
#endif