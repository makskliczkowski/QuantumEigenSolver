#include "rbm_general.inl"
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SPINS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////
#ifndef RBM_2_H
#define RBM_2_H

#ifdef SPIN
#	define RBM_SPIN_UPD(_val) -2.0 * _val
#else
#	define RBM_SPIN_UPD(_val) 1.0 - 2.0 * _val
#endif

// #define RBM_ACCU_ARMA arma::accu
#define RBM_ACCU_ARMA arma::sum

/*
* @brief Template NQS for Spins (hardcore bosons)
*/
template <typename _Ht, typename _T, class _stateType>
class RBM_S<2, _Ht, _T, _stateType> : public RBM<2, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	using NQSLS_p =	typename RBM<2, _Ht, _T, _stateType>::NQSLS_p;
public:
	RBM_S(std::shared_ptr<Hamiltonian<_Ht, 2>>& _H, uint _nHid, double _lr,
		uint _threadNum = 1, int _nParticles = -1, const NQSLS_p& _lower = {}, const std::vector<double>& _beta = {})
		: RBM<2, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles, _lower, _beta) 
	{ 
		this->init();
		this->setRandomState(true);
	};

	/* ----------------------------------------------------------- */
protected:
	// -------------------- P R O B A B I L I T Y -------------------
	virtual auto logPRatio(uint fP, float fV)		-> _T	override;
	virtual auto logPRatio(uint nFlips)				-> _T	override;
	virtual auto logPRatio(const NQSS& _v1,					
					const NQSS& _v2)				-> _T	override;
	virtual auto logPRatio(const NQSS& _v1)			-> _T	override;
	virtual auto logPRatio(std::initializer_list<int> fP,		
				std::initializer_list<double> fV)	-> _T	override;

	/////////////////////////////////////////////////////////////////

	virtual MC_t_p clone() const override {
		return std::make_shared<RBM_S>(*this); 
	}

	/////////////////////////////////////////////////////////////////
};

// ##########################################################################################################################################

#endif