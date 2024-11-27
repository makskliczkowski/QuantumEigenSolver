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

/*
* @brief Template NQS for Spins (hardcore bosons)
*/
template <typename _Ht, typename _T, class _stateType>
class RBM_S<2, _Ht, _T, _stateType> : public RBM<2, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
	using NQSLS_p =	typename RBM<2, _Ht, _T, _stateType>::NQSLS_p;
public:
	RBM_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr,
	 uint _threadNum = 1, int _nParticles = -1, const NQSLS_p& _lower = {}, const std::vector<double>& _beta = {})
		: RBM<2, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles, _lower, _beta) 
	{ 
		this->init();
		this->setRandomState(true);
	};

	/* ----------------------------------------------------------- */
protected:
	// -------------------- P R O B A B I L I T Y -------------------
	virtual auto pRatio(uint fP, float fV)			-> _T	override;
	virtual auto pRatio(uint nFlips)				-> _T	override;
	virtual auto pRatio(const NQSS& _v1,					
					const NQSS& _v2)				-> _T	override;
	virtual auto pRatio(const NQSS& _v1)			-> _T	override;
	virtual auto pRatio(std::initializer_list<int> fP,		
				std::initializer_list<double> fV)	-> _T	override;
};

// !!!!!!!!!!!!!!!!!!! P R O B A B I L I T Y !!!!!!!!!!!!!!!!!!!

// %%%%%%%%%%%%%%%%%%% S I N G L E   F L I P %%%%%%%%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use only one flip
* @param fP place where the flip has been applied
* @param fV value before the flip!
* @returns probability ratio for a given ansatz based on the current state
*/
template<typename _Ht, typename _T, class _stateType>
inline _T RBM_S<2, _Ht, _T, _stateType>::pRatio(uint fP, float fV)
{
	// set the first value of b_visible
	_T val 			= 	RBM_SPIN_UPD(fV);

	// use value as the change already
#ifdef NQS_ANGLES_UPD
	//val				=	val * this->bV_(fP) + arma::sum(arma::log(arma::cosh(this->theta_ + val * this->W_.col(fP)) / this->thetaCOSH_));
	// val				=	std::exp(val * this->bV_(fP)) * arma::prod(arma::cosh(this->theta_ + val * this->W_.col(fP)) / this->thetaCOSH_);
	val				=	val * this->bV_(fP) + arma::sum(arma::log(arma::cosh(this->theta_ + val * this->W_.col(fP))) - arma::log(this->thetaCOSH_));
	val 			= 	std::exp(val);
#else
	// flip the temporary vector
	this->tmpVec_	=	this->curVec_;
	flip(this->tmpVec_, fP, Operators::_SPIN_RBM);
	// calculate
	val				=	val * this->bV_(fP) + arma::sum(arma::log(this->coshF(this->tmpVec_)) - arma::log(this->coshF(this->curVec_)));
	val				=	std::exp(val);
#endif
	return val;
}

// %%%%%%%%%%%%%%%% M U L T I P L E   F L I P S %%%%%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use multiple flips.
* Uses the flips stored within the NQS class (flipPlaces_, flipVals_)
* @param nFlips number of flips to be used
* @returns probability ratio for a given ansatz based on the current state
*/
template<typename _Ht, typename _T, class _stateType>
inline _T RBM_S<2, _Ht, _T, _stateType>::pRatio(uint nFlips)
{
	// you know what to do after one flip
	if (nFlips == 1)
		return RBM_S<2, _Ht, _T, _stateType>::pRatio(this->flipPlaces_[0], this->flipVals_[0]);
	// set the starting point
	_T val				=	0;
	// save the temporary angles
#ifdef NQS_NOT_OMP_MT
	auto thId				= std::this_thread::get_id();
	this->thetaTmp_[thId]	= this->theta_;
#else
	this->thetaTMP		=	this->theta_;
#endif // 
	// iterate through the flips
	for (uint i = 0; i < nFlips; ++i)
	{
		auto flipPlace	=	this->flipPlaces_[i];
		auto flipVal	=	this->flipVals_[i];
		_T currVal		=	RBM_SPIN_UPD(flipVal);

#ifdef NQS_NOT_OMP_MT
		this->thetaTmp_[thId] += currVal * this->W_.col(flipPlace);
#else
		this->thetaTmp_ += currVal * this->W_.col(flipPlace);
#endif
		val				+=	currVal * this->bV_(flipPlace);
	}
	// use value as the change already
#ifdef NQS_ANGLES_UPD
#	ifdef NQS_NOT_OMP_MT
	// val				=	std::exp(val) * arma::prod(arma::cosh(this->thetaTmp_[thId]) / this->thetaCOSH_);
	val				=	std::exp(val) * arma::prod(arma::cosh(this->thetaTmp_[thId]) / this->thetaCOSH_);
#	else
	// val				=	std::exp(val) * arma::prod(arma::cosh(this->thetaTmp_) / this->thetaCOSH_);
	val				=	std::exp(val) * arma::prod(arma::cosh(this->thetaTmp_) / this->thetaCOSH_);
#	endif
#else
	val				= val * this->bV_(fP) + arma::sum(arma::log(this->coshF(this->tmpVec_) / this->coshF(this->curVec_)));
	val				= std::exp(val);
#endif
	return val;
}

// %%%%%%%%%%%%%%%%% U S I N G   V E C T O R S %%%%%%%%%%%%%%%%%

/*
* @brief computes (Psi'/Psi), where (Psi') is the state with certain positions flipped. 
*/
template<typename _Ht, typename _T, class _stateType>
inline _T RBM_S<2, _Ht, _T, _stateType>::pRatio(const NQSS& _v1, const NQSS& _v2)
{
	_T val	=	arma::dot(this->bV_, arma::Col<double>(_v2 - _v1));
	val		+=	arma::sum(arma::log(this->coshF(_v2) / this->coshF(_v1)));
	return std::exp(val);
}

////////////////////////////////////////////////////////////////

template<typename _Ht, typename _T, class _stateType>
inline _T RBM_S<2, _Ht, _T, _stateType>::pRatio(const NQSS& _v1)
{
	_T val	= arma::dot(this->bV_, _v1 - this->curVec_);
#ifdef NQS_ANGLES_UPD
	val		+= arma::sum(arma::log(this->coshF(_v1) / this->thetaCOSH_));
#else
	val		+= arma::sum(arma::log(this->coshF(_v1) / this->coshF(this->curVec_)));
#endif
	return std::exp(val);
}

// %%%%%%%%%%%%% U S I N G   I N I T I A L I Z E R %%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use multiple flips ariving from the external vectors.
* Uses the flips stored within the NQS class (fP, fV)
* @param fP flip places to be used
* @param fV flip values to be used
* @returns probability ratio for a given ansatz based on the current state
*/
template<typename _Ht, typename _T, class _stateType>
inline _T RBM_S<2, _Ht, _T, _stateType>::pRatio(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
	// save the minimum of both sizes as new flip size if someone makes wrong movement
#ifndef NQS_USE_MULTITHREADING
	size_t nFlips = std::min(fP.size(), fV.size());
#else
	size_t nFlips	= fP.size();
	auto thId		= std::this_thread::get_id();
#endif

	// no flips!
	if (nFlips == 0) 
		return 1.0;

	// single flip only
	if (nFlips == 1) 
		return RBM_S<2, _Ht, _T, _stateType>::pRatio(*fP.begin(), *fV.begin());

	// set the starting point
	_T val			= 0;
	auto currVal	= 0.0;
	// make temporary angles vector
#ifdef NQS_NOT_OMP_MT
	this->thetaTmp_[thId] = this->theta_;
#else
	auto thetaTmp_ = this->theta_;
#endif
	// iterate through the flips
	for (uint i = 0; i < nFlips; ++i)
	{
		auto flipPlace	= *(fP.begin() + i);
		auto flipVal	= *(fV.begin() + i);
		currVal			= RBM_SPIN_UPD(flipVal);

		// !TODO speed this up by not creating thetaTMP
#ifdef NQS_NOT_OMP_MT
		this->thetaTmp_[thId]	+= currVal * this->W_.col(flipPlace);
#else
		thetaTmp_				+= currVal * this->W_.col(flipPlace);
#endif
		val						+= currVal * this->bV_(flipPlace);
	}
	// use value as the change already
#ifdef NQS_ANGLES_UPD
#	ifdef NQS_NOT_OMP_MT
	val = std::exp(val) * arma::prod(arma::cosh(this->thetaTmp_[thId]) / this->thetaCOSH_);
#	else
	val = std::exp(val) * arma::prod(arma::cosh(thetaTmp_) / this->thetaCOSH_);
#	endif
#else
	val = val * arma::prod(this->coshF(this->tmpVec) / this->coshF(this->curVec));
#endif
	return val;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#endif