#pragma once
#ifndef RBM_H
#define RBM_H

#ifndef NQS_H
	#include "../nqs_final.hpp"
#endif // !NQS_H

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! B A S E !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

/*
* @brief Restricted Boltzmann Machines ansatz for quantum state - base
*/
template < 	uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class RBM : public NQS_S<_spinModes, _Ht, _T, _stateType>
{
public:
	NQS_PUBLIC_TYPES(_T, _stateType);
	using NQSLS_p =	typename NQS_S<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
protected:
	// architecture parameters
	uint nHid_						=						1;
	u64 rbmSize_					=						1;

	/* ------------------------------------------------------------ */
	// ------------------------ W E I G H T S ------------------------
	NQSW W_;												// weight matrix
	NQSB bV_;												// visible bias
	NQSB bH_;												// hidden bias
	
	// ------------------------- A N G L E S -------------------------
	NQSB theta_, thetaCOSH_, thetaCOSH_log_;				// for storing the angles for the RBM

	// calculate the hiperbolic cosine of the function to obtain the ansatz
	auto coshF(const NQSS& _v)		const -> NQSB			{ return arma::cosh(this->bH_ + this->W_ * _v);		};
	auto coshF()					const -> NQSB			{ return arma::cosh(this->theta_);					};
	// ---------------------- T H R E A D I N G ---------------------
#ifdef NQS_NOT_OMP_MT
    thread_local static inline NQSB thetaTMP_;				// Thread-local storage for thetaTMP
#endif
	NQSB thetaTmpCol_;
	
	/* ------------------------------------------------------------ */
protected:
	// ----------------------- S T A R T E R S -----------------------
	virtual void setInfo()									override;
	virtual void allocate()									override;

	// ------------------------ S E T T E R S ------------------------
	virtual void setState(const NQSS& _st, bool _set)		override;
	virtual void setState(u64 _st, bool _set)				override;

	/* ------------------------------------------------------------ */
	// -------------------- P R O B A B I L I T Y --------------------
	virtual auto logPRatio(uint fP, float fV)				-> _T override = 0;
	virtual auto logPRatio(uint nFlips)						-> _T override = 0;
	virtual auto logPRatio(const NQSS& _v1, const NQSS& _v2)-> _T override = 0;
	virtual auto logPRatio(const NQSS& _v1)					-> _T override = 0;
	virtual auto logPRatio(std::initializer_list<int> fP,
						std::initializer_list<double> fV) 	-> _T override = 0;

	// ------------------------ W E I G H T S ------------------------
public:
	virtual void setWeights(const NQSW& _W, 
							const NQSB& _bV, 
							const NQSB& _bH); 
	virtual void setWeights(std::shared_ptr<RBM<_spinModes, _Ht, _T, _stateType>> _rbm);
	virtual bool setWeights(std::string _path, 
							std::string _file)				override;
	virtual bool saveWeights(std::string _path, 
							std::string _file)				override;
protected:
	virtual void updateWeights()							override;
	// set the angles for the RBM to be updated
	void setTheta()											{ this->setTheta(this->curVec_); };
	void setTheta(const NQSS& v);
	// updaters
#ifdef NQS_ANGLES_UPD
	virtual void update(uint nFlips)						override;
	virtual void update(const NQSS& v, uint nFlips)			override;
#endif

	// ------------------------- T R A I N ------------------------------	
	virtual void grad(const NQSS& _v, uint _plc)			override;

	// -------------------------------------------------------------------
public:
	virtual ~RBM()											{ DESTRUCTOR_CALL; };
	RBM(std::shared_ptr<Hamiltonian<_Ht, _spinModes>>& _H, uint _nHid, double _lr, uint _threadNum = 1, int _nPart = -1,
													const NQSLS_p& _lower = {}, 
													const std::vector<double>& _beta = {});

	// --------------------- S E T T E R S ---------------------
	virtual void init()										override;
	
	// --------------------- G E T T E R S ---------------------
	auto getNhid()					const -> uint			{ return this->nHid_; 		};
	auto getWeights()				const -> NQSW			{ return this->W_; 			};
	auto getVisibleBias()			const -> NQSB			{ return this->bV_; 		};
	auto getHiddenBias()			const -> NQSB			{ return this->bH_; 		};
	
	// --------------------- F I N A L E -----------------------
	virtual auto ansatz(const NQSS& _in) 					const -> _T override;
	virtual auto ansatzlog(const NQSS& _in) 				const -> _T override;
	virtual auto ansatz_ratiolog(const NQSS& _in, 
		NQS<_spinModes, _Ht, _T, _stateType>* _other) 		const -> _T override;
};	

#define RBM_INST_CMB(_Ht, _T, FUN, FUNRET, ARGS, ADD) 							\
					template FUNRET  RBM<2u, _Ht, _T, double>::FUN ARGS ADD; 	\
					template FUNRET  RBM<3u, _Ht, _T, double>::FUN ARGS ADD; 	\
					template FUNRET  RBM<4u, _Ht, _T, double>::FUN ARGS ADD; 	

////////////////////////////////////////////////////////////////////////////

/*
* @brief sets the current angles vector according to arXiv:1606.02318v1
* @param v replaces current vector
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::setTheta(const NQSS& v)
{
	this->theta_		= this->bH_ + this->W_ * v;
	this->thetaCOSH_	= this->coshF();
	this->thetaCOSH_log_= arma::log(this->thetaCOSH_);
}

////////////////////////////////////////////////////////////////////////////

/*
* @brief Updates the weights in the system according to a given gradient
* @warning uses forces vector (member of NQS : F_) to update the gradients - preallocation for optimization
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::updateWeights()
{
	this->bV_	-= this->dF_.subvec(0, this->info_p_.nVis_ - 1);
	this->bH_	-= this->dF_.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1);
	this->W_	-= arma::reshape(this->dF_.subvec(this->info_p_.nVis_ + this->nHid_, this->rbmSize_ - 1),
								this->W_.n_rows, this->W_.n_cols);
}



// ##########################################################################################################################################

// ############################################################ U P D A T E R S #############################################################

// ##########################################################################################################################################

///////////////////////////////////////////////////////////////////////

#ifdef NQS_ANGLES_UPD
/*
* @brief Update angles with the flipped spin (spins)
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::update(uint nFlips)
{
	for (uint i = 0; i < nFlips; ++i)
	{
#ifdef SPIN
		this->theta_	-=	(2.0 * this->flipVals_[i]) * this->W_.col(this->flipPlaces_[i]);
#else
		this->theta_	+=	(1.0 - 2.0 * this->flipVals_[i]) * this->W.col(flipPlaces_[i]);
#endif
	}
	this->thetaCOSH_	=	this->coshF();
	this->thetaCOSH_log_= 	arma::log(this->thetaCOSH_);
}

///////////////////////////////////////////////////////////////////////

/*
* @brief Update angles with the flipped spin (spins)
* @param vector after the flips has been done
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::update(const NQSS& v, uint nFlips)
{
	for (uint i = 0; i < nFlips; i++)
	{
		const auto fP	=	this->flipPlaces_[i];
#ifdef SPIN
		this->theta_	-=	(2.0 * v(fP)) * this->W_.col(fP);
#else
		this->theta_	+=	(1.0 - 2.0 * v(fP)) * this->W.col(fP);
#endif
	}
	this->thetaCOSH_	=	this->coshF();
	this->thetaCOSH_log_= 	arma::log(this->thetaCOSH_);
}
#endif

#endif