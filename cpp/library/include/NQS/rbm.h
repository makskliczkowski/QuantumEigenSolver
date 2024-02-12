#pragma once
#ifndef RBM_H
#define RBM_H

#ifndef NQS_H
	#include "../nqs.h"
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
protected:
	// architecture parameters
	uint nHid_						=						1;


	/* ------------------------------------------------------------ */
	// ------------------------ W E I G H T S ------------------------
	NQSW W_;												// weight matrix
	NQSB bV_;												// visible bias
	NQSB bH_;												// hidden bias
	
	// ------------------------- A N G L E S -------------------------
	NQSB theta_;
	NQSB thetaCOSH_;
	// calculate the hiperbolic cosine of the function to obtain the ansatz
	auto coshF(const NQSS& _v)		const -> NQSB			{ return arma::cosh(this->bH_ + this->W_ * _v);		};
	auto coshF()					const -> NQSB			{ return arma::cosh(this->theta_);					};
	// ---------------------- T H R E A D I N G ---------------------
#if defined NQS_USE_CPU && not defined NQS_USE_OMP && not defined _DEBUG
	// create the map for thetas for a given thread
	std::map<std::thread::id, NQSB> thetaTmp_;
#endif
	
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
	virtual auto pRatio(uint fP, float fV)			-> _T	override = 0;
	virtual auto pRatio(uint nFlips)				-> _T	override = 0;
	virtual auto pRatio(const NQSS& _v1,
					const NQSS& _v2)				-> _T	override = 0;
	virtual auto pRatio(const NQSS& _v1)			-> _T	override = 0;
	virtual auto pRatio(std::initializer_list<int> fP,
						std::initializer_list<double> fV) -> _T	override = 0;

	// ------------------------ W E I G H T S ------------------------
public:
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
	RBM(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum = 1, int _nPart = -1);

	// --------------------- S E T T E R S ---------------------
	virtual void init()										override;
	
	// --------------------- G E T T E R S ---------------------
	auto getNhid()					const -> uint			{ return this->nHid_; };
	
	// --------------------- F I N A L E -----------------------
	virtual auto ansatz(const NQSS& _in) const->_T			override;
};
 
// ##########################################################################################################################################
// ##########################################################################################################################################
// ########################################################### C O N S T R U C T ############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM<_spinModes, _Ht, _T, _stateType>::RBM(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum, int _nPart)
	: NQS_S<_spinModes, _Ht, _T, _stateType>(_H, _lr, _threadNum, _nPart)
{
	this->nHid_ = _nHid;
	this->fullSize_ = this->nHid_ + this->nVis_ + this->nHid_ * this->nVis_;
	this->allocate();
	this->setInfo();
	this->init();
	this->setRandomState(true);
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################## A N S A T Z ###############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief reproduces the RBM NQS ANSATZ.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
_T RBM<_spinModes, _Ht, _T, _stateType>::ansatz(const NQSS& _in) const
{
	return (std::exp(arma::dot(this->bV_, _in)) * arma::prod(this->coshF(_in))) / std::sqrt(this->nVis_);
};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ######################################################## I N I T I A L I Z E R S #########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Allocate the memory for the NQS parameters. 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::allocate()
{
	// allocate weights
	this->bV_.resize(this->nVis_);
	this->bH_.resize(this->nHid_);
	this->theta_.resize(this->nHid_);
	this->thetaCOSH_.resize(this->nHid_);
	this->W_.resize(this->nHid_, this->nVis_);
	// create thread map
#if defined NQS_USE_CPU && not defined NQS_USE_OMP && not defined _DEBUG
	// allocate the vector for using it in the RBM
	for (int _thread = 0; _thread < this->threadNum_; _thread++)
		this->thetaTmp_[this->threads_[_thread].get_id()] = NQSB(this->nHid_);
#endif
	// allocate the rest
	NQS_S<_spinModes, _Ht, _T, _stateType>::allocate();
}

////////////////////////////////////////////////////////////////////////////

/*
* @brief Initializes the weights and biases for the RBM NQS model
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::init()
{
	// initialize biases visible
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum_)
#endif
	for (uint i = 0; i < this->nVis_; i++)
		this->bV_(i) = algebra::cast<_T>(0.05 * (this->ran_.template random<double>(-1.0, 1.0) + I * this->ran_.template randomNormal<double>(-1.0, 1.0)));
	// initialize biases hidden
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum_)
#endif
	for (uint i = 0; i < this->nHid_; i++)
		this->bH_(i) = algebra::cast<_T>(0.05 * (this->ran_.template random<double>(-1.0, 1.0) + I * this->ran_.template randomNormal<double>(-1.0, 1.0)));
	// weights matrix
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum_)
#endif
	for (uint i = 0; i < this->W_.n_rows; i++)
		for (uint j = 0; j < this->W_.n_cols; j++)
			this->W_(i, j) = algebra::cast<_T>(0.05 * (this->ran_.template random<double>(-1.0, 1.0) + I * this->ran_.template randomNormal<double>(-1.0, 1.0)));
	// initialize with a random state
	this->setRandomState();
}

////////////////////////////////////////////////////////////////////////////

/*
* @brief Sets the info about the RBM model NQS.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::setInfo()
{
	this->info_ = "";
	strSeparatedS(this->info_, ',', "RBM", 
		VEQV(mod, this->H_->getType()), 
		VEQV(nv, this->nVis_), 
		VEQV(nh, this->nHid_), 
		VEQV(nS, this->spinModes_),
		VEQVS(lr, this->lr_));
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# S E T T E R S ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Sets the new state according to a given vector. Updates the value if NQS_ANGLES_UPD is set.
* @param _st Column vector to be set as a new state
* @param _set if one wants to additionaly set the angles
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::setState(const NQSS& _st, bool _set)
{
	NQS_S<_spinModes, _Ht, _T, _stateType>::setState(_st);
#ifdef NQS_ANGLES_UPD
	if (_set) this->setTheta();
#endif
}

////////////////////////////////////////////////////////////////////////////

/*
* @brief Sets the new state according to a given integer state. Updates the value if NQS_ANGLES_UPD is set.
* @param _st Column vector to be set as a new state
* @param _set if one wants to additionaly set the angles
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::setState(u64 _st, bool _set)
{
	NQS_S<_spinModes, _Ht, _T, _stateType>::setState(_st);
#ifdef NQS_ANGLES_UPD
	if (_set) this->setTheta();
#endif
}

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
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# W E I G H T S ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @returns whether the load has been successful
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool RBM<_spinModes, _Ht, _T, _stateType>::setWeights(std::string _path, std::string _file)
{
	BEGIN_CATCH_HANDLER
	{
		// set the forces vector for the weights
		if(!NQS_S<_spinModes, _Ht, _T, _stateType>::setWeights(_path, _file))
			return false;
		this->bV_	= this->F_.subvec(0, this->nVis_ - 1);
		this->bH_	= this->F_.subvec(this->nVis_, this->nVis_ + this->nHid_ - 1);
		this->W_	= arma::reshape(this->F_.subvec(this->nVis_ + this->nHid_, this->nVis_ + this->nHid_ + this->W_.n_rows * this->W_.n_cols - 1),
									this->W_.n_rows, this->W_.n_cols);
	}
	END_CATCH_HANDLER("Couldn't set the weights for the RBM NQS...", return false);
	return true;
}

////////////////////////////////////////////////////////////////////////////

/*
* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @returns whether the load has been successful
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool RBM<_spinModes, _Ht, _T, _stateType>::saveWeights(std::string _path, std::string _file)
{
	BEGIN_CATCH_HANDLER
	{
		this->F_.subvec(0, this->nVis_ - 1) = this->bV_;
		this->F_.subvec(this->nVis_, this->nVis_ + this->nHid_ - 1) = this->bH_;
		this->F_.subvec(this->nVis_ + this->nHid_, this->nVis_ + this->nHid_ + this->W_.n_rows * this->W_.n_cols - 1) = this->W_.as_col();
		
		// set the forces vector for the weights
		if(!NQS_S<_spinModes, _Ht, _T, _stateType>::saveWeights(_path, _file))
			return false;
	}
	END_CATCH_HANDLER("Couldn't save the weights for the RBM NQS...", return false);
	return true;
}

////////////////////////////////////////////////////////////////////////////

/*
* @brief Updates the weights in the system according to a given gradient
* @warning uses forces vector (member of NQS : F_) to update the gradients - preallocation for optimization
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::updateWeights()
{
	this->bV_ -= this->F_.subvec(0, this->nVis_ - 1);
// #ifndef _DEBUG
// #pragma omp parallel for num_threads(this->threadsNumLeft_)
// #endif		
	//for (int i = 0; i < this->nVis_; i++)
			//this->bV_(i) -= this->F_(i);

	this->bH_ -= this->F_.subvec(this->nVis_, this->nVis_ + this->nHid_ - 1);
// #ifndef _DEBUG
// #pragma omp parallel for num_threads(this->threadsNumLeft_)
// #endif	
		for (int i = 0; i < this->nHid_; i++)
			this->bH_(i) -= this->F_(i + this->nVis_);
	this->W_ -= arma::reshape(this->F_.subvec(this->nVis_ + this->nHid_, this->fullSize_ - 1), this->W_.n_rows, this->W_.n_cols);
// #ifndef _DEBUG
// #pragma omp parallel for num_threads(this->threadsNumLeft_)
// #endif	
		//for (int i = 0; i < this->nHid_; i++)
		//	for (auto j = 0; j < this->nVis_; j++)
		//		this->W_(i, j) -= this->F_((this->nVis_ + this->nHid_) + i + j * this->nHid_);

}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ G R A D I E N T #############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief At each step calculates the variational derivatives and stores them in the _derivatives matrix.
* @param _v vector to calculate the derivatives for
* @param _plc row at which to store the derivatives
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::grad(const NQSS& _v, uint _plc)
{
	// update the angles if it is necessary
#ifndef NQS_ANGLES_UPD
	this->setTheta(_v);
#endif

	// calculate the flattened part
	//this->derivatives_.submat(_plc, 0, _plc, this->nVis_ - 1) = arma::conv_to<arma::Row<_T>>::from(_v);
//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threadsNumLeft_)
//#endif
	for (uint i = 0; i < this->nVis_; ++i)
		this->derivatives_(_plc, i) = _v(i);

	this->derivatives_.submat(_plc, this->nVis_, _plc, this->nVis_ + this->nHid_ - 1) = arma::tanh(this->theta_).as_row();
//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threadsNumLeft_)
//#endif
	//for (uint i = 0; i < this->nHid_; ++i)
		//this->derivatives_(_plc, i + this->nVis_) = std::tanh(this->theta_(i));

//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threadsNumLeft_)
//#endif
	for (int i = 0; i < this->nHid_; ++i)
		for (auto j = 0; j < this->nVis_; j++)
			this->derivatives_(_plc, (this->nVis_ + this->nHid_) + i + j * this->nHid_) = this->derivatives_(_plc, i + this->nVis_) * _v(j);
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ U P D A T E R S #############################################################
// ##########################################################################################################################################
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

}
#endif
////////////////////////////////////////////////////////////////////////////

// ##########################################################################################################################################
// ##########################################################################################################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

//////////////////////////////////////////////////////////////////////////////////////////
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GENERAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//////////////////////////////////////////////////////////////////////////////////////////
#	ifndef RBM_GEN_H
#		define RBM_GEN_H
template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class RBM_S : public RBM<_spinModes, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
public:
	RBM_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum = 1, int _nParticles = -1)
		: RBM<_spinModes, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles)
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
#	endif
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SPINS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//////////////////////////////////////////////////////////////////////////////////////////
#	ifndef RBM_2_H
#		define RBM_2_H

/*
* @brief Template NQS for Spins (hardcore bosons)
*/
template <typename _Ht, typename _T, class _stateType>
class RBM_S<2, _Ht, _T, _stateType> : public RBM<2, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);

	RBM_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum = 1, int _nParticles = -1)
		: RBM<2, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles) { };

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

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!! P R O B A B I L I T Y !!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
#ifdef SPIN
	_T val			=	-2.0 * fV;
#else
	_T val			=	1.0 - 2.0 * fV;
#endif
	// use value as the change already
#ifdef NQS_ANGLES_UPD
	//val				=	val * this->bV_(fP) + arma::sum(arma::log(arma::cosh(this->theta_ + val * this->W_.col(fP)) / this->thetaCOSH_));
	//val				=	val * this->bV_(fP) + arma::sum(arma::log(arma::cosh(this->theta_ + val * this->W_.col(fP)) / arma::cosh(this->theta_)));
	val				=	std::exp(val * this->bV_(fP)) * arma::prod(arma::cosh(this->theta_ + val * this->W_.col(fP)) / this->thetaCOSH_);
#else
	// flip the temporary vector
	this->tmpVec_	=	this->curVec_;
	flip(this->tmpVec_, fP, Operators::_SPIN_RBM);
	// calculate
	val				=	val * this->bV_(fP) + arma::sum(arma::log(this->coshF(this->tmpVec_) / this->coshF(this->curVec_)));
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
		return NQS_S<2, _Ht, _T, _stateType>::pRatio();
	// set the starting point
	_T val				=	0;
	// save the temporary angles
	NQSB thetaTMP		=	this->theta_;
	// iterate through the flips
	for (uint i = 0; i < nFlips; ++i)
	{
		auto flipPlace	=	this->flipPlaces_[i];
		auto flipVal	=	this->flipVals_[i];
		// set the first value of b_visible
#ifdef SPIN
		_T currVal		=	-2.0 * flipVal;
#else
		_T currVal		=	1.0 - 2.0 * flipVal;
#endif
		thetaTMP		+=	currVal * this->W_.col(flipPlace);
		val				+=	currVal * this->bV_(flipPlace);
	}
	// use value as the change already
#ifdef NQS_ANGLES_UPD
	val				=	std::exp(val) * arma::prod(arma::cosh(thetaTMP) / this->thetaCOSH_);
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
	_T val	= arma::dot(this->bV_, arma::Col<double>(_v1 - this->curVec_));
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
#ifdef _DEBUG
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
		return this->pRatio(*fP.begin(), *fV.begin());

	// set the starting point
	_T val			= 0;
	auto currVal	= 0.0;
	// make temporary angles vector
#if defined NQS_USE_CPU && not defined NQS_USE_OMP && not defined _DEBUG
	this->thetaTmp_[thId] = this->theta_;
#else
	auto thetaTmp_ = this->theta_;
#endif
	// iterate through the flips
	for (uint i = 0; i < nFlips; ++i)
	{
		auto flipPlace	= *(fP.begin() + i);
		auto flipVal	= *(fV.begin() + i);
		// set the first value of b_visible
#ifdef SPIN
		currVal		= -2.0 * flipVal;
#else
		currVal		= 1.0 - 2.0 * flipVal;
#endif
		// !TODO speed this up by not creating thetaTMP
#if defined NQS_USE_CPU && not defined NQS_USE_OMP && not defined _DEBUG
		this->thetaTmp_[thId]	+= currVal * this->W_.col(flipPlace);
#else
		thetaTmp_				+= currVal * this->W_.col(flipPlace);
#endif
		val						+= currVal * this->bV_(flipPlace);
	}
	// use value as the change already
#ifdef NQS_ANGLES_UPD
#	if defined NQS_USE_CPU && not defined NQS_USE_OMP && not defined _DEBUG
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

#	endif
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FERMIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//////////////////////////////////////////////////////////////////////////////////////////
#	ifndef RBM_4_H
#		define RBM_4_H

#	endif
//////////////////////////////////////////////////////////////////////////////////////////

// ##########################################################################################################################################
// ##########################################################################################################################################
// ##########################################################################################################################################
// ##########################################################################################################################################


#endif