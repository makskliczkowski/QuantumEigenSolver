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
	u64 rbmSize_					=						1;

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
#if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP 
	// create the map for thetas for a given thread
	std::map<std::thread::id, NQSB> thetaTmp_;
#else
	NQSB thetaTmp_;
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
	this->nHid_ 			= _nHid;
	this->rbmSize_  		= this->nHid_ + this->info_p_.nVis_ + this->nHid_ * this->info_p_.nVis_;
	this->info_p_.fullSize_ = this->rbmSize_;
	this->allocate();
	this->setInfo();
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
	return (std::exp(arma::dot(this->bV_, _in)) * arma::prod(this->coshF(_in))) / std::sqrt(this->info_p_.nVis_);
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
	this->bV_.resize(this->info_p_.nVis_);
	this->bH_.resize(this->nHid_);
	this->theta_.resize(this->nHid_);
	this->thetaCOSH_.resize(this->nHid_);
	this->W_.resize(this->nHid_, this->info_p_.nVis_);
	// create thread map
#if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP
	// allocate the vector for using it in the RBM
	for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++)
		this->thetaTmp_[this->threads_.threads_[_thread].get_id()] = NQSB(this->nHid_);
#else
	this->thetaTmp_ = NQSB(this->nHid_);
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
#pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
	for (int i = 0; i < this->info_p_.nVis_; i++)
		this->bV_(i) = algebra::cast<_T>(0.05 * (this->ran_.template random<double>(-1.0, 1.0) + I * this->ran_.template randomNormal<double>(-1.0, 1.0)));
	// initialize biases hidden
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
	for (int i = 0; i < this->nHid_; i++)
		this->bH_(i) = algebra::cast<_T>(0.05 * (this->ran_.template random<double>(-1.0, 1.0) + I * this->ran_.template randomNormal<double>(-1.0, 1.0)));
	// weights matrix
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
	for (int i = 0; i < this->W_.n_rows; i++)
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
		VEQV(nv, this->info_p_.nVis_), 
		VEQV(nh, this->nHid_), 
		VEQV(nS, this->spinModes_),
		VEQVS(lr, this->info_p_.nVis_));
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
		this->bV_	= this->F_.subvec(0, this->info_p_.nVis_ - 1);
		this->bH_	= this->F_.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1);
		this->W_	= arma::reshape(this->F_.subvec(this->info_p_.nVis_ + this->nHid_, this->info_p_.nVis_ + this->nHid_ + this->W_.n_rows * this->W_.n_cols - 1),
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
		this->F_.subvec(0, this->info_p_.nVis_ - 1) = this->bV_;
		this->F_.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1) = this->bH_;
		this->F_.subvec(this->info_p_.nVis_ + this->nHid_, this->rbmSize_ - 1) = this->W_.as_col();
		
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
	this->bV_	-= this->F_.subvec(0, this->info_p_.nVis_ - 1);
	this->bH_	-= this->F_.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1);
	this->W_	-= arma::reshape(this->F_.subvec(this->info_p_.nVis_ + this->nHid_, this->rbmSize_ - 1),
								this->W_.n_rows, this->W_.n_cols);
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
	auto _currDerivative	= this->derivatives_.row(_plc);
	// update the angles if it is necessary
#ifndef NQS_ANGLES_UPD
	this->setTheta(_v);
#endif

	// calculate the flattened part
	_currDerivative.subvec(0, this->info_p_.nVis_ - 1) = arma::conv_to<arma::Row<_T>>::from(_v);

	auto _hiddDerivative	= _currDerivative.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1);
	_hiddDerivative			= arma::tanh(this->theta_).as_row();

	auto _weightsDerivative	= _currDerivative.subvec(this->info_p_.nVis_ + this->nHid_, this->rbmSize_ - 1);

//#ifndef _DEBUG
//#	pragma omp parallel for num_threads(this->threadsNumLeft_)
//#endif
	for (int j = 0; j < this->info_p_.nVis_; ++j)
		_weightsDerivative.subvec(j * this->nHid_, (j + 1) * this->nHid_ - 1) = _v(j) * _hiddDerivative;
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
//////////////////////////////////////////////////////////////////////////////////////////

#include "./RBM_base/rbm_general.h"
#include "./RBM_base/rbm_spins.h"
#include "./RBM_base/rbm_fermions.h"

//////////////////////////////////////////////////////////////////////////////////////////

#endif