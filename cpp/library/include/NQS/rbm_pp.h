#pragma once
#ifndef RBMPP_H
#define RBMPP_H

/*
* RBM - PP wave function ansatz for NN quantum states.
* See the:
* - Journal of the Physical Society of Japan Vol. 77, No. 11, November, 2008, 114701
* - https://doi.org/10.1103/PhysRevB.96.205152
* - https://doi.org/10.1103/PhysRevX.11.031034
* for details. 
* 
* Copyright:
* Maksymilian Kliczkowski, Wroclaw University of Science and Technology
* 2024
*/

#ifndef RBM_H
	#include "rbm.h"
#endif // !NQS_H

// ############################## D E F I N I T I O N S   F O R   R B M - P P ##############################

/*
* @brief Restricted Boltzmann Machines ansatz with Pair Product reference state for NQS
*/
template <typename _Ht, 
			uint _spinModes		= 2,
			typename _T				= std::complex<double>,
			class _stateType		= double>
class RBM_PP_S : public RBM_S<_Ht, _spinModes, _T, _stateType>
{
public:
	using NQSS						=						arma::Col<double>;
	using NQSB						=						arma::Col<_T>;
	using NQSW						=						arma::Mat<_T>;
protected:
	// architecture parameters
	uint nPP_						=						1;
	/* ------------------------------------------------------------ */
	// ------------------------ W E I G H T S ------------------------
	NQSW Pfaffian_;

	/* ------------------------------------------------------------ */
protected:
	// ----------------------- S T A R T E R S -----------------------
	void setInfo()														override final;
	void allocate()													override final;

	// ------------------------ S E T T E R S ------------------------

	/* ------------------------------------------------------------ */
	// -------------------- P R O B A B I L I T Y --------------------

	auto pRatio(uint fP, float fV)			-> _T				override final;
	auto pRatio(uint nFlips)					-> _T				override final;
	auto pRatio(const NQSS& _v1,
					const NQSS& _v2)				-> _T				override final;
	auto pRatio(std::initializer_list<int> fP,
		std::initializer_list<double> fV)	-> _T				override final;
	// ------------------------ W E I G H T S ------------------------
public:
	bool setWeights(std::string _path, std::string _file)	override final;
	bool saveWeights(std::string _path, std::string _file)override final;
protected:
	void updateWeights()												override final;

#ifdef NQS_ANGLES_UPD
	void update(uint nFlips)										override final;
	void update(const NQSS& v, uint nFlips)					override final;
#endif

	// ------------------------- T R A I N ------------------------------	
	void grad(const NQSS& _v, uint _plc)						override final;

	// -------------------------------------------------------------------
public:
	~RBM_PP_S()
	{
		LOGINFO(this->info_ + " - destructor called.", LOG_TYPES::INFO, 4);
	};
	RBM_PP_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum = 1)
		: NQS<_Ht, _spinModes, _T, _stateType>(_H, _lr, _threadNum), nHid_(_nHid)
	{
		//this->fullSize_ = this->nHid_ + this->nVis_ + this->nHid_ * this->nVis_;
		this->allocate();
		this->setInfo();
		this->init();
		this->setRandomState(true);
	}

	// --------------------- S E T T E R S ---------------------
	void init()							override final;
	
	// --------------------- G E T T E R S ---------------------
	auto getNPP()						const -> uint			{ return this->nPP_; };
	
	// --------------------- F I N A L E -----------------------
	auto ansatz(const NQSS& _in)	const->_T				override final;
};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################## A N S A T Z ###############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief reproduces the RBM-PP NQS ANSATZ.
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
_T RBM_PP_S<_Ht, _spinModes, _T, _stateType>::ansatz(const NQSS& _in) const
{
	//!TODO
	return (std::exp(arma::dot(this->bV_, _in)) * arma::prod(this->coshF(_in))) / std::sqrt(this->nVis_); //* std::pow(2.0, this->n_hidden)
};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ######################################################## I N I T I A L I Z E R S #########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Allocate the memory for the NQS parameters. 
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void RBM_PP_S<_Ht, _spinModes, _T, _stateType>::allocate()
{
	// allocate weights
	//!TODO
	// allocate the rest
	RBM_S<_Ht, _spinModes, _T, _stateType>::allocate();
}

// ##############################################################################################################################################

/*
* @brief Initializes the weights and biases for the RBM-PP NQS model
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void RBM_PP_S<_Ht, _spinModes, _T, _stateType>::init()
{

	// !TODO
	RBM_S<_Ht, _spinModes, _T, _stateType>::init();
}

// ##############################################################################################################################################

/*
* @brief Sets the info about the RBM-PP model NQS.
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void RBM_PP_S<_Ht, _spinModes, _T, _stateType>::setInfo()
{
	this->info_ = "";
	strSeparatedS(this->info_, ',', "RBMPP", 
		VEQV(mod, this->H_->getType()), 
		VEQV(nv, this->nVis_), 
		VEQV(nh, this->nHid_), 
		VEQV(npp, this->nPP_), 
		VEQVS(lr, this->lr_));
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ######################################################### P R O B A B I L I T Y ##########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

// %%%%%%%%%%%%%%%%%%% S I N G L E   F L I P %%%%%%%%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use only one flip
* @param fP place where the flip has been applied
* @param fV value before the flip!
* @returns probability ratio for a given ansatz based on the current state
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline _T RBM_PP_S<_Ht, _spinModes, _T, _stateType>::pRatio(uint fP, float fV)
{
	//!TODO
	return _T(0);
}

// %%%%%%%%%%%%%%%% M U L T I P L E   F L I P S %%%%%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use multiple flips.
* Uses the flips stored within the NQS class (flipPlaces_, flipVals_)
* @param nFlips number of flips to be used
* @returns probability ratio for a given ansatz based on the current state
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline _T RBM_PP_S<_Ht, _spinModes, _T, _stateType>::pRatio(uint nFlips)  
{
	//!TODO
	return _T(0);
}

// %%%%%%%%%%%%%%%%% U S I N G   V E C T O R S %%%%%%%%%%%%%%%%%

/*
* @brief computes (Psi'/Psi), where (Psi') is the state with certain positions flipped. 
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline _T RBM_PP_S<_Ht, _spinModes, _T, _stateType>::pRatio(const NQSS& _v1, const NQSS& _v2)
{
	//!TODO
	return _T(0);
}

// %%%%%%%%%%%%% U S I N G   I N I T I A L I Z E R %%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use multiple flips ariving from the external vectors.
* Uses the flips stored within the NQS class (fP, fV)
* @param fP flip places to be used
* @param fV flip values to be used
* @returns probability ratio for a given ansatz based on the current state
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline _T RBM_PP_S<_Ht, _spinModes, _T, _stateType>::pRatio(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
	//!TODO
	return _T(0);
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# S E T T E R S ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

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
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline bool RBM_PP_S<_Ht, _spinModes, _T, _stateType>::setWeights(std::string _path, std::string _file)
{
	//!TODO
	BEGIN_CATCH_HANDLER
	{
		// set the forces vector for the weights
		if(!NQS<_Ht, _spinModes, _T, _stateType>::setWeights(_path, _file))
			return false;
		this->bV_	= this->F_.subvec(0, this->nVis_ - 1);
		this->bH_	= this->F_.subvec(this->nVis_, this->nVis_ + this->nHid_ - 1);
		this->W_		= arma::reshape(this->F_.subvec(this->nVis_ + this->nHid_, this->fullSize_ - 1), this->W_.n_rows, this->W_.n_cols);
	}
	END_CATCH_HANDLER("Couldn't set the weights for the RBM NQS...", return false);
	return true;
}

/*
* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @returns whether the load has been successful
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline bool RBM_PP_S<_Ht, _spinModes, _T, _stateType>::saveWeights(std::string _path, std::string _file)
{
	BEGIN_CATCH_HANDLER
	{
		this->F_.subvec(0, this->nVis_ - 1) = this->bV_;
		this->F_.subvec(this->nVis_, this->nVis_ + this->nHid_ - 1) = this->bH_;
		this->F_.subvec(this->nVis_ + this->nHid_, this->fullSize_ - 1) = this->W_.as_col();
		
		// set the forces vector for the weights
		if(!NQS<_Ht, _spinModes, _T, _stateType>::saveWeights(_path, _file))
			return false;
	}
	END_CATCH_HANDLER("Couldn't save the weights for the RBM NQS...", return false);
	return true;
}

// ##########################################################################################################################################

/*
* @brief Updates the weights in the system according to a given gradient
* @warning uses forces vector (member of NQS : F_) to update the gradients - preallocation for optimization
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void RBM_PP_S<_Ht, _spinModes, _T, _stateType>::updateWeights()
{
	this->bV_ -= this->F_.subvec(0, this->nVis_ - 1);
	//#pragma omp parallel for num_threads(this->threadNum_)
	//	for (int i = 0; i < this->nVis_; i++)
	//		this->bV_(i) -= this->F_(i);

	this->bH_ -= this->F_.subvec(this->nVis_, this->nVis_ + this->nHid_ - 1);
	//#pragma omp parallel for num_threads(this->threadNum_)
	//	for (int i = 0; i < this->nHid_; i++)
	//		this->bH_(i) -= this->F_(i + this->nVis_);
	this->W_ -= arma::reshape(this->F_.subvec(this->nVis_ + this->nHid_, this->fullSize_ - 1), this->W_.n_rows, this->W_.n_cols);
	//#pragma omp parallel for num_threads(this->threadNum_)
	//	for (int i = 0; i < this->nHid_; i++)
	//		for (auto j = 0; j < this->nVis_; j++)
	//			this->W_(i, j) -= this->F_((this->nVis_ + this->nHid_) + i + j * this->nHid_);

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
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void RBM_PP_S<_Ht, _spinModes, _T, _stateType>::grad(const NQSS& _v, uint _plc)
{
	// update the angles if it is necessary
#ifndef NQS_ANGLES_UPD
	this->setTheta(_v);
#endif

	// calculate the flattened part
	this->derivatives_.submat(_plc, 0, _plc, this->nVis_ - 1) = arma::conv_to<arma::Row<_T>>::from(_v);

//#pragma omp parallel for num_threads(this->threadNum_)
//	for (int i = 0; i < this->nVis_; i++)
//		this->Derivatives_(_plc, i) = _v(i);
	this->derivatives_.submat(_plc, this->nVis_, _plc, this->nVis_ + this->nHid_ - 1) = arma::tanh(this->theta_).as_row();
//#pragma omp parallel for num_threads(this->threadNum_)
//	for (int i = 0; i < this->nHid_; i++)
//		this->Derivatives_(_plc, i + this->nVis_) = std::tanh(this->theta_(i));

//#pragma omp parallel for num_threads(this->threadNum_)
	for (int i = 0; i < this->nHid_; i++)
		for (auto j = 0; j < this->nVis_; j++)
			this->derivatives_(_plc, (this->nVis_ + this->nHid_) + i + j * this->nHid_) = this->derivatives_(_plc, i + this->nVis_) * _v(j);
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ U P D A T E R S #############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

#ifdef NQS_ANGLES_UPD
/*
* @brief Update angles with the flipped spin (spins)
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void RBM_PP_S<_Ht, _spinModes, _T, _stateType>::update(uint nFlips)
{
	for (uint i = 0; i < nFlips; ++i){
#ifdef SPIN
		this->theta_	-=	(2.0 * this->flipVals_[i]) * this->W_.col(this->flipPlaces_[i]);
#else
		this->theta_	+=	(1.0 - 2.0 * this->flipVals_[i]) * this->W.col(flipPlaces_[i]);
#endif
	}
	this->thetaCOSH_	=	this->coshF();
}

/*
* @brief Update angles with the flipped spin (spins)
* @param vector after the flips has been done
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void RBM_PP_S<_Ht, _spinModes, _T, _stateType>::update(const NQSS& v, uint nFlips)
{
	for (uint i = 0; i < this->flipPlaces_.size(); i++){
#ifdef SPIN
		this->theta_	-=	(2.0 * v[this->flipPlaces_[i]]) * this->W_.col(this->flipPlaces_[i]);
#else
		this->theta_	+=	(1.0 - 2.0 * v[this->flipPlaces_[i]]) * this->W.col(flipPlaces_[i]);
#endif
	}
	this->thetaCOSH_	=	this->coshF();

}
#endif

#endif