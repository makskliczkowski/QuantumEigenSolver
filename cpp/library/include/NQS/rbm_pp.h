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

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! B A S E !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

/*
* @brief Restricted Boltzmann Machines ansatz with Pair Product reference state for NQS
*/
template < 	uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class RBM_PP : public RBM_S<_spinModes, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
protected:
	// architecture parameters
	uint nPP_					= 1;
	u64 rbmPPSize_				= 1;

	/* ------------------------------------------------------------ */
	// ------------------------ W E I G H T S ------------------------
	// go through the quarters (spin sectors)
	v_2d<bool> spinSectors_;
	_T pfaffian_				= 0.0;							// store the last pfaffian value
	_T pfaffianNew_				= 0.0;							// store the last pfaffian value for the candidate
	NQSW X_;													// for stroing the matrix for Pfaffian calculation at each step
	NQSW Xinv_;													// for stroing the matrix inverse for Pfaffian calculation at each step
	NQSW Xnew_;													// for stroing the matrix for Pfaffian calculation at each step - new candidate
	
	NQSB Fpp_;													// for storing the additional variational parameters from the PP
	u64 getFIndex(bool _spini, bool _spinj, uint ri, uint rj)	const;

	/* ------------------------------------------------------------ */
protected:
	// ----------------------- S T A R T E R S -----------------------
	void setInfo()												override final;
	void allocate()												override final;

	// ------------------------ S E T T E R S ------------------------
	virtual void setState(const NQSS& _st, bool _set)			override final;
	virtual void setState(u64 _st, bool _set)					override final;

	/* ------------------------------------------------------------ */
	// -------------------- P R O B A B I L I T Y --------------------
	auto pRatio(uint fP, float fV)			-> _T				override final;
	auto pRatio(uint nFlips)				-> _T				override final;
	auto pRatio(const NQSS& _v1,
					const NQSS& _v2)		-> _T				override final;
	auto pRatio(std::initializer_list<int> fP,
				std::initializer_list<double> fV) -> _T			override final;

	// ------------------------ W E I G H T S ------------------------
public:
	bool setWeights(std::string _path, std::string _file)		override final;
	bool saveWeights(std::string _path, std::string _file)		override final;

protected:
	void updateWeights()										override final;
	// updates
#ifdef NQS_ANGLES_UPD
	void update(uint nFlips)									override final;
	void update(const NQSS& v, uint nFlips)						override final;
	void unupdate(uint nFlips)									override final;
#endif

	// ---------------------------- T R A I N ----------------------------	
	void grad(const NQSS& _v, uint _plc)						override final;

	// --------------------------- A N S A T Z ---------------------------
	virtual void updPffafianC(uint fP, float fV)				= 0;
	virtual void updPffafianC(std::initializer_list<int> fP,
							  std::initializer_list<double> fV)	= 0;
	virtual void updPffafian(uint fP, float fV)					= 0;
	virtual void updPffafian(std::initializer_list<int> fP,
							  std::initializer_list<double> fV)	= 0;
public:
	~RBM_PP() override											{ DESTRUCTOR_CALL;									};
	RBM_PP(std::shared_ptr<Hamiltonian<_Ht>> _H, uint _nHid, double _lr, uint _threadNum = 1, int _nPart = -1);

	// --------------------- S E T T E R S ---------------------
	void init()													override final;
	// for the previous pfaffian matrix
	void setPffMat()											{ this->X_ = this->getPffMat();						};
	void setPffMat(const NQSS& _n)								{ this->X_ = this->getPffMat(_n);					};
	void setPffMat(u64 _n)										{ this->X_ = this->getPffMat(_n);					};
	// for the new pfaffian matrix
	void setPffMatC()											{ this->Xnew_ = this->getPffMat();					};
	void setPffMatC(const NQSS& _n)								{ this->Xnew_ = this->getPffMat(_n);				};
	void setPffMatC(u64 _n)										{ this->Xnew_ = this->getPffMat(_n);				};
	// for the pfaffian value
	void setPffVal()											{ this->pfaffian_ = this->getPff();					};
	void setPffVal(const NQSS& _n)								{ this->setPffMat(_n); this->setPffVal();			};
	void setPffVal(const NQSW& _M)								{ this->pfaffian_ = this->getPff(_M);				};
	// for the candidate pfaffian value 
	void setPffValC()											{ this->pfaffianNew_ = this->getPffC();				};
	void setPffValC(const NQSS& _n)								{ this->setPffMatC(_n); this->setPffValC();			};
	void setPffValC(const NQSW& _M)								{ this->pfaffianNew_ = this->getPff(_M);			};
	// --------------------- G E T T E R S ---------------------
	virtual auto getPffMat(const NQSS& _n)	const -> NQSW		= 0;
#ifndef NQS_USE_VEC_ONLY
	virtual auto getPffMat(u64 _n)			const -> NQSW		= 0;
#endif
	// based on the current vector, obtain a matrix for Pffafian calcuation
	virtual auto getPffMat()				const -> NQSW		{ return this->getPffMat(this->curVec_);								};
	// get the Pffafian value for a given vector
	auto getPffV(const NQSS& _in)			const -> _T			{ return this->getPff(this->getPffMat(_in));							};
	// get size of the PP function matrix
	auto getNPP()							const -> uint		{ return this->nPP_;													};
	// based on the current matrix, obtain the Pffafian
	auto getPff()							const -> _T			{ return algebra::pfaffian<_T>(this->X_, this->nParticles_);			};
	// based on given matrix, obtain the Pffafian
	auto getPff(const NQSW& _M)				const -> _T			{ return algebra::pfaffian<_T>(_M, _M.n_rows);							};
	// based on the current candidate matrix, obtain the Pffafian
	auto getPffC()							const -> _T			{ return algebra::pfaffian<_T>(this->Xnew_, this->nParticles_);			};
	// get the current Pffafian value
	auto getPffV()							const -> _T			{ return this->pfaffian_;												};
	// get the current Pffafian candidate value
	auto getPffVC()							const -> _T			{ return this->pfaffianNew_;											};
	// --------------------- F I N A L E -----------------------
	auto ansatz(const NQSS& _in)			const -> _T			override final;
};

// ##########################################################################################################################################

// ##########################################################################################################################################
// ########################################################### C O N S T R U C T ############################################################
// ##########################################################################################################################################

// ##########################################################################################################################################

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM_PP<_spinModes, _Ht, _T, _stateType>::RBM_PP(std::shared_ptr<Hamiltonian<_Ht>> _H, uint _nHid, double _lr, uint _threadNum, int _nPart)
	: RBM_S<_spinModes, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nPart)
{
	// create the spin sectors
	spinSectors_.push_back({ 1, 1 });
	spinSectors_.push_back({ 1, 0 });
	spinSectors_.push_back({ 0, 1 });
	spinSectors_.push_back({ 0, 0 });

	// !TODO make this changable
	this->nPP_		= this->spinSectors_.size() * this->nSites_ * this->nSites_; // for both spin channels
	this->rbmPPSize_= this->rbmSize_ + this->nPP_;
	this->fullSize_ = this->rbmPPSize_;
	this->allocate();
	this->setInfo();
}

// ##########################################################################################################################################

// ##########################################################################################################################################
// ############################################################## A N S A T Z ###############################################################
// ##########################################################################################################################################

// ##########################################################################################################################################

/*
* @brief reproduces the RBM-PP NQS ANSATZ.
* @param _in the state for which the ansatz shall be calculated
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
_T RBM_PP<_spinModes, _Ht, _T, _stateType>::ansatz(const NQSS& _in) const
{
	// set the Jacobian before!
	return RBM_S<_spinModes, _Ht, _T, _stateType>::ansatz(_in) * this->getPffV(_in); //* std::pow(2.0, this->n_hidden)
};

// ##########################################################################################################################################

// ##########################################################################################################################################
// ######################################################## I N I T I A L I Z E R S #########################################################
// ##########################################################################################################################################

// ##########################################################################################################################################

/*
* @brief Based on the spin of the particle at r_i position and the particle at r_j position, returns the index of 
* the coefficient in the flattened F matrix.
* @param _spini spin of the particle at r_i position
* @param _spinj spin of the particle at r_j position
* @param ri position of the particle in the lattice
* @param rj position of the particle in the lattice
* @returns index of the coefficient in the flattened F matrix
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline u64 RBM_PP<_spinModes, _Ht, _T, _stateType>::getFIndex(bool _spini, bool _spinj, uint ri, uint rj) const
{
	if (_spini)
	{
		if (_spinj)
			return ri * this->nParticles_ + rj;
		else
			return this->nParticles_ * this->nParticles_ + ri * this->nParticles_ + rj;
	}
	else
	{
		if (_spinj)
			return 2 * this->nParticles_ * this->nParticles_ + ri * this->nParticles_ + rj;
		else
			return 3 * this->nParticles_ * this->nParticles_ + ri * this->nParticles_ + rj;
	}
}

////////////////////////////////////////////////////////////////////////

/*
* @brief Allocate the memory for the NQS parameters. 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::allocate()
{
	// !TODO implement changable number of fermions
	// allocate weights
	// Pfaffian matrix for each step
	this->X_		= NQSW(this->nParticles_, this->nParticles_, arma::fill::zeros);
	this->Xinv_		= NQSW(this->nParticles_, this->nParticles_, arma::fill::zeros);

	// allocate the weights themselves !TODO - make this symmetric? 
	// square matrix with spin changes F_{ij}^{\\sigma, \\sigma '}
	this->Fpp_		= NQSB(this->nPP_);
	// allocate the rest
	RBM_S<_spinModes, _Ht, _T, _stateType>::allocate();
}

////////////////////////////////////////////////////////////////////////

/*
* @brief Initializes the weights and biases for the RBM-PP NQS model
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::init()
{
	// matrix for the PP wave function
	this->Fpp_	= 1e-2 * NQSB(this->nPP_, arma::fill::randn);
	auto _lat	= this->H_->getLat();
	
	// go through the lattice
	for (uint i = 0; i < this->nSites_; i++)
	{
		for (uint j = 0; j < this->nSites_; j++)
		{
			// get the distance between the sites
			auto distance = _lat->getSiteDistance(i, j);
			for (const auto& spinSec: this->spinSectors_)
			{
				// make the weights proportional to the distance
				if (distance != 0)
					this->Fpp_(this->getFIndex(spinSec[0], spinSec[1], i, j)) /= distance * distance;
			}
		}
	}

	// !TODO
	RBM<_spinModes, _Ht, _T, _stateType>::init();
}

////////////////////////////////////////////////////////////////////////

/*
* @brief Sets the info about the RBM-PP model NQS.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::setInfo()
{
	this->info_ = "";
	strSeparatedS(this->info_, ',', "RBMPP", 
		VEQV(mod, this->H_->getType()), 
		VEQV(nv, this->nVis_), 
		VEQV(nh, this->nHid_), 
		VEQV(npp, this->nPP_),
		VEQV(nS, this->spinModes_),
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
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::pRatio(uint fP, float fV)
{
	// update pfaffian candidate matrix and its corresponding value
	this->updPffafianC(fP, fV);
	return RBM_S<_spinModes, _Ht, _T, _stateType>::pRatio(fP, fV) * this->pfaffianNew_ / this->pfaffian_;
}

// %%%%%%%%%%%%%%%% M U L T I P L E   F L I P S %%%%%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use multiple flips.
* Uses the flips stored within the NQS class (flipPlaces_, flipVals_)
* @param nFlips number of flips to be used
* @returns probability ratio for a given ansatz based on the current state
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
[[deprecated("Use pRatio(uint fP, float fV) instead")]]
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::pRatio(uint nFlips)
{
	// update pfaffian candidate matrix and its corresponding value
	for (auto i = 0; i < nFlips; ++i)
		this->updPffafianC(this->flipPlaces_[i], this->flipVals_[i]);
	return RBM_S<_spinModes, _Ht, _T, _stateType>::pRatio(nFlips) * this->pfaffianNew_ / this->pfaffian_;
}

// %%%%%%%%%%%%%%%%% U S I N G   V E C T O R S %%%%%%%%%%%%%%%%%

/*
* @brief computes (Psi'/Psi), where (Psi') is the state with certain positions flipped. 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::pRatio(const NQSS& _v1, const NQSS& _v2)
{
	return RBM_S<_spinModes, _Ht, _T, _stateType>::pRatio(_v1, _v2) * this->getPffV(_v2) / this->getPffV(_v1);
}

// %%%%%%%%%%%%% U S I N G   I N I T I A L I Z E R %%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use multiple flips ariving from the external vectors.
* Uses the flips stored within the NQS class (fP, fV)
* @param fP flip places to be used
* @param fV flip values to be used
* @returns probability ratio for a given ansatz based on the current state
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::pRatio(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
	this->updPffafianC(fP, fV);
	return RBM_S<_spinModes, _Ht, _T, _stateType>::pRatio(fP, fV) * this->pfaffianNew_ / this->pfaffian_;
}

// ##########################################################################################################################################

// ##########################################################################################################################################
// ############################################################# S E T T E R S ##############################################################
// ##########################################################################################################################################

// ##########################################################################################################################################

/*
* @brief Sets the state and the corresponding PP state as well.
* Updates the pfaffian matrix.
* @param _st column state to be set
* @param _set set the matrices?
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::setState(const NQSS& _st, bool _set)
{
	RBM_S<_spinModes, _Ht, _T, _stateType>::setState(_st, _set);
#ifdef NQS_ANGLES_UPD
	if (_set)
	{
		// set pfaffian matrix and its corresponding value
		this->setPffMat();
		// set pfaffian value for newly set matrix
		this->setPffVal();
		// set candidate to be the same for the angles update
		this->Xnew_ = this->X_;
		// set its value to be the same as well
		this->pfaffianNew_ = this->pfaffian_;
	}
#endif
}

////////////////////////////////////////////////////////////////////////

/*
* @brief Sets the state and the corresponding PP state as well.
* Updates the pfaffian matrix.
* @param _st integer state to be set
* @param _set set the matrices?
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::setState(u64 _st, bool _set)
{
	RBM_S<_spinModes, _Ht, _T, _stateType>::setState(_st, _set);
#ifdef NQS_ANGLES_UPD
	if (_set)
	{
		// set pfaffian matrix and its corresponding value
		this->setPffMat();
		// set pfaffian value for newly set matrix
		this->setPffVal();
		// set candidate to be the same for the angles update
		this->Xnew_			= this->X_;
		// set its value to be the same as well
		this->pfaffianNew_	= this->pfaffian_;
	}
#endif
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
inline bool RBM_PP<_spinModes, _Ht, _T, _stateType>::setWeights(std::string _path, std::string _file)
{
	// set for RBM
	if(!RBM_S<_spinModes, _Ht, _T, _stateType>::setWeights(_path, _file))
		return false;

	BEGIN_CATCH_HANDLER
	{
		// Fmat is a vector
		this->Fpp_ = this->F_.subvec(this->rbmSize_, this->rbmPPSize_ - 1);
	}
	END_CATCH_HANDLER("Couldn't set the weights for the RBM PP NQS...", return false);
	return true;
}

////////////////////////////////////////////////////////////////////////

/*
* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @link https://arxiv.org/pdf/1102.3440.pdf - for arxiv reference
* @link https://math.stackexchange.com/questions/4426574/pfaffian-skew-symmetric-using-armadillo - for stackexchange
* @returns whether the load has been successful
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool RBM_PP<_spinModes, _Ht, _T, _stateType>::saveWeights(std::string _path, std::string _file)
{
	BEGIN_CATCH_HANDLER
	{
		// Fmat is a vector
		this->F_.subvec(this->rbmSize_, this->rbmPPSize_ - 1) = this->Fpp_;
		// set the forces vector for the weights
		if(!RBM_S<_spinModes, _Ht, _T, _stateType>::saveWeights(_path, _file))
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
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::updateWeights()
{
	// update RBM weights
	RBM_S<_spinModes, _Ht, _T, _stateType>::updateWeights();
	// Fmat is a vector
	this->Fpp_ -= this->F_.subvec(this->rbmSize_, this->rbmPPSize_ - 1);
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
* !TODO - not half filling / not spins
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::grad(const NQSS& _v, uint _plc)
{
	// calculate the RBM derivative
	RBM_S<_spinModes, _Ht, _T, _stateType>::grad(_v, _plc);
	// calculate the derivative of the Pffafian 
	// as 1/Pf(X) * \partial Pf(X) / \partial x = 1/2 Tr(X^{-1} \partial X / \partial x)
	arma::Mat<_T> Xinv	= arma::inv(this->X_);
	Xinv				= ((Xinv.t() - Xinv) * 0.5) * this->pfaffian_;

	// find the correct indices on the values that can be updated
	this->derivatives_.submat(_plc, this->rbmSize_, _plc, this->rbmPPSize_ - 1).zeros();

	int _spinIter = 0;
	for (const auto& s: this->spinSectors_)
	{
		for (uint i = 0; i < this->nParticles_; ++i)
		{
			// if left spin differs from the current left spin, just continue
			if (checkBit(this->curVec_, i) != s[0])
				continue;

			for (uint j = 0; j < this->nParticles_; ++j)
			{

				// if right spin differs from the current right spin, just continue
				if (checkBit(this->curVec_, j) != s[1])
					continue;

				// skip the diagonal part
				if (j == i)
					continue;

				this->derivatives_(_plc, this->getFIndex(s[0], s[1], i, j)) = Xinv(i, j);
			}
		}
		_spinIter += this->nParticles_ * this->nParticles_;
	}
}

// ##########################################################################################################################################

// ##########################################################################################################################################
// ############################################################ U P D A T E R S #############################################################
// ##########################################################################################################################################

// ##########################################################################################################################################

////////////////////////////////////////////////////////////////////////
#	ifdef NQS_ANGLES_UPD

/*
* @brief Update angles with the flipped spin (spins)
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::update(uint nFlips)
{
	// update the angles in the RBM
	RBM_S<_spinModes, _Ht, _T, _stateType>::update(nFlips);
	// update the Pffafian matrix
	// as the candidate pfaffian shall be already updated, use it instead of calculating everything all the time (probably not as efficient)
	// replace updating the pfaffian back
	this->X_		= this->Xnew_;
	this->pfaffian_ = this->pfaffianNew_;
}

////////////////////////////////////////////////////////////////////////

/*
* @brief If a new pfaffian state has not been accepted, unupdate the values
* @param nFlips number of flips to be used
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::unupdate(uint nFlips)
{
	// unupdate the Pffafian matrix
	// as the candidate pfaffian shall be already updated, use it instead of calculating everything all the time (probably not as efficient)
	// replace updating the pfaffian back
	this->Xnew_			= this->X_;
	this->pfaffianNew_	= this->pfaffian_;
}

////////////////////////////////////////////////////////////////////////

/*
* @brief Update angles with the flipped spin (spins)
* @param vector after the flips has been done
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::update(const NQSS& v, uint nFlips)
{
	// update the angles in the RBM
	RBM_S<_spinModes, _Ht, _T, _stateType>::update(v, nFlips);
	// update the Pffafian matrix
	// as the candidate pfaffian shall be already updated, use it instead of calculating everything all the time (probably not as efficient)
	// replace updating the pfaffian back
	this->X_		= this->Xnew_;
	this->pfaffian_ = this->pfaffianNew_;
}
#	endif

//////////////////////////////////////////////////////////////////////////////////////////

// #######################################################################################
// #######################################################################################
// #######################################################################################
// #######################################################################################

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GENERAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////
#	ifndef RBM_PP_GEN_H
#		define RBM_PP_GEN_H
template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class RBM_PP_S : public RBM_PP<_spinModes, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);

	RBM_PP_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum = 1, int _nParticles = -1)
		: RBM_PP<_spinModes, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles)
																	 { NQS_LOG_ERROR_SPIN_MODES; };

	/* ------------------------------------------------------- */
	// --------------------- G E T T E R S ---------------------
	virtual auto getPffMat(const NQSS& _n)	const -> NQSW	override { NQS_LOG_ERROR_SPIN_MODES; return NQSW(); };
#ifndef NQS_USE_VEC_ONLY
	virtual auto getPffMat(u64 _n)			const -> NQSW	override { NQS_LOG_ERROR_SPIN_MODES; return NQSW(); };
#endif
protected:
	// --------------------------- A N S A T Z ---------------------------
	void updPffafianC(uint fP, float fV)					override { NQS_LOG_ERROR_SPIN_MODES; };
	void updPffafianC(std::initializer_list<int> fP,
					 std::initializer_list<double> fV)		override { NQS_LOG_ERROR_SPIN_MODES; };
	void updPffafian(uint fP, float fV)						override { NQS_LOG_ERROR_SPIN_MODES; };
	void updPffafian(std::initializer_list<int> fP,
					 std::initializer_list<double> fV)		override { NQS_LOG_ERROR_SPIN_MODES; };
};
#	endif
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SPINS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////
#	ifndef RBM_PP_2_H
#		define RBM_PP_2_H

/*
* @brief Restricted Boltzmann Machines ansatz with Pair Product reference state for NQS
*/
template <typename _Ht, typename _T, class _stateType>
class RBM_PP_S<2, _Ht, _T, _stateType> : public RBM_PP<2, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
	/* ------------------------------------------------------- */
public:
	RBM_PP_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum = 1, int _nParticles = -1)
		: RBM_PP<2, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles) 
	{
		this->init();
		// sets the thetas and Pfaffian as well
		this->setRandomState(true);
	};

	/* ------------------------------------------------------- */
	// --------------------- G E T T E R S ---------------------
	virtual auto getPffMat(const NQSS& _n)	const -> NQSW	override;
#ifndef NQS_USE_VEC_ONLY
	virtual auto getPffMat(u64 _n)			const -> NQSW	override;
#endif
protected:
	// --------------------------- A N S A T Z -----------------
	void updPffafianC(uint fP, float fV)					override;
	void updPffafianC(std::initializer_list<int> fP,
					 std::initializer_list<double> fV)		override;
	void updPffafian(uint fP, float fV)						override;
	void updPffafian(std::initializer_list<int> fP,
					 std::initializer_list<double> fV)		override;
};

// !!!!!!!!!!!!!!!! P F F A F I A N   S T A T E !!!!!!!!!!!!!!!!

/*
* @brief Allows one to find the new Pfaffian matrix to calculate the overlap <x|\\phi_ref> in the PP wave function.
* This is done as (N/2)! Pf[X] for the standard
* |\\phi _ref> = (\\sum _{i,j} \\sum _{\\sigma, \\sigma'} F_{ij}^{\\sigma,\\sigma'} c_{i\\sigma}^\\dag c_{j\\sigma}^\\dag
* One seeks for the position of spins and obtains X_ij = F_{ri,rj}^{\\sigma_i, \\sigma_j} - F_{rj,ri}^{\\sigma_j, \\sigma_i}.
* For now we calculate this as a double loop
* This is for the spins and in this loop it just checks the spin ad hoc, as half filling is imposed.
[!TODO optimize].
*
* @param state defined as a column vector of ones and zeros:
*	for fermions one has vector of #Nsite spin ups and then #Nsite spin downs
*	for spins one has only vector of #Nsite spins that can be transformed to Abrikosov fermions
* @returns the Pfaffian matrix
*/
template <typename _Ht, typename _T, class _stateType>
typename RBM_PP_S<2, _Ht, _T, _stateType>::NQSW RBM_PP_S<2, _Ht, _T, _stateType>::getPffMat(const NQSS& _n) const
{
	// remember that FF is such that is starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN})
	NQSW _out(this->nParticles_, this->nParticles_, arma::fill::zeros);

	// go through ri, aka, the ith particle site
	for (uint i = 0; i < this->nSites_; ++i)
	{
		// check the spin at a given position i
		bool spin_ri = checkBit(_n, i);

		// go through rj, aka, the jth particle site
		for (uint j = i + 1; j < this->nSites_; ++j)
		{
			// check the spin at a given position
			bool spin_rj =	checkBit(_n, j);
			{
				// F_{ri,rj}^{\\sigma_i, \\sigma_j} - find the index corresponding to those particles in F
				// F_{rj,ri}^{\\sigma_j, \\sigma_i} - find the index corresponding to those particles in F
				// get the Pffafian please X_ij = F_{ri,rj}^{\\sigma_i, \\sigma_j} - F_{rj,ri}^{\\sigma_j, \\sigma_i}
				_out(i, j) = this->Fpp_(this->getFIndex(spin_ri, spin_rj, i, j)) - this->Fpp_(this->getFIndex(spin_rj, spin_ri, j, i));
				_out(j, i) = -_out(i, j);
			}
		}
	}
	return _out;
}

////////////////////////////////////////////////////////////////

#ifndef NQS_USE_VEC_ONLY
template <typename _Ht, typename _T, class _stateType>
typename RBM_PP_S<2, _Ht, _T, _stateType>::NQSW RBM_PP_S<2, _Ht, _T, _stateType>::getPffMat(u64 _n) const
{
	NQSW _out(this->nParticles_, this->nParticles_, arma::fill::zeros);
	// go through ri
	for (uint i = 0; i < this->nSites_; ++i)
	{
		// check the spin at a given position
		bool spinI = checkBit(_n, i);
		for (uint j = 0; j < this->nSites_; ++j)
		{
			// check the spin at a given position
			bool spinJ		=	checkBit(_n, j);
			{
				// F_{ri,rj}^{\\sigma_i, \\sigma_j}
				uint positionLeft	=	(spinI ? 0 : 2 * this->nSites_ * this->nSites_) + 
										(spinJ ? 0 : this->nSites_ * this->nSites_)		+ 
										i * this->nSites_ + j;
				// F_{rj,ri}^{\\sigma_j, \\sigma_i}
				uint positionRight	=	(spinI ? 2 * this->nSites_ * this->nSites_ : 0) + 
										(spinJ ? this->nSites_ * this->nSites_ : 0)		+ 
										j * this->nSites_ + i;
				// get the Pffafian please
				_out(i, j) = this->Fpp_(positionLeft) - this->Fpp_(positionRight);
			}
		}
	}
	return _out;
}
#endif

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// !!!!!!!!!!!! P F F A F I A N   C A N D I D A T E !!!!!!!!!!!!

/*
* @brief Updates the pfaffian candidate according to a single flip
* @param fP place of the flip in the current vector
* @param fV value before the flip at fP
*/
template <typename _Ht, typename _T, class _stateType>
void RBM_PP_S<2, _Ht, _T, _stateType>::updPffafianC(uint fP, float fV)
{
	// remember that FF is such that is starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN} = {I, II} , {III, IV})
	// if the fP is known, only the column and row containing it shall be updated
	// we will use current state as well to see where the other places are
	// remember, fV is the other way around (so it's before the flip)
	// new k, when particle on fP has been set for F_{ki}^{\sigma_k', \sigma _i}
	// run over the columns
	for (uint i = 0; i < this->nParticles_; ++i)
	{
		// the same is not allowed (no double occupations)
		if (i == fP)
			continue;

		// check the bit on the i'th place to know in which place you'll end up
		bool spin_next	= checkBit(this->curVec_, i);

		// F_{ri,rj}^{\\sigma_i, \\sigma_j} - find the index corresponding to those particles in F
		auto posLeft	= this->getFIndex(fV < 0, spin_next, fP, i);
		// F_{rj,ri}^{\\sigma_j, \\sigma_i} - find the index corresponding to those particles in F
		auto posRight	= this->getFIndex(spin_next, fV < 0, i, fP);

		// update 
		this->Xnew_(fP, i) = this->Fpp_(posLeft) - this->Fpp_(posRight);
		this->Xnew_(i, fP) = -this->Xnew_(fP, i);
	}
	// set the value
	this->setPffValC();
}

////////////////////////////////////////////////////////////////

/*
* @brief Updates the pfaffian according to a single flip
* @param fP place of the flip in the current vector
* @param fV value before the flip at fP
*/
template <typename _Ht, typename _T, class _stateType>
void RBM_PP_S<2, _Ht, _T, _stateType>::updPffafian(uint fP, float fV)
{
	// remember that FF is such that is starts with an up spin ({UP^UP, UP^DOWN}, {DOWN^UP, DOWN^DOWN} = {I, II} , {III, IV})
	// if the fP is known, only the column and row containing it shall be updated
	// we will use current state as well to see where the other places are
	// remember, fV is the other way around (so it's before the flip)
	// new k, when particle on fP has been set for F_{ki}^{\sigma_k', \sigma _i}
	// run over the columns
	for (uint i = 0; i < this->nParticles_; ++i)
	{
		// the same is not allowed (no double occupations)
		if (i == fP)
			continue;

		// check the bit on the i'th place to know in which place you'll end up
		bool spin_next	= checkBit(this->curVec_, i);

		// F_{ri,rj}^{\\sigma_i, \\sigma_j} - find the index corresponding to those particles in F
		auto posLeft	= this->getFIndex(fV < 0, spin_next, fP, i);
		// F_{rj,ri}^{\\sigma_j, \\sigma_i} - find the index corresponding to those particles in F
		auto posRight	= this->getFIndex(spin_next, fV < 0, i, fP);

		// update 
		this->X_(fP, i) = this->Fpp_(posLeft) - this->Fpp_(posRight);
		this->X_(i, fP) = -this->Xnew_(fP, i);
	}
	// set the value
	this->setPffVal();
}

////////////////////////////////////////////////////////////////

template <typename _Ht, typename _T, class _stateType>
void RBM_PP_S<2, _Ht, _T, _stateType>::updPffafianC(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
	// if the fP is known, only the column and row containing it shall be updated
	// we will use current state as well to see where the other places are
//	// remember, fV is the other way around
//	uint ns2 = this->nSites_ * this->nSites_;
//	// save the minimum of both sizes as new flip size if someone makes wrong movement
//#ifdef _DEBUG
//	size_t nFlips = std::min(fP.size(), fV.size());
//#else
//	size_t nFlips = fP.size();
//#endif
//	// go through the flips
//	for (uint k = 0; k < nFlips; ++k)
//	{
//		bool spink = *(fV.begin() + k) > 0;
//		bool kP = *(fP.begin() + k);
//		uint kFL = (spink < 0) ? kP : kP + 2 * ns2;
//		uint kFR = (spink < 0) ? kP : kP + ns2;
//		// run over the columns as the row is set
//		for (uint i = 0; i < this->nParticles_; ++i)
//		{
//			// the same is not allowed
//			if (i == kP)
//				continue;
//			// check the bit on the i'th place
//			bool spini = checkBit(this->curVec_, i);
//			// set the left place in the weights (column)
//			uint iFL = spini ? i : i + ns2;
//			uint iFR = spini ? i + 2 * ns2 : i;
//			// update 
//			this->Xnew_(kP, i) = this->Fmat_(kFL + iFL) - this->Fmat_(kFR + iFR);
//			this->Xnew_(i, kP) = -this->Xnew_(kP, i);
//		}
//	}
//	// set the value
//	this->setPffValC();
}

////////////////////////////////////////////////////////////////

template <typename _Ht, typename _T, class _stateType>
void RBM_PP_S<2, _Ht, _T, _stateType>::updPffafian(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
	// if the fP is known, only the column and row containing it shall be updated
	// we will use current state as well to see where the other places are
	// remember, fV is the other way around
//	uint ns2 = this->nSites_ * this->nSites_;
//	// save the minimum of both sizes as new flip size if someone makes wrong movement
//#ifdef _DEBUG
//	size_t nFlips = std::min(fP.size(), fV.size());
//#else
//	size_t nFlips = fP.size();
//#endif
//	// go through the flips
//	for (uint k = 0; k < nFlips; ++k)
//	{
//		bool spink = *(fV.begin() + k) > 0;
//		bool kP = *(fP.begin() + k);
//		uint kFL = (spink < 0) ? kP : kP + 2 * ns2;
//		uint kFR = (spink < 0) ? kP : kP + ns2;
//		// run over the columns as the row is set
//		for (uint i = 0; i < this->nParticles_; ++i)
//		{
//			// the same is not allowed
//			if (i == kP)
//				continue;
//			// check the bit on the i'th place
//			bool spini = checkBit(this->curVec_, i);
//			// set the left place in the weights (column)
//			uint iFL = spini ? i : i + ns2;
//			uint iFR = spini ? i + 2 * ns2 : i;
//			// update 
//			this->X_(kP, i) = this->Fmat_(kFL + iFL) - this->Fmat_(kFR + iFR);
//			this->X_(i, kP) = -this->X_(kP, i);
//		}
//	}
}

////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#	endif
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FERMIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////
#	ifndef RBM_PP_4_H
#		define RBM_PP_4_H

///*
//* @brief Allows one to find the new Pfaffian matrix to calculate the overlap <x|\\phi_ref> in the PP wave function.
//* This is done as (N/2)! Pf[X] for the standard
//* |\\phi _ref> = (\\sum _{i,j} \\sum _{\\sigma, \\sigma'} F_{ij}^{\\sigma,\\sigma'} c_{i\\sigma}^\\dag c_{j\\sigma}^\\dag
//* One seeks for the position of spins and obtains X_ij = F_{ri,rj}^{\\sigma_i, \\sigma_j} - F_{rj,ri}^{\\sigma_j, \\sigma_i}.
//* For now we calculate this as a double loop
//* This is for the Fermions and in this loop it just checks the spin ad hoc, as half filling is imposed.
//[!TODO optimize].
//*
//* @param state defined as a column vector of ones and zeros:
//*	for fermions one has vector of #Nsite spin ups and then #Nsite spin downs
//*	for spins one has only vector of #Nsite spins that can be transformed to Abrikosov fermions
//*/
//template <typename _Ht, typename _T, class _stateType>
//void RBM_PP_S<4, _Ht, _T, _stateType>::ansatzPP(const _stateType& _n)
//{
//	// !TODO
//	LOGINFO("Function " + __FUNCTION__ + " not implemented yet!", LOG_TYPES::ERROR, 0);
//	throw (std::runtime_error(__FUNCTION__));
//}
#	endif
//////////////////////////////////////////////////////////////////////////////////////////

// ##########################################################################################################################################
// ##########################################################################################################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

#endif