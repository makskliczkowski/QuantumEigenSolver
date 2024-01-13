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

	/* ------------------------------------------------------------ */
	// ------------------------ W E I G H T S ------------------------
	_T pfaffianVal_				= 0.0;							// store the last pfaffian value
	_T pfaffianValCandidate_	= 0.0;							// store the last pfaffian value
	NQSW Pfaffian_;												// for stroing the Pfaffian at each step
	NQSW PfaffianCandidate_;
	NQSB Fmat_;													// for storing the additional waves from the PP

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
		std::initializer_list<double> fV)	-> _T				override final;

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
#endif

	// ---------------------------- T R A I N ----------------------------	
	void grad(const NQSS& _v, uint _plc)						override final;

	// --------------------------- A N S A T Z ---------------------------
	// sets the Pffafian matrix inside the Pfaffian_

public:
	~RBM_PP() override											{ DESTRUCTOR_CALL; };
	RBM_PP(std::shared_ptr<Hamiltonian<_Ht>> _H, uint _nHid, double _lr, uint _threadNum = 1, int _nPart = -1);

	// --------------------- S E T T E R S ---------------------
	void init()							override final;
	// for the previous pfaffian matrix
	void setPffMat()											{ this->Pfaffian_ = this->getPffMat(); };
	void setPffMat(const NQSS& _n)								{ this->Pfaffian_ = this->getPffMat(_n); };
	void setPffMat(u64 _n)										{ this->Pfaffian_ = this->getPffMat(_n); };
	// for the new pfaffian matrix
	void setPffMatC()											{ this->PfaffianCandidate_ = this->getPffMat(); };
	void setPffMatC(const NQSS& _n)								{ this->PfaffianCandidate_ = this->getPffMat(_n); };
	void setPffMatC(u64 _n)										{ this->PfaffianCandidate_ = this->getPffMat(_n); };
	// for the pfaffian value
	void setPffVal()											{ this->pfaffianVal_ = this->getPff(); };
	void setPffVal(const NQSS& _n)								{ this->setPffMat(_n); this->setPffVal(); };
	void setPffVal(const NQSW& _M)								{ this->pfaffianVal_ = this->getPff(_M); };
	// for the candidate pfaffian value 
	void setPffValC()											{ this->pfaffianValCandidate_ = this->getPff(); };
	void setPffValC(const NQSS& _n)								{ this->setPffMatC(_n); this->setPffValC(); };
	void setPffValC(const NQSW& _M)								{ this->pfaffianValCandidate_ = this->getPff(_M); };
	// --------------------- G E T T E R S ---------------------
	virtual auto getPffMat(const NQSS& _n)	const -> NQSW		= 0;
	virtual auto getPffMat(u64 _n)			const -> NQSW		= 0;
	virtual auto getPffMat()			const -> NQSW			{ return this->getPffMat(this->curState_); }
	auto getNPP()						const -> uint			{ return this->nPP_; };
	auto getPff()						const -> _T				{ return algebra::pfaffian<_T>(this->Pfaffian_, this->nParticles_); }
	auto getPff(const NQSW& _M)			const -> _T				{ return algebra::pfaffian<_T>(_M, _M.n_rows); }
	auto getPffC()						const -> _T				{ return algebra::pfaffian<_T>(this->PfaffianCandidate_, this->nParticles_); }
	auto getPffV()						const -> _T				{ return this->pfaffianVal_; }
	auto getPffV(const NQSS& _in)		const -> _T				{ return this->getPff(this->getPffMat(_in)); }
	auto getPffVC()						const -> _T				{ return this->pfaffianValCandidate_; }
	// --------------------- F I N A L E -----------------------
	auto ansatz(const NQSS& _in)		const -> _T				override final;
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
	// !TODO make this changable
	this->nPP_		= 4 * this->nSites_ * this->nSites_; // for both spin channels
	this->fullSize_ = this->nHid_ + this->nVis_ + this->nHid_ * this->nVis_ + this->nPP_;
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
* @brief Allocate the memory for the NQS parameters. 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::allocate()
{
	// !TODO implement changable number of fermions
	// allocate weights
	// Pfaffian matrix for each step
	this->Pfaffian_	= NQSW(this->nParticles_, this->nParticles_, arma::fill::zeros);
	// allocate the weights themselves !TODO - make this symmetric? 
	// square matrix with spin changes F_{ij}^{\\sigma, \\sigma '}
	this->Fmat_		= NQSB(this->nPP_);
	// allocate the rest
	RBM_S<_spinModes, _Ht, _T, _stateType>::allocate();
}

///////////////////////////////////////////////////////////////////////

/*
* @brief Initializes the weights and biases for the RBM-PP NQS model
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::init()
{
	// matrix for the PP wave function
	this->Fmat_	= 0.05 * NQSB(this->nSites_ * this->nSites_ * 4, arma::fill::randn);
	auto _lat	= this->H_->getLat();
	
	for (uint i = 0; i < this->nSites_; i++)
	{
		for (uint j = 0; j < this->nSites_; j++)
		{
			auto distance = _lat->getSiteDistance(i, j);
			for (uint _spin = 0; _spin < 4; ++_spin)
			{
				// calculate the position in the matrix (vector)
				auto p = this->nSites_ * i + j + _spin * this->nSites_ * this->nSites_;
				// make the weights proportional to the distance
				if (distance != 0)
					this->Fmat_(p) /= distance * distance;
			}
		}
	}
	// !TODO
	RBM<_spinModes, _Ht, _T, _stateType>::init();
}

///////////////////////////////////////////////////////////////////////

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
	return RBM_S<_spinModes, _Ht, _T, _stateType>::pRatio(fP, fV) / this->pfaffianVal_;
}

// %%%%%%%%%%%%%%%% M U L T I P L E   F L I P S %%%%%%%%%%%%%%%%

/*
* @brief Calculates the probability ratio whenever we use multiple flips.
* Uses the flips stored within the NQS class (flipPlaces_, flipVals_)
* @param nFlips number of flips to be used
* @returns probability ratio for a given ansatz based on the current state
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::pRatio(uint nFlips)
{
	//!TODO
	return _T(0);
}

// %%%%%%%%%%%%%%%%% U S I N G   V E C T O R S %%%%%%%%%%%%%%%%%

/*
* @brief computes (Psi'/Psi), where (Psi') is the state with certain positions flipped. 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::pRatio(const NQSS& _v1, const NQSS& _v2)
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
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::pRatio(std::initializer_list<int> fP, std::initializer_list<double> fV)
{
	//!TODO
	return _T(0);
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# S E T T E R S ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::setState(const NQSS& _st, bool _set)
{
	RBM_S<_spinModes, _Ht, _T, _stateType>::setState(_st, _set);
#ifdef NQS_ANGLES_UPD
	if (_set) this->setPffVal();
#endif
}

////////////////////////////////////////////////////////////////////////////

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::setState(u64 _st, bool _set)
{
	RBM_S<_spinModes, _Ht, _T, _stateType>::setState(_st, _set);
#ifdef NQS_ANGLES_UPD
	if (_set) this->setPffVal();
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
		this->Fmat_ = arma::reshape(this->F_.subvec(this->nVis_ + this->nHid_ + this->W_.n_rows * this->W_.n_cols,
									this->fullSize_ - 1),
									this->W_.n_rows, this->W_.n_cols);
	}
	END_CATCH_HANDLER("Couldn't set the weights for the RBM PP NQS...", return false);
	return true;
}

///////////////////////////////////////////////////////////////////////

/*
* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @returns whether the load has been successful
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool RBM_PP<_spinModes, _Ht, _T, _stateType>::saveWeights(std::string _path, std::string _file)
{
	BEGIN_CATCH_HANDLER
	{
		this->F_.subvec(this->nVis_ + this->nHid_ + this->W_.n_rows * this->W_.n_cols, this->fullSize_ - 1) = this->Fmat_;
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
	// update the rest
	// !TODO
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
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::grad(const NQSS& _v, uint _plc)
{
	// calculate the RBM derivative
	RBM_S<_spinModes, _Ht, _T, _stateType>::grad(_v, _plc);
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ U P D A T E R S #############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

////////////////////////////////////////////////////////////////////////////
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
	// this->ansatzPP();
	//for (uint i = 0; i < this->flipPlaces_.size(); i++)
	//{
	//	for (uint j = 0; i < this->nParticles_; ++i)
	//	{
	//		// update X_{i,j} -> F_{ij}^{s'_i, s_j} - F{ji}^{s_j, s'_i}
	//		auto pos_i = (v(i) > 0 ? 0 : this->nSites_) + i;

	//	}
	//}
}

////////////////////////////////////////////////////////////////////////////

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
	//this->ansatzPP(v);
}
#	endif
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
	virtual auto getPffMat(u64 _n)			const -> NQSW	override { NQS_LOG_ERROR_SPIN_MODES; return NQSW(); };
};
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
	/* ------------------------------------------------------------ */
public:
	RBM_PP_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, double _lr, uint _threadNum = 1, int _nParticles = -1)
		: RBM_PP<2, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nParticles) {};

	/* ------------------------------------------------------- */
	// --------------------- G E T T E R S ---------------------
	virtual auto getPffMat(const NQSS& _n)	const -> NQSW override;
	virtual auto getPffMat(u64 _n)			const -> NQSW override;
};

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!! P F F A F I A N   S T A T E !!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
RBM_PP_S<2, _Ht, _T, _stateType>::NQSW RBM_PP_S<2, _Ht, _T, _stateType>::getPffMat(const NQSS& _n) const
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
				_out(i, j) = this->Fmat_(positionLeft) - this->Fmat_(positionRight);
			}
		}
	}
	return _out;
}

////////////////////////////////////////////////////////////////

template <typename _Ht, typename _T, class _stateType>
RBM_PP_S<2, _Ht, _T, _stateType>::NQSW RBM_PP_S<2, _Ht, _T, _stateType>::getPffMat(u64 _n) const
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
				_out(i, j) = this->Fmat_(positionLeft) - this->Fmat_(positionRight);
			}
		}
	}
	return _out;
}

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