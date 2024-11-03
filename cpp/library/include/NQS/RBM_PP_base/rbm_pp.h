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
	#include "../rbm_final.hpp"
#endif // !NQS_H

#define NQS_RBM_PP_USE_PFAFFIAN_UPDATE

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! B A S E !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

/*
* @brief Restricted Boltzmann Machines ansatz with Pair Product reference state for NQS
*/
template < 	uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class RBM_PP : public RBM_S<_spinModes, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
	using NQSLS_p = typename RBM_S<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
protected:
	// architecture parameters
	uint nPP_					= 1;
	uint nSites2_				= 1;
	uint nParticles2_			= 1;
	u64 rbmPPSize_				= 1;

	/* ------------------------------------------------------------ */
	// ------------------------ W E I G H T S ------------------------
	NQSB Fpp_;													// for storing the additional variational parameters from the PP
	v_2d<bool> spinSectors_;									// go through the quarters (spin sectors)

	_T pfaffian_				= 0.0;							// store the last pfaffian value
	_T pfaffianNew_				= 0.0;							// store the last pfaffian value for the candidate
	NQSW X_;													// for stroing the matrix for Pfaffian calculation at each step
	NQSW Xinv_;													// for stroing the matrix inverse for Pfaffian calculation at each step
	NQSW XinvSkew_;												// for stroing the matrix inverse for Pfaffian calculation at each step
	NQSW Xnew_;													// for stroing the matrix for Pfaffian calculation at each step - new candidate
	
// for calculating the Pfaffian probabilities from the Hamiltonian
#if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP 
	// create the map for thetas for a given thread
	std::map<std::thread::id, NQSW> XTmp_;
#else
	NQSW XTmp_;
#endif

	/* ------------------------------------------------------------ */
	
	u64 getFPPIndex(bool _spini, bool _spinj, uint ri, uint rj)	const;

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
	virtual void updFPP_C(uint fP, float fV)					= 0;
	virtual void updFPP_C(std::initializer_list<int> fP,
						std::initializer_list<double> fV)		= 0;
	virtual void updFPP(uint fP, float fV)						= 0;
	virtual void updFPP(std::initializer_list<int> fP,
						std::initializer_list<double> fV)		= 0;
	virtual void updFPP_F(std::initializer_list<int> fP, 
						std::initializer_list<double> fV,
						arma::Mat<_T>& _Xtmp)					= 0;
public: 				  
	~RBM_PP() override											{ DESTRUCTOR_CALL;												};
	RBM_PP(std::shared_ptr<Hamiltonian<_Ht>> _H, uint _nHid, double _lr, uint _threadNum = 1, 
													int _nPart = -1, 
													const NQSLS_p& _lower = {}, 
													std::vector<double> _beta = {});

	// --------------------- S E T T E R S ---------------------
	void init()													override final;
	// for the PP matrix
	void setFPP()												{ this->X_ = this->getPPMat();									};
	void setFPP(const NQSS& _n)									{ this->X_ = this->getPPMat(_n);								};
	void setFPP(u64 _n)											{ this->X_ = this->getPPMat(_n);								};
	// for the new PP matrix
	void setFPP_C()												{ this->Xnew_ = this->getPPMat();								};
	void setFPP_C(const NQSS& _n)								{ this->Xnew_ = this->getPPMat(_n);								};
	void setFPP_C(u64 _n)										{ this->Xnew_ = this->getPPMat(_n);								};
	// for the pfaffian value
	void setPfaffian()											{ this->pfaffian_ = this->getPfaffian();						};
	void setPfaffian(const NQSS& _n)							{ this->setFPP(_n); this->setPfaffian();						};
	void setPfaffian(const NQSW& _M)							{ this->pfaffian_ = this->getPfaffian(_M);						};
	// for the candidate pfaffian value 
	void setPfaffian_C()										{ this->pfaffianNew_ = this->getPfaffian_C();					};
	void setPfaffian_C(const NQSS& _n)							{ this->setFPP_C(_n); this->setPfaffian_C();					};
	void setPfaffian_C(const NQSW& _M)							{ this->pfaffianNew_ = this->getPfaffian(_M);					};

	// -------------------- U P D A T E R S --------------------

#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	void updateXInv(uint _row);
	void updateXInv_C(uint _row);
	void updatePfaffian(uint _row);
	void updatePfaffian(uint _row, _T& _pfaffian);
	void updatePfaffian(uint _row, _T& _pfaffian, const arma::Mat<_T>& _X);
	void updatePfaffian_C(uint _row);
#endif

	// --------------------- G E T T E R S ---------------------
	
	virtual auto getPPMat(const NQSS& _n)	const -> NQSW		= 0;
#ifndef NQS_USE_VEC_ONLY
	virtual auto getPPMat(u64 _n)			const -> NQSW		= 0;
#endif
	// based on the current vector, obtain a matrix for Pffafian calcuation
	virtual auto getPPMat()					const -> NQSW		{ return this->getPPMat(this->curVec_);										};
	// based on the current matrix, obtain the Pffafian
	auto getPfaffian()						const -> _T			{ return algebra::Pfaffian::pfaffian<_T>(this->X_, this->info_p_.nParticles_);		};
	// based on the current candidate matrix, obtain the Pffafian
	auto getPfaffian_C()					const -> _T			{ return algebra::Pfaffian::pfaffian<_T>(this->Xnew_, this->info_p_.nParticles_);	};
	// get the Pffafian value for a given vector
	auto getPfaffian(const NQSS& _in)		const -> _T			{ return this->getPfaffian(this->getPPMat(_in));							};
	// based on given matrix, obtain the Pffafian
	static auto getPfaffian(const NQSW& _M)	-> _T				{ return algebra::Pfaffian::pfaffian<_T>(_M, _M.n_rows);					};
	
	// --------------------------------------------------------

	// get the current Pffafian value
	auto getCurrentPfaffian()				const -> _T			{ return this->pfaffian_;													};
	// get the current Pffafian candidate value
	auto getCurrentPfaffian_C()				const -> _T			{ return this->pfaffianNew_;												};
	// get size of the PP function matrix
	auto getNPP()							const -> uint		{ return this->nPP_;														};
	// --------------------- F I N A L E -----------------------
	auto ansatz(const NQSS& _in)			const -> _T			override final;
	auto ansatzlog(const NQSS& _in)			const -> _T			override final;
	auto ansatz_ratio(const NQSS& _in, 
		NQS<_spinModes, _Ht, _T, _stateType>* _other) 			const -> _T override final;
};

// ##########################################################################################################################################

// ########################################################### C O N S T R U C T ############################################################

// ##########################################################################################################################################

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM_PP<_spinModes, _Ht, _T, _stateType>::RBM_PP(std::shared_ptr<Hamiltonian<_Ht>> _H, uint _nHid, 
							double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta)
	: RBM_S<_spinModes, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nPart, _lower, _beta)
{
	// create the spin sectors
	spinSectors_.push_back({ 1, 1 });
	spinSectors_.push_back({ 1, 0 });
	spinSectors_.push_back({ 0, 1 });
	spinSectors_.push_back({ 0, 0 });

	// !TODO make this changable
	this->nSites2_		= this->info_p_.nSites_ * this->info_p_.nSites_;
	this->nParticles2_	= this->info_p_.nParticles_ * this->info_p_.nParticles_;
	this->nPP_			= this->spinSectors_.size() * this->nSites2_; // for both spin channels
	this->rbmPPSize_	= this->rbmSize_ + this->nPP_;
	this->info_p_.fullSize_		= this->rbmPPSize_;
	this->allocate();
	this->setInfo();
}

// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

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
#ifndef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	// calculate the inverse only if necessary (without the updates at each step).
	this->Xinv_				= arma::inv(this->X_);
#endif // !NQS_RBM_PP_USE_PFAFFIAN_UPDATE

	// not multiplying by pfaffian, as it's dividied by it later in the definition of the derivative
	this->XinvSkew_			= (this->Xinv_.st() - this->Xinv_);
	//this->XinvSkew_			= (this->Xinv_ - this->Xinv_.st());
	auto _currDerivative	= this->derivatives_.row(_plc).subvec(this->rbmSize_, this->rbmPPSize_ - 1);

	// find the correct indices on the values that can be updated
	_currDerivative.zeros();

	int _spinIter [[maybe_unused]] = 0;
//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threads_.threadNum_)
//#endif
	for (const auto& s: this->spinSectors_)
	{
		for (uint i = 0; i < this->info_p_.nParticles_; ++i)
		{
			// if left spin differs from the current left spin, just continue
			if (checkBit(this->curVec_, i) != s[0])
				continue;

			for (uint j = 0; j < this->info_p_.nParticles_; ++j)
			{
				// skip the diagonal part
				if (j == i)
					continue;

				// if right spin differs from the current right spin, just continue
				if (checkBit(this->curVec_, j) != s[1])
					continue;

				_currDerivative(this->getFPPIndex(s[0], s[1], i, j)) = this->Xinv_(i, j);
			}
		}
		_spinIter += this->nParticles2_;
	}
}

// ##########################################################################################################################################

// ############################################################ U P D A T E R S #############################################################

// ##########################################################################################################################################

////////////////////////////////////////////////////////////////////////
#ifdef NQS_ANGLES_UPD

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
	//for (uint i = 0; i < nFlips; i++)
	//{
	//	// update the Pffafian matrix for the candidate
	//	const auto fP = this->flipPlaces_[i];
	//	const auto fV = this->flipVals_[i];
	//	this->updFPP(fP, fV);
	//}
	// update the Pffafian matrix
	// as the candidate pfaffian shall be already updated, use it instead of calculating everything all the time (probably not as efficient)
	// replace updating the pfaffian back
#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	for(uint i = 0; i < nFlips; i++)
		this->Xinv_	= algebra::scherman_morrison_skew(this->Xinv_, this->flipPlaces_[i], this->Xnew_.row(this->flipPlaces_[i]));
#endif
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
	RBM_S<_spinModes, _Ht, _T, _stateType>::unupdate(nFlips);
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
	//for (uint i = 0; i < nFlips; i++)
	//{
	//	// update the Pffafian matrix for the candidate
	//	const auto fP = this->flipPlaces_[i];
	//	this->updFPP(fP, v(fP));
	//}
#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	for(uint i = 0; i < nFlips; i++)
		this->Xinv_	= algebra::scherman_morrison_skew(this->Xinv_, this->flipPlaces_[i], this->Xnew_.row(this->flipPlaces_[i]));
#endif
	this->X_		= this->Xnew_;
	this->pfaffian_ = this->pfaffianNew_;
}
#endif

#endif