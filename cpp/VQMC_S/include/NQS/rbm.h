#pragma once
#ifndef RBM_H
#define RBM_H

#ifndef NQS_H
	#include "../nqs.h"
#endif // !NQS_H

// ############################## D E F I N I T I O N S   F O R   R B M ##############################

#ifdef NQS_RBM_USESR
	// calculation of the inverse 
	#ifdef NQS_RBM_PINV
		constexpr auto tolPinv = 1e-5;
		#ifdef NQS_RBM_SREG
			#undef NQS_RBM_SREG
		#endif
	#else
		#ifndef NQS_RBM_SREG
			#define NQS_RBM_SREG
		#endif
		constexpr double regB = 0.95;
		constexpr double regLambda0 = 100;
		constexpr double regLambdaMin = 1e-4;
	#endif
#endif

/*
* @brief Restricted Boltzmann Machines ansatz for quantum state
*/
template <typename _T, typename _Ht>
class RBM_S : public NQS<_T, _Ht> 
{
public:
	using NQSS						=					NQS<_T, _Ht>::NQSS;
	using NQSB						=					NQS<_T, _Ht>::NQSB;
	using NQSW						=					NQS<_T, _Ht>::NQSW;
protected:
	uint nHid_						=					1;
	// general parameters
#ifdef NQS_RBM_SREG
	double regBCurrent				=					0;				// parameter for regularisation, changes with Monte Carlo steps
	double regBMult					=					regB;			// starting parameter for regularisation
#endif

	// ------------------------- W E I G H T S ---------------------------
	NQSW S_;															// positive semi-definite covariance matrix
	NQSW W_;															// weight matrix
	NQSB bV_;															// visible bias
	NQSB bH_;															// hidden bias
	NQSB F_;															// forces
	
	// ------------------------- A N G L E S -----------------------------
	NQSB theta_;
	auto hiperbolicF(const NQSS& _v) const -> NQSB		{ return arma::cosh(this->bH_ + this->W_ * _v); };
	auto hiperbolicF() const -> NQSB					{ return arma::cosh(this->theta_); };
	

	// -------------------------------------------------------------------
protected:
	// ------------------------- A L O C A T O R S -----------------------
	void setInfo()										override final;
	void allocate()										override final;

	// ------------------------- S E T T E R S --------------------------
	void setState(NQSS _st)								override final;
	void setState(NQSS _st, bool _set)					override final;
	void setState(u64 _st)								override final;
	void setState(u64 _st, bool _set)					override final;


	// ------------------------- W E I G H T S --------------------------
	void setTheta();
	void setTheta(const NQSS& v);
	void updTheta();
	void updTheta(const NQSS& v);

	// ------------------------- T R A I N ------------------------------
#ifdef NQS_RBM_SREG 
	void rescaleS() {
		this->S.diag() += (regLambda0 < regLambdaMin) ? regLambdaMin : (regLambda0 * this->regBCurrent);
	};

#endif
	void grad(const NQSS& _v, uint _plc)				override final;
	void collectGrad(const NQSB& _energies)				override final;
	void updateWeights()								override final;
	void saveWeights(std::string _dir)					override final;

	void derivativeSR(int step);

	// ------------------------- P R O B A B I L I T Y --------------------------
	auto pRatio()					const -> _T			override final;
	auto pRatio(uint fP, float fV)  const -> _T			override final;
	auto pRatio(uint nFlips)		const -> _T			override final;
	auto pRatio(const NQSS& _v)		const -> _T			override final;
	auto pRatio(const NQSS& _v1,
				const NQSS& _v2)	const -> _T			override final;
	auto pRatioChange(_T _val, 
					  u64 _state, 
					  uint _vid)	-> _T				override final;

	// -------------------------------------------------------------------
public:
	~RBM_S()						=					default;
	RBM_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _nHid, uint _batch, uint _threadNum, double _lr)
		: NQS<_T, _Ht>(_H, _batch, _threadNum, _lr), nHid_(_nHid)
	{
		this->fullSize_ = this->nHid_ + this->nVis_ + this->nHid_ * this->nVis_;
	}

	// --------------------- S E T T E R S ---------------------
	void init()											override final;
	void init(const std::string& _p)					override final {};

	// --------------------- G E T T E R S ---------------------
	auto getNhid()					const -> uint		{ return this->nHid_; };
	
	// --------------------- F I N A L E -----------------------
	/*
	* @brief reproduces the RBM NQS ANSATZ
	*/
	auto ansatz(const NQSS& _in)	const -> _T			override final
	{
		return (std::exp(arma::dot(this->bV_, _in)) * arma::prod(this->hiperbolicF(_in))) / std::sqrt(this->nVis_); //* std::pow(2.0, this->n_hidden)
	};

	// --------------------- S A M P L I N G -------------------
	
	arma::Col<_T> train(uint nSam, uint nThrm,
		uint nBlck, uint bSize,
		uint nFlip = 1, uint progPrc = 25)
	{
		const int _stps = nBlck - nThrm;
#ifdef NQS_RBM_SREG
		this->regBCurrent = this->regBMult;
#endif
		this->Derivatives_.resize(_stps, this->fullSize_);
		return NQS<_T, _Ht>::train(nSam, nThrm, nBlck, bSize, nFlip, progPrc);
	};
};

// ##############################################################################################################################################


/*
* @brief Calculates the probability ratio whenever we use only one flip
*/
template<typename _T, typename _Ht>
inline auto RBM_S<_T, _Ht>::pRatio() const -> _T
{
	// preset the value
#ifdef NQS_RBM_ANGLES_UPD
	auto hiperbolic	= this->hiperbolicF();
#else
	auto hiperbolic = this->hiperbolicF(this->curVec);
#endif
	uint flipPlace	=	this->flipPlaces_[0];
	uint flipVal	=	this->flipVals_[0];
	// set the first value of b_visible
#ifdef SPIN
	_T val			=	(-2.0 * flipVal);
#else
	_T val			=	(1.0 - 2.0 * flipVal);
#endif
		// use value as the change already
#ifdef NQS_RBM_ANGLES_UPD
	val = val * this->bV_(flipPlace) + arma::sum(arma::log(arma::cosh(this->theta_ + val * this->W_.col(flipPlace)) / hiperbolic));
#else
	val = val * this->bV_(flipPlace) + arma::sum(arma::log(this->hiperbolicF(this->tmpVec) / hiperbolic));
#endif
	return exp(val);
}

template<typename _T, typename _Ht>
inline auto RBM_S<_T, _Ht>::pRatio(uint fP, float fV) const -> _T
{
	// preset the value
#ifdef NQS_RBM_ANGLES_UPD
	auto hiperbolic = this->hiperbolicF();
#else
	auto hiperbolic = this->hiperbolicF(this->curVec);
#endif
	// set the first value of b_visible
#ifdef SPIN
	_T val = (-2.0 * fV);
#else
	_T val = (1.0 - 2.0 * fV);
#endif
	// use value as the change already
#ifdef NQS_RBM_ANGLES_UPD
	val = val * this->bV_(fP) + arma::sum(arma::log(arma::cosh(this->theta_ + val * this->W_.col(fP)) / hiperbolic));
#else
	val = val * this->bV_(fP) + arma::sum(arma::log(this->hiperbolicF(this->tmpVec) / hiperbolic));
#endif
	return exp(val);
}

template<typename _T, typename _Ht>
inline auto RBM_S<_T, _Ht>::pRatio(uint nFlips) const -> _T
{
	_T val = 0;
	// preset the value
#ifdef NQS_RBM_ANGLES_UPD
	_T hiperbolic = this->hiperbolicF();
#else
	auto hiperbolic = this->hiperbolicF(this->curVec);
#endif
	// iterate through the flips
	for (uint i = 0; i < this->flipVals_.size(); i++)
	{
		uint flipPlace = this->flipPlaces_[i];
		uint flipVal = this->flipVals_[i];
		// set the first value of b_visible
#ifdef SPIN
		val += (-2.0 * flipVal);
#else
		val += (1.0 - 2.0 * flipVal);
#endif
		// use value as the change already
#ifdef NQS_RBM_ANGLES_UPD
		val = val * this->bV_(flipPlace) + arma::sum(arma::log(arma::cosh(this->theta_ + val * this->W_.col(flipPlace)) / hiperbolic));
#else
		val = val * this->bV_(flipPlace) + arma::sum(arma::log(this->hiperbolicF(this->tmpVec) / hiperbolic));
#endif
	}
	return exp(val);
}

template<typename _T, typename _Ht>
inline auto RBM_S<_T, _Ht>::pRatio(const NQSS& _v) const -> _T
{
	_T val = arma::dot(this->bV_, arma::Col<double>(_v - this->curVec));
	val += arma::sum(arma::log(this->hiperbolicF(_v) / this->hiperbolicF()));
	return std::exp(val);
}

/*
* @brief computes Log Psi'/Psi, where Psi' is the state with certain flipped spins
*/
template<typename _T, typename _Ht>
inline auto RBM_S<_T, _Ht>::pRatio(const NQSS& _v1, const NQSS& _v2) const -> _T
{
	_T val = arma::dot(this->bV_, arma::Col<double>(_v2 - _v1));
	val += arma::sum(arma::log(this->hiperbolicF(_v2) / this->hiperbolicF(_v1)));
	return std::exp(val);
}

// ##############################################################################################################################################

/*
* @brief probability ratio change due to the state change
* @param v value of the ratio
* @param state the state that we change onto
*/
template<typename _T, typename _Ht>
inline auto RBM_S<_T, _Ht>::pRatioChange(_T _val, u64 _state, uint _vid) -> _T
{
	INT_TO_BASE(_state, this->tmpVecs[_vid], Operators::_SPIN_RBM);
#ifndef NQS_RBM_ANGLES_UPD
	return _val * this->pRatio(this->curVec, this->tmpVecs[_vid]);
#else
	return _val * this->pRatio(this->tmpVecs[_vid]);
#endif
}

// ##############################################################################################################################################

template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::setState(NQSS _st, bool _set)
{
	// make sure it is the same base
	this->curVec			=			_st / _st(0) * Operators::_SPIN_RBM;
	this->curState			=			BASE_TO_INT<u64>(_st, Operators::_SPIN_RBM);
#ifdef NQS_RBM_ANGLES_UPD
	if (_set)
		this->setAngles();
#endif
}

template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::setState(NQSS _st)
{
	this->curVec			=			_st / _st(0) * Operators::_SPIN_RBM;
	this->curState			=			BASE_TO_INT<u64>(_st, Operators::_SPIN_RBM);
}

template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::setState(u64 _st, bool _set)
{
	this->curState			=			_st;
	INT_TO_BASE(_st, this->curVec, Operators::_SPIN_RBM);
#ifdef NQS_RBM_ANGLES_UPD
	if (_set)
		this->setAngles();
#endif
}

template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::setState(u64 _st)
{
	this->curState			=			_st;
	INT_TO_BASE(_st, this->curVec, Operators::_SPIN_RBM);
	this->curVec			*=			Operators::_SPIN_RBM;
}

// ##############################################################################################################################################

template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::allocate()
{
	// allocate weights
	this->bV_.resize(		this->nVis_);
	this->bH_.resize(		this->nHid_);
	this->theta_.resize(	this->nHid_);
	this->W_.resize(		this->nHid_,		this->nVis_);
	// allocate gradients
	this->F_.resize(		this->fullSize_);
#ifdef NQS_RBM_USESR
	this->S_.resize(		this->fullSize_,	this->fullSize_);
#endif
	this->curVec		=	arma::ones(this->nVis_);
}

// ##############################################################################################################################################

/*
* @brief Initializes the weights and biases for the RBM NQS model
*/
template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::init()
{
	// initialize with a random state
	this->setRandomState();
	// initialize biases visible
	for (uint i = 0; i < this->nVis_; i++)
		this->bV_(i) = 0.1 * (this->ran_.random<double>() + I * this->ran_.randomNormal<double>());
	// initialize biases hidden
	for (uint i = 0; i < this->nHid_; i++)
		this->bH_(i) = 0.1 * (this->ran_.random<double>() + I * this->ran_.randomNormal<double>());
	// weights matrix
	for (uint i = 0; i < this->W_.n_rows; i++)
		for (uint j = 0; j < this->W_.n_cols; j++)
			this->W_(i, j) = 0.1 * (this->ran_.random<double>() + I * this->ran_.randomNormal<double>());
}

// ##############################################################################################################################################

/*
* @brief Sets the info about the RBM model NQS
*/
template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::setInfo()
{
	this->info_ = "";
	strSeparatedS(this->info_, ',',
		"RBM", VEQV(nv, this->nVis_), VEQV(nh, this->nHid_), VEQV(b, this->batch_), VEQVS(lr, this->lr_));
}

// ##############################################################################################################################################

template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::grad(const NQSS& _v, uint _plc)
{
	// update the angles if it is necessary
#ifndef NQS_RBM_ANGLES_UPD
	this->setTheta(_v);
#endif

	// calculate the flattened part
#pragma omp parallel for num_threads(this->threadNum_)
	for (uint i = 0; i < this->nVis_; i++)
		this->Derivatives_(_plc, i) = _v(i);

#pragma omp parallel for num_threads(this->threadNum_)
	for (uint i = 0; i < this->nHid_; i++)
		this->Derivatives_(_plc, i + this->nVis_) = std::tanh(this->theta_(i));

#pragma omp parallel for num_threads(this->threadNum_)
	for (uint i = 0; i < this->nHid_; i++)
		for (auto j = 0; j < this->nVis_; j++)
			this->Derivatives_(_plc, (this->nVis_ + this->nHid_) + i + j * this->nHid_) = this->Derivatives_(_plc, i + this->nVis_) * _v(j);

}

// ##############################################################################################################################################

/*
* @brief Based on energies collected in steps, calculates the derivatives
*/
template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::collectGrad(const NQSB& _energies)
{
	this->F_ = arma::cov(arma::conj(this->Derivatives_), _energies);
#ifdef NQS_RBM_USE_SR
	this->S_ = arma::cov(arma::conj(this->Derivatives_), this->Derivatives_);
	
	// update model
	this->derivativeSR(0);

	#ifdef NQS_RBM_SREG
	this->regBCurrent *= regBMult;
	#endif // S_REGULAR
#else
	// standard updater
	this->F_ *= this->lr_;
#endif
}

// ##############################################################################################################################################

/*
* @brief Updates the weights in the system according to a given gradient
* @warning uses this->F_ forces vector to update them
*/
template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::updateWeights()
{
#pragma omp parallel for num_threads(this->threadNum_)
	for (auto i = 0; i < this->nVis_; i++)
		this->bV_(i) -= this->F_(i);

#pragma omp parallel for num_threads(this->threadNum_)
	for (auto i = 0; i < this->nHid_; i++)
		this->bH_(i) -= this->F_(i + this->nVis_);

#pragma omp parallel for num_threads(this->threadNum_)
	for (auto i = 0; i < this->nHid_; i++)
		for (auto j = 0; j < this->nVis_; j++)
			this->W_(i, j) -= this->F_((this->nVis_ + this->nHid_) + i + j * this->nHid_);

}

// ##############################################################################################################################################

/*
* @brief sets the current angles vector according to arXiv:1606.02318v1
*/
template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::setTheta()
{
	this->theta_ = this->bH_ + this->W_ * this->curVec;
}

/*
* @brief sets the current angles vector according to arXiv:1606.02318v1
* @param v replaces current vector
*/
template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::setTheta(const NQSS& v)
{
	this->theta_ = this->bH_ + this->W_ * v;
}

// ##############################################################################################################################################

/*
* @brief Update angles with the flipped spin/spins 
* @warning values are stored in flipValues_ before the flip, hence "-" is taken
* @warning flipPlaces_ stores the flip places
*/
template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::updTheta()
{
	for (uint i = 0; i < this->flipPlaces_.size(); i++)
#ifdef SPIN
		this->theta_ -= (2.0 * this->flipVals_[i]) * this->W_.col(this->flipPlaces_[i]);
#else
		this->theta_ += (1.0 - 2.0 * this->flipVals_[i]) * this->W.col(flipPlaces_[i]);
#endif
}

template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::updTheta(const NQSS& v)
{
	for (uint i = 0; i < this->flipPlaces_.size(); i++)
#ifdef SPIN
		this->theta_ -= (2.0 * v[this->flipPlaces_[i]]) * this->W_.col(this->flipPlaces_[i]);
#else
		this->theta_ += (1.0 - 2.0 * v[this->flipPlaces_[i]]) * this->W.col(flipPlaces_[i]);
#endif
}

// ##############################################################################################################################################

/*
* @brief updates the weights using stochastic gradient descent or stochastic reconfiguration [SR]
*/
template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::derivativeSR(int step)
{
	// update flat vector
#ifdef PINV
	this->F_ = this->lr_ * (arma::pinv(this->S, tolPinv) * this->F_);
	return;
#elif defined NQS_RBM_SREG 
	this->rescaleS();
#else 
	this->S_.diag() = this->S_.diag() + (this->lr_ * arma::ones(this->S_.n_rows));
#endif 
	this->F_ = this->lr_ * arma::solve(this->S_, this->F_);
}

// ##############################################################################################################################################

template<typename _T, typename _Ht>
inline void RBM_S<_T, _Ht>::saveWeights(std::string _dir)
{
	std::string dir = _dir + kPS + "RBM" + kPS;
	std::string filename = dir + "weights_" + prettyTime(clk::now()) + ".h5";
	createDir(dir);
	this->W_.save(arma::hdf5_name(filename, "W"));
	this->bH_.save(arma::hdf5_name(filename, "hidden", arma::hdf5_opts::append));
	this->bV_.save(arma::hdf5_name(filename, "visible", arma::hdf5_opts::append));
}

#endif