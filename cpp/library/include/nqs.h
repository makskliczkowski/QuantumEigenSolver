#pragma once

#ifndef NQS_H
#define NQS_H

// ##############################
// include the hamiltonian	 // #
#ifndef HAMIL_H				 // #
	#include "hamil.h"       // #
#endif // !HAMIL_H           // #
// ##############################

// include Machine Learning Stuff
// ##############################
#ifndef ML_H				 // #
	#include "ml.h"			 // #
#endif // !ML_H				 // #
// ##############################

// ######### NQS TYPES ##########
enum NQSTYPES {				 // #
	RBM						 // #
};							 // #
							 // #
BEGIN_ENUM(NQSTYPES)		 // #
{							 // #
	DECL_ENUM_ELEMENT(RBM)	 // #
}							 // #
END_ENUM(NQSTYPES)			 // #
// ##############################

#ifdef NQS_SAVE_WEIGHTS
	#define NQS_SAVE_DIR "DATA" + kPS + "NQS" + kPS + "WEIGHTS" + kPS
#endif

template <typename _Ht, typename _T>
class NQS {
public:
	using NQSS							=		arma::Col<double>;
	using NQSB							=		arma::Col<_T>;
	using NQSW							=		arma::Mat<_T>;
	std::shared_ptr<Hamiltonian<_Ht>> H_;									// pointer to the Hamiltonian

protected:
	std::string info_;
	pBar pBar_;																// for printing out the progress

	uint threadNum_						=		1;							// number of threads that works on this
	uint batch_							=		1;							// size of the batch
	uint nVis_							=		1;							// number of visible neurons
	uint fullSize_						=		1;							// full number of the parameters
	double lr_							=		0.0;						// specific learnign rate

	// Hamiltonian

	// Hilbert space info
	u64 Nh_								=		1;

	// Random number generator
	randomGen ran_;														// consistent quick random number generator

	// --------------------- O P T I M I Z E R S --------------------


	// --------------------- T R A I N I N G ------------------------
	v_1d<uint> flipPlaces_;
	v_1d<double> flipVals_;
	u64 curState;
	NQSW Derivatives_;
	NQSS curVec;
	NQSS tmpVec;
	v_1d<NQSS> tmpVecs;
	// --------------------------------------------------------------
protected:
	// --------------------- S T A R T E R S ---------------------
	virtual void setInfo()							=						0;
	virtual void allocate()							=						0;
	
	// --------------------- P R O B A B I L I T Y ---------------
	virtual auto pRatio()							->_T =					0;
	virtual auto pRatio(uint fP, float fV)			->_T =					0;
	virtual auto pRatio(std::initializer_list<int>,
					std::initializer_list<double>)	->_T =					0;
	virtual auto pRatio(uint nFlips)				->_T =					0;
	virtual auto pRatio(const NQSS& _v)				->_T =					0;
	virtual auto pRatio(const NQSS& _v1,
						const NQSS& _v2)			->_T =					0;
	virtual auto pRatioChange(_T _val,
							  u64 _state,
							  uint _vid)			-> _T =					0;

	// --------------------- S E T T E R S -----------------------
	virtual void setState(NQSS _st, bool _set)		=						0;
	virtual void setState(NQSS _st)					=						0;
	virtual void setState(u64 _st)					=						0;
	virtual void setState(u64 _st, bool _set)		=						0;
	
	// --------------------- W E I G H T S -----------------------
	virtual void updateWeights()					=						0;
	virtual void saveWeights(std::string _dir)		=						0;

	// --------------------- T R A I N   E T C -------------------
	virtual void grad(const NQSS& _v, uint _plc)	=						0;
	virtual void collectGrad(const NQSB& _energies)	=						0;

	// --------------------- E N E R G Y -------------------------
	auto chKernel(std::initializer_list<int>  fP,
				  std::initializer_list<double> fV)	-> _T					{ return this->pRatio(fP, fV); };
	_T locEnKernel();

	// -----------------------------------------------------------
public:
	// --------------------- S E T T E R S -----------------------
	virtual	void init()								=						0; 
	virtual	void update()							=						0; 
	virtual void setRandomState(bool _upd = true)							{ this->setState(this->ran_.template randomInt<u64>(0, this->Nh_), _upd); };
	
	// --------------------- F L I P S ---------------------------
	virtual void chooseRandomFlips();
	template<bool _TMP>
	void applyFlips(typename std::enable_if<_TMP, int>::type _tmp = 0)		{ for (auto& i : this->flipPlaces_) flip(this->tmpVec, i, 0, Operators::_SPIN_RBM); };
	template<bool _TMP>
	void applyFlips(typename std::enable_if<!_TMP, int>::type _tmp = 0)		{ for (auto& i : this->flipPlaces_) flip(this->curVec, i, 0, Operators::_SPIN_RBM); };

	// --------------------- G E T T E R S -----------------------
	auto getInfo()								    const -> std::string	{ return this->info_; };
	auto getHilbertSize()							const -> u64			{ return this->Nh_; };
	auto getNvis()									const -> uint			{ return this->nVis_; };
	auto getOMPID()									const -> uint
	{
#ifndef DEBUG
		if (this->threadNum_ == 1)
			return 0;
		else
			return omp_get_thread_num() % this->threadNum_;
#else
		return 0;
#endif
	};

	// --------------------- S A M P L I N G ---------------------
	virtual void blockSample(uint _bSize,
				 u64 _start, uint _nFlip,
				 bool _therm = false);
	virtual void blockSample(uint _bSize, u64 _start,
				 bool _therm = false);
	
	virtual arma::Col<_T> train(uint nSam, uint nThrm, 
						uint nBlck, uint bSize, 
						uint nFlip = 1, uint progPrc = 25);
	arma::Col<_T> collect(uint nSam, uint nThrm, 
						  uint nBlck, uint bSize, 
						  uint nFlip = 1)	{};

	// --------------------- F I N A L E -------------------------
	virtual auto ansatz(const NQSS& _in)			const ->_T =			0;


	// ------------------ C O N S T R U C T -------------------------

	public:
	virtual ~NQS()					=			default;
	/*
	* @brief interface constructor for Neural Network Quantum State
	*/
	NQS(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _batch, uint _threadNum, double _lr)
		: batch_(_batch), lr_(_lr)
	{
		this->H_					=			_H;
		this->nVis_					=			_H->getNs();
		this->Nh_					=			_H->getHilbertSize();
		this->ran_					=			_H->ran_;
#ifdef DEBUG
		this->threadNum_			=			1;
#else
		this->threadNum_			=			_threadNum;
#endif // DEBUG
		// Use threads for all consecutive parallel regions
		omp_set_num_threads(this->threadNum_);                  
	};
	NQS()							=			default;
};
// ##############################################################################################################################################

template<typename _Ht, typename _T>
inline void NQS<_Ht, _T>::chooseRandomFlips()
{
	for (uint i = 0; i < this->flipPlaces_.size(); i++) {
		this->flipPlaces_[i]	=		this->ran_.template randomInt<uint>(0, this->nVis_);
		this->flipVals_[i]		=		this->tmpVec(this->flipPlaces_[i]);
	}
}

// ##############################################################################################################################################

/*
* @brief block updates the current state according to Metropolis-Hastings algorithm
* @param _bSize the size of the correlation block
* @param _start the state to start from
* @param _nFlip number of flips in a single step
*/
template<typename _Ht, typename _T>
inline void NQS<_Ht, _T>::blockSample(uint _bSize, u64 _start, uint _nFlip, bool _therm)
{
	if (_start != this->curState || _therm)
		this->setState(_start, _therm);

	// set the temporary state
	this->tmpVec = this->curVec;
	for (uint bStep = 0; bStep < _bSize; bStep++)
	{
		this->chooseRandomFlips();

		// flip the vector
		this->applyFlips<true>();

		// check the probability
#ifndef NQS_RBM_ANGLES_UPD
		double proba = std::abs(this->pRatio(this->curVec, this->tmpVec));
#else
		double proba = std::abs(this->pRatio(_nFlip));
#endif
		if (this->ran_.template random<float>() <= proba * proba) {
			// update current state and vector
			this->curVec = this->tmpVec;

			// update angles if needed
			this->update();
		}
		// set the vector back to normal
		else
			this->tmpVec = this->curVec;
	}
}

template<typename _Ht, typename _T>
inline void NQS<_Ht, _T>::blockSample(uint _bSize, u64 _start, bool _therm)
{
	if (_start != this->curState || _therm)
		this->setState(_start, _therm);

	// set the temporary state
	this->tmpVec = this->curVec;
	for (uint bStep = 0; bStep < _bSize; bStep++)
	{
		this->flipPlaces_[0]	=	this->ran_.template randomInt<uint>(0, this->nVis_);
		this->flipVals_[0]		=	this->tmpVec(this->flipPlaces_[0]);

		// flip the vector
		flip(this->tmpVec, this->flipPlaces_[0], 0, Operators::_SPIN_RBM);

		// check the probability
#ifndef NQS_RBM_ANGLES_UPD
		double proba = std::abs(this->pRatio(this->curVec, this->tmpVec));
#else
		auto proba = std::abs(this->pRatio(this->flipPlaces_[0], this->flipVals_[0]));
#endif
		if (this->ran_.template random<float>() <= proba * proba) {
			// update current state and vector
			this->curVec(this->flipPlaces_[0]) = this->tmpVec(this->flipPlaces_[0]);

			// update angles if needed
			this->update();
		}
		// set the vector back to normal
		else
			this->tmpVec(this->flipPlaces_[0]) = this->curVec(this->flipPlaces_[0]);
	}
	this->curState = BASE_TO_INT<u64>(this->curVec, Operators::_SPIN_RBM);
#ifndef NQS_RBM_ANGLES_UPD
	this->setState(this->curState, true);
#endif
}

// ##############################################################################################################################################

/*
* @brief Calculate the local energy depending on the given Hamiltonian - kernel with OpenMP
*/
template<typename _Ht, typename _T>
inline _T NQS<_Ht, _T>::locEnKernel()
{
	double energyR = 0.0;
	double energyI = 0.0;
#ifndef DEBUG
	#pragma omp parallel for reduction(+ : energyR, energyI) num_threads(this->threadNum_)
#endif
	for (int site = 0; site < this->nVis_; site++) {
		//uint vid	=	this->getOMPID();			// gets the current vector id for the openMP
		auto energy	=	this->H_->locEnergy(this->curState, site, 
											std::bind(&NQS<_Ht, _T>::chKernel, 
												this,
												std::placeholders::_1,
												std::placeholders::_2
												)
											);
		energyR		+=	algebra::real(energy);
		energyI		+=	algebra::imag(energy);
	};

	return toType<_T>(energyR, energyI);
}

// ##############################################################################################################################################

template<typename _Ht, typename _T>
inline arma::Col<_T> NQS<_Ht, _T>::train(uint nSam, uint nThrm, uint nBlck, uint bSize, uint nFlip, uint progPrc)
{
	std::string outstr = "";

	// start the timer!
	auto _start = std::chrono::high_resolution_clock::now();

	// set the info about training
	strSeparatedP(outstr, '\t', 2,
		VEQV("mc", nSam),
		VEQV("mcTherm", nThrm),
		VEQV("nb", nBlck),
		VEQV("bs", bSize),
		VEQV("bs", nFlip)
	);
	LOGINFOG("Train: " + outstr, LOG_TYPES::TRACE, 1);

	// make the pbar!
	this->pBar_ = pBar(progPrc, nSam);

	// check if the batch is not bigger than the blocks number
	const int _stps = nBlck - nThrm;
	if (_stps < 0)
	{
		LOGINFOG("Number of steps is too small - thermalisaton too high!", LOG_TYPES::ERROR, 0);
		return {};
	};

	// save all average weights for covariance matrix
	arma::Col<_T> meanEn(nSam, arma::fill::zeros);
	arma::Col<_T> En(_stps, arma::fill::zeros);
	//NQSS derivatives(, arma::fill::zeros);

	this->setRandomState();
	this->flipPlaces_.resize(nFlip);
	this->flipVals_.resize(nFlip);
	for (uint i = 0; i < nSam; i++) {
		//this->setRandomState();
		// thermalize!
		if(nFlip == 1)
			this->blockSample(nThrm, this->curState, true);
		else
			this->blockSample(nThrm, this->curState, nFlip, true);

		// iterate blocks
		for (int _taken = 0; _taken < _stps; _taken++) {

			// sample them guys!
			if (nFlip == 1)
				this->blockSample(bSize, this->curState, false);
			else
				this->blockSample(bSize, this->curState, nFlip, false);

			// calculate the gradient!
			this->grad(this->curVec, _taken);

			// energy
			En(_taken) = this->locEnKernel();
		}
		//this->Derivatives_.print();
		this->collectGrad(En);
		this->updateWeights();
		// save the mean energy
		meanEn(i) = arma::mean(En);

		// update the progress bar
		if (i % this->pBar_.percentageSteps == 0) {
			this->pBar_.printWithTime(LOG_LVL2 + SSTR("PROGRESS"));
#ifdef NQS_SAVE_WEIGHTS
			this->saveWeights(NQS_SAVE_DIR);
#endif // NQS_SAVE_WEIGHTS
		}
	}
	LOGINFOG("FINISHED NQS TRAIN. TIME: " + TMS(_start), LOG_TYPES::TIME, 1);
	LOGINFOG("--- FINISHED NQS TRAIN ---", LOG_TYPES::FINISH, 1);
	return meanEn;
}

#endif