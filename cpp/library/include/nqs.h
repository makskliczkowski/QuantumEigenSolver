#pragma once
/***************************************
* Defines the generic NQS Solver class. 
* Allows for later inhertiance
* for a fine model specialization.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#ifndef NQS_H
#	define NQS_H

// #################################
#ifndef HAMIL_H					// #
#	include "hamil.h"			// #
#endif // !HAMIL_H				// #
// #################################

// #################################
#ifndef ML_H					// #
#	include "ml.h"				// #
#endif // !ML_H					// #
// #################################

// ######### NQS TYPES #############
enum NQSTYPES					// #
{								// #
	RBM							// #
};								// #
								// #
BEGIN_ENUM(NQSTYPES)			// #
{								// #
	DECL_ENUM_ELEMENT(RBM)		// #
}								// #
END_ENUM(NQSTYPES)				// #
// #################################

#include <thread>

// ##########################################################################################################################################
// ##########################################################################################################################################
// ########################################################## N Q S   S O L V E R ###########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief General Neural Network Quantum States eigensolver class
*/
template <typename _Ht,
	uint _spinModes			= 2, 
	double _discreteValue	= Operators::_SPIN_RBM,
	typename _T				= _Ht, 
	class _stateType		= double>
class NQS 
{
public:
	using NQSS							=		arma::Col<_stateType>;		// Quantum state -> for occupation representation purpose
	using NQSB							=		arma::Col<_T>;				// for parameter representation purpose
	using NQSW							=		arma::Mat<_T>;				// for weights representation purpose
	const uint spinModes_				=		_spinModes;					// number of spin modes - 2 for hardcore bosons, 4 for fermions
	const double discVal_				=		_discreteValue;
	std::shared_ptr<Hamiltonian<_Ht, _spinModes>> H_;						// pointer to the Hamiltonian

protected:
	std::string info_					=		"General NQS";
	pBar pBar_;																// for printing out the progress

	uint threadNum_						=		1;							// number of threads that works on this
	uint fullSize_						=		1;							// full number of the parameters (for memory purpose)
	uint batch_							=		1;							// size of the batch for optimization
	uint nVis_							=		1;							// number of visible neurons (input variables)
	uint nSites_						=		1;							// number of lattice sites
	double lr_							=		0.0;						// specific learning rate

	// Hilbert space info
	u64 Nh_								=		1;

	// Random number generator
	randomGen ran_;															// consistent quick random number generator

	// --------------------- O P T I M I Z E R S --------------------

	// create thread pool
#if true || defined NQS_USE_CPU && not defined NQS_USE_OMP && not defined _DEBUG
	v_1d<std::thread> threads_;
	// mutex and condition variable for the speedup
	std::mutex mutex;
	std::condition_variable cv;
	v_1d<bool> flagThreadKill_;
	v_1d<bool> flagThreadRun_;
	v_1d<_T> kernelValues_;
#endif

	// ----------------------- T R A I N I N G ----------------------
	// for flipping
	uint nFlip_							=		1;
	v_1d<uint> flipPlaces_;													// stores flip spots to be flipped during one sampling step
	v_1d<_stateType> flipVals_;												// stores values before the flip to be used for the gradients
	
	// current state
	u64 curState						=		0;							// currently processed state
	NQSS curVec;															// currebtly processed state vector for convenience
	// temporary placeholders
	NQSS tmpVec;
	v_1d<NQSS> tmpVecs;
	// derivatives storage
	NQSW Derivatives_;														// store the variational derivatives F_k

protected:
	// ----------------------- S T A R T E R S ----------------------
	virtual void setInfo()							=						0;
	virtual void allocate()							=						0;
	
	// -------------------- P R O B A B I L I T Y -------------------
	
	// The probability of accepting new state s' from state s
	// comming from the Metropolis - Hastings algorithm is of the 
	// form: 
	// $A(s', s) = min [1, \frac{P(s')}{P(s)}\frac{r(s'->s)}{r(s->s')}],$ 
	// where r(s'->s) is chosen such that the ratio is 1.0. 
	// For example one can choose 
	// $ r(s'->s) = \frac{|<s'|H|s>|(1-\delta _{s,s'})}{\sum _{s!=s'} |<s'|H|s>| }.$
	// Then, the acceptance ratio decides on accepting the new parameter. 
	// The probability:
	// $ P(s) = \frac{||<\psi|s>||^2}{<\psi|\psi>}.$ 
	// Therefore, in the ratio the lower part dissapears and one 
	// needs to calculate the:
	// $ P(flip) = || <\psi|s'> / <\psi|s> ||^2. $
	// The value of <\psi | s> can be stored effectively 
	// as it's updates are easy. Effectively, one needs to calculate
	// only the new <\psi|s'> given from the new state s'.

	virtual auto pRatio()							->_T =					0;
	virtual auto pRatio(uint fP, float fV)			->_T =					0;
	virtual auto pRatio(std::initializer_list<int>,
					std::initializer_list<double>)	->_T =					0;
	// for probability ratio with updating weights
	virtual auto pRatio(uint nFlips)				->_T =					0;
	virtual auto pRatio(const NQSS& _v)				->_T =					0;
	// for probability ratio with updating weights according to a new vector set
	virtual auto pRatio(const NQSS& _v1,
						const NQSS& _v2)			->_T =					0;
	virtual auto pRatioChange(_T _val,
							  u64 _state,
							  uint _vid)			-> _T =					0;

	// --------------------- S E T T E R S -----------------------
	// column vector state
	virtual void setState(NQSS _st, bool _set)		=						0;
	virtual void setState(NQSS _st)					=						0;
	// integer state
	virtual void setState(u64 _st, bool _set)		=						0;
	virtual void setState(u64 _st)					=						0;
	
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
#ifndef NQS_USE_OMP
	virtual auto locEnKernel(uint _start, uint _end, uint _threadNum)		-> void;
#endif

	// -----------------------------------------------------------
public:
	// --------------------- S E T T E R S -----------------------
	virtual	void init()								=						0; 
	virtual	void update()							=						{}; 
	virtual void setRandomState(bool _upd = true)							{ this->setState(this->ran_.template randomInt<u64>(0, this->Nh_), _upd);			};
	
	// ----------------------- F L I P S -------------------------
	virtual void chooseRandomFlips();

	// apply flips to the temporary vector or the current vector according the template
	template<bool _USE_TEMPORARY>
	virtual void applyFlips(typename std::enable_if<_USE_TEMPORARY, int>::type _tmp = 0)
	{ 
		for (auto& i : flipPlaces_) flip(tmpVec, i, 0, discVal_);							
	};
	template<bool _USE_TEMPORARY>
	virtual void applyFlips(typename std::enable_if<!_USE_TEMPORARY, int>::type _tmp = 0)
	{ 
		for (auto& i : flipPlaces_) flip(curVec, i, 0, discVal_);							
	};
	// unapply flips of the temporary vector or the current vector according the template
	template<bool _TMP>
	virtual void unapplyFlips()												{ this->applyFlips<_TMP>;															};


	// --------------------- G E T T E R S -----------------------
	auto getInfo()								    const -> std::string	{ return this->info_;																};
	auto getHilbertSize()							const -> u64			{ return this->Nh_;																	};
	auto getNvis()									const -> uint			{ return this->nVis_;																};
#ifdef NQS_USE_CPU
	auto getThreadID()								const -> uint;
#endif

	// --------------------- S A M P L I N G ---------------------
	virtual void blockSample(uint _bSize,
							 u64 _start, 
							 bool _therm = false);
	
	virtual arma::Col<_T> train(uint mcSteps,					// number of Monte Carlo Steps
								uint nThrm,						// number of mcSteps to thermalize
								uint nBlck,						// number of such blocks for one average step
								uint bSize,						// for killing correlations
								uint nFlip			= 1,		// number of flips to set
								bool quiet			= false,	// shall talk?
								clk::time_point _t	= NOW,		// time!
								uint progPrc		= 25);

	virtual arma::Col<_T> collect(	uint nSam, 
									uint nThrm, 
									uint nBlck, 
									uint bSize, 
									uint nFlip = 1)	{};

	// ----------------------- F I N A L E -----------------------
	virtual auto ansatz(const NQSS& _in)			const ->_T =			0;


	// -------------------- C O N S T R U C T --------------------

public:
	virtual ~NQS();
	NQS(std::shared_ptr<Hamiltonian<_Ht>> _H, double _lr = 1e-2, uint _threadNum = 1);
	NQS()											=						default;
};

// ##############################################################################################################################################

/*
* @brief Returns the thread ID of the currently working thread
* @returns the thread ID integer modulated by this thread number
*/
#ifdef NQS_USE_CPU
template<typename _Ht, uint _spinModes, double _discreteValue, typename _T, class _stateType>
uint NQS<_Ht, _spinModes, _discreteValue, _T, _stateType>::getThreadID() const
{
#ifndef _DEBUG
	if (this->threadNum_ == 1)
		return 0;
	else
#	ifdef NQS_USE_OMP
		return omp_get_thread_num() % this->threadNum_;
#	else
		return std::this_thread::get_id();
#	endif
#else
	return 0;
#endif
};
#endif

// ##########################################################################################################################################

template<typename _Ht, uint _spinModes, double _discreteValue, typename _T, class _stateType>
inline NQS<_Ht, _spinModes, _discreteValue, _T, _stateType>::~NQS()
{
	LOGINFO(LOG_TYPES::TRACE, "Destroying the general NQS class", 2);
}

// ##########################################################################################################################################

/*
* @brief General constructor of the NQS solver
* @param _H Hamiltonian to be used for correct data sampling
* @param _lr learning rate to be used for the training
* @param _threadNum thread number to be used for the solver - available on setting corresponding thread pragma
*/
template<typename _Ht, uint _spinModes, double _discreteValue, typename _T, class _stateType>
inline NQS<_Ht, _spinModes, _discreteValue, _T, _stateType>::NQS(std::shared_ptr<Hamiltonian<_Ht>> _H, double _lr, uint _threadNum)
	: H_(_H), lr_(_lr)
{
	// set the visible layer (for hardcore-bosons we have the same number as sites but fermions introduce twice the complication)
	this->nVis_					=			_H->getNs() * (this->spinModes_ % 2);
	this->nSites_				=			_H->getNs();
	// check the Hilbert space
	this->Nh_					=			_H->getHilbertSize();
	// set the random number generator
	this->ran_					=			_H->ran_;

	// set threads
	{
#ifdef _DEBUG
		this->threadNum_		=			1;
#else
		this->threadNum_		=			std::min(_threadNum, this->nSites_);
#endif
	// Use threads for all consecutive parallel regions
#if true || defined NQS_USE_CPU && not defined _DEBUG
#	ifdef NQS_USE_OMP
		omp_set_num_threads(this->threadNum_);   
#	else
		this->threads_.reserve(this->threadNum_);
		this->flagThreadKill_	=			v_1d<bool>(this->threadNum_, false);
		this->flagThreadRun_	=			v_1d<bool>(this->threadNum_, false);
		
		// calculate how many sites goes to one thread
		uint _siteStep			=			std::ceil(this->Ns / 1.0 / this->threadNum_);

		// start the threads that calculate the energy
		for (auto i = 0; i < this->threadNum_; i++)
		{
			this->threads_.emplace_back(std::thread(&NQS<_Ht, _spinModes, 
													_discreteValue, _T, _stateType>::locEnKernel, 
													this, 
													i * _siteStep,
													std::min((i+1) * _siteStep, this->nSites_),
													i));
		}
#	endif
#endif
	}
	LOGINFO(LOG_TYPES::TRACE, "Constructed the general NQS class", 2);
};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ F L I P P I N G #############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Randomly flip the discrete variables at chosen flip places. Sets the random flips to the vector already saved
*/
template<typename _Ht, uint _spinModes, double _discreteValue, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _discreteValue, _T, _stateType>::chooseRandomFlips()
{
	for (auto i = 0; i < this->flipPlaces_.size(); ++i) 
	{
		// choose the flip place of the vector
		this->flipPlaces_[i]	=	this->ran_.template randomInt<uint>(0, this->nVis_);
		// save the element of a vector before the flip
		this->flipVals_[i]		=	this->tmpVec(this->flipPlaces_[i]);
	}
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ S A M P L I N G #############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Block updates the current state according to Metropolis-Hastings algorithm. The block size is chosen so that
* it gives the given statistics of the averages found.
* @param _bSize the size of the correlation block; how many flips one needs to do in order to get rid of the MC correlations - bin size 
* @param _start the state to start from
* @param _nFlip number of flips in a single step
*/
template<typename _Ht, uint _spinModes, double _discreteValue, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _discreteValue, _T, _stateType>::blockSample(uint _bSize, u64 _start, bool _therm)
{
	// check whether we should set a state again or thermalize the whole process
	if (_start != this->curState || _therm)
		this->setState(_start, _therm);

	// set the temporary state - the vectors are usefull so set them accordingly
	this->tmpVec = this->curVec;

	// go through each block step
	for (uint bStep = 0; bStep < _bSize; bStep++)
	{
		// set the random flip sites
		this->chooseRandomFlips();

		// flip the vector - use temporary vector tmpVec
		this->applyFlips<true>();

		// check the probability (choose to use the iterative update of presaved weights or calculate ratio from scratch)
#ifndef NQS_ANGLES_UPD
		double proba = std::abs(this->pRatio(this->curVec, this->tmpVec));
#else
		double proba = std::abs(this->pRatio(this->nFlip_));
#endif
		// we need to take into account the probability comming from the ratio of states (after and before the flip)
		if (this->ran_.template random<float>() <= proba * proba)
		{
			// update current state and vector when the flip has been accepted
			this->applyFlips<false>();
			// update angles if needed
			this->update();
		}
		// set the vector back to normal
		else
			this->unapplyFlips<true>();
	}
	this->curState = BASE_TO_INT<u64>(this->curVec, discVal_);
#ifndef NQS_ANGLES_UPD
	this->setState(this->curState, true);
#endif
}

// ##########################################################################################################################################

/*
* @brief Calculate the local energy depending on the given Hamiltonian - kernel with OpenMP
*/
template<typename _Ht, uint _spinModes, double _discreteValue, typename _T, class _stateType>
inline _T NQS<_Ht, _spinModes, _discreteValue, _T, _stateType>::locEnKernel()
{
#if defined _DEBUG || defined NQS_USE_OMP
	double energyR	= 0.0;
	double energyI	= 0.0;

#	ifndef _DEBUG
	#pragma omp parallel for reduction(+ : energyR, energyI) num_threads(this->threadNum_)
#	endif
	for (int site = 0; site < this->nSites_; site++) 
	{
		auto energy	=	this->H_->locEnergy(	this->curState, 
												site, 
												std::bind(&NQS<_Ht, _spinModes,
														_discreteValue, _T, _stateType>::chKernel, 
												this,
												std::placeholders::_1,
												std::placeholders::_2));
		energyR		+=	algebra::real(energy);
		energyI		+=	algebra::imag(energy);
	}
	return toType<_T>(energyR, energyI);
#else
	_T energy		= 0.0;
	// run all threads
	for (int _thread = 0; _thread < this->threadNum_; _thread++)
	{
		this->flagThreadRun_[_thread] = true;
		cv.notify_one();
	}
	// wait for all threads
	for (int _thread = 0; _thread < this->threadNum_; _thread++)
	{
		while (this->flagThreadRun_[_thread]);
		energy += kernelValues_[_thread];
	}
	return energy;
#endif
}

#ifndef NQS_USE_OMP
/*
* @brief Allows to run a thread pool based on the condition that all threads wait for a mutex to further run the program
* @param _start starting site for a given thread
* @param _end ending site for a given thread
* @param _threadNum number of the thread currently run
*/
template<typename _Ht, uint _spinModes, double _discreteValue, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _discreteValue, _T, _stateType>::locEnKernel(uint _start, uint _end, uint _threadNum)
{
	// does not go in if the simulation is finished
	while (!flagThreadKill_[_threadNum])
	{
		// wait for the lock to end
		{
			// aquire mutex lock as required by condition variable
			unique_lock<mutex> lock(mutex);									
			// thread will suspend here and release the lock if the expression does not return true
			cv.wait(lock, [this] {return flagThreadRun_[_threadNum]; });	
		}
		
		this->kernelValues_[_threadNum] = 0.0;
		for (auto site = _start; site < _end; ++site)
		{
			this->kernelValues_[_threadNum] +=   this->H_->locEnergy(this->curState,
																	 site, 
																	 std::bind(&NQS<_Ht, _spinModes,
																				_discreteValue, _T, _stateType>::chKernel, 
																	 this,
																	 std::placeholders::_1,
																	 std::placeholders::_2));
		}
		flagThreadRun_[_threadNum] = false;
	}
}
#endif

// ##########################################################################################################################################

template<typename _Ht, uint _spinModes, double _discreteValue, typename _T, class _stateType>
inline arma::Col<_T> NQS<_Ht, _spinModes, _discreteValue, _T, _stateType>::train(uint mcSteps,
																				 uint nThrm,
																				 uint nBlck,
																				 uint bSize,
																				 uint nFlip,
																				 bool quiet,
																				 clk::time_point _t,
																				 uint progPrc)
{
	std::string outstr	= "";
	
	// set the info about training
	strSeparatedP(outstr, '\n\t', 2,
								VEQV("Monte Carlo Steps", mcSteps),
								VEQV("Thermalization Steps", nThrm),
								VEQV("Block Number", nBlck),
								VEQV("Size of the single block", bSize),
								VEQV("Number of flips taken at each step", nFlip)
	);
	LOGINFOG("Train: " + outstr, LOG_TYPES::TRACE, 1);

	// make the pbar!
	this->pBar_			= pBar(progPrc, mcSteps);

	// check if the number of thermal samples is not bigger than the MC steps
	const int _stps		= nBlck - nThrm;
	if (_stps < 0)
	{
		LOGINFOG("Number of steps is too small - thermalisaton too high!", LOG_TYPES::ERROR, 0);
		throw std::runtime_error("Number of steps is too small - thermalisaton too high!");
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