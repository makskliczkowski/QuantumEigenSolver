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

// ###############################
#ifndef HAMIL_H				   // #
#	include "hamil.h"			   // #
#endif // !HAMIL_H			   // #
// ###############################

// ###############################
#ifndef ML_H					   // #
#	include "ml.h"				   // #
#endif // !ML_H				   // #
// ###############################

// ######### NQS TYPES ###########
enum NQSTYPES						// #
{										// #
	RBM								// #
};										// #
										// #
BEGIN_ENUM(NQSTYPES)				// #
{										// #
	DECL_ENUM_ELEMENT(RBM)		// #
}										// #
END_ENUM(NQSTYPES)				// #
// ###############################

#include <thread>

// ##########################################################################################################################################
// ##########################################################################################################################################
// ########################################################## N Q S   S O L V E R ###########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief General Neural Network Quantum States eigensolver class
* Contains all the necessary methods to override
*/
template <typename _Ht,
		  uint _spinModes				= 2, 
		  typename _T					= std::complex<double>, 
		  class _stateType			= double>
class NQS 
{
public:
	using NQSS							=		arma::Col<_stateType>;		// Quantum state -> for occupation representation purpose
	using NQSB							=		arma::Col<_T>;					// for parameter representation purpose
	using NQSW							=		arma::Mat<_T>;					// for weights representation purpose

protected:
	// for the Hamiltonian information
	const uint spinModes_			=		_spinModes;						// number of spin modes - 2 for hardcore bosons, 4 for fermions
	const double discVal_			=		Operators::_SPIN_RBM;
	std::shared_ptr<Hamiltonian<_Ht, _spinModes>> H_;					// pointer to the Hamiltonian

protected:
	// information about the NQS method
	std::string info_					=		"General NQS";
	pBar pBar_;																		// for printing out the progress

	// simulation specific
	uint batch_							=		1;									// size of the batch for optimization
	double lr_							=		0.0;								// specific learning rate
	
	// architecture specific
	uint nVis_							=		1;									// number of visible neurons (input variables)
	uint nSites_						=		1;									// number of lattice sites
	uint fullSize_						=		1;									// full number of the parameters (for memory purpose)

	// Hilbert space info
	u64 Nh_								=		1;

	// Random number generator
	randomGen ran_;																// consistent quick random number generator

	/* ----------------------------------------------------------- */

	// ---------------------- T H R E A D I N G ---------------------

	uint threadNum_					=		1;									// number of threads that works on this
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

	/* ----------------------------------------------------------- */

	// ----------------------- T R A I N I N G ----------------------
	// for flipping
	uint nFlip_							=		1;
	v_1d<uint> flipPlaces_;														// stores flip spots to be flipped during one sampling step
	v_1d<_stateType> flipVals_;												// stores values before the flip to be used for the gradients
	
	// current state
	u64 curState_						=		0;									// currently processed state
	NQSS curVec_;																	// currently processed state vector for convenience
	
	// temporary placeholders
	v_1d<NQSS> tmpVecs_;
	u64 tmpState_						=		0;
	NQSS tmpVec_;
	
	// ------------------------ W E I G H T S -----------------------

	// derivatives storage
	NQSW derivatives_;															// store the variational derivatives F_k
#ifdef NQS_USESR
	NQSW S_;																			// positive semi-definite covariance matrix
#endif
	NQSB F_;																			// forces

protected:
	// ----------------------- S T A R T E R S ----------------------
	virtual void setInfo()			=		0;
	virtual void allocate();
	
	/* ----------------------------------------------------------- */

	// ------------------------ S E T T E R S -----------------------
	// column vector state
	virtual void setState(const NQSS& _st, bool _set)	=				0;
	virtual void setState(const NQSS& _st);
	// integer state
	virtual void setState(u64 _st, bool _set)				=				0;
	virtual void setState(u64 _st);
	
	/* ------------------------------------------------------------ */
	// -------------------- P R O B A B I L I T Y --------------------
	
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

	// ratio when 1 flip is used and flips are stored inside (flipPlaces_ & flipVals_)
	virtual auto pRatio()									-> _T				{ return this->pRatio(this->flipPlaces_[0], this->flipVals_[0]);	};
	// ratio when 1 flip is used and flips are provided to the function (fP, fV)
	virtual auto pRatio(uint fP, float fV)				->_T =			0;
	// ratio when multiple flips are used, provided the number of flips taken
	virtual auto pRatio(uint nFlips)						->_T =			0;
	// ratio when a new vector is provided (after flip) - (uses current vector as well)
	virtual auto pRatio(const NQSS& _v)					->_T				{ return this->pRatio(this->curVec_, _v);									};			
	// ratio when a new vector is provided (after flip) - (uses two different vectors)
	virtual auto pRatio(const NQSS& _v1,
							  const NQSS& _v2)				->_T =			0;
	// ratio when exact points are provided (used for the Hamiltonian)
	virtual auto pRatio(std::initializer_list<int> fP, 
						std::initializer_list<double> fV)->_T =			0;
	/* ------------------------------------------------------------ */

	// ----------------------- W E I G H T S -------------------------
#ifdef NQS_ANGLES_UPD
	virtual void update(uint nFlips = 1)									{};
	virtual void update(const NQSS& v, uint nFlips = 1)				{};
#endif
	virtual void updateWeights()							=					0;
	virtual bool saveWeights(std::string _path, std::string _file);
	virtual bool setWeights(std::string _path, std::string _file);

	// --------------------- T R A I N   E T C -----------------------
	virtual void grad(const NQSS& _v, uint _plc)		=					0;
	virtual void gradFinal(const NQSB& _energies);
	// stochastic reconfiguration
#ifdef NQS_USESR
	virtual void gradSR(uint step = 0);
#endif
	 // regularization
#ifdef NQS_SREG 
	double covMatrixRegCurrent = 1e-4;										// parameter for regularisation, changes with Monte Carlo steps
	double covMatrixRegStart	= 1e-4;										// starting parameter for regularisation
	virtual void covMatrixReg();
#endif
	
	/* ------------------------------------------------------------ */

	// ------------------------ E N E R G Y --------------------------
	auto pKernel(	std::initializer_list<int>  fP,
						std::initializer_list<double> fV)	-> _T			{ return this->pRatio(fP, fV); };
	_T locEnKernel();
#ifndef NQS_USE_OMP
	virtual void locEnKernel(uint _start, uint _end, uint _threadNum);
#endif

	/* ------------------------------------------------------------ */

public:
	// ------------------------ S E T T E R S ------------------------
	virtual void init()										=					0; 
	virtual void setRandomState(bool _upd = true)						{ this->setState(this->ran_.template randomInt<u64>(0, this->Nh_), _upd);	};
	
	// -------------------------- F L I P S --------------------------
	virtual void chooseRandomFlips();

	// apply flips to the temporary vector or the current vector according the template
	virtual void applyFlipsT()													{ for (auto& i : flipPlaces_) flip(tmpVec_, i, 0, discVal_);					};
	virtual void applyFlipsC()													{ for (auto& i : flipPlaces_) flip(curVec_, i, 0, discVal_);					};
	// unapply flips of the temporary vector or the current vector according the template
	virtual void unapplyFlipsT()												{ this->applyFlipsT();																		};
	virtual void unapplyFlipsC()												{ this->applyFlipsC();																		};

	/* ------------------------------------------------------------ */

	// ------------------------ G E T T E R S ------------------------
	auto getInfo()										const -> std::string	{ return this->info_;					};
	auto getNvis()										const -> uint			{ return this->nVis_;					};
	auto getF()											const -> NQSB			{ return this->F_;						};
	auto getCovarianceMat()							const -> NQSW			{ return this->S_;						};	
	// Hilbert
	auto getHilbertSize()							const -> u64			{ return this->Nh_;						};
	// Hamiltonian
	auto getHamiltonianInfo()						const -> std::string { return this->H_->getInfo();			};
	auto getHamiltonianEigVal(u64 _idx)			const -> double		{ return this->H_->getEigVal(_idx); };
	auto getHamiltonian()							const -> std::shared_ptr<Hamiltonian<_Ht>> { return this->H_; };

#ifdef NQS_USE_CPU
	auto getThreadID()								const -> uint;
#endif

	// ----------------------- S A M P L I N G -----------------------
	virtual void blockSample(	uint _bSize,
										u64 _start, 
										bool _therm = false);
	
	virtual arma::Col<_T> train(uint mcSteps,								// number of Monte Carlo Steps
										 uint nThrm,								// number of mcSteps to thermalize
										 uint nBlck,								// number of such blocks for one average step
										 uint bSize,								// for killing correlations
										 std::string dir,							// saving directory
										 uint nFlip			= 1,					// number of flips to set
										 bool quiet			= false,				// shall talk?
										 clk::time_point _t	= NOW,			// time!
										 uint progPrc		= 25);

	virtual arma::Col<_T> collect(uint nSam, 
											uint nThrm, 
											uint nBlck, 
											uint bSize, 
											uint nFlip = 1)	{};

	// ----------------------- F I N A L E -----------------------
	virtual auto ansatz(const NQSS& _in)			const ->_T =			0;


	// -------------------- C O N S T R U C T --------------------

public:
	virtual ~NQS();
	NQS(std::shared_ptr<Hamiltonian<_Ht>>& _H, double _lr = 1e-2, uint _threadNum = 1);
	NQS()											=						default;
};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# W E I G H T S ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Attempting to save the weights as a vector to a given filename.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @returns whether the save has been successful
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline bool NQS<_Ht, _spinModes, _T, _stateType>::saveWeights(std::string _path, std::string _file)
{
	LOGINFO(LOG_TYPES::INFO, "Saving the checkpoint configuration to", 2);
	LOGINFO(LOG_TYPES::INFO, _path + _file, 3);
	createDir(_path);
	bool _isSaved	= false;
#ifdef HAS_CXX20
	if (_file.ends_with(".h5"))
#else
	if (endsWith(_file, ".h5"))
#endif
		_isSaved	=	this->F_.save(arma::hdf5_name(_path + _file, "weights"));
#ifdef HAS_CXX20
	else if (_file.ends_with(".bin"))
#else
	else if (endsWith(_file, ".bin"))
#endif
		_isSaved	=	this->F_.save(_path + _file);
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	else if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
#endif
		_isSaved	=	this->F_.save(_path + _file, arma::arma_ascii);

	// if not saved properly
	if (!_isSaved && (_file != "weights.h5"))
	{
		LOGINFO(LOG_TYPES::ERROR, "Couldn't save the weights to the given path.", 3);
		LOGINFO(LOG_TYPES::ERROR, "Saving to default... ", 3);
		return this->saveWeights(_path, "weights.h5");
	}
	return _isSaved;
}

/*
* @brief Load the weights from the file specified by the user.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @returns whether the load has been successful
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline bool NQS<_Ht, _spinModes, _T, _stateType>::setWeights(std::string _path, std::string _file)
{
	LOGINFO(LOG_TYPES::INFO, "Loading the checkpoint weights from", 2);
	LOGINFO(LOG_TYPES::INFO, _path + _file, 3);
#ifdef HAS_CXX20
	if (_file.ends_with(".h5"))
#else
	if (endsWith(_file, ".h5"))
#endif
	{
		this->F_.load(arma::hdf5_name(_path + _file, "weights"));
		return true;
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".bin"))
#else
	else if (endsWith(_file, ".bin"))
#endif
	{
		this->F_.load(_path + _file);
		return true;
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	else if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
#endif
	{
		this->F_.load(_path + _file, arma::arma_ascii);
		return true;
	}
	//throw std::runtime_error("Couldn't read the file: " + _path + _file);
	return false;
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ########################################################### T H R E A D I N G ############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Returns the thread ID of the currently working thread
* @returns the thread ID integer modulated by this thread number
*/
#ifdef NQS_USE_CPU
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
uint NQS<_Ht, _spinModes, _T, _stateType>::getThreadID() const
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
// ##########################################################################################################################################
// ########################################################### C O N S T R U C T ############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Allocates the main gradient parameters and the temporary and current vectors.
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _T, _stateType>::allocate()
{
	// allocate gradients
	this->F_.resize(this->fullSize_);
#ifdef NQS_USESR
	this->S_.resize(this->fullSize_, this->fullSize_);
#endif
	this->curVec_ = arma::ones(this->nVis_);
	this->tmpVec_ = arma::ones(this->nVis_);
}

// ##########################################################################################################################################

/*
* @brief General destructor of the model.
* Waits on the threads if multithreading is run!
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline NQS<_Ht, _spinModes, _T, _stateType>::~NQS()
{
	LOGINFO(LOG_TYPES::TRACE, "Destroying the general NQS class", 2);
#if not defined NQS_USE_OMP && not defined _DEBUG
	for (int _thread = 0; _thread < this->threadNum_; _thread++)
	{
		flagThreadKill_[_thread] = 1;
		threads_[_thread].join();
	}
#endif
}

// ##########################################################################################################################################

/*
* @brief General constructor of the NQS solver
* @param _H Hamiltonian to be used for correct data sampling
* @param _lr learning rate to be used for the training
* @param _threadNum thread number to be used for the solver - available on setting corresponding thread pragma
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline NQS<_Ht, _spinModes, _T, _stateType>::NQS(std::shared_ptr<Hamiltonian<_Ht>>& _H, double _lr, uint _threadNum)
	: H_(_H), lr_(_lr)
{
	// set the visible layer (for hardcore-bosons we have the same number as sites but fermions introduce twice the complication)
	this->nVis_					=			_H->getNs() * (this->spinModes_ / 2);
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
#if defined NQS_USE_CPU && not defined _DEBUG
#	ifdef NQS_USE_OMP
		omp_set_num_threads(this->threadNum_);   
#	else
		this->threads_.reserve(this->threadNum_);
		this->flagThreadKill_	=			v_1d<bool>(this->threadNum_, false);
		this->flagThreadRun_	=			v_1d<bool>(this->threadNum_, false);
		
		// calculate how many sites goes to one thread
		uint _siteStep			=			std::ceil(this->Ns / 1.0 / this->threadNum_);

		// start the threads that calculate the energy with the local energy kernel function
		// this function waits for the specific energy calculation to be ready on each thread
		// this is handled through "flagThreadRun_" member
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
* @brief Randomly flip the discrete variables at chosen flip places. Sets the random flips to the vector already saved.
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _T, _stateType>::chooseRandomFlips()
{
	// go through the vector elements
	for (auto i = 0; i < this->flipPlaces_.size(); ++i) 
	{
		// choose the flip place of the vector
		this->flipPlaces_[i]	=	this->ran_.template randomInt<uint>(0, this->nVis_);
		// save the element of a vector before the flip
		this->flipVals_[i]		=	this->tmpVec_(this->flipPlaces_[i]);
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
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _T, _stateType>::blockSample(uint _bSize, u64 _start, bool _therm)
{
	// check whether we should set a state again or thermalize the whole process
	if (_start != this->curState_ || _therm)
		this->setState(_start, _therm);

	// set the temporary state - the vectors are usefull so set them accordingly
	this->tmpVec_ = this->curVec_;

	// go through each block step
	for (uint bStep = 0; bStep < _bSize; bStep++)
	{
		// set the random flip sites
		this->chooseRandomFlips();

		// flip the vector - use temporary vector tmpVec
		this->applyFlipsT();

		// check the probability (choose to use the iterative update of presaved weights or calculate ratio from scratch)
#ifndef NQS_ANGLES_UPD
		double proba = std::abs(this->pRatio(this->curVec_, this->tmpVec_));
#else
		double proba = std::abs(this->pRatio(this->nFlip_));
#endif
		// we need to take into account the probability comming from the ratio of states (after and before the flip)
		if (this->ran_.template random<float>() <= proba * proba)
		{
			// update current state and vector when the flip has been accepted
			this->applyFlipsC();
			// update angles if needed
			this->update();
		}
		// set the vector back to normal
		else
			this->unapplyFlipsT();
	}
	this->curState_ = BASE_TO_INT<u64>(this->curVec_, discVal_);
#ifndef NQS_ANGLES_UPD
	this->setState(this->curState_, true);
#endif
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ########################################################## L O C   E N E R G Y ###########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Calculate the local energy depending on the given Hamiltonian - kernel with OpenMP is used
* when the omp pragma NQS_USE_OMP is set or multithreading is not used, otherwise threadpool is used
* to calculate the local energies.
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline _T NQS<_Ht, _spinModes, _T, _stateType>::locEnKernel()
{
#if defined _DEBUG || defined NQS_USE_OMP
	double energyR	= 0.0;
	double energyI	= 0.0;

#	ifndef _DEBUG
	#pragma omp parallel for reduction(+ : energyR, energyI) num_threads(this->threadNum_)
#	endif
	for (int site = 0; site < this->nSites_; site++) 
	{
		auto energy	=	this->H_->locEnergy(	this->curState_, 
														site, 
														std::bind(&NQS<_Ht, _spinModes, _T, _stateType>::pKernel,
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
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _T, _stateType>::locEnKernel(uint _start, uint _end, uint _threadNum)
{
	// does not go in if the simulation is finished
	while (!flagThreadKill_[_threadNum])
	{
		// wait for the lock to end
		{
			// aquire mutex lock as required by condition variable
			std::unique_lock<std::mutex> lock(mutex);									
			// thread will suspend here and release the lock if the expression does not return true
			cv.wait(lock, [this, _threadNum] {return flagThreadRun_[_threadNum]; });	
		}
		
		this->kernelValues_[_threadNum] = 0.0;
		for (auto site = _start; site < _end; ++site)
		{
			this->kernelValues_[_threadNum] +=   this->H_->locEnergy(this->curState_,
																	 site, 
																	 std::bind(&NQS<_Ht, _spinModes, _T, _stateType>::pKernel,
																	 this,
																	 std::placeholders::_1,
																	 std::placeholders::_2));
		}
		flagThreadRun_[_threadNum] = false;
	}
}
#endif

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# S E T T E R S ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Updates the current state processed by the NQS.
* @param _st Column vector to be set as a new state
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _T, _stateType>::setState(const NQSS& _st)
{
#ifdef _DEBUG
	// check the previous value to be sure
	auto _val		= this->discVal_;
	for (auto& _valin : _st)
	{
		if (_valin != 0)
		{
			_val	= _valin;
			break;
		}
	}
	this->curVec_	= _st / _val * this->discVal_;
#else
	this->curVec_	= _st;
#endif
	this->curState_ = BASE_TO_INT<u64>(_st, this->discVal_);
}

/*
* @brief Updates the current state processed by the NQS.
* @param _st Integer vector value to be set as a new state
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _T, _stateType>::setState(u64 _st)
{
	this->curState_ = _st;
	INT_TO_BASE(_st, this->curVec_, this->discVal_);
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ G R A D I E N T #############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

#ifdef NQS_USESR
/*
* @brief Calculates the update parameters for weights to be updated in a form:
* weights[new] <- weights[old] - lr * S^{-1} * F --> second order optimization method
* @param step current step of updating - for the regularization purpose
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _T, _stateType>::gradSR(uint step)
{
	// update the flattened vector for the gradients
#	ifdef NQS_SREG 
	// regularize the covariance matrix before inverting it
	this->covMatrixReg();
#	endif
#	ifdef NQS_PINV
	// calculate the pseudoinverse
	this->F_ = this->lr_ * (arma::pinv(this->S_, NQS_PINV) * this->F_);
	return;
#	else 
	// solve normally
	this->F_ = this->lr_ * arma::solve(this->S_, this->F_);
#	endif 
}
#endif

/*
* @brief	Calculate the final step for the weights to be updated.
* First of all, it calculates the generalized forces
* 
* @warning Uses the forces vector (member F_) to store the forces that update the weights
* 
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _T, _stateType>::gradFinal(const NQSB& _energies)
{
	// calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}>
	this->F_ = arma::cov(arma::conj(this->derivatives_), _energies, 1);
#ifdef NQS_USESR
	this->S_ = arma::cov(arma::conj(this->derivatives_), this->derivatives_, 1);
	
	// update model
	this->gradSR(0);
	
	// use the regularization
	#ifdef NQS_SREG
	this->covMatrixRegCurrent *= this->covMatrixRegStart;
	#endif // S_REGULAR
#else
	// standard updater with the gradient only!
	this->F_ *= this->lr_;
#endif
}

// ##########################################################################################################################################

#ifdef NQS_SREG
/*
! TODO 
* @brief The inverse of the covariance matrix is poorly defined at the begining of training. 
* Use regularization to fix that issue.
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline void NQS<_Ht, _spinModes, _T, _stateType>::covMatrixReg()
{
	this->S_.diag() += this->covMatrixRegCurrent;
}
#endif

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ T R A I N I N G #############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Perform single training of the NQS.
* @param mcSteps Monte Carlo steps to be used in the training
* @param nThrm Monte Carlo steps spent for thermalizing the system
* @param nBlck	number of blocks to be used for calculating the covariance matrix
* @param bSize number of update steps within a single block
* @param nFlip number of flips for a single update of a state
* @param quiet wanna talk?
* @param _t timepoint for timestamping
* @param progPrc progress percentage to be displayed
*/
template<typename _Ht, uint _spinModes, typename _T, class _stateType>
inline arma::Col<_T> NQS<_Ht, _spinModes, _T, _stateType>::train(uint mcSteps,
																											uint nThrm,
																											uint nBlck,
																											uint bSize,
																											std::string dir,
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
								VEQV("Number of flips taken at each step", nFlip));
	LOGINFOG("Train: " + outstr, LOG_TYPES::TRACE, 1);

	// make the pbar!
	this->pBar_			= pBar(progPrc, mcSteps);

	// check if the number of thermal samples is not bigger than the MC steps
	const int _stps	= nBlck - nThrm;
	if (_stps < 0)
	{
		LOGINFOG("Number of steps is too small - thermalisaton too high!", LOG_TYPES::ERROR, 0);
		throw std::runtime_error("Number of steps is too small - thermalisaton too high!");
	};
	// set the derivatives matrix
#ifdef NQS_SREG
	this->covMatrixRegCurrent = this->covMatrixRegStart;
#endif
	this->derivatives_.resize(_stps, this->fullSize_);

	// save all average weights for covariance matrix
	arma::Col<_T> meanEn(mcSteps, arma::fill::zeros);
	// history of energies
	arma::Col<_T> En(_stps, arma::fill::zeros);
	// set the random state at the begining
	this->setRandomState();
	this->nFlip_		= nFlip;
	this->flipPlaces_.resize(nFlip_);
	this->flipVals_.resize(nFlip_);

	// go through the Monte Carlo steps
	for (uint i = 0; i < mcSteps; ++i)
	{
		// thermalize!
		this->blockSample(nThrm, this->curState_, true);

		// iterate blocks
		for (uint _taken = 0; _taken < _stps; ++_taken) {

			// sample them!
			this->blockSample(bSize, this->curState_, false);

			// calculate the gradient!
			this->grad(this->curVec_, _taken);

			// energy
			En(_taken) = this->locEnKernel();
		}
		// calculate the final update vector
		this->gradFinal(En);
		// finally, update the weights
		this->updateWeights();
		// save the mean energy
		meanEn(i) = arma::mean(En);

		// update the progress bar
		PROGRESS_UPD_Q(i, this->pBar_, "PROGRESS NQS", !quiet);
#ifdef NQS_SAVE_WEIGHTS
		if (i % this->pBar_.percentageSteps == 0) this->saveWeights(dir + NQS_SAVE_DIR, "weights.h5");
#endif
	}
	LOGINFO(_t, "NQS_EQ", 1);
	return meanEn;
}

#endif