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

//////////////////////////////////////////////////////////////////////////////////////////
#ifndef HAMIL_H
#	include "hamil.h"
#	include "NQS/nqs_operator.h"
#endif
//////////////////////////////////////////////////////////////////////////////////////////

// ######### NQS TYPES #############
enum NQSTYPES					// #
{								// #
	RBM_T,						// #
	RBMPP_T						// #
};								// #
								// #
BEGIN_ENUM(NQSTYPES)			// #
{								// #
	DECL_ENUM_ELEMENT(RBM),		// #
	DECL_ENUM_ELEMENT(RBMPP)	// #
}								// #
END_ENUM(NQSTYPES)				// #
// #################################

#include <future>
#include <functional>
#include <any>

#define NQS_PUBLIC_TYPES(_type, _stateType) public:	using NQSS = arma::Col<_stateType>;	using NQSB = arma::Col<_type>; using NQSW = arma::Mat<_type>;			
#define NQS_LOG_ERROR_SPIN_MODES LOG_ERROR("IMPLEMENT ME FOR THIS NUMBER OF SPIN MODES")

//////////////////////////////////////////////////////////////////////////////////////////
// Kernel for multithreading
#	if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP

/*
* @brief structure with condition variables for the NQS to perfom multithread operations
*/
template <typename _T>
struct CondVarKernel
{
	std::mutex mutex;
	std::condition_variable cv;
	std::atomic<bool> end_	= false;
	bool flagThreadKill_	= false;
	bool flagThreadRun_		= false;
	_T kernelValue_			= 0.0;
};

#	endif 
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! B A S E !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

/*
* @brief General Neural Network Quantum States eigensolver class - base
* Contains all the necessary methods to override
*/
template <uint _spinModes, 
		  typename _Ht,
		  typename _T					= _Ht, 
		  class _stateType				= double>
class NQS 
{
public:
	using NQSS							=		arma::Col<_stateType>;	// Quantum state -> for occupation representation purpose
	using NQSB							=		arma::Col<_T>;			// for parameter representation purpose
	using NQSW							=		arma::Mat<_T>;			// for weights representation purpose

protected:
	// for the Hamiltonian information
	const uint spinModes_				=		_spinModes;				// number of spin modes - 2 for hardcore bosons, 4 for fermions
	const double discVal_				=		Operators::_SPIN_RBM;
	std::shared_ptr<Hamiltonian<_Ht, _spinModes>> H_;					// pointer to the Hamiltonian

protected:
	// information about the NQS method
	std::string info_					=		"General NQS";
	pBar pBar_;															// for printing out the progress

	// simulation specific
	double lr_							=		0.0;					// specific learning rate
	
	// architecture specific
	uint nVis_							=		1;						// number of visible neurons (input variables)
	uint nSites_						=		1;						// number of lattice sites
	uint fullSize_						=		1;						// full number of the parameters (for memory purpose)

	// Hilbert space info
	u64 Nh_								=		1;
	uint nParticles_					=		1;
	bool conservesParticles_			=		true;

	// Random number generator
	randomGen ran_;														// consistent quick random number generator

	/* ----------------------------------------------------------- */

	// ---------------------- T H R E A D I N G ---------------------
							
	uint threadNum_						=		1;						// number of threads that works on this
	// create thread pool
#if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP
	uint threadsNumLeft_				=		0;
	v_1d<std::thread> threads_;
	v_1d<CondVarKernel<_T>> kernels_;
	std::mutex mutex;
#endif

	/* ----------------------------------------------------------- */

	// ----------------------- T R A I N I N G ----------------------
	// for flipping
	uint nFlip_							=		1;
	v_1d<uint> flipPlaces_;												// stores flip spots to be flipped during one sampling step
	v_1d<_stateType> flipVals_;											// stores values before the flip to be used for the gradients
	
	// current state
	NQSS curVec_;														// currently processed state vector for convenience
#ifndef NQS_USE_VEC_ONLY
	u64 curState_						=		0;						// currently processed state
#endif
	
	// temporary placeholders
	v_1d<NQSS> tmpVecs_;
	u64 tmpState_						=		0;
	NQSS tmpVec_;
	
	// ------------------------ W E I G H T S -----------------------

	// derivatives storage
	NQSW derivatives_;													// store the variational derivatives F_k
	NQSW derivativesC_;													// derivatives conjugated
#ifdef NQS_USESR
	NQSW S_;															// positive semi-definite covariance matrix
#endif
	NQSB F_;															// forces

protected:
	// ----------------------- S T A R T E R S ----------------------
	virtual void setInfo()								=			0;
	virtual void allocate();
	
	/* ----------------------------------------------------------- */

	// ------------------------ S E T T E R S -----------------------
	// column vector state
	virtual void setState(const NQSS& _st, bool _set)	=			0;
	virtual void setState(const NQSS& _st);
	// integer state
	virtual void setState(u64 _st, bool _set)			=			0;
	virtual void setState(u64 _st);
	
	// -------------------------- F L I P S --------------------------
	virtual void chooseRandomFlips()					=			0;

	// apply flips to the temporary vector or the current vector according the template
	virtual void applyFlipsT()							=			0;
	virtual void applyFlipsC()							=			0;			
	virtual void setRandomFlipNum(uint _nFlips)			=			0;

	/* ------------------------------------------------------------ */
	// -------------------- P R O B A B I L I T Y --------------------
	
	// The probability of accepting new state s' from state s
	// comming from the Metropolis - Hastings algorithm is of the 
	// form: 
	// $A(s', s) = min [1, \frac{P(s')}{P(s)}\frac{r(s'->s)}{r(s->s')}],$ 
	// where r(s'->s) is chosen such that the ratio is 1.0. 
	// For example one can choose 
	// $ r(s'->s) = \frac{|<s'|H|s>|(1-\delta _{s,s'})}{\\sum _{s!=s'} |<s'|H|s>| }.$
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
	virtual auto pRatio()								-> _T			{ return this->pRatio(this->flipPlaces_[0], this->flipVals_[0]);	};
	// ratio when 1 flip is used and flips are provided to the function (fP, fV)
	virtual auto pRatio(uint fP, float fV)				->_T			= 0;
	// ratio when multiple flips are used, provided the number of flips taken
	virtual auto pRatio(uint nFlips)					->_T			= 0;
	// ratio when a new vector is provided (after flip) - (uses current vector as well)
	virtual auto pRatio(const NQSS& _v)					->_T			{ return this->pRatio(this->curVec_, _v);							};			
	// ratio when a new vector is provided (after flip) - (uses two different vectors)
	virtual auto pRatio(const NQSS& _v1,
						const NQSS& _v2)				->_T			= 0;
	// ratio when exact points are provided (used for the Hamiltonian)
	virtual auto pRatio(std::initializer_list<int> fP, 
						std::initializer_list<double> fV)->_T			= 0;
	/* ------------------------------------------------------------ */

	// ----------------------- W E I G H T S -------------------------
#ifdef NQS_ANGLES_UPD
	virtual void update(uint nFlips = 1)								{};
	virtual void update(const NQSS& v, uint nFlips = 1)					{};
	virtual void unupdate(uint nFlips = 1)								{};
#endif
	virtual void updateWeights()						=				0;
public:
	virtual bool saveWeights(std::string _path, std::string _file);
	virtual bool setWeights(std::string _path, std::string _file);

	/* ------------------------------------------------------------ */
 protected:
	// --------------------- T R A I N   E T C -----------------------
	virtual void grad(const NQSS& _v, uint _plc)		=				0;
	virtual void gradFinal(const NQSB& _energies);
	// stochastic reconfiguration
#ifdef NQS_USESR
	virtual void gradSR(uint step = 0);
#endif
	 // regularization
#ifdef NQS_SREG 
	double covMatrixRegMult		= 0.95;									// multiplier for the regularization
	double covMatrixRegStart	= 0.02;									// starting parameter for regularisation (epsilon1)
	double covMatrixRegStart2	= 0.0;									// starting parameter for regularisation (epsilon2)
	virtual void covMatrixReg();
#endif
	
	/* ------------------------------------------------------------ */

	// ------------------------ E N E R G Y --------------------------
	auto pKernel(std::initializer_list<int>  fP,
				 std::initializer_list<double> fV)		-> _T			{ return this->pRatio(fP, fV); };

	/* ------------------------------------------------------------ */
protected:
	_T locEnKernel();
#if not defined NQS_USE_OMP && defined NQS_USE_MULTITHREADING
	virtual void locEnKernel(uint _start, uint _end, uint _threadNum);
#endif

	/* ------------------------------------------------------------ */

public:
	// ------------------------ S E T T E R S ------------------------
	virtual void init()									=				0; 
	virtual void setRandomState(bool _upd = true)						{ this->setState(this->ran_.template randomInt<u64>(0, this->Nh_), _upd);	};
	
	// unapply flips of the temporary vector or the current vector according the template
	virtual void unapplyFlipsT()										{ this->applyFlipsT();																};
	virtual void unapplyFlipsC()										{ this->applyFlipsC();																};
	
	/* ------------------------------------------------------------ */

	// ------------------------ G E T T E R S ------------------------
	auto getInfo()								const -> std::string	{ return this->info_;					};
	auto getNvis()								const -> uint			{ return this->nVis_;					};
	auto getF()									const -> NQSB			{ return this->F_;						};
	auto getCovarianceMat()						const -> NQSW			{ return this->S_;						};	
	// Hilbert
	auto getHilbertSize()						const -> u64			{ return this->Nh_;						};
	// Hamiltonian
	auto getHamiltonianInfo()					const -> std::string	{ return this->H_->getInfo();			};
	auto getHamiltonianEigVal(u64 _idx)			const -> double			{ return this->H_->getEigVal(_idx);		};
	auto getHamiltonian() const -> std::shared_ptr<Hamiltonian<_Ht>>	{ return this->H_;						};
	auto getHilbertSpace() const -> Hilbert::HilbertSpace<_Ht>			{ return this->H_->getHilbertSpace();	};

	// ----------------------- S A M P L I N G -----------------------
#ifndef NQS_USE_VEC_ONLY
	virtual void blockSample(	uint _bSize,
								u64 _start, 
								bool _therm = false);
#else
	virtual void blockSample(	uint _bSize,
								const NQSS& _start,
								bool _therm = false);
#endif
	virtual arma::Col<_T> train(uint mcSteps,							// number of Monte Carlo Steps
								uint nThrm,								// number of mcSteps to thermalize
								uint nBlck,								// number of such blocks for one average step
								uint bSize,								// for killing correlations
								std::string dir,						// saving directory
								uint nFlip			= 1,				// number of flips to set
								bool quiet			= false,			// shall talk?
								clk::time_point _t	= NOW,				// time!
								uint progPrc		= 25);

	virtual arma::Col<_T> collect(uint nSam, 
								  uint nThrm, 
								  uint nBlck, 
								  uint bSize, 
								  uint nFlip		= 1,
								  bool quiet		= false,
								  clk::time_point _t	= NOW,			// time!
								  NQSAv::MeasurementNQS<_T>& _mes = {});

	// ----------------------- F I N A L E -----------------------
	virtual auto ansatz(const NQSS& _in)		const ->_T				= 0;


	// -------------------- C O N S T R U C T --------------------

public:
	virtual ~NQS();
	NQS(std::shared_ptr<Hamiltonian<_Ht>>& _H, double _lr = 1e-2, uint _threadNum = 1, int _nParticles = -1);
	NQS()										=						default;
};

// ##########################################################################################################################################

// ############################################################# W E I G H T S ##############################################################

// ##########################################################################################################################################

/*
* @brief Attempting to save the weights as a vector to a given filename.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @returns whether the save has been successful
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool NQS<_spinModes, _Ht, _T, _stateType>::saveWeights(std::string _path, std::string _file)
{
	LOGINFO("Saving the checkpoint configuration.", LOG_TYPES::INFO, 2, '#');

	// save the weights to a given path
	auto _isSaved = saveAlgebraic<_T>(_path, _file, this->F_, "weights");

	// if not saved properly
	if (!_isSaved && (_file != "weights.h5"))
	{
		LOGINFO("Couldn't save the weights to the given path.", LOG_TYPES::ERROR, 3);
		LOGINFO("Saving to default... ", LOG_TYPES::ERROR, 3);
		return this->saveWeights(_path, "weights.h5");
	}
	return _isSaved;
}

///////////////////////////////////////////////////////////////////////

/*
* @brief Load the weights from the file specified by the user.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @returns whether the load has been successful
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool NQS<_spinModes, _Ht, _T, _stateType>::setWeights(std::string _path, std::string _file)
{
	LOGINFO("Loading the checkpoint weights:", LOG_TYPES::INFO, 2);
	return loadAlgebraic<_T>(_path, _file, this->F_, "weights");
}

// ##########################################################################################################################################

// ########################################################### C O N S T R U C T ############################################################

// ##########################################################################################################################################

/*
* @brief Allocates the main gradient parameters and the temporary and current vectors.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::allocate()
{
	// allocate gradients
	this->F_.resize(this->fullSize_);
#ifdef NQS_USESR
	this->S_.resize(this->fullSize_, this->fullSize_);
#endif
	this->curVec_ = arma::ones(this->nVis_);
	this->tmpVec_ = arma::ones(this->nVis_);
}

///////////////////////////////////////////////////////////////////////

/*
* @brief General destructor of the model.
* Waits on the threads if multithreading is run!
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline NQS<_spinModes, _Ht, _T, _stateType>::~NQS()
{
	DESTRUCTOR_CALL;
#if not defined NQS_USE_OMP && defined NQS_USE_MULTITHREADING
	for (int _thread = 0; _thread < this->threadNum_; _thread++)
	{
		std::unique_lock<std::mutex> lock(this->kernels_[_thread].mutex);
		this->kernels_[_thread].flagThreadKill_		= true;
		this->kernels_[_thread].end_				= true;
		this->kernels_[_thread].flagThreadRun_		= 1;
		this->kernels_[_thread].cv.notify_all();
	}
	for (int _thread = 0; _thread < this->threadNum_; _thread++)
		if (threads_[_thread].joinable())
			threads_[_thread].join();
#endif
}

// ##########################################################################################################################################

// ############################################################ S A M P L I N G #############################################################

// ##########################################################################################################################################

/*
* @brief Block updates the current state according to Metropolis-Hastings algorithm. The block size is chosen so that
* it gives the given statistics of the averages found.
* @param _bSize the size of the correlation block; how many flips one needs to do in order to get rid of the MC correlations - bin size 
* @param _start the state to start from
* @param _nFlip number of flips in a single step
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
#ifndef NQS_USE_VEC_ONLY
inline void NQS<_spinModes, _Ht, _T, _stateType>::blockSample(uint _bSize, u64 _start, bool _therm)
#else
inline void NQS<_spinModes, _Ht, _T, _stateType>::blockSample(uint _bSize, const NQSS& _start, bool _therm)
#endif
{
	// check whether we should set a state again or thermalize the whole process
#ifndef NQS_USE_VEC_ONLY
	if (_start != this->curState_ || _therm)
		this->setState(_start, _therm);
#else
	if (_therm)
		this->setState(_start, _therm);
#endif

	// set the temporary state - the vectors are usefull so set them accordingly
	this->tmpVec_ = this->curVec_;

	// go through each block step
	for (uint bStep = 0; bStep < _bSize; ++bStep)
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
			this->update(this->nFlip_);
		}
		// set the vector back to normal
		else
		{
			this->unupdate();
			this->unapplyFlipsT();
		}
	}
	// set the current integer state
#ifndef NQS_USE_VEC_ONLY
	this->curState_ = BASE_TO_INT<u64>(this->curVec_, discVal_);
#endif
#ifndef NQS_ANGLES_UPD
#	ifndef NQS_USE_VEC_ONLY
	this->setState(this->curState_, true);
#	else
	this->setState(this->curVec_, true);
#	endif
#endif
}

// ##########################################################################################################################################

// ########################################################## L O C   E N E R G Y ###########################################################

// ##########################################################################################################################################

/*
* @brief Calculate the local energy depending on the given Hamiltonian - kernel with OpenMP is used
* when the omp pragma NQS_USE_OMP is set or multithreading is not used, otherwise threadpool is used
* to calculate the local energies.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T NQS<_spinModes, _Ht, _T, _stateType>::locEnKernel()
{
#if defined NQS_USE_OMP
	double energyR	= 0.0;
	double energyI	= 0.0;

#	ifndef _DEBUG
	#pragma omp parallel for reduction(+ : energyR, energyI) num_threads(this->threadNum_)
#	endif
	for (int site = 0; site < this->nSites_; site++) 
	{
#	ifndef NQS_USE_VEC_ONLY
		auto energy	=	this->H_->locEnergy(this->curState_, 
											site, 
											std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
											this,
											std::placeholders::_1,
											std::placeholders::_2));
#	else
		auto energy	=	this->H_->locEnergy(this->curVec_, 
											site, 
											std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
											this,
											std::placeholders::_1,
											std::placeholders::_2));
#	endif
		energyR		+=	algebra::real(energy);
		energyI		+=	algebra::imag(energy);
	}
	return toType<_T>(energyR, energyI);
#else
	_T energy = 0.0;
	// run all threads
	for (int _thread = 0; _thread < this->threadNum_; _thread++)
	{
		std::lock_guard<std::mutex> lock(this->kernels_[_thread].mutex);
		this->kernels_[_thread].flagThreadRun_	= true;
		this->kernels_[_thread].end_			= false;
		this->kernels_[_thread].cv.notify_one();
	}

	// wait for all threads
	for (int _thread = 0; _thread < this->threadNum_; _thread++)
	{
		{
			std::unique_lock<std::mutex> lock(this->kernels_[_thread].mutex);
			this->kernels_[_thread].cv.wait(lock, [this, _thread] { return !this->kernels_[_thread].flagThreadRun_; });
		}
		energy += this->kernels_[_thread].kernelValue_;
	}
	return energy;
#endif
}

///////////////////////////////////////////////////////////////////////

#if not defined NQS_USE_OMP && defined NQS_USE_MULTITHREADING
/*
* @brief Allows to run a thread pool based on the condition that all threads wait for a mutex to further run the program
* @param _start starting site for a given thread
* @param _end ending site for a given thread
* @param _threadNum number of the thread currently run
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::locEnKernel(uint _start, uint _end, uint _threadNum)
{
	// does not go in if the simulation is finished
	while (!this->kernels_[_threadNum].flagThreadKill_)
	{
		// wait for the lock to end
		{
			// aquire mutex lock as required by condition variable
			std::unique_lock<std::mutex> lock(this->kernels_[_threadNum].mutex);	
			// thread will suspend here and release the lock if the expression does not return true
			this->kernels_[_threadNum].cv.wait(lock, [this, _threadNum] { return this->kernels_[_threadNum].flagThreadRun_; });	

			// if kill then kill
			if (this->kernels_[_threadNum].flagThreadKill_)
			{
				this->kernels_[_threadNum].end_ = true;
				break;
			}
		}

		// Process the work
		this->kernels_[_threadNum].kernelValue_ = 0.0;
		for (auto site = _start; site < _end; ++site)
		{
#ifndef NQS_USE_VEC_ONLY
			this->kernels_[_threadNum].kernelValue_ += algebra::cast<_T>(this->H_->locEnergy(this->curState_,
																							 site, 
																							 std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
																							 this,
																							 std::placeholders::_1,
																							 std::placeholders::_2)));
#else
			this->kernels_[_threadNum].kernelValue_ += algebra::cast<_T>(this->H_->locEnergy(this->curVec_,
																							site, 
																							std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
																							this,
																							std::placeholders::_1,
																							std::placeholders::_2)));
#endif
		}
		// lock again
		{
			std::lock_guard<std::mutex> lock(this->kernels_[_threadNum].mutex);
			this->kernels_[_threadNum].flagThreadRun_	= false;
			this->kernels_[_threadNum].end_				= true;
			// Notify waiting threads if needed
		}
		this->kernels_[_threadNum].cv.notify_one();
	}
}
#endif

///////////////////////////////////////////////////////////////////////

// ##########################################################################################################################################

// ############################################################# S E T T E R S ##############################################################

// ##########################################################################################################################################

/*
* @brief Updates the current state processed by the NQS.
* @param _st Column vector to be set as a new state
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setState(const NQSS& _st)
{
	this->curVec_	= _st;
#ifndef NQS_USE_VEC_ONLY
	this->curState_ = BASE_TO_INT<u64>(_st, this->discVal_);
#endif
}

///////////////////////////////////////////////////////////////////////

/*
* @brief Updates the current state processed by the NQS.
* @param _st Integer vector value to be set as a new state
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setState(u64 _st)
{
#ifndef NQS_USE_VEC_ONLY
	this->curState_ = _st;
#endif
	INT_TO_BASE(_st, this->curVec_, this->discVal_);
}

// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

// ##########################################################################################################################################

///////////////////////////////////////////////////////////////////////
#ifdef NQS_USESR

/*
* @brief Calculates the update parameters for weights to be updated in a form:
* weights[new] <- weights[old] - lr * S^{-1} * F --> second order optimization method
* @param step current step of updating - for the regularization purpose
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradSR(uint step)
{
	// update the flattened vector for the gradients
#	ifdef NQS_SREG 
	// regularize the covariance matrix before inverting it
	this->covMatrixReg();
#	endif
#	ifdef NQS_PINV
	// calculate the pseudoinverse
	if (NQS_PINV > 0)
		this->F_ = this->lr_ * (arma::pinv(this->S_, NQS_PINV) * this->F_);
	else
		this->F_ = this->lr_ * (arma::pinv(this->S_) * this->F_);
	return;
#	else 
	// solve normally
	//this->F_ = this->lr_ * (arma::inv(this->S_) * this->F_);
	this->F_ = this->lr_ * arma::solve(this->S_, this->F_, arma::solve_opts::likely_sympd);
#	endif 
}
#endif

///////////////////////////////////////////////////////////////////////

/*
* @brief	Calculate the final step for the weights to be updated.
* First of all, it calculates the generalized forces
* 
* @warning Uses the forces vector (member F_) to store the forces that update the weights
* 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradFinal(const NQSB& _energies)
{
	// calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}>
	this->derivativesC_ = arma::conj(this->derivatives_);
	this->F_			= arma::cov(this->derivativesC_, _energies, 1);
#ifdef NQS_USESR
	this->S_			= arma::cov(this->derivativesC_, this->derivatives_, 1);
	
	// update model
	this->gradSR(0);

#else
	// standard updater with the gradient only!
	this->F_ *= this->lr_;
#endif
}

// ##########################################################################################################################################

///////////////////////////////////////////////////////////////////////
#ifdef NQS_SREG
/*
! TODO 
* @brief The inverse of the covariance matrix is poorly defined at the begining of training. 
* Use regularization to fix that issue.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::covMatrixReg()
{
	if (this->covMatrixRegStart != 0)
	{
		this->S_.diag() *= (1.0 + this->covMatrixRegStart);
	}
	if (this->covMatrixRegStart2 != 0)
	{
		auto maximal_re	=	arma::max(arma::real(this->S_.diag()));
		this->S_.diag()	+= this->covMatrixRegStart2 * maximal_re;
	}
}
#endif
///////////////////////////////////////////////////////////////////////

// ##########################################################################################################################################

// ############################################################ T R A I N I N G #############################################################

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
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline arma::Col<_T> NQS<_spinModes, _Ht, _T, _stateType>::train(uint mcSteps,
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
	strSeparatedP(outstr, '\t', 2,
				  VEQV(Monte Carlo Steps, mcSteps),
				  VEQV(Thermalization Steps, nThrm),
				  VEQV(Block Number, nBlck),
				  VEQV(Size of the single block, bSize),
				  VEQV(Number of flips taken at each step, nFlip));
	LOGINFOG("Train: " + outstr, LOG_TYPES::TRACE, 1);

	// make the pbar!
	this->pBar_			= pBar(progPrc, mcSteps);
	this->derivatives_.resize(nBlck, this->fullSize_);
	this->derivativesC_.resize(nBlck, this->fullSize_);

	// save all average weights for covariance matrix
	arma::Col<_T> meanEn(mcSteps, arma::fill::zeros);
	// history of energies
	arma::Col<_T> En(nBlck, arma::fill::zeros);
	// set the random state at the begining
	this->setRandomState();
	this->setRandomFlipNum(nFlip);

	// go through the Monte Carlo steps
	for (uint i = 1; i <= mcSteps; ++i)
	{
		this->setRandomState();
		// thermalize
#ifndef NQS_USE_VEC_ONLY
		this->blockSample(nThrm, this->curState_, false);
#else
		this->blockSample(nThrm, this->curVec_, false);
#endif

		// iterate blocks
		for (uint _taken = 0; _taken < nBlck; ++_taken) {

			// sample them!
#ifndef NQS_USE_VEC_ONLY
			this->blockSample(bSize, this->curState_, false);
#else
			this->blockSample(bSize, this->curVec_, false);
#endif
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
		meanEn(i - 1) = arma::mean(En);
		LOGINFO(VEQ(meanEn(i - 1)), LOG_TYPES::TRACE, 1);

		// update the progress bar
		PROGRESS_UPD_Q(i, this->pBar_, "PROGRESS NQS", !quiet);
#ifdef NQS_SAVE_WEIGHTS
		if (i % this->pBar_.percentageSteps == 0) 
			this->saveWeights(dir + NQS_SAVE_DIR, "weights.h5");
#endif
	}
	LOGINFO(_t, "NQS_EQ", 1);
	return meanEn;
}

// ##########################################################################################################################################

// ########################################################## C O L L E C T I O N ###########################################################

// ##########################################################################################################################################

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline arma::Col<_T> NQS<_spinModes, _Ht, _T, _stateType>::collect(	uint nSam,
																	uint nThrm, 
																	uint nBlck,
																	uint bSize,
																	uint nFlip,
																	bool quiet,
																	clk::time_point _t,
																	NQSAv::MeasurementNQS<_T>& _meas)
{																							
	std::string outstr= "";
	// set the info about training
	strSeparatedP(outstr, '\t', 2,
				  VEQV(Thermalization Steps, nThrm),
				  VEQV(Block Number, nBlck),
				  VEQV(Size of the single block, bSize),
				  VEQV(Number of flips taken at each step, nFlip));
	LOGINFOG("Collect: " + outstr, LOG_TYPES::TRACE, 1);

	// make the pbar!
	this->pBar_	= pBar(20, nSam);
	arma::Col<_T> meanEn(nSam, arma::fill::zeros);
	// history of energies
	arma::Col<_T> En(nBlck, arma::fill::zeros);

	// set the random state at the begining
	this->setRandomFlipNum(nFlip);
	std::function<_T(const NQSS& _v)> opFun = [&](const NQSS& v) { return this->pRatio(v); };
	
	// go through the number of samples to be collected
	for (uint i = 1; i <= nSam; ++i)
	{
		//this->setRandomState();
		// thermalize!
#ifndef NQS_USE_VEC_ONLY
		this->blockSample(nThrm, this->curState_, false);
#else
		this->blockSample(nThrm, this->curVec_, false);
#endif
		// iterate blocks
		for (uint _taken = 0; _taken < nBlck; ++_taken) 
		{
			// sample them!
#ifndef NQS_USE_VEC_ONLY
			this->blockSample(bSize, this->curState_, false);
#else
			this->blockSample(bSize, this->curVec_, false);
#endif 
			// energy
			En(_taken) = this->locEnKernel();

			// local operators
			_meas.measure(BASE_TO_INT<u64>(this->curVec_, this->discVal_), opFun);
		}

		// normalize operators to be saved 
		_meas.normalize(nBlck);

		// save the mean energy
		meanEn(i - 1) = arma::mean(En);

		// update the progress bar
		PROGRESS_UPD_Q(i, this->pBar_, "PROGRESS NQS", !quiet);
	}
	LOGINFO(_t, "NQS_COLLECTION", 1);
	return meanEn;
}

// ##########################################################################################################################################

// ########################################################### C O N S T R U C T ############################################################

// ##########################################################################################################################################

/*
* @brief General constructor of the NQS solver
* @param _H Hamiltonian to be used for correct data sampling
* @param _lr learning rate to be used for the training
* @param _threadNum thread number to be used for the solver - available on setting corresponding thread pragma
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline NQS<_spinModes, _Ht, _T, _stateType>::NQS(std::shared_ptr<Hamiltonian<_Ht>>& _H, double _lr, uint _threadNum, int _nParticles)
	: H_(_H), lr_(_lr)
{
	// set the number of particles
	// set the visible layer (for hardcore-bosons we have the same number as sites but fermions introduce twice the complication)
	this->nVis_					=			_H->getNs() * (this->spinModes_ / 2);
	this->nSites_				=			_H->getNs();
	// make it half filling if necessary
	this->nParticles_			=			(_nParticles < 0 || this->spinModes_ == 2) ? this->nSites_ : (uint)_nParticles;
	// check the Hilbert space
	this->Nh_					=			_H->getHilbertSize();
	// set the random number generator
	this->ran_					=			_H->ran_;

	// set threads
	{
#ifndef NQS_USE_MULTITHREADING
		this->threadNum_		=			1;
#else
		this->threadNum_		=			std::min(_threadNum, this->nSites_);
		this->threadsNumLeft_	=			std::max(_threadNum - this->threadNum_, (uint)1);
#endif
	// Use threads for all consecutive parallel regions
#if defined NQS_USE_MULTITHREADING
#	ifdef NQS_USE_OMP
		omp_set_num_threads(this->threadNum_);   
#	else
		this->threads_.reserve(this->threadNum_);
		this->kernels_			=			v_1d<CondVarKernel<_T>>(this->threadNum_);
		
		// calculate how many sites goes to one thread
		uint _siteStep			=			std::ceil(this->nSites_ / 1.0 / this->threadNum_);

		// start the threads that calculate the energy with the local energy kernel function
		// this function waits for the specific energy calculation to be ready on each thread
		// this is handled through "flagThreadRun_" member
		for (uint i = 0; i < this->threadNum_; i++)
		{
			std::function<void()> lambda = [this, i, _siteStep]() 
				{ 
					this->locEnKernel(i * _siteStep, std::min((i + 1) * _siteStep, this->nSites_), i); 
				};
			this->threads_.emplace_back(std::thread(lambda));
		}
#	endif
#endif
	}
	LOGINFO("Constructed the general NQS class", LOG_TYPES::TRACE, 2);
};

// ##########################################################################################################################################

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GENERAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
class NQS_S : public NQS<_spinModes, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);
public:
	NQS_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, double _lr, uint _threadNum, int _nParticles)
		: NQS<_spinModes, _Ht, _T, _stateType>(_H, _lr, _threadNum, _nParticles)
														 { NQS_LOG_ERROR_SPIN_MODES; };

	protected:
	// -------------------------- F L I P S --------------------------
	virtual void chooseRandomFlips()			override { NQS_LOG_ERROR_SPIN_MODES; };

	// apply flips to the temporary vector or the current vector according the template
	virtual void applyFlipsT()					override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual void applyFlipsC()					override { NQS_LOG_ERROR_SPIN_MODES; };
	virtual void setRandomFlipNum(uint _nFlips)	override { NQS_LOG_ERROR_SPIN_MODES; };
};

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SPINS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

#	ifndef NQS_2_H
#		define NQS_2_H

/*
* @brief Template NQS for Spins (hardcore bosons)
*/
template <typename _Ht, typename _T, class _stateType>
class NQS_S<2, _Ht, _T, _stateType> : public NQS<2, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);

	NQS_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, double _lr, uint _threadNum, int _nParticles)
		: NQS<2, _Ht, _T, _stateType>(_H, _lr, _threadNum, _H->getNs()) 
	{	};

protected:
	// -------------------------- F L I P S --------------------------
	virtual void chooseRandomFlips()			override;

	// apply flips to the temporary vector or the current vector according the template
	virtual void applyFlipsT()					override { for (auto& i : this->flipPlaces_) flip(this->tmpVec_, i, 0, this->discVal_);	};
	virtual void applyFlipsC()					override { for (auto& i : this->flipPlaces_) flip(this->curVec_, i, 0, this->discVal_);	};
	virtual void setRandomFlipNum(uint _nFlips) override;
};

// !!!!!!!!!!!!!!!!!! F L I P S !!!!!!!!!!!!!!!!!!

/*
* @brief Randomly flip the discrete variables at chosen flip places. Sets the random flips to the vector already saved.
*/
template<typename _Ht, typename _T, class _stateType>
inline void NQS_S<2, _Ht, _T, _stateType>::chooseRandomFlips()
{
	// go through the vector elements
	for (auto i = 0; i < this->flipPlaces_.size(); ++i)
	{
		auto fP					= this->ran_.template randomInt<uint>(0, this->nVis_);
		// choose the flip place of the vector
		this->flipPlaces_[i]	= fP;
		// save the element of a vector before the flip
		this->flipVals_[i]		= this->tmpVec_(fP);
	}
}

//////////////////////////////////////////////////

/*
* @brief Set the number of random flips.
* @param _nFlips number of flips to be used
*/
template<typename _Ht, typename _T, class _stateType>
inline void NQS_S<2, _Ht, _T, _stateType>::setRandomFlipNum(uint _nFlips)
{
	this->nFlip_ = _nFlips;
	this->flipPlaces_.resize(this->nFlip_);
	this->flipVals_.resize(this->nFlip_);
}

#	endif

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FERMIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

#	ifndef NQS_4_H
#		define NQS_4_H

/*
* @brief Template NQS for Fermions
*/
template <  typename _Ht, typename _T, class _stateType>
class NQS_S<4, _Ht, _T, _stateType> : public NQS<4, _Ht, _T, _stateType>
{
	NQS_PUBLIC_TYPES(_T, _stateType);

	NQS_S(std::shared_ptr<Hamiltonian<_Ht>>& _H, double _lr, uint _threadNum, int _nParticles)
		: NQS<4, _Ht, _T, _stateType>(_H, _lr, _threadNum, _nParticles) {};
	
protected:
	// -------------------------- F L I P S --------------------------
	virtual void chooseRandomFlips();

	// apply flips to the temporary vector or the current vector according the template
	virtual void applyFlipsT() override								{ LOG_ERROR("NOT IMPLEMENTED FOR FERMIONS YET"); };
	virtual void applyFlipsC() override								{ LOG_ERROR("NOT IMPLEMENTED FOR FERMIONS YET"); };			
	virtual void setRandomFlipNum(uint _nFlips) override;
};

// !!!!!!!!!!!!!!!!!! F L I P S !!!!!!!!!!!!!!!!!!

/*
* @brief Randomly flip the discrete variables at chosen flip places. Sets the random flips to the vector already saved.
*/
template<typename _Ht, typename _T, class _stateType>
inline void NQS_S<4, _Ht, _T, _stateType>::chooseRandomFlips()
{
	LOG_ERROR("NOT IMPLEMENTED FOR FERMIONS YET");
}

/*
* @brief Set the number of random flips.
* @param _nFlips number of flips to be used
*/
template<typename _Ht, typename _T, class _stateType>
inline void NQS_S<4, _Ht, _T, _stateType>::setRandomFlipNum(uint _nFlips)
{
	// check if the NQS shall conserve particles
	this->nFlip_ = this->conservesParticles_ ? 2 * _nFlips : _nFlips;
	this->flipPlaces_.resize(this->nFlip_);
	this->flipVals_.resize(this->nFlip_);
}

#	endif
//////////////////////////////////////////////////////////////////////////////////////////

// ##########################################################################################################################################

#endif