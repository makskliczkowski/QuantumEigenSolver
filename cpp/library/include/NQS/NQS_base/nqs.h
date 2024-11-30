/***************************************
* Defines the generic NQS Solver class. 
* Allows for later inhertiance
* for a fine model specialization.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#include "../nqs_operator.h"
#include "../../algebra/general_operator.h"
#include "../../algebra/operators.h"
#include "armadillo"
#include <cstddef>
#include <functional>
#include <memory>
#ifndef NQS_H
#define NQS_H

// include all the definions
#include "nqs_definitions_base.h"
#include "nqs_definitions_lower.tpp"

/*
* @class NQS
* @brief General template class for Neural Quantum States (NQS) solver.
* 
* This class defines a flexible framework for building Neural Quantum State solvers, supporting multiple spin modes and Hamiltonians.
* It is designed for efficient sampling and optimization of neural network weights using Monte Carlo-based methods. The class can be 
* specialized by inheriting and overriding key methods.
* 
* @tparam _spinModes Number of spin modes (e.g., 2 for hardcore bosons, 4 for fermions).
* @tparam _Ht Type of Hamiltonian used for energy calculations.
* @tparam _T General type for numerical precision (default: same as _Ht).
* @tparam _stateType Type of quantum state (default: double).
*/
template <uint _spinModes, 
		typename _Ht,
		typename _T			= _Ht, 
		class _stateType	= double>
class NQS 
{	
	// type definitions 
	NQS_PUBLIC_TYPES(_T, double);	
public:
	using NQSLS_p 						=		typename NQS_lower_t<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
	NQS_info_t info_p_;													// information about the NQS
	NQS_lower_t<_spinModes, _Ht, _T, _stateType> lower_states_;			// information about the training

	// ensure numerical stability
    const _T epsilon_ 					= 		std::numeric_limits<_T>::epsilon();
	u64 accepted_ 						= 		0;						// number of accepted flips
	u64 total_ 							= 		0;						// total number of flips

protected:
	// for the Hamiltonian information, types and the Hilbert space
	const uint spinModes_				=		_spinModes;				// number of spin modes -> e.g. 2 for hardcore bosons, 4 for fermions
	const double discVal_				=		Operators::_SPIN_RBM;   // discrete value for the numbers in vector representation
	bool updateWeights_ 				= 		true;					// shall update the weights in current step? (default is true)

	std::shared_ptr<Hamiltonian<_Ht, _spinModes>> H_;					// pointer to the Hamiltonian instance (for the energy calculation)

protected:
	std::string info_					=		"General NQS";
	pBar pBar_;															// for printing out the progress
	randomGen ran_;														// consistent quick random number generator

	// ---------------------- T H R E A D I N G ---------------------
#ifdef NQS_NOT_OMP_MT
	bool initThreads(uint _threadNum = 0);
	NQS_thread_t<_T> threads_;											// thread information
#endif

	// ----------------------- T R A I N I N G ----------------------
	uint nFlip_							=		1;						// number of flips to be done in one step (each flip is a change in the state)
	v_1d<uint> flipPlaces_;												// stores flip spots to be flipped during one sampling step
	v_1d<_stateType> flipVals_;											// stores values before (!!!) the flip to be used for the gradients
	
	NQSS curVec_;														// currently processed state vector for convenience
	u64 curState_						=		0;						// currently processed state - may or may not be used
	
	v_1d<NQSS> tmpVecs_;												// temporary vectors for the flips
	u64 tmpState_						=		0;						// temporary state for the flips
	NQSS tmpVec_;														// temporary vector for the flips (for the current state)
	
	// ------------------------ W E I G H T S -----------------------
	NQSW derivatives_;													// store the variational derivatives F_k (nBlocks x fullSize), where nBlocks is the number of consecutive observations
	NQS_ROW_T derivativesMean_;											// store the mean of the derivatives (F_k) - for the SR (fullSize)
	NQSW derivativesCentered_;											// store the centered derivatives (F_k - <F_k>) - for the SR (nBlocks x fullSize), where nBlocks is the number of consecutive observations
	NQSW derivativesCenteredH_;											// store the centered derivatives (F_k - <F_k>) - for the SR (fullSize x nBlocks), where nBlocks is the number of consecutive observations	
	void derivativesReset(size_t nBlocks = 1)							{ this->derivatives_ = NQSW(nBlocks, this->info_p_.fullSize_, arma::fill::zeros); this->derivativesCentered_ = this->derivatives_; this->derivativesCenteredH_ = this->derivatives_.t(); }; 
#ifdef NQS_USESR_MAT_USED
	NQSW S_;															// positive semi-definite covariance matrix - to be optimized (inverse of the Fisher information matrix)
#else 
	algebra::Solvers::Preconditioners::Preconditioner<_T, true>* precond_ 	= nullptr;	// preconditioner for the conjugate gradient
	algebra::Solvers::General::Solver<_T, true>* solver_ 					= nullptr;	// solver for the Fisher matrix inversion
#endif
public:
	void setSolver(int _s, double _t, int _mi = 1000, double _r = -1.0);// solving method with the tolerance						
	void setPreconditioner(int _pre);									// set the preconditioner						
	void setPinv(double _pinv);											// if the pseudoinverse is used

protected:
	NQSB dF_;															// forces acting on the weights (F_k) - final gradient (dF)
	NQSB F_;															// forces acting on the weights (F_k) - to be used for the SR

protected:
	// ----------------------- S T A R T E R S ----------------------
	virtual void setInfo()								=			0; 	// set the information about the NQS (e.g., type, number of hidden units, etc.)
	virtual void allocate();											// allocate the memory for the NQS (e.g., vectors, matrices, etc.)
		
	// ------------------------ S E T T E R S -----------------------
	virtual void setState(const NQSS& _st, bool _set)	=			0;	// column vector state
	virtual void setState(const NQSS& _st);								// column vector state (set the current state)
	virtual void setState(u64 _st, bool _set)			=			0; 	// set the state (integer)
	virtual void setState(u64 _st);
	
	// -------------------------- F L I P S --------------------------
	virtual void chooseRandomFlips()					=			0;	// important for the flipPlaces_ and flipVals_ to be set! - choose random flips
	virtual void applyFlipsT()							=			0;	// apply flips to the temporary vector (tmpVec_)
	virtual void applyFlipsC()							=			0;	// apply flips to the current vector (curVec_)	
	virtual void unapplyFlipsT()						{ this->applyFlipsT(); }; // unapply flips of the temporary vector according the template 
	virtual void unapplyFlipsC()						{ this->applyFlipsC(); }; // unapply flips of the current vector according the template
	virtual void setRandomFlipNum(uint _nFlips)			=			0;	// set the number of flips to be done

public:

	// -------------------- P R O B A B I L I T Y --------------------
	// mucho importante for the flips to be done - most of those functions need to be implemented
	// in the derived class as they are model specific and depend on the architecture of the NQS.
	
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
	// only the new <\psi|s'> given from the new state s'. This is exactly what the NQS does.

	virtual auto pRatio()								->_T			{ return this->pRatio(this->flipPlaces_[0], this->flipVals_[0]);	};
	virtual auto pRatio(uint fP, float fV)				->_T			= 0;	// ratio when only 1 (single) flip is used and flips are provided to the function (fP, fV)
	virtual auto pRatio(uint nFlips)					->_T			= 0;
	virtual auto pRatio(const NQSS& _v)					->_T			{ return this->pRatio(this->curVec_, _v);							};			
	virtual auto pRatio(const NQSS& _v1,
						const NQSS& _v2)				->_T			= 0;	// ratio when a new vector is provided (after flip) - (uses two different vectors - one for the current state and one for the new state)
	virtual auto pRatio(std::initializer_list<int> fP, 
						std::initializer_list<double> fV)->_T			= 0; 	// ratio when exact points are provided (used for the Hamiltonian probability ratio - when the Hamiltonian changes the state)
	std::function<_T(const NQSS&)> pRatioFunc_;									// function for the probability ratio

	// ----------------------- W E I G H T S -------------------------
#ifdef NQS_ANGLES_UPD
	virtual void update(uint nFlips = 1)								{};
	virtual void update(const NQSS& v, uint nFlips = 1)					{};
	virtual void unupdate(uint nFlips = 1)								{};
#endif
	virtual void updateWeights()										= 0;
public:
	virtual bool saveWeights(std::string _path, std::string _file);
	virtual bool setWeights(std::string _path, std::string _file);

protected:
	// --------------------- T R A I N   E T C -----------------------
	virtual void grad(const NQSS& _v, uint _plc)						= 0;
	virtual void gradFinal(const NQSB& _energies, int step = 0, _T _cL = 0.0);
#ifdef NQS_USESR
	virtual void gradSR(uint step = 0, _T _cL = 0.0);					// stochastic reconfiguration
	virtual void covMatrixReg(int _step = 0, _T _cL = 0.0);
#endif
	
	// ------------------------ E N E R G Y --------------------------
	std::function<_T(std::initializer_list<int>, std::initializer_list<double>)> pKernelFunc_;	// function for the probability ratio

	/* ------------------------------------------------------------ */
protected:
	_T locEnKernel();
#ifdef NQS_NOT_OMP_MT
	virtual void locEnKernel(uint _start, uint _end, uint _threadNum);
#endif

	/* ------------------------------------------------------------ */

public:
	// ------------------------ S E T T E R S ------------------------
	virtual void init()													= 0; // initialize the NQS (e.g., set the random state, etc.)
	virtual void setRandomState(bool _upd = true);						// set the random state of the NQS

	// --------------------------------------------------------------- 
	
	void setEarlyStopping(size_t _pat, double _minDlt);	
	void setSregScheduler(int _sch = 0, double _sreg = 1e-7, double _sregd = 0.96, size_t _epo = 10, size_t _pat = 5);							
	void setTrainParExc(const NQS_train_t& _par);  						// training the excited states (if needed)
	void setScheduler(int _sch = 0, double _lr = 1e-3, double _lrd = 0.96, size_t _epo = 10, size_t _pat = 5);									
	/* ------------------------------------------------------------ */

	// ------------------------ G E T T E R S ------------------------
	auto saveInfo(const std::string& _dir, const std::string& _name, int i = 0) const -> void { this->info_p_.saveInfo(_dir, _name, i); };
	auto getInfo()								const -> std::string	{ return this->info_;					};
	auto getNvis()								const -> uint			{ return this->info_p_.nVis_;			};
	auto getF()									const -> NQSB			{ return this->F_;						};
#ifdef NQS_USESR_MAT_USED
	auto getCovarianceMat()						const -> NQSW			{ return this->S_;						};	
#endif
	// Hilbert
	auto getHilbertSize()						const -> u64			{ return this->info_p_.Nh_;				};
	// Hamiltonian
	auto getHamiltonianInfo()					const -> std::string	{ return this->H_->getInfo();			};
	auto getHamiltonianEigVal(u64 _idx)			const -> double			{ return this->H_->getEigVal(_idx);		};
	auto getHamiltonian() const -> std::shared_ptr<Hamiltonian<_Ht, _spinModes>> { return this->H_;						};
	auto getHilbertSpace() const -> Hilbert::HilbertSpace<_Ht, _spinModes> { return this->H_->getHilbertSpace();	};

	// ----------------------- S A M P L I N G -----------------------
	virtual void blockSample(uint _bSize, NQS_STATE_T _start, bool _therm = false);

	// single
	virtual bool trainStop(size_t i, const NQS_train_t& _par, _T _currLoss, _T _currstd = 0.0, bool _quiet = false);	
	virtual bool trainStep(size_t i, arma::Col<_T>& En,
									arma::Col<_T>& meanEn, 
									arma::Col<_T>& stdEn, 
									const NQS_train_t& _par, 
									const bool quiet, 
									const bool randomStart,
									Timer& _timer);

	virtual std::pair<arma::Col<_T>, arma::Col<_T>> train(const NQS_train_t& _par,
														  bool quiet			= false,	// shall talk? (default is false)
														  bool randomStart 		= false,	// random start (default is false)
														  clk::time_point _t	= NOW,		// time! (default is NOW)
														  uint progPrc		= 25);

	// --------------------- C O L L E C T I N G ---------------------

	// single 
	template <typename _CT>
	bool collectStep(size_t i, const NQS_train_t& _par, Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies = nullptr);
	template <typename _CT>
	void collect(const NQS_train_t& _par, Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies = nullptr, bool reset = false);

	// measurement
	virtual bool collectStep(size_t i, const NQS_train_t& _par, 
							NQSAv::MeasurementNQS<_T>& _meas,
							arma::Col<_T>* _E, 
							arma::Col<_T>* _EM,
							arma::Col<_T>* _ES, 
							const bool quiet,
							const bool randomStart,
							Timer& _timer);
	virtual void collect(const NQS_train_t& _par,
						NQSAv::MeasurementNQS<_T>& _mes 	= {},
						arma::Col<_T>* _energies			= nullptr,
						arma::Col<_T>* _energiesStd			= nullptr,
						bool quiet							= false,
						bool randomStart 					= false,
						clk::time_point _t					= NOW,
						uint progPrc						= 25);
	// other
	virtual void collect(const NQS_train_t& _par, NQSAv::MeasurementNQS<_T>& _mes);
	// for collecting the \sum _s f(s) / \psi(s) - used for the gradient calculation
	virtual void collect_ratio(const NQS_train_t& _par, std::function<_T(const NQSS&)> _f, arma::Col<_T>& _container);
	virtual void collect_ratio(const NQS_train_t& _par, NQS<_spinModes, _Ht, _T, _stateType>* other, arma::Col<_T>& _container);

	// ----------------------- F I N A L E -----------------------
	virtual auto ansatz(const NQSS& _in)		const ->_T				= 0;
	virtual auto ansatzlog(const NQSS& _in)		const ->_T				= 0;
	virtual auto ansatz_ratio(const NQSS& _in, NQS<_spinModes, _Ht, _T, _stateType>* _other) const -> _T = 0;

	// -------------------- C O N S T R U C T --------------------

public:
	virtual ~NQS();
	NQS() = default;
	NQS(const NQS& _n)
		: info_p_(_n.info_p_), H_(_n.H_), info_(_n.info_), pBar_(_n.pBar_), 
		ran_(_n.ran_), nFlip_(_n.nFlip_), flipPlaces_(_n.flipPlaces_), flipVals_(_n.flipVals_)
	{
		// this->threads_ 		= _n.threads_;
		// initialize the information 
		this->info_p_  		= _n.info_p_;
		this->lower_states_ = _n.lower_states_;
		// this->init();
#ifdef NQS_NOT_OMP_MT
		this->initThreads(_n.threads_.threadNum_);
#endif
	}
	NQS(NQS&& _n)
		: info_p_(_n.info_p_), H_(_n.H_), info_(_n.info_), pBar_(_n.pBar_), 
		ran_(_n.ran_), nFlip_(_n.nFlip_), flipPlaces_(_n.flipPlaces_), flipVals_(_n.flipVals_)
	{
		// this->threads_ 		= std::move(_n.threads_);
		// initialize the information
		this->info_p_  		= std::move(_n.info_p_);
		this->lower_states_ = std::move(_n.lower_states_);
		// this->init();
#ifdef NQS_NOT_OMP_MT
		this->initThreads(_n.threads_.threadNum_);
#endif
	}
	NQS &operator=(const NQS & _n)
	{	
		this->info_p_				= _n.info_p_;
		this->H_					= _n.H_;
		this->info_					= _n.info_;
		this->pBar_					= _n.pBar_;
		this->ran_					= _n.ran_;
		this->nFlip_				= _n.nFlip_;
		this->flipPlaces_			= _n.flipPlaces_;
		this->flipVals_				= _n.flipVals_;
		// initialize the information
		this->info_p_				= _n.info_p_;
		this->lower_states_			= _n.lower_states_;

		this->init();
#ifdef NQS_NOT_OMP_MT
	this->initThreads(_n.threads_.threadNum_);
#endif
		return *this;
	}
	NQS &operator=(NQS &&_n)
	{
		this->info_p_				= _n.info_p_;
		this->H_					= _n.H_;
		this->info_					= _n.info_;
		this->pBar_					= _n.pBar_;
		this->ran_					= _n.ran_;
		this->nFlip_				= _n.nFlip_;
		this->flipPlaces_			= _n.flipPlaces_;
		this->flipVals_				= _n.flipVals_;
		// initialize the information
		this->info_p_				= std::move(_n.info_p_);
		this->lower_states_			= std::move(_n.lower_states_);

		this->init();
#ifdef NQS_NOT_OMP_MT
		this->initThreads(_n.threads_.threadNum_);
#endif
		return *this;
	}
	NQS(std::shared_ptr<Hamiltonian<_Ht,_spinModes>> &_H, double _lr = 1e-2,
		uint _threadNum = 1, int _nParticles = -1, 
		const NQSLS_p& _lower = {}, const std::vector<double>& _beta = {});
};

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setRandomState(bool _upd)
{ 
	if (this->discVal_ == 0.5) {
#ifdef SPIN
		NQSS randomState = arma::randi<NQSS>(this->info_p_.nVis_, arma::distr_param(0, 1)) - 0.5;
#else
		NQSS randomState = arma::randi<NQSS>(this->info_p_.nVis_, arma::distr_param(0, 1)) * 0.5;
#endif
		this->setState(randomState, _upd);
	} else if (this->discVal_ == 1){
		NQSS randomState = arma::randi<NQSS>(this->info_p_.nVis_, arma::distr_param(0, 1));
		this->setState(randomState, _upd);
	} else {
		// set from integer
		this->setState(this->ran_.template randomInt<u64>(0, this->info_p_.Nh_ - 1), _upd);
	}
};

// ##########################################################################################################################################

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
bool NQS<_spinModes, _Ht, _T, _stateType>::initThreads(uint _threadNum)
{
#ifndef NQS_USE_MULTITHREADING
		this->threads_.threadNum_		=	1;
#else
		this->threads_.threadNum_		=	std::min(_threadNum, this->info_p_.nSites_);
		this->threads_.threadsNumLeft_	=	std::max(_threadNum - this->threads_.threadNum_, (uint)1);
#endif
	// Use threads for all consecutive parallel regions
#if defined NQS_USE_MULTITHREADING
#	ifdef NQS_USE_OMP
		omp_set_num_threads(this->threadNum_);   
#	else
		// initialize the threads
		this->threads_.threads_.reserve(this->threads_.threadNum_);
		this->threads_.kernels_	=			v_1d<CondVarKernel<_T>>(this->threads_.threadNum_);
		
		// calculate how many sites goes to one thread
		uint _siteStep			=			std::ceil(this->info_p_.nSites_ / 1.0 / this->threads_.threadNum_);

		// start the threads that calculate the energy with the local energy kernel function
		// this function waits for the specific energy calculation to be ready on each thread
		// this is handled through "flagThreadRun_" member
		for (uint i = 0; i < this->threads_.threadNum_; i++)
		{
			std::function<void()> lambda = [this, i, _siteStep]() 
				{ 
					this->locEnKernel(i * _siteStep, std::min((i + 1) * _siteStep, this->info_p_.nSites_), i); 
				};
			this->threads_.threads_.emplace_back(std::thread(lambda));
		}
#	endif
#endif
	return true;
}

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
	auto _isSaved = saveAlgebraic(_path, _file, this->F_, "weights");

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
	return loadAlgebraic(_path, _file, this->F_, "weights");
}

// ##########################################################################################################################################

// ########################################################### C O N S T R U C T ############################################################

// ##########################################################################################################################################

/*
* @brief Allocates the main gradient parameters and the temporary and current vectors.
* The vectors are set to ones for the start.  
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::allocate()
{
	// allocate gradients
	this->F_.resize(this->info_p_.fullSize_);
#ifdef NQS_USESR
#	ifndef NQS_USESR_NOMAT
	this->S_.resize(this->info_p_.fullSize_, this->info_p_.fullSize_);
#	endif
#endif
	this->curVec_ = arma::ones(this->info_p_.nVis_);
	this->tmpVec_ = arma::ones(this->info_p_.nVis_);
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
#if defined NQS_NOT_OMP_MT
	for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++)
	{
		std::unique_lock<std::mutex> lock(this->threads_.kernels_[_thread].mutex);
		this->threads_.kernels_[_thread].flagThreadKill_	= true;
		this->threads_.kernels_[_thread].end_				= true;
		this->threads_.kernels_[_thread].flagThreadRun_		= 1;
		this->threads_.kernels_[_thread].cv.notify_all();
	}
	for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++)
		if (this->threads_.threads_[_thread].joinable())
			this->threads_.threads_[_thread].join();
#endif
	// ######################################################################################################################################
	if (this->precond_ != nullptr) {
		delete this->precond_;
		this->precond_ = nullptr;
	}

	if (this->solver_ != nullptr) {
		delete this->solver_;
		this->solver_ = nullptr;
	}
	// ######################################################################################################################################
}

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

// ########################################################### C O N S T R U C T ############################################################

// ##########################################################################################################################################

/*
* @brief General constructor of the NQS solver
* @param _H Hamiltonian to be used for correct data sampling
* @param _lr learning rate to be used for the training
* @param _threadNum thread number to be used for the solver - available on setting corresponding thread pragma
* @param _lower 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline NQS<_spinModes, _Ht, _T, _stateType>::NQS(std::shared_ptr<Hamiltonian<_Ht, _spinModes>>& _H, 
													double _lr, 
													uint _threadNum, 
													int _nParticles,
													const NQSLS_p& _lower, 
													const std::vector<double>& _beta)
	: H_(_H)
{	
	const size_t _Ns			= 			_H->getNs();	
	this->pRatioFunc_			= 			[&](const NQSS& _v) { return this->pRatio(_v); };
	this->pKernelFunc_			= 			[&](std::initializer_list<int> fP, std::initializer_list<double> fV) { return this->pRatio(fP, fV); };

	this->lower_states_			= 			NQS_lower_t<_spinModes, _Ht, _T, _stateType>(_Ns, _lower, _beta, this);
	this->lower_states_.exc_ratio_ = 		[&](const NQSS& _v) { return this->pRatio(_v); };
#ifdef NQS_LOWER_RATIO_LOGDIFF
	this->lower_states_.exc_ansatz_ = 		[&](const NQSS& _v) { return this->ansatzlog(_v); };
#else
	this->lower_states_.exc_ansatz_ = 		[&](const NQSS& _v) { return this->ansatz(_v); };
#endif
	this->info_p_.lr_			= 			_lr;

	// set the number of particles
	// set the visible layer (for hardcore-bosons we have the same number as sites but fermions introduce twice the complication)
	this->info_p_.nVis_ 		= 			_Ns * (this->spinModes_ / 2);
	this->info_p_.nSites_		=			_Ns;

	// make it half filling if necessary
	this->info_p_.nParticles_	=			(_nParticles < 0 || this->spinModes_ == 2) ? this->info_p_.nSites_ : (uint)_nParticles;
	// check the Hilbert space
	this->info_p_.Nh_			=			_H->getHilbertSize();
	// set the random number generator
	this->ran_					=			_H->ran_;
	// set threads
	this->initThreads(_threadNum);

	LOGINFO("Constructed the general NQS class", LOG_TYPES::TRACE, 2);
};

#endif