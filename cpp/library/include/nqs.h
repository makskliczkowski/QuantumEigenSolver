/***************************************
* Defines the generic NQS Solver class. 
* Allows for later inhertiance
* for a fine model specialization.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#include "armadillo"
#include <memory>
#include <stdexcept>
#ifndef NQS_H
#define NQS_H

// include all the definions
#include "NQS/nqs_definitions_base.h"
#include "NQS/nqs_definitions_lower.h"


/*
* @brief General Neural Network Quantum States eigensolver class - base
* Contains all the necessary methods to override for the specific NQS model.
*/
template <uint _spinModes, 
		  typename _Ht,
		  typename _T					= _Ht, 
		  class _stateType				= double>
class NQS 
{
	// type definitions 
	NQS_PUBLIC_TYPES(_T, double);
	using NQSLS_p						=		NQS_lower_t<_spinModes, _Ht, _T, _stateType>::NQSLS_p;	// for the lower states information
public:
	NQS_info_t info_p_;													// information about the NQS
	NQS_lower_t<_spinModes, _Ht, _T, _stateType> lower_states_;			// information about the training

protected:
	// for the Hamiltonian information, types and the Hilbert space
	const uint spinModes_				=		_spinModes;				// number of spin modes -> e.g. 2 for hardcore bosons, 4 for fermions
	const double discVal_				=		Operators::_SPIN_RBM;   // discrete value for the numbers in vector representation
	std::shared_ptr<Hamiltonian<_Ht, _spinModes>> H_;					// pointer to the Hamiltonian instance (for the energy calculation)

protected:
	std::string info_					=		"General NQS";
	pBar pBar_;															// for printing out the progress

	// Random number generator
	randomGen ran_;														// consistent quick random number generator

	/* ----------------------------------------------------------- */

	// ---------------------- T H R E A D I N G ---------------------
							
	bool initThreads(uint _threadNum = 0);
	NQS_thread_t<_T> threads_;												// thread information

	/* ----------------------------------------------------------- */

	// ----------------------- T R A I N I N G ----------------------
	uint nFlip_							=		1;						// number of flips to be done in one step (each flip is a change in the state)
	v_1d<uint> flipPlaces_;												// stores flip spots to be flipped during one sampling step
	v_1d<_stateType> flipVals_;											// stores values before (!!!) the flip to be used for the gradients
	
	NQSS curVec_;														// currently processed state vector for convenience
	u64 curState_						=		0;						// currently processed state - may or may not be used
	
	// temporary placeholders for the vectors
	v_1d<NQSS> tmpVecs_;												// temporary vectors for the flips
	u64 tmpState_						=		0;						// temporary state for the flips
	NQSS tmpVec_;														// temporary vector for the flips (for the current state)
	
	// ------------------------ W E I G H T S -----------------------
	NQSW derivatives_;													// store the variational derivatives F_k (nBlocks x fullSize), where nBlocks is the number of consecutive observations
	NQSW derivativesC_;													// derivatives conjugated (F_k^*) - for the SR (nBlocks x fullSize), where nBlocks is the number of consecutive observations
	void derivativesReset(size_t nBlocks = 1)							{ this->derivatives_ = NQSW(nBlocks, this->info_p_.fullSize_, arma::fill::zeros); this->derivativesC_ = this->derivatives_; };
#ifdef NQS_USESR
#	ifndef NQS_USESR_NOMAT
	NQSW S_;															// positive semi-definite covariance matrix - to be optimized (inverse of the Fisher information matrix)
#	else
	NQSB derivativesM_;													// store the mean of the derivatives (F_k) (fullSize)
#	endif
#endif
	NQSB F_;															// forces acting on the weights (F_k)

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
	virtual void chooseRandomFlips()					=			0;	// important for the flipPlaces_ and flipVals_ to be set!

	// apply flips to the temporary vector or the current vector according the template
	virtual void applyFlipsT()							=			0;	// apply flips to the temporary vector
	virtual void applyFlipsC()							=			0;	// apply flips to the current vector
	
	virtual void unapplyFlipsT()						{ this->applyFlipsT(); }; // unapply flips of the temporary vector according the template
	virtual void unapplyFlipsC()						{ this->applyFlipsC(); }; // unapply flips of the current vector according the template
	
	virtual void setRandomFlipNum(uint _nFlips)			=			0;	// set the number of flips to be done

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

	// ratio when only 1 (single) flip is used and flips are stored inside (flipPlaces_ & flipVals_)
	virtual auto pRatio()								-> _T			{ return this->pRatio(this->flipPlaces_[0], this->flipVals_[0]);	};
	// ratio when only 1 (single) flip is used and flips are provided to the function (fP, fV)
	virtual auto pRatio(uint fP, float fV)				->_T			= 0;
	// ratio when multiple flips are used, provided the number of flips taken
	virtual auto pRatio(uint nFlips)					->_T			= 0;
	// ratio when a new vector is provided (after flip) - (uses current vector as well)
	virtual auto pRatio(const NQSS& _v)					->_T			{ return this->pRatio(this->curVec_, _v);							};			
	// ratio when a new vector is provided (after flip) - (uses two different vectors - one for the current state and one for the new state)
	virtual auto pRatio(const NQSS& _v1,
						const NQSS& _v2)				->_T			= 0;
	// ratio when exact points are provided (used for the Hamiltonian probability ratio - when the Hamiltonian changes the state)
	virtual auto pRatio(std::initializer_list<int> fP, 
						std::initializer_list<double> fV)->_T			= 0;

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

protected:
	// --------------------- T R A I N   E T C -----------------------
	virtual void grad(const NQSS& _v, uint _plc)		=				0;
	virtual void gradFinal(const NQSB& _energies);
#ifdef NQS_USESR
	// stochastic reconfiguration
	virtual void gradSR(uint step = 0);

	#ifdef NQS_USESR_NOMAT
	
	auto getSRMatrixElement(size_t i, size_t j)							-> _T; 		// stochastic reconfiguration without the matrix construction
	auto getSRMatVec(const arma::Col<_T>& x, arma::Col<_T>& y, size_t) 	-> void;	// matrix-vector multiplication for the SR

	// helping variables for conjugate gradient method
	NQSB r_;															// residual
	NQSB p_;															// search direction
	NQSB Ap_;															// matrix-vector multiplication result
	NQSB x_;															// solution

	#endif
#endif
	// ---------------------------------------------------------------
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
#ifdef NQS_NOT_OMP_MT
	virtual void locEnKernel(uint _start, uint _end, uint _threadNum);
#endif

	/* ------------------------------------------------------------ */

public:
	// ------------------------ S E T T E R S ------------------------
	virtual void init()									=				0; 
	virtual void setRandomState(bool _upd = true)						{ this->setState(this->ran_.template randomInt<u64>(0, this->info_p_.Nh_), _upd);	};
	virtual double setNormalization();

	/* ------------------------------------------------------------ */

	// ------------------------ G E T T E R S ------------------------
	auto getInfo()								const -> std::string	{ return this->info_;					};
	auto getNvis()								const -> uint			{ return this->info_p_.nVis_;					};
	auto getF()									const -> NQSB			{ return this->F_;						};
	auto getCovarianceMat()						const -> NQSW			{ return this->S_;						};	
	// Hilbert
	auto getHilbertSize()						const -> u64			{ return this->info_p_.Nh_;						};
	// Hamiltonian
	auto getHamiltonianInfo()					const -> std::string	{ return this->H_->getInfo();			};
	auto getHamiltonianEigVal(u64 _idx)			const -> double			{ return this->H_->getEigVal(_idx);		};
	auto getHamiltonian() const -> std::shared_ptr<Hamiltonian<_Ht>>	{ return this->H_;						};
	auto getHilbertSpace() const -> Hilbert::HilbertSpace<_Ht>			{ return this->H_->getHilbertSpace();	};
	auto getNorm()								const -> double			{ return this->info_p_.norm_ <= 0 ? this->setNormalization() : this->info_p_.norm_;	};

	// ----------------------- S A M P L I N G -----------------------
	virtual void blockSample(uint _bSize, NQS_STATE_T _start, bool _therm = false);

	virtual arma::Col<_T> train(const NQS_train_t& _par,
								bool quiet			= false,			// shall talk? (default is false)
								clk::time_point _t	= NOW,				// time! (default is NOW)
								uint progPrc		= 25);

	virtual arma::Col<_T> collect(const NQS_train_t& _par,
								  bool quiet						= false,
								  clk::time_point _t				= NOW,
								  NQSAv::MeasurementNQS<_T>& _mes 	= {},
								  bool _collectEn					= true);
	virtual void collect(const NQS_train_t& _par, NQSAv::MeasurementNQS<_T>& _mes);


	// ----------------------- F I N A L E -----------------------
	virtual auto ansatz(const NQSS& _in)		const ->_T				= 0;

	// -------------------- C O N S T R U C T --------------------

public:
	virtual ~NQS();
	NQS() = default;
	NQS(const NQS& _n)
		: info_p_(_n.info_p_), spinModes_(_n.spinModes_), discVal_(_n.discVal_), H_(_n.H_), info_(_n.info_), pBar_(_n.pBar_), 
		ran_(_n.ran_), nFlip_(_n.nFlip_), flipPlaces_(_n.flipPlaces_), flipVals_(_n.flipVals_)
	{
		this->threads_ 		= _n.threads_;
		// initialize the information 
		this->info_p_  		= _n.info_p_;
		this->lower_states_ = _n.lower_states_;
		this->init();
	}
	NQS(NQS&& _n)
		: info_p_(_n.info_p_), spinModes_(_n.spinModes_), discVal_(_n.discVal_), H_(_n.H_), info_(_n.info_), pBar_(_n.pBar_), 
		ran_(_n.ran_), nFlip_(_n.nFlip_), flipPlaces_(_n.flipPlaces_), flipVals_(_n.flipVals_)
	{
		this->threads_ 		= std::move(_n.threads_);
		// initialize the information
		this->info_p_  		= std::move(_n.info_p_);
		this->lower_states_ = std::move(_n.lower_states_);
		this->init();
	}
	NQS &operator=(const NQS & _n)
	{	
		this->info_p_				= _n.info_p_;
		this->spinModes_			= _n.spinModes_;
		this->discVal_				= _n.discVal_;
		this->H_					= _n.H_;
		this->info_					= _n.info_;
		this->pBar_					= _n.pBar_;
		this->ran_					= _n.ran_;
		this->threadNum_			= _n.threadNum_;
		this->nFlip_				= _n.nFlip_;
		this->flipPlaces_			= _n.flipPlaces_;
		this->flipVals_				= _n.flipVals_;
		// initialize the information
		this->info_p_				= _n.info_p_;
		this->lower_states_			= _n.lower_states_;
		this->threads_				= _n.threads_;		

		this->init();
		return *this;
	}
	NQS &operator=(NQS &&_n)
	{
		this->info_p_				= _n.info_p_;
		this->spinModes_			= _n.spinModes_;
		this->discVal_				= _n.discVal_;
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
		this->threads_				= std::move(_n.threads_);

		this->init();
		return *this;
	}
	NQS(std::shared_ptr<Hamiltonian<_Ht>> &_H, double _lr = 1e-2,
		uint _threadNum = 1, int _nParticles = -1, 
		const NQSLS_p& _lower = {}, std::vector<double> _beta = {});
};

// ##########################################################################################################################################

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
double NQS<_spinModes, _Ht, _T, _stateType>::setNormalization [[deprecated]] () 
{
	// calculate the normalization factor
	double _norm = 0.0;
	for (u64 i = 0; i < this->info_p_.Nh_; i++)
	{
		Binary::int2base(i, this->curVec_, this->discVal_);
		auto _val 	= 	this->ansatz(this->curVec_);
		_norm 		+= 	algebra::real(_val * algebra::conjugate(_val));
	}

	this->info_p_.norm_ = _norm;
	return _norm;
}

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
	#ifndef NQS_USESR_NOMAT
	this->S_.resize(this->info_p_.fullSize_, this->info_p_.fullSize_);
	#else
	{
		this->r_ 	= NQSB(this->info_p_.fullSize_, arma::fill::zeros);
		this->p_ 	= NQSB(this->info_p_.fullSize_, arma::fill::zeros);
		this->Ap_ 	= NQSB(this->info_p_.fullSize_, arma::fill::zeros);
		this->x_ 	= NQSB(this->info_p_.fullSize_, arma::fill::zeros);
	}
	#endif
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
inline void NQS<_spinModes, _Ht, _T, _stateType>::blockSample(uint _bSize, NQS_STATE_T _start, bool _therm)
{
	// check whether we should set a state again or thermalize the whole process (applies on integer state)
	// Set state based on whether thermalization is required or _start differs from current state
	if (_therm 
#ifndef NQS_USE_VEC_ONLY
		|| _start != this->curState_
#endif
		) this->setState(_start, _therm);

	// set the temporary state - the vectors are usefull so set them accordingly
	this->tmpVec_ = this->curVec_;

	for (uint bStep = 0; bStep < _bSize; ++bStep) // go through each block step
	{
		this->chooseRandomFlips(); 	// set the random flip sites - it depends on a given implementation of the NQS
		this->applyFlipsT();		// flip the vector - use temporary vector tmpVec to store the flipped vector

		// check the probability (choose to use the iterative update of presaved weights [the angles previously updated] or calculate ratio from scratch)
#ifndef NQS_ANGLES_UPD
		double proba = std::abs(this->pRatio(this->curVec_, this->tmpVec_));
#else
		double proba = std::abs(this->pRatio(this->nFlip_));
#endif
		// we need to take into account the probability comming from the ratio of states (after and before the flip)
		if (this->ran_.template random<float>() <= proba * proba)
		{
			// update current state and vector when the flip has been accepted (the probability is higher than the random number)
			this->applyFlipsC();
			// update angles if needed
			this->update(this->nFlip_);
		}
		// set the vector back to normal (unflip)
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

	// Set the state again if angles update is disabled
#ifndef NQS_ANGLES_UPD
	this->setState(NQS_STATE, true);
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
#ifdef NQS_USE_OMP
	{
		double energyR	= 0.0;
		double energyI	= 0.0;

#ifndef _DEBUG
		#pragma omp parallel for reduction(+ : energyR, energyI) num_threads(this->threadNum_)
#endif
		for (int site = 0; site < this->info_p_.nSites_; site++) 
		{
			auto energy	=	this->H_->locEnergy(NQS_STATE, 
												site, 
												std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
												this,
												std::placeholders::_1,
												std::placeholders::_2));
			energyR		+=	algebra::real(energy);
			energyI		+=	algebra::imag(energy);
		}
		return toType<_T>(energyR, energyI);
	}
#else
	{
		_T energy = 0.0;
		for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++) // run all threads
		{
			std::lock_guard<std::mutex> lock(this->threads_.kernels_[_thread].mutex);
			this->threads_.kernels_[_thread].flagThreadRun_	= true;
			this->threads_.kernels_[_thread].end_			= false;
			this->threads_.kernels_[_thread].cv.notify_one();
		}

		for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++) // wait for all threads
		{
			{
				std::unique_lock<std::mutex> lock(this->threads_.kernels_[_thread].mutex);
				this->threads_.kernels_[_thread].cv.wait(lock, [this, _thread] { return !this->threads_.kernels_[_thread].flagThreadRun_; });
			}
			energy += this->threads_.kernels_[_thread].kernelValue_;
		}

		// for the lower states - only if the lower states are used
		{
			for (int _low = 0; _low < this->lower_states_.f_lower.size(); _low++)
			{
				// go through the other overlapping states 
				
			}
		}

		return energy;
	}
#endif
}

///////////////////////////////////////////////////////////////////////

#ifdef NQS_NOT_OMP_MT
/*
* @brief Allows to run a thread pool based on the condition that all threads wait for a mutex to further run the program.
* The threads are killed when the program is finished. The function calculates the local energy for a given thread.
* @param _start starting site for a given thread
* @param _end ending site for a given thread
* @param _threadNum number of the thread currently run
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::locEnKernel(uint _start, uint _end, uint _threadNum)
{
	while (!this->threads_.kernels_[_threadNum].flagThreadKill_)	// does not go in if the simulation is finished
	{
		// wait for the lock to end
		{
			// aquire mutex lock as required by condition variable
			std::unique_lock<std::mutex> lock(this->threads_.kernels_[_threadNum].mutex);	
			// thread will suspend here and release the lock if the expression does not return true
			this->threads_.kernels_[_threadNum].cv.wait(lock, [this, _threadNum] { return this->threads_.kernels_[_threadNum].flagThreadRun_; });	

			if (this->threads_.kernels_[_threadNum].flagThreadKill_)
			{
				this->threads_.kernels_[_threadNum].end_ = true;
				break;
			}
		}

		// Process the work
		this->threads_.kernels_[_threadNum].kernelValue_ = 0.0;
		for (auto site = _start; site < _end; ++site)
		{
			this->threads_.kernels_[_threadNum].kernelValue_ += algebra::cast<_T>(this->H_->locEnergy(NQS_STATE,
																							site, 
																							std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
																							this,
																							std::placeholders::_1,
																							std::placeholders::_2)));
		}

		// lock again
		{
			std::lock_guard<std::mutex> lock(this->threads_.kernels_[_threadNum].mutex);
			this->threads_.kernels_[_threadNum].flagThreadRun_	= false;
			this->threads_.kernels_[_threadNum].end_				= true; 
		}
		this->threads_.kernels_[_threadNum].cv.notify_one(); // Notify waiting threads if needed
	}
}
#endif

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

// -----------------------------------------------------------------------------------------------------------------------------------
#ifdef NQS_USESR

	#ifdef NQS_USESR_NOMAT
// -----------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Calculates the matrix element of the geometric tensor S_ij = <\Delta _i* \Delta _j> - <\Delta _i*><\Delta _j> 
* without constructing the matrix explicitly. This is done by calculating the matrix element directly. 
* @param i row index - the matrix is symmetric
* @param j column index - the matrix is symmetric
* @returns the matrix element of the geometric tensor S_ij
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T NQS<_spinModes, _Ht, _T, _stateType>::getSRMatrixElement(size_t i, size_t j)
{
	// get the subviews of the columns (each column has multiple observations now)
	auto _colO_i 		= this->derivativesC_.col(i);
	auto _colO_j 		= this->derivatives_.col(j);

	// compute <\Delta _i* \Delta _j>
	_T avg_Oi_Oj 		= arma::dot(_colO_i, _colO_j) / static_cast<double>(_colO_i.n_elem);

	// compute <\Delta _i*><\Delta _j>
	_T avg_Oi_avg_Oj 	= algebra::conjugate(this->derivativesM_(i)) * this->derivativesM_(j);

		#ifdef NQS_SREG
		{
		// regularize the covariance matrix before inverting it (if needed and set) by adding the regularization term
		if (i == j)
			avg_Oi_Oj += this->covMatrixRegStart * avg_Oi_avg_Oj;
		}
		#endif
	return avg_Oi_Oj - avg_Oi_avg_Oj;
}

// -----------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Calculates the matrix-vector multiplication using the elements obtained from the getSRMatrixElement. 
* This is done without constructing the matrix explicitly. Once the matrix is known, the multiplication is done
* by calculating the matrix-vector multiplication directly.
* @param x vector to be multiplied with the matrix
* @param y resulting vector from the multiplication
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::getSRMatVec(const arma::Col<_T>& x, arma::Col<_T>& y, size_t n)
{	
	// implement the matrix-vector multiplication using the elements obtained from the getSRMatrixElement
	// #pragma omp parallel for num_threads(this->threadNum_)
	for (size_t i = 0; i < n; ++i)
	{
		// initialize the y vector
		y(i) = 0.0;

		// y_i = \sum _j S_ij x_j
		// Compute the matrix-vector multiplication using only upper triangular matrix and reflection
		// as the matrix is square and symmetric
		for (size_t j = 0; j < n; j++)
			y(i) += this->getSRMatrixElement(i, j) * x(j);
// 		for (size_t j = i; j < n; j++) {
//             const _T S_ij = this->getSRMatrixElement(i, j);
//             y(i) += S_ij * x(j);

//             // Since S is symmetric, reflect the result to y(j)
//             if (i != j) {
// // #pragma omp atomic
//                 y(j) += S_ij * x(i);
//             }
//         }
	}

}
// ----------------------------------------------------------------------------------------------------------------------------------- 
	#endif 

/*
* @brief Calculates the update parameters for weights to be updated in a form:
* weights[new] <- weights[old] - lr * S^{-1} * F --> second order optimization method (stochastic reconfiguration)
* It uses the matrix S = <\Delta _k* \Delta _k> - <\Delta _k*><\Delta _k> to calculate the update. This matrix is 
* called the geometric tensor. It is used as a regularizer for the gradient descent method.
* This method can be either calculated with the pseudoinverse or with the direct inversion of the matrix.
* Otherwise, one can skip the stochastic reconfiguration and use the standard gradient descent.
* @note The method can be run without calculating the geometric tensor S explicitly.
* @param step current step of updating - for the regularization purpose
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradSR(uint step)
{
	#ifndef NQS_USESR_NOMAT
	{
		// regularize the covariance matrix before inverting it (if needed and set)
		{
		#ifdef NQS_SREG
		this->covMatrixReg();
		#endif
		}
		
		#ifdef NQS_PINV
		{
			// calculate the pseudoinverse
			if (NQS_PINV > 0)
				this->F_ = this->info_p_.lr_ * (arma::pinv(this->S_, NQS_PINV) * this->F_);
			else
				this->F_ = this->info_p_.lr_ * (arma::pinv(this->S_) * this->F_);
			return;
		}
		#else 
		// solve normally
		//this->F_ = this->info_p_.lr_ * (arma::inv(this->S_) * this->F_);
		this->F_ = this->info_p_.lr_ * arma::solve(this->S_, this->F_, arma::solve_opts::likely_sympd);
		#endif 
	}
	#else
	{
		std::function<void(const arma::Col<_T>&, arma::Col<_T>&, size_t)> _Fun = std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::getSRMatVec, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

		arma::Col<_T> _F = this->F_;

		// solve the system manually
		algebra::Solvers::solve(this->F_, _Fun, _F, algebra::Solvers::SolverType::MY_CONJ_GRAD, 1.0e-5);
		// algebra::Solvers::ConjugateGradient::solve_my_conj_grad<_T>(this->F_, _Fun, this->x_, this->r_, this->p_, this->Ap_, 1e-6);

		_F *= this->info_p_.lr_;
		// exchange the vectors
		this->F_ = std::move(_F);
		// this->F_ = this->info_p_.lr_ * this->x_;
	}
	#endif
}
// -----------------------------------------------------------------------------------------------------------------------------------
#endif

///////////////////////////////////////////////////////////////////////

/*
* @brief Calculate the final step for the weights to be updated.
* First of all, it calculates the generalized forces for the weights.
* The forces are calculated as the covariance of the derivatives and the local energies.
* The forces are then used to update the weights.
* 
* @warning Uses the forces vector (member F_) to store the forces that update the weights
* 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::gradFinal(const NQSB& _energies)
{
	// calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}> [+ sum_i ^{n-1} \beta _i <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> <Psi _W/Psi_W(i)>]
	this->derivativesC_ = arma::conj(this->derivatives_);				// precalculate the conjugate of the derivatives
	this->F_			= arma::cov(this->derivativesC_, _energies, 1);	// calculate the covariance vector for the gradient !TODO modify this for excited states! 
#ifdef NQS_USESR
	{
	#ifndef NQS_USESR_NOMAT
		// form the covariance matrix explicitly
		this->S_		= arma::cov(this->derivativesC_, this->derivatives_, 1);
	#else 
		{
		arma::Mat<_T> S1 	= arma::cov(this->derivativesC_, this->derivatives_, 0);
		std::cout << "Covariance matrix: " << S1.n_rows << ", " << S1.n_cols << std::endl;


		// calculate the matrix without forming it explicitly and use the mean of the derivatives
		this->derivativesM_ = arma::mean(this->derivatives_, 0).as_col();
		std::cout << "Mean of the derivatives: " << this->derivativesM_.n_elem << std::endl;

		arma::Mat<_T> S2	= S1;
		S2.zeros();

		for (size_t i = 0; i < this->derivativesC_.n_cols; ++i)
			for (size_t j = 0; j < this->derivativesC_.n_cols; ++j)
				S2(i, j) = this->getSRMatrixElement(i, j);
		// check the difference
		arma::Mat<double> diff = arma::abs(S1 - S2);
		diff.print("Difference: ");
		}
	#endif
		// update model by recalculating the gradient (applying the stochastic reconfiguration)
		this->gradSR(0);
	}
#else
	{
		// standard updater with the gradient only!
		this->F_ *= this->info_p_.lr_;
	}
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
* @param mcSteps Monte Carlo steps to be used in the training - this is an outer loop for the training (number of iterations)
* @param nThrm Monte Carlo steps spent for thermalizing the system - burn-in
* @param nBlck	number of blocks to be used for calculating the covariance matrix - number of iterations for the gradient calculation (single iteration step)
* @param bSize number of update steps within a single block - size of the block (number of flip trials (global))
* @param nFlip number of flips for a single update of a state (default is 1)
* @param quiet wanna talk? (default is false)
* @param _t timepoint for timestamping the training
* @param progPrc progress percentage to be displayed in the progress bar
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline arma::Col<_T> NQS<_spinModes, _Ht, _T, _stateType>::train(const NQS_train_t& _par,
																bool quiet,
																clk::time_point _t,
																uint progPrc)
{
	{
		// make the pbar!
		this->pBar_ = pBar(progPrc, _par.mcSteps);

		// set the info about training
		_par.hi();

		this->derivativesReset(_par.nBlck);
	}
	TIMER_CREATE(_timer);

	// save all average weights for covariance matrix
	arma::Col<_T> meanEn(_par.mcSteps, arma::fill::zeros);
	// history of energies - here we save the local energies at each block
	arma::Col<_T> En(_par.nBlck, arma::fill::zeros);
	// set the random state at the begining and the number of flips
	{
		this->setRandomState();
		this->setRandomFlipNum(_par.nFlip);
	}

	// go through the Monte Carlo steps
	for (uint i = 1; i <= _par.mcSteps; ++i)
	{
		// set the random state at the begining
		this->setRandomState();
		// thermalize
		this->blockSample(_par.nThrm, NQS_STATE, false);

		// iterate blocks - this ensures the calculation of a stochastic gradient 
		for (uint _taken = 0; _taken < _par.nBlck; ++_taken) {

			// sample them!
			this->blockSample(_par.bSize, NQS_STATE, false);

			// calculate the gradient at each point of the iteration! - this is implementation specific!!!
			this->grad(this->curVec_, _taken);

			// local energy - stored at each point within the estimation of the gradient (stochastic)
			En(_taken) = this->locEnKernel();
		}
		
		// calculate the final update vector - either use the stochastic reconfiguration or the standard gradient descent !TODO: implement optimizers
		TIMER_START_MEASURE(this->gradFinal(En), (i % this->pBar_.percentageSteps == 0), _timer, STR(i));

		// finally, update the weights with the calculated gradient (force) [can be done with the stochastic reconfiguration or the standard gradient descent] - implementation specific!!!=
		this->updateWeights();
		
		// save the mean energy
		meanEn(i - 1) = arma::mean(En);
		LOGINFO(VEQ(meanEn(i - 1)), LOG_TYPES::TRACE, 1);

		// update the progress bar
		PROGRESS_UPD_Q(i, this->pBar_, "PROGRESS NQS", !quiet);
		
#ifdef NQS_SAVE_WEIGHTS
		{
		if (i % this->pBar_.percentageSteps == 0)  
			this->saveWeights(_par.dir + NQS_SAVE_DIR, "weights.h5");
		}
#endif
	}
	LOGINFO(_t, "NQS_EQ", 1);
	return meanEn;
}

// ##########################################################################################################################################

// ########################################################## C O L L E C T I O N ###########################################################

// ##########################################################################################################################################

/*
* @brief Collects the samples for the given number of Monte Carlo steps. The samples are collected in the form of the measurements
* of the operator. The operator is given as a function that calculates the probability of the operator.
* @param _par parameters for the training
* @param _meas measurement object to store the measurements
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>:: collect(const NQS_train_t& _par, NQSAv::MeasurementNQS<_T>& _meas)
{
	// set the random state at the begining
	this->setRandomFlipNum(_par.nFlip);

	// allows to calculate the probability of the operator (for operator measurements)
	std::function<_T(const NQSS& _v)> opFun = [&](const NQSS& v) { return this->pRatio(v); };
	
	// go through the number of samples to be collected
	for (uint i = 1; i <= _par.mcSteps; ++i)
	{	
		// random flip
		this->setRandomState();

		// remove autocorrelations
		this->blockSample(_par.nThrm, NQS_STATE, false);

		// iterate blocks - allows to collect samples outside of the block
		for (uint _taken = 0; _taken < _par.nBlck; ++_taken) 
		{
			// sample them!
			this->blockSample(_par.bSize, NQS_STATE, false);
			
			// measure 
			_meas.measure(this->curVec_, opFun);
		}
		// normalize the measurements - this also creates a new block of measurements
		_meas.normalize(_par.nBlck);												
	}
}

/*
* @brief Collects the samples for the given number of Monte Carlo steps. The samples are collected in the form of the measurements
* of the operator. The operator is given as a function that calculates the probability of the operator.
* @param _par parameters for the training
* @param _meas measurement object to store the measurements
* @param _collectEn collect the energy as well
* @param _t timepoint for timestamping the training
* @param quiet wanna talk? (default is false)
* @returns the mean energy of the system
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline arma::Col<_T> NQS<_spinModes, _Ht, _T, _stateType>::collect(	const NQS_train_t& _par,
																	bool quiet,
																	clk::time_point _t,
																	NQSAv::MeasurementNQS<_T>& _meas,
																	bool _collectEn)
{																							
	{
		this->pBar_	= pBar(20, _par.mcSteps);
		_par.hi("Collect: ");
	}
	TIMER_CREATE(_timer);

	arma::Col<_T> meanEn, En;
	if (_collectEn)
	{
		meanEn 	= arma::Col<_T>(_par.mcSteps, arma::fill::zeros);
		En 		= arma::Col<_T>(_par.nBlck, arma::fill::zeros);
	}

	// set the random state at the begining
	this->setRandomFlipNum(_par.nFlip);

	// allows to calculate the probability of the operator (for operator measurements)
	std::function<_T(const NQSS& _v)> opFun = [&](const NQSS& v) { return this->pRatio(v); };
	
	// go through the number of samples to be collected
	for (uint i = 1; i <= _par.mcSteps; ++i)
	{
		// random flip
		this->setRandomState();

		// remove autocorrelations
		this->blockSample(_par.nThrm, NQS_STATE, false);

		// iterate blocks - allows to collect samples outside of the block
		for (uint _taken = 0; _taken < _par.nBlck; ++_taken) 
		{
			// sample them!
			this->blockSample(_par.bSize, NQS_STATE, false);

			if (_collectEn) En(_taken) = this->locEnKernel();
			
			// collect other										// one can calculate the local energy here (either of the ground state or the excited state)
			TIMER_START_MEASURE(_meas.measure(this->curVec_, opFun), (i % this->pBar_.percentageSteps == 0 && _taken == 0), _timer, STR(i)); 	
			//TIMER_START_MEASURE(_meas.measure(BASE_TO_INT<u64>(this->curVec_, this->discVal_), opFun), (i % this->pBar_.percentageSteps == 0 && _taken == 0), _timer, STR(i)); 	
		}

		_meas.normalize(_par.nBlck);												// normalize the measurements
		meanEn(i - 1) = arma::mean(En); 											// save the mean energy
		PROGRESS_UPD_Q(i, this->pBar_, "PROGRESS NQS", !quiet); 					// update the progress bar

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
* @param _lower 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline NQS<_spinModes, _Ht, _T, _stateType>::NQS(std::shared_ptr<Hamiltonian<_Ht>>& _H, 
													double _lr, 
													uint _threadNum, 
													int _nParticles,
													const NQSLS_p& _lower, 
													std::vector<double> _beta)
	: H_(_H)
{
	this->info_p_.lr_			= _lr;
	// set the number of particles
	// set the visible layer (for hardcore-bosons we have the same number as sites but fermions introduce twice the complication)
	this->info_p_.nVis_ 		= 			_H->getNs() * (this->spinModes_ / 2);
	this->info_p_.nSites_		=			_H->getNs();

	// make it half filling if necessary
	this->info_p_.nParticles_	=			(_nParticles < 0 || this->spinModes_ == 2) ? this->info_p_.nSites_ : (uint)_nParticles;
	// check the Hilbert space
	this->info_p_.Nh_			=			_H->getHilbertSize();
	// set the random number generator
	this->ran_					=			_H->ran_;

	// setup the lower lying states, whenever they were previously set
	{
		if (_lower.size() == _beta.size() && _lower.size() != 0)
		{
			this->lower_states_.isSet_		=	true;
			this->lower_states_.f_lower 	= _lower;
			this->lower_states_.f_lower_b 	= _beta;
		}
	}

	// set threads
	this->initThreads(_threadNum);

	LOGINFO("Constructed the general NQS class", LOG_TYPES::TRACE, 2);
};

// ##########################################################################################################################################

#include "./NQS/NQS_base/nqs_general.h"
#include "./NQS/NQS_base/nqs_spins.h"
#include "./NQS/NQS_base/nqs_fermions.h"

// ##########################################################################################################################################

#endif