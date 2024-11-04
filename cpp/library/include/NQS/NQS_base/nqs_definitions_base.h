/*
* NQS Definitions - all the definitions for the NQS class
* @file nqs_definitions.h
*/

#pragma once
// ######################### NQS ############################

// saving the weights and the directory
#define NQS_SAVE_WEIGHTS					
#ifdef NQS_SAVE_WEIGHTS						
#	define NQS_SAVE_DIR ( "WEIGHTS" + kPS )		
#endif										

// ----------------------------------------------------------		

//#define NQS_USE_GPU						
#define NQS_USE_CPU							
											
#ifdef NQS_USE_CPU							
#	define NQS_USE_MULTITHREADING			

//#	define NQS_USE_OMP						
# 	if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP
#		define NQS_NOT_OMP_MT 
#	endif

#elif defined NSQ_USE_GPU					
// something !TODO								
#endif		

// ----------------------------------------------------------					
											
// shall one update the angles or calculate them from scratch						
#define NQS_ANGLES_UPD						
											
// use vector only?							
#define NQS_USE_VEC_ONLY

#if defined NQS_USE_VEC_ONLY
	#define NQS_STATE this->curVec_
	#define NQS_STATE_T const NQSS&
	#define NQS_ROW_T arma::Row<_T>
#else 
	#define NQS_STATE this->curState_
	#define NQS_STATE_T u64 
#endif

// optimize the gradient descent with Stochastic Reconfiguration (SR)
#define NQS_USESR
#define NQS_LOWER_RATIO_LOGDIFF 
// #define NQS_CHECK_NAN

// --------------- STOCHASTIC RECONFIGURATION ---------------
#ifdef NQS_USESR						

// skip the matrix construction for the SR
#	define NQS_USESR_NOMAT

// check whether we have SR without matrix
#	if defined NQS_USESR_NOMAT
#		define NQS_USESR_NOMAT_USED
// # 		define NQS_USESR_NOMAT_USEPRECONDITIONING
# 	else
#		define NQS_USESR_MAT_USED
#	endif


// how to handle the inverse of the matrix (if needed)
#	define NQS_SREG_ATTEMPTS 5
#   define NQS_SREG_GRAD_NORM_THRESHOLD 1e3
#endif										
// ##########################################################

// ######### NQS TYPES #############
enum NQSTYPES					// #
{								// #
	RBM_T,						// #
	RBMPP_T						// #
};								// #
// #################################

// ##########################################################################################################################################

// all the types that are to be used in each NQS implementation
#define NQS_PUBLIC_TYPES(_type, _stateType) public:	using NQSS = arma::Col<_stateType>;	\
											using NQSB = arma::Col<_type>; 				\
											using NQSW = arma::Mat<_type>;			 	
											


#define NQS_LOG_ERROR_SPIN_MODES LOG_ERROR("IMPLEMENT ME FOR THIS NUMBER OF SPIN MODES")

//////////////////////////////////////////////////////////////////////////////////////////

// Kernel for multithreading
#ifdef NQS_NOT_OMP_MT
	#include <functional>
	#include <condition_variable>
	#include <future>
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
#endif 
//////////////////////////////////////////////////////////////////////////////////////////

// ##########################################################################################################################################

#include <vector>
#include <string>

// #######################################################################################

// #################################
#ifndef ML_H					// #
#	include "../../../source/src/Include/ml.h"
#endif // !ML_H					// #
// #################################

// #######################################################################################

struct NQS_train_t
{
	NQS_train_t() 	= default;
	NQS_train_t(uint _mcs, uint _mcth, uint _nblck, uint _bsize, uint _nFlip, const std::string& _dir = "") 
		: MC_sam_(_mcs), MC_th_(_mcth), nblck_(_nblck), bsize_(_bsize), dir(_dir), nFlip(_nFlip) {};

	uint MC_sam_	=	10;					// number of Monte Carlo Steps (outer loops for the training or collecting)
	uint MC_th_		=	0;					// number of mcSteps to thermalize (burn-in)
	uint nblck_		= 	32;					// number of such blocks for one average step (single iteration step after which the gradient is calculated)
	uint bsize_		= 	4;					// for killing correlations - (single block size)
	std::string dir	=	"";					// saving directory (for the weights) - try to flip this many times (many flips)
	uint nFlip		= 	1;					// number of flips to set (default is 1)

	void hi(const std::string& _in = "Train: ") const;
};

//////////////////////////////////////////////////////////////////////////////////////////


struct NQS_info_t
{
    using u64                       =       uint64_t;

    // simulation specific
	MachineLearning::Parameters* p_	=		nullptr;

	double pinv_ 					= 		-1;						// pseudoinverse for the NQS
    // architecture specific
    uint nVis_						=		1;						// number of visible neurons (input variables)
    uint nSites_					=		1;						// number of lattice sites or fermionic modes
    uint fullSize_					=		1;						// full number of the parameters (for memory purpose)

    // Hilbert space info
    u64 Nh_							=		1;						// Hilbert space size (number of basis states)
    uint nParticles_				=		1;						// number of particles in the system (if applicable)
    bool conservesParticles_		=		true;					// whether the system conserves the number of particles

    // normalization
    double norm_					=		0.0;					// normalization factor for the state vector

	double lr_						=		1e-3;					// learning rate
	double lr(size_t epoch, double _metric) const					{ return this->p_ ? (*this->p_)(epoch, _metric) : this->lr_; };

	MachineLearning::Parameters* s_ =		nullptr;				// regularization scheduler
	double sreg_					=		1e-7;					// regularization for the covariance matrix
	double sreg(size_t epoch, double _metric) const					{ return this->s_ ? (*this->s_)(epoch, _metric) : this->sreg_; };

	// ---------------------------------------------------------------
	void setEarlyStopping(size_t _pat, double _minDlt = 1e-3)		{ if (this->p_) this->p_->set_early_stopping(_pat, _minDlt); };	
	bool stop(size_t epoch, double _metric = 0.0)					{ if (this->p_) return this->p_->stop(epoch, _metric); else return false; };
	bool stop(size_t epoch, std::complex<double> _metric)			{ if (this->p_) return this->p_->stop(epoch, std::real(_metric)); else return false; };
	double best() const												{ return this->p_ ? this->p_->best() : 0.0; };
	
	// ---------------------------------------------------------------

	NQS_info_t() 					= 		default;
	~NQS_info_t();

	// ---------------------------------------------------------------

	void saveInfo(const std::string& _dir, const std::string& _name, int i = 0) const;
};

//////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
struct NQS_thread_t
{
	int threadNum_					=	1;						// number of threads to be used for the NQS
	#ifdef NQS_NOT_OMP_MT
		uint threadsNumLeft_		=	0;						// other threads that are left to be processed
		std::vector<std::thread> threads_;
		std::vector<CondVarKernel<_T>> kernels_;				// condition variables for the threads
		std::mutex mutex;										// global mutex for the threads	
	#endif
};