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
#else 
	#define NQS_STATE this->curState_
	#define NQS_STATE_T u64 
#endif

// optimize the gradient descent with Stochastic Reconfiguration (SR)
#define NQS_USESR							
#ifdef NQS_USESR							

// skip the matrix construction for the SR
	// #define NQS_USESR_NOMAT

// how to handle the inverse of the matrix (if needed)
#	define NQS_PINV					
// regularization for the covariance matrix	
//#	define NQS_SREG													
// shall one use the iterative solver without constructing the full matrix explicitly?						  
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
    double lr_						=		1e-3;					// specific learning rate for the NQS - either for gradient descent or stochastic reconfiguration
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