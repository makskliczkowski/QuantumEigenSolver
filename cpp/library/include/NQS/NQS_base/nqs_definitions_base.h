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
# 	if defined NQS_USE_GPU
#		undef NQS_USE_GPU
#	endif
#elif defined NQS_USE_GPU					
#	include <cuda_runtime.h>						
#endif		

// ----------------------------------------------------------					
#define NQS_ANGLES_UPD						// shall one update the angles or calculate them from scratch			
#define NQS_USE_VEC_ONLY					// use vector only?							
#if defined NQS_USE_VEC_ONLY
	#define NQS_STATE this->curVec_
	#define NQS_STATE_T Config_cr_t
	#define NQS_ROW_T arma::Row<_T>
	#define NQS_COL_T arma::Col<_T>
#else 
	#define NQS_STATE this->curState_
	#define NQS_STATE_T u64 
#endif
// ----------------------------------------------------------
#define NQS_USESR							// optimize the gradient descent with Stochastic Reconfiguration (SR)
#define NQS_LOWER_RATIO_LOGDIFF 			// use the logarithm of the ratio of the probabilities
// --------------- STOCHASTIC RECONFIGURATION ---------------
#ifdef NQS_USESR							// if we use the SR
#	define NQS_USESR_NOMAT					// skip the matrix construction for the SR - use iterative solvers
#	if defined NQS_USESR_NOMAT				// if we do not use the matrix
#		define NQS_USESR_NOMAT_USED			
# 	else
#		define NQS_USESR_MAT_USED
#	endif
#	define NQS_SREG_ATTEMPTS 5				// how to handle the inverse of the matrix (if needed)
#   define NQS_SREG_GRAD_NORM_THRESHOLD 1e3	// threshold for the gradient norm
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
#define NQS_PUBLIC_TYPES(_type, _stateType) public:	using NQSS = arma::Col<_stateType>;															\
											using NQSB = arma::Col<_type>; 																		\
											using NQSW = arma::Mat<_type>;			 															\
											using Solver_t_p = std::shared_ptr<algebra::Solvers::General::Solver<_T, true>>;					\
											using Precond_t_p = std::shared_ptr<algebra::Solvers::Preconditioners::Preconditioner<_T, true>>;	\
											using int_ini_t = std::initializer_list<int>;														\
											using dbl_ini_t = std::initializer_list<double>;									
											


#define NQS_LOG_ERROR_SPIN_MODES LOG_ERROR("IMPLEMENT ME FOR THIS NUMBER OF SPIN MODES")

//////////////////////////////////////////////////////////////////////////////////////////

// Kernel for multithreading
#ifdef NQS_NOT_OMP_MT
	#include <functional>
	#include <condition_variable>
	#include <future>
/**
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

struct NQS_info_t
{
    using u64                       			=       uint64_t;

    // simulation specific
	std::unique_ptr<MachineLearning::Parameters> p_;							// scheduler for the learning rate

	// regarding the iterative solvers
	int solver_									=		1;						// solver for the NQS with SR
	int maxIter_								= 		5000;					// maximum number of iterations
	double tol_									=		1e-5;					// tolerance for iterative solvers
	void setSolver(int _s, int i, double _tol)									{ this->solver_ = _s; this->maxIter_ = i; this->tol_ = _tol; };

	// pseudoinverse
	double pinv_ 								= 		-1;						// pseudoinverse for the NQS

    // architecture specific			
    uint nVis_									=		1;						// number of visible neurons (input variables)
    uint nSites_								=		1;						// number of lattice sites or fermionic modes
    uint fullSize_								=		1;						// full number of the parameters (for memory purpose)

    // Hilbert space info			
    u64 Nh_										=		1;						// Hilbert space size (number of basis states)
    uint nParticles_							=		1;						// number of particles in the system (if applicable)
    bool conservesParticles_					=		true;					// whether the system conserves the number of particles

	// training related
	double lr_									=		1e-3;					// learning rate
	double lr(size_t epoch, double _metric) const								{ return this->p_ ? (this->p_)->operator()(epoch, _metric) : this->lr_; 				};

	// early stopping
	void setEarlyStopping(size_t _pat, double _minDlt = 1e-3)					{ if (this->p_) this->p_->set_early_stopping(_pat, _minDlt); 							};	
	template <typename _T> bool stop(size_t epoch, _T _metric);
	double best() const															{ return this->p_ ? this->p_->best() : 0.0; };

	// regularization related
	std::unique_ptr<MachineLearning::Parameters> s_;							// regularization scheduler
	double sreg_					=		1e-7;								// regularization for the covariance matrix
	double sreg(size_t epoch, double _metric) const								{ return this->s_ ? (*this->s_)(epoch, _metric) : this->sreg_; };

	// ---------------------------------------------------------------

	NQS_info_t() 					= 		default;
	~NQS_info_t();

	// ---------------------------------------------------------------

	void saveInfo(const std::string& _dir, const std::string& _name, int i = 0) const;

	// ---------------------------------------------------------------

	NQS_info_t(const NQS_info_t& other);
    NQS_info_t(NQS_info_t&& other) noexcept;

	// ---------------------------------------------------------------

    NQS_info_t& operator=(const NQS_info_t& other);
    NQS_info_t& operator=(NQS_info_t&& other) noexcept;

	// ---------------------------------------------------------------
};

//////////////////////////////////////////////////////////////////////////////////////////

template <typename _T = double>
struct NQS_thread_t
{
	int threadNum_					=	1;						// number of threads to be used for the NQS
	#ifdef NQS_NOT_OMP_MT
		uint threadsNumLeft_		=	0;						// other threads that are left to be processed
		std::vector<std::thread> threads_;
		std::vector<CondVarKernel<_T>> kernels_;				// condition variables for the threads
		std::mutex mutex;										// global mutex for the threads	
	#endif

	// ----------
	NQS_thread_t()					=	default;
	~NQS_thread_t()					{ this->threads_.clear(); };
};

// ##########################################################################################################################################

// forward declaration
namespace Operators
{
	template <typename _T, typename ..._Ts>
	class OperatorComb;
};

/**
* @brief Struct to modify the ansatz.
* 
* This struct is used to apply modifications to the ansatz in a quantum eigen solver.
* 
* @tparam _T The data type used for the modifications, default is double.
*/
template <typename _T = double>
struct AnsatzModifier 
{
	std::shared_ptr<Operators::OperatorComb<_T>> modifier_;
	bool modified_ 			= false;
	std::string modtype_ 	= "none";

	// used only when the ansatz is modified
	bool modifies_state_	= false;
	_T logAMod_ 			= 0.0;
	_T logTmpAMod_			= 0.0;
};

// ##########################################################################################################################################

#define NQS_INST_CMB(_Ht, _T, FUN, FUNRET, ARGS) 									\
					template FUNRET  NQS<2u, _Ht, _T, double>::FUN ARGS; 			\
					template FUNRET  NQS<3u, _Ht, _T, double>::FUN ARGS; 			\
					template FUNRET  NQS<4u, _Ht, _T, double>::FUN ARGS; 
#define NQS_INST_CMB_ALL(FUN, FUNRET, ARGS) 										\
					NQS_INST_CMB(double, double, FUN, FUNRET, ARGS)					\
					NQS_INST_CMB(double, std::complex<double>, FUN, FUNRET, ARGS)	\
					NQS_INST_CMB(std::complex<double>, double, FUN, FUNRET, ARGS)	\
					NQS_INST_CMB(std::complex<double>, std::complex<double>, FUN, FUNRET, ARGS)

// ##########################################################################################################################################