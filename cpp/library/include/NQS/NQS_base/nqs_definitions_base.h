/**
* @brief NQS Definitions - all the definitions for the NQS class
* @file nqs_definitions.h
*/
// ##########################################################################################################################################

#ifndef NQS_DEFINITIONS_H
#define NQS_DEFINITIONS_H

#include "../../algebra/general_operator.h"
#include <cstddef>
#include <memory>
// ----------------------------------------------------------

// Directories

// ----------------------------------------------------------
#define NQS_SAVE_WEIGHTS					
#ifdef NQS_SAVE_WEIGHTS						
#	define NQS_SAVE_DIR ( "WEIGHTS" + kPS )		
#endif										

// ----------------------------------------------------------

// MULTITHREADING

// ----------------------------------------------------------
//#define NQS_USE_OMP
//#define NQS_USE_GPU						
#define NQS_USE_CPU							
											
// !CPU
#ifdef NQS_USE_CPU							
#	define NQS_USE_MULTITHREADING			
# 	if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP
#		define NQS_NOT_OMP_MT 
#	endif
# 	if defined NQS_USE_GPU
#		undef NQS_USE_GPU
#	endif
#elif defined NQS_USE_GPU		
#	include <cuda_runtime.h>						
#endif		
//!GPU: TODO			

// ----------------------------------------------------------					
#define NQS_ANGLES_UPD						// shall one update the angles or calculate them from scratch			
#define NQS_USE_ARMA						// use Armadillo for the matrix operations (if not, use the custom implementation)
// ----------------------------------------------------------

// STATES REPRESENTATION

// ----------------------------------------------------------
#define NQS_USE_VEC_ONLY					// use vector only? - if not, use the integer representation		
#ifdef NQS_USE_VEC_ONLY
	#define NQS_STATE_R_T arma::Col
	#define NQS_STATE this->curVec_			// current state
	#define NQS_STATE_T Config_cr_t			// current state type 
#elif defined NQS_USE_STDVEC
	#define NQS_STATE_R_T std::vector
	#define NQS_STATE this->curState_
	#define NQS_STATE_T u64 
#else

#endif
// ----------------------------------------------------------

// REPRESENTATION

// ----------------------------------------------------------
#ifdef NQS_USE_ARMA
#	include <armadillo>
#	define NQS_ROW_T arma::Row<_type>		// row type for weights etc representation
#	define NQS_COL_T arma::Col<_type>		// column type for weights etc representation
#else		
#	define NQS_ROW_T std::vector<_type>		// row type for weights etc representation
#	define NQS_COL_T std::vector<_type>		// column type for weights etc representation
#endif
// ----------------------------------------------------------

// OPTIMIZATIONS

// ----------------------------------------------------------
#define NQS_USESR							// optimize the gradient descent with Stochastic Reconfiguration (SR)
#define NQS_LOWER_RATIO_LOGDIFF 			// use the logarithm of the ratio of the probabilities (instead of the ratio itself) for the lower states
// --------------- STOCHASTIC RECONFIGURATION ---------------
#ifdef NQS_USESR							// if we use the SR
#	define NQS_USESR_NOMAT					// skip the matrix construction for the SR - use iterative solvers
#	ifdef NQS_USESR_NOMAT					// if we do not use the matrix
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
	RBMPP_T,					// #
	CONV_T						// #
};								// #
// #################################

// ##########################################################################################################################################
// all the types that are to be used in each NQS implementation
#ifdef NQS_USE_ARMA
#	define NQS_PUBLIC_TYPES(_type, _stateType) public:																\
				using NQSS = arma::Col<_stateType>;																	\
				using NQSB = arma::Col<_type>; 																		\
				using NQSW = arma::Mat<_type>;			 															\
				using Solver_t_p = std::shared_ptr<algebra::Solvers::General::Solver<_type, true>>;					\
				using Precond_t_p = std::shared_ptr<algebra::Solvers::Preconditioners::Preconditioner<_type, true>>;\
				using int_ini_t = std::initializer_list<int>;														\
				using dbl_ini_t = std::initializer_list<double>;		
#elif defined NQS_USE_STDVEC
#	define NQS_PUBLIC_TYPES(_type, _stateType) public:																							\	
				using NQSS = std::vector<_stateType>;																\
				using NQSB = std::vector<_type>; 																	\
				using NQSW = std::vector<std::vector<_type>>;														\
				using Solver_t_p = std::shared_ptr<algebra::Solvers::General::Solver<_type, false>>;					\
				using Precond_t_p = std::shared_ptr<algebra::Solvers::Preconditioners::Preconditioner<_type, false>>;	\
				using int_ini_t = std::initializer_list<int>;														\
				using dbl_ini_t = std::initializer_list<double>;
# else
# 	define NQS_PUBLIC_TYPES 
#endif
// ##########################################################################################################################################
#define NQS_HAMIL_TYPES(_Ht, _spinModes) public:									\
	using Hamil_t 						= Hamiltonian<_Ht, _spinModes>;				\
	using Hamil_t_p 					= std::shared_ptr<Hamil_t>;					\
	using Hilbert_t 					= Hilbert::HilbertSpace<_Ht, _spinModes>;	\
	using Hilbert_cr_t 					= const Hilbert_t&;											
// ##########################################################################################################################################
#define NQS_LOG_ERROR_SPIN_MODES LOG_ERROR("IMPLEMENT ME FOR THIS NUMBER OF SPIN MODES")
// ##########################################################################################################################################

//////////////////////////////////////////////////////////////////////////////////////////
// Kernel for multithreading without OpenMP
#ifdef NQS_NOT_OMP_MT
	#include <functional>
	#include <condition_variable>
	#include <future>

/**
* @brief A structure that encapsulates a condition variable kernel.
* 
* This structure is designed to manage synchronization between threads using
* a condition variable and a mutex. It also includes atomic and non-atomic 
* flags to control thread execution and a kernel value of a templated type.
* 
* @tparam _T The type of the kernel value.
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

#include <vector>
#include <string>

// #######################################################################################

// #################################
#ifndef ML_H					// #
#	include "../../../source/src/Include/ml.h"
#endif // !ML_H					// #
// #################################

// #################################################################################################################################

/**
* @brief Generates a time space vector using either logarithmic or regular spacing.
*
* This function creates a vector of time points between 0 and _tmax, with spacing
* determined by _dt. If _log is greater than 0, the spacing will be logarithmic,
* otherwise it will be regular.
*
* @param _log Determines the type of spacing. If greater than 0, logarithmic spacing is used.
* @param _dt The time step size for regular spacing or the base for logarithmic spacing.
* @param _tmax The maximum time value.
* @return arma::vec A vector of time points.
*/
inline arma::vec time_space_nqs(int _log = 0, double _dt = 0.01, double _tmax = 1.0)
{
	if (_log > 0)
		return arma::logspace(_dt, _tmax, _log);
	return arma::regspace(_dt, _dt, _tmax);
}

// #######################################################################################
enum class NQS_init_st_t
{
	NO_INIT,			// no initialization
	RANDOM,				// random initialization
	FERRO,				// ferromagnetic initialization
	ANTI_FERRO			// anti ferromagnetic initialization
};
//////////////////////////////////////////////////////////////////////////////////////////
struct NQS_reg_t
{
	double reg_		= 1e-7;														// regularization for the covariance matrix
	std::unique_ptr<MachineLearning::Parameters> p_;							// scheduler for the regularization
	// getters
	const double reg(size_t epoch, double _metric) const;	
	// save
	void save(const std::string& _dir, const std::string& _name, int i, const std::string& _namepar, bool _append) const;							
	// default constructor
	NQS_reg_t()		= default;
	// copy constructor
	NQS_reg_t(const NQS_reg_t& other);
	// move constructor
	NQS_reg_t(NQS_reg_t&& other) noexcept;
	// copy assignment
	NQS_reg_t& operator=(const NQS_reg_t& other);
	// move assignment
	NQS_reg_t& operator=(NQS_reg_t&& other) noexcept;
};
//////////////////////////////////////////////////////////////////////////////////////////
struct NQS_scheduler_t
{
	double best_								=		0.0;					// best value of the metric
	double lr_									=		1e-3;					// learning rate
	std::unique_ptr<MachineLearning::Parameters> p_;							// scheduler for the learning rate
	// setters
	void set_early_stopping(size_t _pat, double _minDlt = 1e-3)					{ if (!this->p_) this->p_->set_early_stopping(_pat, _minDlt);	};
	// getters
	void lr_update(size_t epoch, double _metric)								{ this->lr_ = this->lr(epoch, _metric);							};
	const double lr(size_t epoch, double _metric) const							{ return this->p_ ? (*this->p_)(epoch, _metric) : this->lr_;	};
	const double best() const													{ return this->p_ ? this->p_->best() : this->best_;				};
	const double best(double _currLoss)											{ if (this->p_) return this->p_->best(); this->best_ = _currLoss; return best_; };
	template <typename _T> bool stop(size_t epoch, _T _metric);
	// save
	void save(const std::string& _dir, const std::string& _name, int i, const std::string& _namepar, bool _append) const;
	// default constructor
	NQS_scheduler_t()							=		default;
	// copy constructor
	NQS_scheduler_t(const NQS_scheduler_t& other);
	// move constructor
	NQS_scheduler_t(NQS_scheduler_t&& other) noexcept;
	// copy assignment
	NQS_scheduler_t& operator=(const NQS_scheduler_t& other);
	// move assignment
	NQS_scheduler_t& operator=(NQS_scheduler_t&& other) noexcept;
};
//////////////////////////////////////////////////////////////////////////////////////////
struct NQS_solver_t
{
	int solver_		= 1;														// solver for the NQS with SR
	int maxIter_	= 5000;														// maximum number of iterations
	double tol_		= 1e-5;														// tolerance for iterative solvers
	void set(int _s, int i, double _tol)										{ this->solver_ = _s; this->maxIter_ = i; this->tol_ = _tol; };
	NQS_solver_t()	= default;
};
//////////////////////////////////////////////////////////////////////////////////////////
struct NQS_info_t
{
    using u64                       			=       uint64_t;

	// solver
	NQS_solver_t solver_;														// solver for the NQS with SR
	
	// regularization for the matrix
	NQS_reg_t reg_;																// regularization for the covariance matrix
	
	// scheduler
	NQS_scheduler_t sched_;														// scheduler for the learning rate

    // architecture specific			
    uint nVis_									=		1;						// number of visible neurons (input variables)
    uint nSites_								=		1;						// number of lattice sites or fermionic modes
	size_t nSitesSquared_						=		1;						// number of lattice sites squared
    uint fullSize_								=		1;						// full number of the parameters (for memory purpose)

    // Hilbert space info			
    u64 Nh_										=		1;						// Hilbert space size (number of basis states)
    uint nParticles_							=		1;						// number of particles in the system (if applicable)
	size_t nParticlesSquared_					=		1;						// number of particles squared
    bool conservesParticles_					=		true;					// whether the system conserves the number of particles

	// early stopping

	// ---------------------------------------------------------------
	NQS_info_t()								=		default;
	~NQS_info_t()								=		default;
	// ---------------------------------------------------------------

	void saveInfo(const std::string& _dir, const std::string& _name, int i = 0) const;

	// ---------------------------------------------------------------

	NQS_info_t(	int _nVis,
				int _nSites,
				int _nParticles,
				size_t _Nh)
		: nVis_(_nVis), nSites_(_nSites), 
		nSitesSquared_(static_cast<size_t>(_nSites) * static_cast<size_t>(_nSites)),
		Nh_(_Nh), nParticles_(_nParticles),  
		nParticlesSquared_(static_cast<size_t>(_nParticles) * static_cast<size_t>(_nParticles))
	{

	}

	// ---------------------------------------------------------------

	NQS_info_t(const NQS_info_t& other);
    NQS_info_t(NQS_info_t&& other) noexcept;

	// ---------------------------------------------------------------

    NQS_info_t& operator=(const NQS_info_t& other);
    NQS_info_t& operator=(NQS_info_t&& other) noexcept;

	// ---------------------------------------------------------------
	const int solver() const 					{ return this->solver_.solver_; 		};
	const double tol() const 					{ return this->solver_.tol_; 			};
	const size_t maxIter() const 				{ return this->solver_.maxIter_;		};
	// ---------------------------------------------------------------
	const double reg() const 					{ return this->reg_.reg_; 				};
	// ---------------------------------------------------------------
	const double lr() const 					{ return this->sched_.lr_; 				};
	const double best(double _currLoss) 		{ return this->sched_.best(_currLoss); 	};
	const double best() const 					{ return this->sched_.best(); 			};
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
//////////////////////////////////////////////////////////////////////////////////////////
#define NQS_DERIV_INST_TYPES(ret, fun, args) 									\
					template ret NQS_deriv<double, double>::fun args; 			\
					template ret NQS_deriv<cpx, double>::fun args; 				\
					template ret NQS_deriv<double, cpx>::fun args; 				\
					template ret NQS_deriv<cpx, cpx>::fun args;
//////////////////////////////////////////////////////////////////////////////////////////
template <typename _stateType = double, typename _type = double>
struct NQS_deriv
{
	//************************************************************************************
	NQS_PUBLIC_TYPES(_type, _stateType);
	//************************************************************************************

    // Member variables (fullSize is the number of parameters in the NQS, nBlocks is the number of consecutive observations)
    NQS_ROW_T derivativesMean;         // Mean of the derivatives \bar O_k 		(fullSize)
    NQS_COL_T energiesCentered;        // Centered energies (E_k - <E_k>) 		(nBlocks)
    NQSW derivatives;                  // Variational derivatives O_k 			(nBlocks x fullSize) 
    NQSW derivativesCentered;          // Centered derivatives (O_k - <O_k>) 	(nBlocks x fullSize)
    NQSW derivativesCenteredH;         // Centered derivatives transposed 		(fullSize x nBlocks) 

	/////////////////////////////////////////////////////////////////////////////////////
	~NQS_deriv() 						= default;
    NQS_deriv(size_t fullSize = 1, size_t nBlocks = 1);
	NQS_deriv(const NQS_deriv& other);
	NQS_deriv(NQS_deriv&& other) noexcept;
	/////////////////////////////////////////////////////////////////////////////////////
	// operators
	NQS_deriv& operator=(const NQS_deriv& other);
	NQS_deriv& operator=(NQS_deriv&& other) noexcept;
	/////////////////////////////////////////////////////////////////////////////////////
    void reset(size_t fullSize, size_t nBlocks = 1);
	/////////////////////////////////////////////////////////////////////////////////////
	template <typename _CT> void set_centered(const _CT& _energies, _type _meanLoss, const _type _samples);
	/////////////////////////////////////////////////////////////////////////////////////
	NQSB getF() const;
	/////////////////////////////////////////////////////////////////////////////////////
	template <typename _CT> void finalF(const _CT& _energies, int _step, _type _meanLoss, const _type _samples);
	/////////////////////////////////////////////////////////////////////////////////////
	// SPECIFIC
	/////////////////////////////////////////////////////////////////////////////////////
#ifdef NQS_USE_ARMA
	arma::subview_row<_type> row(size_t i) 				{ return this->derivatives.row(i); 			}
	arma::subview_col<_type> col(size_t i) 				{ return this->derivatives.col(i); 			}
	arma::subview_col<_type> colCentered(size_t i) 		{ return this->derivativesCentered.col(i); 	}
	arma::subview_row<_type> rowCentered(size_t i) 		{ return this->derivativesCentered.row(i); 	}
	arma::subview_col<_type> colCenteredH(size_t i) 	{ return this->derivativesCenteredH.col(i); }
	arma::subview_row<_type> rowCenteredH(size_t i) 	{ return this->derivativesCenteredH.row(i); }
#elif NQS_USE_STDVEC
	NQS_COL_T row(size_t i) 							{ return this->derivatives[i]; 				}
	NQS_COL_T rowCentered(size_t i) 					{ return this->derivativesCentered[i]; 		}
	NQS_COL_T rowCenteredH(size_t i) 					{ return this->derivativesCenteredH[i]; 	}
	// columns can be taken by transposing the rows only
#else 
#endif
	NQS_COL_T mean_t() 									{ return this->derivativesMean.t(); 		}
	NQS_ROW_T mean() 									{ return this->derivativesMean; 			}
	/////////////////////////////////////////////////////////////////////////////////////
	void printState() const;
	/////////////////////////////////////////////////////////////////////////////////////
};
// ##########################################################################################################################################
namespace Operators
{
	template <typename _T, typename ..._Ts>
	class OperatorComb;
};
// ##########################################################################################################################################
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

template <typename _T, uint _spinModes>
class Hamiltonian;

template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double>
struct NQS_Const_par_t 
{
	// **********************************************************************************************************************
	MCS_PUBLIC_TYPES(_T, _stateType, NQS_STATE_R_T);
	NQS_PUBLIC_TYPES(_T, _stateType);
	NQS_HAMIL_TYPES(_Ht, _spinModes);
	// **********************************************************************************************************************
	Hamil_t_p H_;											// Hamiltonian
	std::vector<double> nHid_;								// number of hidden units 
	double lr_			= 1e-1;								// learning rate
	uint threadNum_		= 1;								// number of threads
	int nPart_			= -1;								// number of particles
	// **********************************************************************************************************************
	NQS_Const_par_t()										= default;
	~NQS_Const_par_t()										= default;
	// **********************************************************************************************************************
	NQS_Const_par_t(const NQS_Const_par_t& other)
		: H_(other.H_), nHid_(other.nHid_), lr_(other.lr_), threadNum_(other.threadNum_), nPart_(other.nPart_) { }
	// **********************************************************************************************************************
	NQS_Const_par_t(NQS_Const_par_t&& other) noexcept
		: H_(std::move(other.H_)), nHid_(std::move(other.nHid_)), lr_(other.lr_), threadNum_(other.threadNum_), nPart_(other.nPart_) { }
	// **********************************************************************************************************************
	NQS_Const_par_t& operator=(const NQS_Const_par_t& other)
	{
		if (this != &other)
		{
			this->H_ 			= other.H_;
			this->nHid_	 		= other.nHid_;
			this->lr_ 			= other.lr_;
			this->threadNum_ 	= other.threadNum_;
			this->nPart_ 		= other.nPart_;
		}
		return *this;
	}
	// **********************************************************************************************************************
	NQS_Const_par_t& operator=(NQS_Const_par_t&& other) noexcept
	{
		if (this != &other)
		{
			this->H_ = std::move(other.H_);
			this->nHid_ = std::move(other.nHid_);
			this->lr_ = other.lr_;
			this->threadNum_ = other.threadNum_;
			this->nPart_ = other.nPart_;
		}
		return *this;
	}
	// **********************************************************************************************************************
};
// ##########################################################################################################################################


// ##########################################################################################################################################

#endif // !NQS_DEFINITIONS_H