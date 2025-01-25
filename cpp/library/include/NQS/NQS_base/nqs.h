/***************************************
* @file nqs.h
* @brief Defines the generic NQS Solver class.
* 
* This file contains the definition of the Neural Quantum States (NQS) solver class, 
* which provides a flexible framework for building and optimizing NQS models. 
* The class supports multiple spin modes and Hamiltonians, 
* and is designed for efficient sampling and optimization using Monte Carlo-based methods.
* 
* @date January 2025
* @version 1.12
* @note This code is under constant development.
* @author Maksymilian Kliczkowski
* @institution WUST, Poland
***************************************/
#ifndef NQS_H
#define NQS_H

// include all the definions
#include "../nqs_operator.h"
#include "nqs_definitions_lower.tpp"
// ######################################################################################################################################
namespace NQS_NS
{
	/**
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
	* @!TODO Add more flexible configuration type for the state 
	*/
	template <uint _spinModes, 
			typename _Ht,
			typename _T			= _Ht, 
			class _stateType	= double>
	class NQS : public MonteCarlo::MonteCarloSolver<_T, _stateType, NQS_STATE_R_T<_stateType>>
	{	
	public:
		// **********************************************************************************************************************************
		NQS_PUBLIC_TYPES(_T, _stateType);									// type definitions 
		MCS_PUBLIC_TYPES(_T, _stateType, NQS_STATE_R_T); 					// type definitions for the Monte Carlo solver
		NQS_HAMIL_TYPES(_Ht, _spinModes);									// type definitions for the Hamiltonian
		// **********************************************************************************************************************************
		using NQSLS_p 						= typename NQS_lower_t<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
		// **********************************************************************************************************************************
		using NQS_evo_f_t 					= std::function<NQSB(double, double, const NQSB&)>;			// function for the evolution - returns the new state
		using NQS_evo_fr_t 					= std::function<void(double, double, const NQSB&, NQSB*)>;	// instead of the return value, the function modifies the input
		// **********************************************************************************************************************************
		NQS_info_t info_p_;													// information about the NQS
		NQS_lower_t<_spinModes, _Ht, _T, _stateType> lower_states_;			// information about the training
		// ******************************************clear****************************************************************************************
	protected:
		const double discVal_				=		::Operators::_SPIN;   	// discrete value for the numbers in vector representation
		const uint spinModes_				=		_spinModes;				// number of spin modes -> e.g. 2 for hardcore bosons, 4 for fermions
		bool updateWeights_ 				= 		true;					// shall update the weights in current step? (default is true)
		Container_t E_;														// container for the energies (during the training or the sampling)
		// **********************************************************************************************************************************
		Hamil_t_p H_;														// pointer to the Hamiltonian instance (for the energy calculation)
		// **********************************************************************************************************************************
	protected:																// ---------------------- T H R E A D I N G ---------------------
#ifdef NQS_NOT_OMP_MT
		bool initThreads(uint _threadNum = 0);
		NQS_thread_t<_T> threads_;											// thread information
#endif

	protected:																// ----------------------- T R A I N I N G ----------------------
		// **********************************************************************************************************************************
		uint nFlip_							=		1;						// number of flips to be done in one step (each flip is a change in the state)
		v_1d<uint> flipPlaces_;												// stores flip spots to be flipped during one sampling step
		v_1d<_stateType> flipVals_;											// stores values before (!!!) the flip to be used for the gradients
		// **********************************************************************************************************************************
		Config_t curVec_;													// currently processed state vector for convenience
		u64 curState_						=		0;						// currently processed state - may or may not be used
		// **********************************************************************************************************************************
		v_1d<Config_t> tmpVecs_;											// temporary vectors for the flips
		Config_t tmpVec_;													// temporary vector for the flips (for the current state)
		u64 tmpState_						=		0;						// temporary state for the flips
		// **********************************************************************************************************************************
		// ------------------------ W E I G H T S -----------------------
		NQS_deriv<_stateType, _T> derivatives_;								// derivatives of the NQS (contains the gradients and the parameters)
#ifdef NQS_USESR_MAT_USED
		NQSW S_;															// positive semi-definite covariance matrix - to be optimized (inverse of the Fisher information matrix)
#else 
		Precond_t_p precond_ 				= 		nullptr;				// preconditioner for the conjugate gradient
		Solver_t_p solver_ 					= 		nullptr;				// solver for the Fisher matrix inversion
	public:
		void setSolver(int _s, double _t, int _mi = 1000, double _r = -1.0);// solving method with the tolerance						
		void setPreconditioner(int _pre);									// set the preconditioner						
	#endif
	protected:
		NQSB Weights_;														// weights of the NQS - column of all the weights
		NQSB dF_;															// forces acting on the weights (F_k) - final gradient (dF) - used to update the weights
		NQSB F_;															// forces acting on the weights (F_k) - is the original gradient (F_k)
	protected:																// ----------------------- S T A R T E R S ----------------------
		virtual void allocate();											// allocate the memory for the NQS (e.g., vectors, matrices, etc.)
	public:																	// ------------------------ S E T T E R S -----------------------
		// **********************************************************************************************************************************
		virtual void setState(NQS_init_st_t _st = NQS_init_st_t::RANDOM);	// set the state of the NQS
		virtual void setState(Config_cr_t _st, bool _set)	=		0;		// column vector state
		virtual void setState(Config_cr_t _st);								// column vector state (set the current state)
		virtual void setState(u64 _st, bool _set)			=		0; 		// set the state (integer)
		virtual void setState(u64 _st);
		// **********************************************************************************************************************************
	protected:																// -------------------------- F L I P S --------------------------
		virtual void chooseRandomFlips()					=			0;	// important for the flipPlaces_ and flipVals_ to be set! - choose random flips
		virtual void applyFlipsT()							=			0;	// apply flips to the temporary vector (tmpVec_)
		virtual void applyFlipsC()							=			0;	// apply flips to the current vector (curVec_)	
		virtual void unapplyFlipsT()						{ this->applyFlipsT(); }; // unapply flips of the temporary vector according the template 
		virtual void unapplyFlipsC()						{ this->applyFlipsC(); }; // unapply flips of the current vector according the template
		// **********************************************************************************************************************************
	public:
		virtual void setRandomFlipNum(uint _nFlips) 		override	{};	// set the number of flips to be done

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
		// ***********************************************************************************************************************************
		virtual auto logPRatio(uint fP, float fV)			->_T			= 0; // log of the probability ratio - when flip places (single) are used
		virtual auto logPRatio(uint, uint, float, float)	->_T			= 0; // log of the probability ratio - when flip places (double) are used
		virtual auto logPRatio(uint nFlips)					->_T			= 0; // log of the probability ratio - when number of flips is used
		virtual auto logPRatio(int_ini_t fP, dbl_ini_t fV) -> _T			= 0; // ratio when exact points are provided (used for the Hamiltonian probability ratio - when the Hamiltonian changes the state)
		virtual auto logPRatio(Config_cr_t _v)				->_T			{ return this->logPRatio(this->curVec_, _v);						};
		virtual auto logPRatio(Config_cr_t v, Config_cr_t w)->_T			= 0; // log of the probability ratio - when two vectors are used (one for the current state and one for the new state)
		virtual auto logPRatio()							->_T			{ return this->logPRatio(this->flipPlaces_[0], this->flipVals_[0]);	};
		virtual auto pRatio(uint fP, float fV)				->_T			{ return std::exp(this->logPRatio(fP, fV));							};
		virtual auto pRatio(uint f, uint g, float v, float w) ->_T			{ return std::exp(this->logPRatio(f, g, v, w));						}; 
		virtual auto pRatio(uint nFlips)					->_T			{ return std::exp(this->logPRatio(nFlips));							};	
		virtual auto pRatio(Config_cr_t _v, Config_cr_t _w)	->_T			{ return std::exp(this->logPRatio(_v, _w));							};
		virtual auto pRatio(int_ini_t fP, dbl_ini_t fV)		->_T			{ return std::exp(this->logPRatio(fP, fV));							};
		virtual auto pRatio(Config_cr_t _v)					->_T			{ return std::exp(this->logPRatio(_v));								};	
		virtual auto pRatio()								->_T			{ return this->pRatio(this->flipPlaces_[0], this->flipVals_[0]);	};		
		// ansatz modification with a single operator O: O|psi> = \sum _s Psi(s) \sum _s' o_s' |s'>
		void setModifier(std::shared_ptr<Operators::OperatorComb<_T>> _mod);
		void unsetModifier();
		bool modified()														{ return this->a_mod_p_.modified_; 									};
	protected:
		AnsatzModifier<_T> a_mod_p_;										// ansatz modifier - for the probability ratio

		// used only when the ansatz is modified
		auto logAnsatzModifier(uint fP, float fV)			->_T;
		auto logAnsatzModifier(int_ini_t fP, dbl_ini_t fV)	->_T;
		auto logAnsatzModifier(uint, uint, float, float)	->_T;
		auto logAnsatzModifier(Config_cr_t v)				->_T;
		auto logAnsatzModifier(uint nFlips)					->_T;
		// ***********************************************************************************************************************************
		auto logPRatioMod(uint fP, float fV)				->_T;
		auto logPRatioMod(uint f, uint g, float v, float w) ->_T;
		auto logPRatioMod(int_ini_t fP, dbl_ini_t fV)		->_T;
		auto logPRatioMod(Config_cr_t v)					->_T;
		auto logPRatioMod(Config_cr_t v, Config_cr_t w)		->_T;
		auto logPRatioMod(uint nFlips)						->_T;
		auto pRatioMod(uint fP, float fV)					->_T 			{ return std::exp(this->logPRatioMod(fP, fV)); 						};
		auto pRatioMod(uint f, uint g, float v, float w)	->_T 			{ return std::exp(this->logPRatioMod(f, g, v, w)); 					};
		auto pRatioMod(int_ini_t fP, dbl_ini_t fV)			->_T 			{ return std::exp(this->logPRatioMod(fP, fV)); 						};
		auto pRatioMod(Config_cr_t v)						->_T 			{ return std::exp(this->logPRatioMod(v)); 							};
		auto pRatioMod(Config_cr_t v, Config_cr_t w)		->_T 			{ return std::exp(this->logPRatioMod(v, w)); 						};
		std::function<_T(Config_cr_t)> logPRatioFunc_;						// function for the probability ratio (log)
		std::function<_T(uint)> logPRatioFuncFlips_;						// function for the probability ratio (log) - for the number of flips
		std::function<_T(Config_cr_t)> pRatioFunc_;							// function for the probability ratio
		// ***********************************************************************************************************************************
	protected:																// ----------------------- W E I G H T S -------------------------
		// ***********************************************************************************************************************************
#ifdef NQS_ANGLES_UPD
		virtual void update(uint nFlips = 1); 								// update the weights after one flip (if needed and possible)
		virtual void update(Config_cr_t v, uint nFlips = 1);		 		// update the weights after state change (if needed and possible)
		virtual void unupdate(uint nFlips = 1)								{}; // unupdate the weights after one flip (if needed and possible)
#endif
		// ***********************************************************************************************************************************
	public:
		virtual void setWeights()											= 0; // set the weights of the NQS
		virtual void setWeights(const NQSB& _w);
		virtual void setWeights(NQSB&& _w);
		virtual void setWeights(std::shared_ptr<NQS<_spinModes, _Ht, _T, _stateType>> _nqs); 
	protected:
		virtual void updateWeights();
	public:
		virtual void updateWeights(double _dt)								{ this->dF_ *= _dt; this->updateWeights(); 				};
		virtual void updateWeights(const NQSB& _dF)							{ this->dF_ = _dF; this->updateWeights(); 				};
		virtual void updateWeights(NQSB&& _dF)								{ this->dF_ = std::move(_dF); this->updateWeights(); 	};
		// ***********************************************************************************************************************************
		virtual bool saveWeights(std::string _path, std::string _file) override;	// save the weights to the file 
		virtual bool setWeights(std::string _path, std::string _file) override;		// set the weights from the file
		virtual auto saveInfo(const std::string& _dir, const std::string& _name, int i = 0) const -> void { this->info_p_.saveInfo(_dir, _name, i); };
		// ***********************************************************************************************************************************
	protected:																// --------------------- T R A I N   E T C -----------------------
		virtual void gradF(const Container_t& _energies, int step = 0, _T _cL = 0.0);
#ifdef NQS_USESR
		virtual void gradSR(uint step = 0, _T _cL = 0.0);					// stochastic reconfiguration
		virtual void covMatrixReg(int _step = 0, _T _cL = 0.0);
#endif
		virtual void grad(Config_cr_t _v, uint _plc)						= 0;
		virtual void gradFinal(const Container_t& _energies, int step = 0, _T _cL = 0.0);
		// ***********************************************************************************************************************************
	protected:																// ------------------------ T I M E E V O-------------------------
		virtual void gradEvoFinal(const Container_t& _energies, int step = 0, _T _cL = 0.0, NQSB* _dF = nullptr);
		virtual void gradTime(size_t _step = 0, NQSB* _dF = nullptr);
		// ***********************************************************************************************************************************
	protected:																// ------------------------ E N E R G Y --------------------------
		std::function<_T(int_ini_t, dbl_ini_t)> pKernelFunc_;				// function for the probability ratio - for the kernel
		std::function<_T(int_ini_t, dbl_ini_t)> logPKernelFunc_;			// function for the probability ratio - for the log
		auto locEnKernel()									-> _T;			// local energy kernel - for the energy calculation										
#ifdef NQS_NOT_OMP_MT
		virtual void locEnKernel(uint _start, uint _end, uint _threadNum);
#endif
		// ***********************************************************************************************************************************
	public:																	// ------------------------ S E T T E R S ------------------------
		virtual void setRandomState(bool _upd = true) override;				// set the random state of the NQS	
		virtual void setInfo() 						override				{ this->info_ = "General NQS class";	};
		virtual void init()							override;
		void setEarlyStopping(size_t _pat, double _minDlt);	
		void setSregScheduler(int _sch = 0, double _sreg = 1e-7, double _sregd = 0.96, size_t _epo = 10, size_t _pat = 5);							
		void setTrainParExc(const MonteCarlo::MCS_train_t& _par);  										// training the excited states (if needed)
		void setScheduler(int _sch = 0, double _lr = 1e-3, double _lrd = 0.96, size_t _epo = 10, size_t _pat = 5);									
		// ***********************************************************************************************************************************
	public:																	// ------------------------ G E T T E R S ------------------------
		auto getNvis()							const -> uint				{ return this->info_p_.nVis_;			};
		auto getF()								const -> NQSB				{ return this->F_;						};
		auto full_size()						const -> size_t				{ return this->info_p_.fullSize_;		};
		virtual auto size()						const -> size_t				= 0;
#ifdef NQS_USESR_MAT_USED	
		auto getCovarianceMat()					const -> NQSW				{ return this->S_;						};	
#endif
		// Hilbert	
		auto getHilbertSize()					const -> u64				{ return this->info_p_.Nh_;				};
		auto getHilbertSpace() 					const -> Hilbert_cr_t		{ return this->H_->getHilbertSpace();	};
		// Hamiltonian	
		auto getHamiltonianInfo()				const -> std::string		{ return this->H_->getInfo();			};
		auto getHamiltonianEigVal(u64 _idx)		const -> double				{ return this->H_->getEigVal(_idx);		};
		auto getHamiltonian() 					const -> Hamil_t_p			{ return this->H_;						};
		// overriden MonteCarloSolver methods - get the state
		auto getConfigSize() 					const -> size_t override	{ return this->getNvis();				};
		auto getLastConfig() 					const -> Config_t override	{ return NQS_STATE; 					};
		// overriden MonteCarloSolver methods - set the state
		virtual auto setConfig(Config_cr_t _s) 	-> void override 			{ this->setState(_s, true); 			};
		virtual auto swapWeights(MC_t_p _other) -> void;
		virtual auto swapConfig(MC_t_p _other) 	-> void override;
		virtual auto reset(size_t _n) 			-> void override;
		virtual auto clone() 					const -> MC_t_p override 	= 0;
		virtual auto clone(MC_t_p _other) 		-> void override;
		// ***********************************************************************************************************************************
		// weights
		auto getWeights() 						const -> NQSB				{ return this->Weights_;				};
		auto getWeights_ref() 					-> const NQSB&				{ return this->Weights_;				};
		// ***********************************************************************************************************************************
	public:																	// ----------------------- S A M P L I N G -----------------------
		template <bool _setState = false>
		void blockSample(uint _bSize, NQS_STATE_T _start);
		// !TODO - change this to template to make it more efficient
	public:																	// ------------------------ T R A I N I N G ----------------------
		virtual bool trainStop(size_t i, const MCS_train_t& _par, _T _currLoss, _T _currstd = 0.0, bool _quiet = false) override;
		virtual bool trainStep(size_t i, Container_t& En, Container_t& meanEn, Container_t& stdEn, 
										const MCS_train_t& _par, 
										const bool quiet, const bool randomStart,
										Timer* _timer) 						override;
		virtual Container_pair_t train(	const MCS_train_t& _par,
										bool quiet			= false,		// shall talk? (default is false)
										bool randomStart 	= false,		// random start (default is false)
										clk::time_point _t	= NOW,			// time! (default is NOW)
										uint progPrc		= 25) 			override;	
		// ***********************************************************************************************************************************
	public:																	// --------------------- C O L L E C T I N G ---------------------
		// single 
		template <typename _CT>
		bool collectStep(size_t i, const MCS_train_t& _par, ::Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies = nullptr);
		template <typename _CT>
		void collect(const MonteCarlo::MCS_train_t& _par, ::Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies = nullptr, bool reset = false);
		// more 
		template <typename _CT>
		bool collectStep(size_t i, const MonteCarlo::MCS_train_t& _par, const v_1d<::Operators::OperatorNQS<_T>>& _opG, v_1d<_CT>& _opvals, _CT* _energies = nullptr);
		template <typename _CT>
		void collect(const MonteCarlo::MCS_train_t& _par, const v_1d<::Operators::OperatorNQS<_T>>& _opG, v_1d<_CT>& _opvals, _CT* _energies = nullptr, bool reset = false); 
		// measurement
		virtual bool collectStep(size_t i, const MonteCarlo::MCS_train_t& _par, 
								NQSAv::MeasurementNQS<_T>& _meas,
								Container_t* _E, 
								Container_t* _EM,
								Container_t* _ES, 
								const bool quiet,
								const bool randomStart,
								Timer& _timer);
		virtual void collect(const MonteCarlo::MCS_train_t& _par,
							NQSAv::MeasurementNQS<_T>& _mes 	= {},
							Container_t* _energies			= nullptr,
							Container_t* _energiesStd			= nullptr,
							bool quiet							= false,
							bool randomStart 					= false,
							clk::time_point _t					= NOW,
							uint progPrc						= 25);
		// other
		virtual void collect(const MonteCarlo::MCS_train_t& _par, NQSAv::MeasurementNQS<_T>& _mes);
		// for collecting the \sum _s f(s) / \psi(s) - used for the gradient calculation
		virtual void collect_ratio(const MonteCarlo::MCS_train_t& _par, std::function<_T(Config_cr_t)> _f, Container_t& _container);
		virtual void collect_ratio(const MonteCarlo::MCS_train_t& _par, NQS<_spinModes, _Ht, _T, _stateType>* other, Container_t& _container);
		// ***********************************************************************************************************************************
	public:																	// ----------------------- E V O L V E -----------------------
		// single
		virtual void evolveStep(size_t step, double dt,
								const MonteCarlo::MCS_train_t& _par, 
								const bool quiet, 
								const bool randomStart,
								const bool updateWeights 	= true, 
								NQSB* _dF			 		= nullptr);
		virtual void evolveStepSet(size_t step, double dt, algebra::ODE::IVP<_T, arma::Col<_T>>* _ivp);
		virtual void evolveSet(const MonteCarlo::MCS_train_t& _par, const bool quiet, const bool randomStart);
		auto evolveGet() 													-> NQS_evo_f_t&;
	protected:
		NQS_evo_f_t evolveFunc_;											// function for the time evolution - used with the ODE solver
		NQS_evo_fr_t evolveFuncR_;											// function for the time evolution - used with the ODE solver
	public:																	// ----------------------- F I N A L E -----------------------
		// ***********************************************************************************************************************************	
		virtual auto ansatzlog(Config_cr_t _in)															const -> _T 	= 0;
		virtual auto ansatz(Config_cr_t _in)															const -> _T 	= 0;
		virtual auto ansatz_ratiolog(Config_cr_t _in, NQS<_spinModes, _Ht, _T, _stateType>* _other) 	const -> _T 	= 0;
		virtual auto ansatz_ratio(Config_cr_t _in, NQS<_spinModes, _Ht, _T, _stateType>* _other) 		const -> _T 	{ return std::exp(this->ansatz_ratiolog(_in, _other)); };
		// ***********************************************************************************************************************************
	public:																	// -------------------- C O N S T R U C T --------------------
		// ***********************************************************************************************************************************
		virtual ~NQS() override;
		NQS() = default;
		NQS(const NQS& _n);
		NQS(NQS&& _n);
		// -------------------- O P E R A T O R S --------------------
		NQS &operator=(const NQS & _n);
		NQS &operator=(NQS &&_n);
		// ***********************************************************************************************************************************
		NQS(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p);
		NQS(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const v_1d<double>& _beta);
		// ***********************************************************************************************************************************
		static void save_history(const std::string& _dir, 	
								const arma::Col<_T>& _EN_TRAIN,				// training
								const arma::Col<_T>& _EN_TESTS,				// test
								const arma::Col<_T>& _EN_STD,				// stamdard deviations - training
								const arma::Col<_T>& _EN_TESTS_STD,			// standard deviations - test
								const arma::Col<double>& _betas,			// betas - for the training of the excited states
								arma::Col<_T>& _means,						// means of the energies
								arma::Col<_T>& _stds,						// standard deviations of the energies
								const int _i 				= 0,			// index of the excited state
								const bool _append 			= false,		// append the data to the existing file
								const std::string& _name 	= "history.h5");
		// ***********************************************************************************************************************************
	};
}; // namespace NQS_NS

#endif

// ######################################################################################################################################