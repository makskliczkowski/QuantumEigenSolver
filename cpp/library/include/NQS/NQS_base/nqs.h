/***************************************
* Defines the generic NQS Solver class. 
* Allows for later inhertiance
* for a fine model specialization.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#include "../nqs_operator.h"
#include "../../algebra/general_operator.h"
#include "armadillo"
#include <cstddef>
#include <functional>
#include <memory>
#ifndef NQS_H
#define NQS_H

// include all the definions
#include "nqs_definitions_base.h"
#include "nqs_definitions_lower.tpp"

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
class NQS : public MonteCarlo::MonteCarloSolver<_T, _stateType, arma::Col<_stateType>>
{	
	NQS_PUBLIC_TYPES(_T, _stateType);									// type definitions 
	MCS_PUBLIC_TYPES(_T, _stateType, arma::Col); 						// type definitions for the Monte Carlo solver
	using Hamil_t 	= Hamiltonian<_Ht, _spinModes>;						// Hamiltonian type
	using Hamil_t_p = std::shared_ptr<Hamil_t>;							// shared pointer to the Hamiltonian type
public:
	struct NQS_RK_CONTAINER
	{
		NQSB k1, k2, k3, k4;																						
	} RK_;																// container for the Runge-Kutta method

	// for the lower states
	using NQSLS_p 						=		typename NQS_lower_t<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
	NQS_info_t info_p_;													// information about the NQS
	NQS_lower_t<_spinModes, _Ht, _T, _stateType> lower_states_;			// information about the training

protected:
	// for the Hamiltonian information, types and the Hilbert space
	const double discVal_				=		Operators::_SPIN_RBM;   // discrete value for the numbers in vector representation
	const uint spinModes_				=		_spinModes;				// number of spin modes -> e.g. 2 for hardcore bosons, 4 for fermions
	bool updateWeights_ 				= 		true;					// shall update the weights in current step? (default is true)

protected:
	Hamil_t_p H_;														// pointer to the Hamiltonian instance (for the energy calculation)

protected:																// ---------------------- T H R E A D I N G ---------------------
#ifdef NQS_NOT_OMP_MT
	bool initThreads(uint _threadNum = 0);
	NQS_thread_t<_T> threads_;											// thread information
#endif

	// ----------------------- T R A I N I N G ----------------------
	uint nFlip_							=		1;						// number of flips to be done in one step (each flip is a change in the state)
	v_1d<uint> flipPlaces_;												// stores flip spots to be flipped during one sampling step
	v_1d<_stateType> flipVals_;											// stores values before (!!!) the flip to be used for the gradients
	
	Config_t curVec_;													// currently processed state vector for convenience
	u64 curState_						=		0;						// currently processed state - may or may not be used
	
	v_1d<Config_t> tmpVecs_;											// temporary vectors for the flips
	Config_t tmpVec_;													// temporary vector for the flips (for the current state)
	u64 tmpState_						=		0;						// temporary state for the flips
	
	// ------------------------ W E I G H T S -----------------------
	NQSW derivatives_;													// store the variational derivatives F_k (nBlocks x fullSize), where nBlocks is the number of consecutive observations
	NQS_ROW_T derivativesMean_;											// store the mean of the derivatives (F_k) - for the SR (fullSize)
	NQSW derivativesCentered_;											// store the centered derivatives (F_k - <F_k>) - for the SR (nBlocks x fullSize), where nBlocks is the number of consecutive observations
	NQSW derivativesCenteredH_;											// store the centered derivatives (F_k - <F_k>) - for the SR (fullSize x nBlocks), where nBlocks is the number of consecutive observations	
	void derivativesReset(size_t nBlocks = 1);							// reset the derivatives (F_k) - for the SR		
#ifdef NQS_USESR_MAT_USED
	NQSW S_;															// positive semi-definite covariance matrix - to be optimized (inverse of the Fisher information matrix)
#else 
	Precond_t_p precond_ 				= 		nullptr;				// preconditioner for the conjugate gradient
	Solver_t_p solver_ 					= 		nullptr;				// solver for the Fisher matrix inversion
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

	virtual auto logPRatio(uint fP, float fV)			->_T			= 0; // log of the probability ratio	- when flip places (single) are used
	virtual auto pRatio(uint fP, float fV)				->_T			{ return std::exp(this->logPRatio(fP, fV));										};

	virtual auto logPRatio()							->_T			{ return this->logPRatio(this->flipPlaces_[0], this->flipVals_[0]);				};
	virtual auto pRatio()								->_T			{ return this->pRatio(this->flipPlaces_[0], this->flipVals_[0]);				};	

	virtual auto logPRatio(uint nFlips)					->_T			= 0; // log of the probability ratio	- when number of flips is used
	virtual auto pRatio(uint nFlips)					->_T			{ return std::exp(this->logPRatio(nFlips));										};	

	virtual auto logPRatio(const NQSS& _v)				->_T			{ return this->logPRatio(this->curVec_, _v);									};
	virtual auto pRatio(const NQSS& _v)					->_T			{ return std::exp(this->logPRatio(_v));											};	

	virtual auto logPRatio(	const NQSS& _v1, 
							const NQSS& _v2)			->_T			= 0; // log of the probability ratio	- when two vectors are used (one for the current state and one for the new state)
	virtual auto pRatio(const NQSS& _v1,
						const NQSS& _v2)				->_T			{ return std::exp(this->logPRatio(_v1, _v2));									};
	
	virtual auto logPRatio(int_ini_t fP, dbl_ini_t fV) -> _T			= 0; // ratio when exact points are provided (used for the Hamiltonian probability ratio - when the Hamiltonian changes the state)
	virtual auto pRatio(int_ini_t fP, dbl_ini_t fV)		->_T			{ return std::exp(this->logPRatio(fP, fV));										};
	std::function<_T(const NQSS&)> logPRatioFunc_;						// function for the probability ratio (log)
	std::function<_T(const NQSS&)> pRatioFunc_;							// function for the probability ratio

	// ----------------------- W E I G H T S -------------------------
#ifdef NQS_ANGLES_UPD
	virtual void update(uint nFlips = 1)								{};
	virtual void update(const Config_t& v, uint nFlips = 1)				{};
	virtual void unupdate(uint nFlips = 1)								{};
#endif
	virtual void updateWeights()										= 0;
public:
	virtual bool saveWeights(std::string _path, std::string _file);
	virtual bool setWeights(std::string _path, std::string _file);

protected:
	// --------------------- T R A I N   E T C -----------------------
	virtual void grad(const Config_t& _v, uint _plc)					= 0;
	virtual void gradFinal(const NQSB& _energies, int step = 0, _T _cL = 0.0);
	virtual void gradEvoFinal(const NQSB& _energies, int step = 0, double _dt = 1e-4, _T _cL = 0.0, const bool _useRungeKutta = false);
#ifdef NQS_USESR
	virtual void gradSR(uint step = 0, _T _cL = 0.0);					// stochastic reconfiguration
	virtual void covMatrixReg(int _step = 0, _T _cL = 0.0);
#endif
	virtual void gradTime(size_t _step = 0, double _dt = 1e-4, _T _cL = 0.0, const bool _useRungeKutta = false);
	
	// ------------------------ E N E R G Y --------------------------
	std::function<_T(int_ini_t, dbl_ini_t)> pKernelFunc_;				// function for the probability ratio
	std::function<_T(int_ini_t, dbl_ini_t)> logPKernelFunc_;			// function for the probability ratio

protected:
	_T locEnKernel();
#ifdef NQS_NOT_OMP_MT
	virtual void locEnKernel(uint _start, uint _end, uint _threadNum);
#endif

public:
	// ------------------------ S E T T E R S ------------------------
	virtual void setRandomState(bool _upd = true) override;					// set the random state of the NQS	
	virtual void setInfo() 						override					{ this->info_ = "General NQS class";	};
	virtual void init()							override;
	void setEarlyStopping(size_t _pat, double _minDlt);	
	void setSregScheduler(int _sch = 0, double _sreg = 1e-7, double _sregd = 0.96, size_t _epo = 10, size_t _pat = 5);							
	void setTrainParExc(const MonteCarlo::MCS_train_t& _par);  										// training the excited states (if needed)
	void setScheduler(int _sch = 0, double _lr = 1e-3, double _lrd = 0.96, size_t _epo = 10, size_t _pat = 5);									

	// ------------------------ G E T T E R S ------------------------
	auto saveInfo(const std::string& _dir, const std::string& _name, int i = 0) const -> void { this->info_p_.saveInfo(_dir, _name, i); };
	auto getNvis()								const -> uint				{ return this->info_p_.nVis_;			};
	auto getF()									const -> NQSB				{ return this->F_;						};
#ifdef NQS_USESR_MAT_USED	
	auto getCovarianceMat()						const -> NQSW				{ return this->S_;						};	
#endif	
	// Hilbert	
	auto getHilbertSize()						const -> u64				{ return this->info_p_.Nh_;				};
	// Hamiltonian	
	auto getHamiltonianInfo()					const -> std::string		{ return this->H_->getInfo();			};
	auto getHamiltonianEigVal(u64 _idx)			const -> double				{ return this->H_->getEigVal(_idx);		};
	std::shared_ptr<Hamiltonian<_Ht, _spinModes>> getHamiltonian() const	{ return this->H_;						};
	Hilbert::HilbertSpace<_Ht, _spinModes> getHilbertSpace() const   		{ return this->H_->getHilbertSpace();	};
	// overriden MonteCarloSolver methods - get the state
	virtual auto getLastConfig() 				const -> Config_t override 	{ return NQS_STATE; 					};
	// overriden MonteCarloSolver methods - set the state
	virtual auto setConfig(const Config_t& _s) 	-> void override 			{ this->setState(_s); 					};
	virtual auto swapConfig(MC_t_p _other) 		-> void override;
	virtual auto reset(size_t _n) 				-> void override;
	virtual auto clone() 						const -> MC_t_p override 	= 0;
public:

	// ----------------------- S A M P L I N G -----------------------
	virtual void blockSample(uint _bSize, NQS_STATE_T _start, bool _therm = false);

	// single
	virtual bool trainStop(size_t i, const MCS_train_t& _par, _T _currLoss, _T _currstd = 0.0, bool _quiet = false) override;
	virtual bool trainStep(size_t i,Container_t& En,
									Container_t& meanEn, 
									Container_t& stdEn, 
									const MCS_train_t& _par, 
									const bool quiet, 
									const bool randomStart,
									Timer& _timer) override;

	virtual Container_pair_t train(	const MCS_train_t& _par,
									bool quiet			= false,		// shall talk? (default is false)
								  	bool randomStart 	= false,		// random start (default is false)
								  	clk::time_point _t	= NOW,			// time! (default is NOW)
									uint progPrc		= 25) override;	// progress percentage (default is 25)

	// --------------------- C O L L E C T I N G ---------------------

	// single 
	template <typename _CT>
	bool collectStep(size_t i, const MCS_train_t& _par, Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies = nullptr);
	template <typename _CT>
	void collect(const MonteCarlo::MCS_train_t& _par, Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies = nullptr, bool reset = false);

	// measurement
	virtual bool collectStep(size_t i, const MonteCarlo::MCS_train_t& _par, 
							NQSAv::MeasurementNQS<_T>& _meas,
							arma::Col<_T>* _E, 
							arma::Col<_T>* _EM,
							arma::Col<_T>* _ES, 
							const bool quiet,
							const bool randomStart,
							Timer& _timer);
	virtual void collect(const MonteCarlo::MCS_train_t& _par,
						NQSAv::MeasurementNQS<_T>& _mes 	= {},
						arma::Col<_T>* _energies			= nullptr,
						arma::Col<_T>* _energiesStd			= nullptr,
						bool quiet							= false,
						bool randomStart 					= false,
						clk::time_point _t					= NOW,
						uint progPrc						= 25);
	// other
	virtual void collect(const MonteCarlo::MCS_train_t& _par, NQSAv::MeasurementNQS<_T>& _mes);
	// for collecting the \sum _s f(s) / \psi(s) - used for the gradient calculation
	virtual void collect_ratio(const MonteCarlo::MCS_train_t& _par, std::function<_T(const NQSS&)> _f, arma::Col<_T>& _container);
	virtual void collect_ratio(const MonteCarlo::MCS_train_t& _par, NQS<_spinModes, _Ht, _T, _stateType>* other, arma::Col<_T>& _container);

	// ----------------------- E V O L V E -----------------------

	// single
	virtual bool evolveStep(size_t step, double dt, arma::Col<_T>& En,
							const MonteCarlo::MCS_train_t& _par, 
							const bool quiet, 
							const bool randomStart,
							Timer& _timer,
							const bool _useRungeKutta 		= false);
							
	// ----------------------- F I N A L E -----------------------
	virtual auto ansatzlog(const NQSS& _in)															const -> _T 	= 0;
	virtual auto ansatz(const NQSS& _in)															const -> _T 	= 0;
	virtual auto ansatz_ratiolog(const NQSS& _in, NQS<_spinModes, _Ht, _T, _stateType>* _other) 	const -> _T 	= 0;
	virtual auto ansatz_ratio(const NQSS& _in, NQS<_spinModes, _Ht, _T, _stateType>* _other) 		const -> _T 	{ return std::exp(this->ansatz_ratiolog(_in, _other)); };

	// -------------------- C O N S T R U C T --------------------

public:
	virtual ~NQS() override;
	NQS() = default;
	NQS(const NQS& _n);
	NQS(NQS&& _n);

	// -------------------- O P E R A T O R S --------------------
	NQS &operator=(const NQS & _n);
	NQS &operator=(NQS &&_n);
	NQS(Hamil_t_p _H, double _lr = 1e-2, uint _threadNum = 1, int _nParticles = -1, const NQSLS_p& _lower = {}, const v_1d<double>& _beta = {});
};

// ##########################################################################################################################################

#endif