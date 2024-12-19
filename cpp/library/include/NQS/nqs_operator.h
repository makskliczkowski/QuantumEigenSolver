#pragma once
/***************************************
* Defines the generic NQS Solver class.
* Allows for later inhertiance
* for a fine model specialization.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

// #################################
#include "./NQS_base/nqs_definitions_base.h"
#include <initializer_list>
#include <utility>
#ifndef HAMIL_H
#	include "../hamil.h"
#endif
// #################################

//////////////////////////////////////////////////////////////////////////////////////////
// Operators for the NQS

// #define NQS_OP_TEMP(_T, FUN, RET, ARGS, ADD) 							\
// 		template RET OperatorNQS<_T>::FUN() ARGS ADD;					\
// 		template RET OperatorNQS<_T, uint>::FUN() ARGS ADD;				\
// 		template RET OperatorNQS<_T, int>::FUN() ARGS	ADD;			\
// 		template RET OperatorNQS<_T, uint, uint>::FUN() ARGS ADD;		\
// 		template RET OperatorNQS<_T, int, int>::FUN() ARGS ADD;			\
		// no more exotics

#ifndef NQS_OPERATOR_H
#	define NQS_OPERATOR_H
namespace Operators
{
	// ##########################################################################################################################################

	template <typename _T, typename ..._Ts>
	class OperatorNQS : public OperatorComb<_T, _Ts...>
	{
		NQS_PUBLIC_TYPES(_T, double);
	public:
		using baseType 		= OperatorComb<_T, _Ts...>;									// type of the operator - base type
		// -----------------------------------------------------------------------------

		// Inherit constructors from GeneralOperator
		using OperatorComb<_T, _Ts...>::OperatorComb;  									
		using OperatorComb<_T, _Ts...>::operator=;
		using OperatorComb<_T, _Ts...>::operator();
		
		// -----------------------------------------------------------------------------
		using _VT 			= typename baseType::_VT;									// type of the vector to be used for the operator
		using _VT_CR 		= typename baseType::_VT_CR;								// type of the vector to be used for the operator - const reference		
	public:
		using repType 		= typename baseType::repType_;								// type of the function to be used for the operator
		using repTypeV 		= typename baseType::repTypeV_;								// type of the function to be used for the operator - for vectors
		using ReturnType 	= typename baseType::ReturnType;							// return type of the operator
		using ReturnTypeV 	= typename baseType::ReturnTypeV;							// return type of the operator - for vectors
		// -----------------------------------------------------------------------------
	protected:
		using fP_t			= std::initializer_list<int>;								// like a vector of integers and values but separated with a fixed number of elements
		using fV_t			= std::initializer_list<double>;						
		// -----------------------------------------------------------------------------
		using NQSFunCol		= std::function<cpx(const NQSS& _v)>;						// for initializing the pRatio function with a single column vector (state)
		using NQSFun		= std::function<cpx(fP_t, fV_t)>; 							// for initializing the pRatio function with initializer list - for the Hamiltonian probability ratio (or other operators)
		_OP_V_T state_;																	// store the column state vector						

		// -----------------------------------------------------------------------------

	public:
		~OperatorNQS()																{};
		OperatorNQS() : baseType() 													{};
		OperatorNQS(const OperatorComb<_T, _Ts...>& comb) : OperatorComb<_T, _Ts...>(comb) {
			// Additional initialization for OperatorNQS, if necessary
		}
		OperatorNQS(OperatorComb<_T, _Ts...>&& comb) : OperatorComb<_T, _Ts...>(std::move(comb)) {
			// Additional initialization for OperatorNQS, if necessary
		}

		// -----------------------------------------------------------------------------

		// equality operator to the OperatorComb class
		OperatorNQS& operator=(const OperatorComb<_T, _Ts...>& comb) {
			OperatorComb<_T, _Ts...>::operator=(comb);
			return *this;
		}

		// equality operator to the OperatorComb class
		OperatorNQS& operator=(OperatorComb<_T, _Ts...>&& comb) {
			OperatorComb<_T, _Ts...>::operator=(std::move(comb));
			return *this;
		}

		// -----------------------------------------------------------------------------

		// operators are to be applied step by step that add up in the average \sum _ s' <s|O|s'> * \Psi(s') / \Psi(s)
		// !we apply the operator to the left and look for all the states that can be reached from the base state (= <s|)
		// the |s'> are every possible state that can be reached from the base state with the operator action on <s|
		// O may in general create a combination of the base states, leave it as a vector then - OperatorComb
		// @note The class uses general implementation of the operator, so it can be used for any operator in the future

		// for the integer type of the state
		auto operator()(u64 s, NQSFunCol _fun, _Ts... a)		 -> _T;
		// for the column vector type of the state
		auto operator()(_OP_V_T_CR s, NQSFunCol _fun, _Ts... a) const -> _T;

		// ------------------------------------------------------------------------------
	};
};

#endif

//////////////////////////////////////////////////////////////////////////////////////////

// Average operators in the NQS
#ifndef NQS_AV_H
#	define NQS_AV_H
namespace NQSAv
{
	// ##########################################################################################################################################

	/**
	* @brief Class that stores the measurements from the NQS and is able to save them.
	* 
	* This class is responsible for managing and storing measurements from the Neural Quantum State (NQS).
	* It provides functionalities to measure, normalize, and save the measurements. The class supports
	* various types of operators including global, local, and correlation operators.
	* 
	* @tparam _T The data type used by the measurement operator.
	*/
	template <typename _T>
	class MeasurementNQS
	{
		NQS_PUBLIC_TYPES(_T, double);
		// for initializing the pRatio function with a single column vector
		using NQSFunCol		= std::function<cpx(const NQSS& _v)>;
		using NQSFunColP	= cpx(*)(const NQSS& _v);
		using OPG			= v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>>;
		using OPL			= v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>>;
		using OPC			= v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>>;
		// -----------------------------------------------------------------------------

	protected:
		bool usedMB_		= false;
		std::string dir_	= "";
		uint threads_		= 1;
		uint Ns_			= 0;		
		std::shared_ptr<Lattice> lat_;							// lattice for the operators (only if necessary)
		OPG opG_;												// global operators
		OPL opL_;												// local operators
		OPC opC_;												// correlation operators

		// -----------------------------------------------------------------------------

		// create a container for the operators 
		std::vector<Operators::Containers::OperatorContainer<_T>> containersG_;
		std::vector<Operators::Containers::OperatorContainer<_T>> containersL_;
		std::vector<Operators::Containers::OperatorContainer<_T>> containersC_;	

		// -----------------------------------------------------------------------------

		void createContainers();

	public:
		~MeasurementNQS()										= default;
		MeasurementNQS()										= default;
		MeasurementNQS(const MeasurementNQS& _m) 				= default;
		MeasurementNQS(MeasurementNQS&& _m) 					= default;
		MeasurementNQS& operator=(const MeasurementNQS& _m) 	= default;
		MeasurementNQS& operator=(MeasurementNQS&& _m) 			= default;				// copy and move operators

		// ---- CONSTRUCTORS ----
		MeasurementNQS(std::shared_ptr<Lattice> _lat, const strVec& _operators);		// working with the operators and the lattice
		MeasurementNQS(size_t _Ns, const strVec& _operators);							// working with the operators and the size of the degrees of freedom

		// working with the operators
		MeasurementNQS(std::shared_ptr<Lattice> _lat, const std::string& _dir,
													  const OPG& _opG,
													  const OPL& _opL = {},
													  const OPC& _opC = {},
													  uint _threadNum = 1);
		MeasurementNQS(size_t _Ns,  const std::string& _dir,
									const OPG& _opG,
									const OPL& _opL = {},
									const OPC& _opC = {},
									uint _threadNum = 1);
		MeasurementNQS(size_t _Ns,	const OPG& _opG,
									const OPL& _opL = {},
									const OPC& _opC = {},
									uint _threadNum = 1);



		// ---- MEASUREMENT ----

		// !NQS
		void measure(u64 s, NQSFunCol _fun);											// measure the operators for the given state and an NQS function
		void measure(Operators::_OP_V_T_CR, NQSFunCol _fun);							// measure the operators for the given state and an NQS function			
		void normalize(uint _nBlck);													// normalize the operators based on the NQS Monte Carlo samples

		// !ED
		void measure(const arma::Col<_T>& state, const Hilbert::HilbertSpace<_T>&);		// measure the operators for the given state (from the exact diagonalization)
		_T measureGlob(const arma::Col<_T>& state, const Hilbert::HilbertSpace<_T>&, int which);									// measure the global operators for the given state
		_T measureLocl(const arma::Col<_T>& state, const Hilbert::HilbertSpace<_T>&, int which, uint site);						// measure the local operators for the given state
		_T measureCorr(const arma::Col<_T>& state, const Hilbert::HilbertSpace<_T>&, int which, uint site1, uint site2);			// measure the correlation operators for the given state
		arma::Col<_T> applyGlob(const arma::Col<_T>& state, const Hilbert::HilbertSpace<_T>&, int which);							// apply the operator to the state
		arma::Col<_T> applyLocl(const arma::Col<_T>& state, const Hilbert::HilbertSpace<_T>&, int which, uint site);				// apply the operator to the state
		arma::Col<_T> applyCorr(const arma::Col<_T>& state, const Hilbert::HilbertSpace<_T>&, int which, uint site1, uint site2);	// apply the operator to the state
		void save(const strVec& _ext 	= { ".h5" }, 
				std::string _nameGlobal = "", 
				std::string _nameLocal 	= "", 
				std::string _nameCorr 	= "",
				std::string _appName 	= "",
				bool _app 				= true);										// save the measurements
		void saveMB(const strVec& _ext 	= { ".h5" }, 
				std::string _nameGlobal = "", 
				std::string _nameLocal 	= "", 
				std::string _nameCorr 	= "", 
				std::string _appName 	= "",
				bool app 				= true);										// save the many body measurements
		void saveNQS(const strVec& _ext = { ".h5" },
				std::string _nameGlobal = "",
				std::string _nameLocal 	= "",
				std::string _nameCorr 	= "",
				std::string _appName 	= "",
				bool app				= true);										// save the NQS measurements
		

		// ---- MEASUREMENT ---- (STATIC)
		
		static _T measure(Operators::_OP_V_T_CR _state, const Operators::OperatorNQS<_T>& _gO, NQSFunCol _fun, Operators::Containers::OperatorContainer<_T>& _cont);
		static void normalize(uint _nBlck, Operators::Containers::OperatorContainer<_T>& _cont) { _cont.normalize(_nBlck, true); };

		// ---- GETTERS ----
		auto getOpG()				const		->		const OPG& { return this->opG_; };
		auto getOpG(int i)			const		->		const std::shared_ptr<Operators::OperatorNQS<_T>>& { return this->opG_[i]; };
		auto getOpL()				const		->		const OPL& { return this->opL_; };
		auto getOpL(int i)			const		->		const std::shared_ptr<Operators::OperatorNQS<_T, uint>>& { return this->opL_[i]; };
		auto getOpC()				const		->		const OPC& { return this->opC_; };
		auto getOpC(int i)			const		->		const std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>& { return this->opC_[i]; };
		auto getDir()				const		->		const std::string& { return this->dir_; };
		// values from the MB containers	
		auto getMBCont_G(uint which) 					const -> _T;
		auto getMBCont_L(uint which, uint i)			const -> _T;
		auto getMBCont_C(uint which, uint i, uint j) 	const -> _T;

		// values from the containers
		auto getContMean_G(uint i) 	const		-> 		_T { return this->containersG_[i].template mean<_T>()(0,0); };

		// ---- SETTERS ----
		void setDir(const std::string& _dir)			{ this->dir_ = _dir; };
		void setThreads(uint _threads)					{ this->threads_ = _threads; };
		void setLat(std::shared_ptr<Lattice> _lat) 		{ this->lat_ = _lat; this->Ns_ = _lat->get_Ns(); };
		void setLat(size_t _Ns) 						{ this->Ns_ = _Ns; this->lat_ = nullptr; };
		void setOP_G(const OPG& _opG)					{ this->opG_ = _opG; };
		void setOP_L(const OPL& _opL)					{ this->opL_ = _opL; };
		void setOP_C(const OPC& _opC)					{ this->opC_ = _opC; };

		// ----- RESET -----
		void resetContainers();
		void reset();
	};

	// ##########################################################################################################################################

	/*
	* @brief Apply the operators with a value change (!with pRatio). The function is used to calculate the measuremnet of given a state and the operator and container
	* @param _state base state to apply the operators to
	* @param _gO operator to apply
	* @param _fun pRatio function from the NQS
	* @param _cont container to store the results
	* @returns value of the operator acting on the state with the probability ratio applied - the base state. The value is also stored in the container...
	*/
	template <typename _T>
	inline _T NQSAv::MeasurementNQS<_T>::measure(Operators::_OP_V_T_CR _state, const Operators::OperatorNQS<_T>& _gO, NQSFunCol _fun, Operators::Containers::OperatorContainer<_T>& _cont)
	{
		auto val = _gO(_state, _fun);
		_cont.updCurrent(val);
		return val;
	};

	// ##########################################################################################################################################

	// ###################################################### C L A S S   C O N S T U C T #######################################################

	// ##########################################################################################################################################
	
	template <typename _T>
	inline NQSAv::MeasurementNQS<_T>::MeasurementNQS(std::shared_ptr<Lattice> _lat, const strVec& _operators)
		: Ns_(_lat->get_Ns()), lat_(_lat)
	{
		// CONSTRUCTOR_CALL;
	}

	template<typename _T>
	inline  NQSAv::MeasurementNQS<_T>::MeasurementNQS(size_t _Ns, const strVec & _operators)
		: Ns_(_Ns), lat_(nullptr)
	{
		// CONSTRUCTOR_CALL;
		// !TODO 
		// create the operators from the strings and add them to the list
	}

	////////////////////////////////////////////////////////////////////////////

	/**
	* @brief Resets the containers for the operators.
	*
	* This function resets the containers for the operators. It clears the internal containers
	* `containersG_`, `containersL_`, and `containersC_` and then calls `createContainers()`
	* to reset any additional containers or states associated with the measurement operator.
	*
	* @tparam _T The data type used by the measurement operator.
	*/
	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::resetContainers() { this->createContainers(); }

	/**
	* @brief Resets the measurement operator for the NQS (Neural Quantum State).
	*
	* This function clears the internal containers `opG_`, `opL_`, and `opC_` 
	* and then calls `resetContainers()` to reset any additional containers 
	* or states associated with the measurement operator.
	*
	* @tparam _T The data type used by the measurement operator.
	*/
	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::reset() {
		this->opG_.clear();
		this->opL_.clear();
		this->opC_.clear();
		this->resetContainers();
	}
	
	////////////////////////////////////////////////////////////////////////////

	/**
	* @brief Create the containers for the operators.
	* 
	* This function initializes the containers for the global, local, and correlation operators
	* stored in the class. It clears any existing containers and creates new ones based on the 
	* number of sites (Ns_) and the type of operators.
	*/
	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::createContainers()
	{
		// Clear existing global containers and create new ones
		this->containersG_.clear();
		for (const auto& _ [[maybe_unused]] : this->opG_)
		{
			auto _cont = Operators::Containers::OperatorContainer<_T>(this->Ns_);
			_cont.decideSize(); // Decide size for global containers
			this->containersG_.push_back(_cont);
		}

		// Clear existing local containers and create new ones
		this->containersL_.clear();
		for (const auto& _ [[maybe_unused]] : this->opL_)
		{
			auto _cont = Operators::Containers::OperatorContainer<_T>(this->Ns_);
			_cont.template decideSize<uint>(); // Decide size for local containers
			this->containersL_.push_back(_cont);
		}

		// Clear existing correlation containers and create new ones
		this->containersC_.clear();
		for (const auto& _ [[maybe_unused]] : this->opC_)
		{
			auto _cont = Operators::Containers::OperatorContainer<_T>(this->Ns_);
			_cont.template decideSize<uint, uint>(); // Decide size for correlation containers
			this->containersC_.push_back(_cont);
		}
	}

	////////////////////////////////////////////////////////////////////////////
	
	template <typename _T>
	inline NQSAv::MeasurementNQS<_T>::MeasurementNQS(std::shared_ptr<Lattice> _lat,  
													const std::string& _dir,
													const OPG& _opG,
													const OPL& _opL, 
													const OPC& _opC,
													uint _threadNum)
		: dir_(_dir), threads_(_threadNum), Ns_(_lat->get_Ns()), lat_(_lat)
	{
		// create directory
		makeDir(_dir);

		this->opG_ = _opG;
		this->opL_ = _opL;
		this->opC_ = _opC;

		// create containers
		this->createContainers();

		CONSTRUCTOR_CALL;
	}
	
	template<typename _T>
	inline MeasurementNQS<_T>::MeasurementNQS(size_t _Ns, const OPG& _opG, const OPL& _opL, const OPC& _opC, uint _threadNum)
		: threads_(_threadNum), lat_(nullptr), Ns_(_Ns)
	{
		this->opG_ = _opG;
		this->opL_ = _opL;
		this->opC_ = _opC;

		// create containers
		this->createContainers();
	}

	template<typename _T>
	inline MeasurementNQS<_T>::MeasurementNQS(size_t _Ns, const std::string& _dir, const OPG& _opG, const OPL& _opL, const OPC& _opC, uint _threadNum)
		: MeasurementNQS<_T>::MeasurementNQS(_Ns, _opG, _opL, _opC, _threadNum)
	{
		// create directory
		makeDir(_dir);
		CONSTRUCTOR_CALL;
	}

	// ##########################################################################################################################################

	/**
	* @brief Normalize the operators - divide by the number of samples for each operator and 
	* for each container (block). The normalization is done for the global, local, and correlation operators.
	* This happens when one wants to calculate the mean in the bin.
	* @param _nsamples number of samples
	*/
	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::normalize(uint _nsamples)
	{
		BEGIN_CATCH_HANDLER
		{
			for (auto& _cont : this->containersG_)
				_cont.normalize(_nsamples);
		}
		END_CATCH_HANDLER("Problem in the normalization of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			for (auto& _cont : this->containersL_)
				_cont.normalize(_nsamples);
		}
		END_CATCH_HANDLER("Problem in the normalization of local operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			for (auto& _cont : this->containersC_)
				_cont.normalize(_nsamples);
		}
		END_CATCH_HANDLER("Problem in the normalization of correlation operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////
};
#endif 
//////////////////////////////////////////////////////////////////////////////////////////
