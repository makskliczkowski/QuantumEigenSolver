#pragma once
/***************************************
* Defines the generic NQS Solver class.
* Allows for later inhertiance
* for a fine model specialization.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

// #################################
#include "nqs_definitions.h"
#include <initializer_list>
#ifndef HAMIL_H
#	include "../hamil.h"
#endif
// #################################

// #################################
#ifndef ML_H					// #
#	include "../ml.h"			// #
#endif // !ML_H					// #
// #################################


//////////////////////////////////////////////////////////////////////////////////////////
// Operators for the NQS
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
		
		// -----------------------------------------------------------------------------
		using _VT 			= baseType::_VT;											// type of the vector to be used for the operator
		using _VT_CR 		= baseType::_VT_CR;											// type of the vector to be used for the operator - const reference		
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

		// -----------------------------------------------------------------------------

	public:
		~OperatorNQS()																{};
		OperatorNQS() : baseType() 													{};

		// -----------------------------------------------------------------------------

		// operators are to be applied step by step that add up in the average \sum _ s' <s|O|s'> * \Psi(s') / \Psi(s)
		// !we apply the operator to the left and look for all the states that can be reached from the base state (= <s|)
		// the |s'> are every possible state that can be reached from the base state with the operator action on <s|
		// O may in general create a combination of the base states, leave it as a vector then - OperatorComb
		// @note The class uses general implementation of the operator, so it can be used for any operator in the future

		// for the integer type of the state
		auto operator()(u64 s, NQSFunCol _fun, _Ts... a)		-> _T;
		// for the column vector type of the state
		auto operator()(_OP_V_T_CR s, NQSFunCol _fun, _Ts... a) -> _T;

		// ------------------------------------------------------------------------------
	};

	// ##########################################################################################################################################

	/*
	* @brief Apply the operators with a value change (!with pRatio). The function is used to calculate the 
	* probability ratio for the given state and the operator.
	* @param s base state to apply the operators to
	* @param _fun pRatio function from the NQS
	* @param ...a additional parameters to the operators
	* @returns value of the operator acting on the state with the probability ratio applied
	*/
	template<typename _T, typename ..._Ts>
	inline _T Operators::OperatorNQS<_T, _Ts...>::operator()(u64 s, NQSFunCol _fun, _Ts ...a)
	{
		// starting value
		// this->container_.currentIdx_	= 0;
		_T _valTotal = 0.0;

		// go through operator acting on the state
		for (auto& [s2, _val] : OperatorComb<_T, _Ts...>::operator()(s, a...))
		{
			// transform to state
			INT_TO_BASE(s2, this->state_, Operators::_SPIN_RBM);

			// calculate the probability ratio
			_valTotal += _val * algebra::cast<_T>(_fun(this->state_));
		}
		// this->updCurrent(_valTotal, a...);
		return algebra::cast<_T>(_valTotal);
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Apply the operators with a value change (!with pRatio). The function is used to calculate the
	* probability ratio for the given state and the operator.
	* @param s base state to apply the operators to - vector version
	* @param _fun pRatio function from the NQS
	* @param ...a additional parameters to the operators
	* @returns value of the operator acting on the state with the probability ratio applied
	*/
	template<typename _T, typename ..._Ts>
	inline _T Operators::OperatorNQS<_T, _Ts...>::operator()(_OP_V_T_CR s, NQSFunCol _fun, _Ts ...a)
	{
		// starting value
		// this->container_.currentIdx_	= 0;
		_T _valTotal = 0.0;

		// go through operator acting on the state
		for (auto& [s2, _val] : OperatorComb<_T, _Ts...>::operator()(s, a...))
		{
			// calculate the probability ratio
			_valTotal += _val * algebra::cast<_T>(_fun(s2));
		}
		// this->updCurrent(_valTotal, a...);
		return algebra::cast<_T>(_valTotal);
	}

	// ##########################################################################################################################################
};

#endif

//////////////////////////////////////////////////////////////////////////////////////////

// Average operators in the NQS
#ifndef NQS_AV_H
#	define NQS_AV_H
namespace NQSAv
{
//	// CREATE A MAPPING TO DECIDE WHICH OPERATORS TO USE
//#define MEASUREMENT_OPERATORS_CREATE(WHICH) WHICH, WHICH##TOT, WHICH##COR,
//#define MEASUREMENT_OPERATORS_DECLARE(WHICH) DECL_ENUM_ELEMENT(WHICH), DECL_ENUM_ELEMENT(WHICH##TOT), DECL_ENUM_ELEMENT(WHICH##COR),
//	/*
//	* @brief Operators that are available for the measurement
//	* !TODO Generalize me!
//	*/
//	enum class MeasurementNQSOperators
//	{
//		MEASUREMENT_OPERATORS_CREATE(SZ)
//		MEASUREMENT_OPERATORS_CREATE(SX)
//	};
//
//	BEGIN_ENUMC(MeasurementNQSOperators)
//	{
//		MEASUREMENT_OPERATORS_DECLARE(SZ)
//		MEASUREMENT_OPERATORS_DECLARE(SX)
//	}
//	END_ENUMC(MeasurementNQSOperators);
//
//#define MEASUREMENT_OPERATORS_MAP(WHICH, ENUM_NAME)	{getSTR_##ENUM_NAME(WHICH), ENUM_NAME::##WHICH},				\
//																		{getSTR_##ENUM_NAME(WHICH##TOT), ENUM_NAME::##WHICH##TOT},	\
//																		{getSTR_##ENUM_NAME(WHICH##COR), ENUM_NAME::##WHICH##COR},
//	// create a map
//	std::map<std::string, MeasurementNQSOperators> MeasurementNQSOperatorsMap = {
//		MEASUREMENT_OPERATORS_MAP(SZ, MeasurementNQSOperators)
//		MEASUREMENT_OPERATORS_MAP(SX, MeasurementNQSOperators)
//	};

	/*
	* @brief Class that stores the measurements from the NQS and is able to save them.
	*/
	template <typename _T>
	class MeasurementNQS
	{
		NQS_PUBLIC_TYPES(_T, double);
		// for initializing the pRatio function with a single column vector
		using NQSFunCol		= std::function<cpx(const NQSS& _v)>;
		using OPG			= v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>>;
		using OPL			= v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>>;
		using OPC			= v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>>;
	protected:
		std::string dir_	= "";
		uint threads_		= 1;

		// lattice (if needed)
		std::shared_ptr<Lattice> lat_;
		uint Ns_			= 0;

		// global operators
		OPG opG_;
		// local operators
		OPL opL_;
		// correlation operators
		OPC opC_;

		// -----------------------------------------------------------------------------

		// create a container for the operators 
		std::vector<Operators::Containers::OperatorContainer<_T>> containersG_;
		std::vector<Operators::Containers::OperatorContainer<_T>> containersL_;
		std::vector<Operators::Containers::OperatorContainer<_T>> containersC_;	

		// -----------------------------------------------------------------------------

		void createContainers();

	public:
		~MeasurementNQS() 
		{ 
			LOGINFO("Destroying the measurement of the NQS averages.", LOG_TYPES::TRACE, 3);
			for (auto& x : opG_) x.reset(); 
			for (auto& x : opL_) x.reset();
			for (auto& x : opC_) x.reset();
		}
		MeasurementNQS()						= default;
		MeasurementNQS(const MeasurementNQS&)	= default;
		MeasurementNQS(MeasurementNQS&&)		= default;

		// ---- CONSTRUCTORS ----
		MeasurementNQS(std::shared_ptr<Lattice> _lat, const strVec& _operators);
		MeasurementNQS(size_t _Ns, const strVec& _operators);

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



		// ---- MEASUREMENT ----

		void measure(u64 s, NQSFunCol _fun);
		void measure(Operators::_OP_V_T_CR, NQSFunCol _fun);
		void measure(arma::Col<_T> _state, const Hilbert::HilbertSpace<_T>&);
		void normalize(uint _nBlck);
		void save(const strVec& _ext = { ".h5" });

		// ---- GETTERS ----
		auto getOpG()				const		->		const OPG& { return this->opG_; };
		auto getOpL()				const		->		const OPL& { return this->opL_; };
		auto getOpC()				const		->		const OPC& { return this->opC_; };
		auto getDir()				const		->		const std::string& { return this->dir_; };

	};

	// ##########################################################################################################################################

	// ###################################################### C L A S S   C O N S T U C T #######################################################

	// ##########################################################################################################################################
	
	template <typename _T>
	inline NQSAv::MeasurementNQS<_T>::MeasurementNQS(std::shared_ptr<Lattice> _lat, const strVec& _operators)
		: Ns_(_lat->get_Ns()), lat_(_lat)
	{
		CONSTRUCTOR_CALL;
	}

	template<typename _T>
	inline MeasurementNQS<_T>::MeasurementNQS(size_t _Ns, const strVec & _operators)
		: Ns_(_Ns), lat_(nullptr)
	{
		CONSTRUCTOR_CALL;
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Create the containers for the operators
	* @note The containers are created for the operators that are stored in the class.
	*/
	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::createContainers()
	{
		// create containers
		this->containersG_.clear();
		for (const auto& _ : this->opG_)
		{
			auto _cont = Operators::Containers::OperatorContainer<_T>(this->Ns_);
			// global containers
			_cont.decideSize();
			// add to the list
			this->containersG_.push_back(_cont);
		}

		this->containersL_.clear();
		for (const auto& _ : this->opL_)
		{
			auto _cont = Operators::Containers::OperatorContainer<_T>(this->Ns_);
			// local containers
			_cont.template decideSize<uint>();
			// add to the list
			this->containersL_.push_back(_cont);
		}

		this->containersC_.clear();		
		for (const auto& _ : this->opC_)
		{
			auto _cont = Operators::Containers::OperatorContainer<_T>(this->Ns_);
			// correlation containers
			_cont.template decideSize<uint, uint>();
			// add to the list
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
		: dir_(_dir), threads_(_threadNum), lat_(_lat), Ns_(_lat->get_Ns())
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
	inline MeasurementNQS<_T>::MeasurementNQS(size_t _Ns, const std::string& _dir, const OPG& _opG, const OPL& _opL, const OPC& _opC, uint _threadNum)
		: dir_(_dir), threads_(_threadNum), lat_(nullptr), Ns_(_Ns)
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

	// ##########################################################################################################################################
	
	// ####################################################### C L A S S   M E A S U R E ########################################################
	
	// ##########################################################################################################################################
	
	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::measure(u64 s, NQSFunCol _fun)
	{
		BEGIN_CATCH_HANDLER
		{
			// measure global
			for (int i = 0; i < this->opG_.size(); ++i)
			{
				auto& _op 	= this->opG_[i];
				auto val 	= _op->operator()(s, _fun);
				// update the container
			 	this->containersG_[i].updCurrent(val);
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure local
			for (int i = 0; i < this->opL_.size(); ++i)
			{
				auto& _op 	= this->opL_[i];
				// go through the local operators
				for (auto j = 0; j < this->Ns_; ++j)
				{
					auto val = _op->operator()(s, _fun, j);
					// update the container
					this->containersL_[i].updCurrent(val, j);
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure correlation
			for (int k = 0; k < this->opC_.size(); ++k)
			{
				auto& _op = this->opC_[k];
				for (auto i = 0; i < this->Ns_; ++i)
				{
					for (auto j = 0; j < this->Ns_; ++j)
					{
						auto val = _op->operator()(s, _fun, i, j);
						// update the container
						this->containersC_[k].updCurrent(val, i, j);
					}
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////

	template<typename _T>
	inline void MeasurementNQS<_T>::measure(Operators::_OP_V_T_CR s, NQSFunCol _fun)
	{
		BEGIN_CATCH_HANDLER
		{
			// measure global
			for (int i = 0; i < this->opG_.size(); ++i)
			{
				auto& _op 	= this->opG_[i];
				auto val 	= _op->operator()(s, _fun);
				// update the container
			 	this->containersG_[i].updCurrent(val);
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure local
			for (int i = 0; i < this->opL_.size(); ++i)
			{
				auto& _op 	= this->opL_[i];
				// go through the local operators
				for (auto j = 0; j < this->Ns_; ++j)
				{
					auto val = _op->operator()(s, _fun, j);
					// update the container
					this->containersL_[i].updCurrent(val, j);
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure correlation
			for (int k = 0; k < this->opC_.size(); ++k)
			{
				auto& _op = this->opC_[k];
				for (auto i = 0; i < this->Ns_; ++i)
				{
					for (auto j = 0; j < this->Ns_; ++j)
					{
						auto val = _op->operator()(s, _fun, i, j);
						// update the container
						this->containersC_[k].updCurrent(val, i, j);
					}
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Measure the operators for the given state - uses the operator representation acting on 
	* the state in a full Hilbert space. Therefore, one needs to provide the Hilbert space and the state.
	* @param _state state to measure the operators for
	* @param _H Hilbert space to measure the operators in
	*/
	template<typename _T>
	inline void NQSAv::MeasurementNQS<_T>::measure(arma::Col<_T> _state, const Hilbert::HilbertSpace<_T>& _H)
	{
		BEGIN_CATCH_HANDLER
		{
			// measure global
			for (int i = 0; i < this->opG_.size(); ++i)
			{
				auto& _op 	= this->opG_[i];
				auto& _cont = this->containersG_[i];
				// set the many body matrix
				_cont.resetMB();
				_cont.setManyBodyMat(_H, _op.get());
				auto _val 	= Operators::applyOverlap(_state, _cont.mbmat());
				// update the container
				_cont.setManyBodyVal(_val);
				// reset the many body matrix
				_cont.resetMB();
			}	
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure local
			for (int i = 0; i < this->opL_.size(); ++i)
			{
				auto& _op 	= this->opL_[i];
				auto& _cont = this->containersL_[i];
				// reset
				_cont.resetMB();

				// go through the local operators
				for (auto i = 0; i < _op->getNs(); ++i)
				{
					// set the many body matrix
					_cont.setManyBodyMat(_H, _op.get(), (uint)i);
					auto _val = Operators::applyOverlap(_state, _cont.mbmat());
					// update the container
					_cont.setManyBodyVal(_val, (uint)i);					
				}
				// reset the many body matrix
				_cont.resetMB();
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of local operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure correlation
			for (int k = 0; k < this->opC_.size(); ++k)
			{
				auto& _op = this->opC_[k];
				auto& _cont = this->containersC_[k];
				// reset
				_cont.resetMB();

				for (auto i = 0; i < _op->getNs(); ++i)
				{
					for (auto j = 0; j < _op->getNs(); ++j)
					{
						// set the many body matrix
						_cont.setManyBodyMat(_H, _op.get(), (uint)i, (uint)j);
						auto _val = Operators::applyOverlap(_state, _cont.mbmat());
						// update the container
						_cont.setManyBodyVal(_val, (uint)i, (uint)j);
					}
				}
				// reset the many body matrix
				_cont.resetMB();
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Normalize the operators - divide by the number of samples
	* @param _nsamples number of samples
	*/
	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::normalize(uint _nsamples)
	{
		BEGIN_CATCH_HANDLER
		{
			// measure global
			for (auto& _cont : this->containersG_)
				_cont.normalize(_nsamples);
		}
		END_CATCH_HANDLER("Problem in the normalization of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure local
			for (auto& _cont : this->containersL_)
				_cont.normalize(_nsamples);
		}
		END_CATCH_HANDLER("Problem in the normalization of local operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure correlation
			for (auto& _cont : this->containersC_)
				_cont.normalize(_nsamples);
		}
		END_CATCH_HANDLER("Problem in the normalization of correlation operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////

	template<typename _T>
	inline void NQSAv::MeasurementNQS<_T>::save(const strVec& _ext)
	{
		BEGIN_CATCH_HANDLER
		{
			// save global
			for (int i = 0; i < this->opG_.size(); ++i)
			{
				auto& _cont = this->containersG_[i];
				auto& _op 	= this->opG_[i];
				// nqs
				{
					auto M = _cont.template mean<cpx>();
					// save!
					for (const auto& ext : _ext)
						saveAlgebraic(dir_, _op->getNameS() + ext, M, "values");
				}
				// many body 
				{
					const arma::Mat<_T>& M = _cont.mbval();
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, "mb_" + _op->getNameS() + ext, M, "values");
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// save local
			for (int i = 0; i < this->opL_.size(); ++i)
			{
				auto& _cont = this->containersL_[i];
				auto& _op 	= this->opL_[i];
				// nqs
				{
					auto M = _cont.template mean<cpx>();
					// save!
					for (const auto& ext : _ext)
						saveAlgebraic(dir_, _op->getNameS() + ext, M, "values");
				}
				// many body
				{
					const arma::Mat<_T>& M = _cont.mbval();
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, "mb_" + _op->getNameS() + ext, M, "values");
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of local operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// save correlation
			for (int i = 0; i < this->opC_.size(); ++i)
			{
				auto& _cont = this->containersC_[i];
				auto& _op 	= this->opC_[i];
				// nqs
				{
					auto M = _cont.template mean<cpx>();
					// save!
					for (const auto& ext : _ext)
						saveAlgebraic(dir_, _op->getNameS() + ext, M, "values");
				}
				// many body
				{
					const arma::Mat<_T>& M = _cont.mbval();
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, "mb_" + _op->getNameS() + ext, M, "values");
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////
};
#endif 
//////////////////////////////////////////////////////////////////////////////////////////
