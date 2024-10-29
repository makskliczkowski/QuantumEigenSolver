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
		auto operator()(u64 s, NQSFunCol _fun, _Ts... a)		const -> _T;
		// for the column vector type of the state
		auto operator()(_OP_V_T_CR s, NQSFunCol _fun, _Ts... a) const -> _T;

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
	inline _T Operators::OperatorNQS<_T, _Ts...>::operator()(u64 s, NQSFunCol _fun, _Ts ...a) const
	{
		// starting value
		// this->container_.currentIdx_	= 0;
		_T _valTotal = 0.0;

		// go through operator acting on the state
		for (auto& [s2, _val] : this->operator()(s, a...))
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
	inline _T Operators::OperatorNQS<_T, _Ts...>::operator()(_OP_V_T_CR s, NQSFunCol _fun, _Ts ...a) const
	{
		// starting value
		// this->container_.currentIdx_	= 0;
		_T _valTotal = 0.0;

		// go through operator acting on the state
		for (auto& [s2, _val] : this->operator()(s, a...))
		{
			// calculate the probability ratio
			_T _functionVal = CAST<_T>(_fun(s2));
			_valTotal 		= _valTotal + _functionVal * _val;
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
	/*
	* @brief Class that stores the measurements from the NQS and is able to save them.
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
		MeasurementNQS(const MeasurementNQS& _m)
		{
			this->containersC_ = _m.containersC_;
			this->containersG_ = _m.containersG_;
			this->containersL_ = _m.containersL_;
			this->dir_ = _m.dir_;
			this->lat_ = _m.lat_;
			this->Ns_ = _m.Ns_;
			this->opC_ = _m.opC_;
			this->opG_ = _m.opG_;
			this->opL_ = _m.opL_;
			this->threads_ = _m.threads_;
		}
		MeasurementNQS(MeasurementNQS&& _m)
		{
			this->containersC_ = std::move(_m.containersC_);
			this->containersG_ = std::move(_m.containersG_);
			this->containersL_ = std::move(_m.containersL_);
			this->dir_ = std::move(_m.dir_);
			this->lat_ = std::move(_m.lat_);
			this->Ns_ = std::move(_m.Ns_);
			this->opC_ = std::move(_m.opC_);
			this->opG_ = std::move(_m.opG_);
			this->opL_ = std::move(_m.opL_);
			this->threads_ = std::move(_m.threads_);
		}

		// copy and move operators
		MeasurementNQS& operator=(const MeasurementNQS& _m)
		{
			this->containersC_ = _m.containersC_;
			this->containersG_ = _m.containersG_;
			this->containersL_ = _m.containersL_;
			this->dir_ = _m.dir_;
			this->lat_ = _m.lat_;
			this->Ns_ = _m.Ns_;
			this->opC_ = _m.opC_;
			this->opG_ = _m.opG_;
			this->opL_ = _m.opL_;
			this->threads_ = _m.threads_;
			return *this;
		}

		MeasurementNQS& operator=(MeasurementNQS&& _m)
		{
			this->containersC_ = std::move(_m.containersC_);
			this->containersG_ = std::move(_m.containersG_);
			this->containersL_ = std::move(_m.containersL_);
			this->dir_ = std::move(_m.dir_);
			this->lat_ = std::move(_m.lat_);
			this->Ns_ = std::move(_m.Ns_);
			this->opC_ = std::move(_m.opC_);
			this->opG_ = std::move(_m.opG_);
			this->opL_ = std::move(_m.opL_);
			this->threads_ = std::move(_m.threads_);
			return *this;
		}

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
		MeasurementNQS(size_t _Ns,	const OPG& _opG,
									const OPL& _opL = {},
									const OPC& _opC = {},
									uint _threadNum = 1);



		// ---- MEASUREMENT ----

		void measure(u64 s, NQSFunCol _fun);
		void measure(Operators::_OP_V_T_CR, NQSFunCol _fun);
		void measure(const arma::Col<_T>& state, const Hilbert::HilbertSpace<_T>&);
		void normalize(uint _nBlck);
		void save(const strVec& _ext = { ".h5" });

		// ---- MEASUREMENT ---- (STATIC)

		static std::pair<bool, _T> measure(Operators::_OP_V_T_CR _state, const Operators::OperatorNQS<_T>& _gO, 
							NQSFunCol _fun, Operators::Containers::OperatorContainer<_T>& _cont)
		{
			bool ok  = true;
			auto val = _gO(_state, _fun);

			// Check if val is valid (no NaN or Inf)
			// if constexpr (std::is_arithmetic_v<_T>) {  // For real numbers (float, double)
			// 	if (std::isfinite(val)) {
			// 		_cont.updCurrent(val);  // Update only if val is valid
			// 	} else {
			// 		_cont.updCurrent(0.0);  // Update with 0.0 if val is not valid
			// 		ok = false;
			// 	}
			// } else if constexpr (std::is_same_v<_T, std::complex<typename _T::value_type>>) {  // For complex numbers
			// 	if (std::isfinite(val.real()) && std::isfinite(val.imag())) {
			// 		_cont.updCurrent(val);  // Update only if both parts of val are finite
			// 	} else {
			// 		_cont.updCurrent(0.0);  // Update with 0.0 if val is not valid
			// 		ok = false;
			// 	}
			// }
			return std::make_pair(ok, val);
		};

		static void normalize(uint _nBlck, Operators::Containers::OperatorContainer<_T>& _cont)
		{
			_cont.normalize(_nBlck, true);
		};

		// ---- GETTERS ----
		auto getOpG()				const		->		const OPG& { return this->opG_; };
		auto getOpL()				const		->		const OPL& { return this->opL_; };
		auto getOpC()				const		->		const OPC& { return this->opC_; };
		auto getDir()				const		->		const std::string& { return this->dir_; };

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

		void resetContainers();
		void reset();
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
	inline  NQSAv::MeasurementNQS<_T>::MeasurementNQS(size_t _Ns, const strVec & _operators)
		: Ns_(_Ns), lat_(nullptr)
	{
		CONSTRUCTOR_CALL;
	}

	////////////////////////////////////////////////////////////////////////////

	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::resetContainers()
	{
		this->createContainers();
	}

	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::reset()
	{
		this->opG_.clear();
		this->opL_.clear();
		this->opC_.clear();
		this->resetContainers();
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
	inline void NQSAv::MeasurementNQS<_T>::measure(const arma::Col<_T>& _state, const Hilbert::HilbertSpace<_T>& _H)
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
				_cont.resetMBMat();
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
				for (auto j = 0; j < _op->getNs(); ++j)
				{
					// set the many body matrix
					_cont.setManyBodyMat(_H, _op.get(), (uint)j);
					auto _val = Operators::applyOverlap(_state, _cont.mbmat());
					// update the container
					_cont.setManyBodyVal(_val, (uint)j);					
				}
				// reset the many body matrix
				_cont.resetMBMat();
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
				_cont.resetMBMat();
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Normalize the operators - divide by the number of samples for each operator and 
	* for each container (block). The normalization is done for the global, local, and correlation operators.
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
				auto _name  = _op->getNameS();
				_name 		= _name.size() == 0 ? "OP" + std::to_string(i) : _name;
				// nqs
				{
					auto M = _cont.template mean<cpx>();
					// save!
					for (const auto& ext : _ext)
						saveAlgebraic(dir_, "NQS_OP" + ext, M, _name);
				}
				// many body 
				{
					const arma::Mat<_T>& M = _cont.mbval();
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, "ED_OP" + ext, M, _name);
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
				auto _name  = _op->getNameS();
				_name 		= _name.size() == 0 ? "OP" + std::to_string(i) : _name;
				// nqs
				{
					auto M = _cont.template mean<cpx>();
					// save!
					for (const auto& ext : _ext)
						saveAlgebraic(dir_, "NQS_OP_L" + ext, M, _name);
				}
				// many body
				{
					const arma::Mat<_T>& M = _cont.mbval();
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, "ED_OP_L" + ext, M, _name);
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
				auto _name  = _op->getNameS();
				_name 		= _name.size() == 0 ? "OP" + std::to_string(i) : _name;
				// nqs
				{
					auto M = _cont.template mean<cpx>();
					// save!
					for (const auto& ext : _ext)
						saveAlgebraic(dir_, "NQS_OP_C" + ext, M, _name);
				}
				// many body
				{
					const arma::Mat<_T>& M = _cont.mbval();
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, "ED_OP_C" + ext, M, _name);
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////
	
	/*
	* @brief Measure the given operator for the given basis state and the probability ratio function. 
	* The function is used to measure the operator for the given state and the probability ratio function.
	* @param _state basis state to measure the operator for
	* @param _gO operator to measure - global
	* @param _fun probability ratio function
	*/

};
#endif 
//////////////////////////////////////////////////////////////////////////////////////////
