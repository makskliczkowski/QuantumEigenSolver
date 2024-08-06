#pragma once
/***************************************
* Defines the generic NQS Solver class.
* Allows for later inhertiance
* for a fine model specialization.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

// #################################
#ifndef HAMIL_H					// #
#	include "../hamil.h"		// #
#	include "../algebra/general_operator.h"
#endif // !HAMIL_H				// #
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
	// ##########################################################################################################################################
	// ######################################################## N Q S   O P E R A T O R #########################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	/*
	* @brief Class for storing the operators that act on the quantum state taking into account the probability 
	* ratio for the averages
	* @todo Implement for more than spin operators
	*/
	template <typename _T, typename ..._Ts>
	class OperatorNQS
	{
	protected:
		using NQSS			= arma::Col<double>;
		// for initializing the pRatio function with a single column vector
		using NQSFunCol		= std::function<cpx(const NQSS& _v)>;
		// for initializing the pRatio function with initializer listr
		using NQSFun		= std::function<cpx(std::initializer_list<int>, std::initializer_list<double>)>;
		// decide about the size of values
		void decideSize();
	public:
		// ##### OPERATOR NAME #####
		std::string name_					= "";
		uint sizeX_							= 1;
		uint sizeY_							= 1;
		uint Ns_							= 1;

		// ##### CURRENT VALUE #####
		arma::Mat<_T> currentValue_;
		// save samples here
		v_1d<arma::Mat<_T>> samples_;

		// ####### MANY BODY #######
		// store the matrix for the many body average basded on a given quantum state
		arma::Mat<_T> manyBodyMatrix_;
		arma::Mat<_T> manyBodyVal_;

		// ######## INDICES ########
		// for finding out the index in the variadic variable
		v_1d<uint> indices_;
		uint currentIdx_					= 0;

		// ######## HELPER #########
		// store the column state
		NQSS state_;

		// operators to apply step by step that add up in the average \sum _ s' <s|O|s'> * \Psi(s') / \Psi(s)
		// we apply the operator to the left and look for all the states that can be reached from the base state
		// the s' are every possible state that can be reached from the base state with the operator action on s
		// O may in general create a combination of the base states, leave it as a vector then
		v_1d<Operators::Operator<_T, _Ts...>> op_;
		
		// ####### CONSTRUCT #######
		~OperatorNQS();

		// standard constructor
		OperatorNQS(const Operators::Operator<_T, _Ts...>& _op, const std::string& _name = "");
		OperatorNQS(const v_1d<Operators::Operator<_T, _Ts...>>& _opV, const std::string& _name = "");

		// copy and move constructors
		OperatorNQS(const Operators::OperatorNQS<_T, _Ts...>& _other);
		OperatorNQS(Operators::OperatorNQS<_T, _Ts...>&& _other);

		// ###### APPLICATION ######
		auto operator()(u64 s, _Ts... a)						const -> v_1d<typename _OP<_T>::R>;
		auto operator()(u64 s, NQSFunCol _fun, _Ts... a)		-> _T;
		auto operator[](uint i)									const -> arma::Mat<_T> { return this->samples_[i];		};

		// updates current value
		template <class _Tt = uint>
		typename std::enable_if<std::is_arithmetic<_Tt>::value, void>::type
		updCurrent(_T _val, _Tt i);
		template <class _Tt, typename ..._Tss>
		typename std::enable_if<std::is_arithmetic<_Tt>::value, void>::type
		updCurrent(_T _val, _Tt i, _Tss...a)
		{
			indices_[currentIdx_] = i;
			currentIdx_++;
			updCurrent(_val, a...);
		};
		template <typename ..._Tss>
		void updCurrent(_T _val, _Tss...a)												{ this->updCurrent(_val, a...);			};
		
	public:
		// ######## SETTERS ########
		auto resetSamples()				-> void { this->samples_ = {};															};
		auto resetMB()					-> void { manyBodyVal_ = arma::Mat<_T>(sizeX_, sizeY_, arma::fill::zeros);				};
		auto resetValue()				-> void { currentValue_ = arma::Mat<_T>(sizeX_, sizeY_, arma::fill::zeros);				};
		auto reset()					-> void;
		auto normalize(uint N)			-> void; 
		template <typename _T2>
		auto setManyBodyMat(const Hilbert::HilbertSpace<_T2>&, _Ts... a)				-> void;
		auto applyManyBody(const arma::Col<_T>& _C, uint i = 0, uint j = 0)				-> arma::Col<_T>;

		// ######## GETTERS ########
		auto mbmat()					const -> arma::Mat<_T>							{ return this->manyBodyMatrix_;			};
		auto mbval()					const -> arma::Mat<_T>							{ return this->manyBodyVal_;			};
		auto name()						const -> std::string							{ return this->name_;					};
		auto var()						const -> arma::Mat<cpx>							{ return algebra::cast<cpx>(Vectors::var(samples_));		};
		auto mean()						const -> arma::Mat<cpx>							{ return algebra::cast<cpx>(Vectors::mean(samples_));		};
		auto value()					const -> arma::Mat<cpx>							{ return currentValue_;					};
		auto value(uint i)				const -> arma::Mat<cpx>							{ return samples_[i];					};
		auto samples()					const -> v_1d<arma::Mat<cpx>>					{ return this->samples_;				};
		auto getOperator(uint i)		const -> Operators::Operator<_T, _Ts...>		{ return this->op_[i];					};
	};


	// ##########################################################################################################################################
	
	/*
	* @brief Resets the samples and the current value
	*/
	template <typename _T, typename ..._Ts>
	void OperatorNQS<_T, _Ts...>::reset()
	{ 
		this->resetValue(); 
		this->resetSamples(); 
	};

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ###################################################### C L A S S   C O N S T U C T #######################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################
	
	/*
	* @brief Resize the current value so one can store only the necessary values in the matrix (sizeX, sizeY)
	* Global		has sizeX = sizeY = 1,
	* Local			has sizeX = Ns, sizeY = 1,
	* Correlation	has sizeX = Ns, sizeY = Ns.
	* @template _T type of the operator
	* @template _Ts types of the additional parameters
	* @throws runtime_error if the number of arguments is not 0, 1 or 2
	*/
	template<typename _T, typename ..._Ts>
	inline void Operators::OperatorNQS<_T, _Ts...>::decideSize()
	{
		// get the size of template operators to decide on the opeartor type
		constexpr size_t numArgs		= sizeof...(_Ts);
		if (numArgs > 2) 
			throw std::runtime_error("Not implemented for more than two arguments!");

		if (numArgs == 0)
		{
			this->sizeX_				= 1;
			this->sizeY_				= 1;
			this->indices_				= {};
		}
		else if (numArgs == 1)
		{
			this->sizeX_				= this->Ns_;
			this->sizeY_				= 1;
			this->indices_				= { 0 };
		}
		else if (numArgs == 2)
		{
			this->sizeX_				= this->Ns_;
			this->sizeY_				= this->Ns_;
			this->indices_				= { 0, 0 };
		}
		else
			throw std::runtime_error("Not implemented for more than two arguments!");

		// store the matrix for the many body average basded on a given quantum state
		this->manyBodyVal_	=  arma::Mat<_T>(sizeX_, sizeY_, arma::fill::zeros);
	};

	// ##########################################################################################################################################

	template <typename _T, typename ..._Ts>
	OperatorNQS<_T, _Ts...>::OperatorNQS(const Operators::Operator<_T, _Ts...>& _op, const std::string& _name)
		: name_(_name), Ns_(_op.getNs()), samples_({}), op_({ _op })
	{
		this->decideSize();
		// create the state (basis state)
		this->state_.resize(Ns_);
		this->reset();
	};

	template <typename _T, typename ..._Ts>
	OperatorNQS<_T, _Ts...>::OperatorNQS(const v_1d<Operators::Operator<_T, _Ts...>>& _opV, const std::string& _name)
		: name_(_name), Ns_(_opV[0].getNs()), samples_({}), op_(_opV)
	{
		this->decideSize();
		// create the state (basis state)
		this->state_.resize(Ns_);
		this->reset();
	};

	template<typename _T, typename ..._Ts>
	inline Operators::OperatorNQS<_T, _Ts...>::OperatorNQS(const Operators::OperatorNQS<_T, _Ts...>& _other)
		: name_(_other.name_), Ns_(_other.Ns_), currentValue_(_other.currentValue_), samples_(_other.samples_), op_(_other.op_)
	{
		this->decideSize();
		// create the state (basis state)
		this->state_.resize(Ns_);
		this->reset();
	}

	template<typename _T, typename ..._Ts>
	inline Operators::OperatorNQS<_T, _Ts...>::OperatorNQS(Operators::OperatorNQS<_T, _Ts...>&& _other)
		: name_(std::move(_other.name_)), Ns_(std::move(_other.Ns_)), currentValue_(std::move(_other.currentValue_)), samples_(std::move(_other.samples_)), op_(std::move(_other.op_))
	{
		this->decideSize();
		// create the state (basis state)
		this->state_ = std::move(_other.state_);
		this->reset();
	}

	template<typename _T, typename ..._Ts>
	inline Operators::OperatorNQS<_T, _Ts...>::~OperatorNQS()
	{
		for (auto& x : this->samples_) x.reset();
		this->samples_.clear();
		op_.clear();
	}

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ##################################################### C L A S S   O P E R A T O R S ######################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################
	
	template<typename _T, typename ..._Ts>
	template<class _Tt>
	inline typename std::enable_if<std::is_arithmetic<_Tt>::value, void>::type
	Operators::OperatorNQS<_T, _Ts...>::updCurrent(_T _val, _Tt i)
	{
		this->indices_[this->currentIdx_] = i;
		this->currentIdx_++;

		// check the size of the indices
		if (indices_.size() == 0)
			this->currentValue_(0, 0) += _val;
		else if (indices_.size() == 1)
			this->currentValue_(this->indices_[0], 0) += _val;
		else if (indices_.size() == 2)
			this->currentValue_(this->indices_[0], this->indices_[1]) += _val;
		else
			throw std::runtime_error("Not implemented such exotic operators");
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Sets the many body matrix to apply it later for the many body states
	* @param _H Hilbert space
	* @param ...a additional parameters to the operators
	*/
	template<typename _T, typename ..._Ts>
	template <typename _T2>
	inline void Operators::OperatorNQS<_T, _Ts...>::setManyBodyMat(const Hilbert::HilbertSpace<_T2>& _H, _Ts ...a)
	{
		// get the common type
		using res_typ = typename std::common_type<_T, _T2>::type;
		// store all the measured values
		this->manyBodyMatrix_ = arma::Mat<_T>(_H.getHilbertSize(), _H.getHilbertSize(), arma::fill::zeros);

		// go through all operators, check the corresponding Hilbert sizes!
		for (const Operators::Operator<_T, _Ts...>& _op : this->op_)
		{
			// if we don't need to apply the symmetries
			if (_H.getHilbertSize() == _H.getFullHilbertSize())
			{
				auto _Min = _op.template generateMat<false, res_typ, typename arma::Mat>(_H.getFullHilbertSize(), a...);
				this->manyBodyMatrix_ += algebra::cast<_T>(_Min);
			}
			else
			{
				auto _Min = _op.template generateMat<false, res_typ, typename arma::Mat>(_H, a...);
				this->manyBodyMatrix_ += algebra::cast<_T>(_Min);
			}
		}
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Applies the many body matrix to a given state and saves the overlap <\Psi|O|\Psi>
	* to a specific place (i, j) in the matrix. The place corresponds to the locality of the operator.
	* @param _C many body state
	* @param i first matrix save element
	* @param j second matrix save element
	*/
	template<typename _T, typename ..._Ts>
	inline arma::Col<_T> Operators::OperatorNQS<_T, _Ts...>::applyManyBody(const arma::Col<_T>& _C, uint i, uint j)
	{
		auto _Cout = this->manyBodyMatrix_ * _C;
		manyBodyVal_(i, j) = arma::cdot(_C, _Cout);
		return _Cout;
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Apply the operators without a value change with pRatio.
	* @param s base state to apply the operators to
	* @param _fun pRatio function from the NQS
	* @param ...a additional parameters to the operators
	* @returns vector of changed base states with their corresponding values
	*/
	template<typename _T, typename ..._Ts>
	inline _T Operators::OperatorNQS<_T, _Ts...>::operator()(u64 s, NQSFunCol _fun, _Ts ...a)
	{
		// starting value
		this->currentIdx_	= 0;
		_T _valTotal		= 0.0;
		// go through operators
		for (auto& _op : op_)
		{
			// take value and new vector (written as an integer) 
			auto [s2, _val] = _op(s, a...);
			// transform to state
			INT_TO_BASE(s2, this->state_, Operators::_SPIN_RBM);
			// calculate the probability ratio
			_valTotal += _val * algebra::cast<_T>(_fun(this->state_));
		}
		this->updCurrent(_valTotal, a...);
		return algebra::cast<_T>(_valTotal);
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Apply the operators without a value change with pRatio.
	* @param s base state to apply the operators to
	* @param ...a additional parameters to the operators
	* @returns vector of changed base states with their corresponding values
	*/
	template<typename _T, typename ..._Ts>
	inline v_1d<typename _OP<_T>::R> Operators::OperatorNQS<_T, _Ts...>::operator()(u64 s, _Ts ...a) const
	{
		v_1d<typename _OP<_T>::R> _out;
		// go through operators
		for (auto& _op : op_)
			// take value and new vector (written as an integer) 
			_out.push_back(_op(s, a...));
		return _out;
	}

	////////////////////////////////////////////////////////////////////////////

	/*
	* @brief Normalize the values after given block sample. After that, it resets the value
	* @param N size of the block
	*/
	template<typename _T, typename ..._Ts>
	inline void Operators::OperatorNQS<_T, _Ts...>::normalize(uint N)
	{
		samples_.push_back(currentValue_ / (long double)(N)); 
		this->resetValue();
	}
};
#endif
//////////////////////////////////////////////////////////////////////////////////////////
// Average operators in the NQS
#	ifndef NQS_AV_H
#		define NQS_AV_H
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
		using NQSS			= arma::Col<double>;
		// for initializing the pRatio function with a single column vector
		using NQSFunCol		= std::function<cpx(const NQSS& _v)>;
		using OPG			= v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>>;
		using OPL			= v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>>;
		using OPC			= v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>>;
	protected:
		std::string dir_	= "";
		uint threads_		= 1;
		uint Ns_			= 0;
		// operator vector
		//v_1d<NQSAv::MeasurementNQSOperators> measOp_;
		// lattice pointer
		std::shared_ptr<Lattice> lat_;
		// global operators
		OPG opG_;
		// local operators
		OPL opL_;
		// correlation operators
		OPC opC_;
	public:
		~MeasurementNQS() 
		{ 
			LOGINFO("Destroying the measurement of the NQS averages.", LOG_TYPES::TRACE, 3);
			for (auto& x : opG_) x.reset(); 
			for (auto& x : opL_) x.reset();
			for (auto& x : opC_) x.reset();
		}
		MeasurementNQS(std::shared_ptr<Lattice> _lat, const strVec& _operators);
		MeasurementNQS(std::shared_ptr<Lattice> _lat, const std::string& _dir,
													  const OPG& _opG,
													  const OPL& _opL,
													  const OPC& _opC,
													  uint _threadNum = 1);

		void measure(u64 s, NQSFunCol _fun);
		void measure(arma::Col<_T> _state, const Hilbert::HilbertSpace<_T>&);
		void normalize(uint _nBlck);
		void save(const strVec& _ext = { ".h5" });
	};

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ###################################################### C L A S S   C O N S T U C T #######################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################
	
	template <typename _T>
	inline NQSAv::MeasurementNQS<_T>::MeasurementNQS(std::shared_ptr<Lattice> _lat, const strVec& _operators)
		: Ns_(_lat->get_Ns()), lat_(_lat)
	{}

	////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////
	template <typename _T>
	inline NQSAv::MeasurementNQS<_T>::MeasurementNQS(std::shared_ptr<Lattice> _lat,  const std::string& _dir,
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
		CONSTRUCTOR_CALL;
	}

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ####################################################### C L A S S   M E A S U R E ########################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################
	
	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::measure(u64 s, NQSFunCol _fun)
	{
		BEGIN_CATCH_HANDLER
		{
			// measure global
			for (auto& _op : this->opG_)
				auto val [[maybe_unused]] = _op->operator()(s, _fun);
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure local
			for (auto& _op : this->opL_)
			{
				// go through the local operators
				for (auto i = 0; i < this->Ns_; ++i)
				{
					auto val [[maybe_unused]] = _op->operator()(s, _fun, i);
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure correlation
			for (auto& _op : this->opC_)
			{
				for (auto i = 0; i < this->Ns_; ++i)
				{
					for (auto j = 0; j < this->Ns_; ++j)
					{
						auto val [[maybe_unused]] = _op->operator()(s, _fun, i, j);
					}
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////

	template<typename _T>
	inline void NQSAv::MeasurementNQS<_T>::measure(arma::Col<_T> _state, const Hilbert::HilbertSpace<_T>& _H)
	{
		BEGIN_CATCH_HANDLER
		{
			// measure global
			for (auto& _op : this->opG_)
			{
				_op->resetMB();
				_op->setManyBodyMat(_H);
				_op->applyManyBody(_state, 0, 0);
			}
				
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure local
			for (auto& _op : this->opL_)
			{
				_op->resetMB();
				// go through the local operators
				for (auto i = 0; i < this->Ns_; ++i)
				{
					_op->setManyBodyMat(_H, i);
					_op->applyManyBody(_state, i, 0);
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure correlation
			for (auto& _op : this->opC_)
			{
				_op->resetMB();
				for (auto i = 0; i < this->Ns_; ++i)
				{
					for (auto j = 0; j < this->Ns_; ++j)
					{
						_op->setManyBodyMat(_H, i, j);
						_op->applyManyBody(_state, i, j);
					}
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////

	template <typename _T>
	inline void NQSAv::MeasurementNQS<_T>::normalize(uint _nBlck)
	{
		BEGIN_CATCH_HANDLER
		{
			// measure global
			for (auto& _op : this->opG_)
				_op->normalize(_nBlck);
		}
		END_CATCH_HANDLER("Problem in the normalization of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure local
			for (auto& _op : this->opL_)
				_op->normalize(_nBlck);
		}
		END_CATCH_HANDLER("Problem in the normalization of local operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure correlation
			for (auto& _op : this->opC_)
				_op->normalize(_nBlck);
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
			for (auto& _op : this->opG_)
			{
				// nqs
				{
					arma::Mat<cpx> M = _op->mean();
					// save!
					for (const auto& ext : _ext)
						saveAlgebraic(dir_, _op->name() + ext, M, "values");
				}
				// many body
				{
					arma::Mat<_T> M = _op->mbval();
					if(M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, "mb_" + _op->name() + ext, M, "values");
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure local
			for (auto& _op : this->opL_)
			{
				{
					arma::Mat<cpx> M = _op->mean();
					// save!
					for (const auto& ext : _ext)
						saveAlgebraic(dir_, _op->name() + ext, M, "values");
				}
				// many body
				{
					arma::Mat<_T> M = _op->mbval();
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, "mb_" + _op->name() + ext, M, "values");
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of local operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure correlation
			for (auto& _op : this->opC_)
			{
				{
					arma::Mat<cpx> M = _op->mean();
					// save!
					for (const auto& ext : _ext)
						saveAlgebraic(dir_, _op->name() + ext, M, "values");
				}
				// many body
				{
					arma::Mat<_T> M = _op->mbval();
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, "mb_" + _op->name() + ext, M, "values");
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);
	}

	////////////////////////////////////////////////////////////////////////////
};
#	endif 
//////////////////////////////////////////////////////////////////////////////////////////
