#pragma once
/***************************************
* Defines a class for a general operator.
* It can be later expanded for some more
* complicated operators acting on Hiblert
* space.
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#ifndef LATTICE_H
#include "../../source/src/lattices.h"
#endif

#ifndef BINARY_H
#include "../../source/src/binary.h"
#endif

#ifndef OPALG_H
#include "operator_algebra.h"
#endif

// ################################################################ FORWARD DECLARATIONS ########################################################################

namespace Hilbert {
	template <typename _T, uint _spinModes = 2>
	class HilbertSpace;
};

// ################################################ D E F I N I T I O N S   O F   I M P L E M E N T A T I O N ###################################################

namespace Operators {

	// ##########################################################################################################################################

	/*
	* @brief Standarizes the operator so that it is traceless and has a unit norm 1. The norm is calculated as the Frobenius norm.
	* @param _mat the matrix to standarize 
	*/
	template<typename _MatT>
	inline void standarizeOperator(_MatT& _mat)
	{
		auto _nrows = _mat.n_rows;
		_mat.diag() -= arma::trace(_mat) / double(_nrows);
		auto _Hs	= arma::trace(arma::square(_mat)) / (double)_nrows;
		_mat		= _mat / std::sqrt(_Hs);	
	}

	// ################################################################# G E N E R A L ##########################################################################


	/*
	* @brief Implemented symmetry types
	*/
	enum SymGenerators { E, T, Tr, R, PX, PY, PZ, OTHER, SX, SY, SZ };

	BEGIN_ENUM(SymGenerators)
	{
			DECL_ENUM_ELEMENT(E),
			DECL_ENUM_ELEMENT(T),
			DECL_ENUM_ELEMENT(Tr),
			DECL_ENUM_ELEMENT(R),
			DECL_ENUM_ELEMENT(PX),
			DECL_ENUM_ELEMENT(PY),
			DECL_ENUM_ELEMENT(PZ),
			// other
			DECL_ENUM_ELEMENT(OTHER),
			DECL_ENUM_ELEMENT(SX),
			DECL_ENUM_ELEMENT(SY),
			DECL_ENUM_ELEMENT(SZ)

	}
	END_ENUM(SymGenerators);

	/*
	* @brief Implemented fermionic operators
	*/
	enum FermionicOperators { C_UP_DAG, C_UP, C_DOWN_DAG, C_DOWN };

	BEGIN_ENUM(FermionicOperators)
	{
		DECL_ENUM_ELEMENT(C_UP_DAG),
		DECL_ENUM_ELEMENT(C_UP),
		DECL_ENUM_ELEMENT(C_DOWN_DAG),
		DECL_ENUM_ELEMENT(C_DOWN),
	}
	END_ENUM(FermionicOperators);
};

// ################################################################### G E N E R A L ############################################################################

namespace Operators
{
	/*
	* @brief A class describing the local operator acting on specific states. It returns a value and changes a state.
	* Can take several template arguments
	*/
	template<typename _T, typename ..._Ts>
	class Operator 
	{
	protected:
		/*
		* @brief Function for creating the matrix (overriden). If this is set, the matrix will be created using this function. Otherwise it is created 
		* via standard procedure of acting on the states. This is useful for the operators that are not acting on the states directly or are random operators.
		*/
		GeneralizedMatrixFunction<_T> overridenMatFun_;										// function for creating the matrix (overriden)

		typedef typename _OP<_T>::template INP<_Ts...> repType;								// type returned for representing, what it does with state and value it returns
		size_t Ns_											=			1;					// number of elements in the vector (for one to know how to act on it)
		std::shared_ptr<Lattice> lat_;														// lattice type to be used later on, !! the lattice can be empty if not needed !!
		_T eigVal_											=			1.0;				// eigenvalue for symmetry generator (if there is an inner value)
		repType fun_										=			E;					// function allowing to use the symmetry operation
		
		// quadratic
		bool isQuadratic_									=			false;				// based on this, we will create the operator differently (we want act on a many body state per se)

		// used for checking on which states the operator acts when forgetting and using the matrix only
		u64 acton_											=			0;					// check on states the operator acts, this is stored as a number and the bitmask is applied! For many body
		SymGenerators name_									=			SymGenerators::E;   // name of the operator
		std::string nameS_									=			"E";				// name of the operator in string
		
	public:
		// ----------------------------------------------------------------------------------------------------

		virtual ~Operator()									=			default;			// default destructor

		// standard constructor
		Operator() 
		{ 
			init(); 
		};
		
		// with the usage of the elements in the state vector
		Operator(size_t Ns, const std::string& _nameS = "")
			: Ns_(Ns), nameS_(_nameS)
		{ 
			init(); 
		};

		// with the eigenvalue and the name of the operator
		Operator(size_t Ns, _T _eigVal, const std::string& _nameS = "") 
			: Ns_(Ns), eigVal_(_eigVal), nameS_(_nameS)
		{ 
			init(); 
		};

		// with the eigenvalue, the function and the name of the operator
		Operator(size_t Ns, _T _eigVal, repType _fun, SymGenerators _name = SymGenerators::E, const std::string& _nameS = "") 
			: Ns_(Ns), eigVal_(_eigVal), fun_(_fun), name_(_name), nameS_(_nameS)
		{ 
			init(); 
		};

		// for the usage with the lattice (mostly for spin models, spinless fermions and hardcore bosons)
		Operator(std::shared_ptr<Lattice> _lat, const std::string& _nameS = "")
			: Ns_(_lat->get_Ns()), lat_(_lat), nameS_(_nameS)
		{
			init();
		};

		// for the usage with the lattice and the eigenvalue
		Operator(std::shared_ptr<Lattice> _lat, _T _eigVal, const std::string& _nameS = "")
			: Ns_(_lat->get_Ns()), lat_(_lat), eigVal_(_eigVal), nameS_(_nameS)
		{
			init();
		};

		// for the usage with the lattice, the eigenvalue, the function and the name of the operator
		Operator(std::shared_ptr<Lattice> _lat, _T _eigVal, repType _fun, SymGenerators _name = SymGenerators::E, const std::string& _nameS = "")
			: Ns_(_lat->get_Ns()), lat_(_lat), eigVal_(_eigVal), fun_(_fun), name_(_name), nameS_(_nameS)
		{
			init();
		};

		// copy constructor
		Operator(const Operator<_T, _Ts...>& o)
			: overridenMatFun_(o.overridenMatFun_),
			Ns_(o.Ns_), 
			lat_(o.lat_), 
			eigVal_(o.eigVal_), 
			fun_(o.fun_),
			isQuadratic_(o.isQuadratic_), 
			acton_(o.acton_),
			name_(o.name_), 
			nameS_(o.nameS_)
		{
			init();
		};

		// move constructor
		Operator(Operator<_T, _Ts...>&& o)
			: overridenMatFun_(std::move(o.overridenMatFun_)), 
			Ns_(std::move(o.Ns_)), 
			lat_(std::move(o.lat_)),
			eigVal_(std::move(o.eigVal_)),
			fun_(std::move(o.fun_)), 
			isQuadratic_(std::move(o.isQuadratic_)), 
			acton_(std::move(o.acton_)), 
			name_(std::move(o.name_)), 
			nameS_(std::move(o.nameS_))
		{
			init();
		};

		// ----------------------------------------------------------------------------------------------------

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>& _other)
		{
			if (this != &_other)
			{
				this->Ns_		=		_other.Ns_;
				this->lat_		=		_other.lat_;
				this->fun_		=		_other.fun_;
				this->eigVal_	=		_other.eigVal_;
				this->name_		=		_other.name_;
			}
			return *this;
		}

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>&& _other)
		{
			if (this != &_other)
			{
				this->Ns_		=		std::move(_other.Ns_);
				this->lat_		=		std::move(_other.lat_);
				this->fun_		=		std::move(_other.fun_);
				this->eigVal_	=		std::move(_other.eigVal_);
				this->name_		=		std::move(_other.name_);
			}
			return *this;
		}

		// -------------- O P E R A T O R ( ) -------------
		
		virtual auto operator()(u64 s, _Ts... a)		const -> typename _OP<_T>::R	{ auto [s2, _val] = this->fun_(s, a...); return std::make_pair(s2, eigVal_ * _val); };
		virtual auto operator()(u64 s, _Ts... a)		-> typename _OP<_T>::R			{ auto [s2, _val] = this->fun_(s, a...); return std::make_pair(s2, eigVal_ * _val); };
		//virtual std::function<std::pair<u64, _T>(_Ts...)> operator*(u64 s)	const	{ return std::bind(this->fun_, s, std::placeholders::_1); };
		//virtual std::function<std::pair<u64, _T>(_Ts...)> operator*(u64 s)			{ return std::bind(this->fun_, s, std::placeholders::_1); };

		// ----------------------------------------------------------------------------------------------------

		// -------------------- STATIC --------------------
		static auto E(u64 s, _Ts...)					-> typename _OP<_T>::R			{ return std::make_pair(s, _T(1.0));	};

		// ----------------- V I R T U A L ----------------
		virtual void init() {};
		
		// -------------------- SETTERS -------------------
		
		auto setIsQuadratic(bool _is)					-> void							{ this->isQuadratic_ = _is;								};	
		auto setActOn(u64 _acton)						-> void							{ this->acton_ = _acton;								};
		// functions
		auto setFun(const repType& _fun)				-> void							{ this->fun_ = _fun;									};
		auto setFun(repType&& _fun)						-> void							{ this->fun_ = std::move(_fun);							};
		// names
		auto setName(SymGenerators _name)				-> void							{ this->name_ = _name;									};
		auto setNameS(const std::string& _name)			-> void							{ this->nameS_ = _name;									};
		auto setVal(_T _val)							-> void							{ this->eigVal_ = _val;									};
		auto setNs(size_t Ns)							-> void							{ this->Ns_ = Ns;										};
		// quadratic matrices and overriden functions creating the matrix
		auto setFun(GeneralizedMatrixFunction<_T>&& _fun)-> void						{ this->overridenMatFun_ = std::move(_fun);				};
		auto setFun(GeneralizedMatrixFunction<_T>& _fun) -> void						{ this->overridenMatFun_ = _fun;						};

		//void setQMatSparse(qMatType<arma::SpMat<_T>>&& _qMat)							{ this->qMatSparse_ = std::move(_qMat);				};
		//void setQMatDense(qMatType<arma::Mat<_T>>&& _qMat)								{ this->qMatDense_ = std::move(_qMat);				};
		
		// -------------------- GETTERS --------------------
		auto getIsQuadratic()							const -> bool					{ return this->isQuadratic_;							};
		auto getActOn()									const -> u64					{ return this->acton_;									};
		auto getNs()									const -> size_t					{ return this->Ns_;										};
		auto getVal()									const -> _T						{ return this->eigVal_;									};
		auto getFun()									const -> repType				{ return this->fun_;									};
		auto getName()									const -> SymGenerators			{ return this->name_;									};
		auto getNameG()									const -> std::string			{ return SSTR(getSTR_SymGenerators(this->name_));		};
		auto getNameS()									const -> std::string			{ return this->nameS_;									};

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   J O I N %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		/*
		* Joins the operators into one operator. This combines the operators acting on the same Hilbert space.
		*/
		template <typename T_ = _T, 
			typename std::enable_if<std::is_same<T_, cpx>::value>::type* = nullptr>
		Operator<T_, _Ts...> operator%(const Operator<double, _Ts...>& op) const
		{
			return Operators::Operator<cpx, _Ts...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_, Operators::SymGenerators::OTHER);
		}

		template <typename T_ = _T, 
			typename std::enable_if<!std::is_same<T_, cpx>::value>::type* = nullptr> 
		Operator<T_, _Ts...> operator%(const Operator<double, _Ts...>& op) const
		{
			return Operators::Operator<T_, _Ts...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_, Operators::SymGenerators::OTHER);
		}

		Operator<cpx, _Ts...> operator%(const Operator<cpx, _Ts...>& op) const
		{
			return Operators::Operator<cpx, _Ts...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_, Operators::SymGenerators::OTHER);
		}

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   C O N C A T %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		template <typename T_ = _T, typename ..._T2s,
			typename std::enable_if<std::is_same<T_, cpx>::value>::type* = nullptr>
		Operator<T_, _Ts..., _T2s...> operator*(const Operator<double, _T2s...>& op) const
		{
			return Operators::Operator<cpx, _Ts..., _T2s...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_, Operators::SymGenerators::OTHER);
		}

		template <typename T_ = _T, typename ..._T2s,
			typename std::enable_if<!std::is_same<T_, cpx>::value>::type* = nullptr>
		Operator<T_, _Ts..., _T2s...> operator*(const Operator<double, _T2s...>& op) const
		{
			return Operators::Operator<T_, _Ts..., _T2s...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_, Operators::SymGenerators::OTHER);
		}
		
		template <typename ..._T2s>
		Operator<cpx, _Ts..., _T2s...> operator%(const Operator<cpx, _T2s...>& op) const
		{
			return Operators::Operator<cpx, _Ts..., _T2s...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_, Operators::SymGenerators::OTHER);
		}

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   P O W E R %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		template <typename _T1, typename std::enable_if<std::is_integral<_T1>::value>::type* = nullptr>
		[[nodiscard]]
		Operator<_T, _Ts...> operator^(_T1 _n);
			
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F R I E N D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		/*
		* @brief representative eigenvalue calculator
		*/
		friend _T chi(const Operator<_T, _Ts...>& _op)									{ return _op.eigVal_;};

		/*
		* @brief calculate operator acting on state num eigenvalue
		*/
		friend _T chi(const Operator<_T, _Ts...>& _op, u64 _s, _Ts... _a)				{ auto [state, val] = _op(_s, std::forward<_Ts>(_a)...); return val * _op.eigVal_; };
	
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% H I L B E R T   S P A C E %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T,  template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, uint _spinModes = 2>
		_MatType<typename std::common_type<_TinMat, _T1>::type>
		generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, typename _T2, uint _spinModes = 2>
		_MatType<typename std::common_type<_TinMat, _T1, _T2>::type> 
		generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil1, const Hilbert::HilbertSpace<_T2, _spinModes>& _Hil2, _Ts... _arg);

		// calculates the matrix element of operator given a single state
		template <typename _T1, typename _T2>
		[[deprecated]]
		static _T avOp(const arma::Col<_T1>& _alfa, const arma::Col<_T2>& _beta, const Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace);
		template <typename _T1>
		[[deprecated]]
		static _T avOp(const arma::Col<_T1>& _alfa, const Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace);
	};

	// ##########################################################################################################################################
	
	// for the containers on the vectors
	//using OpVec_glb_t = v_1d<std::variant<Operators::Operator<cpx>, Operators::Operator<double>>>;
	//using OpVec_loc_t = v_1d<std::variant<Operators::Operator<cpx, uint>, Operators::Operator<double, uint>>>;
	//using OpVec_cor_t = v_1d<std::variant<Operators::Operator<cpx, uint, uint>, Operators::Operator<double, uint, uint>>>;

};

// ##########################################################################################################################################

/*
* @brief Raising the operator to the power of n (n is an integer) 
* @param _n the power to which the operator is raised
* @returns the operator raised to the power of n
*/
template<typename _T, typename ..._Ts>
template<typename _T1, typename std::enable_if<std::is_integral<_T1>::value>::type*>
inline Operators::Operator<_T,_Ts...> Operators::Operator<_T, _Ts...>::operator ^(_T1 _n)
{
	if (_n == 0)
		return Operator<_T, _Ts...>(this->lat_, 1.0, Operators::Operator<_T, _Ts...>::E);
	else if (_n == 1)
		return *this;

	auto _f = [_n, this](u64 _s, _Ts... _args) {
		_T val = 1.0;
		do {
			auto [newS, newV]	= this->operator()(_s, _args...);
			_s					= newS;
			val					*= newV;
			--_n;
			} while (_n);
		return std::make_tuple(_s, val);
	};
	return Operator<_T, _Ts...>(this->lat_, std::pow(this->eigVal_, _n), _f);
}

// ##########################################################################################################################################

//!TODO
template<typename _T, typename ..._Ts>
template<typename _T1, typename _T2>
inline _T Operators::Operator<_T, _Ts...>::avOp(const arma::Col<_T1>& _alfa, const arma::Col<_T2>& _beta, const Operators::Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace)
{
	return _T();
}

//!TODO
template<typename _T, typename ..._Ts>
template<typename _T1>
inline _T Operators::Operator<_T, _Ts...>::avOp(const arma::Col<_T1>& _alfa, const Operators::Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace)
{
	return _T();
}

// #################################################### M A T R I X   G E N E R A T I O N ####################################################

/*
* @brief Creates a most basic operator matrix knowing only the dimension of the Hilbert space. 
* The operator is acting on the same Hilbert space as the one it is acting on.
* For the total Hilbert space known to be without symmetries - not looking for representatives
* @brief _dim A dimension of the Hilbert space
* @returns A matrix representing the operator
*/
template<typename _T, typename ..._Ts> 
template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _InT>
typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
	Operators::Operator<_T, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
{
	_MatType<_TinMat> op(_dim, _dim);

	// check whether the operator is quadratic
	if (this->overridenMatFun_ && this->isQuadratic_)
	{
#ifdef _DEBUG
		LOGINFO("Operator is quadratic, going into!", LOG_TYPES::INFO, 1);
#endif
		if constexpr (std::is_same_v<_MatType<_TinMat>, arma::Mat<_TinMat>>)
			op = this->overridenMatFun_(_dim).getDense();
		else if constexpr (std::is_same_v<_MatType<_TinMat>, arma::SpMat<_TinMat>>)
			op = this->overridenMatFun_(_dim).getSparse();
		else
		{
			LOGINFO("Type is neither arma::Mat nor arma::SpMat", LOG_TYPES::ERROR, 0);
			throw std::logic_error("Unsupported matrix type in operator class");
		}
	}
	else
	{
		// otherwise create the operator matrix
	#pragma omp parallel for
		for (u64 _base = 0; _base < _dim; ++_base) 
		{
			auto [_idx, _val]	=	this->operator()(_base, _arg...);
			op(_idx, _base)		+=	_val;
		}

	}

	// standarize the operator
	if(_standarize)
		Operators::standarizeOperator(op);
	return op;
}

// ##########################################################################################################################################

/*
* @brief Creates a most basic operator matrix knowing only the dimension of the Hilbert space. 
* The operator is acting on the same Hilbert space as the one it is acting on.
* For the total Hilbert space known to be without symmetries - not looking for representatives
* @brief _dim A dimension of the Hilbert space
* @returns A matrix representing the operator
*/
template<typename _T, typename ..._Ts> 
template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _InT>
typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
Operators::Operator<_T, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
{
	GeneralizedMatrix<_TinMat> op(_dim);

	// check whether the operator is quadratic
	if (this->overridenMatFun_ && this->isQuadratic_)
	{
#ifdef _DEBUG
		LOGINFO("Operator is quadratic, going into!", LOG_TYPES::INFO, 1);
#endif
		op = this->overridenMatFun_(_dim);
	}
	else
	{
		// otherwise create the operator matrix
#pragma omp parallel for
		for (u64 _base = 0; _base < _dim; ++_base) 
		{
			auto [_idx, _val]	=	this->operator()(_base, _arg...);
			op.add(_idx, _base, _val);
		}

	}

	// standarize the operator
	if (_standarize)
		op.standarize();
	return op;
}

// ##########################################################################################################################################

/*
* @brief Creates an operator matrix whenever the operator is not transforming the state from a different symmetry sector.
* Uses the Hilbert space that stores the state transformations from the representative base.
* @param _Hil the Hilbert space in which we operate
* @param _arg arguments for the operator
* @returns A matrix representing the operator
*/
template<typename _T, typename ..._Ts>
template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _T1, uint _spinModes>
inline _MatType<typename std::common_type<_TinMat, _T1>::type> Operators::Operator<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil, _Ts ..._arg) const
{
	using res_typ	=	typename std::common_type<_T1, _TinMat>::type;
	u64 Nh			=	_Hil.getHilbertSize();
	_MatType<res_typ> op(Nh, Nh);

	for (u64 _idx = 0; _idx < Nh; _idx++)
	{
		auto [_newState, _val]		=	this->operator()(_Hil.getMapping(_idx), _arg...);

		// why even bother?
		[[unlikely]] if (EQP(std::abs(_val), 0.0, 1e-14))
			continue;

		// looking for the representative
		auto [_newIdx, _eigval]		=	_Hil.findRep(_newState, _Hil.getNorm(_idx));

		// go to it manually
		[[likely]]
		if(_newIdx < Nh)
			op(_newIdx, _idx)		+=	_val * _eigval;
	}
	// standarize the operator
	//if(_standarize)
	//	Operators::standarizeOperator(op);
	return op;
}

// ##########################################################################################################################################

/*
* @brief Creates an operator matrix whenever the operator is transforming the state to a different symmetry sector 
* @param _Hil the Hilbert space in which we operate
* @warning (takes into account that we are going to a different symmetry sector so matrix is not square)
* @trace O = \sum _{i \in A} \sum _{j \in _B} |i>_A <j|_B O_{ij}
*/
template<typename _T, typename ..._Ts>
template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _T1, typename _T2, uint _spinModes>
inline _MatType<typename std::common_type<_TinMat, _T1, _T2>::type> Operators::Operator<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil1, const Hilbert::HilbertSpace<_T2, _spinModes>& _Hil2, _Ts ..._arg)
{
	// using res_typ		=	typename std::common_type<_T1, _T, _T2>::type;
	u64 NhA				=	_Hil1.getHilbertSize();
	u64 NhB				=	_Hil2.getHilbertSize();
	arma::SpMat<_TinMat> op(NhA, NhB);

	for (u64 _idxB = 0; _idxB < NhB; _idxB++) 
	{
		// act with an operator on beta sector (right)
		auto [_newStateB, _valB]				=	this->operator()(_Hil2.getMapping(_idxB), _arg...);

		// why even bother?
		[[unlikely]] if (EQP(std::abs(_valB), 0.0, 1e-14))
			continue;

		// find the corresponding index and value in the A sector (left)
		auto [newIdxA, symValA]					=	_Hil1.findRep(_newStateB, _Hil2.getNorm(_idxB));

		// check if the state is there
		if (newIdxA < NhA)
			op(newIdxA, _idxB)					+=	_valB * algebra::conjugate(symValA);
	}
	// standarize the operator
	if(_standarize)
		return standarizeOperator(op);
	return op;
}

// ##########################################################################################################################################

// ############################################### E X T E N S I V E   O P E R A T O R S ####################################################

// ##########################################################################################################################################

 namespace Operators
 {
 	template <typename _T, typename ..._Ts>
 	class OperatorExt : std::vector<Operator<_T, _Ts...>>
 	{
 		using base_t = Operator<_T, _Ts...>;
 		//typedef typename _OP<_T>::template INP_EXT<_Ts...> repType;	 // type returned for representing, what it does with state and values!

		SymGenerators name_									=			SymGenerators::E;   // name of the operator
		std::string nameS_									=			"Extensive";		// name of the operator in string

		// normalization constant
		_T norm_											=			1.0;				// normalization constant

		// ----------------- V I R T U A L ----------------
		virtual void init()									{};

	public:
		// ----------------------------------------------------------------------------------------------------

		virtual ~OperatorExt()								=			default;			// default destructor

		// standard constructor
		OperatorExt() 
		{ 
			this->init(); 
			this->resize(0);
		};

		// with the usage of the elements in the state vector
		OperatorExt(const std::vector<base_t>& _operators, const std::string& _nameS, _T _norm = 0.0)
			: nameS_(_nameS), norm_(_norm)
		{ 
			this->assign(_operators.begin(), _operators.end());
			init(); 
		};

		// Copy constructor
		OperatorExt(const OperatorExt& other)
			: std::vector<Operator<_T, _Ts...>>(other), name_(other.name_), nameS_(other.nameS_), norm_(other.norm_)
		{
			this->init();
		}

		// Move constructor
		OperatorExt(OperatorExt&& other) noexcept
			: std::vector<Operator<_T, _Ts...>>(std::move(other)), name_(std::move(other.name_)), nameS_(std::move(other.nameS_)), norm_(std::move(other.norm_))
		{
			this->init();
		}

		// Copy assignment operator
		OperatorExt& operator=(const OperatorExt& other)
		{
			if (this != &other) 
			{
				std::vector<Operator<_T, _Ts...>>::operator=(other);
				name_	= other.name_;
				nameS_	= other.nameS_;
				norm_	= other.norm_;
			}
			return *this;
		}

		// Move assignment operator
		OperatorExt& operator=(OperatorExt&& other) noexcept
		{
			if (this != &other) {
				std::vector<Operator<_T, _Ts...>>::operator=(std::move(other));
				name_	= std::move(other.name_);
				nameS_	= std::move(other.nameS_);
				norm_	= std::move(other.norm_);
			}
			return *this;
		}

		// ----------------------------------------------------------------------------------------------------

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>& _other)
		{
			this->Ns_		=		_other.Ns_;
			this->lat_		=		_other.lat_;
			this->fun_		=		_other.fun_;
			this->eigVal_	=		_other.eigVal_;
			this->name_		=		_other.name_;
			return *this;
		}

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>&& _other)
		{
			this->Ns_		=		std::move(_other.Ns_);
			this->lat_		=		std::move(_other.lat_);
			this->fun_		=		std::move(_other.fun_);
			this->eigVal_	=		std::move(_other.eigVal_);
			this->name_		=		std::move(_other.name_);
			return *this;
		}
 	
		// ----------------------------------------------------------------------------------------------------

		// -------------------- SETTERS -------------------

		auto setName(SymGenerators _name)					-> void							{ this->name_ = _name;											};
		auto setNameS(const std::string& _name)				-> void							{ this->nameS_ = _name;											};
		auto setNorm(_T _norm)								-> void							{ this->norm_ = _norm;											};
		auto setOperators(const std::vector<base_t>& _ops)	-> void							{ this->assign(_ops.begin(), _ops.end());						};
		auto setOperators(std::vector<base_t>&& _ops)		-> void							{ this->assign(std::move(_ops.begin()), std::move(_ops.end())); };
		auto setOperators(const OperatorExt& _ops)			-> void							{ this->assign(_ops.begin(), _ops.end());						};
		auto setOperators(OperatorExt&& _ops)				-> void							{ this->assign(std::move(_ops.begin()), std::move(_ops.end())); };
		auto setOperator(size_t _idx, const base_t& _op)	-> void							{ this->at(_idx) = _op;											};
		auto setOperator(size_t _idx, base_t&& _op)			-> void							{ this->at(_idx) = std::move(_op);								};

		// -------------------- ADDERS ---------------------

		auto addOperator(const base_t& _op)					-> void							{ this->push_back(_op);											};
		auto addOperator(base_t&& _op)						-> void							{ this->push_back(std::move(_op));								};

		// -------------------- GETTERS --------------------

		auto getName()										const -> SymGenerators			{ return this->name_;											};
		auto getNameS()										const -> std::string			{ return this->nameS_;											};
		auto getNameG()										const -> std::string			{ return SSTR(getSTR_SymGenerators(this->name_));				};
		auto getOperators()									const -> std::vector<base_t>	{ return *this;													};
		auto getOperator(size_t _idx)						const -> const base_t&			{ return this->at(_idx);										};
		auto getNorm()										const -> _T						{ return this->norm_;											};
		
		// ----------------------------------------------------------------------------------------------------

	public:

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T,  template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, uint _spinModes = 2>
		_MatType<typename std::common_type<_TinMat, _T1>::type> 
			generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, typename _T2, uint _spinModes = 2>
		_MatType<typename std::common_type<_TinMat, _T1, _T2>::type> 
			generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil1, const Hilbert::HilbertSpace<_T2, _spinModes>& _Hil2, _Ts... _arg);
	};

	// ##########################################################################################################################################

	// ##########################################################################################################################################

	// ##########################################################################################################################################
 
	/*
	* @brief Creates a most basic operator matrix knowing only the dimension of the Hilbert space. 
	* The operator is acting on the same Hilbert space as the one it is acting on.
	* For the total Hilbert space known to be without symmetries - not looking for representatives
	* @brief _dim A dimension of the Hilbert space
	* @returns A matrix representing the operator
	*/
	template<typename _T, typename ..._Ts> 
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _InT>
	typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
		Operators::OperatorExt<_T, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
	{

		_MatType<_TinMat> op(_dim, _dim);
		for (auto& _op : *this)
		{
			_MatType<_TinMat> _opMat = _op.template generateMat<_TinMat, _MatType, _InT, false>(_dim, _arg...);
			op += _opMat;
		}

		// standarize the operator
		if(_standarize)
			return standarizeOperator(op);
		return op;
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates a most basic operator matrix knowing only the dimension of the Hilbert space. 
	* The operator is acting on the same Hilbert space as the one it is acting on.
	* For the total Hilbert space known to be without symmetries - not looking for representatives
	* @brief _dim A dimension of the Hilbert space
	* @returns A matrix representing the operator
	*/
	template<typename _T, typename ..._Ts> 
	template<bool _standarize, typename _TinMat,  template <class> class _MatType, typename _InT>
	typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
		Operators::OperatorExt<_T, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
	{

		GeneralizedMatrix<_TinMat> op(_dim);

		for (auto& _op : *this)
		{
			auto _opMat = _op.template generateMat<_TinMat, _InT, false>(_dim, _arg...);
			op += _opMat;
		}

		// standarize the operator
		op.standarize();

		return op;
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates an operator matrix whenever the operator is not transforming the state from a different symmetry sector.
	* Uses the Hilbert space that stores the state transformations from the representative base.
	* @param _Hil the Hilbert space in which we operate
	* @param _arg arguments for the operator
	* @returns A matrix representing the operator
	*/
	template<typename _T, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _T1, uint _spinModes>
	inline _MatType<typename std::common_type<_TinMat, _T1>::type> Operators::OperatorExt<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil, _Ts ..._arg) const
	{
		using res_typ	=	typename std::common_type<_T1, _TinMat>::type;
		u64 Nh			=	_Hil.getHilbertSize();
		_MatType<res_typ> op(Nh, Nh);

		for(auto& _op : *this)
		{
			_MatType<res_typ> _opMat = _op.template generateMat<_TinMat, _MatType, _T1, _spinModes, false>(_Hil, _arg...);
			op += _opMat;
		}

		// standarize the operator
		if(_standarize)
			return standarizeOperator(op);
		return op;
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates an operator matrix whenever the operator is transforming the state to a different symmetry sector 
	* @param _Hil the Hilbert space in which we operate
	* @warning (takes into account that we are going to a different symmetry sector so matrix is not square)
	* @trace O = \sum _{i \in A} \sum _{j \in _B} |i>_A <j|_B O_{ij}
	*/
	template<typename _T, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _T1, typename _T2, uint _spinModes>
	inline _MatType<typename std::common_type<_TinMat, _T1, _T2>::type> Operators::OperatorExt<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil1, const Hilbert::HilbertSpace<_T2, _spinModes>& _Hil2, _Ts ..._arg)
	{
		// using res_typ		=	typename std::common_type<_T1, _T, _T2>::type;
		u64 NhA				=	_Hil1.getHilbertSize();
		u64 NhB				=	_Hil2.getHilbertSize();
		arma::SpMat<_TinMat> op(NhA, NhB);

		for(auto& _op : *this)
		{
			_MatType<_TinMat> _opMat = _op.template generateMat<_TinMat, _MatType, _T1, _T2, _spinModes, false>(_Hil1, _Hil2, _arg...);
			op += _opMat;
		}

		// standarize the operator
		if(_standarize)
			return standarizeOperator(op);
		return op;
	}

	// ##########################################################################################################################################

}