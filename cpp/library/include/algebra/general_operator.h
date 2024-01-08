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
		typedef typename _OP<_T>::template INP<_Ts...> repType;					// type returned for representing, what it does with state and value it returns
		std::shared_ptr<Lattice> lat_;													// lattice type to be used later on
		_T eigVal_												=			1.0;				// eigenvalue for symmetry generator (if there is inner value)
		repType fun_											=			E;					// function allowing to use symmetry
		SymGenerators name_									=			SymGenerators::E;
	
	public:
		// ----------------------------------------------------------------------------------------------------

		virtual ~Operator()									=			default;
		Operator() 
		{ 
			init(); 
		};
		Operator(std::shared_ptr<Lattice> _lat)
			: lat_(_lat)
		{
			init();
		};
		Operator(std::shared_ptr<Lattice> _lat, _T _eigVal)
			: lat_(_lat), eigVal_(_eigVal)
		{
			init();
		};
		Operator(std::shared_ptr<Lattice> _lat, _T _eigVal, repType _fun, SymGenerators _name = SymGenerators::E)
			: lat_(_lat), eigVal_(_eigVal), fun_(_fun), name_(_name)
		{
			init();
		};
		Operator(const Operator<_T, _Ts...>& o)
			: lat_(o.lat_), eigVal_(o.eigVal_), fun_(o.fun_)
		{
			init();
		};
		Operator(Operator<_T, _Ts...>&& o)
			: lat_(std::move(o.lat_)), eigVal_(std::move(o.eigVal_)), fun_(std::move(o.fun_))
		{
			init();
		};

		// ----------------------------------------------------------------------------------------------------

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>& _other)
		{
			this->lat_		=		_other.lat_;
			this->fun_		=		_other.fun_;
			this->eigVal_	=		_other.eigVal_;
			this->name_		=		_other.name_;
			return *this;
		}

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>&& _other)
		{
			this->lat_		=		std::move(_other.lat_);
			this->fun_		=		std::move(_other.fun_);
			this->eigVal_	=		std::move(_other.eigVal_);
			this->name_		=		std::move(_other.name_);
			return *this;
		}

		// -------------- O P E R A T O R ( ) -------------
		
		virtual auto operator()(u64 s, _Ts... a)		const -> typename _OP<_T>::R	{ auto [s2, _val] = this->fun_(s, a...); return std::make_pair(s2, eigVal_ * _val); };
		virtual auto operator()(u64 s, _Ts... a)		-> typename _OP<_T>::R			{ auto [s2, _val] = this->fun_(s, a...); return std::make_pair(s2, eigVal_ * _val); };
		//virtual std::function<std::pair<u64, _T>(_Ts...)> operator*(u64 s)	const	{ return std::bind(this->fun_, s, std::placeholders::_1); };
		//virtual std::function<std::pair<u64, _T>(_Ts...)> operator*(u64 s)			{ return std::bind(this->fun_, s, std::placeholders::_1); };

		// ----------------------------------------------------------------------------------------------------

		// -------------------- STATIC --------------------
		static auto E(u64 s, _Ts...)						-> typename _OP<_T>::R			{ return std::make_pair(s, _T(1.0)); };

		// ----------------- V I R T U A L ----------------
		virtual void init() {};
		
		// -------------------- SETTERS -------------------
		auto setFun(const repType& _fun)					-> void								{ this->fun_ = _fun; };
		auto setFun(repType&& _fun)						-> void								{ this->fun_ = std::move(_fun); };
		auto setName(SymGenerators _name)				-> void								{ this->name_ = _name; };
		auto setVal(_T _val)									-> void								{ this->eigVal_ = _val; };

		// -------------------- GETTERS --------------------
		auto getVal()											const -> _T							{ return this->eigVal_; };
		auto getFun()											const -> repType					{ return this->fun_; };
		auto getName()											const -> SymGenerators			{ return this->name_; };
		auto getNs()											const -> uint						{ return this->lat_->get_Ns(); };

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   J O I N %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   C O N C A T %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   C A S T %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		//template <typename T_ = _T,
		//	typename std::enable_if<std::is_same<T_, cpx>::value>::type* = nullptr> 
		//[[maybe_unused]] operator Operator<cpx, _Ts...>()							{ return *this; };

		//template <typename T_ = _T,
		//	typename std::enable_if<!std::is_same<T_, cpx>::value>::type* = nullptr> 
		//[[maybe_unused]] operator Operator<cpx, _Ts...>()
		//{
		//	auto _fun = [&](u64 s, _Ts... args) {
		//		const auto [s1, v1] = this->fun_(s, args...);
		//		return std::make_pair(s1, cpx(v1));
		//	};
		//	return Operator<cpx, _Ts...>(this->lat_, cpx(this->eigVal_), _fun);
		//};

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   P O W E R %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		/*
		* @brief operator to the N'th power
		*/
		template <typename _T1, typename std::enable_if<std::is_integral<_T1>::value>::type* = nullptr>
		[[nodiscard]]
		Operator<_T, _Ts...> operator^(_T1 _n) 
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
			
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F R I E N D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		/*
		* @brief representative eigenvalue calculator
		*/
		friend _T chi(const Operator<_T, _Ts...>& _op)										{ return _op.eigVal_;};

		/*
		* @brief calculate operator acting on state num eigenvalue
		*/
		friend _T chi(const Operator<_T, _Ts...>& _op, u64 _s, _Ts... _a)				{ auto [state, val] = _op(_s, std::forward<_Ts>(_a)...); return val * _op.eigVal_; };
	
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% H I L B E R T   S P A C E %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		template<template <class _TM = _T> class _MatType, HasMatrixType _Concept = _MatType<_T>>
		_MatType<_T> generateMat(u64 _dim, _Ts... _arg) const;
		template<template <class _TM = _T> class _MatType, typename _T1, HasMatrixType _Concept = _MatType<_T>>
		_MatType<_T> generateMat(const Hilbert::HilbertSpace<_T1>& _Hil, _Ts... _arg) const;
		template<template <class _TM = _T> class _MatType, typename _T1, typename _T2, HasMatrixType _Concept = _MatType<_T>>
		_MatType<_T> generateMat(const Hilbert::HilbertSpace<_T1>& _Hil1, const Hilbert::HilbertSpace<_T2>& _Hil2, _Ts... _arg);


		// calculates the matrix element of operator given a single state
		template <typename _T1, typename _T2>
		[[deprecated]]
		static _T avOp(const arma::Col<_T1>& _alfa, const arma::Col<_T2>& _beta, const Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace);
		template <typename _T1>
		[[deprecated]]
		static _T avOp(const arma::Col<_T1>& _alfa, const Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace);

	};
};

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

// ################################################ M A T R I X   G E N E R A T I O N ##############################################################

/*
* @brief Creates a most basic operator matrix knowing only the dimension of the Hilbert space. 
* For the total Hilbert space known to be without symmetries - not looking for representatives
* @brief _dim A dimension of the Hilbert space
*/
template<typename _T, typename ..._Ts> 
template<template <class> class _MatType, HasMatrixType _Concept>
inline _MatType<_T> Operators::Operator<_T, _Ts...>::generateMat(u64 _dim, _Ts ..._arg) const
{
	_MatType<_T> op(_dim, _dim);
#pragma omp parallel for
	for (u64 _base = 0; _base < _dim; _base++) 
	{
		auto [_idx, _val]		=	this->operator()(_base, _arg...);
		op(_idx, _base)			+=	_val;
	}
	return op;
}

/*
* @brief Creates an operator matrix whenever the operator is not transforming the state to a different symmetry sector.
* Uses the Hilbert space that stores the state transformations from the representative base.
* @param _Hil the Hilbert space in which we operate
* @param _arg arguments for the operator
*/
template<typename _T, typename ..._Ts>
template<template <class> class _MatType, typename _T1, HasMatrixType _Concept>
inline _MatType<_T> Operators::Operator<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1>& _Hil, _Ts ..._arg) const
{
	u64 Nh							=	_Hil.getHilbertSize();
	_MatType<_T> op(Nh, Nh);
#pragma omp parallel for
	for (u64 _idx = 0; _idx < Nh; _idx++)
	{
		auto [_newState, _val]		=	this->operator()(_Hil.getMapping(_idx), _arg...);

		// why even bother?
		[[unlikely]] if (EQP(std::abs(_val), 0.0, 1e-14))
			continue;

		// looking for the representative
		auto [_newIdx, _eigval]		=	_Hil.findRep(_newState, _Hil.getNorm(_idx));
		if(_newIdx < Nh)
			op(_newIdx, _idx)		+=	_val * _eigval;
	}
	return op;
}

/*
* @brief Creates an operator matrix whenever the operator is transforming the state to a different symmetry sector 
* @param _Hil the Hilbert space in which we operate
* @warning (takes into account that we are going to a different symmetry sector so matrix is not square)
* @trace O = \sum _{i \in A} \sum _{j \in _B} |i>_A <j|_B O_{ij}
*/
template<typename _T, typename ..._Ts>
template<template <class> class _MatType, typename _T1, typename _T2, HasMatrixType _Concept>
inline _MatType<_T> Operators::Operator<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1>& _Hil1, const Hilbert::HilbertSpace<_T2>& _Hil2, _Ts ..._arg)
{
	u64 NhA										=	_Hil1.getHilbertSize();
	u64 NhB										=	_Hil2.getHilbertSize();
	arma::SpMat<_T> op(NhA, NhB);
#pragma omp parallel for
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
}