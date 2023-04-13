#ifndef LATTICE_H
#include "../../source/src/lattices.h"
#endif

#ifndef BINARY_H
#include "../../source/src/binary.h"
#endif

#ifndef OPALG_H
#include "operator_algebra.h"
#endif

// ############################################### FORWARD DECLARATIONS

namespace Hilbert {
	template <typename _T>
	class HilbertSpace;
};

// ##########################################################################################################################################

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
};

// ################################################################### G E N E R A L ###############################################################################

namespace Operators{

	/*
	* @brief A class describing the local operator acting on specific states. It returns a value and changes a state.
	* Can take several arguments
	*/
	template<typename _T, typename ..._Ts>
	class Operator {
		typedef _OP<_T>::INP<_Ts...> repType;									// type returned for representing, what it does with state and value it returns
		std::shared_ptr<Lattice> lat_;											// lattice type to be used later on
		_T eigVal_											=			1.0;	// eigenvalue for symmetry generator (if there is inner value)
		repType fun_										=			E;		// function allowing to use symmetry
		SymGenerators name_									=			SymGenerators::E;
	
	// ----------------------------------------------------------------------------------------------------
	
	public:
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
		//Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal, repType&& _fun)
		//	: lat_(_lat), eigVal_(_eigVal),
		//	fun_(std::move(_fun))					{ init(); };
		Operator(const Operator<_T, _Ts...>& o)
			: eigVal_(o.eigVal_), fun_(o.fun_), lat_(o.lat_)
		{
			init();
		};
		Operator(Operator<_T, _Ts...>&& o)
			: eigVal_(std::move(o.eigVal_)), fun_(std::move(o.fun_)), lat_(std::move(o.lat_))
		{
			init();
		};

		// ----------------------------------------------------------------------------------------------------

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>& _other)
		{
			this->lat_		=		_other.lat_;
			this->fun_		=		_other.fun_;
			this->eigVal_	=		_other.eigVal_;
			return *this;
		}

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>&& _other)
		{
			this->lat_		=		std::move(_other.lat_);
			this->fun_		=		std::move(_other.fun_);
			this->eigVal_	=		std::move(_other.eigVal_);
			return *this;
		}

		// ---------- override operators -----------
		
		virtual auto operator()(u64 s, _Ts... a)		const -> _OP<_T>::R			{ auto [s2, _val] = this->fun_(s, a...); return std::make_pair(s2, eigVal_ * _val); };
		virtual auto operator()(u64 s, _Ts... a)		-> _OP<_T>::R				{ auto [s2, _val] = this->fun_(s, a...); return std::make_pair(s2, eigVal_ * _val); };

		// ----------------------------------------------------------------------------------------------------

		// ---------- STATIC ----------
		static auto E(u64 s, _Ts...)					-> _OP<_T>::R				{ return std::make_pair(s, _T(1.0)); };

		// calculates the matrix element of operator
		template <typename _T1, typename _T2>
		static _T avOp(const arma::Col<_T1>& _alfa, const arma::Col<_T2>& _beta, const Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace);
		template <typename _T1>
		static _T avOp(const arma::Col<_T1>& _alfa, const Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace);

		// ---------- virtual functions to override ----------
		virtual void init() {};

		// ---------- SETTERS -----------
		auto setFun(const repType& _fun) -> void { this->fun_ = _fun; };
		auto setFun(repType&& _fun) -> void { this->fun_ = std::move(_fun); };
		auto setName(SymGenerators _name) -> void { this->name_ = _name; };

		// ---------- GETTERS -----------
		auto getVal()									const -> _T					{ return this->eigVal_; };
		auto getFun()									const -> repType			{ return this->fun_; };
		auto getName()									const -> SymGenerators		{ return this->name_; };

		// ----------------------------------------------------------------------------------------------------

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

		// ----------------------------------------------------------------------------------------------------

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

		// ----------------------------------------------------------------------------------------------------

		// ---------- OPERATORS CAST ----------

		template <typename T_ = _T,
			typename std::enable_if<std::is_same<T_, cpx>::value>::type* = nullptr>
		operator Operator<cpx, _Ts...>()
		{
			return *this;
		};

		template <typename T_ = _T,
			typename std::enable_if<!std::is_same<T_, cpx>::value>::type* = nullptr>
		operator Operator<cpx, _Ts...>()
		{
			auto _fun = [&](u64 s, _Ts... args) {
				const auto [s1, v1] = this->fun_(s, args...);
				return std::make_pair(s1, cpx(v1));
			};
			return Operator<cpx, _Ts...>(this->lat_, cpx(this->eigVal_), _fun);
		};

		
		// ---------- FRIENDS ----------

		/*
		* @brief representative eigenvalue calculator
		*/
		friend _T chi(const Operator<_T, _Ts...>& _op) {
			return _op.eigVal_;
		};

		/*
		* @brief calculate operator acting on state num eigenvalue
		*/
		friend _T chi(const Operator<_T, _Ts...>& _op, u64 _s, _Ts... _a) {
			auto [state, val] = _op(_s, std::forward<_Ts>(_a)...);
			return val * _op.eigVal_;
		};
	};
};

// ##########################################################################################################################################

template<typename _T, typename ..._Ts>
template<typename _T1, typename _T2>
inline _T Operators::Operator<_T, _Ts...>::avOp(const arma::Col<_T1>& _alfa, const arma::Col<_T2>& _beta, const Operators::Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace)
{
	return _T();
}

template<typename _T, typename ..._Ts>
template<typename _T1>
inline _T Operators::Operator<_T, _Ts...>::avOp(const arma::Col<_T1>& _alfa, const Operators::Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace)
{
	return _T();
}

// ##########################################################################################################################################
