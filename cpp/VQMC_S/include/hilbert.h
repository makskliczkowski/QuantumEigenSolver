#pragma once
#ifndef HILBERT_H
#define HILBERT_H


/*******************************
* Definitions for the operators 
* and Hiblert space that handles
* the symmetries etc.
*******************************/

#ifndef LATTICE_H
#include "../source/src/lattices.h"
#endif

#ifndef BINARY_H
#include "../source/src/binary.h"
#endif

#ifndef OPALG_H
#include "./algebra/operator_algebra.h"
#endif

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
		DECL_ENUM_ELEMENT(SZ),

	}
	END_ENUM(SymGenerators);

	/*
	* @brief A class describing the local operator acting on specific states. It returns a value and changes a state. 
	* Can take several arguments
	*/
	template<typename _T, typename ..._Ts>
	class Operator {
	public:
		typedef _OP<_T>::INP<_Ts...> repType;									// type returned for representing, what it does with state and value it returns
		std::shared_ptr<Lattice> lat_;											// lattice type to be used later on
		_T eigVal_										= 1.0;					// eigenvalue for symmetry generator (if there is inner value)
		repType fun_									= E;					// function allowing to use symmetry
		SymGenerators name_								= SymGenerators::E;
	public:
		Operator()										{ init(); };
		Operator(std::shared_ptr<Lattice>& _lat) 
			: lat_(_lat)								{ init(); };
		Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal) 
			: lat_(_lat), eigVal_(_eigVal)				{ init(); };
		Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal, repType _fun, SymGenerators _name = SymGenerators::E)
			: lat_(_lat), eigVal_(_eigVal), fun_(_fun), name_(_name)
														{ init(); };
		//Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal, repType&& _fun)
		//	: lat_(_lat), eigVal_(_eigVal),
		//	fun_(std::move(_fun))					{ init(); };
		Operator(const Operator<_T, _Ts...>& o)
			: eigVal_(o.eigVal_), fun_(o.fun_),
			lat_(o.lat_)								{ init(); };
		Operator(Operator<_T, _Ts...>&& o)
			: eigVal_(std::move(o.eigVal_)), fun_(std::move(o.fun_)), lat_(std::move(o.lat_)) 
														{ init(); };

		// ----------------------------------------------------------------------------------------------------

		Operator& operator=(const Operator& _other)
		{
			this->lat_				=				_other.lat_;
			this->fun_				=				_other.fun_;
			this->eigVal_			=				_other.eigVal_;
			return *this;
		}

		Operator& operator=(const Operator&& _other)
		{
			this->lat_				=				std::move(_other.lat_);
			this->fun_				=				std::move(_other.fun_);
			this->eigVal_			=				std::move(_other.eigVal_);
			return *this;
		}

		// ---------- override operators -----------
		virtual auto operator()(u64 s, _Ts... a)		const -> _OP<_T>::R			{ auto [s2, _val] = this->fun_(s); return std::make_pair(s2, eigVal_ * _val); };
		virtual auto operator()(u64 s, _Ts... a)		-> _OP<_T>::R				{ auto [s2, _val] = this->fun_(s); return std::make_pair(s2, eigVal_ * _val); };

		// ----------------------------------------------------------------------------------------------------

		// ---------- STATIC ----------
		static auto E(u64 s, _Ts...)					-> _OP<_T>::R				{ std::make_pair(s, _T(1.0)); };

		// ---------- virtual functions to override ----------
		virtual void init() {};

		// ---------- SETTERS -----------
		auto setFun(const repType& _fun)				-> void						{ this->fun_ = _fun; };
		auto setFun(repType&& _fun)						-> void						{ this->fun_ = std::move(_fun); };
		auto setName(SymGenerators _name)				-> void						{ this->name_ = _name; };

		// ---------- GETTERS -----------
		auto getVal()									const -> _T					{ return this->eigVal_; };
		auto getFun()									const -> repType			{ return this->fun_; };
		auto getName()									const -> SymGenerators		{ return this->name_; };

		// ----------------------------------------------------------------------------------------------------

		// ---------- OPERATORS JOIN ----------

		template <typename _T2, typename..._T2s>
		Operator<_T2, _Ts..., _T2s...> operator*(const Operator<_T2, _T2s...>& op) const
		{
			return Operator< _T2, _Ts..., _T2s... >(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_);
		}

		//template<typename _T1, typename _T2, typename ..._T1s, typename ..._T2s>
		//friend Operator<_T2, _T1s..., _T2s...> operator*(const Operator<_T1, _T2s...>& A, const Operator<_T2, _T2s...>& B) {
		//	Operator<_T2, _T1s..., _T2s...> op(A.lat_, A.eigVal_ * B.eigVal_, A.fun_ * B.fun_);
		//	return op;
		//};

		template <typename _T2, typename..._T2s>
		Operator<_T2, _Ts..., _T2s...> operator%(const Operator<_T2, _T2s...>& op) const
		{
			return Operator< _T2, _Ts..., _T2s... >(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_);
		}

		//template<typename _T1, typename _T2, typename ..._T1s, typename ..._T2s>
		//friend Operator<_T2, _T1s..., _T2s...> operator%(const Operator<_T1, _T2s...>& A, const Operator<_T2, _T2s...>& B) {
		//	Operator<_T2, _T1s..., _T2s...> op(A.lat_, A.eigVal_ * B.eigVal_, A.fun_ % B.fun_);
		//	return op;
		//};

		// ---------- OPERATORS CAST ----------
		template <class _TOut>
		operator Operator<_TOut>() {
			auto _fun = [&](u64 s) {
				const auto [s1, v1] = this->fun_(s);
				return std::make_pair(s1, _TOut(v1));
			};
			Operator<_TOut> op(this->lat_,
				static_cast<_TOut>(this->eigVal_),
				_fun);
			return op;
		}

		// ---------- FRIENDS ----------

		/*
		* @brief representative eigenvalue calculator
		*/
		friend _T chi(const Operator& _op) {
			return _op.eigVal_;
		}

		/*
		* @brief calculate operator acting on state num eigenvalue
		*/
		friend _T chi(const Operator& _op, u64 _s, _Ts... _a) {
			auto [state, val] = _op(_s, std::forward<_Ts>(_a)...);
			return val * _op.eigVal_;
		}
	};
};

// ##########################################################################################################################################

namespace Operators {

	// ##########################################################################################################################################
	
	template<typename _T>
	inline auto translation(std::shared_ptr<Lattice>& lat)
	{
		// cyclic shift function with boundary preservation
		_GLB<_T> cyclicShift;
		switch (lat->get_Dim())
		{
		case 1:
			cyclicShift = std::function(
				[lat](u64 state)
				{
					return std::make_pair(rotateLeft(state, lat->get_Ns()), _T(1.0));
				}
			);
			break;
		case 2:
			cyclicShift = std::function(
				[lat](u64 state)
				{
					auto tmpState = state;
					for (auto i = 0; i < lat->get_Lx(); i++)
						tmpState = rotateLeft(state, lat->get_Ns());
					return std::make_pair(tmpState, _T(1.0));
				}
			);
			break;
		case 3:
			cyclicShift = std::function(
				[lat](u64 state)
				{
					auto tmpState = state;
					for (auto i = 0; i < lat->get_Lx() * lat->get_Ly(); i++)
						tmpState = rotateLeft(state, lat->get_Ns());
					return std::make_pair(tmpState, _T(1.0));
				}
			);
			break;
		};
		return cyclicShift;
	}

	/*
	* @brief generates translation symmetry
	* @param kx K sector in X direction
	* @param ky K sector in Y direction
	* @param kz K sector in Z direction
	* @param dim dimension of translation
	* @param base base of the Hilbert space
	*/
	template <typename _T>
	inline Operator<_T> makeTranslation(std::shared_ptr<Lattice>& lat, int kx, int ky = 0, int kz = 0, int8_t dim = 1) {
		auto Kx = TWOPI * kx / double(lat->get_Lx());
		auto Ky = TWOPI * ky / double(lat->get_Ly());
		auto Kz = TWOPI * kz / double(lat->get_Lz());
		auto k = Kx;
		if (dim == 2) k = Ky;
		else if (dim == 3) k = Kz;
		// exponent used as generator
		_T val = std::exp(I * _T(k));
		// return operator		
		return Operator<_T>(lat, val, translation<_T>(lat), SymGenerators::T);
	}

	template <>
	inline Operator<double> makeTranslation(std::shared_ptr<Lattice>& lat, int kx, int ky, int kz, int8_t dim) {
		auto Kx = TWOPI * kx / double(lat->get_Lx());
		auto Ky = TWOPI * ky / double(lat->get_Ly());
		auto Kz = TWOPI * kz / double(lat->get_Lz());

		auto k = Kx;
		if (dim == 2) k = Ky;
		else if (dim == 3) k = Kz;
		// exponent used as generator
		double val = std::real(std::exp(I * double(k)));
		// return operator
		return Operator<double>(lat, val, translation<double>(lat), SymGenerators::T);
	}

	// ##########################################################################################################################################

	template<typename _T>
	inline auto reflection(std::shared_ptr<Lattice>& lat, int base)
	{
		_GLB<_T> fun = [lat, base](u64 state) {
			return std::make_pair(revBits(state, lat->get_Ns(), base), _T(1.0));
		};
		return fun;
	};

	/*
	* @brief Reflection symmetry
	*/
	template <typename _T>
	Operator<_T> makeReflection(std::shared_ptr<Lattice>& lat, int sec, int base = 2) {
		auto reflect = reflection<_T>(lat, base);
		return Operator<_T>(lat, _T(sec), reflect, SymGenerators::R);
	};

	// ##########################################################################################################################################

	template<typename _T>
	inline auto flipZ(std::shared_ptr<Lattice>& lat)
	{
		_GLB<_T> fun = [lat](u64 state) {
			int spinUps = lat->get_Ns() - std::popcount(state);
			return std::make_pair(state, _T(spinUps % 2 == 0 ? 1.0 : -1.0));
		};
		return fun;
	};

	template<typename _T>
	inline auto flipY(std::shared_ptr<Lattice>& lat)
	{
		_GLB<_T> fun = [lat](u64 state)
		{
			int spinUps = lat->get_Ns() - std::popcount(state);
			return std::make_pair(flipAll(state, lat->get_Ns()),
				(double(spinUps % 2 == 0 ? 1.0 : -1.0) * std::pow(I, lat->get_Ns())));
		};
		return fun;
	};

	template<typename _T>
	inline auto flipX(std::shared_ptr<Lattice>& lat)
	{
		_GLB<_T> fun = [lat](u64 state)
		{
			return std::make_pair(flipAll(state, lat->get_Ns()), _T(1.0));
		};
		return fun;
	};

	/*
	* @brief Parity with \sigma^x
	*/
	template <typename _T>
	inline Operator<_T> makeFlipX(std::shared_ptr<Lattice>& lat, int sec) {
		_GLB<_T> fX = flipX<_T>(lat);
		return Operator<_T>(lat, _T(sec), fX, SymGenerators::PX);
	};

	/*
	* @brief Parity with \sigma^y
	*/
	template <typename _T>
	inline Operator<_T> makeFlipY(std::shared_ptr<Lattice>& lat, int sec) {
		_GLB<_T> fY = flipY<_T>(lat);
		return Operator<_T>(lat, _T(sec), fY, SymGenerators::PY);
	};

	/*
	* @brief Parity with \sigma^z
	*/
	template <typename _T>
	inline Operator<_T> makeFlipZ(std::shared_ptr<Lattice>& lat, int sec) {
		_GLB<_T> fZ = flipZ<_T>(lat);
		return Operator<_T>(lat, _T(sec), fZ, SymGenerators::PZ);
	};

	// ############################

	template <typename _T>
	inline Operator<_T> symChoice(std::pair<SymGenerators, int> _g, std::shared_ptr<Lattice>& _lat) {
		auto [gen, eig] = _g;
		switch (gen) {
		case SymGenerators::T:
			return makeTranslation<_T>(_lat, eig, 0, 0, 1);
			break;
		case SymGenerators::R:
			return makeReflection<_T>(_lat, eig, 2);
			break;
		case SymGenerators::PX:
			return makeFlipX<_T>(_lat, eig);
			break;
		case SymGenerators::PY:
			return makeFlipY<_T>(_lat, eig);
			break;
		case SymGenerators::PZ:
			return makeFlipZ<_T>(_lat, eig);
			break;
		case SymGenerators::E:
			return Operator<_T>(_lat);
			break;
		default:
			return Operator<_T>(_lat);
			break;
		};
	};
}

#endif // !HILBERT_H