#pragma once

/*******************************
* Definitions for the operators 
* and Hiblert space that handles
* the symmetries etc.
*******************************/

#include "../source/src/common.h"
#include "../source/src/lattices.h"
#include "../source/src/binary.h"
#include "./algebra/operator_algebra.h"

// ##########################################################################################################################################

namespace Operators {

	/*
	* @brief A class describing the local operator acting on specific states. It returns a value and changes a state. 
	* Can take several arguments
	*/
	template<typename _T, typename ..._Ts>
	class Operator {
	public:
		typedef  _OP<_T>::INP<_Ts...> repType;									// type returned for representing, what it does with state and value it returns
		std::shared_ptr<Lattice> lat_;											// lattice type to be used later on
		_T eigVal_										= 1.0;					// eigenvalue for symmetry generator (if there is inner value)
		repType fun_									= E;					// function allowing to use symmetry

	public:
		Operator()										{ init(); };
		Operator(std::shared_ptr<Lattice>& _lat) 
			: lat_(_lat)								{ init(); };
		Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal) 
			: lat_(_lat), eigVal_(_eigVal)				{ init(); };
		Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal, repType _fun)
			: lat_(_lat), eigVal_(_eigVal), fun_(_fun)	{ init(); };
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
		virtual auto operator()(u64 s, _Ts... a)		const -> _OP<_T>::R			{ const auto [s, _val] = std::apply(this->fun_, s); return std::make_pair(s, eigVal_ * _val); };
		virtual auto operator()(u64 s, _Ts... a)		-> _OP<_T>::R				{ const auto [s, _val] = std::apply(this->fun_, s); return std::make_pair(s, eigVal_ * _val); };

		// ----------------------------------------------------------------------------------------------------

		// ---------- STATIC ----------
		static auto E(u64 s, _Ts...)					-> _OP<_T>::R				{ std::make_pair(s, _T(1.0)); };

		// ---------- virtual functions to override ----------
		virtual void init() {};

		// ---------- SETTERS -----------
		auto setFun(const repType& _fun)				-> void						{ this->fun_ = _fun; };
		auto setFun(repType&& _fun)						-> void						{ this->fun_ = std::move(_fun); };

		// ---------- GETTERS -----------
		auto getVal()									const -> _T					{ return this->eigVal_; };
		auto getFun()									const -> repType			{ return this->fun_; };

		// ----------------------------------------------------------------------------------------------------

		// ---------- OPERATORS JOIN ----------

		template <typename _T2, typename..._T2s>
		Operator<_T2, _Ts..., _T2s...> operator*(const Operator<_T2, _T2s...>& op) const
		{
			return Operator< _T2, _Ts..., _T2s... >(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_);
		}

		template<typename _T1, typename _T2, typename ..._T1s, typename ..._T2s>
		friend Operator<_T2, _T1s..., _T2s...> operator*(const Operator<_T1, _T2s...>& A, const Operator<_T2, _T2s...>& B) {
			Operator<_T2, _T1s..., _T2s...> op(A.lat_, A.eigVal_ * B.eigVal_, A.fun_ * B.fun_);
			return op;
		};

		template <typename _T2, typename..._T2s>
		Operator<_T2, _Ts..., _T2s...> operator%(const Operator<_T2, _T2s...>& op) const
		{
			return Operator< _T2, _Ts..., _T2s... >(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_);
		}

		template<typename _T1, typename _T2, typename ..._T1s, typename ..._T2s>
		friend Operator<_T2, _T1s..., _T2s...> operator%(const Operator<_T1, _T2s...>& A, const Operator<_T2, _T2s...>& B) {
			Operator<_T2, _T1s..., _T2s...> op(A.lat_, A.eigVal_ * B.eigVal_, A.fun_ % B.fun_);
			return op;
		};

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
	};
};

// ##########################################################################################################################################

namespace Operators {

	/*
	* @brief Implemented symmetry types
	*/
	enum SymGenerators { T, Tr, R, PX, PY, PZ };


	// ##########################################################################################################################################

	template<typename _T>
	inline auto translation(std::shared_ptr<Lattice> lat) -> _GLBR
	{
		// cyclic shift function with boundary preservation
		_GLBR cyclicShift;
		switch (lat->get_Dim())
		{
		case 1:
			cyclicShift = std::function(
				[lat](u64 state)
				{
					return std::make_pair(rotateLeft(state, lat->get_Ns()), _T(1.0));
				}
			);
			//val = std::exp(I * cpx(Kx));
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
			//val = ;
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
			//val = std::exp(I * cpx(Kz));
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
	Operator<_T> makeTranslation(std::shared_ptr<Lattice>& lat, int kx, int ky = 0, int kz = 0, int8_t dim = 1) {
		auto Kx = TWOPI * kx / double(lat->get_Lx());
		auto Ky = TWOPI * ky / double(lat->get_Ly());
		auto Kz = TWOPI * kz / double(lat->get_Lz());
		auto k = Kx;
		if (dim == 2) k = Ky;
		else if (dim == 3) k = Kz;
		// exponent used as generator
		_T val = std::exp(I * _T(k));
		// return operator		
		return Operator<_T>(lat, val, translation<_T>(lat));
	}

	template <>
	Operator<double> makeTranslation(std::shared_ptr<Lattice>& lat, int kx, int ky, int kz, int8_t dim) {
		auto Kx = TWOPI * kx / double(lat->get_Lx());
		auto Ky = TWOPI * ky / double(lat->get_Ly());
		auto Kz = TWOPI * kz / double(lat->get_Lz());

		auto k = Kx;
		if (dim == 2) k = Ky;
		else if (dim == 3) k = Kz;
		// exponent used as generator
		double val = std::real(std::exp(I * double(k)));
		// return operator
		return Operator<double>(lat, val, translation<double>(lat));
	}

	// ##########################################################################################################################################

	template<typename _T>
	inline auto reflection(std::shared_ptr<Lattice>& lat, int base) -> _GLBR
	{
		_GLBR fun = [lat, base](u64 state) {
			return std::make_pair(revBits(state, lat->get_Ns(), base), _T(1.0));
		};
		return fun;
	};

	/*
	* @brief Reflection symmetry
	*/
	template <typename _T>
	Operator<_T> makeReflection(std::shared_ptr<Lattice> lat, int sec, int base = 2) {
		auto reflect = reflection<_T>(lat, base);
		return Operator<_T>(lat, _T(sec), reflect);
	};

	// ##########################################################################################################################################

	template<typename _T>
	inline auto flipZ(std::shared_ptr<Lattice>& lat) -> _GLBR
	{
		_GLBR fun = [lat](u64 state) {
			int spinUps = lat->get_Ns() - __builtin_popcountll(state);
			return std::make_pair(state, _T(spinUps % 2 == 0 ? 1.0 : -1.0));
		};
		return fun;
	};

	template<typename _T>
	inline auto flipY(std::shared_ptr<Lattice>& lat) -> _GLBR
	{
		_GLBR fun = [lat](u64 state)
		{
			int spinUps = lat->get_Ns() - __builtin_popcountll(state);
			return std::make_pair(flipAll(state, lat->get_Ns()),
				_T((spinUps % 2 == 0 ? 1.0 : -1.0) * std::pow(I, lat->get_Ns())));
		};
		return fun;
	};

	template<typename _T>
	inline auto flipX(std::shared_ptr<Lattice>& lat) -> _GLBR
	{
		_GLBR fun = [lat](u64 state)
		{
			return std::make_pair(flipAll(state, lat->get_Ns()), _T(1.0));
		};
		return fun;
	};

	/*
	* @brief Parity with \sigma^x
	*/
	template <typename _T>
	Operator<_T> makeFlipX(std::shared_ptr<Lattice> lat, int sec) {
		_OP<_T>::GLB fX = flipX<_T>(lat);
		return Operator<_T>(lat, _T(sec), fX);
	};

	/*
	* @brief Parity with \sigma^y
	*/
	template <typename _T>
	Operator<_T> makeFlipY(std::shared_ptr<Lattice> lat, int sec) {
		_OP<_T>::GLB fY = flipY<_T>(lat);
		return Operator<_T>(lat, _T(sec), fY);
	};

	/*
	* @brief Parity with \sigma^z
	*/
	template <typename _T>
	Operator<_T> makeFlipZ(std::shared_ptr<Lattice> lat, int sec) {
		_OP<_T>::GLB fZ = flipZ<_T>(lat);
		return Operator<_T>(lat, _T(sec), fZ);
	};

}
