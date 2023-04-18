#pragma once
#ifndef OPALG_H
#define OPALG_H

#include <complex>
#include <functional>
using u64 = unsigned long long;
using cpx = std::complex<double>;

namespace Operators {
	template <typename _T>
	using _OPx = std::pair<u64, _T>;
	typedef _OPx<cpx> _OPCx;
	typedef _OPx<double> _OPRe;

	/*
	* @brief All possible correlators
	*/
	template <typename _RET>
	struct _OP {
		using RET		=		_RET;
		using R			=		std::pair<u64, _RET>;
		using GLB		=		std::function<R(u64)>;
		using LOC		=		std::function<R(u64, int)>;
		using COR		=		std::function<R(u64, int, int)>;

		/*
		* @brief contains all possible functions in a template
		*/
		template <typename... _T>
		using INP		=		std::function<R(u64, _T...)>;
	};
	using _GLBC			=		typename _OP<cpx>::GLB;			//<! global function acting on whole product state
	using _LOCC			=		typename _OP<cpx>::LOC;			//<! local function acting on single site
	using _CORC			=		typename _OP<cpx>::COR;			//<! correlation function acting on pair of sites	
#define _INPC _OP<cpx>::INP
	using _GLBR			=		typename _OP<double>::GLB;		//<! global function acting on whole product state
	using _LOCR			=		typename _OP<double>::LOC;		//<! local function acting on single site
	using _CORR			=		typename _OP<double>::COR;		//<! correlation function acting on pair of sites
#define _INPR _OP<double>::INP
	template<typename _T>
	using _GLB			=		typename _OP<_T>::GLB;				
	template<typename _T>
	using _LOC			=		typename _OP<_T>::LOC;
	template<typename _T>
	using _COR			=		typename _OP<_T>::COR;
	template<typename _T, typename ..._Ts>
	using _INP			=		typename _OP<_T>::INP<_Ts...>;

	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	//template <typename _T, typename _T2, typename ..._Ts, typename ..._T2s>
	//inline auto operator*(_INP<_T, _Ts...> f, _INP<_T2, _T2s...> g) {
	//	return [f, g](u64 s, _Ts... a1, _T2s... a2)
	//		{
	//			auto [s1, v1] = g(s, a2...);
	//			auto [s2, v2] = f(s1, a1...);
	//			return std::make_pair(s2, v1 * v2);
	//		};
	//};

	/*
	* @brief Imitates the behavior f \dot g \equiv f(g(n,...),...) 
	* Leaves more types to handle thereafter
	*/
	template <typename... _T1, typename... _T2>
	inline auto operator*(_INPC<_T1...> f, _INPC<_T2...> g)
	{
		return [f, g](u64 s, _T1... a1, _T2... a2) -> _OPCx
		{
			auto [s1, v1] = g(s, a2...);
			auto [s2, v2] = f(s1, a1...);
			return std::make_pair(s2, v1 * v2);
		};
	};

	template <typename... _T1, typename... _T2>
	inline auto operator*(_INPR<_T1...> f, _INPR<_T2...> g)
	{
		return [f, g](u64 s, _T1... a1, _T2... a2) -> _OPRe
		{
			auto [s1, v1] = g(s, a2...);
			auto [s2, v2] = f(s1, a1...);
			return std::make_pair(s2, v1 * v2);
		};
	};

	template <typename... _T1, typename... _T2>
	inline auto operator*(_INPR<_T1...> f, _INPC<_T2...> g)
	{
		return [f, g](u64 s, _T1... a1, _T2... a2) -> _OPCx
		{
			auto [s1, v1] = g(s, a2...);
			auto [s2, v2] = f(s1, a1...);
			return std::make_pair(s2, v1 * v2);
		};
	};

	template <typename... _T1, typename... _T2>
	inline auto operator*(_INPC<_T1...> f, _INPR<_T2...> g)
	{
		return [f, g](u64 s, _T1... a1, _T2... a2) -> _OPCx
		{
			auto [s1, v1] = g(s, a2...);
			auto [s2, v2] = f(s1, a1...);
			return std::make_pair(s2, v1 * v2);
		};
	};

	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	//template <typename _T, typename _T2, typename... _Ts, typename _OPL = _OP<_T>::INP<_Ts...>, typename _OPR = _OP<_T2>::INP<_Ts...>>
	//inline auto operator%(_OPL f, _OPR g)
	//{
	//	return [f, g](u64 s, _Ts... a)
	//	{
	//		auto [s1, v1] = g(s, a...);
	//		auto [s2, v2] = f(s1, a...);
	//		return std::make_pair(s2, static_cast<_T2>(v1 * v2));
	//	};
	//};
	
	template <typename... _Ts>
	inline auto operator%(_OP<double>::INP<_Ts...> f, _OP<double>::INP<_Ts...> g)
	{
		return [f, g](u64 s, _Ts... a)
		{
			auto [s1, v1] = g(s, a...);
			auto [s2, v2] = f(s1, a...);
			return std::make_pair(s2, static_cast<double>(v1 * v2));
		};
	};

	template <typename... _Ts>
	inline auto operator%(_OP<cpx>::INP<_Ts...> f, _OP<double>::INP<_Ts...> g)
	{
		return [f, g](u64 s, _Ts... a)
		{
			auto [s1, v1] = g(s, a...);
			auto [s2, v2] = f(s1, a...);
			return std::make_pair(s2, v1 * v2);
		};
	};

	template <typename... _Ts>
	inline auto operator%(_OP<cpx>::INP<_Ts...> f, _OP<cpx>::INP<_Ts...> g)
	{
		return [f, g](u64 s, _Ts... a)
		{
			auto [s1, v1] = g(s, a...);
			auto [s2, v2] = f(s1, a...);
			return std::make_pair(s2, v1 * v2);
		};
	};

	template <typename... _Ts>
	inline auto operator%(_OP<double>::INP<_Ts...> f, _OP<cpx>::INP<_Ts...> g)
	{
		return [f, g](u64 s, _Ts... a)
		{
			auto [s1, v1] = g(s, a...);
			auto [s2, v2] = f(s1, a...);
			return std::make_pair(s2, v1 * v2);
		};
	};

	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

	template <typename... _T>
	inline _OP<cpx>::INP<_T...> castINP(_OP<double>::INP<_T...> _in) {
		auto fun = [&](u64 s, _T...) {
			auto [s, v] = std::apply(_in, _T...);
			return std::make_pair(_in, cpx(v, 0.0));
		};
		return fun;
	}
};

#endif