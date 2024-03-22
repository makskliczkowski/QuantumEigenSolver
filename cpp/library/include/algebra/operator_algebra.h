#pragma once
#ifndef OPALG_H
#define OPALG_H

#ifndef LINALG_H
#include "../../source/src/Include/linalg/diagonalizers.h"
#endif 

#include <functional>
using u64 = unsigned long long;
using cpx = std::complex<double>;


/*
* @brief Contains the definisions for the operator functions
* Normally, they return a pair of unsigned long long and a complex number - new state and the value of the operator
* We also define extensive operators for the case of multiple return values - new states vector and the values vector
*/
namespace Operators {
	template <typename _T>
	using _OPx = std::pair<u64, _T>;
	template <typename _T>
	using _OPxEXT = std::vector<std::pair<u64, _T>>;
	// standard operators
	typedef _OPx<cpx> _OPCx;
	typedef _OPx<double> _OPRe;
	// extensive operators
	typedef _OPxEXT<cpx> _OPCxEXT;
	typedef _OPxEXT<double> _OPReEXT;

	/*
	* @brief All possible correlators
	*/
	template <typename _RET>
	struct _OP {
		// standard operators
		using RET		=		_RET;
		using R			=		std::pair<u64, _RET>;
		// functions
		using GLB		=		std::function<R(u64)>;
		using LOC		=		std::function<R(u64, int)>;
		using COR		=		std::function<R(u64, int, int)>;

		/*
		* @brief contains all possible functions in a template
		*/
		template <typename... _T>
		using INP		=		std::function<R(u64, _T...)>;

		// ---------------------------------------------------------
		
		// extensive operators
		using RETEXT	=		_RET;									// <! vector of return values
		using REXT		=		std::vector<std::pair<u64, RETEXT>>;	// <! pair of return values
		// functions extensive
		using GLBEXT	=		std::function<REXT(u64)>;
		using LOCEXT	=		std::function<REXT(u64, int)>;
		using COREXT	=		std::function<REXT(u64, int, int)>;

		/*
		* @brief contains all possible functions in a template as extensive operator
		*/
		template <typename... _T>
		using INPEXT	=		std::function<REXT(u64, _T...)>;
	};

	// ######### S T A N D A R D #########

	using _GLBC			=		typename _OP<cpx>::GLB;			//<! global function acting on whole product state
	using _LOCC			=		typename _OP<cpx>::LOC;			//<! local function acting on single site
	using _CORC			=		typename _OP<cpx>::COR;			//<! correlation function acting on pair of sites	

#define _INPC _OP<cpx>::template INP
#define _INPR _OP<double>::template INP

	using _GLBR			=		typename _OP<double>::GLB;		//<! global function acting on whole product state
	using _LOCR			=		typename _OP<double>::LOC;		//<! local function acting on single site
	using _CORR			=		typename _OP<double>::COR;		//<! correlation function acting on pair of sites

	template<typename _T>
	using _GLB			=		typename _OP<_T>::GLB;				
	template<typename _T>
	using _LOC			=		typename _OP<_T>::LOC;
	template<typename _T>
	using _COR			=		typename _OP<_T>::COR;
	template<typename _T, typename ..._Ts>
	using _INP			=		typename _OP<_T>::template INP<_Ts...>;

	// ######### E X T E N S I V E #########

	// extensive
	using _GLBC_EXT		=		typename _OP<cpx>::GLBEXT;		//<! global function acting on whole product state
	using _LOCC_EXT		=		typename _OP<cpx>::LOCEXT;		//<! local function acting on single site
	using _CORC_EXT		=		typename _OP<cpx>::COREXT;		//<! correlation function acting on pair of sites

#define _INPC_EXT _OP<cpx>::template INPEXT
#define _INPR_EXT _OP<double>::template INPEXT

	using _GLBR_EXT		=		typename _OP<double>::GLBEXT;	//<! global function acting on whole product state
	using _LOCR_EXT		=		typename _OP<double>::LOCEXT;	//<! local function acting on single site
	using _CORR_EXT		=		typename _OP<double>::COREXT;	//<! correlation function acting on pair of sites

	template<typename _T>
	using _GLB_EXT		=		typename _OP<_T>::GLBEXT;
	template<typename _T>
	using _LOC_EXT		=		typename _OP<_T>::LOCEXT;
	template<typename _T>
	using _COR_EXT		=		typename _OP<_T>::COREXT;
	template<typename _T, typename ..._Ts>
	using _INP_EXT		=		typename _OP<_T>::template INPEXT<_Ts...>;

	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	template <typename _T, typename _T2, typename ..._Ts, typename ..._T2s>
	inline typename _INP_EXT<std::common_type<_T, _T2>, _Ts..., _T2s...> operator*(_INP_EXT<_T, _Ts...> f, _INP_EXT<_T2, _T2s...> g) 
	{
		using _retype = std::common_type<_T, _T2>;
		return [f, g](u64 s, _Ts... a1, _T2s... a2)
			{
				_OP<_retype>::REXT _out;
				// go through the output of the first function (actually the second one as we multiply)
				for (const auto& [s1, v1] : f(s, a1...))
				{
					// go through the output of the second function (actually the first one as we multiply)
					for (auto [s2, v2] : g(s1, a2...))
					{
						_out.push_back(std::make_pair(s2, v1 * v2));
					}
				}
				return _out;
			};
	};

	/*
	* @brief Imitates the behavior f \\dot g \\equiv f(g(n,...),...) 
	* Leaves more types to handle thereafter (the function takes combined number of argumentrs)
	* Fixes both the inputs to be complex  - thus returning complex
	* @param f first function to apply
	* @param g second function to apply
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

	/*
	* @brief Imitates the behavior f \\dot g \\equiv f(g(n,...),...) 
	* Leaves more types to handle thereafter (the function takes combined number of argumentrs)
	* Fixes both inputs to be real - thus returning real
	* @param f first function to apply
	* @param g second function to apply
	*/
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

	/*
	* @brief Imitates the behavior f \\dot g \\equiv f(g(n,...),...)
	* Leaves more types to handle thereafter (the function takes combined number of argumentrs)
	* Fixes first input to be real and the second input to be complex  - thus returning complex
	* @param f first function to apply
	* @param g second function to apply
	*/
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

	/*
	* @brief Imitates the behavior f \\dot g \\equiv f(g(n,...),...) 
	* Leaves more types to handle thereafter (the function takes combined number of argumentrs)
	* Fixes first input to be complex and the second input to be real - thus returning complex
	* @param f first function to apply
	* @param g second function to apply
	*/
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
		auto fun = [&](u64 s, _T... a) {
			auto [sn, v] = std::apply(_in, a...);
			return std::make_pair(sn, cpx(v, 0.0));
		};
		return fun;
	}
};

#endif