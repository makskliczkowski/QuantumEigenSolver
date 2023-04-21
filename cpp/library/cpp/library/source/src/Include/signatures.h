#pragma once

#define RETURNS(...)		-> decltype((__VA_ARGS__))		{ return (__VA_ARGS__); }								// for quickly returning values
#define DOES(...)											{ return (__VA_ARGS__); }																// for single line void functions

// ######################################################## C P P   V E R S I O N ########################################################

//! check compiler version, only C++17 or newer currently valid for this library
//! older versions are not suppeortd
#if !defined(_MSVC_LANG)
	#if (__cplusplus >= 202002L)
		#define HAS_CXX20
	#elif (__cplusplus >= 201703L)
		#define HAS_CXX17
	#else
		#error "--> at least C++17 compiler required; older versions are not supported"
	#endif
#else 
	#if (_MSVC_LANG >= 202002L)
		#define HAS_CXX20 true
	#elif (_MSVC_LANG >= 201703L)
		#define HAS_CXX17 true
	#else
		#error "--> at least C++17 compiler required; older versions are not supported"
	#endif
#endif

// WHAT IF 2020?
#ifdef HAS_CXX20
	#pragma message ("--> Compiling with c++20 compiler. Metaprogramming fully unlocked.")
	#define callable_type std::invocable
	#ifdef __has_include
		#if __has_include(<format>)
			//#include <format>
			//#define HAS_FORMAT
		#endif
	#endif
	#elif defined HAS_CXX17
		#pragma message ("--> Compiling with c++17 compiler. Failed to use type_traits and concepts in metaprogramming")
		#define callable_type typename
#endif

// ######################################################## E N U M S ########################################################

#define DECL_ENUM_ELEMENT( element )		#element
#define BEGIN_ENUM( ENUM_NAME )				static const char* eSTR##ENUM_NAME [] =
#define END_ENUM( ENUM_NAME )				; inline const char* getSTR_##ENUM_NAME(enum		\
																		ENUM_NAME index)		\
													{ return eSTR##ENUM_NAME [index]; };
#define BEGIN_ENUM_INLINE(ENUM_NAME)		static const inline char* eSTR##ENUM_NAME [] =
#define END_ENUM_INLINE(ENUM_NAME, CLASS)	; static const char* getSTR_##ENUM_NAME(enum		\
																		ENUM_NAME index)		\
													{ return CLASS##::eSTR##ENUM_NAME [index]; };

// ######################################################## F U N C T I O N S ########################################################

/*
* Define function signatures to use in debug scenarios
*/
#if !defined(FUN_SIGNATURE)
	#if defined (__GNUG__)
		#define FUN_SIGNATURE  __PRETTY_FUNCTION__
	#elif defined (_MSC_VER)
		#define FUN_SIGNATURE  __FUNCSIG__ 
	#elif defined(__INTEL_COMPILER)
		#define FUN_SIGNATURE  __FUNCTION__
	#else 
		#define FUN_SIGNATURE  __func__
	#endif
#endif

// ######################################################## C O U N T E R S ########################################################

#ifdef _MSC_VER
	#include <intrin.h>
	#include <nmmintrin.h>
	#define __builtin_popcount __popcnt
	#define __builtin_popcountll _mm_popcnt_u64
#endif

// ######################################################## C A L L S ########################################################

#if defined(DEBUG)
#define DESTRUCTOR_CALL		std::cout << FUN_SIGNATURE << "->\t destructor called" << std::endl << std::endl;
#define CONSTRUCTOR_CALL	std::cout << FUN_SIGNATURE << "->\t constructor called" << std::endl << std::endl;
#else 
#define DESTRUCTOR_CALL 
#define CONSTRUCTOR_CALL
#endif