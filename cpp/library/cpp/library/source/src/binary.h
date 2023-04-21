#pragma once

/************************************
* Defines the most general methods 
* for binary representation numbers
* manipulation for the simulation use
************************************/

#ifndef BINARY_H
#define BINARY_H

#ifndef COMMON_H
#include "common.h"
#endif

#include <bit>
#include <bitset>
#include <cstdint>
#include <iostream>

// --------------------------------------------------------				SUPPRESS WARNINGS				--------------------------------------------------------
#if defined(_MSC_VER)
#define DISABLE_WARNING_PUSH           __pragma(warning( push ))
#define DISABLE_WARNING_POP            __pragma(warning( pop )) 
#define DISABLE_WARNING(warningNumber) __pragma(warning( disable : warningNumber ))

#define DISABLE_OVERFLOW								 DISABLE_WARNING(26451)
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER    DISABLE_WARNING(4100)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION            DISABLE_WARNING(4505)
// other warnings you want to deactivate...

#elif defined(__GNUC__) || defined(__clang__)
#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH           DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP            DO_PRAGMA(GCC diagnostic pop) 
#define DISABLE_WARNING(warningName)   DO_PRAGMA(GCC diagnostic ignored #warningName)

#define DISABLE_OVERFLOW								 DISABLE_WARNING(-Wstrict-overflow)
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER    DISABLE_WARNING(-Wunused-parameter)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION            DISABLE_WARNING(-Wunused-function)
// other warnings you want to deactivate... 

#else
	// another compiler: intel,...
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
// other warnings you want to deactivate... 
#endif

#define NO_OVERFLOW(X)\
	DISABLE_WARNING_PUSH;\
	DISABLE_OVERFLOW;\
	X;\
	DISABLE_WARNING_POP;

//#include <mkl.h>
DISABLE_WARNING_PUSH // include <armadillo> and suppress its warnings, cause developers suck

// ########################################################				 Macros to generate the lookup table (at compile-time) 				########################################################

#define R2(n) n, n + 2*64, n + 1*64, n + 3*64
#define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
#define REVERSE_BITS R6(0), R6(2), R6(1), R6(3)
#define ULLPOW(k) 1ULL << k

#define SPIN

// use binary representation 0/1 instead of -1/1
#ifdef USE_BINARY
	#undef SPIN
#endif

#ifdef SPIN
	#define INT_TO_BASE intToBaseSpin
	#define BASE_TO_INT baseToIntSpin
#else
	#define INT_TO_BASE_BIT intToBase
	#define BASE_TO_INT baseToInt
#endif // SPIN

// The macro `REVERSE_BITS` generates the table
const ull lookup[256] = { REVERSE_BITS };

// Vector containing powers of 2 from 2^0 to 2^(L-1) - after 32 lattice sites we need to handle it with vectors
const v_1d<ull> BinaryPowers = { ULLPOW(0) , ULLPOW(1) , ULLPOW(2) , ULLPOW(3),
								 ULLPOW(4) , ULLPOW(5) , ULLPOW(6) , ULLPOW(7),
								 ULLPOW(8) , ULLPOW(9) , ULLPOW(10), ULLPOW(11),
								 ULLPOW(12), ULLPOW(13), ULLPOW(14), ULLPOW(15),
								 ULLPOW(16), ULLPOW(17), ULLPOW(18), ULLPOW(19),
								 ULLPOW(20), ULLPOW(21), ULLPOW(22), ULLPOW(23),
								 ULLPOW(24), ULLPOW(25), ULLPOW(26), ULLPOW(27),
								 ULLPOW(28), ULLPOW(29), ULLPOW(30), ULLPOW(31),
								 ULLPOW(32), ULLPOW(33), ULLPOW(34), ULLPOW(35),
								 ULLPOW(36), ULLPOW(37), ULLPOW(38), ULLPOW(39)};
// ########################################################				 binary search				 ########################################################

// ---------------------------------- check bit ----------------------------------

/*
* @brief Check the k'th bit
* @param n Number on which the bit shall be checked
* @param k Number of bit (from 0 to 63) - count from right!!!
* @returns Bool on if the bit is set or not
*/
template <typename _T>
inline bool checkBit(_T n, int k) {
	return _T(n & (_T(1) << k));
}

template <typename _T>
inline bool checkBit(_T n, int k, int base) {
	if (base == 2)
		return checkBit(n, k);
	// iterate more than 1 bit to get the real number at that position
	_T val = 0;
	_T exp = 1;
	for (auto i = 0; i < base / 2; i++) {
		val += checkBit(n, k + i) * exp;
		exp *= 2;
	}
	return val;
}

template<typename _T1, typename _T2=_T1>
inline bool checkBit(const v_1d<_T2>& n, uint L) {
	return n[L];
}

template<typename _T1, typename _T2 = _T1>
inline bool checkBit(const arma::Col<_T2>& n, uint L) {
	return n(L);
}

// ########################################################  				  transformations   				 ########################################################

template<typename _T1, typename _T2>
inline void intToBase(_T1 idx, arma::Col<_T2>& vec, float _spin = 1.0) {
	const uint size = (uint)vec.n_elem;
	for (uint k = 0; k < size; k++)
		vec(k) = checkBit(idx, (size - 1) - k);

}

template<typename _T1, typename _T2>
inline void intToBase(_T1 idx, v_1d<_T2>& vec, float _spin = 1.0) {
	const int size = vec.size();
	for (int k = 0; k < size; k++)
		vec[k] = checkBit(idx, (size - 1) - k);

}

template<typename _T1, typename _T2>
inline void intToBaseSpin(_T1 idx, arma::Col<_T2>& vec, float _spin = 1.0) {
	const auto size = vec.n_elem;
	for (int k = 0; k < size; k++)
		vec(k) = checkBit(idx, (size - 1) - k) ? _spin : -_spin;
}

template<typename _T1, typename _T2>
inline void intToBaseSpin(_T1 idx, v_1d<_T2>& vec, float _spin = 1.0) {
	const auto size = vec.n_elem;
	for (int k = 0; k < size; k++)
		vec[k] = checkBit(idx, (size - 1) - k) ? _spin : -_spin;
}

/*
* @brief Translates the integer to a vector in a given base (with bitwise check) (arma)
* @param idx index (integer) of a state
* @param vec vector to be transformed onto
* @param base base of the int
*/
template<typename _T1, typename _T2>
inline void intToBase(_T1 idx, arma::Col<_T2>& vec, int base) {
	if (base == 2)
		INT_TO_BASE(idx, vec);
	else
	{
		auto iter = 0;
		while (idx) {
			vec(iter++) = idx % base;
			idx /= base;
		}
	}
}

template<typename _T1, typename _T2>
inline void intToBase(_T1 idx, v_1d<_T2>& vec, int base) {
	if (base == 2)
		INT_TO_BASE(idx, vec);
	else
	{
		auto iter = 0;
		while (idx) {
			vec[iter++] = idx % base;
			idx /= base;
		}
	}
}

// ########################################################  				  base change


template <typename _T1, typename _T2>
inline _T1 baseToInt(const v_1d<_T2>& vec, float _spin = 1.0) {
	const auto size = vec.size();
	_T1 val = 0;
	for (auto k = 0; k < size; k++)
		val += static_cast<_T1>(vec[size - 1 - k]) * BinaryPowers[k];
	return val;
}

template <typename _T1, typename _T2>
inline _T1 baseToInt(const arma::Col<_T2>& vec, float _spin = 1.0) {
	const auto size = vec.size();
	_T1 val = 0;
	for (auto k = 0; k < size; k++)
		val += static_cast<_T1>(vec(size - 1 - k)) * BinaryPowers[k];
	return val;
}

template <typename _T1, typename _T2>
inline _T1 baseToIntSpin(const v_1d<_T2>& vec, float _spin = 1.0) {
	const auto size = vec.size();
	_T1 val = 0;
	for (auto k = 0; k < size; k++)
		val += static_cast<_T1>((vec[size - 1 - k] / _spin + 1.0) / 2.0) * BinaryPowers[k];
	return val;
}

template <typename _T1, typename _T2>
inline _T1 baseToIntSpin(const arma::Col<_T2>& vec, float _spin = 1.0) {
	const auto size = vec.size();
	_T1 val = 0;
	for (auto k = 0; k < size; k++)
		val += static_cast<_T1>((vec(size - 1 - k) / _spin + 1.0) / 2.0) * BinaryPowers[k];
	return val;
}

/*
*@brief Conversion from base vector to an integer
*@param vec string
*@param base base to covert to
*@returns unsigned long long integer
*/
template <typename _T1, typename _T2>
inline _T1 baseToInt(const v_1d<_T2>& vec, int base) {
	if (base == 2)
		return BASE_TO_INT(vec);

	const auto size = vec.size();
	_T1 val = 0;
	_T1 exp = 1;
	for (auto k = 0; k < size; k++) {
		val += static_cast<_T1>(vec[size - 1 - k]) * exp;
		exp *= base;
	}
	return val;
}

// ########################################################   				 for states operation   				 ########################################################

template<typename _T1, typename _T2>
inline _T1 cdotm(arma::Col<_T1> lv, arma::Col<_T2> rv) {
	_T1 acc = 0;
	for (auto i = 0; i < lv.n_elem; i++)
		acc += std::conj(lv(i)) * rv(i);
	return acc;
}

template<typename _T1, typename _T2>
inline _T1 dotm(arma::Col<_T1> lv, arma::Col<_T2> rv) {
	_T1 acc = 0;
	for (auto i = 0; i < lv.n_elem; i++)
		acc += lv(i) * rv(i);
	return acc;
}

// ########################################################    				 manipulations   				  ########################################################

// ---------------------------------- rotate ----------------------------------

/*
*@brief Rotates the binary representation of the input decimal number by one left shift
*@param n number to rotate
*@returns rotated number
*/
template <typename _T>
inline _T rotateLeft(_T n, uint L) {
	_T maxPower = BinaryPowers[uint(L - 1)];
	return (n >= maxPower) ? (((int64_t)n - (int64_t)maxPower) * 2 + 1) : n * 2;
}

template <typename _T>
inline _T rotateLeft(_T n, uint L, int base) {
	_T val = rotateLeft(n, L);
	for (int i = 0; i < base / 2 - 1; i++)
		val = rotateLeft(val, L);
	return val;
}

template<typename _T>
inline void rotateLeft(v_1d<_T>& n, uint m) {
	std::ranges::rotate(n.begin(), n.begin() + m, n.end());
}

template<typename _T>
inline v_1d<_T> rotateLeft(const v_1d<_T>& n, uint m, int placeholder) {
	v_1d<_T> tmp = n;
	std::ranges::rotate(tmp.begin(), tmp.begin() + m, tmp.end());
	return tmp;
}


// ---------------------------------- flip all bits ----------------------------------

/*
*@brief Flip the bits in the number. The flipping is done via substracting the maximal number we can get for a given bitnumber
*@param n number to be flipped
*@param maxBinaryNum maximal power of 2 for given bit number(maximal length is 64 for ULL)
*@returns flipped number
*/
template <typename _T>
inline _T flipAll(_T n, int L) {
	return BinaryPowers[L] - n - 1;
}

template <typename _T>
inline v_1d<_T> flipAll(const v_1d<_T>& n, int placeholder) {
	v_1d<_T> tmp = n;
	for (auto i = 0; i < tmp.size(); i++) {
#ifdef SPIN
		tmp[i] *= -1;
#else 
		tmp[i] = (tmp[i] == 1) ? 0 : 1;
#endif
	}
	return tmp;
}

template <typename _T>
inline void flipAll(const v_1d<_T>& n) {
	for (auto i = 0; i < n.size(); i++) {
#ifdef SPIN
		n[i] *= -1;
#else 
		n[i] = (n[i] == 1) ? 0 : 1;
#endif
	}
}

template <typename _T>
inline arma::Col<_T> flipAll(const arma::Col<_T>& n, int placeholder) {
	arma::Col<_T> tmp = n;
	for (auto i = 0; i < tmp.size(); i++) {
#ifdef SPIN
		tmp(i) *= -1;
#else 
		tmp(i) = (tmp(i) == 1) ? 0 : 1;
#endif
	}
	return tmp;
}

template <typename _T>
inline void flipAll(const arma::Col<_T>& n) {
	for (auto i = 0; i < n.size(); i++) {
#ifdef SPIN
		n(i) *= -1;
#else 
		n(i) = (n(i) == 1) ? 0 : 1;
#endif
	}
}

// ---------------------------------- flip single bit ----------------------------------

/*
*@brief Flip the bit on k'th site and return the number it belongs to. The bit is checked from right to left!
*@param n number to be checked
*@param k k'th site for flip to be checked
*@returns number with k'th bit from the right flipped
*/
template <typename _T>
inline _T flip(_T n, int k) {
	return checkBit(n, k) ? (_T(n) - (_T)BinaryPowers[k]) : (n + BinaryPowers[k]);
}

template<typename _T>
inline v_1d<_T> flip(const v_1d<_T>& n, int k) {
	auto tmp = n;
#ifdef SPIN
	tmp[k] *= -1;
#else 
	tmp[k] = tmp[k] == 1 ? 0 : 1;
#endif
	return tmp;
}

template<typename _T>
inline arma::Col<_T> flip(const arma::Col<_T>& n, int k) {
	auto tmp = n;
#ifdef SPIN
	tmp(k) *= -1;
#else 
	tmp(k) = tmp(k) == 1 ? 0 : 1;
#endif
	return tmp;
}

template<typename _T>
inline void flip(v_1d<_T>& n, int k, int placeholder) {
#ifdef SPIN
	n[k] *= -1;
#else 
	n[k] = n[k] == 1 ? 0 : 1;
#endif
}

template<typename _T>
inline void flip(arma::Col<_T>& n, int k, int placeholder) {
#ifdef SPIN
	n(k) *= -1;
#else 
	n(k) = (n(k) > 0) ? 0.0 : 1.0;
#endif
}

// ---------------------------------- revelse all bits ----------------------------------

/*
* @brief Function that calculates the bit reverse, note that 64 bit representation
* is now taken and one has to be sure that it doesn't exceede it (which it doesn't, we sure)
* @param L We need to know how many bits does the number really take because the function can take up to 64
* @returns number with reversed bits moved to be maximally of size L again
*/
template <typename _T>
inline _T revBits(_T n, int L, int base = 2) {
	_T rev = (lookup[n & 0xffULL] << 56)	|				// consider the first 8 bits
		(lookup[(n >> 8) & 0xffULL] << 48)	|				// consider the next 8 bits
		(lookup[(n >> 16) & 0xffULL] << 40) |				// consider the next 8 bits
		(lookup[(n >> 24) & 0xffULL] << 32) |				// consider the next 8 bits
		(lookup[(n >> 32) & 0xffULL] << 24) |				// consider the next 8 bits
		(lookup[(n >> 40) & 0xffULL] << 16) |				// consider the next 8 bits
		(lookup[(n >> 48) & 0xffULL] << 8)	|				// consider the next 8 bits
		(lookup[(n >> 54) & 0xffULL]);						// consider last 8 bits
	return (rev >> (64 - L * (base / 2)));					// get back to the original maximal number
}

template <typename _T>
inline v_1d<_T> revBits(const v_1d<_T>& n, int placeholder) {
	v_1d<_T> tmp = n;
	std::ranges::reverse(tmp.begin(), tmp.end());
	return tmp;
}

template <typename _T>
inline void revBits(v_1d<_T>& n) {
	std::ranges::reverse(n.begin(), n.end());
}

template <typename _T>
inline arma::Col<_T> reverseBitsV(const arma::Col<_T>& n, int L) {
	return arma::reverse(n);
}

#endif