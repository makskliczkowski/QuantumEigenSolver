#pragma once
#ifndef BINARY_H
#define BINARY_H

#ifndef COMMON_H
#include "./common.h"
#endif

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

// ----------------------------------------------------------------------------- Macros to generate the lookup table (at compile-time) -----------------------------------------------------------------------------
#define R2(n) n, n + 2*64, n + 1*64, n + 3*64
#define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
#define REVERSE_BITS R6(0), R6(2), R6(1), R6(3)
#define ULLPOW(k) 1ULL << k
#define RETURNS(...) -> decltype((__VA_ARGS__)) { return (__VA_ARGS__); }
//

#define SPIN

// use binary representation 0/1 instead of -1/1
#ifdef USE_BINARY
	#undef SPIN
#endif

#ifdef SPIN
	#define INT_TO_BASE_BIT intToBaseBitSpin
	#define BASE_TO_INT baseToIntSpin
#else
	#define INT_TO_BASE_BIT intToBaseBit
	#define BASE_TO_INT baseToInt
#endif // SPIN






// The macro `REVERSE_BITS` generates the table
const u64 lookup[256] = { REVERSE_BITS };

// Vector containing powers of 2 from 2^0 to 2^(L-1) - after 32 lattice sites we need to handle it with vectors
const v_1d<u64> BinaryPowers = { ULLPOW(0), ULLPOW(1), ULLPOW(2), ULLPOW(3),
								ULLPOW(4), ULLPOW(5), ULLPOW(6), ULLPOW(7),
								ULLPOW(8), ULLPOW(9), ULLPOW(10), ULLPOW(11),
								ULLPOW(12), ULLPOW(13), ULLPOW(14), ULLPOW(15),
								ULLPOW(16), ULLPOW(17), ULLPOW(18), ULLPOW(19),
								ULLPOW(20), ULLPOW(21), ULLPOW(22), ULLPOW(23),
								ULLPOW(24), ULLPOW(25), ULLPOW(26), ULLPOW(27),
								ULLPOW(28), ULLPOW(29), ULLPOW(30), ULLPOW(31) };                   
// ----------------------------------------------------------------------------- 				  binary search  				 -----------------------------------------------------------------------------

/*
* @brief Finding index of base vector in mapping to reduced basis
* @typeparam T 
* @param arr arary/vector conataing the mapping to the reduced basis 
* @param l_point left maring for binary search 
* @param r_point right margin for binary search 
* @param element element to search in the array 
* @returns -1 if not found else index of @ref element
*/
template <class T>
inline u64 binary_search(const std::vector<T>& arr, u64 l_point, u64 r_point, T element) {
	if (l_point < 0) assert(false && "What?");
	if (r_point >= arr.size()) {
		return -1;
	}
	if (r_point >= l_point) {
		u64 middle = l_point + (r_point - l_point) / 2;
		if (arr[middle] == element) return middle;
		else if (arr[middle] < element) return binary_search(arr, middle + 1, r_point, element);
		else return binary_search(arr, l_point, middle - 1, element);
	}
	return -1;
}

// double instance
template <>
inline u64 binary_search(const std::vector<double>& arr, u64 l_point, u64 r_point, double element) {
	if (l_point < 0) assert(false && "What?");
	if (r_point >= arr.size()) {
		return -1;
	}
	if (r_point >= l_point) {
		u64 middle = l_point + (r_point - l_point) / 2;
		if (abs(arr[middle] - element) < 1e-12) return middle;
		else if (arr[middle] < element) return binary_search(arr, middle + 1, r_point, element);
		else return binary_search(arr, l_point, middle - 1, element);
	}
	return -1;
}


// ---------------------------------- check bit ----------------------------------
/*
*@brief Check the k'th bit
*@param n Number on which the bit shall be checked
*@param k Number of bit (from 0 to 63) - count from right!!!
*@returns Bool on if the bit is set or not
*/
inline bool checkBit(u64 n, int k) {
	return n & (1ULL << k);
}

/*
*@brief Check the k'th bit
*@param n Number on which the bit shall be checked
*@param L Number of bit - counted from left
*@returns Bool on if the bit is set or not
*/
template<typename _type>
inline _type checkBitV(const v_1d<_type>& n, int L) {
	return n[L];
}

/*
*@brief Check the k'th bit
*@param n Number on which the bit shall be checked
*@param L Number of bit - counted from left
*@returns Bool on if the bit is set or not
*/
template<typename _type>
inline _type checkBitV(const Col<_type>& n, int L) {
	return n(L);
}

// -----------------------------------------------------------------------------  				  transformations   				 -----------------------------------------------------------------------------

/*
* @brief translates the integer to a vector in a given (binary) base (with bitwise check) (arma)
* @param idx index (integer) of a state
* @param vec vector to be transformed onto
*/
template<typename T>
inline void intToBaseBit(u64 idx, Col<T>& vec) {
	const u64 size = vec.size();
#ifdef DEBUG_BINARY
	auto start = std::chrono::high_resolution_clock::now();
#endif // DEBUG
	for (int k = 0; k < size; k++)
		vec(k) = checkBit(idx, size - 1 - k);
#ifdef DEBUG_BINARY
	stout << "->\n\t->Check bit binary change time taken: " << tim_mus(start) << "mus" << EL;
#endif // DEBUG
}

/*
* @brief translates the integer to a vector in a given (binary) base (with bitwise check)
* @param idx index (integer) of a state
* @param vec vector to be transformed onto
*/
template<typename T>
inline void intToBaseBit(u64 idx, v_1d<T>& vec) {
	const u64 size = vec.size();
#ifdef DEBUG_BINARY
	auto start = std::chrono::high_resolution_clock::now();
#endif // DEBUG
	for (int k = 0; k < size; k++)
		vec[k] = checkBit(idx, size - 1 - k);
#ifdef DEBUG_BINARY
	stout << "->\n\t->Check bit binary change time taken: " << tim_mus(start) << "mus" << EL;
#endif // DEBUG
}


/*
* @brief translates the integer to a vector in a given (binary) base but here we use spin as values in a vector (with bitwise check)
* @param idx index (integer) of a state
* @param vec vector to be transformed onto
*/
template<typename T>
inline void intToBaseBitSpin(u64 idx, Col<T>& vec) {
	const u64 size = vec.size();
#ifdef DEBUG_BINARY
	auto start = std::chrono::high_resolution_clock::now();
#endif // DEBUG
	for (int k = 0; k < size; k++)
		vec(k) = checkBit(idx, size - 1 - k) ? 1.0 : -1.0;
#ifdef DEBUG_BINARY
	stout << "->\n\t->Check bit binary change time taken: " << tim_mus(start) << "mus" << EL;
#endif // DEBUG
}

/*
* @brief translates the integer to a vector in a given (binary) base but here we use spin as values in a vector (with bitwise check) (arma)
* @param idx index (integer) of a state
* @param vec vector to be transformed onto
*/
template<typename T>
inline void intToBaseBitSpin(u64 idx, v_1d<T>& vec) {
	const u64 size = vec.size();
#ifdef DEBUG_BINARY
	auto start = std::chrono::high_resolution_clock::now();
#endif // DEBUG
//#pragma omp parallel for
	for (int k = 0; k < size; k++)
		vec[k] = checkBit(idx, size - 1 - k) ? 1.0 : -1.0;
#ifdef DEBUG_BINARY
	stout << "->\n\t->Check bit binary change time taken: " << tim_mus(start) << "mus" << EL;
#endif // DEBUG
}

// -----------------------------------------------------------------------------  				  transformations using modulo

/*
*@brief Conversion to system vector of a given base (modulo)
*@param idx numner for conversion 
*@param vec vector containing the binary string 
*@param base base to covert to
*/
inline void intToBase(u64 idx, v_1d<int>& vec, int base = 2) {
#ifdef DEBUG_BINARY
	auto start = std::chrono::high_resolution_clock::now();
#endif // DEBUG
	u64 temp = idx;
	const u64 size = vec.size();
	for (int k = 0; k < size; k++) {
		vec[size - 1 - k] = temp % base;
		temp = temp / u64(base);
	}
#ifdef DEBUG_BINARY
	stout << "->\n\t\t->Standard binary change time taken: " << tim_mus(start) << "mus" << EL;
#endif // DEBUG
}

/*
*@brief Conversion to system vector of a given base (modulo)
*@param idx numner for conversion
*@param vec vector containing the binary string
*@param base base to covert to
*@param powers powers to be used in conversion
*/
inline void intToBase(u64 idx, v_1d<int>& vec, const v_1d<u64>& powers) {
	u64 temp = idx;
	const u64 size = vec.size();
	for (int k = size - 1; k >= 0; k--) {
		vec[size - 1 - k] = static_cast<int>(temp / powers[k]);
		temp -= vec[size - 1 - k] * powers[k];
	}
}

/*
*@brief Conversion to system vector of a given base (modulo)
*@param idx numner for conversion
*@param vec vector (arma) containing the binary string
*@param base base to covert to
*/
template<typename T>
inline void intToBase(u64 idx, Col<T>& vec, int base = 2) {
#ifdef DEBUG_BINARY
	auto start = std::chrono::high_resolution_clock::now();
#endif // DEBUG	
	u64 temp = idx;
	const u64 size = vec.size();
	for (int k = 0; k < size; k++) {
		vec(size - 1 - k) = temp % base;
		temp = temp / u64(base);
	}
#ifdef DEBUG_BINARY
	stout << "->\n\t\t->Standard binary change time taken: " << tim_mus(start) << "mus" << EL;
#endif // DEBUG
}

/*
*Conversion to system vector of a given base (modulo)
*@param idx numner for conversion
*@param vec vector (arma) containing the binary string
*@param base base to covert to
*@param powers powers to be used in conversion
*/
template<typename T>
inline void intToBase(u64 idx, Col<T>& vec, const v_1d<u64>& powers) {
	u64 temp = idx;
	const u64 size = vec.size();
	for (int k = size - 1; k >= 0; k--) {
		vec[size - 1 - k] = static_cast<int>(temp / powers[k]);
		temp -= vec[size - 1 - k] * powers[k];
	}
}
// -----------------------------------------------------------------------------  				  base change

/*
*@brief Conversion from base vector to an integer
*@param vec string 
*@param base base to covert to
*@returns unsigned long long integer 
*/
inline u64 baseToInt(const v_1d<int>& vec, int base = 2) {
	u64 val = 0;
	u64 exp = 1;
	const u64 size = vec.size();
	for (int k = 0; k < size; k++) {
		val += static_cast<u64>(vec[size - 1 - k]) * exp;
		exp *= base;
	}
	return val;
}

/*
*@brief Conversion from base vector to an integer
*@param vec string 
*@param powers precalculated powers vector
*@param base base to covert to
*@returns unsigned long long integer 
*/
inline u64 baseToInt(const v_1d<int>& vec, const v_1d<u64>& powers) {
	u64 val = 0;
	const u64 size = vec.size();
//#pragma omp parallel for reduction(+:val)
	for (int k = 0; k < size; k++)
		val += static_cast<u64>(vec[size - 1 - k]) * powers[k];
	return val;
}

/*
*@brief Conversion from base vector to an integer
*@param vec string
*@param powers precalculated powers vector
*@param base base to covert to
*@returns unsigned long long integer
*/
template<typename T>
inline u64 baseToInt(const Col<T>& vec, const v_1d<u64>& powers) {
	u64 val = 0;
	const u64 size = vec.size();
//#pragma omp parallel for reduction(+:val)
	for (int k = 0; k < size; k++)
		val += static_cast<u64>(vec(size - 1 - k)) * powers[k];
	return val;
}

/*
*@brief Conversion from base vector to an integer
*@param vec string
*@param powers precalculated powers vector
*@param base base to covert to
*@returns unsigned long long integer
*/
template<typename T>
inline u64 baseToInt(const Col<T>& vec) {
	u64 val = 0;
	const u64 size = vec.size();
//#pragma omp parallel for reduction(+:val)
	for (int k = 0; k < size; k++)
		val += static_cast<u64>(vec(size - 1 - k)) * BinaryPowers[k];
	return val;
}


/*
*@brief Conversion from base vector to an integer
*@param vec string
*@param powers precalculated powers vector
*@param base base to covert to
*@returns unsigned long long integer
*/
template<typename T>
inline u64 baseToIntSpin(const Col<T>& vec, const v_1d<u64>& powers) {
	u64 val = 0;
	const u64 size = vec.size();
//#pragma omp parallel for reduction(+:val)
	for (int k = 0; k < size; k++)
		val += static_cast<u64>((vec(size - 1 - k) + 1.0)/2.0) * powers[k];
	return val;
}

template<typename T>
inline u64 baseToIntSpin(const Col<T>& vec) {
	u64 val = 0;
	const u64 size = vec.size();
//#pragma omp parallel for reduction(+:val)
	for (int k = 0; k < size; k++)
		val += static_cast<u64>((std::real(vec(size - 1 - k)) + 1.0) / 2.0) * BinaryPowers[k];
	return val;
}

// -----------------------------------------------------------------------------   				 for states operation   				 -----------------------------------------------------------------------------

template<typename T1, typename T2>
inline T1 cdotm(arma::Col<T1> lv, arma::Col<T2> rv, int numthreads = 1) {
	//if (lv.size() != rv.size()) throw "not matching sizes";
	T1 acc = 0;
//#pragma omp parallel for reduction(+ : acc) numthreads(numthreads)
	for (auto i = 0; i < lv.size(); i++)
		acc += std::conj(lv(i)) * rv(i);
	return acc;
}

template<typename T1, typename T2>
inline T1 dotm(arma::Col<T1> lv, arma::Col<T2> rv, int numthreads = 1) {
	//if (lv.size() != rv.size()) throw "not matching sizes";
	T1 acc = 0;
//#pragma omp parallel for reduction(+ : acc) numthreads(numthreads)
	for (auto i = 0; i < lv.size(); i++)
		acc += (lv(i)) * rv(i);
	return acc;
}

// -----------------------------------------------------------------------------    				 manipulations   				  -----------------------------------------------------------------------------

// ---------------------------------- rotate ----------------------------------
/*
*@brief Rotates the binary representation of the input decimal number by one left shift
*@param n number to rotate
*@param maxPower maximal power of 2
*@returns rotated number
*/
inline u64 rotateLeft(u64 n, int L) {
	u64 maxPower = BinaryPowers[L - int32_t(1)];
	return (n >= maxPower) ? (((int64_t)n - (int64_t)maxPower) * 2 + 1) : n * 2;
}

/*
*@brief Rotates the binary representation of the input decimal number by one left shift
*@param n vector to rotate
*@returns rotated number
*/
template<typename _type>
inline void rotateLeftV(v_1d<_type>& n, int L) {
	std::ranges::rotate(n.begin(), n.begin() + L, n.end());
	return v_1d<_type>();
}

/*
*@brief Rotates the binary representation of the input decimal number by one left shift - copy
*@param n vector to rotate
*@returns rotated number
*/
template<typename _type>
inline v_1d<_type> rotateLeftV_C(const v_1d<_type>& n, int L) {
	auto tmp = n;
	std::ranges::rotate(tmp.begin(), tmp.begin() + L, tmp.end());
	return tmp;
}


// ---------------------------------- flip all bits ----------------------------------

/*
*@brief Flip the bits in the number. The flipping is done via substracting the maximal number we can get for a given bitnumber
*@param n number to be flipped
*@param maxBinaryNum maximal power of 2 for given bit number(maximal length is 64 for ULL)
*@returns flipped number
*/
inline u64 flipAll(u64 n, int L) {
	return BinaryPowers[L] - n - 1;
}

/*
*@brief Flip the bits in the number. The flipping is done via substracting the maximal number we can get for a given bitnumber
*@param n number to be flipped
*@param maxBinaryNum maximal power of 2 for given bit number(maximal length is 64 for ULL)
*@returns flipped number
*/
template<typename _type>
inline v_1d<_type> flipAllV_C(const v_1d<_type>& n, int L) {
	auto tmp = n;
	for (auto i = 0; i < L; i++) {
#ifdef SPIN
		tmp[i] *= -1;
#else 
		tmp[i] = tmp[i] == 1 ? 0 : 1;
#endif // SPIN
	}
	return tmp;

}

/*
*@brief Flip the bits in the number. The flipping is done via substracting the maximal number we can get for a given bitnumber
*@param n number to be flipped
*@param maxBinaryNum maximal power of 2 for given bit number(maximal length is 64 for ULL)
*@returns flipped number
*/
template<typename _type>
inline Col<_type> flipAllV_C(const Col<_type>& n, int L) {
	auto tmp = n;
	for (auto i = 0; i < L; i++) {
#ifdef SPIN
		tmp(i) *= -1;
#else 
		tmp(i) = tmp(i) == 1 ? 0 : 1;
#endif // SPIN
	}
	return tmp;

}


/*
*@brief Flip the bits in the number. The flipping is done via substracting the maximal number we can get for a given bitnumber
*@param n number to be flipped
*@param maxBinaryNum maximal power of 2 for given bit number(maximal length is 64 for ULL)
*@returns flipped number
*/
template<typename _type>
inline void flipAllV(v_1d<_type>& n, int L) {
	for (auto i = 0; i < L; i++) {
#ifdef SPIN
		n[i] *= -1;
#else 
		n[i] = n[i] == 1 ? 0 : 1;
#endif // SPIN
	}
}

/*
*@brief Flip the bits in the number. The flipping is done via substracting the maximal number we can get for a given bitnumber
*@param n number to be flipped
*@param maxBinaryNum maximal power of 2 for given bit number(maximal length is 64 for ULL)
*@returns flipped number
*/
template<typename _type>
inline void flipAllV(Col<_type>& n, int L) {
	for (auto i = 0; i < L; i++) {
#ifdef SPIN
		n(i) *= -1;
#else 
		n(i) = n(i) == 1 ? 0 : 1;
#endif // SPIN
	}
}

// ---------------------------------- flip single bit ----------------------------------

/*
*@brief Flip the bit on k'th site and return the number it belongs to. The bit is checked from right to left!
*@param n number to be checked
*@param k k'th site for flip to be checked
*@returns number with k'th bit from the right flipped
*/
inline u64 flip(u64 n, int k) {
	return checkBit(n, k) ? (int64_t(n) - (int64_t)BinaryPowers[k]) : (n + BinaryPowers[k]);
}

/*
*@brief Flip the bit on k'th site and return the number it belongs to. The bit is checked from right to left!
*@param n number to be checked
*@param k k'th site for flip to be checked
*@returns number with k'th bit from the right flipped
*/
template<typename _type>
inline v_1d<_type> flipV_C(const v_1d<_type>& n, int k) {
	auto tmp = n;
#ifdef SPIN
	tmp[k] *= -1;
#else 
	tmp[k] = tmp[k] == 1 ? 0 : 1;
#endif // SPIN
	return tmp;
}

/*
*@brief Flip the bit on k'th site and return the number it belongs to. The bit is checked from right to left!
*@param n number to be checked
*@param k k'th site for flip to be checked
*@returns number with k'th bit from the right flipped
*/
template<typename _type>
inline Col<_type> flipV_C(const Col<_type>& n, int k) {
	auto tmp = n;
#ifdef SPIN
	tmp(k) *= -1;
#else 
	tmp(k) = tmp(k) == 1 ? 0 : 1;
#endif // SPIN
	return tmp;
}


/*
*@brief Flip the bit on k'th site and return the number it belongs to. The bit is checked from right to left!
*@param n number to be checked
*@param k k'th site for flip to be checked
*@returns number with k'th bit from the right flipped
*/
template<typename _type>
inline void flipV(v_1d<_type>& n, int k) {
#ifdef SPIN
	n[k] *= -1;
#else 
	n[k] = n[k] == 1 ? 0 : 1;
#endif // SPIN
}

/*
*@brief Flip the bit on k'th site and return the number it belongs to. The bit is checked from right to left!
*@param n number to be checked
*@param k k'th site for flip to be checked
*@returns number with k'th bit from the right flipped
*/
template<typename _type>
inline void flipV(Col<_type>& n, int k) {
#ifdef SPIN
	n(k) *= -1;
#else 
	//stout << "flippin from: " << VEQ(n(k));
	n(k) = (n(k) > 0) ? 0.0 : 1.0;
	//stout << "to " << VEQ(n(k)) << EL;
#endif // SPIN
}

// ---------------------------------- revelse all bits ----------------------------------

/*
* @brief Function that calculates the bit reverse, note that 64 bit representation
* is now taken and one has to be sure that it doesn't exceede it (which it doesn't, we sure)
* @param L We need to know how many bits does the number really take because the function can take up to 64
* @returns number with reversed bits moved to be maximally of size L again
*/
inline u64 reverseBits(u64 n, int L) {
	u64 rev = (lookup[n & 0xffULL] << 56) |					// consider the first 8 bits
		(lookup[(n >> 8) & 0xffULL] << 48) |				// consider the next 8 bits
		(lookup[(n >> 16) & 0xffULL] << 40) |				// consider the next 8 bits
		(lookup[(n >> 24) & 0xffULL] << 32) |				// consider the next 8 bits
		(lookup[(n >> 32) & 0xffULL] << 24) |				// consider the next 8 bits
		(lookup[(n >> 40) & 0xffULL] << 16) |				// consider the next 8 bits
		(lookup[(n >> 48) & 0xffULL] << 8) |				// consider the next 8 bits
		(lookup[(n >> 54) & 0xffULL]);						// consider last 8 bits
	return (rev >> (64 - L));								// get back to the original maximal number
}

/*
* @brief Function that calculates the bit reverse, note that 64 bit representation
* is now taken and one has to be sure that it doesn't exceede it (which it doesn't, we sure)
*@param L We need to know how many bits does the number really take because the function can take up to 64
*@returns number with reversed bits moved to be maximally of size L again
*/
template<typename _type>
inline v_1d<_type> reverseBitsV_C(const v_1d<_type>& n, int L) {
	v_1d<_type> tmp = n;
	std::ranges::reverse(tmp.begin(), tmp.end());
	return tmp;
}

/*
* @brief Function that calculates the bit reverse, note that 64 bit representation
* is now taken and one has to be sure that it doesn't exceede it (which it doesn't, we sure)
* @param L We need to know how many bits does the number really take because the function can take up to 64
* @returns number with reversed bits moved to be maximally of size L again
*/
template<typename _type>
inline void reverseBitsV(v_1d<_type>& n, int L) {
	std::ranges::reverse(n.begin(), n.end());
}

/*
* @brief Function that calculates the bit reverse, note that 64 bit representation
* is now taken and one has to be sure that it doesn't exceede it (which it doesn't, we sure)
* @param L We need to know how many bits does the number really take because the function can take up to 64
* @returns number with reversed bits moved to be maximally of size L again
*/
template<typename _type>
inline Col<_type> reverseBitsV(const Col<_type>& n, int L) {
	return arma::reverse(n);
}

#endif