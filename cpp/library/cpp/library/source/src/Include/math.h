#pragma once
#include <cmath>
#include <complex>

/*******************************
* Contains the possible methods
* for using math in simulation.
*******************************/

/*
* @brief Check the sign of a value
* @param val value to be checked
* @return sign of a variable
*/
template <typename T1, typename T2>
inline T1 sgn(T2 val) {
	return (T2(0) < val) - (val < T2(0));
}

/*
* @brief Defines an euclidean modulo denoting also the negative sign
* @param a left side of modulo
* @param b right side of modulo
* @return euclidean a%b
* @link https://en.wikipedia.org/wiki/Modulo_operation
*/
template <typename _T>
inline _T modEUC(_T a, _T b)
{
	_T m = a % b;
	if (m < 0) m = (b < 0) ? m - b : m + b;
	return m;
}