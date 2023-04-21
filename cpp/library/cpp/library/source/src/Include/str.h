#pragma once
#include <string>
#include <vector>
#include <iostream> 
#include <complex>

template<class T>
using v_3d = std::vector<std::vector<std::vector<T>>>;				// 3d vector
template<class T>
using v_2d = std::vector<std::vector<T>>;							// 2d vector
template<class T>
using v_1d = std::vector<T>;										// 1d vector
template<class T>
using t_3d = std::tuple<T, T, T>;									// 3d tuple
template<class T>
using t_2d = std::pair<T, T>;										// 2d tuple - pair

#define SSTR std::string
#define STR std::to_string
#define STRP(str,prec) str_p(str, prec)
typedef v_1d<std::string> strVec;

// ############################################################### P R E C I S E   S T R I N G ###############################################################

/*
*@brief Changes a value to a string with a given precision
*@param v Value to be transformed
*@param n Precision default 2
*@return String of a value
*/
template <typename _T>
inline std::string str_p(const _T v, const int n = 2, bool scientific = false) {
	std::ostringstream out;
	out.precision(n);
	if (scientific)
		out << std::scientific;
	else
		out << std::fixed;
	out << v;
	return out.str();
}
template <>
inline std::string str_p(const int v, const int n, bool scientific) {
	std::ostringstream out;
	if (scientific)
		out << std::scientific;
	out << v;
	return out.str();
}
template <>
inline std::string str_p(const std::complex<double> v, const int n, bool scientific) {
	std::ostringstream out;
	out.precision(n);
	if (scientific)
		out << std::scientific;
	else
		out << std::fixed;
	out << "[" << std::real(v) << ", " << std::imag(v) << "]";
	return out.str();
}
template <>
inline std::string str_p(const std::string v, const int n, bool scientific) {
	return v;
}
template <>
inline std::string str_p(const char* v, const int n, bool scientific) {
	return SSTR(v);
}

// ############################################################### V E C T O R I Z E ###############################################################

strVec splitStr(const std::string& s, std::string delimiter = "\t");
strVec fromPtr(int argc, char** argv, unsigned int offset = 1);

// ############################################################### S E P A R A T E D   S T R I N G ###############################################################

template <typename Type>
inline void strSepP(std::string& _out, char _sep, uint16_t prec, bool scien, Type arg) {
	_out += str_p(arg, prec, scien);
}

template <typename Type, typename... Types>
inline void strSepP(std::string& _out, char _sep, uint16_t prec, bool scien, Type arg, Types... elements) {
	strSepP(_out, _sep, prec, scien, arg);	
	_out += std::string(1, _sep);
	strSepP(_out, _sep, prec, scien, elements...);
}

template <typename... Types>
inline void strSeparatedP(std::string& out, char sep, uint16_t prec, Types... elements) {
	strSepP(out, sep, prec, false, elements...);
}

template <typename... Types>
inline void strSeparatedS(std::string& out, char sep, Types... elements) {
	strSepP(out, sep, 2, true, elements...);
}

template <typename... Types>
inline void strSeparated(std::string& out, char sep, Types... elements) {
	strSepP(out, sep, 2, false, elements...);
}