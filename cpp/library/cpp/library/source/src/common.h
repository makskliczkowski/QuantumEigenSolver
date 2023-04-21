#pragma once

/*******************************
* Stores all the common functions
* used throughtly in the codes
*******************************/

#ifndef COMMON_H
#define COMMON_H

// ########################################################				 ARMA				########################################################

#include "Include/statistical.h"

// ########################################################				 OTHER				########################################################

#include "Include/random.h"
#include "Include/math.h"
#include "flog.h"

#include <omp.h>
#include <thread>

// ########################################################				CONCEPTS			########################################################

template<class T> struct is_complex						: std::false_type	{};
template<class T> struct is_complex<std::complex<T>>	: std::true_type	{};

// ########################################################				DEFINITIONS				########################################################

// using types
using cpx						=					std::complex<double>;
using uint						=					unsigned int;
using ul						=					unsigned long;
using ull						=					unsigned long long;
using u64						=					ull;
using ld						=					long double;

// constexpressions
constexpr long double PI		=					3.141592653589793238462643383279502884L;			// it is me, pi
constexpr long double TWOPI		=					2.0L * PI;											// it is me, 2pi
constexpr long double PIHALF	=					PI / 2.0L;											// it is me, half a pi
constexpr cpx I					=					cpx(0, 1);											// imaginary unit
const auto global_seed			=					std::random_device{}();								// global seed for classes

#define EL std::endl
#define stout std::cout << std::setprecision(8) << std::fixed											// standard out
#define stoutc(c) if(c) stout <<  std::setprecision(8) << std::fixed	

// debug printers
#ifdef DEBUG
    #define stoutd(str) do { stout << str << EL } while(0)
    #define PRT(time_point, cond) do { stoutc(cond) << #cond << " -> time : " << tim_mus(time_point) << "mus" << EL; } while (0);
#else
    #define stoutd(str) do { break; } while (0)
    #define PRT(time_point, cond) do { } while (0)
#endif

// ########################################################				COMMON UTILITIES				 ########################################################

#define SPACE_VEC(Lx, Ly, Lz, T) v_3d<T>(Lx, v_2d<T>(Ly, v_1d<T>(Lz)))

template<class T>
using v_Mat = v_1d<arma::Mat<T>>;																		// 1d vector of arma::mat

// ########################################################				 STREAM OVERLOADED				 ########################################################

/*
*@brief Overwritten standard stream redirection operator for 2D vectors separated by commas
*@param out outstream to be used
*@param v 1D vector
*/
template <typename T>
inline std::ostream& operator<< (std::ostream& out, const v_1d<T>& v) {
	if (!v.empty()) {
		for (int i = 0; i < v.size(); i++)
			out << v[i] << ",";
		out << "\b"; // use two ANSI backspace characters '\b' to overwrite final ", "
	}
	return out;
}

/*
* @brief Overwritten standard stream redirection operator for 2D vectors
* @param out outstream to be used
* @param v 2D vector
*/
template <typename T>
inline std::ostream& operator<< (std::ostream& out, const v_2d<T>& v) {
	if (!v.empty())
		for (auto it : v)
			out << "\t" << it << EL;
	return out;
}

// ########################################################				   VALUE EQUALS				########################################################

/*
* @brief Checks if value is equal to some param up to given tolerance
*/
template <typename _T1, typename _T2, typename _T3>
inline auto valueEqualsPrecision(_T1 value, _T2 equals, _T3 tolerance) RETURNS(std::abs(value - equals) <= tolerance);
template <typename _T1, typename _T2>
inline auto valueEqualsPrecision(_T1 value, _T2 equals) RETURNS(value == equals);

#define VEQ(name)					valueEquals(#name,(name)	, 2)
#define VEQS(name)					valueEqualsS(#name,(name)	, 2)
#define VEQV(name,val)				valueEquals(#name,(val)		, 2)
#define VEQVS(name,val)				valueEqualsS(#name,(val)	, 2)
#define VEQP(name,prec)				valueEquals(#name,(name)	, prec)
#define VEQVP(name,val,prec)		valueEquals(#name,(val)		, prec)
#define EQP(value, equals, prec)	valueEqualsPrecision(value, equals, prec)
#define EQ(value, equals) valueEqualsPrecision(value, equals)
/*
* @brief Overloads printing to standard stream for complex numbers
*/
inline std::ostream& operator<< (std::ostream& out, const cpx v)
{
	auto prec = 1e-4;
	auto phase = std::arg(v) / PI;
	while (phase < 0) phase += 2.0;
	std::string absolute = "+" + STRP(std::abs(v), 2);
	std::string phase_str = "";
	if (EQP(phase, 0.0, prec) || EQP(phase, 2.0, prec))
		phase_str = "";
	else if (EQP(phase, 1.0, prec)) {
		absolute = "-" + STRP(std::abs(v), 2);
		phase_str = "";
	}
	else
		phase_str = "*exp(" + STRP(phase, 2) + "*pi*i)";
	out << absolute + phase_str;
	return out;
}

/*
* @brief Given the char* name it prints its value in a format "name=val"
* @param name name of the variable
* @param value value of the variable
* @returns "name=val" string
*/
template <typename T>
inline auto valueEquals(const char name[], T value, int prec = 2)			RETURNS(std::string(name) + "=" + str_p(value, prec));
template <typename T>
inline auto valueEqualsS(const char name[], T value, int prec = 2)			RETURNS(std::string(name) + "=" + str_p(value, prec, true));
inline auto valueEquals(const char name[], std::string value, int prec)		RETURNS(std::string(name) + "=" + value);


// ########################################################				BINARY SEARCH				########################################################

/*
* @brief Finding index of base vector in mapping to reduced basis
* @typeparam T
* @param arr arary/vector conataing the mapping to the reduced basis
* @param l_point left maring for binary search
* @param r_point right margin for binary search
* @param element element to search in the array
* @returns -1 if not found else index of @ref element
*/
template <typename _T>
inline long long binarySearch(const v_1d<_T>& arr, ull l_point, ull r_point, _T elem) {
	if (l_point < 0 || r_point >= arr.size())
		return -1;

	if (l_point <= r_point) {
		ull middle = l_point + (r_point - l_point) / 2;												// find the middle point
		if (arr[middle] == elem) return middle;														// if found return
		else if (arr[middle] < elem) return binarySearch(arr, middle + 1, r_point, elem);			// else check the other boundaries
		else return binarySearch(arr, l_point, middle - 1, elem);
	}
	return -1;
}

template <>
inline long long binarySearch(const v_1d<double>& arr, ull l_point, ull r_point, double elem) {
	if (l_point < 0 || r_point >= arr.size())
		return -1;

	if (l_point <= r_point) {
		ull middle = l_point + (r_point - l_point) / 2;
		if (EQP(arr[middle], elem, 1e-12)) return middle;
		else if (arr[middle] < elem) return binarySearch(arr, middle + 1, r_point, elem);
		else return binarySearch(arr, l_point, middle - 1, elem);
	}
	return -1;
}

// ########################################################				PROGRESS BAR				########################################################

#ifndef PROGRESS_H
#define PROGRESS_H
class pBar {
public:
	void update(double newProgress);
	void print();
	void printWithTime(std::string message);
	pBar() : timer(std::chrono::high_resolution_clock::now()) { };							// constructor
	pBar(double percentage, int discreteSteps)
		: percentage(percentage)
		, timer(std::chrono::high_resolution_clock::now())
		, percentageSteps(static_cast<int>(percentage* discreteSteps / 100.0))
	{};
protected:
	// --------------------------- STRING ENDS
	std::string firstPartOfpBar = "\t\t\t\t[";
	std::string lastPartOfpBar = "]";
	std::string pBarFiller = "|";
	std::string pBarUpdater = "|\\/";
	// --------------------------- PROGRESS
	clk::time_point timer;														            // inner clock
	int amountOfFiller = 0;															            // length of filled elements
	int pBarLength = 50;														            // length of a progress bar
	int currUpdateVal = 0;														            //
	double currentProgress = 0;													            // current progress
	double neededProgress = 100;												            // final progress
public:
	auto get_start_time()												const { return this->timer; };
	double percentage = 34;																	// print percentage
	int percentageSteps = 1;
};
#endif // !PROGRESS_H

// ########################################################				OTHER				########################################################


#endif // !COMMON_H
