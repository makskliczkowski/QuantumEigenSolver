#pragma once
#include <chrono>
#include <iostream>
#include <algorithm>
#include <time.h>
#include "exceptions.h"

/*******************************
* Contains the possible methods
* for handling the sim time.
*******************************/

// ########################################################	T I M E   F U N C T I O N S ########################################################

using clk						=				std::chrono::steady_clock;
using clkS						=				std::chrono::system_clock;
#define DUR										std::chrono::duration
#define DURCAST									std::chrono::duration_cast
#define NOW										std::chrono::high_resolution_clock::now()	    											// standard out microseconds
#define DURATION(t1, t2)						static_cast<long double>(DURCAST<std::chrono::microseconds>(DUR(t1 - t2)).count())
/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_s(clk::time_point start)			RETURNS(DURATION(NOW, start) / 1e6);

/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_ms(clk::time_point start)			RETURNS(DURATION(NOW, start) / 1e3);

/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_mus(clk::time_point start)		RETURNS(DURATION(NOW, start));

#define stouts(text, start)		stout	<< text <<	" -> time : " << tim_s(start)	<< "s"		<< EL					// standard out seconds
#define stoutms(text, start)	stout	<< text <<	" -> time : " << tim_ms(start)	<< "ms"		<< EL					// standard out miliseconds
#define stoutmus(text, start)	stout	<< text <<	" -> time : " << tim_mus(start) << "mus"	<< EL	

// strftime format
constexpr auto PRETTY_TIME_FORMAT				= "%Y-%m-%d_%H:%M:%S";

// printf format
constexpr auto PRETTY_TIME_FORMAT_MS			= ".%03Id";

/*
* @brief convert current time to milliseconds since unix epoch
*/
template <typename _T>
static auto to_ms(const std::chrono::time_point<_T>& tp)
{
	return DURCAST<std::chrono::milliseconds>(tp.time_since_epoch()).count();
};

/*
* @brief pretty prints the time point
*/
static std::string prettyTime(clk::time_point _tp)
{
#ifdef HAS_FORMAT
	return std::format("{0:%F_%T}", _tp);
#else
	__time64_t curTime	=	\
		clkS::to_time_t(clkS::now() + DURCAST<clkS::duration>(_tp - clk::now()));

	// this function use static global pointer. so it is not thread safe solution
	std::tm timeInfo;
	_localtime64_s(&timeInfo, &curTime);

	// create a buffer
	char buffer[128];

	auto size			=	strftime(buffer,		
									(int)sizeof(buffer),
									PRETTY_TIME_FORMAT,
									&timeInfo
									);

	//auto ms				=	to_ms(_tp) % 1000;

	//size				+=	std::snprintf(buffer + size, 
	//								(size_t)sizeof(buffer) - size,
	//								PRETTY_TIME_FORMAT_MS, 
	//								ms
	//								);
	return std::string(buffer, buffer + size);
#endif
}
