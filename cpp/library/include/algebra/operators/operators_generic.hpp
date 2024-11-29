/**
* @file operators_generic.hpp
* @brief Contains the most common operators used in quantum algebra.
* 
* This file defines various operators that act on a Hilbert space. It includes
* general operators, operator algebra, and specific operators for entropy calculations.
* 
* The following headers are included:
* - general_operator.h: Defines the base class for all operators.
* - operator_algebra.h: Provides functions for operator algebra, such as addition and multiplication.
* - entropy.h: Contains functions for calculating entropy, which is a measure of disorder or randomness.
* 
* @note This file is part of the Quantum EigenSolver project and is under constant development.
* 
* @date December 2024
* @author Maksymilian Kliczkowski
* @institution WUST, Poland
*/
#pragma once

#include "../general_operator.h"
#include "../operator_algebra.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>

#ifndef ENTROPY_H
#	include "../quantities/entropy.h"
#endif // !ENTROPY_H

/**
* Separators used for the later parsing of the operators and their names:
* - OPERATOR_SEP: separator between the operator type and the operator name.
*   Example: "type/name" where "type" is the operator type and "name" is the operator name.
* - OPERATOR_SEP_CORR: separator between the operator name and the correlation site.
*   Example: "nn-1-2" where "nn" is the operator name and "1-2" indicates correlation between sites 1 and 2.
* - OPERATOR_SEP_MULT: separator between the operator name and multiple sites.
*   Example: "nn,1,2" where "nn" is the operator name and "1,2" indicates the operator acts on sites 1 and 2 separately.
* - OPERATOR_SEP_DIFF: separator between the operator name and the site.
*   Example: "nnm1" where "nn" is the operator name and "m1" indicates the difference between sites.
* - OPERATOR_SEP_RANGE: separator between the operator name and the site.
*   Example: "nn:1:5" where "nn" is the operator name and "1:5" indicates a range of sites from 1 to 5.
* - OPERATOR_SEP_RANDOM: separator between the operator name and the site.
*   Example: "nnr" where "nn" is the operator name and "r" indicates a random operator.
* - OPERATOR_SEP_DIV: separator between the operator name and the site.
*   Example: "nn_1_2" where "nn" is the operator name and "_1_2" indicates division of the sites.
*/
constexpr auto OPERATOR_SEP			= "/";
constexpr auto OPERATOR_SEP_CORR	= "-";
constexpr auto OPERATOR_SEP_MULT 	= ",";
constexpr auto OPERATOR_SEP_DIFF	= "m";
constexpr auto OPERATOR_SEP_RANGE	= ":";
constexpr auto OPERATOR_SEP_RANDOM	= "r";
constexpr auto OPERATOR_SEP_DIV		= "_";
constexpr auto OPERATOR_PI			= "pi";
constexpr auto OPERATOR_SITE		= "l";
constexpr auto OPERATOR_SITEU    	= "L";
constexpr auto OPERATOR_SITE_M_1    = true;
#define OPERATOR_INT_CAST(x) static_cast<size_t>(x)
#define OPERATOR_INT_CAST_S(v, x, p) (v ? STR(OPERATOR_INT_CAST(x)) : STRP(x, p))

// ##########################################################################################################################################

namespace Operators 
{
	constexpr double _SPIN			=		0.5;
	constexpr double _SPIN_RBM		=		_SPIN;
};

// ##########################################################################################################################################