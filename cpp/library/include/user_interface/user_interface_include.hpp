/**
* @file user_interface_include.hpp
* @brief This file includes various headers for the Quantum EigenSolver project, 
*        including neural quantum states (NQS), models, lattices, and statistical measures.
* 
* @author Maksymilian Kliczkowski
* @date December 2024
*/

#ifndef USER_INTERFACE_INCLUDE_H
#define USER_INTERFACE_INCLUDE_H

#include "../../source/src/UserInterface/ui.h"
#include <memory>
#include <vector>

#ifdef _DEBUG
//	#define DEBUG_BINARY
#else
//	#define OMP_NUM_THREADS 16;
#	include <thread>
#	include <mutex>
#endif

// ######################### NQS ############################
#include "../NQS/nqs_all.hpp"
// ##########################################################


// ######################### MODELS #########################
#if 1
#ifndef ISING_H											 // #
#	include "../models/ising.h"							 // #
#endif // !ISING_H										 // #
#ifndef XYZ_H											 // #
#	include "../models/XYZ.h"							 // #
#endif // !XYZ_H										 // #
#ifndef HEISENBERG_KITAEV_H								 // #
#	include "../models/heisenberg-kitaev.h"				 // #
#endif													 // #
// random Hamiltonians									 // #
#ifndef QSM_H											 // #
#	include "../models/quantum_sun.h"					 // #
#endif													 // #
#ifndef ROSENZWEIG_PORTER_H								 // #
#	include "../models/rosenzweig-porter.h"				 // #
#endif													 // #
#ifndef ULTRAMETRIC_H									 // #
#	include "../models/ultrametric.h"					 // #
#endif													 // #
#endif
// ##########################################################

// ######################## MODELS Q ########################
#if 1
#ifndef POWER_LAW_RANDOM_BANDED_H
#include "../models/quadratic/PowerLawRandomBanded.h"	 // #
#endif
#ifndef SYK2_M_H										 // #
#include "../models/quadratic/SYK2.h"					 // #
#endif // !SYK2											 // #
#ifndef FF_M_H											 // #
#include "../models/quadratic/FreeFermions.h"			 // #
#endif // !SYK2											 // #
#ifndef AUBRY_ANDRE_M_H									 // #
#include "../models/quadratic/AubryAndre.h"				 // #
#endif // !SYK2											 // #
#endif
// ##########################################################

// ###################### LATTICES ##########################
#if 1													 // #
#ifndef SQUARE_H										 // #
#include "../../source/src/Lattices/square.h"			 // #
#endif													 // #
#ifndef HEXAGONAL_H										 // #
#include "../../source/src/Lattices/hexagonal.h"		 // #
#endif													 // #
#ifndef HONEYCOMB_H										 // #
#include "../../source/src/Lattices/honeycomb.h"		 // #
#endif													 // #
#endif													 // #
// ##########################################################

// ##################### STATISTICAL ########################
#if 1													 // #
#include "../algebra/quantities/measure.h"				 // #
#endif													 // #
// ##########################################################

#endif													 // # 