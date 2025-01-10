/**
* @file user_interface_latp.hpp
* @brief This file contains the definition of the LatP structure within the UI_PARAMS namespace.
* 
* The LatP structure is used to define parameters related to lattice properties and boundary conditions
* for quantum eigen solvers. It includes default values for various lattice parameters and provides
* a method to set these parameters to their default values.
* 
* The structure includes:
* - Boundary conditions (bc)
* - Lattice type (typ)
* - Lattice dimensions (Lx, Ly, Lz, dim)
* - Total number of particles (Ntot)
* - A vector of total number of particles (Ntots)
* - A shared pointer to a Lattice object (lat)
* 
* The setDefault() method sets the lattice parameters to their default values.
*/

#ifndef USER_INTERFACE_INCLUDE_LAT_HPP
#define USER_INTERFACE_INCLUDE_LAT_HPP
#include "./user_interface_modp.hpp"

namespace UI_PARAMS 
{
	struct LatP 
    {
		UI_PARAM_CREATE_DEFAULT(bc, BoundaryConditions, BoundaryConditions::PBC);
		UI_PARAM_CREATE_DEFAULT(typ, LatticeTypes, LatticeTypes::SQ);
		UI_PARAM_CREATE_DEFAULT(Lx, uint, 2);
		UI_PARAM_CREATE_DEFAULT(Ly, uint, 1);
		UI_PARAM_CREATE_DEFAULT(Lz, uint, 1);
		UI_PARAM_CREATE_DEFAULT(dim, uint, 1);
		// for the Hamiltonians that only use the total number of particles, not lattices!
		UI_PARAM_CREATE_DEFAULT(Ntot, uint, 1);
		UI_PARAM_CREATE_DEFAULTV(Ntots, int);

		std::shared_ptr<Lattice> lat;

		void setDefault() {
			UI_PARAM_SET_DEFAULT(typ);
			UI_PARAM_SET_DEFAULT(bc);
			UI_PARAM_SET_DEFAULT(Lx);
			UI_PARAM_SET_DEFAULT(Ly);
			UI_PARAM_SET_DEFAULT(Lz);
			UI_PARAM_SET_DEFAULT(dim);
		};
	};
};

// ##########################################################################################################################################

#endif