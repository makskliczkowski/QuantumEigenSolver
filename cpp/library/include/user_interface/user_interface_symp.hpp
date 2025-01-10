/**
* @file user_interface_symp.hpp
* @brief This file contains the definition of the SymP struct, which is used to manage symmetry parameters 
* and their default values for a quantum eigenvalue solver. It includes methods for setting default values, 
* retrieving local symmetry generators, and checking if the symmetries make the Hamiltonian complex.
*
* The supported symmetries include:
* - Translational symmetry (k)
* - Parity symmetry along the x-axis (px)
* - Parity symmetry along the y-axis (py)
* - Parity symmetry along the z-axis (pz)
* - Rotational symmetry (x)
* - U(1) symmetry (U1)
*
* Each symmetry parameter is initialized to a default value indicating that the symmetry is not used.
* The SymP struct provides functionality to set these parameters, retrieve active symmetry generators, 
* and determine if the symmetries result in a complex Hamiltonian.
*/

#ifndef USER_INTERFACE_SYMP_HPP
#define USER_INTERFACE_SYMP_HPP

// ##########################################################
#define UI_CHECK_SYM(val, gen)  if(this->val##_ != -INT_MAX) syms.push_back(std::make_pair(Operators::SymGenerators::gen, this->val##_));
// ##########################################################

#include "./user_interface_latp.hpp"

namespace UI_PARAMS
{

	/**
	* @brief Defines a container for symmetry eigenvalues.
	* @warning By default, the parameters are -maximal integer in order to tell that no symmetry is used
	*/
	struct SymP 
    {
        // ##########################################################

		// symmetry parameters
		UI_PARAM_CREATE_DEFAULT(S, bool, false);
		UI_PARAM_CREATE_DEFAULT(k, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(px, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(py, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(pz, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(x, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(U1, int, -INT_MAX);

		// other
		UI_PARAM_CREATE_DEFAULT(checkpoint, bool, true);

		void setDefault() {
			UI_PARAM_SET_DEFAULT(S);
			UI_PARAM_SET_DEFAULT(k);
			UI_PARAM_SET_DEFAULT(px);
			UI_PARAM_SET_DEFAULT(py);
			UI_PARAM_SET_DEFAULT(pz);
			UI_PARAM_SET_DEFAULT(x);
			UI_PARAM_SET_DEFAULT(U1);
			UI_PARAM_SET_DEFAULT(checkpoint);
		}

        // ##########################################################

        /**
        * @brief Retrieves a list of local symmetry generators.
        *
        * This function constructs and returns a vector of pairs, where each pair consists of a 
        * symmetry generator from the Operators::SymGenerators enumeration and an integer. 
        * The function checks for various symmetry conditions and adds the corresponding 
        * symmetry generators to the vector.
        *
        * @return v_1d<std::pair<Operators::SymGenerators, int>> A vector of pairs representing 
        * the local symmetry generators and their associated integer values.
        */
		v_1d<std::pair<Operators::SymGenerators, int>> getLocGenerator() {
			v_1d<std::pair<Operators::SymGenerators, int>> syms = {};
			UI_CHECK_SYM(k, T);
			UI_CHECK_SYM(px, PX);
			UI_CHECK_SYM(py, PY);
			UI_CHECK_SYM(pz, PZ);
			UI_CHECK_SYM(x, R);
			return syms;
		}

        // ##########################################################
		
        /**
		* @brief Checks if the symmetries make the Hamiltonian complex
		* @param Ns lattice size
		*/
		bool checkComplex(int Ns) const {
			if (this->k_ == 0 || (this->k_ == Ns / 2 && Ns % 2 == 0) || this->py_ != -INT_MAX)
				return false;
			return true;
		}

        // ##########################################################
	};
};

#endif

