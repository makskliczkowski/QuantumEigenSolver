/**
* @file operator_quadratic.hpp
* @brief This file contains the declaration of various quadratic operators used in quantum eigen solvers.
* 
* The operators are defined within the `Operators` namespace and further categorized under the `QuadraticOperators` namespace.
* These operators act on integer indices rather than configurations.
* 
* The following operators are declared:
* 
* - `site_occupation`: Computes the site occupation operator for a given site.
* - `site_occupation_r`: Computes the site occupation operator with given coefficients.
* - `site_occupation_r`: Computes the site occupation operator for given sites and coefficients.
* - `site_nq`: Computes the site occupation operator for a given momentum.
* - `nn_correlation`: Computes the nearest-neighbor correlation operator for given sites.
* - `quasimomentum_occupation`: Computes the quasimomentum occupation operator for a given momentum.
* - `quasimomentum_occupation`: Computes the quasimomentum occupation operator.
* - `kinetic_energy`: Computes the kinetic energy operator for a given lattice size.
* 
* @namespace Operators
* @namespace QuadraticOperators
*
* @date December 2024
* @version 1.0
* @author Maksymilian Kliczkowski
* @institution WUST, Poland
* @note Is included by operators_final.hpp
*/
#include "./operator_spins.hpp"

// ##########################################################################################################################################

namespace Operators
{
	/**
	* @brief For Quadratic Operators, we will treat the operators as acting on the integer index as it was not the configuration!
	*/
	namespace QuadraticOperators
	{
		// -------- n_i Operators --------

		Operators::Operator<double> site_occupation(size_t _Ns, const size_t _site);
		Operators::Operator<double> site_occupation_r(size_t _Ns, const v_1d<double>& _coeffs);
		Operators::Operator<double> site_occupation_r(size_t _Ns, const v_1d<size_t>& _sites, const v_1d<double>& _coeffs);

		// -------- n_q Operators --------
		
		Operators::Operator<double> site_nq(size_t _Ns, const size_t _momentum);

		// ------ n_i n_j Operators ------

		Operators::Operator<double> nn_correlation(size_t _Ns, const size_t _site_plus, const size_t _site_minus);

		// --- quasimomentum Operators ---

		Operators::Operator<std::complex<double>> quasimomentum_occupation(size_t _Ns, const size_t _momentum);
		Operators::Operator<double> quasimomentum_occupation(size_t _Ns);

		// ----- kinectic Operators ------

		Operators::Operator<double> kinetic_energy(size_t _Nx, size_t _Ny, size_t _Nz);
	}
};

// ##########################################################################################################################################