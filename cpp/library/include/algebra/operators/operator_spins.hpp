/**
* @file operator_spins.hpp
* @brief This file contains the declaration of various spin operators used in quantum mechanics.
* 
* The spin operators are defined within the `Operators::SpinOperators` namespace and include common
* spin operators such as sigma-x, sigma-z, raising, and lowering operators. Additionally, there is
* a namespace for random superposition operators.
* 
* @namespace Operators
* The main namespace for all operator-related functionalities.
* 
* @namespace Operators::SpinOperators
* Contains the most common spin operators used in quantum mechanics.
* 
* @namespace Operators::SpinOperators::RandomSuperposition
* Contains operators related to random superpositions.
* 
* @tparam _T The data type used for the operators, default is double.
* 
* @function sig_x
* @brief Computes the sigma-x operator.
* 
* @function sig_z
* @brief Computes the sigma-z operator.
* 
* @function sig_p
* @brief Computes the raising operator.
* 
* @function sig_m
* @brief Computes the lowering operator.
* 
* @function Operators::SpinOperators::RandomSuperposition::sig_z
* @brief Computes the sigma-z operator for random superpositions.
* 
* @function Operators::SpinOperators::RandomSuperposition::sig_z_vanish
* @brief Computes the sigma-z operator for vanishing random superpositions.
* 
* @staticvar std::vector<double> Operators::SpinOperators::RandomSuperposition::superpositions
* @brief A static inline vector containing random superposition values.
* @date December 2024
* @author Maksymilian Kliczkowski
* @institution WUST, Poland
* @note is included by operators_quadratic.hpp
*/
#include "operators_generic.hpp"

namespace Operators
{
    /*
	* @brief The spin operator namespace. Contains the most common spin operators.
	*/
	namespace SpinOperators
	{	
		template <typename _T = double>
		std::pair<u64, _T> sig_x(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		_T sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref);
	
		// ---
		template <typename _T = double>
		Operators::Operator<_T> sig_x(size_t _Ns, size_t _part);
		template <typename _T = double>
		Operators::Operator<_T> sig_x(size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		Operators::Operator<_T> sig_x(size_t _Ns);
		template <typename _T = double>
		Operators::Operator<_T, uint> sig_x_l(size_t _Ns);
		template <typename _T = double>
		Operators::Operator<_T, uint, uint> sig_x_c(size_t _Ns);

		// --------------------------------------------------------------------------------

		template <typename _T = double>
		std::pair<u64, _T> sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		_T sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref);
		// --- 
		template <typename _T = double>
		Operators::Operator<_T> sig_z(size_t _Ns, size_t _part);
		template <typename _T = double>
		Operators::Operator<_T> sig_z(size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		Operators::Operator<_T> sig_z(size_t _Ns);
		template <typename _T = double>
		Operators::Operator<_T, uint> sig_z_l(size_t _Ns);
		template <typename _T = double>
		Operators::Operator<_T, uint, uint> sig_z_c(size_t _Ns);

		// raising operator
		template <typename _T = double>
		std::pair<u64, _T> sig_p(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_p(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_p(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref);

		// lowering operator
		template <typename _T = double>
		std::pair<u64, _T> sig_m(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_m(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_m(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref);
	
		// --------------------------------------------------------------------------------

		// Plusminus operators
		template <typename _T = double>
		std::pair<u64, _T> sig_pm(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_pm(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_pm(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref);

		// Minusplus operators
		template <typename _T = double>
		std::pair<u64, _T> sig_mp(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_mp(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_mp(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref);

		// --------------------------------------------------------------------------------

		namespace Flux 
		{
			// flux on a given lattice (2D)
			template <typename _T = double>
			std::pair<u64, _T> sig_f(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
			template <typename _T = double>
			std::pair<_OP_V_T, _T> sig_f(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
			template <typename _T = double>
			_T sig_f(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref);
			// ---
			template <typename _T = double>
			Operators::Operator<_T> sig_f(size_t _Ns, const v_1d<uint>& sites);
		}

		// --------------------------------------------------------------------------------

		// sig q
		template <typename _T = double>
		std::pair<u64, _T> sig_k(u64 base_vec, size_t _Ns, const v_1d<uint>& sites, double _k = 1.0);
		template <typename _T = double>
		std::pair<u64, _T> sig_q(u64 base_vec, size_t _Ns, const v_1d<uint>& sites, uint _q = 1) { return sig_k<_T>(base_vec, _Ns, sites, 2.0 * M_PI * _q / (double)_Ns); }
		// ---
		template <typename _T = double>
		std::pair<u64, _T> sig_k(u64 base_vec, size_t _Ns, double _k = 1.0);
		template <typename _T = double>
		std::pair<u64, _T> sig_q(u64 base_vec, size_t _Ns, uint _q = 0) { return sig_k<_T>(base_vec, _Ns, 2.0 * M_PI * _q / (double)_Ns); }
		// ---
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_k(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, double _k = 1.0);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_q(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, uint _q = 0) { return sig_k<_T>(base_vec, _Ns, sites, 2.0 * M_PI * _q / (double)_Ns); }
		// ---
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_k(_OP_V_T_CR base_vec, size_t _Ns, double _k = 1.0);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_q(_OP_V_T_CR base_vec, size_t _Ns, uint _q = 0) { return sig_k<_T>(base_vec, _Ns, 2.0 * M_PI * _q / (double)_Ns); }
		// ---
		template <typename _T = double>
		Operators::Operator<_T> sig_k(size_t _Ns, const v_1d<uint>& sites, double _k = 0);
		template <typename _T = double>
		Operators::Operator<_T> sig_q(size_t _Ns, const v_1d<uint>& sites, uint _q = 0) { return sig_k<_T>(_Ns, sites, 2.0 * M_PI * _q / (double)_Ns); }
		// ---
		template <typename _T = double>
		Operators::Operator<_T> sig_k(size_t _Ns, double _k = 0.0);
		template <typename _T = double>
		Operators::Operator<_T> sig_q(size_t _Ns, uint _q = 0) { return sig_k<_T>(_Ns, 2.0 * M_PI * _q / (double)_Ns); }

		// --------------------------------------------------------------------------------
	};

	namespace SpinOperators
	{
		// ##########################################################################################################################################
		namespace RandomSuperposition {
			static inline std::vector<double> superpositions = { 0.3888, 0.1021, 0.3092, -0.3006, -0.9239, 0.7622, 0.4685, 0.8464, 0.4395, -0.1038, 0.3524, -0.7478, 0.0176, -0.9207, -0.7081, 0.0704 };
			Operators::Operator<double> sig_z(size_t _Ns);
			Operators::Operator<double> sig_z_vanish(size_t _Ns);
			// Operators::Operator<double> sig_z_vanish_r(size_t _Ns);
		};
	};
};