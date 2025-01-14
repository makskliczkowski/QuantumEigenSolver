#include "./include/algebra/operators.h"
#include "include/algebra/general_operator.h"
#include "include/algebra/operator_algebra.h"
#include "include/algebra/operators/operators_generic.hpp"
#include "source/src/Include/str.h"
#include "source/src/binary.h"
#include "source/src/common.h"
#include "source/src/lin_alg.h"
#include <complex>
#include <iostream>
#include <string>

namespace Operators
{

	// ##############################################################################################################################

	// ######################################################### S P I N S ##########################################################

	// ##############################################################################################################################
	
	namespace SpinOperators
	{
		// ############################################################################################# 

		// ######################################## SIGMA X ############################################

		// #############################################################################################

		/**
		* @brief multiplication of sigma_xi | state >
		* @param base_vec the base vector to be acted on. This is given by the copy.
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at. The order of the sites matters!
		* @returns the pair of the new state and the value of the operator
		*/
		template <typename _T>
		std::pair<u64, _T> sig_x(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			for (auto const& site : sites)
			{
				base_vec	=	flip(base_vec, _Ns - 1 - site);
				_val		*=	Operators::_SPIN;
			}
			return std::make_pair(base_vec, _val);
		}

		/**
		* @brief Applies the sigma_x (Pauli-X) operator to a given base vector.
		*
		* This function takes a base vector and a list of sites, and applies the sigma_x
		* operator to each specified site. The sigma_x operator flips the spin at the given
		* site and multiplies the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The input base vector on which the sigma_x operator will be applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_x operator will be applied.
		* @return A pair consisting of the modified base vector and the resulting value after
		*         applying the sigma_x operator to the specified sites.
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			_OP_V_T _base_vec = base_vec;
			for (auto const& site : sites)
			{
				flip(_base_vec, site, 0, Operators::_SPIN);
				_val *= Operators::_SPIN;
			}
			return std::make_pair(_base_vec, _val);
		}

		/**
		* @brief Applies the sigma_x operator to a given base vector.
		*
		* This function applies the sigma_x (Pauli-X) operator to a given base vector
		* at specified sites and returns the resulting vector and a value.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma_x operator is applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_x operator is applied.
		* @param _ref A reference vector that will be modified to store the result (in the process).
		* @return A pair containing the modified vector and the resulting value.
		*/
		template <typename _T>
		_T sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref)
		{
			_T _val = 1.0;
			if (_ref.size() != base_vec.size())
				_ref = base_vec;

			for (auto const& site : sites)
			{
				flip(_ref, site, 0, Operators::_SPIN);
				_val *= Operators::_SPIN;
			}
			return _val;
		}

		// template instantiation
		OP_TMPLT_INST_PAIR(sig_x,, (u64, size_t, const v_1d<uint>&), u64);
		OP_TMPLT_INST_PAIR(sig_x,, (_OP_V_T_CR, size_t, const v_1d<uint>&), _OP_V_T);
		template double sig_x<double>(_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&);
		template std::complex<double> sig_x<std::complex<double>>(_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&);

		// #####################################

		/**
		* @brief Applies the sigma_x (Pauli-X) operator to a specific site in the lattice.
		*
		* This function creates an operator that applies the sigma_x operator to a specific site
		* in the lattice. The sigma_x operator flips the spin at the given site and multiplies
		* the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @param _part The site index where the sigma_x operator will be applied.
		* @return An Operator object that applies the sigma_x operator to the specified site.
		*/
		template <typename _T>
		Operators::Operator<_T> sig_x(size_t _Ns, size_t _part)
		{
			// create the function
			_GLB<_T> fun_ = [_Ns, _part](u64 state) { return sig_x<_T>(state, _Ns, { (uint)_part }); };
			_GLB_V<_T> funV_ = [_Ns, _part](_OP_V_T_CR state) { return sig_x<_T>(state, _Ns, { (uint)_part }); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 1ULL << (_Ns - 1 - _part);

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SX);
			_op.setActOn(_acts);
			_op.setNameS("Sx/" + std::to_string(_part));
			return _op;
		}

		/**
		* @brief Applies the sigma_x (Pauli-X) operator to a given set of sites in the lattice.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_x operator will be applied. This acts as a multiplication of the operators - Pauli strings.
		* @returns The operator acting on the specified sites.
		*/
		template <typename _T>
		Operators::Operator<_T> sig_x(size_t _Ns, const v_1d<uint>& sites)
		{
			// create the function
			_GLB<_T> fun_		= [_Ns, sites](u64 state) { return sig_x<_T>(state, _Ns, sites); };
			_GLB_V<_T> funV_	= [_Ns, sites](_OP_V_T_CR state) { return sig_x<_T>(state, _Ns, sites); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 0;
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1
			for(auto _part : sites)
				_acts |= 1 << (_Ns - 1 - _part);

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SX);
			_op.setActOn(_acts);

			// set the name
			std::string _name = "Sx/";
			for (auto _part : sites)
				_name += std::to_string(_part) + "-";
			_name.pop_back(); // remove the last comma
			_op.setNameS(_name);
			
			return _op;
		}

		/**
		* @brief Creates a global sigma_x (Pauli-X) operator for the entire lattice.
		*
		* This function constructs an operator that applies the sigma_x operator to all sites
		* in the lattice. The sigma_x operator flips the spin at each site and multiplies
		* the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @return An Operator object that applies the sigma_x operator to all sites in the lattice.
		*/
		template <typename _T>
		Operators::Operator<_T> sig_x(size_t _Ns)
		{
			// set the vector of sites
			v_1d<uint> _sites = Vectors::vecAtoB<uint>(_Ns);
			
			// create the function
			_GLB<_T> fun_ = [_Ns, _sites](u64 state) { return sig_x<_T>(state, _Ns, _sites); };
			_GLB_V<_T> funV_ = [_Ns, _sites](_OP_V_T_CR state) { return sig_x<_T>(state, _Ns, _sites); };

			// save on which elements the operator acts (for the sake of correctness)
			u64 _acts = (1ULL << _Ns) - 1; // set the bitmask on the state, covering all sites

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SX);
			_op.setActOn(_acts);

			// set the name
			std::string _name = "Sx/1.L.1";
			_op.setNameS(_name);

			return _op;
		}

		/**
		* @brief Creates a local sigma_x (Pauli-X) operator for a specific site in the lattice.
		*
		* This function constructs an operator that applies the sigma_x operator to a specific site
		* in the lattice. The sigma_x operator flips the spin at the given site and multiplies
		* the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @return An Operator object that applies the sigma_x operator to a specific site in the lattice.
		*/
		template <typename _T>
		Operators::Operator<_T, uint> sig_x_l(size_t _Ns)
		{
			_LOC<_T> fun_ 	= [_Ns](u64 state, uint _part) { return sig_x<_T>(state, _Ns, { _part }); };
			_LOC_V<_T> funV_ 	= [_Ns](_OP_V_T_CR state, uint _part) { return sig_x<_T>(state, _Ns, { _part }); };

			// set the operator
			Operator<_T, uint> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SX);
			_op.setNameS("Sx/L");
			return _op;
		}

		/**
		* @brief Creates a correlation sigma_x (Pauli-X) operator for a set of sites in the lattice.
		*
		* This function constructs an operator that applies the sigma_x operator to a set of sites
		* in the lattice. The sigma_x operator flips the spin at each specified site and multiplies
		* the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @return An Operator object that applies the sigma_x operator to the specified sites in the lattice.
		*/
		template <typename _T>
		Operators::Operator<_T, uint, uint> sig_x_c(size_t _Ns)
		{
			_COR<_T> fun_ 		= [_Ns](u64 state, uint _s1, uint _s2) { return sig_x<_T>(state, _Ns, {_s1, _s2}); };
			_COR_V<_T> funV_ 	= [_Ns](_OP_V_T_CR state, uint _s1, uint _s2) { return sig_x<_T>(state, _Ns, {_s1, _s2}); };

			Operator<_T, uint, uint> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SX);
			_op.setNameS("Sx/C");
			return _op;
		}

		// #############################################################################################

		// ######################################## SIGMA Y ############################################

		// #############################################################################################

		/**
		* @brief multiplication of sigma_yi | state >
		* @param base_vec the base vector to be acted on. This is given by the copy.
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at. The order of the sites matters!
		* @returns the pair of the new state and the value of the operator
		*/
		template <typename _T>
		std::pair<u64, _T> sig_y(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			cpx _val = 1.0;
			for (auto const& site : sites)
			{
				_val		*=	Binary::check(base_vec, _Ns - 1 - site) ? I * Operators::_SPIN : -I * Operators::_SPIN;
				base_vec	=	flip(base_vec, _Ns - 1 - site);
			}
			return std::make_pair(base_vec, algebra::cast<_T>(_val));
		}

		/**
		* @brief Applies the sigma_y (Pauli-Y) operator to a given base vector.
		*
		* This function takes a base vector and a list of sites, and applies the sigma_x
		* operator to each specified site. The sigma_x operator flips the spin at the given
		* site and multiplies the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The input base vector on which the sigma_x operator will be applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_x operator will be applied.
		* @return A pair consisting of the modified base vector and the resulting value after
		*         applying the sigma_x operator to the specified sites.
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_y(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			cpx _val = 1.0;
			_OP_V_T _base_vec = base_vec;
			for (auto const& site : sites)
			{
				_val *= Binary::check(_base_vec, site) ? I * Operators::_SPIN : -I * Operators::_SPIN;
				flip(_base_vec, site, 0, Operators::_SPIN);
			}
			return std::make_pair(_base_vec, algebra::cast<_T>(_val));
		}

		/**
		* @brief Applies the sigma_y operator to a given base vector.
		*
		* This function applies the sigma_y (Pauli-Y) operator to a given base vector
		* at specified sites and returns the resulting vector and a value.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma_x operator is applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_x operator is applied.
		* @param _ref A reference vector that will be modified to store the result (in the process).
		* @return A pair containing the modified vector and the resulting value.
		*/
		template <typename _T>
		_T sig_y(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref)
		{
			cpx _val = 1.0;
			if (_ref.size() != base_vec.size())
				_ref = base_vec;

			for (auto const& site : sites)
			{
				_val *= Binary::check(_ref, site) ? I * Operators::_SPIN : -I * Operators::_SPIN;
				flip(_ref, site, 0, Operators::_SPIN);
			}
			return algebra::cast<_T>(_val);
		}

		// template instantiation
		OP_TMPLT_INST_PAIR(sig_y,, (u64, size_t, const v_1d<uint>&), u64);
		OP_TMPLT_INST_PAIR(sig_y,, (_OP_V_T_CR, size_t, const v_1d<uint>&), _OP_V_T);
		template double sig_y<double>(_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&);
		template std::complex<double> sig_y<std::complex<double>>(_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&);

		/**
		* @brief This function creates an operator that applies the sigma_y operator to a specific site
		* in the lattice. The sigma_y operator flips the spin at the given site and multiplies
		* the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @param _part The site index where the sigma_y operator will be applied.
		* @return An Operator object that applies the sigma_y operator to the specified site.
		*/
		template <typename _T>
		Operators::Operator<_T> sig_y(size_t _Ns, size_t _part)
		{
			_GLB<_T> fun_ = [_Ns, _part](u64 state) { return sig_y<_T>(state, _Ns, { (uint)_part }); };
			_GLB_V<_T> funV_ = [_Ns, _part](_OP_V_T_CR state) { return sig_y<_T>(state, _Ns, { (uint)_part }); };
			u64 _acts = 1ULL << (_Ns - 1 - _part);

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SY);
			_op.setActOn(_acts);
			_op.setNameS("Sy/" + std::to_string(_part));
			return _op;
		}

		/**
		* @brief Applies the sigma_y (Pauli-Y) operator to a given set of sites in the lattice.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_y operator will be applied. This acts as a multiplication of the operators - Pauli strings.
		* @returns The operator acting on the specified sites.
		*/
		template <typename _T>
		Operators::Operator<_T> sig_y(size_t _Ns, const v_1d<uint>& sites)
		{
			// create the function
			_GLB<_T> fun_		= [_Ns, sites](u64 state) { return sig_y<_T>(state, _Ns, sites); };
			_GLB_V<_T> funV_	= [_Ns, sites](_OP_V_T_CR state) { return sig_y<_T>(state, _Ns, sites); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 0;
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1
			for(auto _part : sites)
				_acts |= 1 << (_Ns - 1 - _part);

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SY);
			_op.setActOn(_acts);

			// set the name
			std::string _name = "Sy/";
			for (auto _part : sites)
				_name += std::to_string(_part) + "-";
			_name.pop_back(); // remove the last comma
			_op.setNameS(_name);
			
			return _op;
		}

		/**
		* @brief Creates a global sigma_y (Pauli-Y) operator for the entire lattice.
		*
		* This function constructs an operator that applies the sigma_y operator to all sites
		* in the lattice. The sigma_y operator flips the spin at each site and multiplies
		* the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @return An Operator object that applies the sigma_y operator to all sites in the lattice.
		*/
		template <typename _T>
		Operators::Operator<_T> sig_y(size_t _Ns)
		{
			// set the vector of sites
			v_1d<uint> _sites = Vectors::vecAtoB<uint>(_Ns);
			
			// create the function
			_GLB<_T> fun_ = [_Ns, _sites](u64 state) { return sig_y<_T>(state, _Ns, _sites); };
			_GLB_V<_T> funV_ = [_Ns, _sites](_OP_V_T_CR state) { return sig_y<_T>(state, _Ns, _sites); };

			// save on which elements the operator acts (for the sake of correctness)
			u64 _acts = (1ULL << _Ns) - 1; // set the bitmask on the state, covering all sites

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SY);
			_op.setActOn(_acts);

			// set the name
			std::string _name = "Sy/1.L.1";
			_op.setNameS(_name);

			return _op;
		}

		/**
		* @brief Creates a local sigma_y (Pauli-Y) operator for a specific site in the lattice.
		*
		* This function constructs an operator that applies the sigma_y operator to a specific site
		* in the lattice. The sigma_y operator flips the spin at the given site and multiplies
		* the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @return An Operator object that applies the sigma_y operator to a specific site in the lattice.
		*/
		template <typename _T>
		Operators::Operator<_T, uint> sig_y_l(size_t _Ns)
		{
			_LOC<_T> fun_ 	= [_Ns](u64 state, uint _part) { return sig_y<_T>(state, _Ns, { _part }); };
			_LOC_V<_T> funV_ 	= [_Ns](_OP_V_T_CR state, uint _part) { return sig_y<_T>(state, _Ns, { _part }); };

			// set the operator
			Operator<_T, uint> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SY);
			_op.setNameS("Sy/L");
			return _op;
		}

		/**
		* @brief Creates a correlation sigma_y (Pauli-Y) operator for a set of sites in the lattice.
		*
		* This function constructs an operator that applies the sigma_y operator to a set of sites
		* in the lattice. The sigma_y operator flips the spin at each specified site and multiplies
		* the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @return An Operator object that applies the sigma_y operator to the specified sites in the lattice.
		*/
		template <typename _T>
		Operators::Operator<_T, uint, uint> sig_y_c(size_t _Ns)
		{
			_COR<_T> fun_ 		= [_Ns](u64 state, uint _s1, uint _s2) { return sig_y<_T>(state, _Ns, {_s1, _s2}); };
			_COR_V<_T> funV_ 	= [_Ns](_OP_V_T_CR state, uint _s1, uint _s2) { return sig_y<_T>(state, _Ns, {_s1, _s2}); };

			Operator<_T, uint, uint> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SY);
			_op.setNameS("Sy/C");
			return _op;
		}

		// ############################################################################################# 

		// ######################################## SIGMA Z ############################################

		// #############################################################################################
		
		/**
		* @brief Applies the sigma_z (Pauli-Z) operator to a given base vector.
		*
		* This function takes a base vector and a list of sites, and applies the sigma_z
		* operator to each specified site. The sigma_z operator measures the spin at the given
		* site and multiplies the resulting value by a constant factor.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The input base vector on which the sigma_z operator will be applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_z operator will be applied.
		* @return A pair consisting of the base vector and the resulting value after
		*         applying the sigma_z operator to the specified sites.
		*/
		template <typename _T>
		std::pair<u64, _T> sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			for (auto const& site : sites)
			{
				_val *= Binary::check(base_vec, _Ns - 1 - site) ? Operators::_SPIN : -Operators::_SPIN;
			}
			return std::make_pair(base_vec, _val);
		}

		/**
		* @brief Computes the Z-component of the spin operator for a given base vector.
		*
		* This function calculates the Z-component of the spin operator (σ_z) for a given base vector.
		* It iterates over the specified sites and multiplies the spin value based on the binary check
		* of the base vector at each site.
		*
		* @tparam _T The type of the spin value.
		* @param base_vec The base vector representing the quantum state.
		* @param _Ns The number of sites (not used in the function).
		* @param sites A vector of site indices where the spin operator is applied.
		* @return A pair consisting of the original base vector and the computed Z-component value.
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val 			= 1.0;
			for (auto const& site : sites)
				_val *= Binary::check(base_vec, site) ? Operators::_SPIN : -Operators::_SPIN;

			return std::make_pair(base_vec, _val);
		}


		/**
		* @brief Applies the sigma_z operator to a given base vector.
		*
		* This function calculates the product of the sigma_z operator applied to the specified sites
		* of the base vector. The result is a pair consisting of the original base vector and the 
		* calculated value.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma_z operator is applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector containing the indices of the sites where the sigma_z operator is applied.
		* @param _ref A reference vector used to check the binary state of each site (unnecessary parameter).
		* @return A pair consisting of the original base vector and the calculated value after applying the sigma_z operator.
		*/
		template <typename _T>
		_T sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref)
		{
			_T _val = 1.0;
			for (auto const& site : sites)
				_val *= Binary::check(_ref, site) ? Operators::_SPIN : -Operators::_SPIN;
			return _val;
		}

		OP_TMPLT_INST_PAIR(sig_z,, (u64, size_t, const v_1d<uint>&), u64);
		OP_TMPLT_INST_PAIR(sig_z,, (_OP_V_T_CR, size_t, const v_1d<uint>&), _OP_V_T);
		template double sig_z(_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&);
		template cpx sig_z(_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&);

		/**
		* @brief Creates a sigma-z (Pauli-Z) operator for a quantum system.
		* 
		* This function generates a sigma-z operator that acts on a specific particle in a quantum system.
		* The operator is represented as a functor that can be applied to quantum states.
		* 
		* @tparam _T The data type used for the operator's elements (e.g., float, double).
		* @param _Ns The total number of particles (or qubits) in the quantum system.
		* @param _part The index of the particle (or qubit) on which the sigma-z operator acts.
		* @return Operators::Operator<_T> The constructed sigma-z operator.
		* 
		* The function creates two functors:
		* - `fun_`: A functor that applies the sigma-z operator to a single quantum state.
		* - `funV_`: A functor that applies the sigma-z operator to a vector of quantum states.
		* 
		* The function also sets a bitmask (`_acts`) to indicate which particle the operator acts on.
		* The bitmask is used to ensure the correctness of the operator's application.
		* 
		* The constructed operator is then initialized with the following parameters:
		* - The total number of particles (`_Ns`).
		* - The scaling factor (set to 1.0).
		* - The functors (`fun_` and `funV_`).
		* - The symmetry generator type (`SymGenerators::SZ`).
		* 
		* Finally, the function sets the bitmask on the operator and returns it.
		*/
		template <typename _T>
		Operators::Operator<_T> sig_z(size_t _Ns, size_t _part)
		{
			// create the function
			_GLB<_T> fun_		= [_Ns, _part](u64 state) { return sig_z<_T>(state, _Ns, { (uint)_part }); };
			_GLB_V<_T> funV_ 	= [_Ns, _part](_OP_V_T_CR state) { return sig_z<_T>(state, _Ns, { (uint)_part }); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 0;
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1
			_acts |= 1 << (_Ns - 1 - _part);

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SZ);
			_op.setActOn(_acts);
			_op.setNameS("Sz/" + STR(_part));
			_op.setModifiesState(false);
			return _op;
		}

		/**
		* @brief Creates a sigma_z (Pauli-Z) operator for a given set of sites in the lattice.
		*
		* This function constructs an operator that applies the sigma_z operator to a set of sites
		* in the lattice. The sigma_z operator measures the spin at each specified site and multiplies
		* the resulting value by a constant factor.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_z operator will be applied.
		* @note The name of the operator is set to "Sz_m" - is appended to the name of the operator as we multiply the operators.
		* @return An Operator object that applies the sigma_z operator to the specified sites in the lattice.
		*/
		template <typename _T>
		Operators::Operator<_T> sig_z(size_t _Ns, const v_1d<uint>& sites)
		{
			// create the function
			_GLB<_T> fun_ = [_Ns, sites](u64 state) { return sig_z<_T>(state, _Ns, sites); };
			_GLB_V<_T> funV_ = [_Ns, sites](_OP_V_T_CR state) { return sig_z<_T>(state, _Ns, sites); };

			// save on which elements the operator acts (for the sake of correctness)
			u64 _acts = 0;
			for (auto _part : sites)
				_acts |= 1ULL << (_Ns - 1 - _part);

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SZ);
			_op.setActOn(_acts);

			std::string _name = "Sz/";
			for (auto _part : sites)
				_name += STR(_part) + "-";
			_name.pop_back();

			_op.setNameS(_name);
			_op.setModifiesState(false);
			return _op;
		}

		/**
		* @brief Constructs a sigma_z (Pauli-Z) operator for a quantum system of size _Ns.
		*
		* This function creates a sigma_z operator which acts on a quantum state of size _Ns.
		* The operator is represented using a lambda function that applies the sigma_z operation
		* on a given state. The operator acts on all sites of the quantum system.
		*
		* @tparam _T The data type of the operator elements.
		* @param _Ns The number of sites in the quantum system.
		* @return Operators::Operator<_T> The constructed sigma_z operator.
		*
		* The function performs the following steps:
		* 1. Sets the vector of sites from 0 to _Ns-1.
		* 2. Creates lambda functions to apply the sigma_z operation on a given state.
		* 3. Sets the bitmask to indicate that the operator acts on all sites.
		* 4. Constructs and returns the sigma_z operator.
		*/
		template <typename _T>
		Operators::Operator<_T> sig_z(size_t _Ns)
		{
			// set the vector of sites
			v_1d<uint> _sites		= Vectors::vecAtoB<uint>(_Ns);
			// create the function
			_GLB<_T> fun_		= [_Ns, _sites](u64 state) { return sig_z<_T>(state, _Ns, _sites); };
			_GLB_V<_T> funV_ 	= [_Ns, _sites](_OP_V_T_CR state) { return sig_z<_T>(state, _Ns, _sites); };

			// save on which elements the operator acts (for the sake of the correctness)
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1

			// take all of them!
			u64 _acts				= (ULLPOW(_Ns)) - 1;

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SZ);

			std::string _name = "Sz/1.L.1";
			_op.setNameS(_name);
			_op.setActOn(_acts);
			_op.setModifiesState(false);
			return _op;
		}

		/**
		* @brief Generates a sig_z operator for a given system size.
		*
		* This function creates a sig_z operator for a quantum system of size _Ns.
		* The sig_z operator is used in quantum mechanics to represent the Pauli-Z
		* matrix, which is a fundamental operator in the study of quantum systems.
		*
		* @tparam _T The data type of the operator elements.
		* @param _Ns The size of the quantum system.
		* @return An Operator object representing the sig_z operator for the given system size.
		*
		* The function uses lambda expressions to define the operator's action on
		* quantum states. The operator is then constructed with these lambda functions
		* and a symmetry generator (SymGenerators::SZ).
		*
		* Example usage:
		* @code
		* size_t system_size = 4;
		* auto sig_z_operator = Operators::sig_z_l<double>(system_size);
		* @endcode
		*/
		template <typename _T>
		Operators::Operator<_T, uint> sig_z_l(size_t _Ns)
		{
			_LOC<_T> fun_ 		= [_Ns](u64 state, uint _part) { return sig_z<_T>(state, _Ns, { _part }); };
			_LOC_V<_T> funV_ 	= [_Ns](_OP_V_T_CR state, uint _part) { return sig_z<_T>(state, _Ns, { _part }); };

			// set the operator
			Operator<_T, uint> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SZ);
			_op.setNameS("Sz/L");
			_op.setModifiesState(false);
			return _op;
		}

		/**
		* @brief Generates a sig_z operator for a given system size. The operator acts on a pair of sites.
		
		* This function constructs an operator that applies the sigma_z operator to a set of sites
		* in the lattice. The sigma_z operator measures the spin at each specified site and multiplies
		* the resulting value by a constant factor.
		*
		* @tparam _T The type of the value to be returned.
		* @param _Ns The size of the system (number of sites).
		* @return An Operator object that applies the sigma_z operator to the specified sites in the lattice.
		*/
		template <typename _T>
		Operators::Operator<_T, uint, uint> sig_z_c(size_t _Ns)
		{
			_COR<_T> fun_ 		= [_Ns](u64 state, uint _s1, uint _s2) { return sig_z<_T>(state, _Ns, {_s1, _s2}); };
			_COR_V<_T> funV_ 	= [_Ns](_OP_V_T_CR state, uint _s1, uint _s2) { return sig_z<_T>(state, _Ns, {_s1, _s2}); };

			Operator<_T, uint, uint> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SZ);
			_op.setNameS("Sz/C");
			_op.setModifiesState(false);
			return _op;
		}
		
		// ############################################################################################# 

		// ######################################## SIGMA P ############################################

		// #############################################################################################

		/**
		* @brief Applies the sigma_p (raising) operator to a given base vector.
		*
		* This function takes a base vector and a list of sites, and applies the sigma_p
		* operator to each specified site. The sigma_p operator raises the spin at the given
		* site and multiplies the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The input base vector on which the sigma_p operator will be applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_p operator will be applied.
		* @return A pair consisting of the modified base vector and the resulting value after
		*         applying the sigma_p operator to the specified sites.
		*/
		template <typename _T>
		std::pair<u64, _T> sig_p(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			for (auto const& site : sites)
			{
				if (checkBit(base_vec, _Ns - 1 - site))
				{
					_val = 0.0;
					break;
				}
				base_vec = flip(base_vec, _Ns - 1 - site);
				_val *= Operators::_SPIN;
			}
			return std::make_pair(base_vec, _val);
		}

		/**
		* @brief Applies the sigma plus (σ⁺) operator to a given base vector.
		*
		* This function applies the sigma plus (σ⁺) operator to a given base vector `base_vec`
		* at the specified `sites`. The sigma plus operator flips the spin at the given sites
		* and multiplies the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma plus operator is applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma plus operator is applied.
		* @return A pair consisting of the modified base vector and the resulting value.
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_p(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			_OP_V_T _base_vec = base_vec;
			for (auto const& site : sites)
			{
				if (checkBit(_base_vec, site))
				{
					_val = 0.0;
					break;
				}
				flip(_base_vec, site, 0, Operators::_SPIN);
				_val *= Operators::_SPIN;
			}
			return std::make_pair(_base_vec, _val);
		}

		template <typename _T>
		std::pair<_OP_V_T, _T> sig_p(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref)
		{
			_T _val = 1.0;
			if (_ref.size() != base_vec.size())
				_ref = base_vec;

			for (auto const& site : sites)
			{
				if (checkBit(_ref, site))
				{
					_val = 0.0;
					break;
				}
				flip(_ref, site, 0, Operators::_SPIN);
				_val *= Operators::_SPIN;
			}
			return std::make_pair(_ref, _val);
		}

		// template instantiation
		OP_TMPLT_INST_PAIR(sig_p,, (u64, size_t, const v_1d<uint>&), u64);
		OP_TMPLT_INST_PAIR(sig_p,, (_OP_V_T_CR, size_t, const v_1d<uint>&), _OP_V_T);
		OP_TMPLT_INST_PAIR(sig_p,, (_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&), _OP_V_T);

		// #############################################################################################

		// ######################################## SIGMA M ############################################

		// #############################################################################################

		/**
		* @brief Applies the sigma_m (lowering) operator to a given base vector.
		*
		* This function takes a base vector and a list of sites, and applies the sigma_m
		* operator to each specified site. The sigma_m operator lowers the spin at the given
		* site and multiplies the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The input base vector on which the sigma_m operator will be applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_m operator will be applied.
		* @return A pair consisting of the modified base vector and the resulting value after
		*         applying the sigma_m operator to the specified sites.
		*/
		template <typename _T>
		std::pair<u64, _T> sig_m(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			for (auto const& site : sites)
			{
				if (!checkBit(base_vec, _Ns - 1 - site))
				{
					_val = 0.0;
					break;
				}
				base_vec = flip(base_vec, _Ns - 1 - site);
				_val *= Operators::_SPIN;
			}
			return std::make_pair(base_vec, _val);
		}

		/**
		* @brief Applies the sigma_m (lowering) operator to a given base vector.
		*
		* This function applies the sigma_m (lowering) operator to a given base vector `base_vec`
		* at the specified `sites`. The sigma_m operator flips the spin at the given sites
		* and multiplies the resulting vector by a constant value.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma_m operator is applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma_m operator is applied.
		* @return A pair consisting of the modified base vector and the resulting value.
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_m(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			_OP_V_T _base_vec = base_vec;
			for (auto const& site : sites)
			{
				if (!checkBit(_base_vec, site))
				{
					_val = 0.0;
					break;
				}
				flip(_base_vec, site, 0, Operators::_SPIN);
				_val *= Operators::_SPIN;
			}
			return std::make_pair(_base_vec, _val);
		}

		template <typename _T>
		std::pair<_OP_V_T, _T> sig_m(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref)
		{
			_T _val = 1.0;
			if (_ref.size() != base_vec.size())
				_ref = base_vec;

			for (auto const& site : sites)
			{
				if (!checkBit(_ref, site))
				{
					_val = 0.0;
					break;
				}
				flip(_ref, site, 0, Operators::_SPIN);
				_val *= Operators::_SPIN;
			}
			return std::make_pair(_ref, _val);
		}

		// template instantiation
		OP_TMPLT_INST_PAIR(sig_m,, (u64, size_t, const v_1d<uint>&), u64);
		OP_TMPLT_INST_PAIR(sig_m,, (_OP_V_T_CR, size_t, const v_1d<uint>&), _OP_V_T);
		OP_TMPLT_INST_PAIR(sig_m,, (_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&), _OP_V_T);

		// #############################################################################################

		// ######################################## SIGMA PM ###########################################

		// #############################################################################################

		/**
		* @brief Applies the product of sigma plus (σ⁺) and sigma minus (σ⁻) operators to a given base vector.
		*
		* This function applies the product of sigma plus (σ⁺) and sigma minus (σ⁻) operators to a given base vector `base_vec`
		* at the specified `sites`. The sigma plus operator raises the spin at the given sites, and the sigma minus operator
		* lowers the spin at the given sites. The order of the sites matters.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma plus and sigma minus operators are applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma plus and sigma minus operators are applied.
		* @return A pair consisting of the modified base vector and the resulting value.
		*/
		template <typename _T>
		std::pair<u64, _T> sig_pm(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			for (size_t i = 0; i < sites.size(); ++i)
			{
				if (i % 2 == 0)
				{
					if (checkBit(base_vec, _Ns - 1 - sites[i]))
					{
						_val = 0.0;
						break;
					}
					base_vec = flip(base_vec, _Ns - 1 - sites[i]);
					_val *= Operators::_SPIN;
				}
				else
				{
					if (!checkBit(base_vec, _Ns - 1 - sites[i]))
					{
						_val = 0.0;
						break;
					}
					base_vec = flip(base_vec, _Ns - 1 - sites[i]);
					_val *= Operators::_SPIN;
				}
			}
			return std::make_pair(base_vec, _val);
		}

		/**
		* @brief Applies the product of sigma plus (σ⁺) and sigma minus (σ⁻) operators to a given base vector.
		*
		* This function applies the product of sigma plus (σ⁺) and sigma minus (σ⁻) operators to a given base vector `base_vec`
		* at the specified `sites`. The sigma plus operator raises the spin at the given sites, and the sigma minus operator
		* lowers the spin at the given sites. The order of the sites matters.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma plus and sigma minus operators are applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma plus and sigma minus operators are applied.
		* @return A pair consisting of the modified base vector and the resulting value.
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_pm(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			_OP_V_T _base_vec = base_vec;
			for (size_t i = 0; i < sites.size(); ++i)
			{
				if (i % 2 == 0)
				{
					if (checkBit(_base_vec, sites[i]))
					{
						_val = 0.0;
						break;
					}
					flip(_base_vec, sites[i], 0, Operators::_SPIN);
					_val *= Operators::_SPIN;
				}
				else
				{
					if (!checkBit(_base_vec, sites[i]))
					{
						_val = 0.0;
						break;
					}
					flip(_base_vec, sites[i], 0, Operators::_SPIN);
					_val *= Operators::_SPIN;
				}
			}
			return std::make_pair(_base_vec, _val);
		}

		/**
		* @brief Applies the product of sigma plus (σ⁺) and sigma minus (σ⁻) operators to a given base vector.
		*
		* This function applies the product of sigma plus (σ⁺) and sigma minus (σ⁻) operators to a given base vector `base_vec`
		* at the specified `sites`. The sigma plus operator raises the spin at the given sites, and the sigma minus operator
		* lowers the spin at the given sites. The order of the sites matters.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma plus and sigma minus operators are applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma plus and sigma minus operators are applied.
		* @param _ref A reference vector used to store the result.
		* @return A pair consisting of the modified base vector and the resulting value.
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_pm(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref)
		{
			_T _val = 1.0;
			if (_ref.size() != base_vec.size())
				_ref = base_vec;

			for (size_t i = 0; i < sites.size(); ++i)
			{
				if (i % 2 == 0)
				{
					if (checkBit(_ref, sites[i]))
					{
						_val = 0.0;
						break;
					}
					flip(_ref, sites[i], 0, Operators::_SPIN);
					_val *= Operators::_SPIN;
				}
				else
				{
					if (!checkBit(_ref, sites[i]))
					{
						_val = 0.0;
						break;
					}
					flip(_ref, sites[i], 0, Operators::_SPIN);
					_val *= Operators::_SPIN;
				}
			}
			return std::make_pair(_ref, _val);
		}

		/**
		* @brief Applies the product of sigma minus (σ⁻) and sigma plus (σ⁺) operators to a given base vector.
		*
		* This function applies the product of sigma minus (σ⁻) and sigma plus (σ⁺) operators to a given base vector `base_vec`
		* at the specified `sites`. The sigma minus operator lowers the spin at the given sites, and the sigma plus operator
		* raises the spin at the given sites. The order of the sites matters.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma minus and sigma plus operators are applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma minus and sigma plus operators are applied.
		* @return A pair consisting of the modified base vector and the resulting value.
		*/
		template <typename _T>
		std::pair<u64, _T> sig_mp(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			for (size_t i = 0; i < sites.size(); ++i)
			{
				if (i % 2 == 0)
				{
					if (!checkBit(base_vec, _Ns - 1 - sites[i]))
					{
						_val = 0.0;
						break;
					}
					base_vec = flip(base_vec, _Ns - 1 - sites[i]);
					_val *= Operators::_SPIN;
				}
				else
				{
					if (checkBit(base_vec, _Ns - 1 - sites[i]))
					{
						_val = 0.0;
						break;
					}
					base_vec = flip(base_vec, _Ns - 1 - sites[i]);
					_val *= Operators::_SPIN;
				}
			}
			return std::make_pair(base_vec, _val);
		}

		/**
		* @brief Applies the product of sigma minus (σ⁻) and sigma plus (σ⁺) operators to a given base vector.
		*
		* This function applies the product of sigma minus (σ⁻) and sigma plus (σ⁺) operators to a given base vector `base_vec`
		* at the specified `sites`. The sigma minus operator lowers the spin at the given sites, and the sigma plus operator
		* raises the spin at the given sites. The order of the sites matters.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma minus and sigma plus operators are applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma minus and sigma plus operators are applied.
		* @return A pair consisting of the modified base vector and the resulting value.
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_mp(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			_OP_V_T _base_vec = base_vec;
			for (size_t i = 0; i < sites.size(); ++i)
			{
				if (i % 2 == 0)
				{
					if (!checkBit(_base_vec, sites[i]))
					{
						_val = 0.0;
						break;
					}
					flip(_base_vec, sites[i], 0, Operators::_SPIN);
					_val *= Operators::_SPIN;
				}
				else
				{
					if (checkBit(_base_vec, sites[i]))
					{
						_val = 0.0;
						break;
					}
					flip(_base_vec, sites[i], 0, Operators::_SPIN);
					_val *= Operators::_SPIN;
				}
			}
			return std::make_pair(_base_vec, _val);
		}

		/**
		* @brief Applies the product of sigma minus (σ⁻) and sigma plus (σ⁺) operators to a given base vector.
		*
		* This function applies the product of sigma minus (σ⁻) and sigma plus (σ⁺) operators to a given base vector `base_vec`
		* at the specified `sites`. The sigma minus operator lowers the spin at the given sites, and the sigma plus operator
		* raises the spin at the given sites. The order of the sites matters.
		*
		* @tparam _T The type of the value to be returned.
		* @param base_vec The base vector to which the sigma minus and sigma plus operators are applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the sigma minus and sigma plus operators are applied.
		* @param _ref A reference vector used to store the result.
		* @return A pair consisting of the modified base vector and the resulting value.
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_mp(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref)
		{
			_T _val = 1.0;
			if (_ref.size() != base_vec.size())
				_ref = base_vec;

			for (size_t i = 0; i < sites.size(); ++i)
			{
				if (i % 2 == 0)
				{
					if (!checkBit(_ref, sites[i]))
					{
						_val = 0.0;
						break;
					}
					flip(_ref, sites[i], 0, Operators::_SPIN);
					_val *= Operators::_SPIN;
				}
				else
				{
					if (checkBit(_ref, sites[i]))
					{
						_val = 0.0;
						break;
					}
					flip(_ref, sites[i], 0, Operators::_SPIN);
					_val *= Operators::_SPIN;
				}
			}
			return std::make_pair(_ref, _val);
		}

		// template instantiation
		OP_TMPLT_INST_PAIR(sig_pm,, (u64, size_t, const v_1d<uint>&), u64);
		OP_TMPLT_INST_PAIR(sig_pm,, (_OP_V_T_CR, size_t, const v_1d<uint>&), _OP_V_T);
		OP_TMPLT_INST_PAIR(sig_pm,, (_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&), _OP_V_T);

		OP_TMPLT_INST_PAIR(sig_mp,, (u64, size_t, const v_1d<uint>&), u64);
		OP_TMPLT_INST_PAIR(sig_mp,, (_OP_V_T_CR, size_t, const v_1d<uint>&), _OP_V_T);
		OP_TMPLT_INST_PAIR(sig_mp,, (_OP_V_T_CR, size_t, const v_1d<uint>&, _OP_V_T&), _OP_V_T);

		// #############################################################################################

		/**
		* @brief Applies a series of quantum spin operations on a base vector.
		*
		* This function template applies a series of quantum spin operations (X and Z) on a given base vector.
		* The operations are applied based on the indices provided in the `sites` vector. The function returns
		* a pair consisting of the modified base vector and the resulting value after applying the operations.
		*
		* @tparam _T The type of the value to be returned (e.g., double, std::complex<double>).
		* @param base_vec The initial base vector on which the operations will be applied.
		* @param _Ns The size of the system (number of sites).
		* @param sites A vector of site indices where the operations will be applied.
		* @return A pair consisting of the modified base vector and the resulting value.
		*
		* This function is important for simulating quantum systems on a honeycomb lattice, where specific
		* spin operations need to be applied to certain sites in a controlled manner. The alternation between
		* X and Z operations allows for the simulation of complex quantum behaviors and interactions.
		*/
		namespace Flux
		{
			static constexpr int _X_BOND_NEI [[maybe_unused]] = 0;
			static constexpr int _Y_BOND_NEI = 2;
			static constexpr int _Z_BOND_NEI = 1;					// Z bond represents the diagonal bond

			/**
			* @brief Applies a series of quantum spin operations (X and Z) on a base vector.
			*
			* This function applies a series of quantum spin operations (X and Z) on a given base vector.
			* The operations are applied based on the indices provided in the `sites` vector. The function returns
			* a pair consisting of the modified base vector and the resulting value after applying the operations.
			*
			* @tparam _T The type of the value to be returned (e.g., double, std::complex<double>).
			* @param base_vec The initial base vector on which the operations will be applied.
			* @param _Ns The size of the system (number of sites).
			* @param sites A vector of site indices where the operations will be applied.
			* @return A pair consisting of the modified base vector and the resulting value.
			*/
			template <typename _T>
			std::pair<u64, _T> sig_f(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
			{
				cpx _val = 1.0;
				for (size_t i = 0; i < sites.size(); ++i)
				{
					// Y bond
					if (i % 3 == _Y_BOND_NEI)
					{
						_val 		*= Binary::check(base_vec, static_cast<int>(_Ns - 1 - sites[i])) ? I * Operators::_SPIN : -I * Operators::_SPIN;
						base_vec 	= flip(base_vec, static_cast<int>(_Ns - 1 - sites[i]));
					}
					// Z bond
					else if (i % 3 == _Z_BOND_NEI)
					{
						_val 		*= Binary::check(base_vec, static_cast<int>(_Ns - 1 - sites[i])) ? Operators::_SPIN : -Operators::_SPIN;
					}
					// X bond
					else
					{
						base_vec 	= flip(base_vec, _Ns - 1 - sites[i]);
						_val 		*= Operators::_SPIN;
					}
				}
				return std::make_pair(base_vec, algebra::cast<_T>(_val));
			}

			/**
			* @brief Applies a series of quantum spin operations (X and Z) on a base vector.
			*
			* This function applies a series of quantum spin operations (X and Z) on a given base vector.
			* The operations are applied based on the indices provided in the `sites` vector. The function returns
			* a pair consisting of the modified base vector and the resulting value after applying the operations.
			*
			* @tparam _T The type of the value to be returned (e.g., double, std::complex<double>).
			* @param base_vec The initial base vector on which the operations will be applied.
			* @param _Ns The size of the system (number of sites).
			* @param sites A vector of site indices where the operations will be applied.
			* @return A pair consisting of the modified base vector and the resulting value.
			*/
			template <typename _T>
			std::pair<_OP_V_T, _T> sig_f(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
			{
				cpx _val 			= 1.0;
				_OP_V_T _base_vec 	= base_vec;
				for (size_t i = 0; i < sites.size(); ++i)
				{
					// Y bond 
					if (i % 3 == _Y_BOND_NEI)
					{
						_val 		*= Binary::check(_base_vec, static_cast<int>(sites[i])) ? I * Operators::_SPIN : -I * Operators::_SPIN;
						flip(_base_vec, sites[i], 0, Operators::_SPIN);
					}
					// Z bond
					if (i % 3 == _Z_BOND_NEI)
					{
						_val 		*= Binary::check(base_vec, static_cast<int>(sites[i])) ? Operators::_SPIN : -Operators::_SPIN;
					}
					// X bond
					else
					{
						_val 		*= Operators::_SPIN;
						flip(_base_vec, sites[i], 0, Operators::_SPIN);
					}
				}
				return std::make_pair(base_vec, algebra::cast<_T>(_val));
			}

			/**
			* @brief Applies a series of quantum spin operations (X and Z) on a base vector with a reference vector.
			*
			* This function applies a series of quantum spin operations (X and Z) on a given base vector,
			* using a reference vector to store intermediate results. The operations are applied based on the indices
			* provided in the `sites` vector. The function returns a pair consisting of the modified reference vector
			* and the resulting value after applying the operations.
			*
			* @tparam _T The type of the value to be returned (e.g., double, std::complex<double>).
			* @param base_vec The initial base vector on which the operations will be applied.
			* @param _Ns The size of the system (number of sites).
			* @param sites A vector of site indices where the operations will be applied.
			* @param _ref A reference vector to store intermediate results.
			* @return A pair consisting of the modified reference vector and the resulting value.
			*/
			template <typename _T>
			_T sig_f(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref)
			{
				cpx _val = 1.0;
				if (_ref.size() != base_vec.size())
					_ref = base_vec;

				for (size_t i = 0; i < sites.size(); ++i)
				{
					// Y bond 
					if (i % 3 == _Y_BOND_NEI)
					{
						_val 		*= Binary::check(_ref, sites[i]) ? I * Operators::_SPIN : -I * Operators::_SPIN;
						flip(_ref, sites[i], 0, Operators::_SPIN);
					}
					// Z bond 
					if (i % 3 == _Z_BOND_NEI)
					{
						_val 		*= Binary::check(base_vec, sites[i]) ? Operators::_SPIN : -Operators::_SPIN;
					}
					// X bond 
					else
					{
						_val 		*= Operators::_SPIN;
						flip(_ref, sites[i], 0, Operators::_SPIN);
					}
				}
				return algebra::cast<_T>(_val);
			}

			// template instantiation
			template std::pair<u64, double> sig_f<double>(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
			template std::pair<u64, std::complex<double>> sig_f<std::complex<double>>(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
			template std::pair<_OP_V_T, double> sig_f<double>(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
			template std::pair<_OP_V_T, std::complex<double>> sig_f<std::complex<double>>(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
			template double sig_f<double>(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref);
			template cpx sig_f<std::complex<double>>(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, _OP_V_T& _ref);

			// operator 
			template <typename _T>
			Operators::Operator<_T> sig_f(size_t _Ns, const v_1d<uint>& sites)
			{
				// create the function
				_GLB<_T> fun_ 		= [_Ns, sites](u64 state) 			{ return Flux::sig_f<_T>(state, _Ns, sites); };
				_GLB_V<_T> funV_ 	= [_Ns, sites](_OP_V_T_CR state) 	{ return Flux::sig_f<_T>(state, _Ns, sites); };

				// save on which elements the operator acts (for the sake of the correctness)
				u64 _acts = 0;
				// set the bitmask on the state, remember that this is counted from the left|
				// the first position is leftwise 0, the last is leftwise Ns - 1
				for (auto const& site : sites)
					_acts |= 1 << (_Ns - 1 - site);

				// set the operator
				Operators::Operator<_T> _op(_Ns, 1.0, fun_, funV_);
				_op.setActOn(_acts);

				std::string _sitesname = "";
				for (auto& s : sites)
					_sitesname +=  STR(s) + ",";
				_sitesname.pop_back();

				_op.setNameS("flux_" + _sitesname);
				return _op;
			}

			template Operators::Operator<double> sig_f(size_t _Ns, const v_1d<uint>& sites);
			template Operators::Operator<std::complex<double>> sig_f(size_t _Ns, const v_1d<uint>& sites);
		};

		// #############################################################################################

		namespace RandomSuperposition 
		{

			// #############################################################################################

			/*
			* @brief multiplication of \sum _i c^r_i S_z^i |state> 
			*/
			std::pair<u64, double> sig_z(u64 base_vec, size_t _Ns)
			{
				auto _val = 0.0;
				for (size_t i = 0; i < _Ns; i++)
					_val += superpositions[i] * (Binary::check(base_vec, i) ? Operators::_SPIN : -Operators::_SPIN);
				return std::make_pair(base_vec, _val / std::sqrt(_Ns));
			}

			std::pair<u64, double> sig_z_vanish(u64 base_Vec , size_t _Ns)
			{
				auto _val = 0.0;
				for (size_t i = 0; i < _Ns; i++)
					_val += superpositions[i] / (i + 1) * (Binary::check(base_Vec, i) ? Operators::_SPIN : -Operators::_SPIN) ;
				return std::make_pair(base_Vec, _val / std::sqrt(_Ns));
			}

			// #############################################################################################

			std::pair<_OP_V_T_CR, double> sig_z(_OP_V_T_CR base_vec, size_t _Ns)
			{
				auto _val = 0.0;
				for (size_t i = 0; i < _Ns; i++)
					_val += superpositions[i] * (Binary::check(base_vec, i) ? Operators::_SPIN : -Operators::_SPIN);
				return std::make_pair(base_vec, _val / std::sqrt(_Ns));
			}

			std::pair<_OP_V_T_CR, double> sig_z_vanish(_OP_V_T_CR base_vec, size_t _Ns)
			{
				auto _val = 0.0;
				for (size_t i = 0; i < _Ns; i++)
					_val += superpositions[i] / (i + 1) * (Binary::check(base_vec, i) ? Operators::_SPIN : -Operators::_SPIN);
				return std::make_pair(base_vec, _val / std::sqrt(_Ns));
			}

			// #############################################################################################

			Operators::Operator<double> sig_z(size_t _Ns)
			{
				// create the function
				_OP<double>::GLB fun_ 		= [_Ns](u64 state) 			{ return RandomSuperposition::sig_z(state, _Ns); };
				_OP_V<double>::GLB funV_ 	= [_Ns](_OP_V_T_CR state) 	{ return RandomSuperposition::sig_z(state, _Ns); };

				// save on which elements the operator acts (for the sake of the correctness)
				u64 _acts = 0;
				// |set the bitmask on the state, remember that this is counted from the left|
				// the first position is leftwise 0, the last is leftwise Ns - 1
				for (size_t i = 0; i < _Ns; i++)
					_acts |= 1 << (_Ns - 1 - i);

				// set the operator
				Operator<double> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SZ);
				_op.setActOn(_acts);
				return _op;
			}

			Operators::Operator<double> sig_z_vanish(size_t Ns)
			{
				// create the function
				_OP<double>::GLB fun_ 		= [Ns](u64 state) 			{ return RandomSuperposition::sig_z_vanish(state, Ns); };
				_OP_V<double>::GLB funV_ 	= [Ns](_OP_V_T_CR state) 	{ return RandomSuperposition::sig_z_vanish(state, Ns); };

				// save on which elements the operator acts (for the sake of the correctness)
				u64 _acts = 0;
				// |set the bitmask on the state, remember that this is counted from the left|
				// the first position is leftwise 0, the last is leftwise Ns - 1
				for (size_t i = 0; i < Ns; i++)
					_acts |= 1 << (Ns - 1 - i);

				// set the operator
				Operator<double> _op(Ns, 1.0, fun_, funV_, SymGenerators::SZ);
				_op.setActOn(_acts);
				return _op;
			}

		};

		// #############################################################################################

		/**
		* @brief Calculates the Fourier transform of spin operators for a given momentum q
		* 
		* This function computes the Fourier transform of spin operators (sigma+/sigma-) 
		* for a specific momentum q in a spin chain system. It evaluates the operation 
		* on a basis state and returns both the resulting state and its coefficient.
		* 
		* @tparam _T Template parameter for the coefficient type (typically double or complex<double>)
		* @param base_vec The input basis state represented as a binary number
		* @param _Ns The total number of sites in the system
		* @param sites Vector containing the site indices to apply the operator
		* @param _q The momentum number
		* 
		* @return std::pair<u64, _T> where:
		*         - first: The resulting basis state (unchanged in this case)
		*         - second: The coefficient after applying the operator
		* 
		* @note The function includes phase factors exp(ikr) where k = 2π*q/N
		* @note The result is normalized by 1/sqrt(N) where N is the number of sites
		*/
		template <typename _T>
		std::pair<u64, _T> sig_k(u64 base_vec, size_t _Ns, const v_1d<uint>& sites, double _k)
		{
			_T _val = 1.0;
			for (auto const& site : sites)
			{
				cpx _valin	= 	Binary::check(base_vec, site) ? Operators::_SPIN : -Operators::_SPIN;
				_valin 		*= 	std::exp(std::complex<double>(0.0, 1.0) * _k * (double)site);
				_val 		+= 	algebra::cast<_T>(_valin);
			}
			return std::make_pair(base_vec, _val / std::sqrt(sites.size()));
		}
		// template instantiation
		OP_TMPLT_INST_PAIR(sig_k, , (u64, size_t, const v_1d<uint>&, double), u64);
		// ############################################################################################
		
		/**
		* @brief Calculates the Fourier transform of spin operators for a given quantum state
		* 
		* This function computes the momentum space representation of spin operators
		* by performing a Fourier transform on the real space spin configuration.
		* 
		* @tparam _T Template parameter for return value type (typically double or complex<double>)
		* @param base_vec Input state represented as a bit string where 1/0 corresponds to up/down spins
		* @param _Ns Number of lattice sites/spins in the system
		* @param _q Momentum index q for the Fourier transform
		* @return std::pair<u64,_T> Pair containing:
		*         - First element: The unchanged input state (base_vec)
		*         - Second element: Fourier transformed value normalized by sqrt(N)
		* 
		* The function:
		* 1. Iterates through each site
		* 2. Determines spin value (+/-S) based on bit value
		* 3. Multiplies by phase factor exp(ikx)
		* 4. Accumulates the sum and normalizes by sqrt(N)
		*/
		template <typename _T>
		std::pair<u64, _T> sig_k(u64 base_vec, size_t _Ns, double _k)
		{
			_T _val = 1.0;
			for(size_t site = 0; site < _Ns; ++site)
			{
				cpx _valin	= 	Binary::check(base_vec, site) ? Operators::_SPIN : -Operators::_SPIN;
				_valin 		*= 	std::exp(std::complex<double>(0.0, 1.0) * _k * (double)site);
				_val 		+= 	algebra::cast<_T>(_valin);
			}
			return std::make_pair(base_vec, _val / std::sqrt(_Ns));
		}
		// template instantiation
		OP_TMPLT_INST_PAIR(sig_k, , (u64, size_t, double), u64);

		// #############################################################################################

		/**
		* @brief Calculates quantum spin operator with momentum q
		* 
		* This template function computes the quantum spin operator with momentum q for a given basis vector.
		* It applies the spin operator to specified sites and includes phase factors based on momentum q.
		* 
		* @tparam _T The numeric type used for calculations (typically double or complex<double>)
		* @param base_vec Input basis vector to apply the operator to
		* @param _Ns Total number of sites in the system
		* @param sites Vector containing the site indices to apply the operator
		* @param _q Momentum quantum number
		* 
		* @return std::pair containing:
		*         - first: The resulting basis vector after operator application
		*         - second: The corresponding coefficient/value (_T type)
		* 
		* @note The function normalizes the result by dividing by sqrt(number of sites)
		* @note Uses TWOPI constant and applies exp(ikx) phase factors
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> sig_k(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites, double _k)
		{
			_T _val = 1.0;
			_OP_V_T _base_vec = base_vec;
			for (auto const& site : sites)
			{
				cpx _valin	= 	Binary::check(_base_vec, site) ? Operators::_SPIN : -Operators::_SPIN;
				_valin 		*= 	std::exp(std::complex<double>(0.0, 1.0) * _k * (double)site);
				_val 		+= 	algebra::cast<_T>(_valin);
			}
			return std::make_pair(_base_vec, _val / std::sqrt(sites.size()));
		}
		// template instantiation
		OP_TMPLT_INST_PAIR(sig_k, , (_OP_V_T_CR, size_t, const v_1d<uint>&, double), _OP_V_T)

		// #############################################################################################

		template <typename _T>
		std::pair<_OP_V_T, _T> sig_k(_OP_V_T_CR base_vec, size_t _Ns, double _k)
		{
			_T _val = 1.0;
			_OP_V_T _base_vec = base_vec;
			for(size_t site = 0; site < _Ns; ++site)
			{
				cpx _valin	= 	Binary::check(_base_vec, site) ? Operators::_SPIN : -Operators::_SPIN;
				_valin 		*= 	std::exp(std::complex<double>(0.0, 1.0) * _k * (double)site);
				_val 		+= 	algebra::cast<_T>(_valin);
			}
			return std::make_pair(_base_vec, _val / std::sqrt(_Ns));
		}
		// template instantiation
		OP_TMPLT_INST_PAIR(sig_k, , (_OP_V_T_CR, size_t, double), _OP_V_T)

		// #############################################################################################

		// operator
		template <typename _T>
		Operators::Operator<_T> sig_k(size_t _Ns, const v_1d<uint>& _sites, double _k)
		{
			_GLB<_T> fun_		= [_Ns, _sites, _k](u64 state) { return sig_k<_T>(state, _Ns, _sites, _k); };
			_GLB_V<_T> funV_ 	= [_Ns, _sites, _k](_OP_V_T_CR state) { return sig_k<_T>(state, _Ns, _sites, _k); };
			u64 _acts			= (ULLPOW(_Ns)) - 1;
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
			_op.setActOn(_acts);
			_op.setModifiesState(false);

			// set the name
			const uint _q 		= (uint)std::round(_k * _Ns / TWOPI);
			_op.setNameS("sigk_" + std::to_string(_q));
			return _op;
		}
		// template instantiation
		OP_TMPLT_INST_OPG(sig_k, , (size_t, const v_1d<uint>&, double), double);
		// #############################################################################################
		
		template <typename _T>
		Operators::Operator<_T> sig_k(size_t _Ns, double _k)
		{	
			_GLB<_T> fun_		= [_Ns, _k](u64 state) { return sig_k<_T>(state, _Ns, _k); };
			_GLB_V<_T> funV_ 	= [_Ns, _k](_OP_V_T_CR state) { return sig_k<_T>(state, _Ns, _k); };
			u64 _acts			= (ULLPOW(_Ns)) - 1;
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
			_op.setActOn(_acts);
			_op.setModifiesState(false);

			// set the name
			const uint _q 		= (uint)std::round(_k * _Ns / TWOPI);
			_op.setNameS("sigk_" + std::to_string(_q));
			return _op;
		}
		// template instantiation
		OP_TMPLT_INST_OPG(sig_k, , (size_t, double), double);
		// #############################################################################################

	};
	// sigx 
	OP_TMPLT_INST_OPG(sig_x, SpinOperators::, (size_t, size_t),);
	OP_TMPLT_INST_OPG(sig_x, SpinOperators::, (size_t, const v_1d<uint>&),);
	OP_TMPLT_INST_OPG(sig_x, SpinOperators::, (size_t),);
	OP_TMPLT_INST_OPL(sig_x_l, SpinOperators::, (size_t),);
	OP_TMPLT_INST_OPC(sig_x_c, SpinOperators::, (size_t),);
	// sigy
	OP_TMPLT_INST_OPG(sig_y, SpinOperators::, (size_t, size_t),);
	OP_TMPLT_INST_OPG(sig_y, SpinOperators::, (size_t, const v_1d<uint>&),);
	OP_TMPLT_INST_OPG(sig_y, SpinOperators::, (size_t),);
	OP_TMPLT_INST_OPL(sig_y_l, SpinOperators::, (size_t),);
	OP_TMPLT_INST_OPC(sig_y_c, SpinOperators::, (size_t),);
	// sigz
	OP_TMPLT_INST_OPG(sig_z, SpinOperators::, (size_t, size_t),);
	OP_TMPLT_INST_OPG(sig_z, SpinOperators::, (size_t, const v_1d<uint>&),);
	OP_TMPLT_INST_OPG(sig_z, SpinOperators::, (size_t),);
	OP_TMPLT_INST_OPL(sig_z_l, SpinOperators::, (size_t),);
	OP_TMPLT_INST_OPC(sig_z_c, SpinOperators::, (size_t),);

	// ##############################################################################################################################

	// ###################################################### Q U A D R A T I C #####################################################

	// ##############################################################################################################################

	namespace QuadraticOperators
	{
		// #############################################################################################

		/*
		* @brief Create the occupation operator for single particle basis. Each state corresponds to single particle vector |... 1 ... 0 ...>, with 1 at the site _site.
		* @param _Ns the number of sites
		* @param _site the site to be acted on
		* @param _standarize if the operator should be standarized
		*/
		Operators::Operator<double> site_occupation(size_t _Ns, const size_t _site)
		{
			if (_site >= _Ns) throw std::out_of_range("Site index is out of range.");

			// create the function
			_OP<double>::GLB fun_ = [_site](u64 _state) { return (_site == _state) ? std::make_pair(_state, 1.0) : std::make_pair(_state, 0.0); };

			GeneralizedMatrixFunction<double> _mat = [_site](size_t _Ns)
				{
					GeneralizedMatrix<double> _out(_Ns, true);
					_out.set(_site, _site, 1.0);
					return _out;
				};

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::OTHER);
			_op.setIsQuadratic(true);
			_op.setFun(std::move(_mat));
			return _op;		
		}

		// #############################################################################################

		/*
		* @brief Create the operator that has random coefficients at all diagonal sites. The coefficients are given by the vector _coeffs.
		* @param _Ns the number of sites
		* @param _coeffs the coefficients to be used
		* @param _standarize if the operator should be standarized
		* @returns the operator
		*/
		Operators::Operator<double> site_occupation_r(size_t _Ns, const v_1d<double>& _coeffs)
		{
			// create the function
			_OP<double>::GLB fun_ = [_coeffs](u64 state) 
			{ 
				return std::make_pair(state, _coeffs[state]);
			};

			GeneralizedMatrixFunction<double> _mat = [_coeffs](size_t _Ns)
				{
					GeneralizedMatrix<double> _out(_Ns, true);

					// set the values
					for (size_t i = 0; i < _coeffs.size(); i++)
						_out.set(i, i, _coeffs[i]);

					return _out;
				};

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::OTHER);
			_op.setIsQuadratic(true);
			_op.setFun(std::move(_mat));
			return _op;		
		}

		// #############################################################################################

		/*
		* @brief Create the operator that has a random coefficients at the sites given by the vector _sites. The coefficients are given by the vector _coeffs.
		* @param _Ns the number of sites
		* @param _sites the sites to be acted on
		* @param _coeffs the coefficients to be used
		* @param _standarize if the operator should be standarized
		* @returns the operator
		*/
		Operators::Operator<double> site_occupation_r(size_t _Ns, const v_1d<size_t>& _sites, const v_1d<double>& _coeffs)
		{
			// create the function
			_OP<double>::GLB fun_ = [_sites](u64 state) 
				{
					for (size_t i = 0; i < _sites.size(); i++)
						if (_sites[i] == state)
							return std::make_pair(state, 1.0);
					return std::make_pair(state, 0.0);
				};

			GeneralizedMatrixFunction<double> _mat = [_sites, _coeffs](size_t _Ns)
				{
					GeneralizedMatrix<double> _out(_Ns, true);

					// set the values
					for (auto i = 0; i < _sites.size(); i++)
						_out.set(_sites[i], _sites[i], _coeffs[i]);

					return _out;
				};

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::OTHER);
			_op.setIsQuadratic(true);
			_op.setFun(std::move(_mat));
			return _op;		
		}

		// #############################################################################################
		
		/*
		* @brief Create the operator for the nq modulation of the occupation number
		* @param _Ns the number of sites
		* @param _momentum the momentum to be added
		* @param _standarize if the operator should be standarized
		*/
		Operators::Operator<double> site_nq(size_t _Ns, const size_t _momentum)
		{
			const auto _k = TWOPI * double(_momentum) / double(_Ns);

			_OP<double>::GLB fun_ = [_k](u64 _state) 
				{ 
					return std::make_pair(_state, std::cos(_k * _state));
				};

			GeneralizedMatrixFunction<double> _mat = [_k](size_t _Ns)
				{
					GeneralizedMatrix<double> _out(_Ns, true);

					// set the values
					for (auto i = 0; i < _Ns; i++)
						_out.set(i, i, std::cos(_k * i));	
					return _out;
				};

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::OTHER);
			_op.setIsQuadratic(true);
			_op.setFun(std::move(_mat));
			return _op;			
		}

		// #############################################################################################

		/*
		* @brief Standard hopping!
		* @param _Ns the number of sites
		* @param _site_plus the site to be acted on
		* @param _site_minus the site to be acted on
		* @returns the operator
		*/
		Operators::Operator<double> nn_correlation(size_t _Ns, const size_t _site_plus, const size_t _site_minus)
		{
			_OP<double>::GLB fun_ = [_site_plus, _site_minus](u64 state) 
				{ 
					if(state == _site_minus)
						return std::make_pair((u64)_site_plus, 1.0);
					if(state == _site_plus)
						return std::make_pair((u64)_site_minus, 1.0);
					return std::make_pair(state, 0.0);
				};

			GeneralizedMatrixFunction<double> _mat = [_site_plus, _site_minus](size_t _Ns)
				{
					GeneralizedMatrix<double> _out(_Ns, true);

					// set the values
					_out.set(_site_plus, _site_minus, 1.0);
					_out.set(_site_minus, _site_plus, 1.0);

					return _out;
				};

			// set the operator			
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::OTHER);
			_op.setIsQuadratic(true);
			_op.setFun(std::move(fun_));
			return _op;			
		}

		// #############################################################################################

		Operators::Operator<std::complex<double>> quasimomentum_occupation(size_t _Ns, const size_t _momentum)
		{
			_OP<std::complex<double>>::GLB fun_ = [](u64 _state) 
				{ 
					return std::make_pair(_state, 1.0);
				};

			GeneralizedMatrixFunction<std::complex<double>> _mat = [_momentum](size_t _Ns)
				{
					GeneralizedMatrix<std::complex<double>> _out(_Ns, false);

					// set the values
					for (auto i = 0; i < _Ns; i++)
						for (auto j = 0; j < _Ns; j++)
							_out.set(i, j, std::exp(I * double(TWOPI) * double(_momentum * (i - j) / _Ns)) / (double)(_Ns));

					return _out;
				};


			// set the operator
			Operator<std::complex<double>> _op(_Ns, 1.0, fun_, SymGenerators::OTHER);
			_op.setIsQuadratic(true);
			_op.setFun(std::move(_mat));
			return _op;			
		}

		/*
		* @brief Create the operator for the quasimomentum occupation number
		* @param _Ns the number of sites
		* @returns the operator
		*/
		Operators::Operator<double> quasimomentum_occupation(size_t _Ns)
		{
			_OP<double>::GLB fun_ = [](u64 _state) 
				{ 
					return std::make_pair(_state, 1.0);
				};

			GeneralizedMatrixFunction<double> _mat = [](size_t _Ns)
				{
					GeneralizedMatrix<double> _out(_Ns, false);

					// set the values
					auto _val = 1.0 / (double)(_Ns);
					for (auto i = 0; i < _Ns; i++)
						for (auto j = 0; j < _Ns; j++)
							_out.set(i, j, _val);

					return _out;
				};

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::OTHER);
			_op.setIsQuadratic(true);
			_op.setFun(std::move(_mat));
			return _op;			
		}

		// #############################################################################################

		Operators::Operator<double> kinetic_energy(size_t _Nx, size_t _Ny, size_t _Nz)
		{
			//auto _norm = 2;
			//if (_Ny > 1) _norm = 4;
			//if (_Nz > 1) _norm = 6;
			auto _Ns = _Nx * _Ny * _Nz;

			// create the function
			GeneralizedMatrixFunction<double> _mat = [_Nx, _Ny, _Nz](size_t _Ns)
				{
					GeneralizedMatrix<double> _out(_Ns, true);

					for (auto i = 0; i < _Ns; i++)
					{
						// x
						_out.set(i, modEUC<int>(i + 1, _Nx), -1.0);
						_out.set(i, modEUC<int>(i - 1, _Nx), -1.0);
						_out.set(modEUC<int>(i + 1, _Nx), i, -1.0);
						_out.set(modEUC<int>(i - 1, _Nx), i, -1.0);

						// y 
						if (_Ny > 1)
						{
							_out.set(i, modEUC<int>(i + _Nx, _Nx * _Ny), -1.0);
							_out.set(i, modEUC<int>(i - _Nx, _Nx * _Ny), -1.0);
							_out.set(modEUC<int>(i + _Nx, _Nx * _Ny), i, -1.0);
							_out.set(modEUC<int>(i - _Nx, _Nx * _Ny), i, -1.0);
						}

						// z 
						if (_Nz > 1)
						{
							_out.set(i, modEUC<int>(i + _Nx * _Ny, _Ns), -1.0);
							_out.set(i, modEUC<int>(i - _Nx * _Ny, _Ns), -1.0);
							_out.set(modEUC<int>(i + _Nx * _Ny, _Ns), i, -1.0);
							_out.set(modEUC<int>(i - _Nx * _Ny, _Ns), i, -1.0);
						}

					}
					return _out;
				};

			// set the operator
			Operator<double> _op(_Ns, 1.0, {}, SymGenerators::OTHER);
			_op.setIsQuadratic(true);
			_op.setFun(std::move(_mat));
			return _op;				
		}
	}

}

// ##############################################################################################################################
namespace Operators
{



	// ##########################################################################################################################



};

// ##############################################################################################################################



// ##############################################################################################################################

// ######################################################### S P I N S ##########################################################

// ##############################################################################################################################

/*
* @brief multiplication of sigma_xi | state >checkBit(tmp, L - 1 - site) ? I * Operators::_SPIN : -I * Operators::_SPIN;
* @param L lattice dimensionality (base vector length)
* @param sites the sites to meassure correlation at
*/
std::pair<u64, double> Operators::sigma_x(u64 base_vec, int L, const v_1d<uint>& sites)
{
	auto tmp = base_vec;
	for (auto const& site : sites)
		tmp = flip(tmp, L - 1 - site);
	return std::make_pair(tmp, Operators::_SPIN);
};

Operators::Operator<double> Operators::makeSigmaX(std::shared_ptr<Lattice> lat, uint site) 
{
	_OP<double>::GLB fun_ = [&](u64 state) { return Operators::sigma_x(state, lat->get_Ns(), { site }); };
	return Operator<double>(lat, 1.0, fun_, SymGenerators::SX);
}

// ##############################################################################################################################

/*
* @brief multiplication of sigma_yi | state >
* @param L lattice dimensionality (base vector length)
* @param sites the sites to meassure correlation at
*/
std::pair<u64, cpx> Operators::sigma_y(u64 base_vec, int L, const v_1d<uint>& sites)
{
	auto tmp = base_vec;
	cpx val = 1.0;
	for (auto const& site : sites) {
		val *= Binary::check(base_vec, L - site - 1) ? I * Operators::_SPIN : -I * Operators::_SPIN;
		tmp = flip(tmp, L - 1 - site);
	}
	return std::make_pair(tmp, val);
};

Operators::Operator<cpx> Operators::makeSigmaY(std::shared_ptr<Lattice> lat, uint site)
{
	_OP<cpx>::GLB fun_ = [&](u64 state) { return sigma_y(state, lat->get_Ns(), { site }); };
	return Operator<cpx>(lat, 1.0, fun_, SymGenerators::SY);
}

// ##############################################################################################################################


// ##############################################################################################################################

// ###################################################### F E R M I O N S #######################################################

// ##############################################################################################################################

/*
* @brief Describes the act of c_i1^+ c_i2^+ ... c_in^+ on some state, where n is the length of '_sites'
* |s1, s2, ..., sL> = (1-d_{s1, 1})c_1^+ ... (1-d_{sL, 1})c_L^+ |vac>
* This includes the arrangement of the operators in order to include fermionic sign.
* The code first checks whether any of the orbitals are already occupied 
* (then we can skip the calculation of sign). 
* The order of the _sites matters! Afterwards the sign corresponding to _sites is calculated by swaps in bubble sort.
* @param base_vec vector to be acted on
* @param L lattice size
*/
std::pair<u64, double> Operators::c_dag_up(u64 base_vec, uint L, v_1d<uint> _sites)
{
	// divide by the 2^L to get the integer corresponding to the UP spin only
	u64 tmp		= base_vec / BinaryPowers[L];
	double val	= 1.0;
	double sign = 1.0;
	uint comp	= 0;

	// go through sites - those give different operators at those positions, check for 0
	for (auto const& site : _sites)
		if (val *= (double)!checkBit(tmp, L - 1 - site); !(bool)val)
			break;

	// get the sign corresponding to sorting _sites
	if (val != 0 && _sites.size() > 1)
	{
		Vectors::bubbleSort(_sites.begin(), _sites.end(), std::greater<uint>(), &comp);
		if (comp % 2) sign *= -1;
	}
	else if(val == 0)
		return std::make_pair(tmp, sign * val);

	uint _currentSite	= 0;
	double _signSite	= 1.0;

	// check the Fermionic sign after moving the operators
	// go through all lattice sites, 
	// if the site at current site is one of those, append with sign
	for (auto i = L - 1; i > _sites[_sites.size() - 1]; i--)
	{
		if (checkBit(tmp, L - 1 - i))
			_signSite *= (-1.0);
		if (i == _sites[_currentSite])
		{
			sign	*= _signSite;
			tmp		=	flip(tmp, L - 1 - i);
			_currentSite++;

		}
	}
	return std::make_pair(tmp, sign * val);
}

Operators::Operator<double> Operators::makeCDagUp(std::shared_ptr<Lattice> _lat, uint _site)
{
	_OP<double>::GLB fun_ = [&](u64 state) { return Operators::c_dag_up(state, _lat->get_Ns(), { _site }); };
	return Operator<double>(_lat, 1.0, fun_, (SymGenerators)FermionicOperators::C_UP_DAG);
}

// ##############################################################################################################################

/*
* @brief Describes the act of c_i1^ c_i2^ ... c_in^ on some state, where n is the length of '_sites'
* |s1, s2, ..., sL> = (1-d_{s1, 1})c_1^+ ... (1-d_{sL, 1})c_L^+ |vac>
* This includes the arrangement of the operators in order to include fermionic sign.
* The code first checks whether any of the orbitals are already occupied 
* (then we can skip the calculation of sign). 
* The order of the _sites matters! Afterwards the sign corresponding to _sites is calculated by swaps in bubble sort.
* @param base_vec vector to be acted on
* @param L lattice size
*/
std::pair<u64, double> Operators::c_up(u64 base_vec, uint L, v_1d<uint> _sites)
{
	// divide by the 2^L to get the integer corresponding to the UP spin only
	u64 tmp		= base_vec / BinaryPowers[L];
	double val	= 1.0;
	double sign = 1.0;
	uint comp	= 0;

	// go through sites - those give different operators at those positions, check for 0
	for (auto const& site : _sites)
		if (val *= (double)checkBit(tmp, L - 1 - site); (bool)!val)
			break;

	// get the sign corresponding to sorting _sites
	if (val != 0 && _sites.size() > 1)
	{
		Vectors::bubbleSort(_sites.begin(), _sites.end(), std::greater<uint>(), &comp);
		if (comp % 2) sign *= -1;
	}
	else if(val == 0)
		return std::make_pair(tmp, sign * val);

	uint _currentSite	= 0;
	double _signSite	= 1.0;

	// check the Fermionic sign after moving the operators
	// go through all lattice sites, 
	// if the site at current site is one of those, append with sign
	for (auto i = L - 1; i > _sites[_sites.size() - 1]; i--)
	{
		if (checkBit(tmp, L - 1 - i))
			_signSite *= (-1.0);
		if (i == _sites[_currentSite])
		{
			sign	*= _signSite;
			tmp		=	flip(tmp, L - 1 - i);
			_currentSite++;

		}
	}
	return std::make_pair(tmp, sign * val);
}

Operators::Operator<double> Operators::makeCUp(std::shared_ptr<Lattice> _lat, uint _site)
{
	_OP<double>::GLB fun_ = [&](u64 state) { return Operators::c_up(state, _lat->get_Ns(), { _site }); };
	return Operator<double>(_lat, 1.0, fun_, (SymGenerators)FermionicOperators::C_UP);
}

// ##############################################################################################################################

/*
* @brief Spin down. Describes the act of c_i1^+ c_i2^+ ... c_in^+ on some state, where n is the length of '_sites'
* |s1, s2, ..., sL> = (1-d_{s1, 1})c_1^+ ... (1-d_{sL, 1})c_L^+ |vac>
* This includes the arrangement of the operators in order to include fermionic sign.
* The code first checks whether any of the orbitals are already occupied 
* (then we can skip the calculation of sign). 
* @param base_vec vector to be acted on
* @param L lattice size
*/
std::pair<u64, double> Operators::c_dag_dn(u64 base_vec, uint L, v_1d<uint> _sites)
{
	// modulo by the 2^L to get the integer corresponding to the DN spin only
	u64 tmp		= base_vec / BinaryPowers[L];
	double val	= 1.0;
	double sign = L % 2 == 0 ? (1.0) : (-1.0);
	uint comp	= 0;

	// go through sites - those give different operators at those positions, check for 0
	for (auto const& site : _sites)
		if (val *= (double)!checkBit(tmp, L - 1 - site); (bool)!val)
			break;

	// get the sign corresponding to sorting _sites
	if (val != 0 && _sites.size() > 1)
	{
		Vectors::bubbleSort(_sites.begin(), _sites.end(), std::greater<uint>(), &comp);
		if (comp % 2) sign *= -1;
	}
	else if(val == 0)
		return std::make_pair(tmp, sign * val);

	uint _currentSite	= 0;
	double _signSite	= 1.0;

	// check the Fermionic sign after moving the operators
	// go through all lattice sites, 
	// if the site at current site is one of those, append with sign
	for (auto i = L - 1; i > _sites[_sites.size() - 1]; i--)
	{
		if (checkBit(tmp, L - 1 - i))
			_signSite *= (-1.0);
		if (i == _sites[_currentSite])
		{
			sign	*= _signSite;
			tmp		=	flip(tmp, L - 1 - i);
			_currentSite++;

		}
	}
	return std::make_pair(tmp, sign * val);
}

Operators::Operator<double> Operators::makeCDagDn(std::shared_ptr<Lattice> _lat, uint _site)
{
	_OP<double>::GLB fun_ = [&](u64 state) { return Operators::c_dag_dn(state, _lat->get_Ns(), { _site }); };
	return Operator<double>(_lat, 1.0, fun_, (SymGenerators)FermionicOperators::C_DOWN_DAG);
}

// ##############################################################################################################################

/*
* @brief Spin down. Describes the act of c_i1^ c_i2^ ... c_in^ on some state, where n is the length of '_sites'
* |s1, s2, ..., sL> = (1-d_{s1, 1})c_1^+ ... (1-d_{sL, 1})c_L^+ |vac>
* This includes the arrangement of the operators in order to include fermionic sign.
* The code first checks whether any of the orbitals are already occupied 
* (then we can skip the calculation of sign). 
* @param base_vec vector to be acted on
* @param L lattice size
*/
std::pair<u64, double> Operators::c_dn(u64 base_vec, uint L, v_1d<uint> _sites)
{
	// modulo by the 2^L to get the integer corresponding to the DN spin only
	u64 tmp		= base_vec / BinaryPowers[L];
	double val	= 1.0;
	double sign = L % 2 == 0 ? (1.0) : (-1.0);
	uint comp	= 0;

	// go through sites - those give different operators at those positions, check for 0
	for (auto const& site : _sites)
		if (val *= (double)checkBit(tmp, L - 1 - site); (bool)!val)
			break;

	// get the sign corresponding to sorting _sites
	if (val != 0 && _sites.size() > 1)
	{
		Vectors::bubbleSort(_sites.begin(), _sites.end(), std::greater<uint>(), &comp);
		if (comp % 2) sign *= -1;
	}
	else if(val == 0)
		return std::make_pair(tmp, sign * val);

	uint _currentSite	= 0;
	double _signSite	= 1.0;

	// check the Fermionic sign after moving the operators
	// go through all lattice sites, 
	// if the site at current site is one of those, append with sign
	for (auto i = L - 1; i > _sites[_sites.size() - 1]; i--)
	{
		if (checkBit(tmp, L - 1 - i))
			_signSite *= (-1.0);
		if (i == _sites[_currentSite])
		{
			sign	*= _signSite;
			tmp		=	flip(tmp, L - 1 - i);
			_currentSite++;

		}
	}
	return std::make_pair(tmp, sign * val);
}

Operators::Operator<double> Operators::makeCDn(std::shared_ptr<Lattice> _lat, uint _site)
{
	_OP<double>::GLB fun_ = [&](u64 state) { return Operators::c_dn(state, _lat->get_Ns(), { _site }); };
	return Operator<double>(_lat, 1.0, fun_, (SymGenerators)FermionicOperators::C_DOWN_DAG);
}

// ##############################################################################################################################