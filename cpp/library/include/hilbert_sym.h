#pragma once
#ifndef HILBERTSYM_H
#define HILBERTSYM_H

/*******************************
* Definitions for the operators 
* and Hiblert space that handles
* the symmetries etc.
*******************************/

#include "algebra/general_operator.h"

namespace Operators 
{
	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ########################################################### T R A N S L A T E ############################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################
	
	template<typename _T>
	inline _GLB<_T> translation(std::shared_ptr<Lattice>& lat)
	{
		// cyclic shift function with boundary preservation
		_GLB<_T> cyclicShift;
		uint Lx [[maybe_unused]] = lat->get_Lx();
		uint Ly [[maybe_unused]] = lat->get_Ly();
		uint Lz	[[maybe_unused]] = lat->get_Lz();
		uint Ns		= lat->get_Ns();
		switch (lat->get_Dim())
		{
		case 1:
			cyclicShift = std::function(
				[Ns](u64 state)
				{
					return std::make_pair(rotateLeft(state, Ns), _T(1.0));
				}
			);
			break;
		case 2:
			cyclicShift = std::function(
				[Lx, Ns](u64 state)
				{
					u64 tmpState = state;
					for (uint i = 0; i < Lx; i++)
						tmpState = rotateLeft(state, Ns);
					return std::make_pair(tmpState, _T(1.0));
				}
			);
			break;
		case 3:
			cyclicShift = std::function(
				[Lx, Ly, Ns](u64 state)
				{
					u64 tmpState = state;
					for (uint i = 0; i < Lx * Ly; i++)
						tmpState = rotateLeft(state, Ns);
					return std::make_pair(tmpState, _T(1.0));
				}
			);
			break;
		default:
			cyclicShift = std::function(
				[Ns](u64 state)
				{
					return std::make_pair(rotateLeft(state, Ns), _T(1.0));
				}
			);
			break;
		};
		return cyclicShift;
	}

	/*
	* @brief generates translation symmetry
	* @param kx K sector in X direction
	* @param ky K sector in Y direction
	* @param kz K sector in Z direction
	* @param dim dimension of translation
	* @param base base of the Hilbert space
	*/
	template <typename _T>
	inline Operator<_T> makeTranslation(std::shared_ptr<Lattice> lat, int kx, int ky = 0, int kz = 0, int8_t dim = 1) {
		auto Kx = TWOPI * kx / double(lat->get_Lx());
		auto Ky = TWOPI * ky / double(lat->get_Ly());
		auto Kz = TWOPI * kz / double(lat->get_Lz());
		auto k	= Kx;
		if (dim == 2) 
			k = Ky;
		else if (dim == 3) 
			k = Kz;
		// exponent used as generator
		_T val	= MATH::expI<double, _T>(k);
		// return operator		
		return Operator<_T>(lat, val, translation<_T>(lat), SymGenerators::T);
	}

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ########################################################## R E F L E C T I O N ###########################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	template<typename _T>
	inline _GLB<_T> reflection(std::shared_ptr<Lattice> lat, int base)
	{
		auto Ns			= lat->get_Ns();
		_GLB<_T> fun	= [Ns, base](u64 state) {
			return std::make_pair(revBits(state, Ns, base), _T(1.0));
		};
		return fun;
	};

	/*
	* @brief Reflection symmetry
	*/
	template <typename _T>
	Operator<_T> makeReflection(std::shared_ptr<Lattice> lat, int sec, int base = 2) {
		auto reflect = reflection<_T>(lat, base);
		return Operator<_T>(lat, _T(sec), reflect, SymGenerators::R);
	};

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ############################################################## F L I P   Z ###############################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	template<typename _T>
	inline _GLB<_T> flipZ(std::shared_ptr<Lattice> lat)
	{
		auto Ns = lat->get_Ns();
		_GLB<_T> fun = [Ns](u64 state) {
			int spinUps = Ns - std::popcount(state);
			return std::make_pair(state, _T(spinUps % 2 == 0 ? 1.0 : -1.0));
		};
		return fun;
	};

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ############################################################## F L I P   Y ###############################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	template<typename _T>
	inline _GLB<_T> flipY(std::shared_ptr<Lattice> lat)
	{
		auto Ns = lat->get_Ns();
		_GLB<_T> fun = [Ns](u64 state)
		{
			int spinUps		=	Ns - std::popcount(state);
			_T _val			=	double(spinUps % 2 == 0 ? 1.0 : -1.0) * std::pow(I, Ns);
			return std::make_pair(flipAll(state, Ns), _val);
		};
		return fun;
	};

	// ###################################################################

	template<>
	inline _GLB<double> flipY<double>(std::shared_ptr<Lattice> lat)
	{
		auto Ns = lat->get_Ns();
		_GLB<double> fun = [Ns](u64 state)
		{
			int spinUps = Ns - std::popcount(state);
			double _val = double(spinUps % 2 == 0 ? 1.0 : -1.0) * std::real(std::pow(I, Ns));
			return std::make_pair(flipAll(state, Ns), _val);
		};
		return fun;
	};

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ############################################################## F L I P   X ###############################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	template<typename _T>
	inline _GLB<_T> flipX(std::shared_ptr<Lattice> lat)
	{
		auto Ns = lat->get_Ns();
		auto fun = [Ns](u64 state)
		{
			return std::make_pair(flipAll(state, Ns), _T(1.0));
		};
		return fun;
	};

	// ###################################################################

	/*
	* @brief Parity with \\sigma^x
	*/
	template <typename _T>
	inline Operator<_T> makeFlipX(std::shared_ptr<Lattice> lat, int sec) 
	{
		_GLB<_T> fX = flipX<_T>(lat);
		return Operator<_T>(lat, _T(sec), fX, SymGenerators::PX);
	};

	// ###################################################################

	/*
	* @brief Parity with \\sigma^y
	*/
	template <typename _T>
	inline Operator<_T> makeFlipY(std::shared_ptr<Lattice> lat, int sec) 
	{
		_GLB<_T> fY = flipY<_T>(lat);
		return Operator<_T>(lat, _T(sec), fY, SymGenerators::PY);
	};

	// ###################################################################

	/*
	* @brief Parity with \\sigma^z
	*/
	template <typename _T>
	inline Operator<_T> makeFlipZ(std::shared_ptr<Lattice> lat, int sec) {
		_GLB<_T> fZ = flipZ<_T>(lat);
		return Operator<_T>(lat, _T(sec), fZ, SymGenerators::PZ);
	};

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ############################################################## C H O I C E ###############################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	template <typename _T>
	inline Operator<_T> symChoice(std::pair<SymGenerators, int> _g, std::shared_ptr<Lattice> _lat) {
		auto [gen, eig] = _g;
		switch (gen) {
		case SymGenerators::T:
			return makeTranslation<_T>(_lat, eig, 0, 0, 1);
			break;
		case SymGenerators::R:
			return makeReflection<_T>(_lat, eig, 2);
			break;
		case SymGenerators::PX:
			return makeFlipX<_T>(_lat, eig);
			break;
		case SymGenerators::PY:
			return makeFlipY<_T>(_lat, eig);
			break;
		case SymGenerators::PZ:
			return makeFlipZ<_T>(_lat, eig);
			break;
		case SymGenerators::E:
			return Operator<_T>(_lat);
			break;
		default:
			return Operator<_T>(_lat);
			break;
		};
	};
};

#endif // !HILBERT_H