#include "./include/algebra/operators.h"
#include "include/algebra/general_operator.h"
#include "source/src/Include/str.h"
#include "source/src/lin_alg.h"
#include <complex>
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

		/*
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

		template <typename _T>
		std::pair<_OP_V_T, _T> sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val 			= 1.0;
			_OP_V_T _base_vec 	= base_vec;
			for (auto const& site : sites)
			{
				flip(_base_vec, site, Operators::_SPIN);
				_val *= Operators::_SPIN;
			}
			return std::make_pair(_base_vec, _val);
		}

		// #####################################

		/*
		* @brief multiplication of sigma_xi | state > - local
		* @param _Ns lattice dimensionality (base vector length)
		* @param _part the site to meassure correlation at
		* @returns the operator acting on the _part site
		*/
		template <typename _T>
		Operators::Operator<_T> sig_x(size_t _Ns, size_t _part)
		{
			// create the function
			_GLB<_T> fun_		= [_Ns, _part](u64 state) { return sig_x<_T>(state, _Ns, { (uint)_part }); };
			_GLB_V<_T> funV_	= [_Ns, _part](_OP_V_T_CR state) { return sig_x<_T>(state, _Ns, { (uint)_part }); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 0;
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1
			_acts |= 1 << (_Ns - 1 - _part);

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SX);
			_op.setActOn(_acts);
			return _op;
		}

		/*
		* @brief multiplication of sigma_xi | state > - correlation
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at
		* @returns the operator acting on the _part site
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
			return _op;
		}

		/*
		* @brief multiplication of sigma_xi | state > - correlation - global
		* @param _Ns lattice dimensionality (base vector length)
		* @returns the operator acting on the _part site
		*/
		template <typename _T>
		Operators::Operator<_T> sig_x(size_t _Ns)
		{
			// set the vector of sites
			v_1d<uint> _sites		= Vectors::vecAtoB<uint>(_Ns);
			// create the function
			_GLB<_T> fun_			= [_Ns, _sites](u64 state) { return sig_x<_T>(state, _Ns, _sites); };
			_GLB_V<_T> funV_ 		= [_Ns, _sites](_OP_V_T_CR state) { return sig_x<_T>(state, _Ns, _sites); };

			// save on which elements the operator acts (for the sake of the correctness)
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1

			// take all of them!
			u64 _acts				= (ULLPOW(_Ns)) - 1;

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SX);
			_op.setActOn(_acts);
			return _op;
		}

		template <typename _T>
		Operators::Operator<_T, uint> sig_x_l(size_t _Ns)
		{
			_LOC<_T> fun_ 	= [_Ns](u64 state, uint _part) { return sig_x<_T>(state, _Ns, { _part }); };
			_LOC_V<_T> funV_ 	= [_Ns](_OP_V_T_CR state, uint _part) { return sig_x<_T>(state, _Ns, { _part }); };

			// set the operator
			Operator<_T, uint> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SX);
			return _op;
		}


		// ############################################################################################# 

		// ######################################## SIGMA Z ############################################

		// #############################################################################################
		
		/*
		* @brief multiplication of sigma_zi | state >
		* @param base_vec the base vector to be acted on. This is given by the copy.
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at. The order of the sites matters!
		* @returns the pair of the new state and the value of the operator
		*/
		template <typename _T>
		std::pair<u64, _T> sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val = 1.0;
			for (auto const& site : sites)
				_val *= Binary::check(base_vec, _Ns - 1 - site) ? Operators::_SPIN : -Operators::_SPIN;
			return std::make_pair(base_vec, _val);
		}

		template <typename _T>
		std::pair<_OP_V_T, _T> sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			_T _val 			= 1.0;
			for (auto const& site : sites)
				_val *= Binary::check(base_vec, site) ? Operators::_SPIN : -Operators::_SPIN;

			return std::make_pair(base_vec, _val);
		}

		/*
		* @brief multiplication of sigma_zi | state > - local
		* @param _Ns lattice dimensionality (base vector length)
		* @param _part the site to meassure correlation at
		* @returns the operator acting on the _part site
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
			return _op;
		}

		/*
		* @brief multiplication of sigma_zi | state > - correlation
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at
		* @returns the operator acting on the _part site
		*/
		template <typename _T>
		Operators::Operator<_T> sig_z(size_t _Ns, const v_1d<uint>& sites)
		{
			// create the function
			_GLB<_T> fun_		= [_Ns, sites](u64 state) { return sig_z<_T>(state, _Ns, sites); };
			_GLB_V<_T> funV_	= [_Ns, sites](_OP_V_T_CR state) { return sig_z<_T>(state, _Ns, sites); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 0;
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1
			for(auto _part : sites)
				_acts |= 1 << (_Ns - 1 - _part);

			// set the operator
			Operator<_T> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SZ);
			_op.setActOn(_acts);
			return _op;
		}

		/*
		* @brief multiplication of sigma_zi | state > - correlation - global
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at
		* @returns the operator acting on the _part site
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
			_op.setActOn(_acts);
			return _op;
		}

		template <typename _T>
		Operators::Operator<_T, uint> sig_z_l(size_t _Ns)
		{
			_LOC<_T> fun_ 		= [_Ns](u64 state, uint _part) { return sig_z<_T>(state, _Ns, { _part }); };
			_LOC_V<_T> funV_ 	= [_Ns](_OP_V_T_CR state, uint _part) { return sig_z<_T>(state, _Ns, { _part }); };

			// set the operator
			Operator<_T, uint> _op(_Ns, 1.0, fun_, funV_, SymGenerators::SZ);
			return _op;
		}

		// ############################################################################################# 

		// ######################################## SIGMA P ############################################

		// #############################################################################################

		/*
		* @brief Operator S^+ acting on the state | state >
		* @param base_vec the base vector to be acted on. This is given by the copy.
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at. The order of the sites matters!
		* @returns the pair of the new state and the value of the operator
		*/
		std::pair<u64, double> sig_p(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			auto _val		=	1.0;
			for (auto const& site : sites)
			{
				// this op
				if (checkBit(base_vec, _Ns - 1 - site))
				{
					_val	=	0.0;
					break;
				}
				base_vec	=	flip(base_vec, _Ns - 1 - site);
				_val		*=	Operators::_SPIN;
			}
			return std::make_pair(base_vec, _val);
		}

		/*
		* @brief Operator S^+ acting on the state | state >
		* @param base_vec the base vector to be acted on. This is given by the copy.
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at. The order of the sites matters!
		* @returns the pair of the new state and the value of the operator
		*/
		std::pair<u64, double> sig_m(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			auto _val		=	1.0;
			for (auto const& site : sites)
			{
				// this op
				if (!checkBit(base_vec, _Ns - 1 - site))
				{
					_val	=	0.0;
					break;
				}
				base_vec	=	flip(base_vec, _Ns - 1 - site);
				_val		*=	Operators::_SPIN;
			}
			return std::make_pair(base_vec, _val);
		}

		/*
		* @brief multiplication of S^+S^- | state >
		* @param base_vec the base vector to be acted on. This is given by the copy.
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at. The order of the sites matters!
		* @returns the pair of the new state and the value of the operator
		*/
		std::pair<u64, double> sig_pm(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			uint _size			= sites.size();
			auto _mid			= sites.begin() + std::floor(_size / 2);

			v_1d<uint> _left(sites.begin(), _mid);
			v_1d<uint> _right(_mid, sites.end());

			auto [_out, _val]	= sig_p(base_vec, _Ns, _left);

			// return the value
			return sig_m(_out, _Ns, _right);
		}

		/*
		* @brief multiplication of S^-S^+ | state >
		* @param base_vec the base vector to be acted on. This is given by the copy.
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at. The order of the sites matters!
		* @returns the pair of the new state and the value of the operator
		*/
		std::pair<u64, double> sig_mp(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			uint _size			= sites.size();
			auto _mid			= sites.begin() + std::floor(_size / 2);

			v_1d<uint> _left(sites.begin(), _mid);
			v_1d<uint> _right(_mid, sites.end());

			auto [_out, _val]	= sig_m(base_vec, _Ns, _left);

			// return the value
			return sig_p(_out, _Ns, _right);
		}

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
	};


	// definitions of the tamplates with given types

	// sigx - double 
	template std::pair<u64, double> SpinOperators::sig_x(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
	template std::pair<_OP_V_T, double> SpinOperators::sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
	template Operators::Operator<double> SpinOperators::sig_x(size_t _Ns, size_t _part);
	template Operators::Operator<double> SpinOperators::sig_x(size_t _Ns, const v_1d<uint>& sites);
	template Operators::Operator<double> SpinOperators::sig_x(size_t _Ns);
	template Operators::Operator<double, uint> SpinOperators::sig_x_l(size_t _Ns);
	// sigz - double
	template std::pair<u64, double> SpinOperators::sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
	template std::pair<_OP_V_T, double> SpinOperators::sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
	template Operators::Operator<double> SpinOperators::sig_z(size_t _Ns, size_t _part);
	template Operators::Operator<double> SpinOperators::sig_z(size_t _Ns, const v_1d<uint>& sites);
	template Operators::Operator<double> SpinOperators::sig_z(size_t _Ns);
	template Operators::Operator<double, uint> SpinOperators::sig_z_l(size_t _Ns);
	// sigx - complex
	template std::pair<u64, std::complex<double>> SpinOperators::sig_x(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
	template std::pair<_OP_V_T, std::complex<double>> SpinOperators::sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
	template Operators::Operator<std::complex<double>> SpinOperators::sig_x(size_t _Ns, size_t _part);
	template Operators::Operator<std::complex<double>> SpinOperators::sig_x(size_t _Ns, const v_1d<uint>& sites);
	template Operators::Operator<std::complex<double>> SpinOperators::sig_x(size_t _Ns);
	template Operators::Operator<std::complex<double>, uint> SpinOperators::sig_x_l(size_t _Ns);
	// sigz - complex
	template std::pair<u64, std::complex<double>> SpinOperators::sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
	template std::pair<_OP_V_T, std::complex<double>> SpinOperators::sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
	template Operators::Operator<std::complex<double>> SpinOperators::sig_z(size_t _Ns, size_t _part);
	template Operators::Operator<std::complex<double>> SpinOperators::sig_z(size_t _Ns, const v_1d<uint>& sites);
	template Operators::Operator<std::complex<double>> SpinOperators::sig_z(size_t _Ns);
	template Operators::Operator<std::complex<double>, uint> SpinOperators::sig_z_l(size_t _Ns);

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

// ######################################################### S P I N S ##########################################################

// ##############################################################################################################################

/*
* @brief multiplication of sigma_xi | state >
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
		val *= checkBit(tmp, L - 1 - site) ? I * Operators::_SPIN : -I * Operators::_SPIN;
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