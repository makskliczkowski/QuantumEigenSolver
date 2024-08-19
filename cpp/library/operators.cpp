#include "./include/algebra/operators.h"
#include <complex>

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
		std::pair<u64, double> sig_x(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			auto _val = 1.0;
			for (auto const& site : sites)
			{
				base_vec	=	flip(base_vec, _Ns - 1 - site);
				_val		*=	Operators::_SPIN;
			}
			return std::make_pair(base_vec, _val);
		}

		/*
		* @brief multiplication of sigma_xi | state > - local
		* @param _Ns lattice dimensionality (base vector length)
		* @param _part the site to meassure correlation at
		* @returns the operator acting on the _part site
		*/
		Operators::Operator<double> sig_x(size_t _Ns, size_t _part)
		{
			// create the function
			_OP<double>::GLB fun_ = [_Ns, _part](u64 state) { return sig_x(state, _Ns, { (uint)_part }); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 0;
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1
			_acts |= 1 << (_Ns - 1 - _part);

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::SX);
			_op.setActOn(_acts);
			return _op;
		}

		/*
		* @brief multiplication of sigma_xi | state > - correlation
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at
		* @returns the operator acting on the _part site
		*/
		Operators::Operator<double> sig_x(size_t _Ns, const v_1d<uint>& sites)
		{
			// create the function
			_OP<double>::GLB fun_ = [_Ns, sites](u64 state) { return sig_x(state, _Ns, sites); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 0;
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1
			for(auto _part : sites)
				_acts |= 1 << (_Ns - 1 - _part);

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::SX);
			_op.setActOn(_acts);
			return _op;
		}

		/*
		* @brief multiplication of sigma_xi | state > - correlation - global
		* @param _Ns lattice dimensionality (base vector length)
		* @returns the operator acting on the _part site
		*/
		Operators::Operator<double> sig_x(size_t _Ns)
		{
			// set the vector of sites
			v_1d<uint> _sites		= Vectors::vecAtoB<uint>(_Ns);
			// create the function
			_OP<double>::GLB fun_	= [_Ns, _sites](u64 state) { return sig_x(state, _Ns, _sites); };

			// save on which elements the operator acts (for the sake of the correctness)
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1

			// take all of them!
			u64 _acts				= (ULLPOW(_Ns)) - 1;

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::SX);
			_op.setActOn(_acts);
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
		std::pair<u64, double> sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites)
		{
			auto _val = 1.0;
			for (auto const& site : sites)
				_val *= Binary::check(base_vec, _Ns - 1 - site) ? Operators::_SPIN : -Operators::_SPIN;
			return std::make_pair(base_vec, _val);
		}

		/*
		* @brief multiplication of sigma_zi | state > - local
		* @param _Ns lattice dimensionality (base vector length)
		* @param _part the site to meassure correlation at
		* @returns the operator acting on the _part site
		*/
		Operators::Operator<double> sig_z(size_t _Ns, size_t _part)
		{
			// create the function
			_OP<double>::GLB fun_ = [_Ns, _part](u64 state) { return sig_z(state, _Ns, { (uint)_part }); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 0;
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1
			_acts |= 1 << (_Ns - 1 - _part);

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::SZ);
			_op.setActOn(_acts);
			return _op;
		}

		/*
		* @brief multiplication of sigma_zi | state > - correlation
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at
		* @returns the operator acting on the _part site
		*/
		Operators::Operator<double> sig_z(size_t _Ns, const v_1d<uint>& sites)
		{
			// create the function
			_OP<double>::GLB fun_ = [_Ns, sites](u64 state) { return sig_z(state, _Ns, sites); };

			// save on which elements the operator acts (for the sake of the correctness)
			u64 _acts = 0;
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1
			for(auto _part : sites)
				_acts |= 1 << (_Ns - 1 - _part);

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::SZ);
			_op.setActOn(_acts);
			return _op;
		}

		/*
		* @brief multiplication of sigma_zi | state > - correlation - global
		* @param _Ns lattice dimensionality (base vector length)
		* @param sites the sites to meassure correlation at
		* @returns the operator acting on the _part site
		*/
		Operators::Operator<double> sig_z(size_t _Ns)
		{
			// set the vector of sites
			v_1d<uint> _sites		= Vectors::vecAtoB<uint>(_Ns);
			// create the function
			_OP<double>::GLB fun_	= [_Ns, _sites](u64 state) { return sig_z(state, _Ns, _sites); };

			// save on which elements the operator acts (for the sake of the correctness)
			// |set the bitmask on the state, remember that this is counted from the left|
			// the first position is leftwise 0, the last is leftwise Ns - 1

			// take all of them!
			u64 _acts				= (ULLPOW(_Ns)) - 1;

			// set the operator
			Operator<double> _op(_Ns, 1.0, fun_, SymGenerators::SZ);
			_op.setActOn(_acts);
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

	}


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

// ------------------------------------------------------------------------------------------------------------------------------

// ##############################################################################################################################

// ------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Create a parser for the operator names. It allows to parse the names of the operators and return the corresponding operator.
* @param _inputs the input strings
*/
strVec Operators::OperatorNameParser::parse(const strVec& _inputs)
{
	strVec _out = {};

	// go through all the strings
	for (const auto& _str : _inputs)
	{
		// check if the string is empty
		strVec _outstings = this->parse(_str);
		for (const auto& _outstr : _outstings)
		{
			if (_outstr.size() != 0)
				_out.push_back(_outstr);
		}
	}

	// sort the output
	std::sort(_out.begin(), _out.end());

	// remove duplicates
	auto _last = std::unique(_out.begin(), _out.end());

	_out.erase(_last, _out.end());

	return _out;
}

/*
* @brief Parse the operator name and return the corresponding operator.
* @param _input the input string
* @returns the operator names as strings
*/
strVec Operators::OperatorNameParser::parse(const std::string& _input)
{
	strVec _out = {};

	// go through all the strings
	if(_input.find(OPERATOR_SEP) == std::string::npos) {
		// Assume default format {operator}/1.L.1
		_out = this->parseDefault(_input); 
	} else if(_input.find(OPERATOR_SEP_CORR) != std::string::npos) {
		// This is the correlation operator then, must be handled separately
		// {operator}/{index1}_{index2}_..._{indexN}
		_out = this->parseCorrelationOperator(_input);
	} else if(_input.find(OPERATOR_SEP_MULT) != std::string::npos) {
		// This is the multiple operator then, must be handled separately
		_out = this->parseMultipleOperators(_input);
	} else if (_input.find(OPERATOR_SEP_RANGE) != std::string::npos) {
		_out = this->parseRangeOperators(_input);
	} else {
		_out.push_back(this->parseSingleOperator(_input));
	}

	return _out;
}

// ------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Parse the site given as a string and return the corresponding site.
* The format is {site} or {site}/{div} where div is the divisor of the site.
* @param _input the input string
*/
double Operators::OperatorNameParser::resolveSite(const std::string &_site)
{
	if(_site.length() == 0)
		throw std::invalid_argument("The site: " + _site + " is not valid.");

	if(_site == "L") {
		return this->L_;
	} else if(_site.find(OPERATOR_SEP_DIV) != std::string::npos) {
		auto _div = this->resolveSite(splitStr(_site, OPERATOR_SEP_DIV)[1]);
		return this->L_ / _div;
	}
	else if (_site.find(OPERATOR_SEP_DIFF) != std::string::npos) {
		auto _diff = this->resolveSite(splitStr(_site, OPERATOR_SEP_DIFF)[1]);
		return std::max(0.0, (int)this->L_ - _diff);
	}

	// simply return the site
	auto _siteInt = std::stod(_site);
	if (_siteInt < 0 || _siteInt > this->L_)
		throw std::invalid_argument("The site: " + _site + " is out of range.");
	return std::stod(_site);
}

// ------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Given single site ranges (something that occurs after /) resolve the sites.
* @param _sites the sites to resolve
*/
strVec Operators::OperatorNameParser::resolveSitesMultiple(const std::string &_sites)
{
    strVec _out = {};

	if (_sites.find(OPERATOR_SEP_MULT) != std::string::npos) {
		for (const auto& _str : splitStr(_sites, OPERATOR_SEP_MULT))
			_out.push_back(STRP(this->resolveSite(_str), 3));
	} else if (_sites.find(OPERATOR_SEP_RANGE) != std::string::npos) {
		auto _str = splitStr(_sites, OPERATOR_SEP_RANGE);	
		if (_str.size() == 3)
		{
			// throw std::invalid_argument("The range: " + _sites + " is not valid.");		
			auto _start = this->resolveSite(_str[0]);
			auto _end 	= this->resolveSite(_str[1]);
			auto _step 	= this->resolveSite(_str[2]);

			for (auto i = _start; i <= _end; i += _step)
				_out.push_back(STRP(i, 3));
		}
	} else {
		_out.push_back(STRP(this->resolveSite(_sites), 3));
	}
	return _out;
}

// ------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Parse the list of string sites and change them to the integer sites.
* @param _sites the sites to resolve
*/
std::vector<double> Operators::OperatorNameParser::resolveSites(const strVec &_sites)
{
	std::vector<double> _out = {};

	for (const auto& _site : _sites)
		_out.push_back(this->resolveSite(_site));
	return _out;
}

// ------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Resolve the correlation operator. For a given depth in the list of all combinations, resolve the correlation.
* It can be {{1, 2}, {3, 4}, {5, 6}} and then the depth is recursively resolved.
* @param _list the list of the lists of all the sites
* @param _currentCombination the current combination
* @param _depth the current depth
* @param _out the output
*/
void Operators::OperatorNameParser::resolveCorrelation(const std::vector<strVec>& _list, strVec &_currentCombination, size_t _depth, strVec &_out)
{
	// if we already reached the depth
	if (_depth == _list.size())
	{
		std::string _str = "";
		for (const auto& _el : _currentCombination)
			_str += _el + OPERATOR_SEP_CORR;
	
		// remove the last separator
		_str.pop_back();
		_out.push_back(_str);

		return;
	}

	// go through all the elements of the current depth
	for (const auto& _el : _list[_depth])
	{
		_currentCombination.push_back(_el);
		resolveCorrelation(_list, _currentCombination, _depth + 1, _out);
		_currentCombination.pop_back();
	}
}

// ------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Checks whether the operator name is correctly formatted and returns the operator name and the index.
* The format is {operator}/{index} where the index is the index of the operator (site) or multiple sites (operators).
* @param _input the input string
* @returns the pair of the operator name and the index
*/
std::pair<std::string, std::string> Operators::OperatorNameParser::resolveOperatorSeparator(const std::string &_input)
{
	auto _posSep = _input.find(OPERATOR_SEP);
	if (_posSep == std::string::npos)
		throw std::invalid_argument("The operator name: " + _input + " is not valid.");

	// get the operator name
	const auto _opName		= _input.substr(0, _posSep);
	const auto _indexStr	= _input.substr(_posSep + 1);

	return std::make_pair(_opName, _indexStr);
}

// ------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Parse the operator name and return the corresponding operator for a single operator.
* The format is {operator}/{index} where the index is the index of the operator (site).
* @param _input the input string
*/
std::string Operators::OperatorNameParser::parseSingleOperator(const std::string & _input)
{
	// get the operator name
	const auto [_opName, _indexStr] = this->resolveOperatorSeparator(_input);

	// site index
	auto _index = this->resolveSite(_indexStr);

	// return the operator name
	return _opName + OPERATOR_SEP + STRP(_index, 3);
}

// ------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Parse the operator name and return the corresponding operator for a default operator.
* The format is {operator} and it assumes that we have {operator}/1.L.1.
*/
strVec Operators::OperatorNameParser::parseDefault(const std::string & _input)
{
	return this->parse(_input + "/1.L.1");
}

// #############################################################################################################################

/*
* @brief Parse the operator name and return the corresponding operator for correlation operators.
* The format is {operator}/{index1}_{index2}_..._{indexN} where the index is the index of the operator (site).
*/
strVec Operators::OperatorNameParser::parseCorrelationOperator(const std::string &_input)
{
	// get the operator name
	const auto [_opName, _indexStr] = this->resolveOperatorSeparator(_input);

	// split for the potential indices (for each element there might be multiple sites)
	strVec _potentialIndicies 		= splitStr(_indexStr, OPERATOR_SEP_CORR);

	std::vector<strVec> _out 		= {};

	// go through all the potential indices and resolve them
	for (int i = 0; i < _potentialIndicies.size(); ++i)
		_out.push_back(resolveSitesMultiple(_potentialIndicies[i]));

	if (_out.size() == 0)
		return {};

	strVec _outOps 	= {};
	strVec _current	= {};

	// resolve the correlation
	resolveCorrelation(_out, _current, 0, _outOps);

	for (auto& _o: _outOps)
		_o = _opName + OPERATOR_SEP + _o;

	return _outOps;
}

// #############################################################################################################################

/*
* @brief Multiple operators of the form 
* {operator}/{index1},{index2},...{indexN} where the index is the index of the operator (site).
*/
strVec Operators::OperatorNameParser::parseMultipleOperators(const std::string &_input)
{
	// get the operator name
	const auto [_opName, _indexStr] = this->resolveOperatorSeparator(_input);

	// split for the potential indices
	strVec _potentialIndicies 		= resolveSitesMultiple(_indexStr);

	for (int i = 0; i < _potentialIndicies.size(); ++i)
		_potentialIndicies[i] = _opName + OPERATOR_SEP + _potentialIndicies[i];

	return _potentialIndicies;
}

// #############################################################################################################################

/*
* @brief Parse the operator name and return the corresponding operator for range operators.
* The format is {operator}/{start}.{stop}_{step}
*/
strVec Operators::OperatorNameParser::parseRangeOperators(const std::string &_input)
{
	// get the operator name
	const auto [_opName, _indexStr] = this->resolveOperatorSeparator(_input);

	// split for the potential indices
	strVec _potentialIndicies 		= resolveSitesMultiple(_indexStr);

	for (int i = 0; i < _potentialIndicies.size(); ++i)
		_potentialIndicies[i] = _opName + OPERATOR_SEP + _potentialIndicies[i];

	return _potentialIndicies;
}