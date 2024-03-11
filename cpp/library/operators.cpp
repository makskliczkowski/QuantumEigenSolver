#include "./include/algebra/operators.h"

namespace Operators
{

	// ##############################################################################################################################

	// ######################################################### S P I N S ##########################################################

	// ##############################################################################################################################
	
	namespace SpinOperators
	{
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