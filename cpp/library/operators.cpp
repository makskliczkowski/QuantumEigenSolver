#include "./include/algebra/operators.h"

// ##############################################################################################################################
// ##############################################################################################################################
// ######################################################### S P I N S ##########################################################
// ##############################################################################################################################
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

/*
* @brief multiplication of sigma_zi | state >
* @param L lattice dimensionality (base vector length)
* @param sites the sites to meassure correlation at
*/
std::pair<u64, double> Operators::sigma_z(u64 base_vec, int L, const v_1d<uint>& sites) {
	double val = 1.0;
	for (auto const& site : sites)
		val *= checkBit(base_vec, L - 1 - site) ? Operators::_SPIN : -Operators::_SPIN;
	return std::make_pair(base_vec, val);
};

Operators::Operator<double> Operators::makeSigmaZ(std::shared_ptr<Lattice> lat, uint site)
{
	_OP<double>::GLB fun_ = [&](u64 state) { return Operators::sigma_z(state, lat->get_Ns(), { site }); };
	return Operator<double>(lat, 1.0, fun_, SymGenerators::SZ);
}

// ##############################################################################################################################
// ##############################################################################################################################
// ###################################################### F E R M I O N S #######################################################
// ##############################################################################################################################
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
std::pair<u64, double> Operators::c_dag_up(u64 base_vec, uint L, const v_1d<uint> _sites)
{
	// divide by the 2^L to get the integer corresponding to the UP spin only
	auto tmp	= base_vec / BinaryPowers[L];
	double val	= 1.0;
	double sign = 1.0;

	// go through sites - those give different operators at those positions
	for (auto const& site : _sites)
	{
		if (checkBit(tmp, L - 1 - site))
		{
			val = 0;
			break;
		}
		double _signIn = 1.0;
		// check the Fermionic sign after moving the operators
		for (auto i = L - 1; i > site; i--)
			if (checkBit(tmp, L - 1 - i))
				_signIn *= (-1.0);
		// flip the vector at last
		sign	= sign * _signIn;
		tmp		= flip(tmp, L - 1 - site);
	}

	// calculate the sign of _sites
	//if (val != 0 and _sites.size() > 1)
	//{
	//	uint _comparisons = 0;
	//	v_1d<uint> sites_ = _sites;
	//	VEC::bubbleSort(sites_.begin(), sites_.end(), std::greater<uint>(), &_comparisons);
	//	if (_comparisons % 2)
	//		sign *= -1;
	//}

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
std::pair<u64, double> Operators::c_up(u64 base_vec, uint L, const v_1d<uint> _sites)
{
	// divide by the 2^L to get the integer corresponding to the UP spin only
	auto tmp	= base_vec / BinaryPowers[L];
	double val	= 1.0;
	double sign = 1.0;

	// go through sites - those give different operators at those positions
	for (auto const& site : _sites)
	{
		if (!checkBit(tmp, L - 1 - site))
		{
			val = 0;
			break;
		}
		double _signIn = 1.0;
		// check the Fermionic sign after moving the operators
		for (auto i = L - 1; i > site; i--)
			if (checkBit(tmp, L - 1 - i))
				_signIn *= (-1.0);
		// flip the vector at last
		sign	= sign * _signIn;
		tmp		= flip(tmp, L - 1 - site);
	}

	//// calculate the sign of _sites
	//if (val != 0 and _sites.size() > 1)
	//{
	//	uint _comparisons = 0;
	//	v_1d<uint> sites_ = _sites;
	//	VEC::bubbleSort(sites_.begin(), sites_.end(), std::greater<uint>(), &_comparisons);
	//	if (_comparisons % 2)
	//		sign *= -1;
	//}

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
std::pair<u64, double> Operators::c_dag_dn(u64 base_vec, uint L, const v_1d<uint> _sites)
{
	// modulo by the 2^L to get the integer corresponding to the DOWN spin only
	auto tmp	= base_vec % BinaryPowers[L];
	double val	= 1.0;
	double sign = 1.0;

	// go through sites - those give different operators at those positions
	for (auto const& site : _sites)
	{
		if (checkBit(tmp, L - 1 - site))
		{
			val = 0;
			break;
		}
		double _signIn = 1.0;
		// check the Fermionic sign after moving the operators
		for (auto i = L - 1; i > site; i--)
			if (checkBit(tmp, L - 1 - i))
				_signIn *= (-1.0);
		// flip the vector at last
		sign	= sign * _signIn;
		tmp		= flip(tmp, L - 1 - site);
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
std::pair<u64, double> Operators::c_dn(u64 base_vec, uint L, const v_1d<uint> _sites)
{
	// modulo by the 2^L to get the integer corresponding to the DOWN spin only
	auto tmp	= base_vec % BinaryPowers[L];
	double val	= 1.0;
	double sign = 1.0;

	// go through sites - those give different operators at those positions
	for (auto const& site : _sites)
	{
		if (!checkBit(tmp, L - 1 - site))
		{
			val = 0;
			break;
		}
		double _signIn = 1.0;
		// check the Fermionic sign after moving the operators
		for (auto i = L - 1; i > site; i--)
			if (checkBit(tmp, L - 1 - i))
				_signIn *= (-1.0);
		// flip the vector at last
		sign	= sign * _signIn;
		tmp		= flip(tmp, L - 1 - site);
	}
	return std::make_pair(tmp, sign * val);
}

Operators::Operator<double> Operators::makeCDn(std::shared_ptr<Lattice> _lat, uint _site)
{
	_OP<double>::GLB fun_ = [&](u64 state) { return Operators::c_dn(state, _lat->get_Ns(), { _site }); };
	return Operator<double>(_lat, 1.0, fun_, (SymGenerators)FermionicOperators::C_DOWN_DAG);
}