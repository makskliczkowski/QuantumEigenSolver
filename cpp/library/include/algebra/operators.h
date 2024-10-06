/***********************************
* Contains the most common operators.
* Is used for more general opeartors.
* Also defines various acting on a 
* Hilbert space.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***********************************/
#pragma once

#include "general_operator.h"
#include "operator_algebra.h"
#include <memory>
#include <string>
#ifndef OPERATORS_H
#define OPERATORS_H

#ifndef ENTROPY_H
#	include "quantities/entropy.h"
#endif // !ENTROPY_H

constexpr auto OPERATOR_SEP			= "/";
constexpr auto OPERATOR_SEP_CORR	= "-";
constexpr auto OPERATOR_SEP_MULT 	= ",";
constexpr auto OPERATOR_SEP_DIFF	= "m";
constexpr auto OPERATOR_SEP_RANGE	= ":";
constexpr auto OPERATOR_SEP_RANDOM	= "r";
constexpr auto OPERATOR_SEP_DIV		= "_";
constexpr auto OPERATOR_PI			= "pi";
constexpr auto OPERATOR_SITE		= "l";
constexpr auto OPERATOR_SITEU    	= "L";
constexpr auto OPERATOR_SITE_M_1    = true;
#define OPERATOR_INT_CAST(x) static_cast<size_t>(x)
#define OPERATOR_INT_CAST_S(v, x, p) (v ? STR(OPERATOR_INT_CAST(x)) : STRP(x, p))

namespace Operators 
{
	// ##########################################################################################################################################

	constexpr double _SPIN			=		0.5;
	constexpr double _SPIN_RBM		=		_SPIN;

	// ##########################################################################################################################################
	
	inline std::string createOperatorName(const std::string& _type, const std::string& _name)
	{
		return _type + std::string(OPERATOR_SEP) + _name;
	}

	inline std::string createOperatorName(const std::string& _type, const std::string& _name, const std::string& _site)
	{
		return _type + std::string(OPERATOR_SEP) + _name + OPERATOR_SEP + _site;
	}

	inline std::string createOperatorName(const std::string& _type, const std::string& _name, const std::string& _site, const std::string& _site2)
	{
		return _type + std::string(OPERATOR_SEP) + _name + OPERATOR_SEP + _site + OPERATOR_SEP_CORR + _site2;
	}

	// ##########################################################################################################################################

	/*
	* @brief Applies the many body matrix to a given state O|\Psi>
	* @param _C many body state
	* @param _M many body matrix
	*/
	template<typename _Ct, typename _M>
	inline _Ct apply(const _Ct& _C, const _M& _mat)
	{
		return _mat * _C;
	}

	/*
	* @brief Applies the many body matrix to a given state and saves the overlap <\Psi|O|\Psi>
	* @param _C many body state
	* @param _M many body matrix
	* @returns the overlap <\Psi|O|\Psi>
	*/
	template<typename _M, typename _Ct>
	inline inner_type_t<_Ct> applyOverlap(const _Ct& _C, const _M& _mat)
	{
		return arma::cdot(_C, _mat * _C);
	}

	/*
	* @brief Applies the many body matrix to a given state and saves the overlap <\Psi|O|\Psi>
	* @param _Cleft many body state
	* @param _Cright many body state
	* @param _M many body matrix
	* @returns the overlap <\Psi|O|\Psi>
	*/
	template<typename _Ct, typename _T2>
	inline inner_type_t<_Ct> applyOverlap(const _Ct& _C, const GeneralizedMatrix<_T2>& _mat)
	{	
		if (_mat.isSparse())
			return CAST<inner_type_t<_Ct>>(arma::cdot(_C, _mat.getSparse() * _C));
		else
			return CAST<inner_type_t<_Ct>>(arma::cdot(_C, _mat.getDense() * _C));
	}

	// ##########################################################################################################################################

	/*
	* @brief Applies the overlap between all the states in the matrix.
	* @param _eigvecs the eigenvectors matrix
	* @param _mat the many body matrix
	* @returns the overlap matrix
	*/
	template<typename _T, typename _M2>
	inline arma::Mat<_T> applyOverlapMat(const arma::Mat<_T>& _eigvecs, const _M2& _mat)
	{
		return _eigvecs.t() * (_mat * _eigvecs);
	}

	template<typename _T, typename _T2>
	inline arma::Mat<_T> applyOverlapMat(const arma::Mat<_T>& _eigvecs, const GeneralizedMatrix<_T2>& _mat)
	{
		if (_mat.isSparse())
			return _eigvecs.t() * (arma::SpMat<_T2>(_mat.getSparse()) * _eigvecs);
		else
			return _eigvecs.t() * (_mat.getDense() * _eigvecs);
	}

	// _____________________________________________________________________________________________________________________________

	template<typename _M, typename _Ct>
	inline _Ct applyOverlap(const arma::subview_col<_Ct>& _C, const _M& _mat)
	{
		return arma::cdot(_C, _mat * _C);
	}

	/*
	* @brief Applies the many body matrix to a given state and saves the overlap <\Psi|O|\Psi>
	* @param _Cleft many body state
	* @param _Cright many body state
	* @param _M many body matrix
	* @returns the overlap <\Psi|O|\Psi>
	*/
	template <typename _Ct, typename _M>
	inline inner_type_t<_Ct> applyOverlap(const _Ct& _Cleft, const _Ct& _Cright, const _M& _mat)
	{
		return arma::cdot(_Cleft, _mat * _Cright);
	}

	template <typename _Ct, typename _M>
	inline _Ct applyOverlap(const arma::subview_col<_Ct>& _Cleft, const arma::subview_col<_Ct>& _Cright, const _M& _mat)
	{
		return arma::cdot(_Cleft, _mat * _Cright);
	}

	// ##########################################################################################################################################
	
	namespace GeneralOperators
	{
		// ##########################################################################################################################################

		/*
		* @brief For the states in state occupation representation, check if they
		* correspond to the given state, if yes, return 1.0, otherwise 0.0.
		* @note The function is used not in the full Hilbert space representation, but in the state occupation representation.
		* Therefore, although the states are not diagonal in terms of the dot product, they are in the full Hilbert space
		* Those are represented by the integer numbers - and 1 in the state in full Hilbert space corresponds to using exactly this state.
		* @param base_vec the base vector
		* @param _proj the projected vector
		* @returns the pair of the base vector and the value of the projector
		*/
		template<typename _T>
		std::pair<u64, _T> projector(u64 base_vec, u64 _proj)
		{
			return std::make_pair(base_vec, (base_vec == _proj) ? 1.0 : 0.0);
		}
		
		/*
		* @brief For the states in state occupation representation, check if they
		* correspond to the given state, if yes, return 1.0, otherwise 0.0. 
		* @note The function is used not in the full Hilbert space representation, but in the state occupation representation.
		* Therefore, although the states are not diagonal in terms of the dot product, they are in the full Hilbert space 
		* Those are represented by the integer numbers - and 1 in the state in full Hilbert space corresponds to using exactly this state.
		* @example Let's say we have hardcore bosons on the lattice. State in full Hilbert space |0, 1, 0, 0> corresponds to the state 1 = |01> in the state occupation representation.
		* but state |0, 0, 1, 0> corresponds to the state 2 = |10> in the state occupation representation and |0, 0, 0, 1> corresponds to the state 4 = |11> in the state occupation representation.
		* @param base_vec the base vector
		* @param _proj the projected vector
		* @returns the pair of the base vector and the value of the projector
		* @note Uses vector representation!
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> projector(_OP_V_T_CR base_vec, _OP_V_T_CR _proj)
		{
			if (base_vec.n_elem != _proj.n_elem) throw std::invalid_argument("The states have different dimensions.");
			return std::make_pair(base_vec, arma::approx_equal(base_vec, _proj, "absdiff", 0.0) ? 1.0 : 0.0);
		}

		template <typename _T>
		Operators::Operator<_T> projector(size_t _Ns, u64 _proj)
		{
			_OP_V_T _projv(_Ns);
			Binary::int2base<inner_type_t<_OP_V_T>, arma::Col<inner_type_t<_OP_V_T>>, false>(_proj, _projv);

			typename _OP<_T>::GLB fun_ 		= [_proj](u64 _state) 			{ return projector<_T>(_state, _proj); };
			typename _OP_V<_T>::GLB funV_ 	= [_projv](_OP_V_T_CR _state) 	{ return projector<_T>(_state, _projv); };
			return Operator<_T>(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
		}

		template <typename _T>
		Operators::Operator<_T> projector(size_t _Ns, _OP_V_T_CR _projv)
		{
			u64 _proj = Binary::base2int<inner_type_t<_OP_V_T>, _OP_V_T, false>(_projv);

			typename _OP<_T>::GLB fun_ 		= [_proj](u64 _state) 			{ return projector<_T>(_state, _proj); };
			typename _OP_V<_T>::GLB funV_ 	= [_projv](_OP_V_T_CR _state) 	{ return projector<_T>(_state, _projv); };
			return Operator<_T>(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
		}

		template <typename _T>
		Operators::OperatorComb<_T> projectorComb(size_t _Ns, u64 _proj)
		{
			return OperatorComb<_T>(projector<_T>(_Ns, _proj));
		}

		template <typename _T>
		Operators::OperatorComb<_T> projectorComb(size_t _Ns, _OP_V_T_CR _projv)
		{
			return OperatorComb<_T>(projector<_T>(_Ns, _projv));
		}

		// ##########################################################################################################################################

		template <typename _T>
		std::pair<u64, _T> projectorSum(u64 _projectTo, std::function<_T(u64)> _application)
		{
			return std::make_pair(_projectTo, _application(_projectTo));
		}

		template <typename _T>
		std::pair<_OP_V_T, _T> projectorSum(_OP_V_T_CR _projectTo, std::function<_T(_OP_V_T_CR)> _application)
		{
			return std::make_pair(_projectTo, _application(_projectTo));
		}

		template <typename _T> 
		Operators::Operator<_T> projectorSum(size_t _Ns, const u64 _projectTo, std::function<_T(u64)> _application)
		{
			_OP_V_T _projv(_Ns);
			Binary::int2base<inner_type_t<_OP_V_T>, arma::Col<inner_type_t<_OP_V_T>>, false>(_projectTo, _projv);

			typename _OP<_T>::GLB fun_ 		= [_projectTo, _application](u64 _state) 		{ return projectorSum<_T>(_projectTo, _application); };
			typename _OP_V<_T>::GLB funV_ 	= [_projv, _application](_OP_V_T_CR _state) 	{ return projectorSum<_T>(_projv, _application); };
			return Operator<_T>(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
		}

		template <typename _T>
		Operators::Operator<_T> projectorSum(size_t _Ns, _OP_V_T_CR _projectTo, std::function<_T(_OP_V_T_CR)> _application, bool _projectInt = false)
		{
			const u64 _proj = _projectInt ? Binary::base2int<inner_type_t<_OP_V_T>, _OP_V_T, false>(_projectTo) : 0;

			typename _OP<_T>::GLB fun_ 		= [_proj, _application](u64 _state) 			{ return projectorSum<_T>(0, _application); };
			typename _OP_V<_T>::GLB funV_ 	= [_projectTo, _application](_OP_V_T_CR _state) { return projectorSum<_T>(_projectTo, _application); };
			return Operator<_T>(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
		}

		template <typename _T>
		Operators::OperatorComb<_T> projectorSumComb(size_t _Ns, const u64 _projectTo, std::function<_T(u64)> _application)
		{
			return OperatorComb<_T>(projectorSum<_T>(_Ns, _projectTo, _application));
		}

		template <typename _T>
		Operators::OperatorComb<_T> projectorSumComb(_OP_V_T_CR _projectTo, std::function<_T(_OP_V_T_CR)> _application, bool _projectInt = false)
		{
			return OperatorComb<_T>(projectorSum<_T>(_projectTo, _application, _projectInt));
		}
	};


	/*
	* @brief The spin operator namespace. Contains the most common spin operators.
	*/
	namespace SpinOperators
	{
		std::pair<u64, double> sig_x(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		std::pair<_OP_V_T, double> sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_x(size_t _Ns, size_t _part);
		Operators::Operator<double> sig_x(size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_x(size_t _Ns);

		std::pair<u64, double> sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		std::pair<_OP_V_T, double> sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_z(size_t _Ns, size_t _part);
		Operators::Operator<double> sig_z(size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_z(size_t _Ns);
	}

	// ##########################################################################################################################################

	/*
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

	// ##########################################################################################################################################

	std::pair<u64, double> sigma_x(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<double> makeSigmaX(std::shared_ptr<Lattice> lat, uint site);

	std::pair<u64, cpx> sigma_y(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<cpx> makeSigmaY(std::shared_ptr<Lattice> lat, uint site);

	/*
	* @brief multiplication of sigma_zi | state >
	* @param L lattice dimensionality (base vector length)
	* @param sites the sites to meassure correlation at
	*/
	template <typename _T>
	std::pair<u64, _T> sigma_z(u64 base_vec, int L, const v_1d<uint>& sites)
	{
		_T val = 1.0;
		for (auto const& site : sites)
			val *= checkBit(base_vec, L - 1 - site) ? Operators::_SPIN : -Operators::_SPIN;
		return std::make_pair(base_vec, val);
	}
	template <typename _T>
	Operators::Operator<_T> makeSigmaZ(std::shared_ptr<Lattice> lat, uint site)
	{
		typename _OP<_T>::GLB fun_ = [&](u64 state) { return Operators::sigma_z<_T>(state, lat->get_Ns(), { site }); };
		return Operator<_T>(lat, 1.0, fun_, SymGenerators::SZ);
	}
	
	/*
	* @brief Creates local sigma z
	*/
	template <typename _T>
	Operators::Operator<_T, uint> sigmaZ_L(std::shared_ptr<Lattice> lat)
	{
		typename _OP<_T>::LOC fun_ = [&](u64 state, uint i) { return Operators::sigma_z<_T>(state, lat->get_Ns(), { i }); };
		return Operator<_T, uint>(lat, 1.0, fun_, SymGenerators::SZ);
	}

	/*
	* @brief Creates correlation sigma z
	*/
	template <typename _T>
	Operators::Operator<_T, uint, uint> sigmaZ_C(std::shared_ptr<Lattice> lat)
	{
		typename _OP<_T>::COR fun_ = [&](u64 state, uint i, uint j) { return Operators::sigma_z<_T>(state, lat->get_Ns(), { i, j }); };
		return Operator<_T, uint, uint>(lat, 1.0, fun_, SymGenerators::SZ);
	}

	// ##########################################################################################################################################

	std::pair<u64, double> c_dag_up(u64 base_vec, uint L, v_1d<uint> _sites);
	Operators::Operator<double> makeCDagUp(std::shared_ptr<Lattice> _lat, uint _site);

	std::pair<u64, double> c_up(u64 base_vec, uint L, v_1d<uint> _sites);
	Operators::Operator<double> makeCUp(std::shared_ptr<Lattice> _lat, uint _site);

	std::pair<u64, double> c_dag_dn(u64 base_vec, uint L, v_1d<uint> _sites);
	Operators::Operator<double> makeCDagDn(std::shared_ptr<Lattice> _lat, uint _site);

	std::pair<u64, double> c_dn(u64 base_vec, uint L, v_1d<uint> _sites);
	Operators::Operator<double> makeCDn(std::shared_ptr<Lattice> _lat, uint _site);

	// ##########################################################################################################################################
	namespace OperatorTypes
	{
		// known operators
		enum OperatorsAvailable { 
			E =	0, 
			Sx, 
			Sy, 
			Sz,
			// quadratic
			ni, 			// site occupation
			nq,				// site modulation 
			nn,				// site correlation 
			nk,				// quasi-momentum occupation
			nr, 			// site occupation (Random)
			E2,				// identity squared - not used
		};

		BEGIN_ENUM(OperatorsAvailable)
		{
			DECL_ENUM_ELEMENT(E),
			DECL_ENUM_ELEMENT(Sx),
			DECL_ENUM_ELEMENT(Sy),
			DECL_ENUM_ELEMENT(Sz),
			DECL_ENUM_ELEMENT(ni),
			DECL_ENUM_ELEMENT(nq),
			DECL_ENUM_ELEMENT(nn),
			DECL_ENUM_ELEMENT(nk),
			DECL_ENUM_ELEMENT(nr),
			DECL_ENUM_ELEMENT(E2)
		}
		END_ENUM(OperatorsAvailable)

		/*
		* @brief Checks if the operator needs integer indices
		* @param _op the operator
		* @returns true if the operator needs integer indices	
		*/
		inline bool needsIntegerIdx(OperatorsAvailable _op)
		{
			switch (_op)
			{
			// case OperatorsAvailable::nq:
			case OperatorsAvailable::nk:
			case OperatorsAvailable::nr:
				return false;
			default:
				return true;
			}
		}
		// --------------------------------------------------------------------------------------------

		/*
		* @brief Check if the operator uses Hilbert space dimension rather than the integer indices
		* @param _op the operator
		* @returns true if the operator uses Hilbert space dimension
		*/
		inline bool needsHilbertSpaceDim(OperatorsAvailable _op)
		{
			switch (_op)
			{
			case OperatorsAvailable::E:
			case OperatorsAvailable::E2:
				return false;
			// quadratic operators do use Nh!
			case OperatorsAvailable::ni:
			case OperatorsAvailable::nq:
			case OperatorsAvailable::nn:
			case OperatorsAvailable::nk:
			case OperatorsAvailable::nr:
				return true;
			default:
				return false;
			}
			return false;
		}
	}

	// ##########################################################################################################################################


	class OperatorNameParser
	{
	private:
		static inline int precision_ = 3;
		size_t L_;
		size_t Nh_; 
		std::string Lstr_;
		std::string Nhstr_;

		void initMap()
		{
			for(int fooInt = static_cast<int>(OperatorTypes::OperatorsAvailable::E); fooInt != static_cast<int>(OperatorTypes::OperatorsAvailable::E2); fooInt++ )
			{
				if(fooInt == static_cast<int>(OperatorTypes::OperatorsAvailable::E2) || fooInt == static_cast<int>(OperatorTypes::OperatorsAvailable::E)) 
					continue;
				
				// setup the name 
				std::string fooStr 		= OperatorTypes::getSTR_OperatorsAvailable(static_cast<OperatorTypes::OperatorsAvailable>(fooInt));
				operator_map_[fooStr] 	= static_cast<OperatorTypes::OperatorsAvailable>(fooInt);
			}
		}

	public:
		// create a map of operators
		std::map<std::string, Operators::OperatorTypes::OperatorsAvailable> operator_map_;

		// --------------------------------------------------------------------------------------------

		OperatorNameParser(size_t L) : L_(L), Nh_(L), Lstr_(std::to_string(L)), Nhstr_(Lstr_) 
		{
			this->initMap();
		};

		OperatorNameParser(size_t L, size_t Nh) : L_(L), Nh_(Nh), Lstr_(std::to_string(L)), Nhstr_(std::to_string(Nh)) 
		{
			this->initMap();
		};

		// --------------------------------------------------------------------------------------------

		
		// parse input 
		strVec parse(const strVec& _inputs);

	private:
		// parse single input
		strVec parse(const std::string& _input);

		// parse default (without the site separator)
		strVec parseDefault(const std::string& _input);

		// parse with the sites after "/"
		std::string parseSingleOperator(const std::string& _input);

		// parse with correlation after "-"
		strVec parseCorrelationOperator(const std::string& _input);

		// parse multiple operators
		strVec parseMultipleOperators(const std::string& _input);

		// parse range of sites
		strVec parseRangeOperators(const std::string& _input);

		// --------------------------------------------------------------------------------------------
		
		// resolve the operator name from the input sites
		std::pair<std::string, std::string> resolveOperatorSeparator(const std::string& _input);

		// resolve the site and return a long double (for the indices parsing)
		long double resolveSite(const std::string& _site, bool _usesHilbert = false);

		// std::string resolveSite(double _site);

		std::vector<long double> resolveSites(const strVec& _sites, bool _usesHilbert);

		strVec resolveSitesMultiple(const std::string& _sites, bool _needsIntIdx = true, bool _usesHilbert = false);

		// resolve the correlation recursively
		void resolveCorrelation(const std::vector<strVec>& _list, strVec& _currentCombination, size_t _depth, strVec& _out);

		// --------------------------------------------------------------------------------------------
	public:

		/*
		* @brief Creates a global operator from the input string - this allows for its further usage in the calculations.
		* (creating matrices, acting on states, etc.)
		* @param _input the input string
		* @param _operator the operator to create
		* @returns true if the operator was created successfully
		*/
		template <typename _T>
		bool createGlobalOperator(const std::string& _input, std::shared_ptr<Operator<_T>>& _operator,
				bool _usesRealAllowed 		= true,
				bool _useHilbertAllowed 	= false,
				randomGen* _rgen 			= nullptr)
		{
			// resolve the operator and the sites based on the input
			auto [op, sites] 		= this->resolveOperatorSeparator(_input);
			
			// check if the operator is known
			if (!this->operator_map_.contains(op))
				return false;

			// check if the operator uses the Hilbert space or the lattice size
			bool _usesHilbert 		= OperatorTypes::needsHilbertSpaceDim(this->operator_map_[op]);
			
			// get the dimension - either the Hilbert space or the lattice size (depending on the character of the operator)
			size_t _dimension 		= _usesHilbert ? this->Nh_ : this->L_;

			// check if the sites contain the correlation or random operator
			v_1d<long double> _sites 	= { 0 };
			bool _containsRandom 		= false;
			if (_containsRandom = sites.find(OPERATOR_SEP_RANDOM) != std::string::npos; !_containsRandom)
				_sites = this->resolveSites(splitStr(sites, OPERATOR_SEP_CORR), _usesHilbert);

			// filter the operators
			if (!_useHilbertAllowed && (_usesHilbert || _containsRandom))
				return false;
			else if (!_usesRealAllowed && !_usesHilbert)
				return false;

			// create the operator
			switch (operator_map_[op])
			{
			// !!!!! SPIN OPERATORS !!!!!
			case OperatorTypes::OperatorsAvailable::Sx: 
				_operator = std::make_shared<Operator<_T>>(Operators::SpinOperators::sig_x(_dimension, Vectors::convert<uint>(_sites)));
				break;
			case OperatorTypes::OperatorsAvailable::Sy:
				// return Operators::SpinOperators::sig_y(this->L_, _sites);
				break;
			case OperatorTypes::OperatorsAvailable::Sz:
				_operator = std::make_shared<Operator<_T>>(Operators::SpinOperators::sig_z(_dimension, Vectors::convert<uint>(_sites)));
				break;
			// !!!!! QUADRATIC OPERATORS !!!!!
			case OperatorTypes::OperatorsAvailable::ni:
				_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::site_occupation(_dimension, _sites[0]));	
				break;
			case OperatorTypes::OperatorsAvailable::nq:
				_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::site_nq(_dimension, _sites[0]));
				break;
			case OperatorTypes::OperatorsAvailable::nn:
				if(_sites.size() == 1)
					_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::nn_correlation(_dimension, _sites[0], _sites[0]));
				else if (_sites.size() > 1)
					_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::nn_correlation(_dimension, _sites[0], _sites[1]));
				break;
			case OperatorTypes::OperatorsAvailable::nk:
				if (_sites[0] == 0)
					_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::quasimomentum_occupation(_dimension));
				// else 
					// return Operators::QuadraticOperators::quasimomentum_occupation(this->L_, _sites[0]);
				break;
			// !!!!!! RANDOM OPERATOR !!!!!!
			case OperatorTypes::OperatorsAvailable::nr:
			{
				if (_rgen)
				{
					// create the random operator
					v_1d<double> _rcoefs = _rgen->random<double>(-1.0, 1.0, _dimension);
					_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::site_occupation_r(_dimension, _rcoefs));
				}
				else 
					return false;
				break;	
			}
			default:
				return false;
			};

			return true;
		}

		/*
		* @brief Creates a global operator from the input string - this allows for its further usage in the calculations.
		* (creating matrices, acting on states, etc.)
		* allows for filtering the operators on being quadratic or many-body operators
		* @param _input the input string
		* @param _operator the operator to create
		* @param _uses real if the operator uses real space indices (can be also momentum)
		* @param _usesHilbert if the operator uses the Hilbert space dimension (quadraic operators)
		* @returns a pair of the operator and the names of the operators
		*/
		template <typename _T>
		std::pair<std::vector<std::shared_ptr<Operator<_T>>>, strVec> createGlobalOperators(const strVec& _inputs,
																							bool _usesReal 		= true,
																							bool _usesHilbert 	= false,
																							randomGen* _rgen 	= nullptr)
		{
			std::vector<std::shared_ptr<Operator<_T>>> ops;

			// parse the input strings
			strVec _outStr = this->parse(_inputs);

			// create the operators
			LOGINFO("Using operators: ", LOG_TYPES::INFO, 4);
			strVec _msgs = {};
			for (int i = 0; i < _outStr.size(); i++)
				_msgs.push_back(STR(i) + ")" + _outStr[i]);
			LOGINFO(_msgs, LOG_TYPES::INFO, 4);
			
			// try to parse the operators
			strVec _outOperators = {};
			for (auto& op : _outStr)
			{
				std::shared_ptr<Operator<_T>> _opin;

				// check if the operator is valid
				if (this->createGlobalOperator<_T>(op, _opin, _usesReal, _usesHilbert, _rgen))
				{
					LOGINFO("Correctly parsed operator: " + op, LOG_TYPES::INFO, 4);
					ops.push_back(_opin);
					_outOperators.push_back(op);
				}
			}

			return std::make_pair(ops, _outOperators);
		}

		// --------------------------------------------------------------------------------------------
	};

};

#endif

//class avOperators {
//public:
//	std::string lat_type = "";
//	int Ns = 1;
//	int Lx = 1;
//	int Ly = 1;
//	int Lz = 1;
//
//
//	// sigma z
//	double s_z = 0.0;
//
//	// correlation with z neighbor
//	double s_z_nei = 0.0;
//	//v_3d<double> s_z_cor;
//	mat s_z_cor;
//	vec s_z_i;
//
//	// sigma x
//	cpx s_x = 0.0;
//	// correlation with x neighbor
//	double s_x_nei = 0.0;
//	//v_3d<double> s_x_cor;
//	mat s_x_cor;
//	cx_vec s_x_i;
//
//	// sigma y
//	cpx s_y = 0.0;
//	// correlation with x neighbor
//	cpx s_y_nei = 0.0;
//	//v_3d<double> s_x_cor;
//	//mat s_y_cor;
//	//cx_vec s_y_i;
//
//
//	// entropy
//	vec ent_entro;
//
//	// energy
//	cpx en = 0.0;
//
//	avOperators() = default;
//	avOperators(int Lx, int Ly, int Lz, int Ns, std::string lat_type)
//		: Lx(Lx), Ly(Ly), Lz(Lz), Ns(Ns), lat_type(lat_type)
//	{
//		//v_3d<double> corr_vec;
//		//if (lat_type == "square") {
//		//	corr_vec = SPACE_VEC_D(Lx, Ly, Lz);
//		//}
//		//else if (lat_type == "hexagonal") {
//		//	corr_vec = SPACE_VEC_D(Lx, 2 * Ly, Lz);
//		//}
//
//		this->s_z_cor = mat(Ns, Ns, arma::fill::zeros);
//		this->s_z_i = arma::vec(Ns, arma::fill::zeros);
//		this->s_x_cor = mat(Ns, Ns, arma::fill::zeros);
//		this->s_x_i = arma::cx_vec(Ns, arma::fill::zeros);
//		this->ent_entro = arma::vec(Ns - 1, arma::fill::zeros);
//	};
//
//	void reset() {
//		//v_3d<double> corr_vec;
//		//if (lat_type == "square") {
//		//	corr_vec = SPACE_VEC_D(Lx, Ly, Lz);
//		//}
//		//else if (lat_type == "hexagonal") {
//		//	corr_vec = SPACE_VEC_D(Lx, 2 * Ly, Lz);
//		//}
//		this->s_x = 0.0;
//		this->s_x_nei = 0.0;
//		this->s_y = 0.0;
//		this->s_y_nei = 0.0;
//		this->s_z = 0.0;
//		this->s_z_nei = 0.0;
//
//		this->s_z_cor = mat(Ns, Ns, arma::fill::zeros);
//		this->s_z_i = arma::vec(Ns, arma::fill::zeros);
//		this->s_x_cor = mat(Ns, Ns, arma::fill::zeros);
//		this->s_x_i = arma::cx_vec(Ns, arma::fill::zeros);
//		this->ent_entro = arma::vec(Ns - 1, arma::fill::zeros);
//	};
//
//	void normalise(u64 norm, const v_3d<int>& spatialNorm) {
//		this->s_z /= double(norm);
//		this->s_y /= double(norm);
//		this->s_x /= double(norm);
//		this->s_z_i /= double(norm);
//		this->s_x_i /= double(norm);
//		this->s_z_nei /= double(norm);
//		this->s_x_nei /= double(norm);
//		this->s_y_nei /= double(norm);
//
//		this->s_x_cor /= double(norm);
//		this->s_z_cor /= double(norm);
//
//		//for (int i = 0; i < this->s_x_cor.size(); i++) {
//		//	for (int j = 0; j < this->s_x_cor[i].size(); j++) {
//		//		for (int k = 0; k < this->s_x_cor[i][j].size(); k++) {
//		//			this->s_x_cor[i][j][k] /= spatialNorm[i][j][k] * norm;
//		//			this->s_z_cor[i][j][k] /= spatialNorm[i][j][k] * norm;
//		//		}
//		//	}
//		//}
//		this->en /= double(norm);
//	};
//};



//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op)
//{
//	cpx value = 0;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		for (int j = 0; j < Ns; j++) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, j));
//			value += val * conj(alfa(new_idx)) * beta(k);
//		}
//	}
//	return value / double(this->Ns);
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, std::vector<int> sites)
//{
//	for (auto& site : sites)
//		if (site < 0 || site >= this->Ns) throw "Site index exceeds chain";
//	cpx value = 0;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		for (auto const& site : sites) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, site));
//			value += val * conj(alfa(new_idx)) * beta(k);
//		}
//	}
//	return value;
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, int site_a, int site_b)
//{
//	if (site_a < 0 || site_b < 0 || site_a >= this->Ns || site_b >= this->Ns) throw "Site index exceeds chain";
//	cpx value = 0;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		const auto& [new_idx, val] = op(k, Ns, v_1d<int>{site_a, site_b});
//		value += val * conj(alfa(new_idx)) * beta(k);
//	}
//	return value;
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, int corr_len)
//{
//	return cpx();
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op)
//{
//	cpx value = 0;
//	//stout << alfa << EL;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		for (int j = 0; j < Ns; j++) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, j));
//			value += val * conj(alfa(new_idx)) * alfa(k);
//			//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << EL;
//		}
//	}
//	return value / double(this->Ns);
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, std::vector<int> sites)
//{
//	for (auto& site : sites)
//		if (site < 0 || site >= this->Ns) throw "Site index exceeds chain";
//	cpx value = 0;
//	//stout << alfa << EL;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		for (auto const& site : sites) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, site));
//			value += val * conj(alfa(new_idx)) * alfa(k);
//			//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << "," << VEQ(site) << EL;
//		}
//	}
//	return value;
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, int site_a, int site_b)
//{
//	cpx value = 0;
//	//stout << alfa << EL;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		if (!(site_a < 0 || site_b < 0 || site_a >= this->Ns || site_b >= this->Ns)) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>{site_a, site_b});
//			value += val * conj(alfa(new_idx)) * alfa(k);
//		}
//		//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << "," << VEQ(site_a) << "," << VEQ(site_b) << EL;
//	}
//	return value;
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, int corr_len)
//{
//	return cpx();
//}


//// -----------------   				   HELPERS  				    -------------------
//template<typename _type>
//inline void Operators<_type>::calculate_operators(const Col<_type>& eigvec, avOperators& av_op, bool cal_entro)
//{
//	av_op.reset();
//
//	// --------------------- compare sigma_z ---------------------
//
//	// S_z_vector extensive
//	av_op.s_z = std::real(this->av_operator(eigvec, this->sigma_z));
//
//	// S_z at each site
//	for (auto i = 0; i < Ns; i++)
//		av_op.s_z_i(i) = std::real(this->av_operator(eigvec, this->sigma_z, v_1d<int>(1, i)));
//	// stout << av_op.s_z_i << EL;
//	// S_z correlations
//	for (auto i = 0; i < Ns; i++) {
//		auto z_nei = this->lat->get_z_nn(i);
//		if (z_nei >= 0)
//			av_op.s_z_nei += std::real(this->av_operator(eigvec, this->sigma_z, i, z_nei));
//
//		for (auto j = 0; j < Ns; j++) {
//			av_op.s_z_cor(i, j) += std::real(this->av_operator(eigvec, this->sigma_z, i, j));
//		}
//	}
//	av_op.s_z_nei /= Ns;
//	// --------------------- compare sigma_u ---------------------
//
//	// S_y_vector extensive
//	//av_op.s_y = std::real(this->av_operator(eigvec, this->sigma_y));
//
//	// S_y at each site
//	//for (auto i = 0; i < Ns; i++)
//	//	av_op.s_y_i(i) = std::real(this->av_operator(eigvec, this->sigma_y, v_1d<int>(1, i)));
//
//	// S_y correlations
//	for (auto i = 0; i < Ns; i++) {
//		int y_nei = this->lat->get_y_nn(i);
//		if (y_nei >= 0)
//			av_op.s_y_nei += std::real(this->av_operator(eigvec, this->sigma_y, i, y_nei));
//	}
//	av_op.s_y_nei /= Ns;
//	// --------------------- compare sigma_x ---------------------
//
//	// S_x_vector extensive
//	av_op.s_x = std::real(this->av_operator(eigvec, this->sigma_x));
//
//	// S_x at each site
//	for (auto i = 0; i < Ns; i++)
//		av_op.s_x_i(i) = std::real(this->av_operator(eigvec, this->sigma_x, v_1d<int>(1, i)));
//
//	// S_x correlations
//	for (auto i = 0; i < Ns; i++) {
//		int x_nei = this->lat->get_x_nn(i);
//		if (x_nei >= 0)
//			av_op.s_x_nei += std::real(this->av_operator(eigvec, this->sigma_x, i, x_nei));
//
//		for (auto j = 0; j < Ns; j++) {
//			av_op.s_x_cor(i, j) += std::real(this->av_operator(eigvec, this->sigma_x, i, j));
//		}
//	}
//	av_op.s_x_nei /= Ns;
//
//	// --------------------- entropy ----------------------
//	if (cal_entro)
//		av_op.ent_entro = this->entanglement_entropy_sweep(eigvec);
//}
