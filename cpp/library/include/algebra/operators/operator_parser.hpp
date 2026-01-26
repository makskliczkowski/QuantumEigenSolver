#ifndef OPERATOR_PARSER_H
#define OPERATOR_PARSER_H
#include "./operator_quadratic.hpp"

namespace Operators
{
	namespace OperatorTypes
	{
		// known operators
		enum OperatorsAvailable { 
			E =	0, 
			Sx, 
			Sy, 
			Sz,
			SzR,			// random superposition of Sz
			SzRV,			// random superposition of Sz (vanishing)
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
			DECL_ENUM_ELEMENT(SzR),
			DECL_ENUM_ELEMENT(ni),
			DECL_ENUM_ELEMENT(nq),
			DECL_ENUM_ELEMENT(nn),
			DECL_ENUM_ELEMENT(nk),
			DECL_ENUM_ELEMENT(nr),
			DECL_ENUM_ELEMENT(E2)
		}
		END_ENUM(OperatorsAvailable)

		/**
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

		/**
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
	};

	// ##########################################################################################################################################


	class OperatorNameParser
	{
	private:
		static inline int precision_ = 3;
		size_t L_;
		size_t Nh_; 
		std::string Lstr_;
		std::string Nhstr_;

		// --------------------------------------------------------------------------------------------
		
		void initMap();

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

		// --------------------------------------------------------------------------------------------
		template <typename _T, typename _Op = Operators::Operator<_T>>
		bool createGlobalOperator(const std::string& _input, std::shared_ptr<_Op>& _operator,
				bool _usesRealAllowed 		= true,
				bool _useHilbertAllowed 	= false,
				randomGen* _rgen 			= nullptr)
			requires std::is_base_of_v<Operators::GeneralOperator<_T, typename _Op::repType, typename _Op::repTypeV>, _Op> &&
			std::is_same_v<typename _Op::innerType, _T>;

		// --------------------------------------------------------------------------------------------

		template <typename _T, typename _Op = Operators::Operator<_T>>
		std::pair<v_sp_t<_Op>, strVec> createGlobalOperators(const strVec& _inputs,
																							bool _usesReal 		= true,
																							bool _usesHilbert 	= false,
																							randomGen* _rgen 	= nullptr)
			requires std::is_base_of_v<Operators::GeneralOperator<_T, typename _Op::repType, typename _Op::repTypeV>, _Op> &&
			std::is_same_v<typename _Op::innerType, _T>;

		// --------------------------------------------------------------------------------------------
	};
};

#endif
// ##########################################################################################################################################