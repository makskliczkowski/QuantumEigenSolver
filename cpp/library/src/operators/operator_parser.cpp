#include "../../include/algebra/operators.h"
#include "../../include/NQS/nqs_operator.h"

// ##########################################################################################################################

namespace Operators
{

    // OperatorNameParser

    // ##########################################################################################################################

    /**
    * @brief Creates a global operator based on the input string.
    * 
    * This function parses the input string to determine the type of operator and the sites it acts on.
    * It then creates the corresponding operator and assigns it to the provided shared pointer.
    * 
    * @tparam _T The data type used by the operator.
    * @tparam _Op The template for the operator type.
    * @param _input The input string specifying the operator and sites.
    * @param _operator A shared pointer to the operator to be created.
    * @param _usesRealAllowed Flag indicating if real operators are allowed.
    * @param _useHilbertAllowed Flag indicating if Hilbert space operators are allowed.
    * @param _rgen Pointer to a random number generator, used for creating random operators.
    * @return true If the operator was successfully created.
    * @return false If the operator could not be created.
    * @requires _Op is a subclass of GeneralOperator.
    */
	template <typename _T, typename _Op>
    bool Operators::OperatorNameParser::createGlobalOperator(const std::string& _input, std::shared_ptr<_Op>& _operator,
            bool _usesRealAllowed, bool _useHilbertAllowed, randomGen* _rgen)
			requires std::is_base_of_v<Operators::GeneralOperator<_T, typename _Op::repType, typename _Op::repTypeV>, _Op> &&
					 std::is_same_v<typename _Op::innerType, _T>
    {
        auto [op, sites] 		= this->resolveOperatorSeparator(_input);       // resolve the operator and the sites based on the input
        
        if (!this->operator_map_.contains(op))                                  // check if the operator is known
            return false;

        // check if the operator uses the Hilbert space or the lattice size
        bool _usesHilbert 		= OperatorTypes::needsHilbertSpaceDim(this->operator_map_[op]);
        
        size_t _dimension 		= _usesHilbert ? this->Nh_ : this->L_;          // get the dimension - either the Hilbert space or the lattice size (depending on the character of the operator)
        v_1d<long double> _sites= { 0 };
        
        bool _containsRandom 	= false;                                        // check if the sites contain the random operator
        if (_containsRandom = sites.find(OPERATOR_SEP_RANDOM) != std::string::npos; !_containsRandom)
            _sites = this->resolveSites(splitStr(sites, OPERATOR_SEP_CORR), _usesHilbert);

        // filter the operators
        if (!_useHilbertAllowed && (_usesHilbert || _containsRandom))
            return false;
        else if (!_usesRealAllowed && !_usesHilbert)
            return false;

        switch (operator_map_[op])
        {
        // !!!!! SPIN OPERATORS !!!!!
        case OperatorTypes::OperatorsAvailable::Sx: 
            _operator = std::make_shared<_Op>(Operators::SpinOperators::sig_x<_T>(_dimension, Vectors::convert<uint>(_sites)));
            break;
        case OperatorTypes::OperatorsAvailable::Sy:
            // return Operators::SpinOperators::sig_y(this->L_, _sites);
            break;
        case OperatorTypes::OperatorsAvailable::Sz:
            _operator = std::make_shared<_Op>(Operators::SpinOperators::sig_z<_T>(_dimension, Vectors::convert<uint>(_sites)));
            break;
        case OperatorTypes::OperatorsAvailable::SzR:
			if constexpr (std::is_same_v<_T, double>)
				_operator = std::make_shared<_Op>(Operators::SpinOperators::RandomSuperposition::sig_z(_dimension));
            break;
        case OperatorTypes::OperatorsAvailable::SzRV:
			if constexpr (std::is_same_v<_T, double>)
				_operator = std::make_shared<_Op>(Operators::SpinOperators::RandomSuperposition::sig_z_vanish(_dimension));
            break;
        // !!!!! QUADRATIC OPERATORS !!!!!
        case OperatorTypes::OperatorsAvailable::ni:
			if constexpr (std::is_same_v<_T, double>)
				_operator = std::make_shared<_Op>(Operators::QuadraticOperators::site_occupation(_dimension, _sites[0]));	
            break;
        case OperatorTypes::OperatorsAvailable::nq:
			if constexpr (std::is_same_v<_T, double>)		
				_operator = std::make_shared<_Op>(Operators::QuadraticOperators::site_nq(_dimension, _sites[0]));
            break;
        case OperatorTypes::OperatorsAvailable::nn:
			if constexpr (std::is_same_v<_T, double>)
			{
				if(_sites.size() == 1)
					_operator = std::make_shared<_Op>(Operators::QuadraticOperators::nn_correlation(_dimension, _sites[0], _sites[0]));
				else if (_sites.size() > 1)
					_operator = std::make_shared<_Op>(Operators::QuadraticOperators::nn_correlation(_dimension, _sites[0], _sites[1]));
			}
            break;
        case OperatorTypes::OperatorsAvailable::nk:
			if constexpr (std::is_same_v<_T, double>)
			{
				if (_sites[0] == 0)
					_operator = std::make_shared<_Op>(Operators::QuadraticOperators::quasimomentum_occupation(_dimension));
			}
            // else 
                // return Operators::QuadraticOperators::quasimomentum_occupation(this->L_, _sites[0]);
            break;
        // !!!!!! RANDOM OPERATOR !!!!!!
        case OperatorTypes::OperatorsAvailable::nr:
        {
			if constexpr (std::is_same_v<_T, double>)
			{
				if (_rgen)
				{
					// create the random operator
					v_1d<double> _rcoefs = _rgen->random<double>(-1.0, 1.0, _dimension);
					_operator = std::make_shared<_Op>(Operators::QuadraticOperators::site_occupation_r(_dimension, _rcoefs));
				}
				else 
					return false;
			}
            break;	
        }
        default:
            return false;
        };
        return true;
    };

    // template instantation
    template bool Operators::OperatorNameParser::createGlobalOperator<double, Operators::Operator<double>>(const std::string&, 
                                                                std::shared_ptr<Operators::Operator<double>>&, 
                                                                bool, bool, randomGen*);
    template bool Operators::OperatorNameParser::createGlobalOperator<cpx, Operators::Operator<cpx>>(const std::string&, 
                                                                std::shared_ptr<Operators::Operator<cpx>>&, 
                                                                bool, bool, randomGen*);
    template bool Operators::OperatorNameParser::createGlobalOperator<double, Operators::OperatorNQS<double>>(const std::string&, 
                                                                std::shared_ptr<Operators::OperatorNQS<double>>&, 
                                                                bool, bool, randomGen*);
    template bool Operators::OperatorNameParser::createGlobalOperator<cpx, Operators::OperatorNQS<cpx>>(const std::string&, 
                                                                std::shared_ptr<Operators::OperatorNQS<cpx>>&, 
                                                                bool, bool, randomGen*);
    // ##########################################################################################################################

    /**
    * @brief Creates global operators from the input strings.
    * 
    * This function parses the input strings to determine the types of operators and the sites they act on.
    * It then creates the corresponding operators and returns them along with their names.
    * 
    * @tparam _T The data type used by the operators.
    * @tparam _Op The template for the operator type.
    * @param _inputs The input strings specifying the operators and sites.
    * @param _usesReal Flag indicating if real operators are allowed.
    * @param _usesHilbert Flag indicating if Hilbert space operators are allowed.
    * @param _rgen Pointer to a random number generator, used for creating random operators.
    * @return A pair containing a vector of shared pointers to the created operators and a vector of their names.
    * @requires _Op is a subclass of GeneralOperator.
    */
    template <typename _T, typename _Op>
    std::pair<v_sp_t<_Op>, strVec> OperatorNameParser::createGlobalOperators(const strVec& _inputs,
                                                                                        bool _usesReal,
                                                                                        bool _usesHilbert,
                                                                                        randomGen* _rgen)
        requires std::is_base_of_v<Operators::GeneralOperator<_T, typename _Op::repType, typename _Op::repTypeV>, _Op> &&
            std::is_same_v<typename _Op::innerType, _T>
    {
        v_sp_t<_Op> ops;
        strVec _outStr = this->parse(_inputs);                  // parse the input strings

        // create the operators
        LOGINFO("Using operators: ", LOG_TYPES::INFO, 4);
        strVec _msgs = {};
        for (int i = 0; i < _outStr.size(); i++)
            _msgs.push_back(STR(i) + ")" + _outStr[i]);
        LOGINFO(_msgs, LOG_TYPES::INFO, 4);
        
        strVec _outOperators = {};								// try to parse the operators
        for (auto& op : _outStr)
        {
            std::shared_ptr<_Op> _opin;

            // check if the operator is valid
            if (this->createGlobalOperator<_T>(op, _opin, _usesReal, _usesHilbert, _rgen))
            {
                if (!_opin->getFun())
                    throw std::runtime_error("The operator: " + op + " is not valid.");

                LOGINFO("Correctly parsed operator: " + op, LOG_TYPES::INFO, 4);
                ops.push_back(_opin);
                _outOperators.push_back(op);
            }
        }
        return std::make_pair(ops, _outOperators);
    }

    // template instantation
    template std::pair<v_sp_t<Operators::Operator<double>>, strVec> Operators::OperatorNameParser::createGlobalOperators<double, Operators::Operator<double>>(const strVec&, bool, bool, randomGen*);
    template std::pair<v_sp_t<Operators::Operator<cpx>>, strVec> Operators::OperatorNameParser::createGlobalOperators<cpx, Operators::Operator<cpx>>(const strVec&, bool, bool, randomGen*);
    template std::pair<v_sp_t<Operators::OperatorNQS<double>>, strVec> Operators::OperatorNameParser::createGlobalOperators<double, Operators::OperatorNQS<double>>(const strVec&, bool, bool, randomGen*);
    template std::pair<v_sp_t<Operators::OperatorNQS<cpx>>, strVec> Operators::OperatorNameParser::createGlobalOperators<cpx, Operators::OperatorNQS<cpx>>(const strVec&, bool, bool, randomGen*);
    // ##########################################################################################################################
};

// ##############################################################################################################################

/**
* @brief Initializes the operator map with available operator names and their corresponding enum values.
* 
* This function iterates over the range of operator types defined in the OperatorsAvailable enum,
* excluding the boundary values E and E2. For each valid operator type, it retrieves the string 
* representation of the operator and maps it to its corresponding enum value in the operator_map_.
*/
void Operators::OperatorNameParser::initMap()
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

// ##############################################################################################################################

/**
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

// ------------------------------------------------------------------------------------------------------------------------------

/**
* @brief Parse the operator name and return the corresponding operator.
* @param _input the input string
* @returns the operator names as strings
*/
strVec Operators::OperatorNameParser::parse(const std::string& _input)
{
	strVec _out = {};

	// go through all the strings
	if(_input.find(OPERATOR_SEP) == std::string::npos) {
		// Assume default format {operator}/1:L:1
		_out = this->parseDefault(_input); 
	} else if(_input.find(OPERATOR_SEP_CORR) != std::string::npos) {
		// This is the correlation operator then, must be handled separately
		// {operator}/{index1}_{index2}_..._{indexN}
		_out = this->parseCorrelationOperator(_input);
	} else if (_input.find(OPERATOR_SEP_RANGE) != std::string::npos) {
		_out = this->parseRangeOperators(_input);
	} else if(_input.find(OPERATOR_SEP_MULT) != std::string::npos) {
		// This is the multiple operator then, must be handled separately
		_out = this->parseMultipleOperators(_input);
	} else if (_input.find(std::string(OPERATOR_SEP) + OPERATOR_SEP_RANDOM) != std::string::npos) {
		_out.push_back(_input);
	} else {
		_out.push_back(this->parseSingleOperator(_input));
	}

	return _out;
}

// ------------------------------------------------------------------------------------------------------------------------------

/**
* @brief Parse the site given as a string and return the corresponding site.
* The format is {site} or {site}/{div} where div is the divisor of the site.
* @param _input the input string
*/
long double Operators::OperatorNameParser::resolveSite(const std::string &_site, bool _usesHilbert)
{
	if(_site.length() == 0) throw std::invalid_argument("The site: " + _site + " is not valid.");

	// get the dimension
	const size_t _dimension = _usesHilbert ? this->Nh_ : this->L_;

	// check if site is L or l already - then return the dimension (L-1)
	if(_site == OPERATOR_SITE || _site == OPERATOR_SITEU) {
		return _dimension - (OPERATOR_SITE_M_1 ? 1 : 0);
	} 
	// check if the site is PI
	else if (_site == OPERATOR_PI) {
		return PI;
	}
	// check if the site can be divided - then divide it
	else if(_site.find(OPERATOR_SEP_DIV) != std::string::npos) {
		// contains L or l
		auto _div = this->resolveSite(splitStr(_site, OPERATOR_SEP_DIV)[1]);
		if (_site.find(OPERATOR_SITEU) != std::string::npos || _site.find(OPERATOR_SITE) != std::string::npos) {
			return _dimension / _div;
		}
		// contains PI
		else if (_site.find(OPERATOR_PI) != std::string::npos) {
			return PI / _div;
		}
	}
	// check if the site is a difference
	else if (_site.find(OPERATOR_SEP_DIFF) != std::string::npos) {
		auto _diff = this->resolveSite(splitStr(_site, OPERATOR_SEP_DIFF)[1]);
		return std::max((long double)0.0, _dimension - _diff - (OPERATOR_SITE_M_1 ? 1 : 0));
	}

	// simply return the site as a number
	auto _siteInt = std::stod(_site);
	if (_siteInt < 0 || _siteInt >= _dimension)
		throw std::invalid_argument("The site: " + _site + " is out of range. The dimension is: " + std::to_string(_dimension));
	return std::stod(_site);
}

// ------------------------------------------------------------------------------------------------------------------------------

/**
* @brief Given single site ranges (something that occurs after /) resolve the sites.
* @note The format is {site1}_{site2}_{site3}..._{siteN}
* @param _sites the sites to resolve
*/
strVec Operators::OperatorNameParser::resolveSitesMultiple(const std::string &_sites, bool _needsIntIdx, bool _usesHilbert)
{
    strVec _out = {};

	if (_sites.find(OPERATOR_SEP_MULT) != std::string::npos) 
	{
		for (const auto& _str : splitStr(_sites, OPERATOR_SEP_MULT))
		{
			long double _site = _needsIntIdx ? size_t(this->resolveSite(_str, _usesHilbert)) : this->resolveSite(_str, _usesHilbert);
			_out.push_back(STRP(_site, 3));
		}	
	} else if (_sites.find(OPERATOR_SEP_RANGE) != std::string::npos) 
	{
		auto _str = splitStr(_sites, OPERATOR_SEP_RANGE);	
		if (_str.size() == 3)
		{
			// throw std::invalid_argument("The range: " + _sites + " is not valid.");		
			auto _start = this->resolveSite(_str[0], _usesHilbert);
			auto _end 	= this->resolveSite(_str[1], _usesHilbert);
			auto _step 	= this->resolveSite(_str[2], _usesHilbert);
			
			// based on the index type, resolve the sites
			for (auto i = _start; i <= _end; i += _step)
				_out.push_back(OPERATOR_INT_CAST_S(_needsIntIdx, i, OperatorNameParser::precision_));
		}
	} else {
		_out.push_back(OPERATOR_INT_CAST_S(_needsIntIdx, this->resolveSite(_sites, _usesHilbert), OperatorNameParser::precision_));
	}
	return _out;
}

// ------------------------------------------------------------------------------------------------------------------------------

/**
* @brief Parse the list of string sites and change them to the integer sites.
* @param _sites the sites to resolve
*/
std::vector<long double> Operators::OperatorNameParser::resolveSites(const strVec &_sites, bool _usesHilbert)
{
	std::vector<long double> _out = {};

	for (const auto& _site : _sites)
		_out.push_back(this->resolveSite(_site, _usesHilbert));
	return _out;
}

// ------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------------

/**
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

/**
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

// ------------------------------------------------------------------------------------------------------------------------------

/**
* @brief Parse the operator name and return the corresponding operator for a single operator.
* The format is {operator}/{index} where the index is the index of the operator (site).
* @param _input the input string
*/
std::string Operators::OperatorNameParser::parseSingleOperator(const std::string & _input)
{
	// get the operator name
	const auto [_opName, _indexStr] = this->resolveOperatorSeparator(_input);

	// check if the operator needs Hilbert space or lattice space
	bool _usesHilbert 				= false;
	bool _needsIntIdx 				= true;
	if (this->operator_map_.contains(_opName))
	{
		_needsIntIdx 				= OperatorTypes::needsIntegerIdx(this->operator_map_.at(_opName));
		_usesHilbert 				= OperatorTypes::needsHilbertSpaceDim(this->operator_map_.at(_opName));
	}

	// site index
	long double _index 				= this->resolveSite(_indexStr, _usesHilbert);


	// return the operator name
	return _opName + OPERATOR_SEP + OPERATOR_INT_CAST_S(_needsIntIdx, _index, OperatorNameParser::precision_);
}

// ------------------------------------------------------------------------------------------------------------------------------

/**
* @brief Parse the operator name and return the corresponding operator for a default operator.
* The format is {operator} and it assumes that we have {operator}/1.L.1.
*/
strVec Operators::OperatorNameParser::parseDefault(const std::string & _input)
{
	return this->parse(_input + "/1:L:1");
}

// #############################################################################################################################

/**
* @brief Parse the operator name and return the corresponding operator for correlation operators.
* The format is {operator}/{index1}_{index2}_..._{indexN} where the index is the index of the operator (site).
*/
strVec Operators::OperatorNameParser::parseCorrelationOperator(const std::string &_input)
{
	// get the operator name
	const auto [_opName, _indexStr] = this->resolveOperatorSeparator(_input);

	bool _usesHilbert 				= false;
	bool _needsIntIdx 				= true;
	if (this->operator_map_.contains(_opName))
	{
		_needsIntIdx 				= OperatorTypes::needsIntegerIdx(this->operator_map_.at(_opName));
		_usesHilbert 				= OperatorTypes::needsHilbertSpaceDim(this->operator_map_.at(_opName));
	}

	// split for the potential indices (for each element there might be multiple sites)
	strVec _potentialIndicies 		= StrParser::split(_indexStr, OPERATOR_SEP_CORR);

	std::vector<strVec> _out 		= {};

	// go through all the potential indices and resolve them
	for (int i = 0; i < _potentialIndicies.size(); ++i)
		_out.push_back(resolveSitesMultiple(_potentialIndicies[i], _needsIntIdx, _usesHilbert));

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

/**
* @brief Parse the operator name and return the corresponding operator for multiple operators.
* The format is {operator}/{index1},{index2},...{indexN} where the index is the index of the operator (site).
*/
strVec Operators::OperatorNameParser::parseMultipleOperators(const std::string &_input)
{
	// get the operator name
	const auto [_opName, _indexStr] = this->resolveOperatorSeparator(_input);

	bool _usesHilbert 				= false;
	bool _needsIntIdx 				= true;
	if (this->operator_map_.contains(_opName))
	{
		_needsIntIdx 				= OperatorTypes::needsIntegerIdx(this->operator_map_.at(_opName));
		_usesHilbert 				= OperatorTypes::needsHilbertSpaceDim(this->operator_map_.at(_opName));
	}

	// split for the potential indices
	strVec _potentialIndicies 		= resolveSitesMultiple(_indexStr, _needsIntIdx, _usesHilbert);

	for (int i = 0; i < _potentialIndicies.size(); ++i)
		_potentialIndicies[i] = _opName + OPERATOR_SEP + _potentialIndicies[i];

	return _potentialIndicies;
}

// #############################################################################################################################


/**
* @brief Parses a string representing an operator with potential range indices and returns a vector of strings with fully qualified operator names.
*
* This function takes an input string representing an operator and its indices, resolves the operator name and indices,
* and returns a vector of strings where each string is a fully qualified operator name with its corresponding index.
*
* @param _input The input string representing the operator and its indices.
* @return strVec A vector of strings containing the fully qualified operator names with their indices.
*/
strVec Operators::OperatorNameParser::parseRangeOperators(const std::string &_input)
{
	// get the operator name
	const auto [_opName, _indexStr] = this->resolveOperatorSeparator(_input);

	bool _usesHilbert 				= false;
	bool _needsIntIdx 				= true;
	if (this->operator_map_.contains(_opName))
	{
		_needsIntIdx 				= OperatorTypes::needsIntegerIdx(this->operator_map_.at(_opName));
		_usesHilbert 				= OperatorTypes::needsHilbertSpaceDim(this->operator_map_.at(_opName));
	}

	// split for the potential indices
	strVec _potentialIndicies 		= resolveSitesMultiple(_indexStr, _needsIntIdx, _usesHilbert);

	for (int i = 0; i < _potentialIndicies.size(); ++i)
		_potentialIndicies[i] = _opName + OPERATOR_SEP + _potentialIndicies[i];

	return _potentialIndicies;
}

// #############################################################################################################################
