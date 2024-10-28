#include "../include/algebra/operators.h"

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

// ------------------------------------------------------------------------------------------------------------------------------

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

/*
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
	auto _siteInt = std::stold(_site);
	if (_siteInt < 0 || _siteInt >= _dimension)
		throw std::invalid_argument("The site: " + _site + " is out of range. The dimension is: " + std::to_string(_dimension));
	return std::stold(_site);
}

// ------------------------------------------------------------------------------------------------------------------------------

/*
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

/*
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

/*
* @brief Parse the operator name and return the corresponding operator for a default operator.
* The format is {operator} and it assumes that we have {operator}/1.L.1.
*/
strVec Operators::OperatorNameParser::parseDefault(const std::string & _input)
{
	return this->parse(_input + "/1:L:1");
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

/*
* @brief Multiple operators of the form 
* {operator}/{index1},{index2},...{indexN} where the index is the index of the operator (site).
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

/*
* @brief Parse the operator name and return the corresponding operator for range operators.
* The format is {operator}/{start}.{stop}_{step}
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
