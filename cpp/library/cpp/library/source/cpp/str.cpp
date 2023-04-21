#include "../src/Include/str.h"

// ######################################################## STRING RELATED FUNCTIONS ########################################################

/*
* @brief Splits string according to the delimiter
* @param A a string to be split
* @param delimiter A delimiter. Default = '\\t'
* @return Split string
*/
strVec splitStr(const std::string& s, std::string delimiter)
{
	unsigned long long pos_start = 0;
	unsigned long long pos_end;
	unsigned long long delim_len = delimiter.length();
	std::string token;
	strVec res = {};

	while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
		token = s.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}
	res.push_back(s.substr(pos_start));
	return res;
}

// ######################################################## String vector ########################################################

/*
* We want to handle files so let's make the c-way input a string. This way we will parse the command line arguments
* @param argc number of main input arguments
* @param argv main input arguments
* @returns vector of strings with the arguments from command line
*/
strVec fromPtr(int argc, char** argv, unsigned int offset)
{
	v_1d<std::string> tmp(argc - offset, "");
	for (auto i = 0; i < argc - offset; i++)
		tmp[i] = argv[i + offset];
	return tmp;
};
