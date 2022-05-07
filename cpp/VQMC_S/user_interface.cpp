#include "include/user_interface/user_interface.h"

// --------------------------------------------------------   				 USER INTERFACE   				  --------------------------------------------------------

/*
* @brief Find a given option in a vector of string given from cmd parser
* @param vec vector of strings from cmd
* @param option the option that we seek
* @returnsvalue for given option if exists, if not an empty string
*/
string user_interface::getCmdOption(const v_1d<string>& vec, string option) const
{
	if (auto itr = std::find(vec.begin(), vec.end(), option); itr != vec.end() && ++itr != vec.end())
		return *itr;
	return string();
}

/*
* @brief If the commands are given from file, we must treat them the same as arguments
* @param filename"> the name of the file that contains the command line 
* @returns
*/
std::vector<string> user_interface::parseInputFile(string filename) {
	v_1d<string> commands;
	std::ifstream inputFile(filename);
	string line = "";
	if (!inputFile.is_open()) {
		std::cout << "Cannot open a file " + filename + " that I could parse. Setting all parameters to default. Sorry :c \n";
		this->set_default();
	}
	else {
		if (std::getline(inputFile, line)) {
			// saving lines to out vector if it can be done, then the parser shall treat them normally
			commands = split_str(line, " ");										
		}
	}
	return std::vector<string>(commands.begin(), commands.end());
}

