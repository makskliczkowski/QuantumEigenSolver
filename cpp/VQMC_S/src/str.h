#include <string>
#include <vector>

template<class T>
using v_3d = std::vector<std::vector<std::vector<T>>>;				// 3d double vector
template<class T>
using v_2d = std::vector<std::vector<T>>;							// 2d double vector
template<class T>
using v_1d = std::vector<T>;										// 1d double vector

//! -------------------------------------------------------- STRING RELATED FUNCTIONS --------------------------------------------------------

/*
* Splits string according to the delimiter
* @param s a string to be split
* @param delimiter a delimiter. Default = '\\t'
* @return splitted string
*/
v_1d<std::string> split_str(const std::string& s, std::string delimiter = "\t");

/*
* We want to handle files so let's make the c-way input a string. This way we will parse the command line arguments
* @param argc number of main input arguments 
* @param argv main input arguments 
* @returns vector of strings with the arguments from command line
*/
inline v_1d<std::string> changeInpToVec(int argc, char** argv){
	// -1 because first is the name of the file
	v_1d<std::string> tmp(argc - 1, "");										
	for (int i = 0; i < argc - 1; i++)
		tmp[i] = argv[i + 1];
	return tmp;
};
