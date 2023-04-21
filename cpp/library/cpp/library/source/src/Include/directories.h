#pragma once

/*******************************
* Contains the possible methods
* for directories creation etc.
*******************************/

// --- FILESYSTEM : DIRECTORY CREATION ---
#ifdef __has_include
#	if __has_include(<filesystem>)
#		include <filesystem>
#   	define have_filesystem 1
		namespace fs = std::filesystem;
#	elif __has_include(<experimental/filesystem>)
#		include <experimental/filesystem>
#include <iostream>
#    	define have_filesystem 1
#    	define experimental_filesystem
		namespace fs = std::experimental::filesystem;
#	else
#		define have_filesystem 0
#	endif
#endif

// --- K PATH SEPARATOR : DIRECTORY SEPARATOR ---
static const char* kPSep = 
#ifdef _WIN32 
    R"(\)"; 
#else 
    "/"; 
#endif
const std::string kPS				=				std::string(kPSep);


// ############################################################# DIRECTORIES #############################################################

/*
* @brief Creates a single directory given a string path
* @param dir the directory
*/
void createDir(const std::string& dir);

/*
* @brief Creates a variadic directory set given a string paths
* @param dir the directory
*/
template <typename... _Ty>
inline void createDirs(const std::string& dir, const _Ty&... dirs) {
	createDir(dir);
	createDirs(dirs...);
}


