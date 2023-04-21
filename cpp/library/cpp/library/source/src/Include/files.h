#pragma once
#include <ios>
#include <iostream>
#include <fstream>
#include "time.h"

/*******************************
* Contains the possible methods
* for handling files and print.
*******************************/

// ########################################################			    FILE AND STREAMS			########################################################

/*
* @brief Opens a file that is specified previously
* @param fileName name of the file to be opened
* @param mode std::ios_base::openmode of file to be opened
*/
template <typename T>
inline int openFile(T& file, std::string fileName, std::ios_base::openmode mode = std::ios::out) {
    BEGIN_CATCH_HANDLER
        file.open(fileName, mode);
        if (!file.is_open())
            throw ("Couldn't open a file: " + fileName + "\n");
    END_CATCH_HANDLER("Exception opening/reading/closing file")
    return 1;
}

// ######################################################## PRINT SEPARATED

template <typename Type>
inline void printSepP(std::ostream& output, char sep, uint16_t width, uint16_t prec, Type arg) {
	output.width(width); output << str_p(arg, prec) << std::string(1, sep);
}
template <typename Type, typename... Types>
inline void printSepP(std::ostream& output, char sep, uint16_t width, uint16_t prec, Type arg, Types... elements) {
	printSepP(output, sep, width, prec, arg);	printSepP(output, sep, width, prec, elements...);
}

/*
*@brief printing the separated number of variables using the variadic functions initializer - includes precision call
*@param output output stream
*@param sep separator to be used
*@param width width of one element column for printing
*@param endline shall we add endline at the end?
*@param elements at the very end we give any type of variable to the function
*/
template <typename... Types>
inline void printSeparatedP(std::ostream& output, char sep, uint16_t width, bool endline, uint16_t prec, Types... elements) {
	printSepP(output, sep, width, prec, elements...); if (endline) output << std::endl;
}

/*
* @brief printing the separated number of variables using the variadic functions initializer - LAST CALL
* @param output output stream
* @param sep separator to be used
* @param width width of one element column for printing
* @param endline shall we add endline at the end?
* @param elements at the very end we give any type of variable to the function
*/
template <typename... Types>
inline void printSeparated(std::ostream& output, char sep, uint16_t width, bool endline, Types... elements) {
	printSepP(output, sep, width, 8, elements...); if (endline) output << std::endl;
}

