#pragma once
#include "Include/files.h"
#include "Include/directories.h"

/*******************************
* Contains the possible methods
* for logging the info etc.
*******************************/

// ######################################################## L O G   L E V E L S ########################################################

#define LOG_LVL0		""
#define LOG_LVL1		"\t->"
#define LOG_LVL2		"\t\t->"
#define LOG_LVL3		"\t\t\t->"
#define LOG_LVL4		"\t\t\t\t->"
static int LASTLVL		= 0;

/*
* @brief prints log level based on a given input
* @param _lvl tabulation level
*/
inline void logLvl(unsigned int _lvl) {
	while (_lvl--)
		std::cout << "\t";
	std::cout << "->";
}

enum LOG_TYPES 
{
	INFO,
	TIME,
	ERROR,
	TRACE,
	CHOICE,
	FINISH,
	WARNING
};

BEGIN_ENUM(LOG_TYPES)
{
	DECL_ENUM_ELEMENT(INFO),
	DECL_ENUM_ELEMENT(TIME),
	DECL_ENUM_ELEMENT(ERROR),
	DECL_ENUM_ELEMENT(TRACE),
	DECL_ENUM_ELEMENT(CHOICE),
	DECL_ENUM_ELEMENT(FINISH),
	DECL_ENUM_ELEMENT(WARNING)
}
END_ENUM(LOG_TYPES);

#define LOG_INFO(TYP)								"[" + SSTR(getSTR_LOG_TYPES(TYP)) + "]"

// --- create log file if necessary ---
#ifdef LOG_FILE
	#define LOG_DIR									SSTR("LOG") + kPS
	static inline std::string LOG_FILENAME			= "log";
	static clk::time_point LOG_TIME;
#endif

inline void SET_LOG_TIME() {
#ifdef LOG_FILE
	createDir(LOG_DIR);
	LOG_TIME			=		clk::now();
	LOG_FILENAME		=		"." + kPS + "LOG" + kPS + "log_" + prettyTime(LOG_TIME) + ".txt";
	std::ofstream file(LOG_FILENAME);
#endif
};

/*
* @brief prints log info based on a given input message and type
* @param _msg message to be logged
* @param _typ the type of message
* @param _lvl tabulation level
*/
inline void LOGINFO(const std::string& _msg, LOG_TYPES _typ, unsigned int _lvl) {
	logLvl(_lvl); 
	std::cout << "[" << getSTR_LOG_TYPES(_typ) << "]" << _msg << std::endl;
#ifdef LOG_FILE
	std::ofstream _file;
	openFile(_file, LOG_FILENAME, std::ios::app);
	// check level of log
	if (_lvl > 0)
		_file << "->";
	while (_lvl--)
		_file << "\t";
	// write to the file
	_file << "[" << getSTR_LOG_TYPES(_typ) << "]" << _msg << std::endl;
	_file.close();
#endif
}

/*
* @brief prints log info based on a given input message and type
* @param _msg message to be logged
* @param _
*/
inline void LOGINFOG(const std::string& _msg, LOG_TYPES _typ, unsigned int _lvl) {
	logLvl(LASTLVL + _lvl);
	std::cout << "[" << getSTR_LOG_TYPES(_typ) << "]" << _msg << std::endl;
#ifdef LOG_FILE
	std::ofstream _file;
	openFile(_file, LOG_FILENAME, std::ios::app);
	// check level of log
	auto _lvl = LASTLVL + _lvl;
	if (_lvl > 0)
		_file << "->";
	while (_lvl--)
		_file << "\t";
	// write to the file
	_file << "[" << getSTR_LOG_TYPES(_typ) << "]" << _msg << std::endl;
	_file.close();
#endif
}