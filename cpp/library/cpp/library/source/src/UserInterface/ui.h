#pragma once
#ifndef UI_H
#define UI_H

#ifndef COMMON_H
#include "../common.h"
#endif

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// sets the specific option that is self-explanatory
#define SETOPTION(n, S)							this->setOption(this->n.S##_, argv, SSTR(#S))
#define SETOPTIONV(n, S, v)						this->setOption(this->n.S##_, argv, SSTR(v)	)
// sets the option with steps etc.
#define SETOPTION_STEP(x, S)					SETOPTION(x, S);							\
												SETOPTION(x, S##0);							\
												SETOPTION(x, S##s);							\
												SETOPTION(x, S##n)
// creates both default variable (_ at front) and used variable (_ at the end) for the UI model
#define UI_PARAM_CREATE_DEFAULT(PAR, TYP, VAL)	static const	TYP _##PAR = VAL; TYP PAR##_ = VAL 
// creates both default variable (_ at front) and used variable (_ at the end) for the UI model - nonstatic doubles etc.
#define UI_PARAM_CREATE_DEFAULTD(PAR, TYP, VAL)	const			TYP _##PAR = VAL; TYP PAR##_ = VAL 
// sets default value for single parameters 
#define UI_PARAM_SET_DEFAULT(PAR)				this->##PAR##_	=	this->_##PAR
// specifies parameters in the UI that distinguish between step in range, starting point, number of steps and disorder strength
#define UI_PARAM_STEP(TYP, PAR, VAL)			UI_PARAM_CREATE_DEFAULTD(PAR, TYP, VAL);	\
												UI_PARAM_CREATE_DEFAULTD(PAR##0,TYP,0.0);	\
												UI_PARAM_CREATE_DEFAULTD(PAR##s,TYP,0.0);	\
												UI_PARAM_CREATE_DEFAULTD(PAR##n,int,1)		
// allows to set default values for parameters that precise range, disorder strength etc.
#define UI_PARAM_SET_DEFAULT_STEP(PAR)			this->##PAR##_	=	this->_##PAR;			\
												this->##PAR##0_	=	this->_##PAR##0;		\
												this->##PAR##n_	=	this->_##PAR##n;		\
												this->##PAR##s_	=	this->_##PAR##s
// adds parameters to the map
#define UI_PARAM_MAP(p, v, f)					{ #p					, std::make_tuple(#v , f)								},	\
												{ SSTR(#p ) + SSTR("0")	, std::make_tuple("0.0", FHANDLE_PARAM_HIGHERV(-1e-15))	},	\
												{ SSTR(#p ) + SSTR("s")	, std::make_tuple("0.0", FHANDLE_PARAM_DEFAULT)			},	\
												{ SSTR(#p ) + SSTR("n")	, std::make_tuple("1.0", FHANDLE_PARAM_HIGHER0)			}
// adds other varables to the map
#define UI_OTHER_MAP(p, v, f)					{ #p					, std::make_tuple(STRP(v, 2), f)						}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// -------------------------------------------------------- Make a User interface class --------------------------------------------------------

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
using HANDLE_FUN_TYPE = std::function<std::string(std::string)>;
// #########################################################
#define HANDLE_PARAM_DEFAULT					SSTR("")
inline	std::string		FHANDLE_PARAM_DEFAULT	(std::string s)	
{ 
	return HANDLE_PARAM_DEFAULT; 
};
// #########################################################
#define HANDLE_PARAM_HIGHERV(val)				SSTR("Value must be higher than : ") + STRP(val, 2)
inline	HANDLE_FUN_TYPE	FHANDLE_PARAM_HIGHERV	(double _low = 0.0)
{
	return [=](std::string s) { if (stod(s) <= _low) return HANDLE_PARAM_HIGHERV(_low); return HANDLE_PARAM_DEFAULT; };
};
inline	std::string		FHANDLE_PARAM_HIGHER0	(std::string s)
{
	return FHANDLE_PARAM_HIGHERV()(s);
}
// #########################################################
#define HANDLE_PARAM_BETWEEN(low, high)			SSTR("Value must be between : ") + STRP(low, 2) + SSTR(" and ") + STRP(high, 2)
inline	HANDLE_FUN_TYPE	FHANDLE_PARAM_BETWEEN	(double _low = -1.0, double _high = 1.0) 
{
	return [=](std::string s) { auto v = std::stod(s); if (v > _high || v < _low) return HANDLE_PARAM_BETWEEN(_low, _high); return HANDLE_PARAM_DEFAULT; };
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class UserInterface {
protected:
	typedef v_1d<std::string> cmdArg;
	typedef std::unordered_map<std::string, std::tuple<std::string, std::function<std::string(std::string)>>> cmdMap;

	std::string mainDir									= "." + kPS;																		// main directory - to be saved onto
	uint threadNum										= 1;	
	int chosenFun										= -1;												// chosen function to be used later
	bool quiet											= false;
	
	// ----------------------- CHOICES and OPTIONS and DEFAULTS -----------------------
	cmdMap defaultParams;																					// default parameters

	// ----------------------- FUNCTIONS -----------------------

	std::string getCmdOption(cmdArg& vec, std::string option) const;				 						// get the option from cmd input
	std::string setDefaultMsg(std::string v, std::string opt, std::string message, const cmdMap& map) const;// setting value to default and sending a message
	
	template <typename _T>
	bool setOption(_T& value, cmdArg& argv, std::string choice);											// set an option

	/*
	* @brief initialize the UI
	*/
	void init(int argc, char** argv) {
		strVec input = fromPtr(argc, argv, 1);																// change standard input to vec of strings
		//input = std::vector<string>(input.begin()++, input.end());										// skip the first element which is the name of file
		if (std::string option = this->getCmdOption(input, "-f"); option != "")
			input = this->parseInputFile(option);															// parse input from file
		this->parseModel((int)input.size(), input);
	};
	virtual void setDefaultMap()						= 0;
public:
	virtual ~UserInterface() = default;

	// general functions to override
	virtual void exitWithHelp()							= 0;

	// ----------------------- REAL PARSING -----------------------
	virtual void funChoice()							= 0;												// allows to choose the method without recompilation of the whole code
	virtual void parseModel(int argc, cmdArg& argv)		= 0;												// the function to parse the command line
	virtual cmdArg parseInputFile(std::string filename);													// if the input is taken from file we need to make it look the same way as the command line does
	
	// ----------------------- HELPING FUNCIONS -----------------------
	virtual void setDefault()							= 0;										 		// set default parameters
	
	// ----------------------- NON-VIRTUALS -----------------------
};

/*
* @brief sets option from a given cmd options
* @param value a value to be set onto
* @param argv arguments to find the corresponding option
* @param choice chosen option
*/
template<typename _T>
inline bool UserInterface::setOption(_T& value, cmdArg& argv, std::string choice)
{
	std::string option	=	this->getCmdOption(argv, "-" + choice);
	bool setVal			=	option != "";
	option				=	this->setDefaultMsg(option, choice, choice + ":\n", defaultParams);
	if (setVal)	setVal	=	!option.empty();
	if (setVal) value	=	static_cast<_T>(stod(option));
	return setVal;
}

template<>
inline bool UserInterface::setOption<std::string>(std::string& value, cmdArg& argv, std::string choice) {
	std::string option	=	this->getCmdOption(argv, "-" + choice);
	bool setVal			=	option.empty();
	if (setVal)
		value = this->setDefaultMsg(option, std::string(choice), std::string(choice + ":\n"), defaultParams);
	else
		value			=	option;
	return !setVal;
}

#endif // !UI_H


