#pragma once
#include <exception>
#include <stdexcept>
#include "signatures.h"
#include "str.h"

/*******************************
* Contains the possible methods
* for handling the exceptions.
*******************************/

// ######################################################## E X C E P T I O N S ########################################################

class ExceptionHandler {

	enum EXCEPTIONS 
	{
		UNDEFINED	=	0,
		RUNTIME		=	-1,
		BAD_ALOC	=	-2,
		EXCEPTION	=	-3,
		FILE		=	-4
	};

	BEGIN_ENUM_INLINE(EXCEPTIONS)
	{
		DECL_ENUM_ELEMENT(UNDEFINED),
		DECL_ENUM_ELEMENT(RUNTIME),
		DECL_ENUM_ELEMENT(BAD_ALOC),
		DECL_ENUM_ELEMENT(EXCEPTION),
		DECL_ENUM_ELEMENT(FILE)
	}
	END_ENUM_INLINE(EXCEPTIONS, ExceptionHandler);
public:
	static void printException(const std::string& _what, const std::string& _msg, EXCEPTIONS _ex = UNDEFINED);

	static void handleExceptions(std::exception_ptr _ePtr, const std::string& _msg);
};

#define BEGIN_CATCH_HANDLER				try{
#define END_CATCH_HANDLER(message)		}catch(...){ ExceptionHandler::handleExceptions(std::current_exception(), message); }
