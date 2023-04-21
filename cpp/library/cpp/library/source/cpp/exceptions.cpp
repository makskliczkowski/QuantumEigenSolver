#include "../src/flog.h"

/*
* @brief simple exception printer
*/
void ExceptionHandler::printException(const std::string& _what, const std::string& _msg, EXCEPTIONS _ex)
{
	std::cout << LOG_LVL0 << getSTR_EXCEPTIONS(_ex) << std::endl;
	std::cout << LOG_LVL1 << _what << std::endl;
	std::cout << LOG_LVL1 << _msg << std::endl;
	exit(static_cast<int>(_ex));
}

/*
* @brief handles the most common exceptions
*/
void ExceptionHandler::handleExceptions(std::exception_ptr _ePtr, const std::string& _msg)
{
	try {
		if (_ePtr) std::rethrow_exception(_ePtr);
	}
	catch (const std::runtime_error& err) {
		printException(err.what(), _msg, EXCEPTIONS::RUNTIME);
	}
	catch (const std::bad_alloc& err) {
		printException(err.what(), _msg, EXCEPTIONS::BAD_ALOC);
	}
	catch (const std::exception& err) {
		printException(err.what(), _msg, EXCEPTIONS::EXCEPTION);
	}
	//catch (const std::ifstream::failure& err) {
		//printException(err.what(), _msg, EXCEPTIONS::FILE);
	//}
	catch (...) {
		printException("UNKNOWN EXCEPTION", _msg, EXCEPTIONS::BAD_ALOC);
	};
}