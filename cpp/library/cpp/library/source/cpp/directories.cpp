#include "../src/flog.h"

/*
* @brief create single specified directory
*/
void createDir(const std::string& dir)
{
	if (!dir.empty())
		if(fs::create_directories(dir))
			LOGINFO("CREATED: " + dir, LOG_TYPES::INFO, 1);	
}