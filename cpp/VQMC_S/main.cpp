
#define DEBUG

#include "include/user_interface/user_interface.h"

int main(const int argc, char* argv[]) {
	#ifdef LOG_FILE
		SET_LOG_TIME();
	#endif

	auto ui = std::make_unique<UI>(argc, argv);
	//ui->make_simulation();
	ui->funChoice();

	return 0;
}