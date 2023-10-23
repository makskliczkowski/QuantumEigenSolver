
// %%%%%%%%%%%%% L O G %%%%%%%%%%%%%%%
#define DEBUG						//
//#define LOG_FILE					//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#ifdef _DEBUG
#include "vld.h"
#endif

// %%%%%%%%%%%%% N Q S %%%%%%%%%%%%%%%
/*#define NQS_RBM_USESR*/				//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include "include/user_interface/user_interface.h"

int main(const int argc, char* argv[]) {

	SET_LOG_TIME();

	auto ui = std::make_unique<UI>(argc, argv);
	//ui->make_simulation();
	ui->funChoice();

	return 0;
}