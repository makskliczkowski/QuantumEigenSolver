

#include "include/user_interface/user_interface.h"
#include "user_interface.cpp"

int main(const int argc, char* argv[]) {

	auto ui = std::make_unique<UI>(argc, argv);
	//ui->define_models();
	//ui->make_simulation();
	ui->funChoice();

	return 0;
}