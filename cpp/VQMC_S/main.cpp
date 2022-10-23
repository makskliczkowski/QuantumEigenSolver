

#include "include/user_interface/user_interface.h"
#include "user_interface.cpp"

int main(const int argc, char* argv[]) {

	auto ui = std::make_unique<rbm_ui::ui<cpx, double>>(argc, argv);
	//ui->define_models();
	//ui->make_simulation();
	ui->make_simulation_symmetries();
	//ui->make_symmetries_test();

	// ------------ CLASSICAL -----------
	// ferro
	//ui->make_mc_classical();

	return 0;
}