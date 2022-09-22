
//#define DEBUG
#include "include/user_interface/user_interface.h"

int main(const int argc, char* argv[]) {

	auto ui = std::make_unique<rbm_ui::ui<cpx, double>>(argc, argv);
	//ui->define_models();
	//ui->make_simulation();
	//ui->make_mc_classical(50, 0.1, 0.05, 0.06);
	ui->make_simulation_symmetries();

	return 0;
}