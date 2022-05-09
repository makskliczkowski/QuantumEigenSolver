#define PINV
//#define DEBUG
#define DONT_USE_ADAM
#define S_REGULAR
#define RBM_ANGLES_UPD
#define PLOT

#include "include/user_interface/user_interface.h"

int main(const int argc, char* argv[]) {

	auto ui = std::make_unique<rbm_ui::ui<cpx, double>>(argc, argv);
	ui->define_models();
	ui->make_simulation();
	

	return 0;
}