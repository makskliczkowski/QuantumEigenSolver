
//#define DEBUG
#define USE_SR
//#define USE_ADAM
//#define USE_RMS

#define RBM_ANGLES_UPD
#define PLOT
#define SPIN


#ifdef USE_SR
	#define PINV
	//#define S_REGULAR
#endif






#include "include/user_interface/user_interface.h"

int main(const int argc, char* argv[]) {

	auto ui = std::make_unique<rbm_ui::ui<cpx, double>>(argc, argv);
	ui->define_models();
	ui->make_simulation();
	

	return 0;
}