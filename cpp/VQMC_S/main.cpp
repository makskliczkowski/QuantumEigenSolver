
//#define DEBUG
#define USE_SR
//#define USE_ADAM
//#define USE_RMS

//#define RBM_ANGLES_UPD
#define PLOT
//#define SPIN


#ifdef USE_SR
	//#define PINV
	#define S_REGULAR
#endif

#ifdef PINV
	constexpr auto pinv_tol = 1e-5;
	#ifdef S_REGULAR
		#undef S_REGULAR
	#endif
#elif defined S_REGULAR 
	constexpr double lambda_0_reg = 100;
	constexpr double b_reg = 0.9;
	constexpr double lambda_min_reg = 1e-4;
	#ifdef PINV
		#undef PINV
	#endif
#endif




#include "include/user_interface/user_interface.h"

int main(const int argc, char* argv[]) {

	auto ui = std::make_unique<rbm_ui::ui<cpx, double>>(argc, argv);
	ui->define_models();
	ui->make_simulation();
	

	return 0;
}