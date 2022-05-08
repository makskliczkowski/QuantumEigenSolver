
#include "include/user_interface/user_interface.h"

template<typename _type, typename _hamtype>
void testModel();



int main(const int argc, char* argv[]) {

	auto ui = std::make_unique<rbm_ui::ui<cpx, cpx>>(argc, argv);
	ui->define_models();
	ui->make_simulation();
	

	return 0;
}