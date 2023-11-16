
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
	ui->funChoice();

	// create GOE matrix and test Lanczos on it
	//randomGen* r			= new randomGen;
	//auto N					= 500;
	//auto Nkrylov			= 500;
	//arma::Mat<cpx> matrix	= r->CUE(N, N);
	//matrix = matrix + matrix.t();
	//arma::Col<double> eigvalA;
	//arma::Mat<cpx> eigvecA;
	//arma::Col<double> eigvalM;
	//arma::Mat<cpx> eigvecM;
	//// eig_sym
	//arma::eig_sym(eigvalA, eigvecA, matrix);
	//eigvalA.save("testArma_val.txt", arma::raw_ascii);
	//eigvecA.save("testArma_vec.txt", arma::raw_ascii);
	//// Lanczos
	//LanczosMethod<cpx>::diagS(eigvalM, eigvecM, matrix, Nkrylov, r);
	//delete r;
	//eigvalM.save("testMine_val.txt", arma::raw_ascii);
	//eigvecM.save("testMine_vec.txt", arma::raw_ascii);

	return 0;
}