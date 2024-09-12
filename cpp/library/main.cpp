// %%%%%%%%%%%%% L O G %%%%%%%%%%%%%%%
//#define LOG_FILE					//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//#ifdef _DEBUG
 //#	include "vld.h"
//#endif
constexpr auto ARMA_VEC_SEED = 0;

// %%%%%%%%%%%%% N Q S %%%%%%%%%%%%%%%
/*#define NQS_RBM_USESR*/				//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include "include/user_interface/user_interface.h"


int main(const int argc, char* argv[]) 
{
	// set the seed to a random value
	if (ARMA_VEC_SEED)
		arma::arma_rng::set_seed(arma::arma_rng::seed_type(ARMA_VEC_SEED));
	else
		arma::arma_rng::set_seed_random();

	SET_LOG_TIME();

	auto ui = std::make_unique<UI>(argc, argv);
	ui->funChoice();

	// create GOE matrix and test Lanczos on it
	//randomGen* r			= new randomGen(169);
	//auto N					= std::pow(2, 5);
	//auto Nkrylov			= 30;
	//arma::Mat<cpx> matrix	= r->CUE(N, N);

	//matrix = matrix + matrix.t();
	//arma::Col<double> eigvalA;
	//arma::Mat<cpx> eigvecA;
	//arma::Col<double> eigvalM;
	//arma::Mat<cpx> eigvecM;
	//arma::Mat<cpx> krylovMat;
	//// eig_sym
	//arma::eig_sym(eigvalA, eigvecA, matrix);
	//eigvalA.save("testArma_val.txt", arma::raw_ascii);
	//eigvecA.save("testArma_vec.txt", arma::raw_ascii);
	//// Lanczos
	//LanczosMethod<cpx>::diagS(eigvalM, eigvecM, matrix, Nkrylov, r, krylovMat);
	//delete r;
	//eigvalM.save("testMine_val.txt", arma::raw_ascii);
	//eigvecM.save("testMine_vec.txt", arma::raw_ascii);

	//auto groundED = eigvecA.col(0);
	//groundED.print("ED=");
	//double Sz_0 = 0;
	//for (int i = 0; i < N; i++)
	//{
	//	auto [_, Sz] = Operators::sigma_z(i, 5, { 0 });
	//	Sz_0 += Sz * std::abs(groundED[i] * algebra::conjugate(groundED[i]));
	//}
	//LOGINFO(VEQ(Sz_0), LOG_TYPES::INFO, 0);
	//auto groundKr = LanczosMethod<cpx>::trueState(eigvecM, krylovMat, 0);
	//groundKr.print("Krylov=");
	//Sz_0 = 0;
	//for (int i = 0; i < N; i++)
	//{
	//	auto [_, Sz] = Operators::sigma_z(i, 5, { 0 });
	//	Sz_0 += Sz * std::abs(groundKr[i] * algebra::conjugate(groundKr[i]));
	//}
	//LOGINFO(VEQ(Sz_0), LOG_TYPES::INFO, 0);

	return 0;
}