#define DONT_USE_ADAM
//#define DEBUG

#ifdef DEBUG
//#define DEBUG_BINARY
	#ifdef DEBUG_RBM
	//#define DEBUG_RBM_SAMP
	//#define DEBUG_RBM_LCEN
	//#define DEBUG_RBM_GRAD
	//#define DEBUG_RBM_DRVT
	#endif
#else
	#include <omp.h>
#endif

#include "include/rbm.h"
#include "include/lattices/square.h"
#include "include/models/ising.h"
#include "include/models/heisenberg_dots.h"


template<typename _type, typename _hamtype>
void testModel() {
	
	// define lattice
	int maxEd = 12;
	int Lx = 10;
	int Ly = 1;
	int Lz = 1;
	int dim = 1;
	int _BC = 1;
	auto lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
	auto lattice_type = lat->get_type();
	auto Ns = lat->get_Ns();
	stout << VEQ(lattice_type) << EL;

	// define model
	double J = 1.0;
	double J0 = 0;
	double h = 0.1;
	double w = 0.0;
	double g = 0.0;
	double g0 = 0.0;
	// for dots
	v_1d<int> positions = { 0 , Lx - 1};
	const auto phis = vec({ 0.1, 0.1});
	const auto thetas = vec({ 0.8, 0.5});
	const vec J_dot = { 1.0,0.0, -1.0 };
	double J0_dot = 0.0;

	//auto ham = std::make_shared<IsingModel<_hamtype>>(J, J0, g, g0, h, w, lat);
	auto ham = std::make_shared<Heisenberg<_hamtype>>(J, J0, g, g0, h, w, lat);
	//auto ham = std::make_shared<Heisenberg_dots<_hamtype>>(J, J0, g, g0, h, w, lat, positions, J_dot, J0_dot);
	//ham->set_angles(phis, thetas);

	double ground_ed = 0;
	if (lat->get_Ns() <= maxEd) {
		ham->hamiltonian();
		ham->diag_h(false);
		auto info = ham->get_info();
		stout << VEQ(info) << EL;
		stout << "------------------------------------------------------------------------" << EL;
		stout << "GROUND STATE ED:" << EL;
		SpinHamiltonian<_hamtype>::print_state_pretty(ham->get_eigenState(0), lat->get_Ns(), 0.1);
		stout << "------------------------------------------------------------------------" << EL;
		ground_ed = std::real(ham->get_eigenEnergy(0));
		double sz = 0;
		for (auto i = 0; i < ham->get_hilbert_size(); i++) {
			auto value = ham->get_eigenStateValue(0, i);
			if (abs(value) < 1e-7) continue;
			for (int j = 0; j < lat->get_Ns(); j++) {
				sz += abs(abs(value) * value) * (checkBit(i, lat->get_Ns() - 1 - j) ? 1.0 : -1.0);
			}
		}
		stout << "ED sz: " << VEQ(sz/lat->get_Ns()) << EL;
	}

	// define rbm state
	u64 nhidden = Lx * Ly * Lz;
	u64 nvisible = 2 * nhidden;
	size_t batch = std::pow(2, 8);
	size_t thread_num = 16;
	auto lr = 1e-2;


	auto phi = std::make_unique<rbmState<_type, _hamtype>>(nvisible, nhidden, ham, lr, batch, thread_num);
	auto rbm_info = phi->get_info();
	stout << VEQ(rbm_info) << EL;

	// monte carlo
	auto mcSteps = 400;
	size_t n_blocks = 200;
	auto n_therm = size_t(0.1 * n_blocks);
	size_t block_size = 16;//std::pow(2, 3);
	auto n_flips = 1;
	auto energies = phi->mcSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
	//auto energies2 = phi.mcSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
	// print dem ens
	std::ofstream file("energies" + ham->get_info() + ".dat");
	for (int i = 0; i < energies.size(); i++)
		printSeparatedP(file, '\t', 8, true, 5, i, energies[i].real());
	//for (int i = 0; i < energies2.size(); i++)
	//	printSeparatedP(file, '\t', 8, true, 5, i, energies[i].real());
	file.close();


	energies = v_1d<_type>(energies.end() - block_size, energies.end());
	_type standard_dev = stddev<_type>(energies);
	stout << "\t\t->ENERGIES" << EL;
	_type ground_rbm = 0;
	for (const auto& e : energies)
		//if (std::real(ground_rbm) > std::real(e)) ground_rbm = e;
		ground_rbm += e;
	ground_rbm /= double(energies.size());

	// ------------------- check ground state
	phi->avSampling(200, 50, 16, n_flips);

	stout << "\t\t\t->" << VEQ(ground_rbm) << "+-" << standard_dev << EL;
	if (lat->get_Ns() <= maxEd) {
		stout << "\t\t\t->" << VEQ(ground_ed) << EL;
		auto relative_error = abs(std::real(ground_ed - ground_rbm)) / abs(ground_ed) * 100;
		stout << "\t\t\t->" << VEQ(relative_error) << "%" << EL;
	}

}	


int main() {

	testModel<cpx, double>();

	return 0;
}