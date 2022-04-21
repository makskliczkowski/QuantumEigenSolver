#define DONT_USE_ADAM


#include "include/rbm.h"
#include "include/lattices/square.h"
#include "include/models/ising.h"






int main() {

	// define lattice
	int maxEd = 10;
	int Lx = 10;
	int Ly = 1;
	int Lz = 1;
	int dim = 1;
	int _BC = 1;
	auto lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
	auto lattice_type = lat->get_type();
	stout << VEQ(lattice_type) << EL;

	// define model
	double J = -2;
	double J0 = 0;
	double h = 0.1;
	double g = -1;
	double w = 0.0;
	double g0 = 0.0;
	auto ham = std::make_shared<IsingModelDis<cpx>>(J, J0, g, g0, h, w, lat);
	double ground_ed = 0;
	if (lat->get_Ns() <= maxEd) {
		ham->hamiltonian();
		ham->diag_h(false);
		auto info = ham->get_info();
		stout << VEQ(info) << EL;
		stout << "------------------------------------------------------------------------" << EL;
		stout << "GROUND STATE ED:" << EL;
		SpinHamiltonian<cpx>::print_state_pretty(ham->get_eigenState(0), lat->get_Ns());
		stout << "------------------------------------------------------------------------" << EL;
		ground_ed = std::real(ham->get_eigenEnergy(0));
	}

	// define rbm state
	u64 nhidden = Lx * Ly * Lz;
	u64 nvisible = 2 * nhidden;
	size_t batch = std::pow(2, 12);
	auto lr = 1e-2;

	
	rbmState<cpx> phi(nvisible, nhidden, ham, lr, batch);
	auto rbm_info = phi.get_info();
	stout << VEQ(rbm_info) << EL;

	// monte carlo
	auto mcSteps = 400;
	size_t n_blocks = 200;
	size_t n_therm = int(0.1 * n_blocks);
	size_t block_size = std::pow(2, 3);
	auto n_flips = 1;
	auto energies = phi.mcSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
	// print dem ens
	std::ofstream file("energies"+ham->get_info() + ".dat");
	for (int i = 0; i < energies.size(); i++)
		printSeparatedP(file, '\t', 8, true, 5, i, energies[i].real());
	file.close();


	energies = v_1d<cpx>(energies.end() - block_size, energies.end());
	cpx standard_dev = stddev<cpx>(energies);
	stout << "\t\t->ENERGIES" << EL;
	cpx ground_rbm = 0;
	for (const auto& e : energies)
		ground_rbm += e;
	ground_rbm /= energies.size();
	//ground_rbm = energies[energies.size() - 1];

	stout << "\t\t\t->" << VEQ(ground_rbm) << "+-" << standard_dev << EL;
	if (lat->get_Ns() <= maxEd) {
		stout << "\t\t\t->" << VEQ(ground_ed) << EL;
		auto relative_error = std::abs((ground_ed - ground_rbm) / ground_ed) * 100;
		stout << "\t\t\t->" << VEQ(relative_error) << "%" << EL;
	}

	return 0;
}