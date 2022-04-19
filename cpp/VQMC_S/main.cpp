
#include "include/rbm.h"
#include "include/models/ising.h"
#include "include/lattices/square.h"

int main() {

	// define lattice
	int Lx = 8;
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
	auto info = ham->get_info();
	stout << VEQ(info) << EL;
	//
	//// define rbm state
	u64 nhidden = Lx * Ly * Lz;
	u64 nvisible = 2 * nhidden;
	auto lr = 1e-2;
	auto mom = 1e-5;
	auto xav = 1;
	//
	rbmState<cpx> phi(nvisible, nhidden, ham, lr, mom, xav);
	auto mcSteps = 200;
	auto n_blocks = 200;
	auto n_therm = int(0.1 * n_blocks);
	auto block_size = std::pow(2, 2);
	auto n_flips = 1;
	auto energies = phi.mcSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
	stout << energies[energies.size() - 1] << EL;

	// do ed for comparison
	ham->hamiltonian();
	ham->diag_h(false);
	double ground_en = std::real(ham->get_eigenEnergy(0));
	stout << ground_en << EL;
	SpinHamiltonian<cpx>::print_state_pretty(ham->get_eigenState(0), lat->get_Ns());
	return 0;
}