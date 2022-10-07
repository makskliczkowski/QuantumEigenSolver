#pragma once
#include "../../source/src/UserInterface/ui.h"

#ifdef DEBUG
//#define DEBUG_BINARY
#ifdef DEBUG_RBM
//#define DEBUG_RBM_SAMP
//#define DEBUG_RBM_LCEN
//#define DEBUG_RBM_GRAD
//#define DEBUG_RBM_DRVT
#endif
#else
//#define OMP_NUM_THREADS 16;
#include <thread>
#include <mutex>
#endif

// symmetric models 
#include "../models/symmetries/ising_sym.h"

// RBM
#ifndef RBM_H
#include "../rbm.h"
#endif

#ifndef SQUARE_H
#include "../../source/src/Lattices/square.h"
#endif
#ifndef HEXAGONAL_H
#include "../../source/src/Lattices/hexagonal.h"
#endif


#ifdef PLOT
#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

// plotting
#define WITHOUT_NUMPY
#define WITH_OPENCV
#include "matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;

template<typename _type>
void plot_v_1d(v_1d<_type> v, string xlabel = "", string ylabel = "", string name = "") {
	plt::plot(v);
	plt::xlabel(xlabel);
	plt::ylabel(ylabel);
	plt::title(name);
};
template<typename _type>

void plot_v_1d(Col<_type> v, string xlabel = "", string ylabel = "", string name = "") {
	plt::plot(arma::conv_to<v_1d<double> >::from(v));
	plt::xlabel(xlabel);
	plt::ylabel(ylabel);
	plt::title(name);
};

#define PLOT_V1D(v, x, y, n) plot_v_1d(v, x, y, n)
template<typename _type>
void scatter_v_1d(v_1d<_type> v, string xlabel = "", string ylabel = "", string name = "") {
	std::vector<int> ivec(v.size());
	std::iota(ivec.begin(), ivec.end(), 0); // ivec will become: [0..99]

	plt::scatter_colored(ivec, v);
	plt::xlabel(xlabel);
	plt::ylabel(ylabel);
	plt::title(name);
};
#define SCATTER_V1D(v, x, y, n) plot_v_1d(v, x, y, n)
void inline save_fig(string name, bool show = false) {
	plt::save(name);
	if (show) plt::show();
	plt::close();
}
#define SAVEFIG(name, show) save_fig(name, show)
#else 
#define PLOT_V1D(v, x, y, n)
#define SCATTER_V1D(v, x, y, n)
#define SAVEFIG(name, show)
#endif

// maximal ed size to compare
constexpr int maxed = 20;





// -------------------------------------------------------- Make a User interface class --------------------------------------------------------

template<typename _hamtype>
void calculate_ed(double& ground_ed, double ground_rbm, std::shared_ptr<SpinHamiltonian<_hamtype>> hamiltonian) {
	// compare ED
	auto Ns = hamiltonian->lattice->get_Ns();
	auto maxNs = 16;
	if (maxNs <= 16) {
		stout << "\t\t\t\t->calculating ed" << EL;
		hamiltonian->hamiltonian();
		if (maxNs <= 16)
			hamiltonian->diag_h(false);
		else
			hamiltonian->diag_h(false, 3, 0, 1000);

		ground_ed = std::real(hamiltonian->get_eigenEnergy(0));
		stout << "\t\t\t\t->" << VEQ(ground_ed) << "\t" << VEQ(ground_rbm) << EL;
	}
	else {
		stout << "\t\t\t\t->skipping ed" << EL;
	}
}

// --------------------------------------------------------  				 HUBBARD USER INTERFACE 					  --------------------------------------------------------

namespace rbm_ui {
	// --------------------------------------------------------  				 MAP OF DEFAULTS FOR HUBBARD 				  --------------------------------------------------------

		// ------------------------------------------- 				  CLASS				 -------------------------------------------
	template<typename _type, typename _hamtype>
	class ui : public user_interface {
	private:
		std::unordered_map <string, string> default_params = {
			{"m","300"},								// mcsteps	
			{"b","100"},								// batch
			{"nb","500"},								// number of blocks	
			{"bs","8"},									// block size
			{"nh","2"},									// hidden parameters
			// lattice parameters
			{"d","1"},									// dimension
			{"lx","4"},
			{"ly","1"},
			{"lz","1"},
			{"bc","0"},									// boundary condition
			{"l","0"},									// lattice type (default square)
			{"f",""},									// file to read from directory
			// model parameters
			{"mod","0"},								// choose model
			{"J","1.0"},								// spin coupling
			{"J0","0.0"},								// spin coupling randomness maximum (-J0 to J0)
			{"h","0.1"},								// perpendicular magnetic field constant
			{"w","0.01"},								// disorder strength
			{"g","1.0"},								// transverse magnetic field constant
			{"g0","0.0"},								// transverse field randomness maximum (-g0 to g0)
			// heisenberg
			{"dlt", "0"},								// delta
			// kitaev
			{"kx", "0.0"},								// kitaev x interaction
			{"ky", "0.0"},								// kitaev y interaction
			{"kz", "0.0"},								// kitaev z interaction
			{"k0", "0.0"},								// kitaev interaction disorder
			// other
			{"th","1"},									// number of threads
			{"q","0"},									// quiet?
		};

		// lattice stuff
		impDef::lattice_types lattice_type; 																					// for non_numeric data
		int dim = 1;
		int _BC = 0;
		int Lx = 2;
		int Ly = 1;
		int Lz = 1;
		shared_ptr<Lattice> lat;

		// symmetries stuff
		bool sym = false;
		int k_sym = 0;
		bool p_sym = true;
		bool x_sym = true;
		long double spectrum_size = 0.2;

		// define basic model
		shared_ptr<SpinHamiltonian<_hamtype>> ham;
		shared_ptr<SpinHamiltonian<double>> ham_d;
		shared_ptr<SpinHamiltonian<cpx>> ham_cpx;

		impDef::ham_types model_name;												// the name of the model for parser
		double J = 1.0;																// spin exchange
		double J0 = 0;																// spin exchange coefficient disorder
		double h = 0.0;																// perpendicular magnetic field
		double w = 0.0;																// the distorder strength to set dh in (-disorder_strength, disorder_strength)
		double g = 0.0;																// transverse magnetic field
		double g0 = 0.0;															// disorder in the system - deviation from a constant g0 value

		// heisenberg stuff
		double delta = 0.00;														// 

		// kitaev-heisenberg 
		double Kx = 1;
		double Ky = 1;
		double Kz = 1;
		double K0 = 0.0;

		// heisenberg with classical dots stuff
		v_1d<int> positions = {};
		vec phis = vec({});
		vec thetas = vec({});
		vec J_dot = { 1.0,0.0,-1.0 };
		double J0_dot = 0.0;
		double J_dot_dot = 1.0;														// dot - dot  classical interaction

		// rbm stuff
		unique_ptr<rbmState<_type, _hamtype>> phi;
		int layer_mult = 2;
		u64 nhidden;
		u64 nvisible;
		size_t batch = std::pow(2, 10);
		size_t mcSteps = 1000;
		size_t n_blocks = 500;
		size_t block_size = 8;
		size_t n_therm = size_t(0.1 * n_blocks);
		size_t n_flips = 1;
		double lr = 1e-2;

		// others 
		size_t thread_num = 16;														// thread parameters
		bool quiet;																	// bool flags	


		// averages from operators
		avOperators av_op;

		// -------------------------------------------   					HELPER FUNCTIONS  					-------------------------------------------
		void compare_ed(double ground_rbm);
		void save_operators(clk::time_point start, std::string name, double energy, double energy_error);
	public:
		// -------------------------------------------  					CONSTRUCTORS  					-------------------------------------------
		ui() = default;
		ui(int argc, char** argv);
		// -------------------------------------------  					PARSER FOR HELP  					-------------------------------------------
		void exit_with_help() override;
		// -------------------------------------------  				  REAL PARSER  				 -------------------------------------------
		// the function to parse the command line
		void parseModel(int argc, const v_1d<string>& argv) final;
		// -------------------------------------------   					HELPERS  							-------------------------------------------
		void set_default() override;																		// set default parameters
		// -------------------------------------------  				  SIMULATION  			-------------------------------------------	 
		void define_models();
		void make_mc_classical(bool ferromagnetic = true);
		void symmetries_cpx(clk::time_point start);
		void symmetries_double(clk::time_point start);
		void make_simulation_symmetries();
		void make_symmetries_test(int l = -1);
		void make_simulation() override;
	};
}
// --------------------------------------------------------    				RBM   						--------------------------------------------------------
