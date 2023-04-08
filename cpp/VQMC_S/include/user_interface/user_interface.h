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

// ######################### RBM ############################
#ifndef RBM_H											 // #
#include "../rbm.h"										 // #
#endif													 // #
// ##########################################################


// ######################### MODELS #########################
#ifndef ISING_H											 // #
#include "../models/ising.h"							 // #
#endif // !ISING_H										 // #
// MODELS												 // #
#ifndef XYZ_H											 // #
#include "../models/xyz.h"								 // #
#endif // !XYZ_H										 // #
// ##########################################################

// ###################### LATTICES ##########################
#ifndef SQUARE_H										 // #
#include "../../source/src/Lattices/square.h"			 // #
#endif													 // #
#ifndef HEXAGONAL_H										 // #
#include "../../source/src/Lattices/hexagonal.h"		 // #
#endif													 // #
// ##########################################################


// maximal ed size to compare
constexpr int maxed = 20;





// -------------------------------------------------------- Make a User interface class --------------------------------------------------------

//template<typename _hamtype>
//double calculate_ed(double& ground_ed, double ground_rbm, std::shared_ptr<SpinHamiltonian<_hamtype>> hamiltonian) {
//	// compare ED
//	auto Ns = hamiltonian->lattice->get_Ns();
//	auto maxNs = 16;
//	if (Ns <= maxNs) {
//		stout << "\t\t\t\t->calculating ed" << EL;
//		hamiltonian->hamiltonian();
//		if (Ns <= maxNs)
//			hamiltonian->diag_h(false);
//		else
//			hamiltonian->diag_h(false, 3, 0, 1000);
//
//		ground_ed = std::real(hamiltonian->get_eigenEnergy(0));
//		auto relative_error = abs(std::real(ground_ed - ground_rbm)) / abs(ground_ed) * 100.;
//		stout << "\t\t\t\t->" << VEQP(ground_ed, 7) << "\t" << VEQP(ground_rbm, 7) << "\t" << VEQP(relative_error, 5) << "%" << EL;
//		return relative_error;
//	}
//	else
//		stout << "\t\t\t\t->skipping ed" << EL;
//	return 0.0;
//}

// --------------------------------------------------------  				 HUBBARD USER INTERFACE 					  --------------------------------------------------------

// lattice stuff
struct LatP {
	LatticeTypes typ = SQ; 																					// for non_numeric data
	int bc = 0;
	int dim = 1;
	int Lx = 2;
	int Ly = 1;
	int Lz = 1;
	std::shared_ptr<Lattice> lat;
};

struct SymP {
	bool S = false;
	int kSec = 0;									// translation symmetry sector
	bool pxSec = true;								// x parity sector
	bool pySec = true;								// y parity sector
	bool pzSec = true;								// z parity sector
	bool xSec = true;								// spin flip sector
	int U1Sec = 0;									// particle number conservation sector
};

// !TODO 
// Neural network quantum states params
struct NqsP {
	//unique_ptr<rbmState<_type, _hamtype>> phi;
	u64 nHidden = 1;
	u64 nVisible = 1;
	uint nLayers = 2;
	v_1d<u64> layersDim;

	// single time flips
	uint nFlips = 1;
	double lr = 1e-2;
	uint nBlocks = 500;
	uint blockSize = 8;
	uint mcSteps = 1000;
	uint nTherm = uint(0.1 * nBlocks);
	u64 batch	= (u64)std::pow(2, 10);
};


// --------------------------------------------------------  				 MAP OF DEFAULTS FOR HUBBARD 				  --------------------------------------------------------

// ------------------------------------------- 				  CLASS				 -------------------------------------------

class UI : public UserInterface {
private:
	// set the possible options
	cmdMap default_params = {
	//	std::make_tuple("m", &this->nqsP.mcSteps, "300", std::function(higherThanZero)),			// mcsteps	
	//	std::make_tuple("b", &this->nqsP.batch, "100", std::function(higherThanZero)),				// batch
	//	std::make_tuple("nb", &this->nqsP.nBlocks, "500", std::function(higherThanZero)),			// number of blocks	
	//	std::make_tuple("bs", &this->nqsP.blockSize, "8", std::function(higherThanZero)),			// block size
	//	std::make_tuple("nh", &this->nqsP.nHidden, "2", std::function(higherThanZero))				// hidden parameters
		
		// lattice parameters
		{"d",	std::make_tuple("1"		, higherThanZero)	},				// dimension
		{"lx",	std::make_tuple("4"		, higherThanZero)	},
		{"ly",	std::make_tuple("1"		, higherThanZero)	},
		{"lz",	std::make_tuple("1"		, higherThanZero)	},
		{"bc",	std::make_tuple("0"		, higherThanZero)	},				// boundary condition
		{"l",	std::make_tuple("0"		, higherThanZero)	},				// lattice type (default square)
		{"f",	std::make_tuple(""		, defaultReturn)	},				// file to read from directory
		// model parameters
		{"mod",	std::make_tuple("0"		, higherThanZero)	},				// choose model
		{"J",	std::make_tuple("1.0"	, defaultReturn)	},				// spin coupling
		{"J0",	std::make_tuple("0.0"	, defaultReturn)	},				// spin coupling randomness maximum (-J0 to J0)
		{"h",	std::make_tuple("0.1"	, defaultReturn)	},				// perpendicular magnetic field constant
		{"w",	std::make_tuple("0.01"	, defaultReturn)	},				// disorder strength
		{"g",	std::make_tuple("1.0"	, defaultReturn)	},				// transverse magnetic field constant
		{"g0",  std::make_tuple("0.0"	, defaultReturn)	},				// transverse field randomness maximum (-g0 to g0)
		// heisenberg	
		{"dlt", std::make_tuple("1.0"	, defaultReturn)	},				// delta
		// xyz
		{"dlt2",std::make_tuple("0.9"	, defaultReturn)	},				// delta2
		{"J2",  std::make_tuple("1.0"	, defaultReturn)	},				// J2
		{"eta", std::make_tuple("0.0"	, defaultReturn)	},				// eta
		{"eta2",std::make_tuple("0.0"	, defaultReturn)	},				// eta2

		// kitaev
		{"kx",  std::make_tuple("0.0"	, defaultReturn)	},				// kitaev x interaction
		{"ky",  std::make_tuple("0.0"	, defaultReturn)	},				// kitaev y interaction
		{"kz",  std::make_tuple("0.0"	, defaultReturn)	},				// kitaev z interaction
		{"k0",  std::make_tuple("0.0"	, defaultReturn)	},				// kitaev interaction disorder
		// other
		{"fun", std::make_tuple("-1"	, defaultReturn)	},				// choice of the function to be calculated
		{"th",  std::make_tuple("1"		, defaultReturn)	},				// number of threads
		{"q",   std::make_tuple("0"		, defaultReturn)	}				// quiet?
	};

	// lattice
	LatP latP;

	// symmetry params
	SymP symP;

	// NQS params
	NqsP nqsP;

	// define basic model
	//shared_ptr<SpinHamiltonian<_hamtype>> ham;
	std::shared_ptr<Hamiltonian<double>> hamDouble;
	std::shared_ptr<Hamiltonian<cpx>> hamComplex;

	//impDef::ham_types model_name;												// the name of the model for parser
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
	arma::vec phis = arma::vec({});
	arma::vec thetas = arma::vec({});
	arma::vec J_dot = { 1.0,0.0,-1.0 };
	double J0_dot = 0.0;
	double J_dot_dot = 1.0;														// dot - dot  classical interaction

	// xyz parameters
	double Jb = 1.0;																						// next nearest neighbors J	
	double Delta_b = 0.9;																					// sigma_z*sigma_z next nearest neighbors
	double eta_a = 0.5;
	double eta_b = 0.5;

	// others 
	bool quiet;																	// bool flags	
	uint thread_number = 1;														// thread parameters

	// averages from operators
	//avOperators av_op;

	// -------------------------------------------   		 HELPER FUNCTIONS  		-------------------------------------------
	//void compare_ed(double ground_rbm);
	//void save_operators(clk::time_point start, std::string name, double energy, double energy_error);
protected:

//private:
	// INNER METHODS

	// ####################### SYMMETRIES #######################
		
	//void symmetries_cpx(clk::time_point start);
	//void symmetries_double(clk::time_point start);

	// ####################### CLASSICAL ########################
		
	//void make_mc_classical_angles(double Jdot = 0.0);

	// ####################### KITAEV #######################

	//void make_mc_kitaev(t_3d<double> K);


public:
	// -----------------------------------------------        CONSTRUCTORS  		-------------------------------------------
	UI() = default;
	UI(int argc, char** argv) {
		strVec input = fromPtr(argc, argv, 1);											// change standard input to vec of strings
		//input = std::vector<string>(input.begin()++, input.end());					// skip the first element which is the name of file
		if (std::string option = this->getCmdOption(input, "-f"); option != "")
			input = this->parseInputFile(option);										// parse input from file
		this->parseModel((int)input.size(), input);
	}

	// -----------------------------------------------   	 PARSER FOR HELP  		-------------------------------------------
	void exitWithHelp() override {
		printf(
			"Usage: name [options] outputDir \n"
			"options:\n"
			" The input can be both introduced with [options] described below or with giving the input directory \n"
			" (which also is the flag in the options) \n"
			" options:\n"
			"-f input file for all of the options : (default none) \n"
			"-m monte carlo steps : bigger than 0 (default 300) \n"
			"-d dimension : set dimension (default 2) \n"
			"	1 -- 1D \n"
			"	2 -- 2D \n"
			"	3 -- 3D -> NOT IMPLEMENTED YET \n"
			"-l lattice type : (default square) -> CHANGE NOT IMPLEMENTED YET \n"
			"   square \n"
			// SIMULATIONS STEPS
			"\n"
			"-q : 0 or 1 -> quiet mode (no outputs) (default false)\n"
			"\n"
			"-fun : function to be used in the calculations. There are predefined functions in the model that allow that:\n"
			"   The options divide each other on different categories according to the first number _ \n"
			"   -1 -- default option -> shows help \n"
			"    0 -- this option tests the calculations of various types of Hamiltonians and compares the results\n (w and wo symmetries included)\n"
			"    1 -- this option utilizes the classical degrees of freedom with RBM\n"
			"		11 -- check the difference between AF and FM classical spins configuration\n"
			"		12 -- check the minimum of energy when classical spins are varied with angle and with interaction\n"
			"    2 -- this option utilizes the Hamiltonian with symmetries calculation\n"
			"		21 -- input the symmetries and the model information and a given block will be calculated\n"
			"		22 -- input the symmetries and the model information and a given block will be calculated with a sweep in Jb (to be changed)\n"
			"\n"
			"-h - help\n"
		);
		std::exit(1);
	}
		
	// -----------------------------------------------    	   REAL PARSER          -------------------------------------------
	// the function to parse the command line
	void funChoice() final;
	void parseModel(int argc, cmdArg& argv) final;

	// ----------------------------------------------- 			HELPERS  			-------------------------------------------
	void setDefault() override;

	// -----------------------------------------------  	   SIMULATION  		    -------------------------------------------	 
	//void define_models();

	// #######################		     RBMs               #######################
		
	// ########## CLASSICAL
	//void make_mc_classical();
	//void make_mc_angles_sweep();

	// ########## KITAEV
	//void make_mc_kitaev_sweep();

	// ########## TEST 
	//void make_simulation() override;

	// #######################        SYMMETRIES            #######################
	//void make_simulation_symmetries();
	//void make_simulation_symmetries_sweep();
	//void make_symmetries_test(int l = -1);

};
