#pragma once
#ifndef UI_H
#define UI_H

#define DONT_USE_ADAM
//#define DEBUG
#define PLOT


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

#include "../rbm.h"
#include "../models/ising.h"
#include "../models/heisenberg_dots.h"
#include "../models/heisenberg-kitaev.h"

#ifndef SQUARE_H
	#include "../lattices/square.h"
#endif
#ifndef HEXAGONAL_H
	#include "../lattices/hexagonal.h"
#endif

#ifdef PLOT
	// plotting
	#define WITHOUT_NUMPY
	#define WITH_OPENCV
	#include "matplotlib-cpp/matplotlibcpp.h"
	namespace plt = matplotlibcpp;
#endif

namespace rbm_ui {
	std::unordered_map <string, string> const default_params = {
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
}
// -------------------------------------------------------- Make a User interface class --------------------------------------------------------

class user_interface {
protected:
	int thread_number;																				 			// number of threads
	int boundary_conditions;																		 			// boundary conditions - 0 - PBC, 1 - OBC, 2 - ABC, ...
	string saving_dir;

	string getCmdOption(const v_1d<string>& vec, string option) const;				 							// get the option from cmd input
	template <typename T>
	void set_option(T& value, const v_1d<string>& argv, string choosen_option, bool geq_0 = true);				// set an option

	template <typename T>
	void set_default_msg(T& value, string option, string message, \
		const unordered_map <string, string>& map) const;														// setting value to default and sending a message
	// std::unique_ptr<LatticeModel> model;															 				// a unique pointer to the model used

public:
	virtual ~user_interface() = default;

	// make a simulation
	virtual void make_simulation() = 0;

	// exit
	virtual void exit_with_help() = 0;

	// ------------------------------------------- 				 REAL PARSING				 -------------------------------------------
	virtual void parseModel(int argc, const v_1d<string>& argv) = 0;											// the function to parse the command line
	// ------------------------------------------- 				 HELPING FUNCIONS			 -------------------------------------------
	virtual void set_default() = 0;																	 			// set default parameters
	// ------------------------------------------- 				 NON-VIRTUALS				 -------------------------------------------
	v_1d<string> parseInputFile(string filename);																// if the input is taken from file we need to make it look the same way as the command line does
};


/*
* @brief sets option from a given cmd options
* @param value
* @param argv
* @param choosen_option
* @param geq_0
*/
template<typename T>
void user_interface::set_option(T& value, const v_1d<string>& argv, string choosen_option, bool geq_0)
{
	if (string option = this->getCmdOption(argv, choosen_option); option != "")
		value = static_cast<T>(stod(option));													// set value to an option
	if (geq_0 && value <= 0)																	// if the variable shall be bigger equal 0
		this->set_default_msg(value, choosen_option.substr(1), \
			choosen_option + " cannot be negative\n", rbm_ui::default_params);
}

// string instance
template<>
inline void user_interface::set_option<string>(string& value, const v_1d<string>& argv, string choosen_option, bool geq_0) {
	if (string option = this->getCmdOption(argv, choosen_option); option != "")
		value = option;
}


/*
* @brief sets the message sent to user to default
* @param value
* @param option
* @param message
*/
template<typename T>
void user_interface::set_default_msg(T& value, string option, string message, const std::unordered_map <string, string>& map) const
{
	std::cout << message;																// print warning
	string value_str = "";																// we will set this to value
	auto it = map.find(option);
	if (it != map.end()) {
		value_str = it->second;															// if in table - we take the enum
	}
	value = static_cast<T>(stod(value_str));
}


// --------------------------------------------------------  				 HUBBARD USER INTERFACE 					  --------------------------------------------------------

namespace rbm_ui {
// --------------------------------------------------------  				 MAP OF DEFAULTS FOR HUBBARD 				  --------------------------------------------------------

	// ------------------------------------------- 				  CLASS				 -------------------------------------------
	template<typename _type, typename _hamtype>
	class ui : public user_interface {
	private:
		// lattice stuff
		impDef::lattice_types lattice_type; 																					// for non_numeric data
		int dim = 1;
		int _BC = 0;
		int Lx = 2;
		int Ly = 1;
		int Lz = 1;
		shared_ptr<Lattice> lat;

		// define basic model
		shared_ptr<SpinHamiltonian<_hamtype>> ham;
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

		// rbm stuff
		unique_ptr<rbmState<_type, _hamtype>> phi;
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

		// -------------------------------------------------------- HELPER FUNCTIONS

	public:
		// -------------------------------------------  				  CONSTRUCTORS  				 -------------------------------------------
		ui() = default;
		ui(int argc, char** argv);
		// -------------------------------------------  				  PARSER FOR HELP  				 -------------------------------------------
		void exit_with_help() override;
		// -------------------------------------------  				  REAL PARSER  				 -------------------------------------------
		void parseModel(int argc, const v_1d<string>& argv) override;									// the function to parse the command line
		// ------------------------------------------- HELPERS  				 -------------------------------------------
		void set_default() override;																		// set default parameters
		// -------------------------------------------  				  SIMULATION  			-------------------------------------------	 
		void make_simulation() override;
	};
}
// --------------------------------------------------------    				 RBM   				  --------------------------------------------------------

/*
* @param argc number of cmd parameters
* @param argv cmd parameters
*/
template<typename _type, typename _hamtype>
rbm_ui::ui<_type, _hamtype>::ui(int argc, char** argv)
{
	auto input = changeInpToVec(argc, argv);												// change standard input to vec of strings
	input = std::vector<string>(input.begin()++, input.end());								// skip the first element which is the name of file

	if (string option = this->getCmdOption(input, "-f"); option != "") {
		input = this->parseInputFile(option);												// parse input from file
	}
	this->parseModel(input.size(), input);													// parse input from CMD directly
}

// -------------------------------------------------------- PARSERS

/*
* @brief Prints help for a Hubbard interface
*/
template<typename _type, typename _hamtype>
void rbm_ui::ui<_type, _hamtype>::exit_with_help()
{
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
		"-th outer threads : number of outer threads (default 1)\n"
		"-ti inner threads : number of inner threads (default 1)\n"
		"-q : 0 or 1 -> quiet mode (no outputs) (default false)\n"
		"\n"
		"-h - help\n"
	);
	std::exit(1);
}

/*
* @brief  Setting Hubbard parameters to default
*/
template<typename _type, typename _hamtype>
void rbm_ui::ui<_type, _hamtype>::set_default()
{

	// lattice stuff
	this->lattice_type = impDef::lattice_types::square; 													// for non_numeric data
	this->dim = 1;
	this->_BC = 0;
	this->Lx = 10;
	this->Ly = 1;
	this->Lz = 1;

	// define basic model
	this->model_name = impDef::ham_types::ising;
	this->J = 1.0;
	this->J0 = 0;
	this->h = 0.1;
	this->w = 0.05;
	this->g = 0.2;
	this->g0 = 0.0;

	// heisenberg stuff
	this->delta = 0.00;

	// kitaev-heisenberg 
	this->Kx = 1;
	this->Ky = 1;
	this->Kz = 1;
	this->K0 = 0.0;

	// heisenberg with classical dots stuff
	this->positions = { 0 };
	this->phis = vec({ 0 });
	this->thetas = vec({ 1 });
	this->J_dot = { 0.0,0.0,-1.0 };
	this->J0_dot = 0.0;

	// others 
	this->thread_num = 16;

	// rbm
	this->batch = std::pow(2, 10);
	this->mcSteps = 1000;
	this->n_blocks = 500;
	this->block_size = 8;
	this->n_therm = size_t(0.1 * this->n_blocks);
	this->n_flips = 1;
	this->lr = 1e-2;
}

/*
* @brief model parser
* @param argc number of line arguments
* @param argv line arguments
*/
template<typename _type, typename _hamtype>
void rbm_ui::ui<_type, _hamtype>::parseModel(int argc, const v_1d<string>& argv)
{
	this->set_default();

	string choosen_option = "";

	//---------- SIMULATION PARAMETERS
	// monte carlo steps
	choosen_option = "-m";
	this->set_option(this->mcSteps, argv, choosen_option);

	// batch size
	choosen_option = "-b";
	this->set_option(this->batch, argv, choosen_option);

	// number of blocks
	choosen_option = "-nb";
	this->set_option(this->n_blocks, argv, choosen_option);
	
	// block size
	choosen_option = "-bs";
	this->set_option(this->block_size, argv, choosen_option);
	
	// number of hidden layers
	choosen_option = "-nh";
	this->set_option(this->nhidden, argv, choosen_option, false);

	// ----------- lattice

	// lattice type
	choosen_option = "-l";
	this->set_option(this->lattice_type, argv, choosen_option, false);

	// dimension
	choosen_option = "-d";
	this->set_option(this->dim, argv, choosen_option, false);

	// lx
	choosen_option = "-lx";
	this->set_option(this->Lx, argv, choosen_option);
	// ly
	choosen_option = "-ly";
	this->set_option(this->Ly, argv, choosen_option);
	// lz
	choosen_option = "-lz";
	this->set_option(this->Lz, argv, choosen_option);

	int Ns = Lx * Ly * Lz;

	// boundary conditions
	choosen_option = "-bc";
	this->set_option(this->_BC, argv, choosen_option, false);


	// ---------- model
	choosen_option = "-mod";
	this->set_option(this->model_name, argv, choosen_option, false);

	choosen_option = "-J";
	this->set_option(this->J, argv, choosen_option, false);

	// spin coupling disorder
	choosen_option = "-J0";
	this->set_option(this->J0, argv, choosen_option, false);

	// transverse field
	choosen_option = "-g";
	this->set_option(this->g, argv, choosen_option, false);

	// transverse field disorder
	choosen_option = "-g0";
	this->set_option(this->g0, argv, choosen_option, false);

	// perpendicular field
	choosen_option = "-h";
	this->set_option(this->h, argv, choosen_option, false);

	// perpendicular field disorder
	choosen_option = "-w";
	this->set_option(this->w, argv, choosen_option, false);

	// delta
	choosen_option = "-dlt";
	this->set_option(this->delta, argv, choosen_option, false);

	// --- kitaev
	choosen_option = "-kx";
	this->set_option(this->delta, argv, choosen_option, false);
	choosen_option = "-ky";
	this->set_option(this->delta, argv, choosen_option, false);
	choosen_option = "-kz";
	this->set_option(this->delta, argv, choosen_option, false);
	choosen_option = "-k0";
	this->set_option(this->delta, argv, choosen_option, false);

	// -------- other
	
	//---------- OTHERS
	// quiet
	choosen_option = "-q";
	this->set_option(this->quiet, argv, choosen_option, false);
	// outer thread number
	// inner thread number
	choosen_option = "-ti";
	this->set_option(this->thread_num, argv, choosen_option, false);

	// get help
	choosen_option = "-hlp";
	if (string option = this->getCmdOption(argv, choosen_option); option != "")
		exit_with_help();

	bool set_dir = false;
	choosen_option = "-dir";
	if (string option = this->getCmdOption(argv, choosen_option); option != "") {
		this->set_option(this->saving_dir, argv, choosen_option, false);
		set_dir = true;
	}
	if (!set_dir)
		this->saving_dir = fs::current_path().string() + kPS + "results" + kPS;

	fs::create_directories(this->saving_dir);

#ifndef DEBUG
	omp_set_num_threads(this->thread_num);
#endif // !DEBUG
}

/*
*
*/
template<typename _type, typename _hamtype>
void rbm_ui::ui<_type, _hamtype>::ui::make_simulation()
{
	//stout << "STARTING THE SIMULATION AND USING OUTER THREADS = " << outer_threads << ", INNER THREADS = " << inner_threads << EL;
	//this->set_default();



	// save the log file
//#pragma omp single
//	{
//		std::fstream fileLog(this->saving_dir + "HubbardLog.csv", std::ios::in | std::ios::app);
//		fileLog.seekg(0, std::ios::end);
//		if (fileLog.tellg() == 0) {
//			fileLog.clear();
//			fileLog.seekg(0, std::ios::beg);
//			printSeparated(fileLog, ',', 20, true ,"lattice_type","mcsteps","avsNum","corrTime", "M", "M0", "dtau", "Lx",\
//				"Ly", "Lz", "beta", "U", "mu", "occ", "sd(occ)", "av_sgn", "sd(sgn)", "Ekin",\
//				"sd(Ekin)", "m^2_z", "sd(m^2_z)", "m^2_x", "time taken");
//		}
//		fileLog.close();
//	}
	int maxEd = 12;

	// define the lattice
	switch (this->lattice_type)
	{
	case impDef::lattice_types::square:
		this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
		break;
	case impDef::lattice_types::hexagonal:
		this->lat = std::make_shared<HexagonalLattice>(Lx, Ly, Lz, dim, _BC);
		break;
	default:
		this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
		break;
	}

	auto lat_type = lat->get_type();
	auto Ns = lat->get_Ns();
	stout << VEQ(lat_type) << EL;
	// rbm stuff
	this->nhidden = Ns;
	this->nvisible = 2 * this->nhidden;

	// define the hamiltonian
	switch (this->model_name)
	{
	case impDef::ham_types::ising:
		this->ham = std::make_shared<IsingModel<_hamtype>>(J, J0, g, g0, h, w, lat);
		break;
	case impDef::ham_types::heisenberg:
		this->ham = std::make_shared<Heisenberg<_hamtype>>(J, J0, g, g0, h, w, delta, lat);
		break;
	case impDef::ham_types::heisenberg_dots:
		this->ham = std::make_shared<Heisenberg_dots<_hamtype>>(J, J0, g, g0, h, w, delta, lat, positions, J_dot, J0_dot);
		ham->set_angles(phis, thetas);
		break;
	case impDef::ham_types::kitaev_heisenberg:
		this->ham = std::make_shared<Heisenberg_kitaev<_hamtype>>(J, J0, g, g0, h, w, delta, make_tuple(Kx, Ky, Kz), K0, lat);
		break;
	default:
		this->ham = std::make_shared<IsingModel<_hamtype>>(J, J0, g, g0, h, w, lat);
		break;
	}

	// test ED
	double ground_ed = 0;
	if (Ns <= maxEd) {
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
			//for (int j = 0; j < lat->get_Ns(); j++) {
			auto j = 0;
			sz += abs(abs(value) * value) * (checkBit(i, lat->get_Ns() - 1 - j) ? 1.0 : -1.0);
			//}
		}
		stout << "ED sz: " << VEQ(sz) << EL;
	}

	this->phi = std::make_unique<rbmState<_type, _hamtype>>(nvisible, nhidden, ham, lr, batch, thread_num);
	auto rbm_info = phi->get_info();
	stout << VEQ(rbm_info) << EL;

	// monte carlo
	auto energies = phi->mcSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);

#ifdef PLOT
	plt::plot(arma::conv_to< v_1d<double> >::from(arma::real(energies)));
#endif

	// print dem ens
	auto filename = "energies" + ham->get_info() + ".dat";
	std::ofstream file(filename);
	for (int i = 0; i < energies.size(); i++)
		printSeparatedP(file, '\t', 8, true, 5, i, energies(i).real());
	file.close();


	energies = energies.tail(block_size);
	_type standard_dev = arma::stddev(energies);
	stout << "\t\t->ENERGIES" << EL;
	_type ground_rbm = arma::mean(energies);

	// ------------------- check ground state
	std::map<u64, _type> states = phi->avSampling(200, 50, 16, n_flips);
	Col<_type> states_col = SpinHamiltonian<_type>::map_to_state(states, ham->get_hilbert_size());
	//SpinHamiltonian<_type>::print_state_pretty(states_col, lat->get_Ns(), 0.1);

	// compare sigma_x
	auto ed_av_s_x = this->ham->av_sigma_x(0, 0);
	auto rbm_av_s_x = this->ham->av_sigma_x(states_col, states_col);

	stout << VEQP(ed_av_s_x, 4) << "," << VEQP(rbm_av_s_x, 4) << EL;




	stout << "\t\t\t->" << VEQ(ground_rbm) << "+-" << standard_dev << EL;
	if (lat->get_Ns() <= maxEd) {
		stout << "\t\t\t->" << VEQ(ground_ed) << EL;
		auto relative_error = abs(std::real(ground_ed - ground_rbm)) / abs(ground_ed) * 100;
		stout << "\t\t\t->" << VEQ(relative_error) << "%" << EL;

#ifdef PLOT
		plt::axhline(ground_ed);
		plt::annotate(VEQ(ground_ed) + ",\n" + VEQ(ground_rbm) + ",\n" + VEQ(relative_error) + "%", mcSteps / 3, 0);
	}
	plt::title(ham->get_info() + "\nrbm:" + rbm_info);
	plt::xlabel("#mcstep");
	plt::ylabel("$<E_{est}>$");
	plt::save("energies" + ham->get_info() + ".png");
	plt::show(true);
#else
	}
#endif
	stout << "FINISHED EVERY THREAD" << EL;
}
// -------------------------------------------------------- HELPERS
#endif // !UI_H
