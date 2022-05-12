#pragma once
#ifndef UI_H
#define UI_H



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
	#include <thread>
	#include <mutex>
#endif


#ifndef RBM_H
	#include "../rbm.h"
#endif

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
constexpr int maxed = 10;


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
	// the function to parse the command line
	virtual void parseModel(int argc, const v_1d<string>& argv) = 0;											
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
		void make_simulation() override;
	};
}
// --------------------------------------------------------    				RBM   						--------------------------------------------------------

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
	this->layer_mult = 2;
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
inline void rbm_ui::ui<_type, _hamtype>::parseModel(int argc, const v_1d<string>& argv)
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
	this->n_therm = size_t(0.1 * n_blocks);
	
	// block size
	choosen_option = "-bs";
	this->set_option(this->block_size, argv, choosen_option);
	
	// number of hidden layers
	choosen_option = "-nh";
	this->set_option(this->nhidden, argv, choosen_option, false);

	// number of hidden layers multiplier
	choosen_option = "-lm";
	this->set_option(this->layer_mult, argv, choosen_option, false);

	// learning rate
	choosen_option = "-lr";
	this->set_option(this->lr, argv, choosen_option, false);
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

	// model type
	choosen_option = "-mod";
	this->set_option(this->model_name, argv, choosen_option, false);

	// spin interaction
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

	// --- heisenberg ---

	// delta
	choosen_option = "-dlt";
	this->set_option(this->delta, argv, choosen_option, false);

	// --- kitaev ---
	choosen_option = "-kx";
	this->set_option(this->delta, argv, choosen_option, false);
	choosen_option = "-ky";
	this->set_option(this->delta, argv, choosen_option, false);
	choosen_option = "-kz";
	this->set_option(this->delta, argv, choosen_option, false);
	choosen_option = "-k0";
	this->set_option(this->delta, argv, choosen_option, false);

	//---------- OTHERS

	// quiet
	choosen_option = "-q";
	this->set_option(this->quiet, argv, choosen_option, false);

	// thread number
	choosen_option = "-th";
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

	// create the directories
	fs::create_directories(this->saving_dir);

//#ifndef DEBUG
//	omp_set_num_threads(this->thread_num);
//#endif // !DEBUG
}

/*
*
*/
template<typename _type, typename _hamtype>
void rbm_ui::ui<_type, _hamtype>::ui::make_simulation()
{
	auto start = std::chrono::high_resolution_clock::now();
	stouts("STARTING THE SIMULATION FOR GROUNDSTATE SEEK AND USING: " + VEQ(thread_num), start);
	printSeparated(stout, ',', 5, true, VEQ(mcSteps), VEQ(n_blocks), VEQ(n_therm), VEQ(block_size));
	// monte carlo
	auto energies = this->phi->mcSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);

	// print energies
	string dir = this->saving_dir + kPS + ham->get_info() + kPS + phi->get_info() + kPS;
	fs::create_directories(dir);

	auto fileRbmEn_name = dir + "energies";
	std::ofstream fileRbmEn;
	openFile(fileRbmEn, fileRbmEn_name + ".dat", ios::out);
	for (auto i = 0; i < energies.size(); i++)
		printSeparatedP(fileRbmEn, '\t', 8, true, 5, i, energies(i).real());
	fileRbmEn.close();

	// calculate the statistics of a simulation
	auto energies_tail = energies.tail(block_size);
	_type standard_dev = arma::stddev(energies_tail);
	_type ground_rbm = arma::mean(energies_tail);

	// if the size is small enough
	this->compare_ed(std::real(ground_rbm));
	PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", ham->get_info() + "\nrbm:" + this->phi->get_info());
	SAVEFIG(fileRbmEn_name + ".png", true);
	// ------------------- check ground state
	std::map<u64, _type> states = phi->avSampling(mcSteps, n_therm, n_blocks, block_size, n_flips);
	if (this->lat->get_Ns() <= maxed) {
		// convert to our basis
		Col<_type> states_col = SpinHamiltonian<_type>::map_to_state(states, ham->get_hilbert_size());
		this->av_op.reset();
		Operators<_type> op(this->lat);
		op.calculate_operators(states_col, this->av_op);
		this->save_operators(start, this->phi->get_info(), real(ground_rbm), real(standard_dev));
	}
	else {
		this->av_op = this->phi->get_op_av();
		this->save_operators(start, this->phi->get_info(), real(ground_rbm), real(standard_dev));
	}
	stouts("FINISHED EVERY THREAD", start);
	stout << "\t\t\t->" << VEQ(ground_rbm) << "+-" << standard_dev << EL;
}
// -------------------------------------------------------- HELPERS

/*
*/
template<typename _type, typename _hamtype>
inline void rbm_ui::ui<_type, _hamtype>::define_models()
{
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
	auto lat_info = lat->get_info();
	auto Ns = lat->get_Ns();
	stout << "\t\t-> " << VEQ(lat_info) << EL;
	//stout << "\t\t-> " << VEQ(lat_type) << EL;
	// create operator averages
	this->av_op = avOperators(Lx, Ly, Lz, this->lat->get_Ns(), lat_type);				

	// define the hamiltonian
	//stout << static_cast<int>(this->model_name) << EL;
	switch (static_cast<int>(this->model_name))
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
	auto model_info = this->ham->get_info();
	stout << "\t\t-> " << VEQ(model_info) << EL;


	// rbm stuff
	this->nhidden = Ns;
	this->nvisible = this->layer_mult * this->nhidden;
	this->phi = std::make_unique<rbmState<_type, _hamtype>>(nvisible, nhidden, ham, lr, batch, thread_num);
	auto rbm_info = phi->get_info();
	stout << "\t\t-> " << VEQ(rbm_info) << EL;

}


/*
* if it is possible to do so we can test the exact diagonalization states for comparison
*/
template<typename _type, typename _hamtype>
inline void rbm_ui::ui<_type, _hamtype>::compare_ed(double ground_rbm)
{
	// test ED
	auto Ns = this->lat->get_Ns();
	double ground_ed = 0;

	if (Ns <= maxed) {
		this->av_op.reset();
		auto diag_time = std::chrono::high_resolution_clock::now();
		stout << "\n\n-> starting ED for:\n\t-> " + ham->get_info() << EL;
		// define the operators class
		
		this->ham->hamiltonian();
		this->ham->diag_h(false);

		Operators<_hamtype> op(this->lat); 
		ground_ed = std::real(ham->get_eigenEnergy(0));
		auto excited_ed = std::real(ham->get_eigenEnergy(1));
		Col<_hamtype> eigvec = ham->get_eigenState(0);
		op.calculate_operators(eigvec, this->av_op);
		this->save_operators(diag_time, "", ground_ed, 0);

		u64 state = 0;
		auto sz = this->av_op.s_z;
		auto sx = this->av_op.s_x;
		auto relative_error = abs(std::real(ground_ed - ground_rbm)) / abs(ground_ed) * 100.;
		
		stouts("\t\t-> finished ED", diag_time);
		stout << "\t\t\t->" << VEQ(ground_ed) << EL;
		stout << "\t\t\t->" << VEQ(ground_rbm) << EL;
		stout << "\t\t\t->" << VEQP(relative_error, 4) << "%" << EL;
		stout << "------------------------------------------------------------------------" << EL;
		stout << "GROUND STATE ED ENERGY: " << VEQP(ground_ed, 4) << EL;
		stout << "1ST EXCITED STATE ED ENERGY: " << VEQP(excited_ed, 4) << EL;
		stout << "GROUND STATE ED SIGMA_X EXTENSIVE: " << VEQP(sx, 4) << EL;
		stout << "GROUND STATE ED SIGMA_Z EXTENSIVE: " << VEQP(sz, 4) << EL;
		stout << "\n------------------------------------------------------------------------\n|Psi>=:" << EL;
		stout << "\t->Ground(" + STRP(ground_ed, 3) + "):" << EL;
		SpinHamiltonian<_hamtype>::print_state_pretty(ham->get_eigenState(0), Ns, 0.08);
		stout << "\t->Excited(" + STRP(excited_ed, 3) + "):" << EL;
		SpinHamiltonian<_hamtype>::print_state_pretty(ham->get_eigenState(1), Ns, 0.08);
		stout << "------------------------------------------------------------------------" << EL;
#ifdef PLOT
		plt::axhline(ground_ed);
		plt::axhline(excited_ed);
		plt::axhline(ham->get_eigenEnergy(2));
		plt::axhline(ham->get_eigenEnergy(3));
		plt::axhline(ham->get_eigenEnergy(4));
		plt::annotate(VEQ(ground_ed) + ",\n" + VEQ(excited_ed) + ",\n" + VEQ(ground_rbm) + ",\n" + VEQ(relative_error) + "%", mcSteps / 3, (ground_rbm) / 2);
#endif
	}
}

// -------------------------------- OPERATORS -----------------------------------------


template<typename _type, typename _hamtype>
inline void rbm_ui::ui<_type, _hamtype>::save_operators(clk::time_point start, std::string name, double energy, double energy_error)
{
	std::ofstream fileSave;
	std::fstream log;
	
	string dir = this->saving_dir + kPS + ham->get_info() + kPS; 
	if (name != "") dir = dir + name + kPS;
	fs::create_directories(dir);

	string filename = "";
	auto Ns = this->lat->get_Ns();
	// --------------------- compare sigma_z ---------------------

	// S_z at each site
	filename = dir + "_sz_site";
	openFile(fileSave, filename + ".dat", ios::out);
	print_vector_1d(fileSave, this->av_op.s_z_i);
	fileSave.close();
	PLOT_V1D(this->av_op.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + this->ham->get_info() + "\n" + name);
	SAVEFIG(filename + ".png", false);

	// S_z correlations
	filename = dir + "_sz_corr";
	openFile(fileSave, filename + ".dat", ios::out);
	print_vector_3d(fileSave, this->av_op.s_z_cor);
	fileSave.close();

	// --------------------- compare sigma_x ---------------------
	// S_z at each site
	filename = dir + "_sx_site";
	openFile(fileSave, filename + ".dat", ios::out);
	print_vector_1d(fileSave, this->av_op.s_x_i);
	fileSave.close();
	PLOT_V1D(this->av_op.s_x_i, "lat_site", "$S^x_i$", "$S^x_i$\n" + ham->get_info() + "\n" + name);
	SAVEFIG(filename + ".png", false);

	// S_z correlations
	filename = dir + "_sx_corr_";
	openFile(fileSave, filename + ".dat", ios::out);
	print_vector_3d(fileSave, this->av_op.s_x_cor);
	fileSave.close();

	// --------------------- entropy ----------------------
	filename = dir + "_ent_entro";
	openFile(fileSave, filename + ".dat", ios::out);
	print_vector_1d(fileSave, this->av_op.ent_entro);
	fileSave.close();
	PLOT_V1D(this->av_op.ent_entro, "bond_cut", "$S_0(L)$", "Entanglement entropy\n" + ham->get_info() + "\n" + name);
	SAVEFIG(filename + ".png", false);

	// --------------------- save log ---------------------	// save the log file and append columns if it is empty
	string logname = dir + "log.dat";
#pragma omp single
	{
		openFile(log, logname, ios::app);
		log.seekg(0, std::ios::end);
		if (log.tellg() == 0) {
			log.clear();
			log.seekg(0, std::ios::beg);
			printSeparated(log, '\t', 15, true, "lattice_type", "Lx", \
				"Ly", "Lz", "En", "dEn", "Sz", "Sx", "time taken");
		}
		log.close();
	}
	openFile(log, logname, ios::app);
	printSeparatedP(log, '\t', 15, true, 4, this->lat->get_type(), this->lat->get_Lx(), this->lat->get_Ly(), this->lat->get_Lz(), \
		energy, energy_error, av_op.s_z, std::real(av_op.s_x), tim_s(start));
	log.close();
};


#endif // !UI_H
