#include "include/user_interface/user_interface.h"


// --------------------------------------------------------   				 USER INTERFACE   				  --------------------------------------------------------

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

	// symmetries stuff
	this->k_sym = 0;
	this->p_sym = true;
	this->x_sym = true;
	this->sym = false;
	this->spectrum_size = 0.2;

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
	this->J_dot_dot = 1.0;

	// others 
	this->thread_num = 16;

	// rbm
	this->batch = std::pow(2, 10);
	this->mcSteps = 1000;
	this->n_blocks = 500;
	this->layer_mult = 2;
	this->block_size = 8;
	this->n_therm = uint(0.1 * this->n_blocks);
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
	this->n_therm = uint(0.1 * n_blocks);

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

	// dot - dot classical interaction
	choosen_option = "-Jd";
	this->set_option(this->J_dot_dot, argv, choosen_option, false);

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
	this->set_option(this->Kx, argv, choosen_option, false);
	choosen_option = "-ky";
	this->set_option(this->Ky, argv, choosen_option, false);
	choosen_option = "-kz";
	this->set_option(this->Kz, argv, choosen_option, false);
	choosen_option = "-k0";
	this->set_option(this->K0, argv, choosen_option, false);

	//---------- SYMMETRIES
	// translation
	choosen_option = "-ks";
	this->set_option(this->k_sym, argv, choosen_option, false);
	if (this->k_sym < 0 || this->k_sym >= Ns)
		this->k_sym = 0;

	// parity
	choosen_option = "-ps";
	this->set_option(this->p_sym, argv, choosen_option, false);

	// spin_flip
	choosen_option = "-xs";
	this->set_option(this->x_sym, argv, choosen_option, false);

	// include symmetries
	choosen_option = "-S";
	this->set_option(this->sym, argv, choosen_option, false);

	// spectrum size from the middle of the spectrum to test
	choosen_option = "-SS";
	this->set_option(this->spectrum_size, argv, choosen_option, true);

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

// -------------------------------------------------------- SIMULATIONS

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

	std::string dir = this->saving_dir;// +this->lat->get_type() + kPS + this->ham->get_info() + kPS + this->phi->get_info() + kPS;
	fs::create_directories(dir);

	// print energies
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
	std::map<u64, _type> states = phi->avSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
	bool convert = false;
	if (convert) {
		// convert to our basis
		Col<_type> states_col = SpinHamiltonian<_type>::map_to_state(states, ham->get_hilbert_size());
		this->av_op.reset();
		Operators<_type> op(this->lat);
		op.calculate_operators(states_col, this->av_op, this->nvisible <= maxed);
		this->save_operators(start, this->phi->get_info(), real(ground_rbm), real(standard_dev));
	}
	else {
		this->av_op = this->phi->get_op_av();
		this->save_operators(start, this->phi->get_info(), real(ground_rbm), real(standard_dev));
	}
	stouts("FINISHED EVERY THREAD", start);
	stout << "\t\t\t->" << VEQ(ground_rbm) << "+-" << standard_dev << EL;
}

template<typename _type, typename _hamtype>
inline void rbm_ui::ui<_type, _hamtype>::make_mc_classical()
{
	auto start = std::chrono::high_resolution_clock::now();
	stouts("STARTING THE SIMULATION FOR MINIMIZING CONFIGURATION SEEK AND USING: " + VEQ(thread_num), start);
	printSeparated(stout, ',', 5, true, VEQ(mcSteps), VEQ(n_blocks), VEQ(n_therm), VEQ(block_size));

	auto ran = randomGen();
	// make the lattice
	this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
	auto Ns = this->lat->get_Ns();
	this->nvisible = Ns;
	this->nhidden = this->layer_mult * this->nvisible;
	std::string lat_info = this->lat->get_info();
	this->av_op = avOperators(Lx, Ly, Lz, Ns, this->lat->get_type());
	stout << "->" << lat_info << EL;

	this->positions = v_1d<int>(Ns);
	// use all the positions for the lattice sites
	std::iota(this->positions.begin(), this->positions.end(), 0);

	// use no disorder at classical - quantum interaction
	this->J0_dot = 0;
	// set angles (let's set only the z'th direction (set all phis to 0)
	this->phis = arma::vec(Ns, arma::fill::zeros);
	// set thetas to 1 corresponding to all classical spins up
	this->thetas = arma::vec(Ns, arma::fill::zeros);

	v_1d<double> e_ferro = {};
	v_1d<double> e_ferro_ed = {};
	v_1d<double> e_aferro = {};
	v_1d<double> e_aferro_ed = {};
	double step = 0.1;

	for (int ferro = 0; ferro <= 1; ferro++) {
		stout << "\t\t->DOING " << (bool(ferro) ? "ferromagnet" : "antiferromagnet") << EL;
		// ------------------------ FERROMAGNETIC CLASSICAL SPINS ------------------------
		if (bool(ferro))
			for (int i = 0; i < thetas.size(); i++) {
				this->thetas(i) = 0.0;
			}
		else
		{
			for (int y = 0; y < this->lat->get_Ly(); y++) {
				bool y_start = y % 2 == 0 ? 0 : 1;
				for (int x = 0; x < this->lat->get_Lx(); x++) {
					int n = x + this->lat->get_Lx() * y;
					this->thetas(n) = x % 2 == 0 ? int(y_start) : int(!y_start);
				}
			}
		}
		//this->thetas.print("thetas=");
		this->delta = 1.0;

		vec sin_phis = sin(this->phis * TWOPI);
		vec cos_phis = cos(this->phis * TWOPI);
		vec sin_thetas = sin(this->thetas * PI);
		vec cos_thetas_rbm = cos(this->thetas * PI);
		vec cos_thetas_ed = cos(this->thetas * PI);
		uint threads = this->thread_num / 2 < 1 ? 1 : uint(this->thread_num / 2);
		cos_thetas_rbm.print(VEQ(ferro));
		//#pragma omp parallel for num_threads(threads)
		for (int j0 = 0; j0 <= 40; j0++) {
			v_1d<double> Jd = { 0.0, 0.0, -2.0 + j0 * step };
			stout << "\t-> doing " << VEQ(Jd[2]) << EL;
			// create Hamiltonians
			std::shared_ptr<Heisenberg_dots<double>> hamiltonian_rbm = std::make_shared<Heisenberg_dots<double>>(J, 0.0, 0.0, 0.0, 0.0, 0.0, delta, lat, positions, Jd, 0.0, this->J_dot_dot);
			std::shared_ptr<Heisenberg_dots<double>> hamiltonian_ed = std::make_shared<Heisenberg_dots<double>>(J, 0.0, 0.0, 0.0, 0.0, 0.0, delta, lat, positions, Jd, 0.0, this->J_dot_dot);
			hamiltonian_rbm->set_angles(sin_phis, sin_thetas, cos_phis, cos_thetas_rbm);
			hamiltonian_ed->set_angles(sin_phis, sin_thetas, cos_phis, cos_thetas_ed);
			auto model_info = hamiltonian_rbm->get_info();
			stout << "\t-> " << VEQ(model_info) << EL;

			// rbm stuff
			auto ph = std::make_unique<rbmState<_type, double>>(nhidden, nvisible, hamiltonian_rbm, lr, batch, this->thread_num);
			auto rbm_info = ph->get_info();
			stout << "\t\t->" << VEQ(rbm_info) << EL;
			string order = string(ferro ? "ferro" : "aferro") + ",Lx=" + STR(this->lat->get_Lx()) + ",Ly=" + STR(this->lat->get_Ly()) + ",bc=" + STR(_BC) + ",d=" + STR(this->lat->get_Dim());
			string dir = this->saving_dir + kPS + order + kPS;
			fs::create_directories(dir);
			dir = dir + rbm_info + kPS;
			fs::create_directories(dir);
			dir = dir + model_info + kPS;
			fs::create_directories(dir);

			string dir_ed = this->saving_dir + kPS + order + kPS;
			fs::create_directories(dir_ed);
			dir_ed = dir_ed + "ed" + kPS;
			fs::create_directories(dir_ed);
			dir_ed = dir_ed + model_info + kPS;
			fs::create_directories(dir_ed);

			std::ofstream fileSave;

			// ------------------- calculator rbm -------------------
			// monte carlo for energy
			auto energies = ph->mcSampling(this->mcSteps, n_blocks, n_therm, block_size, n_flips, dir);
			auto energies_tail = energies.tail(block_size);
			double ground_rbm = std::real(arma::mean(energies_tail));

			// ------------------- calculator ed -------------------
			double ground_ed = 0.0;
			avOperators av_operator(Lx, Ly, Lz, Ns, this->lat->get_type());
			if (Ns <= 14) {
				calculate_ed<double>(ground_ed, ground_rbm, hamiltonian_ed);
#ifdef PLOT
				plt::axhline(ground_ed);
				plt::annotate(VEQ(ground_ed) + ",\n" + VEQ(ground_rbm), mcSteps / 3, (ground_rbm) / 2);
				PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", hamiltonian_rbm->get_info() + "\nrbm:" + ph->get_info());
				SAVEFIG(dir + "energy" + ".png", true);
#endif
				Col<_hamtype> eigvec = hamiltonian_ed->get_eigenState(0);
				Operators<_hamtype> op(this->lat);
				op.calculate_operators(eigvec, av_operator, true);
				// --------------------- compare sigma_z ---------------------

				// S_z at each site
				std::string filename = dir_ed + "_sz_site";
				openFile(fileSave, filename + ".dat", ios::out);
				print_vector_1d(fileSave, this->av_op.s_z_i);
				fileSave.close();
				PLOT_V1D(this->av_op.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + model_info + "\n");
				SAVEFIG(filename + ".png", false);

				// S_z correlations
				filename = dir_ed + "_sz_corr";
				openFile(fileSave, filename + ".dat", ios::out);
				print_mat(fileSave, this->av_op.s_z_cor);
				fileSave.close();
			}
			else {
				PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", hamiltonian_rbm->get_info() + "\nrbm:" + ph->get_info());
				SAVEFIG(dir + "energy" + ".png", true);
			}

			// ------------------- sampling rbm -------------------
			ph->avSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
			av_operator.reset();
			av_operator = ph->get_op_av();

			auto fileRbmEn_name = dir + "energies";
			std::ofstream fileRbmEn;
			openFile(fileRbmEn, fileRbmEn_name + ".dat", ios::out);
			for (auto i = 0; i < energies.size(); i++)
				printSeparatedP(fileRbmEn, '\t', 8, true, 5, i, real(energies(i)));
			fileRbmEn.close();

			// other observables

			string filename = "";
			// --------------------- compare sigma_z

			// S_z at each site
			filename = dir + "_sz_site";
			openFile(fileSave, filename + ".dat", ios::out);
			print_vector_1d(fileSave, this->av_op.s_z_i);
			fileSave.close();
			PLOT_V1D(this->av_op.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + model_info + "\n");
			SAVEFIG(filename + ".png", false);

			// S_z correlations
			filename = dir + "_sz_corr";
			openFile(fileSave, filename + ".dat", ios::out);
			print_mat(fileSave, this->av_op.s_z_cor);
			fileSave.close();

			if (bool(ferro)) {
				e_ferro.push_back(ground_rbm);
				e_ferro_ed.push_back(ground_ed);
			}
			else {
				e_aferro.push_back(ground_rbm);
				e_aferro_ed.push_back(ground_ed);
			}
		}
	}
	std::ofstream file;
	openFile(file, this->saving_dir + lat_info + ".dat", ios::out | ios::app);
	for (int i = 0; i <= 40; i++) {
		this->J0 = -2.0 + i * step;
		printSeparatedP(file, '\t', 17, true, 13, this->J, this->J0, e_ferro[i], e_aferro[i], e_ferro_ed[i], e_aferro_ed[i]);
	}
	file.close();
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
	this->nvisible = Ns;
	this->nhidden = this->layer_mult * this->nvisible;
	this->phi = std::make_unique<rbmState<_type, _hamtype>>(nhidden, nvisible, ham, lr, batch, thread_num);
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

	if (Ns <= maxed) {
		this->av_op.reset();
		auto diag_time = std::chrono::high_resolution_clock::now();
		stout << "\n\n-> starting ED for:\n\t-> " + ham->get_info() << EL;
		// define the operators class

		this->ham->hamiltonian();
		this->ham->diag_h(false);

		Operators<_hamtype> op(this->lat);
		auto ground_ed = std::real(ham->get_eigenEnergy(0));
		auto excited_ed = std::real(ham->get_eigenEnergy(1));
		Col<_hamtype> eigvec = ham->get_eigenState(0);

		// calculate operators for a ground state
		op.calculate_operators(eigvec, this->av_op, true);

		// save operators
		this->save_operators(diag_time, "", ground_ed, 0);

		u64 state = 0;
		auto sz = this->av_op.s_z;
		auto sx = this->av_op.s_x;
		auto relative_error = abs(std::real(ground_ed - ground_rbm)) / abs(ground_ed) * 100.;

		stouts("\t\t-> finished ED", diag_time);
		stout << "\t\t\t->" << VEQP(ground_ed, 8) << EL;
		stout << "\t\t\t->" << VEQP(ground_rbm, 8) << EL;
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

	std::string dir = this->saving_dir + kPS + this->lat->get_type() + kPS + this->ham->get_info() + kPS;
	if (name != "") dir = dir + name + kPS;
	fs::create_directories(dir);

	std::string filename = "";
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
	print_mat(fileSave, this->av_op.s_z_cor);
	fileSave.close();

	// --------------------- compare sigma_y ---------------------
	// S_y at each site
	//filename = dir + "_sy_site";
	//openFile(fileSave, filename + ".dat", ios::out);
	//print_vector_1d(fileSave, this->av_op.s_x_i);
	//fileSave.close();
	//PLOT_V1D(this->av_op.s_x_i, "lat_site", "$S^x_i$", "$S^x_i$\n" + ham->get_info() + "\n" + name);
	//SAVEFIG(filename + ".png", false);
	//
	//// S_z correlations
	//filename = dir + "_sx_corr_";
	//openFile(fileSave, filename + ".dat", ios::out);
	//print_mat(fileSave, this->av_op.s_x_cor);
	//fileSave.close();

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
	print_mat(fileSave, this->av_op.s_x_cor);
	fileSave.close();

	// --------------------- entropy ----------------------
	if (Ns <= maxed) {
		filename = dir + "_ent_entro";
		openFile(fileSave, filename + ".dat", ios::out);
		print_vector_1d(fileSave, this->av_op.ent_entro);
		fileSave.close();
		PLOT_V1D(this->av_op.ent_entro, "bond_cut", "$S_0(L)$", "Entanglement entropy\n" + ham->get_info() + "\n" + name);
		SAVEFIG(filename + ".png", false);
	}

	// --------------------- neigbors ----------------------
	filename = dir + "_neighbor_corr_";
	openFile(fileSave, filename + ".dat", ios::out);
	printSeparatedP(fileSave, '\t', 14, true, 5, "<S^x_iS^x_(i+m)>", "<S^y_iS^y_(i+m)>", "<S^z_iS^z_(i+m)>");
	printSeparatedP(fileSave, '\t', 14, true, 5, real(this->av_op.s_x_nei), real(this->av_op.s_y_nei), real(this->av_op.s_z_nei));
	fileSave.close();

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

// -------------------------------- HAMILTONIAN WITH SYMMETRIES --------------------------------

template<typename _type, typename _hamtype>
inline void rbm_ui::ui<_type, _hamtype>::symmetries_double(clk::time_point start)
{
	stout << "->using real" << EL;
	if (sym) {
		if (this->model_name == 0)
			this->ham_d = std::make_shared<ising_sym::IsingModelSym<double>>(J, g, h, lat, k_sym, p_sym, x_sym, this->thread_num);
		else
			this->ham_d = std::make_shared<xyz_sym::XYZSym<double>>(lat, this->thread_num, k_sym, p_sym, x_sym);
	}
	else {
		if (this->model_name == 0)
			this->ham_d = std::make_shared<IsingModel<double>>(J, J0, g, g0, h, w, lat);
		else
			this->ham_d = std::make_shared<XYZ<double>>(lat, true);
		this->ham_d->hamiltonian();
	}

	stouts("\t->finished buiding Hamiltonian", start);
	stout << "\t->" << this->ham_d->get_info() << EL;
	this->ham_d->diag_h(false);

	stouts("\t->finished diagonalizing Hamiltonian", start);
	const u64 N = this->ham_d->get_hilbert_size();

	std::string name = "spectrum_num=" + STR(N);
	stout << "->middle_spectrum_size : " << name << EL;

	std::string dir = this->saving_dir + kPS + this->ham_d->get_info() + kPS;
	fs::create_directories(dir);
	std::ofstream file;
	std::ofstream fileAv;

	// save energies to check
	openFile(file, dir + "energies," + name + ".dat");
	for (u64 i = 0; i < N; i++)
		file << STRP(this->ham_d->get_eigenEnergy(i), 14) << EL;
	file.close();
	this->ham_d->get_eigenvalues().save(dir + "energies," + name + ".bin", arma::raw_binary);
	this->ham_d->get_eigenvalues().save(arma::hdf5_name(dir + "energies," + name + ".h5", "energy"));
	this->ham_d->clear_energies();
	this->ham_d->clear_hamiltonian();

	// calculate the reduced density matrices
	Operators<double> op(this->lat);

	// iterate through bond cut
	int bond_num = this->lat->get_Ns() / 2;
	arma::mat entropies(bond_num, N, arma::fill::zeros);

	for (int i = 1; i <= bond_num; i++) {
		// iterate through the state
		stout << "\t->doing : " << VEQ(i) << EL;

#pragma omp parallel for num_threads(this->thread_num)
		for (u64 j = 0; j < N; j++) {
			u64 idx = j;
			const Col<double> state = this->ham_d->get_eigenStateFull(idx);
			auto entro = op.entanglement_entropy(state, i);
			entropies(i - 1, j) = entro;
		}
		auto mean = arma::mean(entropies.row(i - 1));
		stout << "\t\t->mean : " << VEQP(mean, 6) << EL;
	}

	// save binary file
	std::string filename = dir + "entropies," + name + ".bin";
	entropies.save(filename, arma::raw_binary);
	entropies.save(arma::hdf5_name(dir + "entropies," + name + ".h5", "entropy"));
	if (this->lat->get_Ns() < 16) {
		filename = dir + "entropies," + name + ".txt";
		entropies.save(filename, arma::arma_ascii);
	}

	const u64 av_energy_idx = this->ham_d->get_en_av_idx();
	// iterate through fractions
	for (v_1d<double> fractions = { 0.1, 0.25, 0.5 }; double frac : fractions) {
		u64 spectrum_num = frac <= 1.0 ? frac * N : static_cast<u64>(frac);
		name = ((frac <= 1.0) ? "spectrum_num=" + STRP(frac, 3) + "x" + STR(N) + "=" + STR(spectrum_num) : VEQ(frac));
		// define the window to calculate the entropy
		if (long(av_energy_idx) - long(spectrum_num / 2) < 0 || av_energy_idx + u64(spectrum_num / 2) >= N)
			continue;
		openFile(fileAv, dir + "av_entropies," + name + ".dat", ios::out);

		auto subview = entropies.submat(0, av_energy_idx - u64(spectrum_num / 2), bond_num - 1, av_energy_idx + u64(spectrum_num / 2));
		for (int i = 1; i <= bond_num; i++) {
			double mean = arma::mean(subview.row(i - 1));
			printSeparated(fileAv, '\t', 18, false, i);
			printSeparatedP(fileAv, '\t', 18, true, 12, mean);
		}
		fileAv.close();
	}
}

template<typename _type, typename _hamtype>
inline void rbm_ui::ui<_type, _hamtype>::symmetries_cpx(clk::time_point start)
{
	stout << "->using complex" << EL;
	if (sym) {
		if (this->model_name == 0)
			this->ham_cpx = std::make_shared<ising_sym::IsingModelSym<cpx>>(J, g, h, lat, k_sym, p_sym, x_sym, this->thread_num);
		else
			this->ham_cpx = std::make_shared<xyz_sym::XYZSym<cpx>>(lat, this->thread_num, k_sym, p_sym, x_sym);
	}
	else {
		if (this->model_name == 0)
			this->ham_cpx = std::make_shared<IsingModel<cpx>>(J, J0, g, g0, h, w, lat);
		else
			this->ham_cpx = std::make_shared<XYZ<cpx>>(lat, true);
		this->ham_cpx->hamiltonian();
	}

	stouts("\t->finished buiding Hamiltonian", start);
	stout << "\t->" << this->ham_cpx->get_info() << EL;
	this->ham_cpx->diag_h(false);
	stouts("\t->finished diagonalizing Hamiltonian", start);
	const u64 N = this->ham_cpx->get_hilbert_size();



	std::string name = "spectrum_num=" + STR(N);
	stout << "->middle_spectrum_size : " << name << EL;


	std::string dir = this->saving_dir + kPS + this->ham_cpx->get_info() + kPS;
	fs::create_directories(dir);
	std::ofstream file;
	std::ofstream fileAv;

	// save energies to check
	openFile(file, dir + "energies," + name + ".dat");
	for (u64 i = 0; i < N; i++)
		file << STRP(this->ham_cpx->get_eigenEnergy(i), 14) << EL;
	file.close();
	this->ham_cpx->get_eigenvalues().save(dir + "energies," + name + ".bin", arma::raw_binary);
	this->ham_cpx->get_eigenvalues().save(arma::hdf5_name(dir + "energies," + name + ".h5", "energy"));
	this->ham_cpx->clear_energies();
	this->ham_cpx->clear_hamiltonian();

	// calculate the reduced density matrices
	Operators<cpx> op(this->lat);

	// iterate through bond cut
	int bond_num = this->lat->get_Ns() / 2;
	arma::mat entropies(bond_num, N, arma::fill::zeros);
	//Mat<cpx> U = this->ham_cpx->symmetryRotationMat();
	for (int i = 1; i <= bond_num; i++) {
		// iterate through the state
		stout << "\t->doing : " << VEQ(i) << EL;

#pragma omp parallel for num_threads(this->thread_num)
		for (u64 j = 0; j < N; j++) {
			u64 idx = j;
			const Col<cpx> state = this->ham_cpx->get_eigenStateFull(idx);
			auto entro = op.entanglement_entropy(state, i);
			entropies(i - 1, j) = entro;
		}
		auto mean = arma::mean(entropies.row(i - 1));
		stout << "\t\t->mean : " << VEQP(mean, 6) << EL;
	}
	// save binary file
	std::string filename = dir + "entropies," + name + ".bin";
	entropies.save(filename, arma::raw_binary);
	entropies.save(arma::hdf5_name(dir + "entropies," + name + ".h5", "entropy"));
	if (this->lat->get_Ns() < 16) {
		filename = dir + "entropies," + name + ".txt";
		entropies.save(filename, arma::arma_ascii);
	}
	//openFile(file, filename, ios::out | ios::binary);
	//stout << "\t\t->SAVING: " + filename;
	//for (int i = 0; i < bond_num; i++)
	//	for (int j = 0; j < N; j++) {
	//		double val = entropies(i, j);
	//		file.write(reinterpret_cast<const char*>(&val), sizeof(double));
	//	}
	//if(!file.good())
	//	stout << "\t\t->Error occurred at writing time!" << endl;
	//file.close();

	const u64 av_energy_idx = this->ham_cpx->get_en_av_idx();
	// iterate through fractions
	for (v_1d<double> fractions = { 0.1, 0.25, 0.5 }; double frac : fractions) {
		u64 spectrum_num = frac <= 1.0 ? frac * N : static_cast<u64>(frac);
		name = ((frac <= 1.0) ? "spectrum_num=" + STRP(frac, 3) + "x" + STR(N) + "=" + STR(spectrum_num) : VEQ(frac));
		// define the window to calculate the entropy
		if (long(av_energy_idx) - long(spectrum_num / 2) < 0 || av_energy_idx + u64(spectrum_num / 2) >= N)
			continue;
		openFile(fileAv, dir + "av_entropies," + name + ".dat", ios::out);

		auto subview = entropies.submat(0, av_energy_idx - u64(spectrum_num / 2), bond_num - 1, av_energy_idx + u64(spectrum_num / 2));
		for (int i = 1; i <= bond_num; i++) {
			double mean = arma::mean(subview.row(i - 1));
			printSeparated(fileAv, '\t', 18, false, i);
			printSeparatedP(fileAv, '\t', 18, true, 12, mean);
		}
		fileAv.close();
	}

}

template<typename _type, typename _hamtype>
inline void rbm_ui::ui<_type, _hamtype>::make_simulation_symmetries()
{
	auto start = std::chrono::high_resolution_clock::now();
	stouts("STARTING THE CALCULATIONS FOR QUANTUM ISING HAMILTONIAN: " + VEQ(thread_num), start);
	stout << "->" << (sym ? "" : "not ") << "including symmetries" << EL;

	this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
	stout << "->" << this->lat->get_info() << EL;
	auto Ns = this->lat->get_Ns();

	// initialize hamiltonian
	if (this->sym) {
		if (this->k_sym == 0 || this->k_sym == this->lat->get_Ns() / 2)
			this->symmetries_double(start);
		else
			this->symmetries_cpx(start);
	}
	else
		this->symmetries_double(start);
	stouts("FINISHED THE CALCULATIONS FOR QUANTUM ISING HAMILTONIAN: ", start);

}

template<typename _type, typename _hamtype>
void rbm_ui::ui<_type, _hamtype>::make_symmetries_test(int l)
{

	// -------------------------------------------------------------------- calculate full ed obc
	this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, 1, 1);
	const auto Ns = this->lat->get_Ns();
	stout << "Doing->" << this->lat->get_info() << EL;

	const int La = (l == -1) ? int(Ns / 2) : l;
	std::ofstream file, file2, states;
	Operators<double> op_ed_obc(this->lat);
	this->g = 0.8;
	this->h = 0.9;
	if (this->model_name == 0)
		this->ham_d = std::make_shared<IsingModel<double>>(J, 0.0, g, 0.0, h, 0.0, lat);
	else
		this->ham_d = std::make_shared<XYZ<double>>(lat, true);

	this->ham_d->hamiltonian();
	stout << "\t->" << "finished building: " << this->ham_d->get_info() << EL;
	this->ham_d->diag_h(false);
	stout << "\t->" << "finished diagonalizing: " << this->ham_d->get_info() << EL;


	std::string dir = this->saving_dir + kPS + this->ham_d->get_info() + kPS;
	fs::create_directories(dir);
	openFile(file, dir + "energies.dat");
	for (u64 i = 0; i < this->ham_d->get_hilbert_size(); i++)
		file << this->ham_d->get_eigenEnergy(i) << EL;
	stout << "\t\t->" << "saved energies: " << this->ham_d->get_info() << EL;
	file.close();

	arma::mat entropies_mat(La, this->ham_d->get_hilbert_size(), arma::fill::zeros);
	for (int i = 1; i <= La; i++) {
		// iterate through the state
		stout << "\t->doing : " << VEQ(i) << EL;

#pragma omp parallel for num_threads(this->thread_num)
		for (u64 j = 0; j < this->ham_d->get_hilbert_size(); j++) {
			u64 idx = j;
			const Col<double> state = this->ham_d->get_eigenStateFull(idx);
			auto entro = op_ed_obc.entanglement_entropy(state, i);
			entropies_mat(i - 1, j) = entro;
		}
		auto mean = arma::mean(entropies_mat.row(i - 1));
		stout << "\t\t->obc_mean : " << VEQP(mean, 6) << EL;
	}
	// save binary file
	std::string filename = dir + "entropies.bin";
	entropies_mat.save(filename, arma::raw_binary);

	arma::mat ens_obc = this->ham_d->get_eigenvalues();
	filename = dir + "energies.bin";
	ens_obc.save(filename, arma::raw_binary);
	// save binary file
	if (this->lat->get_Ns() < 16) {
		auto filename = dir + "entropies_obc" + ".txt";
		entropies_mat.save(filename, arma::arma_ascii);
	}

	return;
	// -------------------------------------------------------------------- calculate full ed pbc
	this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, 1, 0);
	stout << "Doing->" << this->lat->get_info() << EL;

	if (this->model_name == 0)
		this->ham_d = std::make_shared<IsingModel<double>>(J, 0.0, g, 0.0, h, 0.0, lat);
	else
		this->ham_d = std::make_shared<XYZ<double>>(lat, false);

	this->ham_d->hamiltonian();
	stout << "\t->" << "finished building: " << this->ham_d->get_info() << EL;
	this->ham_d->diag_h(false);
	stout << "\t->" << "finished diagonalizing: " << this->ham_d->get_info() << EL;

	// -------------------------------------------------------------------- calculate sym ed pbc

	v_1d<double> energies_sym = {};
	v_1d<double> entropies = {};
	v_1d<std::pair<int, int>> sym = {};

	Operators<cpx> op(this->lat);
	dir = this->saving_dir + kPS + this->lat->get_info() + (this->model_name == 0 ? ",ising" : ",xyz") + kPS;
	fs::create_directories(dir);
	openFile(file, dir + "log.dat");
	openFile(states, dir + "states.dat");

	double entro = 0;
	u64 Nh = pow(2, Ns);
	u64 state_num = 0;
	arma::cx_mat H(Nh, Nh, arma::fill::zeros);

	v_1d<int> ps = {};
	for (int k = 0; k < Ns; k++) {
		if (k == 0 || k == int(Ns / 2))
			ps = { 0, 1 };
		else
			ps = { 1 };

		for (auto p : ps) {
			if (this->model_name == 0)
				this->ham_cpx = std::make_shared<ising_sym::IsingModelSym<cpx>>(J, g, h, lat, k, p, 1, this->thread_num);
			else
				this->ham_cpx = std::make_shared<xyz_sym::XYZSym<cpx>>(lat, this->thread_num, k, p, 1);

			file << "\tDoing: " << VEQ(k) << "," << VEQ(p) << ".\tHilbert size = " << this->ham_cpx->get_hilbert_size() << EL;
			file << "\t->" << this->ham_cpx->get_info() << EL;
			this->ham_cpx->diag_h(false);

			// create rotation matrix
			Mat<cpx> U = this->ham_cpx->symmetryRotationMat();
			Mat<cpx> Hafter = U * arma::Mat<cpx>(this->ham_cpx->get_hamiltonian()) * U.t();
			H += Hafter;
			state_num += this->ham_cpx->get_hilbert_size();
			for (u64 i = 0; i < this->ham_cpx->get_hilbert_size(); i++) {
				auto En = this->ham_cpx->get_eigenEnergy(i);
				energies_sym.push_back(En);
				Col<cpx> state = this->ham_cpx->get_eigenState(i);
				Col<cpx> transformed_state = U * state;

				auto en_state = cdot(transformed_state, H * transformed_state);
				if (auto tmp = abs(En - en_state); tmp > 1e-12)
					stout << VEQ(En) << "\t" << VEQ(en_state) << "\t" << VEQ(tmp) << EL;
				if (i == 0 || i == 1) {
					states << VEQ(k) << "," << VEQ(p) << "," << VEQ(en_state) << ":\t";
					for (int z = 0; z < Nh; z++) {
						states << transformed_state(z) << "\t";
					}
					states << EL;
				}
				auto entropy = op.entanglement_entropy(transformed_state, La);
				auto entropy_transform = op.entanglement_entropy(this->ham_cpx->get_eigenStateFull(i), La);
				if (auto tmp = abs(entropy - entropy_transform); tmp > 1e-12)
					stout << VEQ(entropy) << "\t" << VEQ(entropy_transform) << "\t" << VEQ(tmp) << EL;

				entropies.push_back(entropy_transform);
				entro += entropy_transform;
				sym.push_back(std::make_pair(k, p));

				file << "\t\t->" << VEQP(this->ham_cpx->get_eigenEnergy(i), 5) << "\t" << "after_trasform:" << arma::cdot(transformed_state, Hafter * transformed_state) << EL;
			}
		}
	}
	entro /= double(entropies.size());
	stout << VEQ(Nh) << "\t" << VEQ(state_num) << EL;
	// --------------------------- sort stuff ---------------------------
	v_1d<int> index(energies_sym.size(), 0);
	for (int i = 0; i != index.size(); i++) {
		index[i] = i;
	}
	sort(index.begin(), index.end(),
		[&](const int& a, const int& b) {
			return (energies_sym[a] < energies_sym[b]);
		}
	);
	states.close();

	// -------------------------- check the ed with symmeties --------------------------
	double entro_ed = 0;

	dir = this->saving_dir + kPS + this->lat->get_info() + (this->model_name == 0 ? ",ising" : ",xyz") + kPS;
	openFile(file2, dir + "all_compare.dat");
	openFile(states, dir + "states_ed.dat");
	printSeparated(file2, '\t', 15, true, "E_ed", "E_sym", "Symmetry", "S_ed", "S_sym");
	Operators<double> op_ed_pbc(this->lat);

	for (u64 i = 0; i < this->ham_d->get_hilbert_size(); i++) {
		auto E_ed = this->ham_d->get_eigenEnergy(i);
		const Col<double> state = (this->ham_d->get_eigenState(i));
		auto ent_ed = op_ed_pbc.entanglement_entropy(state, Ns / 2);

		// sort
		auto sort_i = index[i];
		auto E_sym = energies_sym[sort_i];
		const auto [k, p] = sym[sort_i];
		auto ent_sym = entropies[sort_i];

		if (i <= 24) {
			states << VEQ(k) << "," << VEQ(p) << "," << VEQ(E_ed) << ":-:" << VEQ(E_sym) << ":\t";
			for (int z = 0; z < Nh; z++) {
				states << state(z) << "\t";
			}
			states << EL;
		}
		printSeparatedP(file2, '\t', 15, true, 7, E_ed, E_sym, STR(k) + "," + STR(p), ent_ed, ent_sym);
	}
	entro_ed /= double(entropies.size());
	stout << VEQP(entro, 6) << "\t<-sym\t" << VEQP(entro_ed, 6) << "\t<-ed\t" << EL;
	file2.close();
	states.close();
	// compare Hamiltonians


	auto H0 = arma::mat(this->ham_d->get_hamiltonian());
	auto N = H0.n_cols;
	Mat<cpx> res = H - (H0);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			if (abs(res(i, j)) > 1e-15)
				printSeparated(file, '\t', 32, true, i, j, res(i, j), H0(i, j));
	file.close();
}

