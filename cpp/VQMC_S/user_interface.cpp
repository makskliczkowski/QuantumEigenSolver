#include "include/user_interface/user_interface.h"

// -------------------------------------------------------- PARSERS

/*
* @brief  Setting parameters to default
*/
void UI::setDefault()
{

	// lattice stuff
	this->latP.typ = SQ; 													// for non_numeric data
	this->latP.dim = 1;
	this->latP.bc = 0;
	this->latP.Lx = 10;
	this->latP.Ly = 1;
	this->latP.Lz = 1;

	// symmetries stuff
	this->symP.kSec = 0;
	this->symP.pxSec = true;
	this->symP.pySec = true;
	this->symP.pzSec = true;
	this->symP.xSec = true;
	this->symP.U1Sec = 0;

	// define basic model
	//this->model_name = impDef::ham_types::ising;
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
	this->phis = arma::vec({ 0 });
	this->thetas = arma::vec({ 1 });
	this->J_dot = { 0.0,0.0,-1.0 };
	this->J0_dot = 0.0;
	this->J_dot_dot = 1.0;

	// others 
	this->thread_number = 1;

	// rbm
	this->nqsP.lr = 1e-2;
	this->nqsP.blockSize = 8;
	this->nqsP.nBlocks = 500;
	this->nqsP.mcSteps = 1000;
	this->nqsP.batch = std::pow(2, 10);
	this->nqsP.nVisible = latP.lat->get_Ns();
	this->nqsP.nHidden = 2ll * latP.lat->get_Ns();
	this->nqsP.layersDim = { this->nqsP.nVisible , this->nqsP.nHidden };
	this->nqsP.nTherm = uint(0.1 * this->nqsP.nBlocks);
	this->nqsP.nFlips = 1;
}

/*
* @brief model parser
* @param argc number of line arguments
* @param argv line arguments
*/
inline void UI::parseModel(int argc, cmdArg& argv)
{
	// set default at first
	this->setDefault();

	std::string choosen_option = "";

	// -------------------- SIMULATION PARAMETERS --------------------

	this->setOption(this->nqsP.mcSteps, argv, "m");
	this->setOption(this->nqsP.batch, argv, "b");
	this->setOption(this->nqsP.nBlocks, argv, "nb"); this->nqsP.nTherm = uint(0.1 * nqsP.nBlocks);
	this->setOption(this->nqsP.blockSize, argv, "bs");
	this->setOption(this->nqsP.nHidden, argv, "nh");
	SETOPTION(this->nqsP, lr);
	
	// -------------------- Lattice --------------------

	this->setOption(this->latP.typ, argv, "-l");
	this->setOption(this->latP.dim, argv, "-d");
	SETOPTION(this->latP, Lx);
	SETOPTION(this->latP, Ly);
	SETOPTION(this->latP, Lz);
	SETOPTION(this->latP, bc);
	int Ns = latP.Lx * latP.Ly * latP.Lz;

	// ---------- model

	// model type
	//this->setOption(this->model_name, argv, "-mod", false);
	this->setOption(this->J, argv, "-J");
	this->setOption(this->J0, argv, "-J0");
	this->setOption(this->J_dot_dot, argv, "-Jd");
	this->setOption(this->g, argv, "-g");
	this->setOption(this->g0, argv, "-g0");
	this->setOption(this->h, argv, "-h");
	this->setOption(this->w, argv, "-w");

	// --- heisenberg ---
	this->setOption(this->delta, argv, "-dlt");
	
	// --- XYZ --- 
	this->setOption(this->Delta_b, argv, "-dlt2");
	this->setOption(this->Jb, argv, "-J2");
	this->setOption(this->eta_a, argv, "-eta");
	this->setOption(this->eta_b, argv, "-eta2");
	
	// --- kitaev ---
	this->setOption(this->Kx, argv, "-kx");
	this->setOption(this->Ky, argv, "-ky");
	this->setOption(this->Kz, argv, "-kz");
	this->setOption(this->K0, argv, "-k0");

	//---------- SYMMETRIES
	
	// translation
	SETOPTION(this->symP, kSec);
	SETOPTION(this->symP, pxSec);
	SETOPTION(this->symP, pySec);
	SETOPTION(this->symP, pzSec);
	SETOPTION(this->symP, xSec);
	SETOPTION(this->symP, U1Sec);
	SETOPTION(this->symP, S);

	// ---------- OTHERS
	this->setOption(this->quiet, argv, "-q");
	this->setOption(this->thread_number, argv, "-th");
	
	// --------- HELP
	if (std::string option = this->getCmdOption(argv, "-hlp"); option != "")
		exitWithHelp();

	// later function choice
	this->setOption(this->chosen_funtion, argv, "-fun");

	//---------- DIRECTORY

	bool set_dir = false;
	choosen_option = "dir";
	if (std::string option = this->getCmdOption(argv, "-" + choosen_option); option != "") {
		this->setOption(this->main_dir, argv, choosen_option);
		set_dir = true;
	}
	if (!set_dir)
		this->main_dir = fs::current_path().string() + kPS + "results" + kPS;

	// create the directories
	createDir(this->main_dir);
}

/*
* @brief chooses the method to be used later based on input -fun argument
*/
void UI::funChoice()
{
	switch (this->chosen_funtion)
	{
	case -1:
		// default case of showing the help
		this->exitWithHelp();
		break;
	//case 0:
	//	// test
	//	this->make_symmetries_test();
	//	break;
	//case 11:
	//	// calculate the simulation with classical degrees of freedom
	//	this->make_mc_classical();
	//	break;
	//case 12:
	//	// check the minimum of energy when classical spins are varied with angle and with interaction
	//	this->make_mc_angles_sweep();
	//	break;
	//case 13:
	//	// check the properties of Kitaev model when the interations are 
	//	this->make_mc_kitaev_sweep();
	//	break;
	//case 14:
	//	// make simulation for a single model based on input parameters
	//	this->make_simulation();
	//	break;
	//case 21:
	//	// this option utilizes the Hamiltonian with symmetries calculation
	//	this->make_simulation_symmetries();
	//	break;
	//case 22:
	//	// this option utilizes the Hamiltonian with symmetries calculation - sweep!
	//	this->make_simulation_symmetries_sweep();
	//	break;
	default:
		// default case of showing the help
		this->exitWithHelp();
		break;
	}
}
// -------------------------------------------------------- SIMULATIONS

/*
*
*/
//template<typename _type, typename _hamtype>
//void rbm_ui::ui<_type, _hamtype>::ui::make_simulation()
//{
//	auto start = std::chrono::high_resolution_clock::now();
//	this->define_models();
//	stouts("STARTING THE SIMULATION FOR GROUNDSTATE SEEK AND USING: " + VEQ(thread_num), start);
//	printSeparated(stout, ',', 5, true, VEQ(mcSteps), VEQ(n_blocks), VEQ(n_therm), VEQ(block_size));
//	// monte carlo
//	auto energies = this->phi->mcSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
//
//	std::string dir = this->saving_dir + this->lat->get_type() + kPS + this->ham->get_info() + kPS + this->phi->get_info() + kPS;
//	fs::create_directories(dir);
//
//	// print energies
//	auto fileRbmEn_name = dir + "energies";
//	std::ofstream fileRbmEn;
//	openFile(fileRbmEn, fileRbmEn_name + ".dat", ios::out);
//	for (auto i = 0; i < energies.size(); i++)
//		printSeparatedP(fileRbmEn, '\t', 8, true, 5, i, energies(i).real());
//	fileRbmEn.close();
//
//	// calculate the statistics of a simulation
//	auto energies_tail = energies.tail(block_size);
//	_type standard_dev = arma::stddev(energies_tail);
//	_type ground_rbm = arma::mean(energies_tail);
//
//	// if the size is small enough
//	this->compare_ed(std::real(ground_rbm));
//	PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", ham->get_info() + "\nrbm:" + this->phi->get_info());
//	SAVEFIG(fileRbmEn_name + ".png", true);
//	// ------------------- check ground state
//	std::map<u64, _type> states = phi->avSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
//	bool convert = false;
//	if (convert) {
//		// convert to our basis
//		Col<_type> states_col = SpinHamiltonian<_type>::map_to_state(states, ham->get_hilbert_size());
//		this->av_op.reset();
//		Operators<_type> op(this->lat);
//		op.calculate_operators(states_col, this->av_op, this->nvisible <= maxed);
//		this->save_operators(start, this->phi->get_info(), real(ground_rbm), real(standard_dev));
//	}
//	else {
//		this->av_op = this->phi->get_op_av();
//		this->save_operators(start, this->phi->get_info(), real(ground_rbm), real(standard_dev));
//	}
//	stouts("FINISHED EVERY THREAD", start);
//	stout << "\t\t\t->" << VEQ(ground_rbm) << "+-" << standard_dev << EL;
//}
//
//template<typename _type, typename _hamtype>
//inline void rbm_ui::ui<_type, _hamtype>::make_mc_classical()
//{
//	auto start = std::chrono::high_resolution_clock::now();
//	stouts("STARTING THE SIMULATION FOR MINIMIZING CONFIGURATION SEEK AND USING: " + VEQ(thread_num), start);
//	printSeparated(stout, ',', 5, true, VEQ(mcSteps), VEQ(n_blocks), VEQ(n_therm), VEQ(block_size));
//
//	auto ran = randomGen();
//	// make the lattice
//	this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
//	auto Ns = this->lat->get_Ns();
//	this->nvisible = Ns;
//	this->nhidden = this->layer_mult * this->nvisible;
//	std::string lat_info = this->lat->get_info();
//	this->av_op = avOperators(Lx, Ly, Lz, Ns, this->lat->get_type());
//	stout << "->" << lat_info << EL;
//
//	this->positions = v_1d<int>(Ns);
//	// use all the positions for the lattice sites
//	std::iota(this->positions.begin(), this->positions.end(), 0);
//
//	// use no disorder at classical - quantum interaction
//	this->J0_dot = 0;
//	// set angles (let's set only the z'th direction (set all phis to 0)
//	this->phis = arma::vec(Ns, arma::fill::zeros);
//	// set thetas to 1 corresponding to all classical spins up
//	this->thetas = arma::vec(Ns, arma::fill::zeros);
//
//	v_1d<double> e_ferro = {};
//	v_1d<double> e_ferro_ed = {};
//	v_1d<double> e_aferro = {};
//	v_1d<double> e_aferro_ed = {};
//	double step = 0.1;
//
//	for (int ferro = 0; ferro <= 1; ferro++) {
//		stout << "\t\t->DOING " << (bool(ferro) ? "ferromagnet" : "antiferromagnet") << EL;
//		// ------------------------ FERROMAGNETIC CLASSICAL SPINS ------------------------
//		if (bool(ferro))
//			for (int i = 0; i < thetas.size(); i++) {
//				this->thetas(i) = 0.0;
//			}
//		else
//		{
//			for (int y = 0; y < this->lat->get_Ly(); y++) {
//				bool y_start = y % 2 == 0 ? 0 : 1;
//				for (int x = 0; x < this->lat->get_Lx(); x++) {
//					int n = x + this->lat->get_Lx() * y;
//					this->thetas(n) = x % 2 == 0 ? int(y_start) : int(!y_start);
//				}
//			}
//		}
//		//this->thetas.print("thetas=");
//		this->delta = 1.0;
//
//		vec sin_phis = sin(this->phis * TWOPI);
//		vec cos_phis = cos(this->phis * TWOPI);
//		vec sin_thetas = sin(this->thetas * PI);
//		vec cos_thetas_rbm = cos(this->thetas * PI);
//		vec cos_thetas_ed = cos(this->thetas * PI);
//		uint threads = this->thread_num / 2 < 1 ? 1 : uint(this->thread_num / 2);
//		cos_thetas_rbm.print(VEQ(ferro));
//		//#pragma omp parallel for num_threads(threads)
//		for (int j0 = 0; j0 <= 40; j0++) {
//			v_1d<double> Jd = { 0.0, 0.0, -2.0 + j0 * step };
//			stout << "\t-> doing " << VEQ(Jd[2]) << EL;
//			// create Hamiltonians
//			std::shared_ptr<Heisenberg_dots<double>> hamiltonian_rbm = std::make_shared<Heisenberg_dots<double>>(J, 0.0, 0.0, 0.0, 0.0, 0.0, delta, lat, positions, Jd, 0.0, this->J_dot_dot);
//			std::shared_ptr<Heisenberg_dots<double>> hamiltonian_ed = std::make_shared<Heisenberg_dots<double>>(J, 0.0, 0.0, 0.0, 0.0, 0.0, delta, lat, positions, Jd, 0.0, this->J_dot_dot);
//			hamiltonian_rbm->set_angles(sin_phis, sin_thetas, cos_phis, cos_thetas_rbm);
//			hamiltonian_ed->set_angles(sin_phis, sin_thetas, cos_phis, cos_thetas_ed);
//			auto model_info = hamiltonian_rbm->get_info();
//			stout << "\t-> " << VEQ(model_info) << EL;
//
//			// rbm stuff
//			auto ph = std::make_unique<rbmState<_type, double>>(nhidden, nvisible, hamiltonian_rbm, lr, batch, this->thread_num);
//			auto rbm_info = ph->get_info();
//			stout << "\t\t->" << VEQ(rbm_info) << EL;
//			string order = string(ferro ? "ferro" : "aferro") + ",Lx=" + STR(this->lat->get_Lx()) + ",Ly=" + STR(this->lat->get_Ly()) + ",bc=" + STR(_BC) + ",d=" + STR(this->lat->get_Dim());
//			string dir = this->saving_dir + kPS + order + kPS;
//			fs::create_directories(dir);
//			dir = dir + rbm_info + kPS;
//			fs::create_directories(dir);
//			dir = dir + model_info + kPS;
//			fs::create_directories(dir);
//
//			string dir_ed = this->saving_dir + kPS + order + kPS;
//			fs::create_directories(dir_ed);
//			dir_ed = dir_ed + "ed" + kPS;
//			fs::create_directories(dir_ed);
//			dir_ed = dir_ed + model_info + kPS;
//			fs::create_directories(dir_ed);
//
//			std::ofstream fileSave;
//
//			// ------------------- calculator rbm -------------------
//			// monte carlo for energy
//			auto energies = ph->mcSampling(this->mcSteps, n_blocks, n_therm, block_size, n_flips, dir);
//			auto energies_tail = energies.tail(block_size);
//			double ground_rbm = std::real(arma::mean(energies_tail));
//
//			// ------------------- calculator ed -------------------
//			double ground_ed = 0.0;
//			avOperators av_operator(Lx, Ly, Lz, Ns, this->lat->get_type());
//			if (Ns <= 14) {
//				calculate_ed<double>(ground_ed, ground_rbm, hamiltonian_ed);
//#ifdef PLOT
//				plt::axhline(ground_ed);
//				plt::annotate(VEQ(ground_ed) + ",\n" + VEQ(ground_rbm), mcSteps / 3, (ground_rbm) / 2);
//				PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", hamiltonian_rbm->get_info() + "\nrbm:" + ph->get_info());
//				SAVEFIG(dir + "energy" + ".png", true);
//#endif
//				Col<_hamtype> eigvec = hamiltonian_ed->get_eigenState(0);
//				Operators<_hamtype> op(this->lat);
//				op.calculate_operators(eigvec, av_operator, true);
//				// --------------------- compare sigma_z ---------------------
//
//				// S_z at each site
//				std::string filename = dir_ed + "_sz_site";
//				openFile(fileSave, filename + ".dat", ios::out);
//				print_vector_1d(fileSave, this->av_op.s_z_i);
//				fileSave.close();
//				PLOT_V1D(this->av_op.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + model_info + "\n");
//				SAVEFIG(filename + ".png", false);
//
//				// S_z correlations
//				filename = dir_ed + "_sz_corr";
//				openFile(fileSave, filename + ".dat", ios::out);
//				print_mat(fileSave, this->av_op.s_z_cor);
//				fileSave.close();
//			}
//			else {
//				PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", hamiltonian_rbm->get_info() + "\nrbm:" + ph->get_info());
//				SAVEFIG(dir + "energy" + ".png", true);
//			}
//
//			// ------------------- sampling rbm -------------------
//			ph->avSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
//			av_operator.reset();
//			av_operator = ph->get_op_av();
//
//			auto fileRbmEn_name = dir + "energies";
//			std::ofstream fileRbmEn;
//			openFile(fileRbmEn, fileRbmEn_name + ".dat", ios::out);
//			for (auto i = 0; i < energies.size(); i++)
//				printSeparatedP(fileRbmEn, '\t', 8, true, 5, i, real(energies(i)));
//			fileRbmEn.close();
//
//			// other observables
//
//			string filename = "";
//			// --------------------- compare sigma_z
//
//			// S_z at each site
//			filename = dir + "_sz_site";
//			openFile(fileSave, filename + ".dat", ios::out);
//			print_vector_1d(fileSave, this->av_op.s_z_i);
//			fileSave.close();
//			PLOT_V1D(this->av_op.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + model_info + "\n");
//			SAVEFIG(filename + ".png", false);
//
//			// S_z correlations
//			filename = dir + "_sz_corr";
//			openFile(fileSave, filename + ".dat", ios::out);
//			print_mat(fileSave, this->av_op.s_z_cor);
//			fileSave.close();
//
//			if (bool(ferro)) {
//				e_ferro.push_back(ground_rbm);
//				e_ferro_ed.push_back(ground_ed);
//			}
//			else {
//				e_aferro.push_back(ground_rbm);
//				e_aferro_ed.push_back(ground_ed);
//			}
//		}
//	}
//	std::ofstream file;
//	openFile(file, this->saving_dir + lat_info + ".dat", ios::out | ios::app);
//	for (int i = 0; i <= 40; i++) {
//		this->J0 = -2.0 + i * step;
//		printSeparatedP(file, '\t', 17, true, 13, this->J, this->J0, e_ferro[i], e_aferro[i], e_ferro_ed[i], e_aferro_ed[i]);
//	}
//	file.close();
//}
//
//template<typename _type, typename _hamtype>
//void rbm_ui::ui<_type, _hamtype>::make_mc_classical_angles(double Jdot)
//{
//	auto start = std::chrono::high_resolution_clock::now();
//	stouts("STARTING THE SIMULATION FOR MINIMIZING CONFIGURATION SEEK AND USING: " + VEQ(thread_num), start);
//	printSeparated(stout, ',', 5, true, VEQ(mcSteps), VEQ(n_blocks), VEQ(n_therm), VEQ(block_size));
//
//	vec Jd(3, arma::fill::ones);
//	Jd = Jd * Jdot;
//
//	// create files
//	std::ofstream file, log;
//	auto ran = randomGen();
//
//	// make the lattice
//	auto lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
//	const auto Ns = lat->get_Ns();
//	auto nvisible = Ns;
//	auto nhidden = this->layer_mult * nvisible;
//	// use all the positions for the lattice sites
//	auto positions = v_1d<int>(Ns);
//	std::iota(positions.begin(), positions.end(), 0);
//	std::string lat_info = lat->get_info();
//
//	// initialize the angles in x-dimension
//	vec phis = arma::vec(Ns, arma::fill::zeros);
//	// set thetas to pi/2 (remember that this is multiplied in the class)
//	vec thetas = arma::vec(Ns, arma::fill::ones) / 2.0;
//
//	// set the allowed angles
//	v_1d<double> allowed_angles_x(int(lat->get_Lx() / 2.0) + 1);
//	v_1d<double> allowed_angles_y(int(lat->get_Ly() / 2.0) + 1);
//	for (int i = 0; i < allowed_angles_x.size(); i++)
//		allowed_angles_x[i] = (i) / double(Ns);
//	for (int i = 0; i < allowed_angles_y.size(); i++)
//		allowed_angles_y[i] = (i) / double(Ns);
//
//
//	// open file for saving the energies and the angles
//	auto hamiltonian_rbm = std::make_shared<Heisenberg_dots<cpx>>(J, 0.0, 0.0, 0.0, 0.0, 0.0, delta, lat, positions, Jd, 0.0, this->J_dot_dot);
//	auto model_info = hamiltonian_rbm->get_info();
//	
//	// rbm stuff
//	unique_ptr<rbmState<_type, cpx>> ph = std::make_unique<rbmState<_type, cpx>>(nhidden, nvisible, hamiltonian_rbm, lr, batch, 1);
//	auto rbm_info = ph->get_info();
//
//	std::string dir = this->saving_dir + kPS + lat_info + rbm_info + kPS;
//	fs::create_directories(dir);
//	dir = dir + rbm_info + kPS;
//	fs::create_directories(dir);
//
//	openFile(log, dir + "log_" + model_info + ".txt");
//	log << "->" << lat_info << EL;
//	log << "\t->" << Jd.t() << EL;
//	log << "\t\t->" << VEQ(rbm_info) << EL;
//	
//	openFile(file, dir + "energies_log_" + model_info + ".dat", ios::app);
//
//	// iterate through all angles
//	std::string dir_start = dir;
//
//	for (int iter = 0; iter < allowed_angles_x.size(); iter++) {
//		double angle = allowed_angles_x[iter];
//		for (int iter2 = 0; iter2 < allowed_angles_y.size(); iter2++) {
//			if (lat->get_Lx() == lat->get_Ly() && iter2 < iter) continue;
//			double angle2 = allowed_angles_y[iter2];
//			dir = dir_start;
//			log << "\t\t\t->DOING " << VEQ(angle) << "," << VEQ(angle2);
//			log << "\n\t\t\t\t->a_x" << STR(iter) + "/" + STR(Ns) << "PI";
//			log << "\n\t\t\t\t->a_y" << STR(iter2) + "/" + STR(Ns) << "PI" << EL << EL;
//
//			// set the angles accordingly
//			for (int site = 0; site < Ns; site++) {
//				auto x = lat->get_coordinates(site, 0);
//				auto y = lat->get_coordinates(site, 1);
//				phis(site) = x * angle + y * angle2;
//			}
//			vec sin_phis = sin(phis * TWOPI);
//			vec cos_phis = cos(phis * TWOPI);
//			vec sin_thetas = sin(thetas * PI);
//			vec cos_thetas = cos(thetas * PI);
//			log << "\t\t\t\tsin_phi: " << sin_phis.t();
//			log << "\t\t\t\tcos_phi: " << cos_phis.t();
//			log << "\t\t\t\tsin_theta: " << sin_thetas.t();
//			log << "\t\t\t\tcos_theta: " << cos_thetas.t() << EL;
//
//			// create Hamiltonians
//			auto hamiltonian_ed = std::make_shared<Heisenberg_dots<cpx>>(J, 0.0, 0.0, 0.0, 0.0, 0.0, delta, lat, positions, Jd, 0.0, this->J_dot_dot);
//			hamiltonian_ed->set_angles(sin_phis, sin_thetas, cos_phis, cos_thetas);
//			hamiltonian_rbm->set_angles(sin_phis, sin_thetas, cos_phis, cos_thetas);
//			model_info = hamiltonian_rbm->get_info();
//			ph->init();
//
//			log << "\t\t\t\t-> " << VEQ(model_info) << EL;
//
//			std::string angle_str = "ax=" + STR(iter) + "," + "ay=" + STR(iter2);
//			dir = dir + angle_str + kPS;
//			fs::create_directories(dir);
//
//			std::string dir_ed = dir + "ed" + kPS;
//			fs::create_directories(dir_ed);
//
//			std::ofstream fileSave;
//			// ------------------- calculator rbm -------------------
//			// monte carlo for energy
//			auto energies = ph->mcSampling(this->mcSteps, n_blocks, n_therm, block_size, n_flips, dir);
//			auto energies_tail = energies.tail(block_size);
//			double max_rbm = arma::max(arma::real(energies));
//			double ground_rbm = std::real(arma::mean(energies_tail));
//			double rbm_difference = max_rbm - ground_rbm;
//			double var_rbm = std::real(arma::var(energies_tail));
//
//			// save the log of energies
//
//			// ------------------- calculator ed -------------------
//			double ground_ed = -1.0;
//			avOperators av_operator(Lx, Ly, Lz, Ns, lat->get_type());
//#pragma omp critical
//			if (Ns <= 14) {
//				auto relative_err = calculate_ed<cpx>(ground_ed, ground_rbm, hamiltonian_ed);
//#ifdef PLOT
//				plt::axhline(ground_ed);
//				plt::annotate(VEQP(ground_ed, 6) + ",\n" + VEQP(ground_rbm, 6) + ",\n" + VEQP(relative_error, 5), 0, max_rbm - (rbm_difference / 1.2));
//				PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", hamiltonian_rbm->get_info() + "\nrbm:" + ph->get_info());
//				SAVEFIG(dir + "en_" + model_info + ".png", false);
//#endif
//				Col<cpx> eigvec = hamiltonian_ed->get_eigenState(0);
//				Operators<cpx> op(lat);
//				op.calculate_operators(eigvec, av_operator, true);
//				// --------------------- compare sigma_z ---------------------
//
//				// S_z at each site
//				std::string filename = dir_ed + "_sz_" + model_info;
//				av_operator.s_z_i.save(arma::hdf5_name(filename + ".h5", "sz_site", arma::hdf5_opts::append));
//				PLOT_V1D(av_operator.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + model_info + "\n");
//				SAVEFIG(dir + "_sz_site.png", false);
//
//				// S_z correlations
//				av_operator.s_z_cor.save(arma::hdf5_name(filename + ".h5", "sz_corr", arma::hdf5_opts::append));
//			}
//			else {
//				PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", hamiltonian_rbm->get_info() + "\nrbm:" + ph->get_info());
//				SAVEFIG(dir + "energy" + ".png", true);
//			}
//				// ------------------- sampling rbm -------------------
//				ph->avSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
//#pragma omp critical
//			{
//				printSeparated(file, '\t', 30, true, model_info, angle_str, ground_rbm, var_rbm, ground_ed);
//				av_operator.reset();
//				av_operator = ph->get_op_av();
//
//				auto fileRbmEn_name = dir + "en_" + model_info;
//				Col<double> en_real = arma::real(energies);
//				en_real.save(arma::hdf5_name(fileRbmEn_name + ".h5", "energy"));
//
//				// other observables
//				string filename = "";
//				// --------------------- compare sigma_z
//
//				// S_z at each site
//				filename = dir + "_sz_" + model_info;
//				av_operator.s_z_i.save(arma::hdf5_name(filename + ".h5", "sz_site", arma::hdf5_opts::append));
//				PLOT_V1D(av_operator.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + model_info + "\n");
//				SAVEFIG(filename + ".png", false);
//
//				// S_z correlations
//				av_operator.s_z_cor.save(arma::hdf5_name(filename + ".h5", "sz_cor", arma::hdf5_opts::append));
//			}
//		}
//	}
//
//	file.close();
//	log.close();
//}
//
//template<typename _type, typename _hamtype>
//void rbm_ui::ui<_type, _hamtype>::make_mc_angles_sweep()
//{
//	// set default parameters
//	// use no disorder at classical - quantum interaction
//	this->g = 0.0;
//	this->h = 0.0;
//	this->J_dot_dot = 0.0;
//	this->delta = 1.0;
//
//	double Jd_start = -2.0;
//	double Jd_end = 2.0;
//	double Jd_step = 0.1;
//	int Jd_num = abs(Jd_end - Jd_start) / Jd_step;
//
//#pragma omp parallel for num_threads(this->thread_num)
//	for (int i = 0; i <= Jd_num; i++)
//	{
//		auto Jd = Jd_start + i * Jd_step;
//		this->make_mc_classical_angles(Jd);
//	}
//}
//
///*
//* @brief make simulation for Kitaev Hamiltonian
//*/
//template<typename _type, typename _hamtype>
//void rbm_ui::ui<_type, _hamtype>::make_mc_kitaev(t_3d<double> K)
//{
//
//	// create files
//	std::ofstream file, log;
//	auto ran = randomGen();
//
//	// make the lattice
//	auto lat = std::make_shared<HexagonalLattice>(Lx, Ly, Lz, dim, _BC);
//	std::string lat_info = lat->get_info();
//	const auto Ns = lat->get_Ns();
//	auto nvisible = Ns;
//	auto nhidden = this->layer_mult * nvisible;
//
//	// open file for saving the energies and the angles
//	auto hamiltonian_rbm = std::make_shared<Heisenberg_kitaev<double>>(J, 0.0, 0.0, 0.0, h, 0.0, this->delta, K, 0.0, lat);
//	auto model_info = hamiltonian_rbm->get_info();
//
//	// rbm stuff
//	unique_ptr<rbmState<_type, double>> ph = std::make_unique<rbmState<_type, double>>(nhidden, nvisible, hamiltonian_rbm, lr, batch, 1);
//	auto rbm_info = ph->get_info();
//
//	std::string dir = this->saving_dir + kPS + lat_info + kPS;
//	fs::create_directories(dir);
//	dir = dir + rbm_info + kPS;
//	fs::create_directories(dir);
//	std::string dir_ed = dir + "ed" + kPS;
//	fs::create_directories(dir_ed);
//
//	openFile(log, dir + "log_" + ".txt", ios::app);
//	log << "->" << lat_info << EL;
//	log << "\t\t->" << VEQ(rbm_info) << EL;
//	log << "\t\t\t->" << VEQ(model_info) << EL;
//
//	openFile(file, dir + "energies_log_" + ".dat", ios::app);
//
//	// ------------------------------------------------------------------- CALCULATORS ------------------------------------------------------------------------------
//	// ------------------- calculator rbm -------------------
//	// monte carlo for energy
//	auto energies = ph->mcSampling(this->mcSteps, n_blocks, n_therm, block_size, n_flips, dir);
//	auto energies_tail = energies.tail(block_size);
//	double max_rbm = arma::max(arma::real(energies));
//	double ground_rbm = std::real(arma::mean(energies_tail));
//	double rbm_difference = max_rbm - ground_rbm;
//	double var_rbm = std::real(arma::var(energies_tail));
//
//	// save the log of energies
//
//	// ------------------- calculator ed -------------------
//	double ground_ed = -1.0;
//	double sz_nei_ed = -1.0; 
//	double sx_nei_ed = -1.0;
//	double sy_nei_ed = -1.0;
//	avOperators av_operator(Lx, Ly, Lz, Ns, lat->get_type());
//#pragma omp critical
//	if (Ns <= 16) {
//		auto relative_err = calculate_ed<double>(ground_ed, ground_rbm, hamiltonian_rbm);
//#ifdef PLOT
//		plt::axhline(ground_ed);
//		plt::annotate(VEQP(ground_ed, 6) + ",\n" + VEQP(ground_rbm, 6) + ",\n" + VEQP(relative_error, 5), 0, max_rbm - (rbm_difference / 1.2));
//		PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", hamiltonian_rbm->get_info() + "\nrbm:" + ph->get_info());
//		SAVEFIG(dir + "en_" + model_info + ".png", false);
//#endif
//		Col<double> eigvec = hamiltonian_rbm->get_eigenState(0);
//		Operators<double> op(lat);
//		op.calculate_operators(eigvec, av_operator, true);
//		// --------------------- compare sigma_z ---------------------
//
//		// S_z at each site
//		std::string filename = dir_ed + "_sz_" + model_info;
//		av_operator.s_z_i.save(arma::hdf5_name(filename + ".h5", "sz_site", arma::hdf5_opts::append));
//		PLOT_V1D(av_operator.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + model_info + "\n");
//		SAVEFIG(dir + "_sz_site.png", false);
//
//		// S_z correlations
//		av_operator.s_z_cor.save(arma::hdf5_name(filename + ".h5", "sz_corr", arma::hdf5_opts::append));
//		sz_nei_ed = av_operator.s_z_nei;
//		sy_nei_ed = real(av_operator.s_y_nei);
//		sx_nei_ed = av_operator.s_x_nei;
//	}
//	else {
//		PLOT_V1D(arma::conv_to< v_1d<double> >::from(arma::real(energies)), "#mcstep", "$<E_{est}>$", hamiltonian_rbm->get_info() + "\nrbm:" + ph->get_info());
//		SAVEFIG(dir + "energy" + ".png", true);
//	}
//	// ------------------- sampling rbm -------------------
//	ph->avSampling(mcSteps, n_blocks, n_therm, block_size, n_flips);
//#pragma omp critical
//	{
//		av_operator.reset();
//		av_operator = ph->get_op_av();
//
//		auto fileRbmEn_name = dir + "en_" + model_info;
//		Col<double> en_real = arma::real(energies);
//		en_real.save(arma::hdf5_name(fileRbmEn_name + ".h5", "energy"));
//
//		// other observables
//		string filename = "";
//		// --------------------- compare sigma_z
//
//		// S_z at each site
//		filename = dir + "_sz_" + model_info;
//		av_operator.s_z_i.save(arma::hdf5_name(filename + ".h5", "sz_site", arma::hdf5_opts::append));
//		PLOT_V1D(av_operator.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + model_info + "\n");
//		SAVEFIG(filename + ".png", false);
//
//		// S_z correlations
//		av_operator.s_z_cor.save(arma::hdf5_name(filename + ".h5", "sz_cor", arma::hdf5_opts::append));
//		printSeparated(file, '\t', 30, true, model_info, ground_rbm,
//			var_rbm, ground_ed,
//			av_operator.s_z_nei, sz_nei_ed,
//			std::real(av_operator.s_y_nei), sy_nei_ed,
//			av_operator.s_x_nei, sx_nei_ed);
//	}
//
//}
//
//template<typename _type, typename _hamtype>
//void rbm_ui::ui<_type, _hamtype>::make_mc_kitaev_sweep()
//{
//	double K_start = -2.0;
//	double K_end = 1.0;
//	double K_step = -0.1;
//	int K_num = abs(K_end - K_start) / abs(K_step);
//
//#pragma omp parallel for num_threads(this->thread_num)
//	for (int i = 0; i <= K_num; i++)
//	{
//		auto k = K_end + i * K_step;
//		this->make_mc_kitaev(std::make_tuple(k, k, k));
//	}
//}
//
//// -------------------------------------------------------- HELPERS
//
///*
//*/
//template<typename _type, typename _hamtype>
//inline void rbm_ui::ui<_type, _hamtype>::define_models()
//{
//	// define the lattice
//	switch (this->lattice_type)
//	{
//	case impDef::lattice_types::square:
//		this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
//		break;
//	case impDef::lattice_types::hexagonal:
//		this->lat = std::make_shared<HexagonalLattice>(Lx, Ly, Lz, dim, _BC);
//		break;
//	default:
//		this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
//		break;
//	}
//	auto lat_type = lat->get_type();
//	auto lat_info = lat->get_info();
//	auto Ns = lat->get_Ns();
//	stout << "\t\t-> " << VEQ(lat_info) << EL;
//	//stout << "\t\t-> " << VEQ(lat_type) << EL;
//	// create operator averages
//	this->av_op = avOperators(Lx, Ly, Lz, this->lat->get_Ns(), lat_type);
//
//	// define the hamiltonian
//	//stout << static_cast<int>(this->model_name) << EL;
//	switch (static_cast<int>(this->model_name))
//	{
//	case impDef::ham_types::ising:
//		this->ham = std::make_shared<IsingModel<_hamtype>>(J, J0, g, g0, h, w, lat);
//		break;
//	case impDef::ham_types::heisenberg:
//		this->ham = std::make_shared<Heisenberg<_hamtype>>(J, J0, g, g0, h, w, delta, lat);
//		break;
//	case impDef::ham_types::heisenberg_dots:
//		this->ham = std::make_shared<Heisenberg_dots<_hamtype>>(J, J0, g, g0, h, w, delta, lat, positions, J_dot, J0_dot);
//		ham->set_angles(phis, thetas);
//		break;
//	case impDef::ham_types::kitaev_heisenberg:
//		this->ham = std::make_shared<Heisenberg_kitaev<_hamtype>>(J, J0, g, g0, h, w, delta, make_tuple(Kx, Ky, Kz), K0, lat);
//		break;
//	default:
//		this->ham = std::make_shared<IsingModel<_hamtype>>(J, J0, g, g0, h, w, lat);
//		break;
//	}
//	auto model_info = this->ham->get_info();
//	stout << "\t\t-> " << VEQ(model_info) << EL;
//
//
//	// rbm stuff
//	this->nvisible = Ns;
//	this->nhidden = this->layer_mult * this->nvisible;
//	this->phi = std::make_unique<rbmState<_type, _hamtype>>(nhidden, nvisible, ham, lr, batch, thread_num);
//	auto rbm_info = phi->get_info();
//	stout << "\t\t-> " << VEQ(rbm_info) << EL;
//
//}
//
///*
//* if it is possible to do so we can test the exact diagonalization states for comparison
//*/
//template<typename _type, typename _hamtype>
//inline void rbm_ui::ui<_type, _hamtype>::compare_ed(double ground_rbm)
//{
//	// test ED
//	auto Ns = this->lat->get_Ns();
//
//	if (Ns <= maxed) {
//		this->av_op.reset();
//		auto diag_time = std::chrono::high_resolution_clock::now();
//		stout << "\n\n-> starting ED for:\n\t-> " + ham->get_info() << EL;
//		// define the operators class
//
//		this->ham->hamiltonian();
//		this->ham->diag_h(false);
//
//		Operators<_hamtype> op(this->lat);
//		auto ground_ed = std::real(ham->get_eigenEnergy(0));
//		auto excited_ed = std::real(ham->get_eigenEnergy(1));
//		Col<_hamtype> eigvec = ham->get_eigenState(0);
//
//		// calculate operators for a ground state
//		op.calculate_operators(eigvec, this->av_op, true);
//
//		// save operators
//		this->save_operators(diag_time, "", ground_ed, 0);
//
//		u64 state = 0;
//		auto sz = this->av_op.s_z;
//		auto sx = this->av_op.s_x;
//		auto relative_error = abs(std::real(ground_ed - ground_rbm)) / abs(ground_ed) * 100.;
//
//		stouts("\t\t-> finished ED", diag_time);
//		stout << "\t\t\t->" << VEQP(ground_ed, 8) << EL;
//		stout << "\t\t\t->" << VEQP(ground_rbm, 8) << EL;
//		stout << "\t\t\t->" << VEQP(relative_error, 4) << "%" << EL;
//		stout << "------------------------------------------------------------------------" << EL;
//		stout << "GROUND STATE ED ENERGY: " << VEQP(ground_ed, 4) << EL;
//		stout << "1ST EXCITED STATE ED ENERGY: " << VEQP(excited_ed, 4) << EL;
//		stout << "GROUND STATE ED SIGMA_X EXTENSIVE: " << VEQP(sx, 4) << EL;
//		stout << "GROUND STATE ED SIGMA_Z EXTENSIVE: " << VEQP(sz, 4) << EL;
//		stout << "\n------------------------------------------------------------------------\n|Psi>=:" << EL;
//		stout << "\t->Ground(" + STRP(ground_ed, 3) + "):" << EL;
//		SpinHamiltonian<_hamtype>::print_state_pretty(ham->get_eigenState(0), Ns, 0.08);
//		stout << "\t->Excited(" + STRP(excited_ed, 3) + "):" << EL;
//		SpinHamiltonian<_hamtype>::print_state_pretty(ham->get_eigenState(1), Ns, 0.08);
//		stout << "------------------------------------------------------------------------" << EL;
//#ifdef PLOT
//		plt::axhline(ground_ed);
//		plt::axhline(excited_ed);
//		plt::axhline(ham->get_eigenEnergy(2));
//		plt::annotate(VEQ(ground_ed) + ",\n" + VEQ(excited_ed) + ",\n" + VEQ(ground_rbm) + ",\n" + VEQ(relative_error) + "%", mcSteps / 3, (ground_rbm) / 2);
//#endif
//	}
//}
//
//// -------------------------------- OPERATORS -----------------------------------------
//
//template<typename _type, typename _hamtype>
//inline void rbm_ui::ui<_type, _hamtype>::save_operators(clk::time_point start, std::string name, double energy, double energy_error)
//{
//	std::ofstream fileSave;
//	std::fstream log;
//
//	std::string dir = this->saving_dir + kPS + this->lat->get_type() + kPS + this->ham->get_info() + kPS;
//	if (name != "") dir = dir + name + kPS;
//	fs::create_directories(dir);
//
//	std::string filename = "";
//	auto Ns = this->lat->get_Ns();
//	// --------------------- compare sigma_z ---------------------
//
//	// S_z at each site
//	filename = dir + "_sz_site";
//	openFile(fileSave, filename + ".dat", ios::out);
//	print_vector_1d(fileSave, this->av_op.s_z_i);
//	fileSave.close();
//	PLOT_V1D(this->av_op.s_z_i, "lat_site", "$S^z_i$", "$S^z_i$\n" + this->ham->get_info() + "\n" + name);
//	SAVEFIG(filename + ".png", false);
//
//	// S_z correlations
//	filename = dir + "_sz_corr";
//	openFile(fileSave, filename + ".dat", ios::out);
//	print_mat(fileSave, this->av_op.s_z_cor);
//	fileSave.close();
//
//	// --------------------- compare sigma_y ---------------------
//	// S_y at each site
//	//filename = dir + "_sy_site";
//	//openFile(fileSave, filename + ".dat", ios::out);
//	//print_vector_1d(fileSave, this->av_op.s_x_i);
//	//fileSave.close();
//	//PLOT_V1D(this->av_op.s_x_i, "lat_site", "$S^x_i$", "$S^x_i$\n" + ham->get_info() + "\n" + name);
//	//SAVEFIG(filename + ".png", false);
//	//
//	//// S_z correlations
//	//filename = dir + "_sx_corr_";
//	//openFile(fileSave, filename + ".dat", ios::out);
//	//print_mat(fileSave, this->av_op.s_x_cor);
//	//fileSave.close();
//
//	// --------------------- compare sigma_x ---------------------
//	// S_z at each site
//	filename = dir + "_sx_site";
//	openFile(fileSave, filename + ".dat", ios::out);
//	print_vector_1d(fileSave, this->av_op.s_x_i);
//	fileSave.close();
//	PLOT_V1D(this->av_op.s_x_i, "lat_site", "$S^x_i$", "$S^x_i$\n" + ham->get_info() + "\n" + name);
//	SAVEFIG(filename + ".png", false);
//
//	// S_z correlations
//	filename = dir + "_sx_corr_";
//	openFile(fileSave, filename + ".dat", ios::out);
//	print_mat(fileSave, this->av_op.s_x_cor);
//	fileSave.close();
//
//	// --------------------- entropy ----------------------
//	if (Ns <= maxed) {
//		filename = dir + "_ent_entro";
//		openFile(fileSave, filename + ".dat", ios::out);
//		print_vector_1d(fileSave, this->av_op.ent_entro);
//		fileSave.close();
//		PLOT_V1D(this->av_op.ent_entro, "bond_cut", "$S_0(L)$", "Entanglement entropy\n" + ham->get_info() + "\n" + name);
//		SAVEFIG(filename + ".png", false);
//	}
//
//	// --------------------- neigbors ----------------------
//	filename = dir + "_neighbor_corr_";
//	openFile(fileSave, filename + ".dat", ios::out);
//	printSeparatedP(fileSave, '\t', 14, true, 5, "<S^x_iS^x_(i+m)>", "<S^y_iS^y_(i+m)>", "<S^z_iS^z_(i+m)>");
//	printSeparatedP(fileSave, '\t', 14, true, 5, real(this->av_op.s_x_nei), real(this->av_op.s_y_nei), real(this->av_op.s_z_nei));
//	fileSave.close();
//
//	// --------------------- save log ---------------------	// save the log file and append columns if it is empty
//	string logname = dir + "log.dat";
//#pragma omp single
//	{
//		openFile(log, logname, ios::app);
//		log.seekg(0, std::ios::end);
//		if (log.tellg() == 0) {
//			log.clear();
//			log.seekg(0, std::ios::beg);
//			printSeparated(log, '\t', 15, true, "lattice_type", "Lx", \
//				"Ly", "Lz", "En", "dEn", "Sz", "Sx", "time taken");
//		}
//		log.close();
//	}
//	openFile(log, logname, ios::app);
//	printSeparatedP(log, '\t', 15, true, 4, this->lat->get_type(), this->lat->get_Lx(), this->lat->get_Ly(), this->lat->get_Lz(), \
//		energy, energy_error, av_op.s_z, std::real(av_op.s_x), tim_s(start));
//	log.close();
//};
//
//// -------------------------------- HAMILTONIAN WITH SYMMETRIES --------------------------------
//
///*
//* @brief calculates properties of the Hamiltonian with symmetries whenever the sector is real
//*/
//template<typename _type, typename _hamtype>
//inline void rbm_ui::ui<_type, _hamtype>::symmetries_double(clk::time_point start)
//{
//	stout << "->using real" << EL;
//	if (sym) {
//		if (this->model_name == 0)
//			this->ham_d = std::make_shared<ising_sym::IsingModelSym<double>>(J, g, h, lat, k_sym, p_sym, x_sym, this->thread_num);
//		else
//			this->ham_d = std::make_shared<xyz_sym::XYZSym<double>>(lat, this->J, this->Jb, this->g, this->h,
//				this->delta, this->Delta_b, this->eta_a, this->eta_b,
//				k_sym, p_sym, x_sym, su2, this->thread_num);
//		this->ham_d->hamiltonian();
//	}
//	else {
//		if (this->model_name == 0)
//			this->ham_d = std::make_shared<IsingModel<double>>(J, J0, g, g0, h, w, lat);
//		else
//			this->ham_d = std::make_shared<XYZ<double>>(lat, J, Jb, g, h, delta, Delta_b, eta_a, eta_b, this->p_break);
//		this->ham_d->hamiltonian();
//	}
//	const u64 N = this->ham_d->get_hilbert_size();
//	bool use_s_i = false;
//	u64 state_num = N;
//
//	stouts("\t->finished buiding Hamiltonian", start);
//	stout << "\t->" << this->ham_d->get_info() << EL;
//	if (N < ULLPOW(8)){
//		stout << "\t->" << "using standard diagonalization" << EL;
//		this->ham_d->diag_h(false);
//	}
//	else
//	{
//		stout << "\t->" << "using S&I" << EL;
//		use_s_i = true;
//		state_num = 100;
//		this->ham_d->diag_h(false, state_num, 0, 1000, 1e-5, "sa");
//	}
//	stouts("\t->finished diagonalizing Hamiltonian", start);
//
//	std::string name = "spectrum_num=" + STR(N);
//	stout << "->middle_spectrum_size : " << name << ".\n\t\t->Taking num states: " << state_num << EL;
//
//	// create the directories
//	std::string dir = this->saving_dir + kPS;
//	fs::create_directories(dir);
//	std::ofstream file;
//	std::ofstream fileAv;
//	std::string model_info = this->ham_d->get_info();
//
//	// save energies to check
//	if (N < ULLPOW(16)) {
//		openFile(file, dir + "energies" + model_info + "," + name + ".dat");
//		for (u64 i = 0; i < state_num; i++)
//			file << this->ham_d->get_eigenEnergy(i) << EL;
//		file.close();
//	}
//	
//	// save energies
//	std::string filename = dir + model_info + "," + name;
//	const vec& energies = this->ham_d->get_eigenvalues();
//	energies.save(arma::hdf5_name(filename + ".h5", "energy", arma::hdf5_opts::append));
//	this->ham_d->clear_energies();
//	this->ham_d->clear_hamiltonian();
//
//	// calculate the reduced density matrices
//	Operators<double> op(this->lat);
//
//	// iterate through bond cut
//	int bond_num = this->lat->get_Ns() / 2;
//	arma::mat entropies(bond_num, state_num, arma::fill::zeros);
//
//	// check the symmetry rotation
//	auto global = this->ham_d->get_global_sym();
//	v_1d<u64> full_map = global.su ? this->ham_d->get_mapping_full() : v_1d<u64>();
//	SpMat<double> symmetryRotationMat = this->ham_d->symmetryRotationMat(full_map);
//
//	// set less number of bonds for quicker calculations
//	v_1d<uint> bonds = { static_cast<uint>(bond_num) };
//#pragma omp parallel for num_threads(this->thread_num)
//	for (u64 idx = 0; idx < state_num; idx++) {
//		Col<double> state = this->ham_d->get_eigenState(idx);
//		if (this->sym)
//			state = symmetryRotationMat * state;
//		for (auto i : bonds) {
//			// iterate through the state
//			//auto entro = op.entanglement_entropy(state, i, full_map);
//			auto entro = op.schmidt_decomposition(state, i, full_map);
//			entropies(i - 1, idx) = entro;
//		}
//		if (state_num > 10 && idx % int(state_num / 10) == 0) {
//			stout << "\t->Done: " << int(idx * 100.0 / state_num) << "%\n";
//			stout << "\t\t->doing : " << VEQ(idx) << "\t" << VEQ(model_info) << EL;
//		}
//	}
//	stout << "\t->" << VEQ(model_info) << "\t : FINISHED ENTROPIES" << EL;
//	
//	// save entropies file
//	entropies.save(arma::hdf5_name(filename + ".h5", "entropy", arma::hdf5_opts::append));
//	if (this->lat->get_Ns() <= 12) 
//		entropies.save(filename + ".dat", arma::arma_ascii);
//
//	if (use_s_i)
//		return;
//
//	// set the average energy index
//	const u64 av_energy_idx = this->ham_d->get_en_av_idx();
//
//	// save states near the mean energy index
//	if (this->lat->get_Ns() == 20) {
//		arma::mat states = this->ham_d->get_eigenvectors().submat(0, av_energy_idx - 50, N-1, av_energy_idx + 50);
//		states.save(arma::hdf5_name(filename + ".h5", "states", arma::hdf5_opts::append));
//	}
//
//	// iterate through fractions
//	v_1d<double> fractions = { 0.25, 0.1, 0.125, 0.5, 50, 200, 500 };
//	v_1d<double> mean_frac(fractions.size());
//	for (int iter = 0; iter < fractions.size(); iter++) {
//		double frac = fractions[iter];
//		u64 spectrum_num = frac <= 1.0 ? frac * N : static_cast<u64>(frac);
//		name = "spectrum_num=" + STR(N);
//		// define the window to calculate the entropy
//		if (long(av_energy_idx) - long(spectrum_num / 2) < 0 || av_energy_idx + u64(spectrum_num / 2) >= N)
//			continue;
//
//		auto subview = entropies.submat(0, av_energy_idx - long(spectrum_num / 2), bond_num - 1, av_energy_idx + u64(spectrum_num / 2));
//		vec bond_mean(bond_num, arma::fill::zeros);
//		for (int i = 1; i <= bond_num; i++)
//			bond_mean(i - 1) = arma::mean(subview.row(i - 1));
//		std::string filename = dir + "av_" + model_info + "," + name;
//		bond_mean.save(arma::hdf5_name(filename + ".h5", STRP(frac, 3), arma::hdf5_opts::append));
//
//		mean_frac[iter] = bond_mean(bond_num - 1);
//	}
//	
//	// save maxima
//	openFile(fileAv, this->saving_dir + kPS + "entropies_log" + ".dat", ios::out | ios::app);
//	vec maxima = arma::max(entropies, 1);
//	printSeparatedP(fileAv, '\t', 18, false, 12, this->ham_d->inf({}, "_", 4), maxima(bond_num - 1));
//	for(auto mean : mean_frac)
//		printSeparatedP(fileAv, '\t', 18, false, 12, mean);
//	printSeparatedP(fileAv, '\t', 18, true, 12, N);
//	fileAv.close();
//}
//
///*
//* @brief calculates properties of the Hamiltonian with symmetries whenever the sector is complex
//*/
//template<typename _type, typename _hamtype>
//inline void rbm_ui::ui<_type, _hamtype>::symmetries_cpx(clk::time_point start)
//{
//	stout << "->using cpx" << EL;
//	if (sym) {
//		if (this->model_name == 0)
//			this->ham_cpx = std::make_shared<ising_sym::IsingModelSym<cpx>>(J, g, h, lat, k_sym, p_sym, x_sym, this->thread_num);
//		else
//			this->ham_cpx = std::make_shared<xyz_sym::XYZSym<cpx>>(lat, this->J, this->Jb, this->g, this->h,
//				this->delta, this->Delta_b, this->eta_a, this->eta_b,
//				k_sym, p_sym, x_sym, su2, this->thread_num);
//		this->ham_cpx->hamiltonian();
//	}
//	else {
//		if (this->model_name == 0)
//			this->ham_cpx = std::make_shared<IsingModel<cpx>>(J, J0, g, g0, h, w, lat);
//		else
//			this->ham_cpx = std::make_shared<XYZ<cpx>>(lat, J, Jb, g, h, delta, Delta_b, eta_a, eta_b, false);;
//		this->ham_cpx->hamiltonian();
//	}
//	
//	// get Hilbert space info
//	const u64 N = this->ham_cpx->get_hilbert_size();
//	bool use_s_i = false;
//	u64 state_num = N;
//
//	// check whether this is already over our range
//	stouts("\t->finished buiding Hamiltonian", start);
//	stout << "\t->" << this->ham_cpx->get_info() << EL;
//	if (N < ULLPOW(8)) {
//		stout << "\t\t->" << "using standard diagonalization" << EL;
//		this->ham_cpx->diag_h(false);
//	}
//	else
//	{
//		use_s_i = true;
//		state_num = 100;
//		this->ham_cpx->diag_h(false, state_num, 0, 1000, 1e-5, "sr");
//	}
//	stouts("\t->finished diagonalizing Hamiltonian", start);
//
//	std::string name = "spectrum_num=" + STR(N);
//	stout << "->middle_spectrum_size : " << name << ".\n\t\t->Taking num states: " << state_num << EL;
//
//	// create the directories
//	std::string dir = this->saving_dir + kPS;
//	fs::create_directories(dir);
//	std::ofstream file;
//	std::ofstream fileAv;
//	std::string model_info = this->ham_cpx->get_info();
//	// save energies to check
//	if (N < ULLPOW(8)) {
//		openFile(file, dir + "energies" + model_info + "," + name + ".dat");
//		for (u64 i = 0; i < N; i++)
//			file << this->ham_cpx->get_eigenEnergy(i) << EL;
//		file.close();
//	}
//	
//	std::string filename = dir + model_info + "," + name;
//	const vec& energies = this->ham_cpx->get_eigenvalues();
//	energies.save(arma::hdf5_name(filename + ".h5", "energy", arma::hdf5_opts::append));
//	this->ham_cpx->clear_energies();
//	this->ham_cpx->clear_hamiltonian();
//
//	// calculate the reduced density matrices
//	Operators<cpx> op(this->lat);
//
//	// iterate through bond cut
//	int bond_num = this->lat->get_Ns() / 2;
//	arma::mat entropies(bond_num, state_num, arma::fill::zeros);
//
//	// check the symmetry rotation
//	auto global = this->ham_cpx->get_global_sym();
//	v_1d<u64> full_map = global.su ? this->ham_cpx->get_mapping_full() : v_1d<u64>();
//	SpMat<cpx> symmetryRotationMat = this->ham_cpx->symmetryRotationMat(full_map);
//
//	// set less number of bonds for quicker calculations
//	v_1d<uint> bonds = { static_cast<uint>(bond_num) };
//#pragma omp parallel for num_threads(this->thread_num)
//	for (u64 idx = 0; idx < state_num; idx++) {
//		Col<cpx> state = this->ham_cpx->get_eigenState(idx);
//		if (this->sym)	
//			state = symmetryRotationMat * state;
//		for (auto i : bonds) {
//			// iterate through the state
//			//auto entro = op.entanglement_entropy(state, i, full_map);
//			auto entro = op.schmidt_decomposition(state, i, full_map);
//			entropies(i - 1, idx) = entro;
//		}
//		if (state_num > 10 && idx % int(state_num / 10) == 0) {
//			stout << "\t->Done: " << int(idx * 100.0 / state_num) << "%\n";
//			stout << "\t\t->doing : " << VEQ(idx) << "\t" << VEQ(model_info) << EL;
//		}
//	}
//	stout << "\t->" << VEQ(model_info) << "\t : FINISHED ENTROPIES" << EL;
//
//	// save the entropies
//	entropies.save(arma::hdf5_name(filename + ".h5", "entropy", arma::hdf5_opts::append));
//	if (this->lat->get_Ns() <= 12)
//		entropies.save(filename + ".dat", arma::arma_ascii);
//
//	// end when using s&i
//	if (use_s_i)
//		return;
//
//	// set the average energy index
//	const u64 av_energy_idx = this->ham_cpx->get_en_av_idx();
//
//	// save states near the mean energy index
//	if (this->lat->get_Ns() == 20) {
//		arma::cx_mat states = this->ham_cpx->get_eigenvectors().submat(0, av_energy_idx - 50, N - 1, av_energy_idx + 50);
//		states.save(arma::hdf5_name(filename + ".h5", "states", arma::hdf5_opts::append));
//	}
//
//
//	// iterate through fractions
//	v_1d<double> fractions = { 0.25, 0.1, 0.125, 0.5, 50, 200, 500 };
//	v_1d<double> mean_frac(fractions.size());
//	for (int iter = 0; iter < fractions.size(); iter++) {
//		double frac = fractions[iter];
//		u64 spectrum_num = frac <= 1.0 ? frac * N : static_cast<u64>(frac);
//		name = "spectrum_num=" + STR(N);
//		// define the window to calculate the entropy
//		if (long(av_energy_idx) - long(spectrum_num / 2) < 0 || av_energy_idx + u64(spectrum_num / 2) >= N)
//			continue;
//
//		auto subview = entropies.submat(0, av_energy_idx - long(spectrum_num / 2), bond_num - 1, av_energy_idx + u64(spectrum_num / 2));
//		vec bond_mean(bond_num, arma::fill::zeros);
//		for (int i = 1; i <= bond_num; i++)
//			bond_mean(i - 1) = arma::mean(subview.row(i - 1));
//		std::string filename = dir + "av_" + model_info + "," + name;
//		bond_mean.save(arma::hdf5_name(filename + ".h5", STRP(frac, 3), arma::hdf5_opts::append));
//
//		mean_frac[iter] = bond_mean(bond_num-1);
//	}
//	// save maxima
//	openFile(fileAv, this->saving_dir + kPS + "entropies_log" + ".dat", ios::out | ios::app);
//	vec maxima = arma::max(entropies, 1);
//	printSeparatedP(fileAv, '\t', 18, false, 12, this->ham_cpx->inf({}, "_", 4), maxima(bond_num - 1));
//	for (auto mean : mean_frac)
//		printSeparatedP(fileAv, '\t', 18, false, 12, mean);
//	printSeparatedP(fileAv, '\t', 18, true, 12, N);
//	fileAv.close();
//}
//
///*
//* @brief make simulation for symmetric case for single symmetry sector and parameters combination
//*/
//template<typename _type, typename _hamtype>
//inline void rbm_ui::ui<_type, _hamtype>::make_simulation_symmetries()
//{
//	auto start = std::chrono::high_resolution_clock::now();
//	stouts("STARTING THE CALCULATIONS FOR QUANTUM HAMILTONIAN: " + VEQ(thread_num), start);
//	stout << "->" << (sym ? "" : "not ") << "including symmetries" << EL;
//
//	this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
//	stout << "->" << this->lat->get_info() << EL;
//	auto Ns = this->lat->get_Ns();
//
//	this->J = 1.0;
//	this->J0 = 0.0;
//	this->w = 0.0;
//	this->g0 = 0.0;
//
//
//	v_1d<int> su2v = {};
//	if (this->eta_a == 0.0 && this->eta_b == 0.0) {
//		su2v.push_back(int(Ns / 2.0));
//	}
//	else {
//		su2v.push_back(-1);
//	}
//
//	// set momenta
//	if (this->lat->get_BC() == 1)
//		this->k_sym = 0;
//
//	// set the fields
//	if (this->h != 0.0)
//		this->x_sym = 1;
//
//
//	if (this->sym) {
//		for (auto su2 : su2v) {
//			//this->su2 = su2;
//			if (this->k_sym == 0 || this->k_sym == this->lat->get_Ns() / 2)
//				this->symmetries_double(start);
//			else
//				this->symmetries_cpx(start);
//		}
//	}
//	else
//		this->symmetries_double(start);
//
//	stouts("FINISHED THE CALCULATIONS FOR QUANTUM ISING HAMILTONIAN: ", start);
//
//}
//
///*
//* @brief sweep symmetries for given parameters in given models 
//*/
//template<typename _type, typename _hamtype>
//void rbm_ui::ui<_type, _hamtype>::make_simulation_symmetries_sweep()
//{
//	auto start = std::chrono::high_resolution_clock::now();
//	stouts("STARTING THE CALCULATIONS FOR QUANTUM HAMILTONIAN: " + VEQ(thread_num), start);
//	stout << "->" << (sym ? "" : "not ") << "including symmetries" << EL;
//
//	this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, _BC);
//	stout << "->" << this->lat->get_info() << EL;
//	int Ns = this->lat->get_Ns();
//
//	// for SU(2) eta_a, eta_b needs to be set to 0.0, hz = 0.0, hx = 0.0, sweep delta_a outside
//	// for no SU(2) eta_a, eta_b need to be set 0.5, delta_b = delta_a needs to be set to 0.9, we sweep hz outside, hx = 0.4
//
//
//	this->J = 1.0;
//	this->J0 = 0.0;
//	//this->g = 0.0;
//
//	this->w = 0.0;
//	this->g0 = 0.0;
//
//
//
//	v_1d<int> su_val = {};
//	bool su = this->eta_a == 0.0 && this->eta_b == 0.0;
//	if (su) {
//		this->h = 0.0;
//		this->g = 0.0;
//		this->Delta_b = 0.3;
//		su_val.push_back(int(Ns / 2.0));
//	}
//	else {
//		this->Delta_b = this->delta;
//		this->g = 0.4;
//		su_val.push_back(-1);
//	}
//
//
//	double J2_min = this->Jb;
//	double J2_step = 0.1;
//	double J2_max = this->Jb + 5 * J2_step;
//	int J2_num = abs(J2_max - J2_min) / J2_step;
//	// over next nearest interaction
//
//	// set momenta
//	v_1d<int> ks(int(Ns/2) + 1);
//	if (this->lat->get_BC() == 0)
//		std::iota(ks.begin(), ks.end(), 0);
//	else
//		ks = { 0 };
//	// set the parities
//	v_1d<int> ps = {};
//	// set the fields
//	v_1d<int> xs = {};
//
//	for (int i = 0; i <= 1; i++) {
//		this->Jb = J2_min + J2_step * i;
//		// initialize hamiltonian
//		if (this->sym) {
//			for (auto su_v : su_val) {
//				this->su2 = su_v;
//				for (auto k : ks) {
//					this->k_sym = k;
//					if (k == 0 || k == int(Ns / 2))
//						ps = { 0, 1 };
//					else
//						ps = { 1 };
//
//					bool su_0 = !su || (su && su_v == Ns / 2);
//					const bool include_sz_flip = su_0 && valueEqualsPrec(this->h, 0.0, 1e-9) && valueEqualsPrec(this->g, 0.0, 1e-9);
//
//					if (include_sz_flip)
//						xs = { 0, 1 };
//					else
//						xs = { 1 };
//
//					for (auto x : xs) {
//						this->x_sym = x;
//						for (auto p : ps)
//						{
//							this->p_sym = p;
//							if (this->k_sym == 0 || this->k_sym == (this->lat->get_Ns() / 2))
//								this->symmetries_double(start);
//							else
//								this->symmetries_cpx(start);
//						}
//					}
//				}
//			}
//		}
//		else
//			this->symmetries_double(start);
//	}
//	stouts("FINISHED THE CALCULATIONS FOR QUANTUM ISING HAMILTONIAN: ", start);
//}
//
///*
//* @brief Test the symmetries for given Hamiltonian. Checks all the possible combinations of symmetries in order to find the energies and compare them.
//*/
//template<typename _type, typename _hamtype>
//void rbm_ui::ui<_type, _hamtype>::make_symmetries_test(int l)
//{
//
//	// -------------------------------------------------------------------- calculate full ed obc
//	this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, 1, 1);
//	const auto Ns = this->lat->get_Ns();
//	stout << "Doing->" << this->lat->get_info() << EL;
//
//	const int La = (l == -1) ? int(Ns / 2) : l;
//
//	std::ofstream file, file2, states;
//	Operators<double> op_ed_obc(this->lat);
//
//	// eta needs to be set, delta_b needs to be set
//	this->J = 1.0;
//	//this->g = 0.0;
//	this->delta = 0.3;
//	// for particle conservation symmetry
//
//	if (this->model_name == 0)
//		this->ham_d = std::make_shared<IsingModel<double>>(J, 0.0, g, 0.0, h, 0.0, lat);
//	else
//		this->ham_d = std::make_shared<XYZ<double>>(lat, this->J, this->Jb, this->g, this->h,
//			this->delta, this->Delta_b, this->eta_a, this->eta_b, false);
//
//	this->ham_d->hamiltonian();
//	file << "\t->" << "finished building: " << this->ham_d->get_info() << EL;
//	this->ham_d->diag_h(false);
//	file << "\t->" << "finished diagonalizing: " << this->ham_d->get_info() << EL;
//
//	// Save the energies for calculating full Hamiltonian
//	std::string dir = this->saving_dir + kPS + this->ham_d->get_info() + kPS;
//	fs::create_directories(dir);
//	openFile(file, dir + "energies.dat");
//	for (u64 i = 0; i < this->ham_d->get_hilbert_size(); i++)
//		file << STRP(this->ham_d->get_eigenEnergy(i), 10) << EL;
//	this->ham_d->get_eigenvalues().save(arma::hdf5_name(dir + "en.h5", "energy", arma::hdf5_opts::append));
//	file << "\t\t->" << "saved energies: " << this->ham_d->get_info() << EL;
//	file.close();
//
//	// save the entropies for the full Hamiltonian
//	arma::mat entropies_mat(La, this->ham_d->get_hilbert_size(), arma::fill::zeros);
//#pragma omp parallel for num_threads(this->thread_num)
//	for (u64 idx = 0; idx < this->ham_d->get_hilbert_size(); idx++) {
//		const Col<double> state = this->ham_d->get_eigenStateFull(idx);
//		for (int i = 1; i <= La; i++) {
//			auto entro = op_ed_obc.entanglement_entropy(state, i);
//			entropies_mat(i - 1, idx) = entro;
//		}
//	}
//
//	// save binary file
//	entropies_mat.save(dir + "entropies.dat", arma::raw_ascii);
//	entropies_mat.save(arma::hdf5_name(dir + "en.h5", "energy", arma::hdf5_opts::append));
//
//	arma::mat ens_obc = this->ham_d->get_eigenvalues();
//	// save binary file
//	if (this->lat->get_Ns() < 16) {
//		auto filename = dir + "entropies_obc" + ".dat";
//		entropies_mat.save(filename, arma::arma_ascii);
//	}
//
//	// -------------------------------------------------------------------- calculate full ed pbc --------------------------------------------------------------------
//	this->lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, 1, 0);
//	stout << "Doing->" << this->lat->get_info() << EL;
//
//	if (this->model_name == 0)
//		this->ham_d = std::make_shared<IsingModel<double>>(J, 0.0, g, 0.0, h, 0.0, lat);
//	else
//		this->ham_d = std::make_shared<XYZ<double>>(lat, this->J, this->Jb, this->g, this->h,
//			this->delta, this->Delta_b, this->eta_a, this->eta_b, false);
//
//	this->ham_d->hamiltonian();
//	stout << "\t->" << "finished building: " << this->ham_d->get_info() << EL;
//	this->ham_d->diag_h(false);
//	stout << "\t->" << "finished diagonalizing: " << this->ham_d->get_info() << EL;
//
//	// -------------------------------------------------------------------- calculate sym ed pbc
//
//	v_1d<double> energies_sym = {};
//	v_1d<double> entropies = {};
//	v_1d<std::tuple<int, int, int, int>> sym = {};
//
//	Operators<cpx> op(this->lat);
//	dir = this->saving_dir + kPS + this->ham_d->get_info() + kPS;
//	fs::create_directories(dir);
//	std::string dir_separated = dir + "values" + kPS;
//	fs::create_directories(dir_separated);
//	openFile(file, dir + "log.dat");
//	
//
//	double entro = 0;
//	u64 Nh = pow(2, Ns);
//	u64 state_num = 0;
//
//	// parity
//	v_1d<int> ps = {};
//	// spin flip
//	v_1d<int> sf = {};
//
//
//	// save particle sectors
//	v_1d<int> su_val = {};
//	bool su = this->eta_a == 0.0 && this->eta_b == 0.0;
//	if (su) {
//		int start = 0;
//		int end = Ns;
//		for (int i = start; i <= end; i++) {
//			su_val.push_back(i);
//		}
//	}
//	else
//		su_val.push_back(-1);
//	// save the Hamiltonian without symmetries
//	arma::sp_mat H0 = this->ham_d->get_hamiltonian();
//	auto lambda = [this](int k, int p, int x, SpMat<cpx>& U, std::shared_ptr<SpinHamiltonian<cpx>> model, arma::sp_cx_mat& H) {
//		arma::sp_cx_mat Hsym = model->get_hamiltonian();
//		H += U * Hsym * U.t();
//	};
//
//	// create sparse matrix to compare the full hamiltonian
//	arma::sp_cx_mat H(H0.n_rows, H0.n_cols);
//	// go through all symmetries
//	for (auto su_v : su_val) {
//		for (int k = 0; k < Ns; k++) {
//			if (k == 0 || k == int(Ns / 2))
//				ps = { 0, 1 };
//			else
//				ps = { 1 };
//
//			bool su_0 = !su || (su && su_v == Ns / 2);
//			const bool include_sz_flip = su_0 && valueEqualsPrec(this->h, 0.0, 1e-9) && valueEqualsPrec(this->g, 0.0, 1e-9);
//
//			if (include_sz_flip)
//				sf = { 0, 1 };
//			else
//				sf = { 1 };
//
//			for (auto x : sf){
//				for (auto p : ps) {
//					if (this->model_name == 0)
//						this->ham_cpx = std::make_shared<ising_sym::IsingModelSym<cpx>>(J, g, h, lat, k, p, 1, this->thread_num);
//					else
//						this->ham_cpx = std::make_shared<xyz_sym::XYZSym<cpx>>(lat, this->J, this->Jb, this->g, this->h,
//							this->delta, this->Delta_b, this->eta_a, this->eta_b,
//							k, p, x, su_v, this->thread_num);
//
//					file << "\t->" << this->ham_cpx->get_info() << EL;
//					if (this->ham_cpx->get_hilbert_size() == 0)
//					{
//						file << "\t\t->EMPTY SECTOR : " << VEQ(k) << "," << VEQ(p) << "," << VEQ(x) << "," << VEQ(su_v) << EL;
//						continue;
//					}
//					this->ham_cpx->hamiltonian();
//					file << "\tDoing: " << VEQ(k) << "," << VEQ(p) << "," << VEQ(x) << "," << VEQ(su_v) << ".\tHilbert size = " << this->ham_cpx->get_hilbert_size() << EL;
//					this->ham_cpx->diag_h(false);
//
//					v_1d<u64> full_map = (this->eta_a == 0.0 && this->eta_b == 0.0) ? this->ham_cpx->get_mapping_full() : v_1d<u64>();
//
//					// create rotation matrix
//					SpMat<cpx> U = this->ham_cpx->symmetryRotationMat(full_map);
//					if (!(this->eta_a == 0.0 && this->eta_b == 0.0))
//						lambda(k, p, x, U, this->ham_cpx, H);
//
//					state_num += this->ham_cpx->get_hilbert_size();
//					vec entro_inner(this->ham_cpx->get_hilbert_size());
//					for (u64 i = 0; i < this->ham_cpx->get_hilbert_size(); i++) {
//						// get the energy to push back
//						auto En = this->ham_cpx->get_eigenEnergy(i);
//						energies_sym.push_back(En);
//
//						// transform the state
//						Col<cpx> state = this->ham_cpx->get_eigenState(i);
//						Col<cpx> transformed_state = U * state;
//
//						// calculate the entanglement entropy
//						auto entropy = op.entanglement_entropy(transformed_state, La, full_map);
//						entro_inner(i) = entropy;
//
//						entropies.push_back(entropy);
//						entro += entropy;
//						// push back symmetry
//						sym.push_back(std::make_tuple(k, p, x, su_v));
//
//						//file << "\t\t->" << VEQP(this->ham_cpx->get_eigenEnergy(i), 5) << "\t" << "after_trasform:" << arma::cdot(transformed_state, Hafter * transformed_state) << EL;
//					}
//				std::string filenameh5 = dir_separated + VEQ(k) + "," + VEQ(p) + "," + VEQ(x) + "," + VEQ(su_v) + ".h5";
//				this->ham_cpx->get_eigenvalues().save(arma::hdf5_name(filenameh5, "energy", arma::hdf5_opts::append));
//				entro_inner.save(arma::hdf5_name(filenameh5, "entropy", arma::hdf5_opts::append));
//				} 
//			}
//		}
//	}
//	/* PLOT MATRIX DIFFERENCES */
//	if (!(this->eta_a == 0.0 && this->eta_b == 0.0)) {
//		arma::sp_mat HH = arma::real(H);
//		auto N = H0.n_cols;
//		arma::sp_cx_mat res = arma::sp_cx_mat(HH - H0, arma::imag(H));
//		printSeparated(std::cout, '\t', 32, true, "index i", "index j", "difference", "original hamil", "symmetry hamil");
//		cpx x = 0;
//		for (int i = 0; i < N; i++) {
//			for (int j = 0; j < N; j++) {
//				cpx val = res(i, j);
//				if (res(i, j).real() > 1e-13 || res(i, j).imag() > 1e-13) {
//					x += val;
//					printSeparated(std::cout, '\t', 32, true, i, j, res(i, j), H0(i, j), H(i, j));
//					printSeparated(std::cout, '\t', 32, true, i, j, res);
//				}
//			}
//		}
//
//		printSeparated(std::cout, '\t', 32, true, "Sum of suspicious elements: ", x);
//	}
//
//	entro /= double(entropies.size());
//	file << VEQ(Nh) << "\t" << VEQ(state_num) << EL;
//	// --------------------------- sort stuff ---------------------------
//	v_1d<int> index(energies_sym.size(), 0);
//	for (int i = 0; i != index.size(); i++) {
//		index[i] = i;
//	}
//	sort(index.begin(), index.end(),
//		[&](const int& a, const int& b) {
//			return (energies_sym[a] < energies_sym[b]);
//		}
//	);
//
//	// -------------------------- check the ed with symmeties --------------------------
//	double entro_ed = 0;
//
//	openFile(file2, dir + "all_compare.dat");
//	openFile(states, dir + "states_ed.dat");
//	file2 << this->ham_d->get_info() << EL;
//	printSeparated(file2, '\t', 15, true, "E_ed", "E_sym", "Symmetry", "S_ed", "S_sym");
//	Operators<double> op_ed_pbc(this->lat);
//
//	for (u64 i = 0; i < this->ham_d->get_hilbert_size(); i++) {
//		// get the ed energy
//		auto E_ed = this->ham_d->get_eigenEnergy(i);
//		const Col<double> state = (this->ham_d->get_eigenState(i));
//		auto ent_ed = op_ed_pbc.entanglement_entropy(state, Ns / 2);
//
//		// sort
//		auto sort_i = index[i];
//		auto E_sym = energies_sym[sort_i];
//		const auto [k, p, x, su2] = sym[sort_i];
//		auto ent_sym = entropies[sort_i];
//
//		if (i <= 24) {
//			states << VEQ(k) << "," << VEQ(p) << "," << VEQ(x) << "," << VEQ(su2) << "," << VEQ(E_ed) << ":-:" << VEQ(E_sym) << ":\t";
//			for (int z = 0; z < Nh; z++) {
//				states << state(z) << "\t";
//			}
//			states << EL;
//		}
//		printSeparatedP(file2, '\t', 15, true, 7, E_ed, E_sym, STR(k) + "," + STR(p) + "," + STR(x) + "," + STR(su2), ent_ed, ent_sym);
//		entro_ed += ent_ed;
//	}
//	entro_ed /= double(entropies.size());
//	file << VEQP(entro, 6) << "\t<-sym\t" << VEQP(entro_ed, 6) << "\t<-ed\t" << EL;
//	file2.close();
//	states.close();
//	// compare Hamiltonians
//
//	//
//	//auto H0 = arma::mat(this->ham_d->get_hamiltonian());
//	//auto N = H0.n_cols;
//	//Mat<cpx> res = H - (H0);
//	//for (int i = 0; i < N; i++)
//	//	for (int j = 0; j < N; j++)
//	//		if (abs(res(i, j)) > 1e-15)
//	//			printSeparated(file, '\t', 32, true, i, j, res(i, j), H0(i, j));
//	file.close();
//}
