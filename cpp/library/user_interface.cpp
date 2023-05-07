#include "include/user_interface/user_interface.h"
int LASTLVL = 0;

/*
* @brief  Setting parameters to default.
*/
void UI::setDefault()
{
	// lattice stuff
	this->latP.setDefault();

	// symmetries stuff
	this->symP.setDefault();

	// define basic model
	this->modP.setDefault();

	// others 
	this->threadNum = 1;

	// rbm
	//this->nqsP.lr = 1e-2;
	//this->nqsP.blockSize = 8;
	//this->nqsP.nBlocks = 500;
	//this->nqsP.mcSteps = 1000;
	//this->nqsP.batch = (u64)std::pow(2, 10);
	//this->nqsP.nVisible = latP.lat->get_Ns();
	//this->nqsP.nHidden = 2ll * latP.lat->get_Ns();
	//this->nqsP.layersDim = { this->nqsP.nVisible , this->nqsP.nHidden };
	//this->nqsP.nTherm = uint(0.1 * this->nqsP.nBlocks);
	//this->nqsP.nFlips = 1;
}

/*
* @brief model parser
* @param argc number of line arguments
* @param argv line arguments
*/
void UI::parseModel(int argc, cmdArg& argv)
{
	// --------- HELP
	if (std::string option = this->getCmdOption(argv, "-hlp"); option != "")
		this->exitWithHelp();

	// set default at first
	this->setDefault();

	std::string choosen_option = "";

	// -------------------- SIMULATION PARAMETERS --------------------
	SETOPTIONV(		nqsP, nMcSteps	,"m" );
	SETOPTIONV(		nqsP, batch		,"b" );
	SETOPTIONV(		nqsP, nBlocks	,"nb");
	SETOPTIONV(		nqsP, blockSize	,"bs");
	SETOPTIONV(		nqsP, nHidden	,"nh");
	SETOPTIONV(		nqsP, nFlips	,"nf");
	SETOPTION(		nqsP, lr			 );
	this->nqsP.nTherm_	= uint(0.1 * nqsP.nBlocks_);
	
	// ---------- LATTICE ----------
	SETOPTIONV(		latP, typ, "l"		);
	SETOPTIONV(		latP, dim, "d"		);
	SETOPTION(		latP, Lx			);
	SETOPTION(		latP, Ly			);
	SETOPTION(		latP, Lz			);
	SETOPTION(		latP, bc			);
	int Ns [[maybe_unused]] = latP.Lx_ * latP.Ly_ * latP.Lz_;

	// ---------- MODEL ----------

	// model type
	SETOPTIONV(		modP, modTyp, "mod"	);
	// --- ising ---
	SETOPTION_STEP(	modP, J1			);
	SETOPTION_STEP(	modP, hx			);
	SETOPTION_STEP(	modP, hz			);
	// --- heisenberg ---
	SETOPTION_STEP(	modP, dlt1			);
	// --- xyz ---
	SETOPTION_STEP(	modP, J2			);
	SETOPTION_STEP(	modP, dlt2			);
	SETOPTION_STEP(	modP, eta1			);
	SETOPTION_STEP(	modP, eta2			);
	// --- kitaev ---
	SETOPTION_STEP( modP, kx			);
	SETOPTION_STEP( modP, ky			);
	SETOPTION_STEP( modP, kz			);

	// ---------- SYMMETRIES ----------
	SETOPTION(		symP, k				);
	SETOPTION(		symP, px			);
	SETOPTION(		symP, py			);
	SETOPTION(		symP, pz			);
	SETOPTION(		symP, x				);
	SETOPTION(		symP, U1			);
	SETOPTION(		symP, S				);

	// ---------- OTHERS
	this->setOption(this->quiet		, argv, "q"	);
	this->setOption(this->threadNum	, argv, "th"	);

	// later function choice
	this->setOption(this->chosenFun	, argv, "fun"	);

	//---------- DIRECTORY

	bool setDir		[[maybe_unused]] =	this->setOption(this->mainDir, argv, "dir");
	this->mainDir	=	fs::current_path().string() + kPS + "DATA" + kPS + this->mainDir + kPS;

	// create the directories
	createDir(this->mainDir);
}

/*
* @brief chooses the method to be used later based on input -fun argument
*/
void UI::funChoice()
{
	LOGINFO_CH_LVL(0);
	switch (this->chosenFun)
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
		case 11:
			// this option utilizes the Hamiltonian with NQS ansatz calculation
			LOGINFO("SIMULATION: HAMILTONIAN WITH NQS", LOG_TYPES::CHOICE, 1);
			this->makeSimNQS();
			break;
	case 20:
		// this option utilizes the Hamiltonian with symmetries calculation
		LOGINFO("SIMULATION: HAMILTONIAN WITH SYMMETRIES - ALL SECTORS", LOG_TYPES::CHOICE, 1);
		this->symmetriesTest(clk::now());
		break;
	case 21:
		// this option utilizes the Hamiltonian with symmetries calculation
		LOGINFO("SIMULATION: HAMILTONIAN WITH SYMMETRIES", LOG_TYPES::CHOICE, 1);
		this->makeSimSymmetries();
		break;
	case 22:
		// this option utilizes the Hamiltonian with symmetries calculation - sweep!
		LOGINFO("SIMULATION: HAMILTONIAN WITH SYMMETRIES - SWEEP ALL", LOG_TYPES::CHOICE, 1);
		this->makeSimSymmetriesSweep();
		break;
	default:
		// default case of showing the help
		this->exitWithHelp();
		break;
	}
	LOGINFO("USING #THREADS=" + STR(this->threadNum), LOG_TYPES::CHOICE, 1);
}

/*
* @brief defines the models based on the input parameters
*/
bool UI::defineModels(bool _createLat) {
	// create lattice
	if (_createLat && !this->latP.lat)
	{
		switch (this->latP.typ_)
		{
		case LatticeTypes::SQ:
			this->latP.lat = std::make_shared<SquareLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_,
				this->latP.dim_, this->latP.bc_);
			break;
		case LatticeTypes::HEX:
			this->latP.lat = std::make_shared<HexagonalLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_,
				this->latP.dim_, this->latP.bc_);
			break;
		default:
			this->latP.lat = std::make_shared<SquareLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_,
				this->latP.dim_, this->latP.bc_);
			break;
		};
	}
	// check if is complex
	this->isComplex_ = this->symP.checkComplex(this->latP.lat->get_Ns());

	// check if is complex and define the Hamiltonian
	bool _ok;
	if (this->isComplex_)
		_ok = this->defineModel(this->hilComplex, this->hamComplex);
	else
		_ok = this->defineModel(this->hilDouble, this->hamDouble);

	return _ok;

}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

std::pair<v_1d<GlobalSyms::GlobalSym>, v_1d<std::pair<Operators::SymGenerators, int>>> UI::createSymmetries()
{
	v_1d<GlobalSyms::GlobalSym> _glbSyms = {};
	v_1d<std::pair<Operators::SymGenerators, int>> _locSyms = {};
	if (this->symP.S_ == true)
	{
		// create Hilbert space
		this->isComplex_ = this->symP.checkComplex(this->latP.lat->get_Ns());
		// ------ LOCAL ------
		_locSyms = this->symP.getLocGenerator();
		// ------ GLOBAL ------
		// check U1
		if (this->symP.U1_ != -INT_MAX) _glbSyms.push_back(GlobalSyms::getU1Sym(this->latP.lat, this->symP.U1_));
	};
	return std::make_pair(_glbSyms, _locSyms);
}

/*
* @brief A placeholder for making the simulation with symmetries.
*/
void UI::makeSimSymmetries()
{
	// reset Hamiltonians - memory release
	if (this->hamComplex)
		this->hamComplex.reset();
	if (this->hamDouble)
		this->hamDouble.reset();
	// define the models
	if (!this->defineModels(true)) 
		return;
	if (this->isComplex_)
		this->symmetries(clk::now(), this->hamComplex);
	else
		this->symmetries(clk::now(), this->hamDouble);
}

/*
* @brief A placeholder for making the simulation with symmetries, sweeping them all
*/
void UI::makeSimSymmetriesSweep()
{
	LOGINFO_CH_LVL(3);
	this->defineModels(true);
	uint Ns				= this->latP.lat->get_Ns();
	auto BC				= this->latP.lat->get_BC();
	u64 Nh				[[maybe_unused]] = 1;

	// parameters
	v_1d<int> kS		= {};
	v_1d<int> Rs		= {};
	v_1d<int> Szs		= {};
	v_1d<int> Sys		= {};
	v_1d<int> U1s		= {};
	v_1d<int> Sxs		= {};

	bool useU1			= (this->modP.modTyp_ == MY_MODELS::XYZ_M) && this->modP.eta1_ == 0 && this->modP.eta2_ == 0;
	bool useSzParity	= (this->modP.modTyp_ == MY_MODELS::XYZ_M);// && (Ns % 2 == 0);
	bool useSyParity	= false; //(this->modP.modTyp_ == MY_MODELS::XYZ_M) && (Ns % 2 == 0);

	if (useSzParity)	Szs = { -1, 1 }; else Szs = { -INT_MAX };
	if (useSyParity)	Sys = { -1, 1 }; else Sys = { -INT_MAX };
	if (useU1)			for (uint i = 0; i <= Ns; i++) U1s.push_back(i); else U1s.push_back(-INT_MAX);
	if (BC == PBC)		for (uint i = 0; i <= int(Ns / 2) + 1; i++) kS.push_back(i); else kS.push_back(-INT_MAX);

	// go through all
	LOGINFO("STARTING ALL SECTORS", LOG_TYPES::INFO, 1);
	for (auto k : kS)
	{
		this->symP.k_ = k;
		// check Reflection
		if (k == 0 || (k == int(Ns / 2) && (Ns % 2) == 0))
			Rs = { -1, 1 };
		else
			Rs = { -INT_MAX };
		for (auto r : Rs) {
			this->symP.x_ = r;
			for (auto pz : Szs) {
				this->symP.pz_ = pz;
				for (auto u1 : U1s) {
					this->symP.U1_ = u1;
					// check Parity X
					if ((!useU1 && (this->modP.hz_ == 0.0)) || (useU1 && (Ns % 2 == 0) && (this->symP.U1_ == Ns / 2) && (this->modP.hz_ == 0.0) && (this->modP.hx_ == 0.0)))
						Sxs = { -1, 1 };
					else
						Sxs = { -INT_MAX };
					for (auto px : Sxs) {
						this->symP.px_ = px;
						this->createSymmetries();
						this->makeSimSymmetries();
						LOGINFO("-----------------------------------", LOG_TYPES::TRACE, 0);
					}
				}
			}
		}
	}
}

/*
* @brief A placeholder for making the simulation with NQS.
*/
void UI::makeSimNQS()
{
	this->defineModels();
	if (this->isComplex_) {
		this->defineNQS<cpx>(this->hamComplex, this->nqsCpx);
		this->nqsSingle(clk::now(), this->nqsCpx);
	}
	else {
		this->defineNQS<double>(this->hamDouble, this->nqsDouble);
		this->nqsSingle(clk::now(), this->nqsDouble);
	}
}


// -------------------------------------------------------- SIMULATIONS



///*
//*
//*/
//template<typename _type, typename _hamtype>
//void rbm_ui::ui<_type, _hamtype>::ui::make_simulation()
//{
//	// monte carlo
//
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
////
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
