#include "include/user_interface/user_interface.h"
#include "source/src/UserInterface/ui.h"
int LASTLVL = 0;

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::exitWithHelp()
{
	UserInterface::exitWithHelp();
	printf(
		"Usage of the VQMC library:\n"
		"options:\n"
		"-m monte carlo steps	: bigger than 0 (default 300) \n"
		"-d dimension			: set dimension (default 2) \n"
		"	1 -- 1D \n"
		"	2 -- 2D \n"
		"	3 -- 3D \n"
		"-l lattice type		: (default square) -> CHANGE NOT IMPLEMENTED YET \n"
		"   square \n"
		// SIMULATIONS STEPS
		"\n"
		"-fun					: function to be used in the calculations. There are predefined functions in the model that allow that:\n"
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
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

	// rbm
	this->nqsP.setDefault();

	// others 
	this->threadNum = 1;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// ####################################### MOLTO IMPORTANTE ######################################

// ###############################################################################################

/*
* @brief model parser
* @param argc number of line arguments
* @param argv line arguments
*/
void UI::parseModel(int argc, cmdArg& argv)
{
	// --------- HELP
	if (std::string option = this->getCmdOption(argv, "-h"); option != "")
		this->exitWithHelp();

	// set default at first
	this->setDefault();

	std::string choosen_option = "";
	// ---------- SIMULATION PARAMETERS ----------
	{
		// training parameters
		SETOPTION(nqsP,	nqs_tr_epo);
		SETOPTION(nqsP,	nqs_tr_mc);
		SETOPTION(nqsP,	nqs_tr_bs);
		SETOPTION(nqsP,	nqs_tr_th);
		SETOPTION(nqsP,	nqs_tr_pinv);
		SETOPTION(nqsP,	nqs_tr_pc);
		SETOPTION(nqsP,	nqs_tr_reg);
		// collecting - excited
		SETOPTION(nqsP,	nqs_ex_mc);
		SETOPTION(nqsP,	nqs_ex_bn);
		SETOPTION(nqsP,	nqs_ex_th);
		SETOPTION(nqsP,	nqs_ex_bs);
		SETOPTIONVECTORRESIZET(nqsP, nqs_ex_beta, 1, double);

		// collecting - all
		SETOPTION(nqsP,	nqs_col_mc);
		SETOPTION(nqsP,	nqs_col_bn);
		SETOPTION(nqsP,	nqs_col_th);	
		SETOPTION(nqsP,	nqs_col_bs);	

		// learming rate
		SETOPTION(nqsP,  nqs_sch);
		SETOPTION(nqsP,	 nqs_lr);
		SETOPTION(nqsP,  nqs_lrd);
		SETOPTION(nqsP,  nqs_lr_pat);

		// early stopping
		SETOPTION(nqsP,  nqs_es_pat);
		SETOPTION(nqsP,  nqs_es_del);

		SETOPTION(nqsP, nqs_nh);
		SETOPTION(nqsP,	 nFlips);
		SETOPTIONV(nqsP, type, 		"nqst");
		SETOPTIONV(nqsP, loadNQS,	"lNQS"	);
		SETOPTION(nqsP,  nqs_ed				);
	}
	// ----------------- LATTICE -----------------
	{
		SETOPTIONV(latP,	typ,	"l"	);
		SETOPTIONV(latP,	dim,	"d"	);
		SETOPTION(latP,		Lx			);
		SETOPTION(latP,		Ly			);
		SETOPTION(latP,		Lz			);
		SETOPTION(latP,		bc			);
		SETOPTION(latP,		Ntot		);
		SETOPTIONVECTORRESIZE(latP, Ntots, 10);
	}
	int Ns [[maybe_unused]] = latP.Lx_ * latP.Ly_ * latP.Lz_;
	if (latP.typ_ == LatticeTypes::HEX && latP.dim_ > 1)
		Ns *= 2;
	// for now, we set the number of sites in the NQS to the total number of sites from the lattice
	// !TODO: change this to be more flexible and allow for different number of sites in the NQS (without the need to have the lattice)
	nqsP.nVisible_ = Ns;
	if (nqsP.nqs_nh_ < 0)
		this->nqsP.nqs_nh_ = (int)std::abs(nqsP.nqs_nh_ * Ns);
	

	// ------------------ MODEL ------------------
	{
		// model type
		SETOPTIONV(modP, modTyp, "mod");
		this->modP.modRanN_ = { 1 };
		SETOPTIONVECTORRESIZET(modP, modRanN, 10, uint);
		SETOPTION(modP, modRanSeed);
		SETOPTION(modP, modMidStates);
		SETOPTION(modP, modEnDiff);

		// eth
		SETOPTION(modP, eth_entro);
		SETOPTION(modP, eth_susc);
		SETOPTION(modP, eth_ipr);
		SETOPTION(modP, eth_offd);
		SETOPTIONVECTORRESIZET(modP, eth_end, 10, double);

		// set operators vector
		this->setOption<std::string>(modP.operators, argv, "op", "sz/L", true);

		// ---- quadratic ----
		{
			SETOPTIONV(modP, q_manybody,		"q_mb");
			SETOPTIONV(modP, q_manifold,		"q_man");
			SETOPTIONV(modP, q_gamma,			"q_gamma");
			SETOPTIONV(modP, q_realizationNum,	"q_R");
			SETOPTIONV(modP, q_randomCombNum,	"q_CN");
			SETOPTIONV(modP, q_shuffle,			"q_S");
			SETOPTIONV(modP, q_broad,			"q_broad");
			
			// -- aubry-andre ---
			{
				SETOPTION_STEP(modP.aubry_andre, aa_beta);
				SETOPTION_STEP(modP.aubry_andre, aa_phi);
				SETOPTION_STEP(modP.aubry_andre, aa_J);
				SETOPTION_STEP(modP.aubry_andre, aa_lambda);
			}

			// --- power-law ---
			{
				SETOPTIONVECTORRESIZET(modP.power_law_random_bandwidth, plrb_a, 10, double);
				// SETOPTION(modP.power_law_random_bandwidth, plrb_a);
				SETOPTION(modP.power_law_random_bandwidth, plrb_b);
				SETOPTION(modP.power_law_random_bandwidth, plrb_mb);
			}
		}
		
		// ------ spin ------
		{
			// ------ ising ------
			{
				SETOPTION_STEP(modP, J1);
				SETOPTION_STEP(modP, hx);
				SETOPTION_STEP(modP, hz);
			}
			// ---- heisenberg ---
			{
				// resize
				this->modP.resizeHeisenberg(Ns);

				SETOPTION(modP, heiJ);
				SETOPTION(modP, heiDlt);
				SETOPTION(modP, heiHz);
				SETOPTION(modP, heiHx);
				SETOPTION_STEP(modP, dlt1);
			}
			// ------- xyz -------
			{
				SETOPTION_STEP(modP, J2);
				SETOPTION_STEP(modP, dlt2);
				SETOPTION_STEP(modP, eta1);
				SETOPTION_STEP(modP, eta2);
			}
			// ------ kitaev -----
			{
				// resize
				this->modP.resizeKitaev(Ns);

				// set options
				SETOPTION(modP, Kx);
				SETOPTION(modP, Ky);
				SETOPTION(modP, Kz);
			}
			// ------ QSM ---------
			{
				SETOPTION(modP.qsm, qsm_gamma);
				SETOPTION(modP.qsm, qsm_g0);
				this->modP.qsm.qsm_Ntot_ = this->latP.Ntot_;
				SETOPTION(modP.qsm, qsm_N);

				// resize
				this->modP.qsm.resizeQSM();
				
				// set
				SETOPTIONVECTORRESIZET(modP.qsm, qsm_alpha, 10, double);
				SETOPTIONVECTOR(modP.qsm, qsm_xi);
				SETOPTIONVECTOR(modP.qsm, qsm_h);
			}
			// - ROSENZWEIG-PORTER -
			{
				SETOPTION(modP.rosenzweig_porter, rp_g_sweep_n);
				modP.rosenzweig_porter.resizeRP();
				SETOPTIONVECTOR(modP.rosenzweig_porter, rp_g);
				SETOPTION(modP.rosenzweig_porter, rp_single_particle);
				SETOPTION(modP.rosenzweig_porter, rp_be_real);
			}
			// ---- ULTRAMETRIC ----
			{
				this->modP.ultrametric.um_Ntot_ = this->latP.Ntot_;
				SETOPTION(modP.ultrametric, um_N);	
				SETOPTION(modP.ultrametric, um_g);
				
				this->modP.ultrametric.resizeUM();

				SETOPTIONVECTOR(modP.ultrametric, um_alpha);

			}
		}
	}
	// --------------- SYMMETRIES ----------------
	{
		SETOPTION(symP, k);
		SETOPTION(symP, px);
		SETOPTION(symP, py);
		SETOPTION(symP, pz);
		SETOPTION(symP, x);
		SETOPTION(symP, U1);
		SETOPTION(symP, S);
	}
	// ----------------- OTHERS ------------------
	this->parseOthers(argv);
	// ---------------- DIRECTORY ----------------
	this->parseMainDir(argv);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief chooses the method to be used later based on input -fun argument
*/
void UI::funChoice()
{
	LOGINFO_CH_LVL(0);
	LOGINFO("USING #THREADS=" + STR(this->threadNum), LOG_TYPES::CHOICE, 1);
	this->_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	
	BEGIN_CATCH_HANDLER
	{
		switch (this->chosenFun)
		{
		case -1:
			// default case of showing the help
			this->exitWithHelp();
			break;
			// ------------------------------- NEURAL QST -------------------------------
		case 10:
			// this option utilizes the Hamiltonian with NQS ansatz calculation
			LOGINFO("SIMULATION: HAMILTONIAN WITH NQS", LOG_TYPES::CHOICE, 1);
			this->makeSimNQS();
			break;
		case 11:
			// this option utilizes the Hamiltonian with NQS ansatz calculation - excited states
			LOGINFO("SIMULATION: HAMILTONIAN WITH NQS - EXCITED STATES", LOG_TYPES::CHOICE, 1);
			this->makeSimNQSExcited();
			break;
			// ------------------------------- SYMMETRIES -------------------------------
		case 20:
			// this option utilizes the Hamiltonian with symmetries calculation
			LOGINFO("SIMULATION: HAMILTONIAN WITH SYMMETRIES - ALL SECTORS", LOG_TYPES::CHOICE, 1);
			this->symmetriesTest();
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
		case 23:
			// this option creates a map between Hamiltonian type and the Hilbert space size with symmetries
			LOGINFO("SIMULATION: HAMILTONIAN WITH SYMMETRIES - SAVE ALL HILBERT", LOG_TYPES::CHOICE, 1);
			this->makeSimSymmetriesSweepHilbert();
			break;
		case 24:
			// this option utilizes the Hamiltonian with symmetries calculation
			LOGINFO("SIMULATION: HAMILTONIAN WITH SYMMETRIES - SAVE STATES", LOG_TYPES::CHOICE, 1);
			this->makeSimSymmetries(true, true);
			break;
		case 25:
			// this option utilizes the Hamiltonian with symmetries calculation
			LOGINFO("SIMULATION: HAMILTONIAN WITH SYMMETRIES - SAVE DEGENERACIES", LOG_TYPES::CHOICE, 1);
			this->makeSimSymmetriesDeg();
			break;
		case 26:
			// this option takes the Hamiltonian in a given symmetry sector and calculates entropy of combination of states
			LOGINFO("SIMULATION: HAMILTONIAN WITH SYMMETRIES - CREATE DEGENERACIES NEAR ZERO", LOG_TYPES::CHOICE, 1);
			this->makeSimSymmetriesCreateDeg();
			break;
			// ------------------------------- QUADRATIC -------------------------------
		case 30:
			// this option utilizes the quadratic Hamiltonian calculation
			LOGINFO("SIMULATION: QUADRATIC HAMILTONIAN - STATES MIXING", LOG_TYPES::CHOICE, 1);;
			this->makeSymQuadraticManifold();
			break;
		case 31:
			// this option utilizes the quadratic Hamiltonian calculation with spectral functions
			LOGINFO("SIMULATION: QUADRATIC HAMILTONIAN - SPECTRAL FUNCTIONS", LOG_TYPES::CHOICE, 1);
			this->makeSimQuadraticSpectralFunction();
			break;
			// ----------------------------------- ETH ---------------------------------
		case 40:
			// this option utilizes the Hamiltonian for ETH
			LOGINFO("SIMULATION: HAMILTONIAN - ETH", LOG_TYPES::CHOICE, 1);
			this->makeSimETH();
			break;
		case 41:
			// this option utilizes the Hamiltonian for ETH
			LOGINFO("SIMULATION: HAMILTONIAN - ETH", LOG_TYPES::CHOICE, 1);
			this->makeSimETHSweep();
			break;
		case 42:
			// this option utilizes the Hamiltonian for ETH statistics
			LOGINFO("SIMULATION: HAMILTONIAN - ETH - statistics", LOG_TYPES::CHOICE, 1);
			this->makeSimETH();
			break;
		case 43:
			// this option utilizes the Hamiltonian for ETH statistics
			LOGINFO("SIMULATION: HAMILTONIAN - ETH - statistics sweep", LOG_TYPES::CHOICE, 1);
			this->makeSimETHSweep();
			break;
		case 44:
			// this option utilizes the Hamiltonian for ETH statistics - offdiagonal elements scaling
			LOGINFO("SIMULATION: HAMILTONIAN - ETH - statistics offdiagonal elements scaling", LOG_TYPES::CHOICE, 1);
			this->makeSimETH();
			break;
		case 45:
			// this option utilizes the Hamiltonian time evolution for ETH statistics
			LOGINFO("SIMULATION: HAMILTONIAN - ETH - statistics time evolution", LOG_TYPES::CHOICE, 1);
			this->makeSimETHSweep();
			break;
		case 46:
			// this option utilizes the Hamiltonian time evolution for ETH statistics
			LOGINFO("SIMULATION: HAMILTONIAN - ETH - statistics time evolution sweep", LOG_TYPES::CHOICE, 1);
			this->makeSimETH();
			break;
		default:
			// default case of showing the help
			this->exitWithHelp();
			break;
		}
	}
	END_CATCH_HANDLER("The function chooser returned with: ", return);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Defines the lattice in the system
*/
bool UI::defineLattice()
{
	return this->defineLattice(this->latP.lat, this->latP.typ_);
}

/*
* @brief Defines the lattice in the system
* @param _lat lattice to be defined
* @return true if the lattice was defined
*/
bool UI::defineLattice(std::shared_ptr<Lattice>& _lat, LatticeTypes _typ)
{
	BEGIN_CATCH_HANDLER
	{
		switch (_typ)
		{
		case LatticeTypes::SQ:
			_lat = std::make_shared<SquareLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_, this->latP.dim_, this->latP.bc_);
			break;
		case LatticeTypes::HEX:
			_lat = std::make_shared<HexagonalLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_, this->latP.dim_, this->latP.bc_);
			break;
		default:
			_lat = std::make_shared<SquareLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_, this->latP.dim_, this->latP.bc_);
			break;
		};
	}
	END_CATCH_HANDLER("Exception in setting the lattices: ", return false;);
	return true;
}

// ----------------------------------------------------------------------------------------------

/*
* @brief defines the models based on the input parameters - interacting
* Using the Hamiltonians defined within the class...
*/
bool UI::defineModels(bool _createLat, bool _checkSyms, bool _useHilbert) 
{
	// create lattice if not created
	if (_createLat && !this->latP.lat)
		this->defineLattice();

	if(_checkSyms)
		this->isComplex_	= this->symP.checkComplex(this->latP.lat->get_Ns());

	// go complex if needed
	const bool _takeComplex	= (this->isComplex_ || this->useComplex_);	
	
	LOGINFO("Making : " + std::string(_takeComplex ? " complex" : " real"), LOG_TYPES::INFO, 3);
	
	if (_useHilbert)
	{
		if (_takeComplex)
			return this->defineModel(this->hilComplex, this->hamComplex);
		else
			return this->defineModel(this->hilDouble, this->hamDouble);
	}
	else
	{
		if (_takeComplex)
			return this->defineModel(this->hamComplex, this->latP.Ntot_);
		else
			return this->defineModel(this->hamDouble, this->latP.Ntot_);
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief defines the models based on the input parameters - quadratic
*/
bool UI::defineModelsQ(bool _createLat)
{
	// create lattice
	LOGINFO("Making quadratic model!", LOG_TYPES::INFO, 1);

	if (_createLat && !this->latP.lat)
		this->defineLattice();

	// check if is complex
	this->isComplex_	= this->modP.checkComplex();
	bool _takeComplex	= (this->isComplex_ || this->useComplex_);	
	LOGINFO("Making : " + std::string(_takeComplex ? " complex" : " real"), LOG_TYPES::INFO, 3);
	
	// check if is complex and define the Hamiltonian
	if (this->latP.lat && _createLat)
	{
		if (_takeComplex)
			return this->defineModel<cpx>(this->hamComplex, this->latP.lat);
		else
			return this->defineModel<double>(this->hamDouble, this->latP.lat);
	}
	else
	{
		if (_takeComplex)
			return this->defineModel<cpx>(this->hamComplex, this->latP.Ntot_);
		else
			return this->defineModel<double>(this->hamDouble, this->latP.Ntot_);
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief A placeholder for making the simulation with symmetries.
* @param Should diagonalize and proceed with the simulation?
*/
void UI::makeSimSymmetries(bool _diag, bool _states)
{
	// define the models
	this->resetEd();
	// define the models
	if (this->defineModels(true))
	{
		if (this->isComplex_)
			this->symmetries(this->hamComplex, _diag, _states);
		else
			this->symmetries(this->hamDouble, _diag, _states);
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Use the simulation and Haar matrices to calculate the degeneracies and entropies
*/
void UI::makeSimSymmetriesDeg()
{
	// define the models
	this->resetEd();
	// force complex Hamiltonian
	this->useComplex_ = true;
	// simulate
	if (this->defineModels(true))
		this->symmetriesDeg();
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Use the simulation and CUE matrices to calculate entanglement entropy of multiple degeneracies
*/
void UI::makeSimSymmetriesCreateDeg()
{
	// define the models
	this->resetEd();
	// force complex Hamiltonian
	this->useComplex_ = true;
	// simulate
	if (this->defineModels(true))
		this->symmetriesCreateDeg();
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief A placeholder for making the simulation with symmetries, sweeping them all
*/
void UI::makeSimSymmetriesSweep()
{
	LOGINFO_CH_LVL(3);
	this->defineModels(true);
	uint Ns = this->latP.lat->get_Ns();
	auto BC = this->latP.lat->get_BC();
	u64 Nh [[maybe_unused]] = 1;

	// parameters
	v_1d<int> kS = {};
	v_1d<int> Rs = {};
	v_1d<int> Szs = {};
	v_1d<int> Sys = {};
	v_1d<int> U1s = {};
	v_1d<int> Sxs = {};

	bool useU1 = (this->modP.modTyp_ == MY_MODELS::XYZ_M) && this->modP.eta1_ == 0 && this->modP.eta2_ == 0;
	bool useSzParity = false;//(this->modP.modTyp_ == MY_MODELS::XYZ_M);// && (Ns % 2 == 0);
	bool useSyParity = false; //(this->modP.modTyp_ == MY_MODELS::XYZ_M) && (Ns % 2 == 0);

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

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief A placeholder for making the simulation with symmetries, sweeping them all. Saves the Hilbert space sizes for future reference of what to run.
*/
void UI::makeSimSymmetriesSweepHilbert()
{
	LOGINFO_CH_LVL(3);
	this->defineModels(true);
	uint Ns					= this->latP.lat->get_Ns();
	auto BC					= this->latP.lat->get_BC();
	u64 Nh [[maybe_unused]] = 0;

	// parameters
	v_1d<int> kS			= {};
	v_1d<int> Rs			= {};
	v_1d<int> Szs			= {};
	v_1d<int> Sys			= {};
	v_1d<int> U1s			= {};
	v_1d<int> Sxs			= {};

	bool useU1				= (this->modP.modTyp_ == MY_MODELS::XYZ_M) && this->modP.eta1_ == 0 && this->modP.eta2_ == 0;
	bool useSzParity		= false; //(this->modP.modTyp_ == MY_MODELS::XYZ_M) && (this->modP.eta1_ != 0 && this->modP.eta2_ != 0);
	bool useSyParity		= false; //(this->modP.modTyp_ == MY_MODELS::XYZ_M) && (Ns % 2 == 0);

	if (useSzParity)	Szs = { -1, 1 }; else Szs = { -INT_MAX };
	if (useSyParity)	Sys = { -1, 1 }; else Sys = { -INT_MAX };
	if (useU1)			for (uint i = 0; i <= Ns; i++) U1s.push_back(i); else U1s.push_back(-INT_MAX);
	if (BC == PBC)		for (uint i = 0; i < Ns; i++) kS.push_back(i); else kS.push_back(-INT_MAX);

	// go through all
	LOGINFO("STARTING ALL SECTORS", LOG_TYPES::INFO, 1);
	for (auto k : kS)
	{
		this->symP.k_ = k;
		// check Reflection
		if (k == -INT_MAX || k == 0 || (k == int(Ns / 2) && (Ns % 2) == 0))
			Rs = { -1, 1 };
		else
			Rs = { -INT_MAX };
		for (auto r : Rs) {
			this->symP.x_ = r;
			for (auto pz : Szs) {
				this->symP.pz_ = pz;
				for (auto u1 : U1s)
				{
					this->symP.U1_ = u1;
					// check Parity X
					if ((!useU1 && (this->modP.hz_ == 0.0)) || (useU1 && (Ns % 2 == 0) && (this->symP.U1_ == Ns / 2) && (this->modP.hz_ == 0.0) && (this->modP.hx_ == 0.0)))
						Sxs = { -1, 1 };
					else
						Sxs = { -INT_MAX };
					for (auto px : Sxs) {
						this->symP.px_ = px;
						// create symmetries vectors
						this->createSymmetries();
						// reset Hamiltonians - memory release
						if (this->hamComplex)
							this->hamComplex.reset();
						if (this->hamDouble)
							this->hamDouble.reset();
						// define the models
						bool empty [[maybe_unused]] = this->defineModels(true);
						// save the Hilbert space sizes
						if (this->isComplex_)
							this->symmetries(this->hamComplex, false);
						else
							this->symmetries(this->hamDouble, false);
						// add Hilbert space size
						if (this->hamComplex)
							Nh += this->hamComplex->getHilbertSize();
						if (this->hamDouble)
							Nh += this->hamDouble->getHilbertSize();
						LOGINFO("-----------------------------------", LOG_TYPES::TRACE, 0);
					}
				}
			}
		}
	}
	LOGINFO("------------------ " + VEQ(Nh) + "/" + VEQ(useU1 ? std::pow(2.0, Ns) : 1) + " ------------------", LOG_TYPES::TRACE, 0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::makeSymQuadraticManifold()
{
	// reset Hamiltonians - memory release
	this->resetQuadratic();
	//this->useComplex_ = true;
	if (check_noninteracting(this->modP.modTyp_))
	{
		if (this->defineModels(false))
		{
			if (this->isComplex_)
				this->quadraticStatesManifold<cpx>(this->hamComplex);
			else
				this->quadraticStatesManifold<double>(this->hamDouble);
		}
	}
	else 
	{
		if (this->defineModelsQ(false))
		{
			if (this->isComplex_)
				this->quadraticStatesManifold<cpx>(this->hamComplex);
			else
				this->quadraticStatesManifold<double>(this->hamDouble);
		}
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::makeSimQuadraticSpectralFunction()
{
	// reset Hamiltonians - memory release
	this->resetQuadratic();
	this->useComplex_ = false;
	if (this->defineModelsQ(isLatticeModel(this->modP.modTyp_)))
	{
		if(this->useComplex_)
			this->quadraticSpectralFunction<cpx>(this->hamComplex);
		else
			this->quadraticSpectralFunction<double>(this->hamDouble);
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// ------------------------------------------------