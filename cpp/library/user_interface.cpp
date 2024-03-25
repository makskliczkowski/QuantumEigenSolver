#include "include/user_interface/user_interface.h"
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

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
		SETOPTIONV(nqsP,	nMcSteps,	"m"		);
		SETOPTIONV(nqsP,	blockSize,	"bs"	);
		SETOPTIONV(nqsP,	nBlocks,	"nb"	);
		SETOPTIONV(nqsP,	nHidden,	"nh"	);
		SETOPTIONV(nqsP,	nFlips,		"nf"	);
		SETOPTIONV(nqsP,	type,		"nqst"	);
		SETOPTIONV(nqsP,	nTherm,		"nt"	);
		SETOPTION(nqsP,		lr					);
		SETOPTIONV(nqsP,	loadNQS,	"lNQS"	);
		// samples
		SETOPTIONV(nqsP,	nMcSamples,	"mcS"	);
		SETOPTIONV(nqsP,	blockSizeS,	"bsS"	);
		SETOPTIONV(nqsP,	nSBlocks,	"nbS"	);
		//this->nqsP.nTherm_ = uint(0.1 * nqsP.nBlocks_);
	}
	// ----------------- LATTICE -----------------
	{
		SETOPTIONV(latP,	typ,	"l"	);
		SETOPTIONV(latP,	dim,	"d"	);
		SETOPTION(latP,		Lx			);
		SETOPTION(latP,		Ly			);
		SETOPTION(latP,		Lz			);
		SETOPTION(latP,		bc			);
	}
	int Ns [[maybe_unused]] = latP.Lx_ * latP.Ly_ * latP.Lz_;
	if (latP.typ_ == LatticeTypes::HEX && latP.dim_ > 1)
		Ns *= 2;
	// ------------------ MODEL ------------------
	{
		// model type
		SETOPTIONV(modP, modTyp, "mod");
		SETOPTION(modP, modRanN);
		SETOPTION(modP, modRanSeed);
		SETOPTION(modP, modMidStates);
		SETOPTION(modP, modEnDiff);

		// ---- quadratic ----
		{
			SETOPTIONV(modP, modTypQ,			"modQ");
			SETOPTIONV(modP, q_manybody,		"q_mb");
			SETOPTIONV(modP, q_manifold,		"q_man");
			SETOPTIONV(modP, q_gamma,			"q_gamma");
			SETOPTIONV(modP, q_realizationNum,	"q_R");
			SETOPTIONV(modP, q_randomCombNum,	"q_CN");
			SETOPTIONV(modP, q_shuffle,			"q_S");
			
			// -- aubry-andres ---
			{
				SETOPTION_STEP(modP, Beta);
				SETOPTION_STEP(modP, Phi);
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
				SETOPTION(modP.qsm, qsm_Ntot);
				SETOPTION(modP.qsm, qsm_N);

				// resize
				this->modP.qsm.resizeQSM();
				
				// set
				SETOPTIONVECTOR(modP.qsm, qsm_alpha);
				SETOPTIONVECTOR(modP.qsm, qsm_xi);
				SETOPTIONVECTOR(modP.qsm, qsm_h);
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
	BEGIN_CATCH_HANDLER
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
	END_CATCH_HANDLER("Exception in setting the lattices: ", return false;);
	return true;
}

// ----------------------------------------------------------------------------------------------

/*
* @brief defines the models based on the input parameters - interacting
*/
bool UI::defineModels(bool _createLat, bool _checkSyms, bool _useHilbert) 
{
	// create lattice if not created
	if (_createLat && !this->latP.lat)
		this->defineLattice();

	if(_checkSyms)
		this->isComplex_	= this->symP.checkComplex(this->latP.lat->get_Ns());

	// go complex if needed
	bool _takeComplex	= (this->isComplex_ || this->useComplex_);	
	
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
			return this->defineModel(this->hamComplex);
		else
			return this->defineModel(this->hamDouble);
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
	this->isComplex_	= this->symP.checkComplex(this->latP.lat->get_Ns());
	bool _takeComplex	= (this->isComplex_ || this->useComplex_);	
	LOGINFO("Making : " + std::string(_takeComplex ? " complex" : " real"), LOG_TYPES::INFO, 3);
	
	// check if is complex and define the Hamiltonian
	if (_takeComplex)
		return this->defineModelQ<cpx>(this->qhamComplex);
	else
		return this->defineModelQ<double>(this->qhamDouble);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Based on symmetry parametes, create local and global symmetries
*/
std::pair<v_1d<GlobalSyms::GlobalSym>, v_1d<std::pair<Operators::SymGenerators, int>>> UI::createSymmetries()
{
	v_1d<GlobalSyms::GlobalSym> _glbSyms					= {};
	v_1d<std::pair<Operators::SymGenerators, int>> _locSyms = {};
	if (this->symP.S_ == true)
	{
		// create Hilbert space
		this->isComplex_	= this->symP.checkComplex(this->latP.lat->get_Ns());

		// ------ LOCAL ------
		_locSyms			= this->symP.getLocGenerator();

		// ------ GLOBAL ------
		// check U1
		if (this->symP.U1_ != -INT_MAX)
			_glbSyms.push_back(GlobalSyms::getU1Sym(this->latP.lat, this->symP.U1_));
	};
	return std::make_pair(_glbSyms, _locSyms);
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

/*
* @brief A placeholder for making the simulation with NQS.
*/
void UI::makeSimNQS()
{
	this->useComplex_ = true;
	this->defineModels(true);
	this->defineNQS<cpx>(this->hamComplex, this->nqsCpx);
	this->nqsSingle(this->nqsCpx);

}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::makeSymQuadraticManifold()
{
	// reset Hamiltonians - memory release
	this->resetQuadratic();
	this->useComplex_ = true;
	if (this->defineModelsQ(true))
		this->quadraticStatesManifold<cpx>(this->qhamComplex);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::makeSimETH()
{
	// define the models
	this->resetEd();
	// force complex Hamiltonian
	this->useComplex_ = false;
	// simulate
	if (this->defineModels(false, false, false))
	{
		if (this->chosenFun == 40)
			this->checkETH(this->hamDouble);
		else if (this->chosenFun == 42)
			this->checkETH_statistics(this->hamDouble);
	}
}

void UI::makeSimETHSweep()
{
	// steps for alpha
	double _alpha = this->modP.qsm.qsm_alpha_[0];
	// get the random seed for this realization
	auto seed [[maybe_unused]] = std::random_device()();

	while (_alpha <= 1.0)
	{
		//this->ran_.newSeed(seed);

		// set the alpha
		for(int i = 0; i < this->modP.qsm.qsm_alpha_.size(); i++)
			this->modP.qsm.qsm_alpha_[i] = _alpha;

		// define the models
		this->resetEd();
		// force complex Hamiltonian
		this->useComplex_ = false;
		// simulate
		if (this->defineModels(false, false, false))
		{
			if (this->chosenFun == 41)
				this->checkETH(this->hamDouble);
			else if (this->chosenFun == 43)
				this->checkETH_statistics(this->hamDouble);
		}
		_alpha += 0.02;
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::checkETH_statistics(std::shared_ptr<Hamiltonian<double>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

	// check the random field
	auto _rH	= this->modP.qsm.qsm_h_r_;
	auto _rA	= this->modP.qsm.qsm_h_ra_;
	size_t _Ns	= this->modP.qsm.qsm_Ntot_;
	u64 _Nh		= ULLPOW(_Ns);

	// get info
	std::string modelInfo	= _H->getInfo();
	std::string randomStr	= FileParser::appWRandom("", _H->ran_);
	std::string dir			= makeDirsC(this->mainDir, "QSM_MAT_ELEM_STAT", modelInfo, (_rH != 0) ? VEQV(dh, _rA) + "_" + STRP(_rH, 3) : "");
	std::string extension	= ".h5";

	// set seed
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	// set the placeholder for the values to save (will save only the diagonal elements and other measures)
	v_1d<arma::Mat<double>> _ent	= v_1d<arma::Mat<double>>(_Ns, -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones));
	arma::Mat<double> _en			= -1e5 * arma::Mat<double>(_H->getHilbertSize(), this->modP.modRanN_, arma::fill::zeros);
	arma::Mat<double> _gaps			= -1e5 * arma::Col<double>(this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _ipr1			= -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _ipr2			= -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones);

	// choose the random position inside the dot for the correlation operators
	uint _pos						= this->ran_.randomInt(0, this->modP.qsm.qsm_N_);

	// create the operators
	auto _sx						= Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, _Ns - 1);
	auto _sxc						= Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, { _pos, (uint)_Ns - 1 });
	auto _szc						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { _pos, (uint)_Ns - 1 });
	//auto _sp_sm

	// for each site
	std::vector<Operators::Operator<double>> _sz_is;
	std::vector<std::string> _sz_i_names;

	// create the diagonal operators for spin z at each side
	for (auto i = 0; i < _Ns; ++i)
	{
		_sz_is.push_back(Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, i));
		_sz_i_names.push_back("sz_" + STR(i));
	}

	// create the matrices
	v_1d<Operators::Operator<double>> _ops	= { _sx, _sxc, _szc			};
	v_1d<std::string> _opsN					= { "sx_l", "sx_c", "sz_c"	};

	// matrices
	for (auto i = 0; i < _Ns; ++i)
	{
		_ops.push_back(_sz_is[i]);
		_opsN.push_back(_sz_i_names[i]);
	}

	// create the measurement class
	Measurement<double> _measure(this->modP.qsm.qsm_Ntot_, dir, _ops, _opsN);
	_measure.initializeMatrices(_Nh);

	// to save the operators (those elements will be stored for each operator separately)
	// a given matrix element <n|O|n> will be stored in i'th column of the i'th operator
	// the n'th row in the column will be the state index
	// the columns corresponds to realizations of disorder
	v_1d<arma::Mat<double>> _diagElems(_ops.size(), -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones));

	// (mean, typical, mean2, typical2, gaussianity, kurtosis, binder cumulant)
	// the columns will correspond to realizations
	u64 _hs_fractions_diag		= SystemProperties::hs_fraction_diagonal_cut(this->modP.modMidStates_, _Nh);
	v_1d<arma::Mat<double>> _diagElemsStat(_ops.size(), arma::Mat<double>(7, this->modP.modRanN_, arma::fill::zeros));
	
	const double off_extensive_threshold	= 1e-1;
	const double off_constant_threshold_up	= off_extensive_threshold;
	const double off_constant_threshold_dn	= 1e-2;
	const double off_vanishing_threshold	= std::min(5e-1 / _Ns, off_constant_threshold_dn);

	// (mean, typical, mean2, typical2, mean4, meanabs, gaussianity, binder cumulant)
	v_1d<arma::Mat<double>> _offdiagElemesStat(_ops.size(), arma::Mat<double>(8, this->modP.modRanN_, arma::fill::zeros));
	v_1d<arma::Mat<double>> _offdiagElemesStatConst(_ops.size(), arma::Mat<double>(8, this->modP.modRanN_, arma::fill::zeros));
	v_1d<arma::Mat<double>> _offdiagElemesStatVanish(_ops.size(), arma::Mat<double>(8, this->modP.modRanN_, arma::fill::zeros));
	v_1d<arma::Mat<double>> _offdiagElemesStatExtensive(_ops.size(), arma::Mat<double>(8, this->modP.modRanN_, arma::fill::zeros));

	// saves the histograms of the second moments for the offdiagonal elements 
	v_1d<HistogramAverage<double>> _histAv(_ops.size(), HistogramAverage<double>());
	v_1d<HistogramAverage<double>> _histAvTypical(_ops.size(), HistogramAverage<double>());
	
	// ----------------------- nbins operators -----------------------
	v_1d<Histogram> _histOperatorsDiag(_ops.size(), Histogram());
	v_1d<Histogram> _histOperatorsOffdiag(_ops.size(), Histogram());
	v_1d<Histogram> _histOperatorsOffdiagVanishing(_ops.size(), Histogram());
	v_1d<Histogram> _histOperatorsOffdiagConst(_ops.size(), Histogram());
	v_1d<Histogram> _histOperatorsOffdiagExtensive(_ops.size(), Histogram());
	uint _nbinOperators = 15 * _Ns;
	for (uint _opi = 0; _opi < _ops.size(); ++_opi) 
	{
		// diagonal
		_histOperatorsDiag[_opi].reset(_nbinOperators);
		_histOperatorsDiag[_opi].uniform(0.5, -0.5); 
		// offdiagonal
		_histOperatorsOffdiag[_opi].reset(_nbinOperators);
		_histOperatorsOffdiagConst[_opi].reset(_nbinOperators);
		_histOperatorsOffdiagVanishing[_opi].reset(_nbinOperators);
		_histOperatorsOffdiagExtensive[_opi].reset(_nbinOperators);
		double _offdiagLimit	= 0.5 - 0.025 * _Ns;
		_histOperatorsOffdiag[_opi].uniform(_offdiagLimit, -_offdiagLimit);
		_histOperatorsOffdiagExtensive[_opi].uniform(_offdiagLimit, -_offdiagLimit);
		_offdiagLimit			*= 0.8;
		_histOperatorsOffdiagConst[_opi].uniform(_offdiagLimit, -_offdiagLimit);
		_histOperatorsOffdiagVanishing[_opi].uniform(_offdiagLimit, -_offdiagLimit);

	}

	// create the saving function
	std::function<void(uint)> _saver = [&](uint _r)
		{
			// save the matrices
			for(int i = 0; i < _Ns; ++i)
			{
				// save the entropies (only append when i > 0)
				saveAlgebraic(dir, "entro" + randomStr + extension, _ent[i], STR(i), i > 0);
			}

			// save the iprs and the energy to the same file
			saveAlgebraic(dir, "stat" + randomStr + extension, _gaps, "gap_ratio", false);
			// participation ratio
			saveAlgebraic(dir, "stat" + randomStr + extension, _ipr1, "part_entropy_q=1", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _ipr2, "part_entropy_q=2", true);
			// energy
			saveAlgebraic(dir, "stat" + randomStr + extension, _en, "energy", true);

			// append statistics from the diagonal elements
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "stat" + randomStr + extension, _diagElemsStat[_opi], "diag_" + _name, true);

				// offdiagonal elements
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStat[_opi], "offdiag_" + _name, true);
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStatConst[_opi], "offdiag_" + _name + "_const", true);
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStatVanish[_opi], "offdiag_" + _name + "_vanish", true);
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStatExtensive[_opi], "offdiag_" + _name + "_extensive", true);
			}

			// diagonal elements (only append when _opi > 0)
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "diag" + randomStr + extension, _diagElems[_opi], _name, _opi > 0);
			}

			// save the histograms
			saveAlgebraic(dir, "hist" + randomStr + extension, _histAv[0].edgesCol(), "omegas", false);
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "hist" + randomStr + extension, _histAv[_opi].averages_av(), _name + "_mean", true);
				saveAlgebraic(dir, "hist" + randomStr + extension, _histAvTypical[_opi].averages_av(true), _name + "_typical", true);
			}
			saveAlgebraic(dir, "hist" + randomStr + extension, _histAv[0].countsCol(), "_counts", true);

			// save the distributions of the operators
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsDiag[_opi].edgesCol(), _name + "_diag_edges", _opi > 0);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsDiag[_opi].countsCol(), _name + "_diag_counts", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag[_opi].edgesCol(), _name + "_offdiag_edges", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag[_opi].countsCol(), _name + "_offdiag_counts", true);
				// additional
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag[_opi].edgesCol(), _name + "_offdiag_edges_vanish", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiagConst[_opi].countsCol(), _name + "_offdiag_counts_const", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiagVanishing[_opi].countsCol(), _name + "_offdiag_counts_vanish", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiagExtensive[_opi].countsCol(), _name + "_offdiag_counts_extensive", true);
			}

			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
		};

	// go through realizations
	for (int _r = 0; _r < this->modP.modRanN_; ++_r)
	{
		// ----------------------------------------------------------------------------
		
		// checkpoints etc
		{
			LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
			_timer.checkpoint(STR(_r));

			// -----------------------------------------------------------------------------

			LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
			_H->clearH();
			_H->randomize(this->modP.qsm.qsm_h_ra_, _rH, { "h" });

			// -----------------------------------------------------------------------------

			// set the Hamiltonian
			_H->buildHamiltonian();
			_H->diagH(false);
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
		}

		// -----------------------------------------------------------------------------
				
		// get the average energy index and the points around it on the diagonal
		u64 _minIdxDiag, _maxIdxDiag = 0; 

		// set
		{
			std::tie(_minIdxDiag, _maxIdxDiag)		= _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
		}

		// -----------------------------------------------------------------------------
		
		// set the uniform distribution of frequencies in logspace!!!
		if (_r == 0)
		{

			double oMax			= std::abs(_H->getEigVal(_maxIdxDiag) - _H->getEigVal(_minIdxDiag)) * 2.5;
			double oMin			= 1.0l / _Nh / 10;
			u64 _nFrequencies	= 12 * _Ns;

			// set the histograms
			for (auto iHist = 0; iHist < _ops.size(); ++iHist)
			{
				_histAv[iHist].reset(_nFrequencies);
				_histAv[iHist].uniformLog(oMax, oMin);

				_histAvTypical[iHist].reset(_nFrequencies);
				_histAvTypical[iHist].uniformLog(oMax, oMin);
			}
		}

		// -----------------------------------------------------------------------------

		// save the energies
		{
			_en.col(_r) = _H->getEigVal();
		}

		// -----------------------------------------------------------------------------
		
		// calculator of the properties
		{
			// -----------------------------------------------------------------------------
			
			// gap ratios
			{
				// calculate the eigenlevel statistics
				_gaps(_r)					= SystemProperties::eigenlevel_statistics(arma::Col<double>(_H->getEigVal().subvec(_minIdxDiag, _maxIdxDiag - 1)));
				LOGINFO(StrParser::colorize(VEQ(_gaps(_r, 0)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
				LOGINFO(_timer.point(STR(_r)), "Gap ratios", 1);
			}

			// -----------------------------------------------------------------------------
			
			// other measures
			{
				// participation ratios
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif							
				for (long long _start = 0; _start < _Nh; ++_start)
				{
					_ipr1(_start, _r) = SystemProperties::information_entropy(_H->getEigVec(_start));
					_ipr2(_start, _r) = std::log(1.0 / SystemProperties::participation_ratio(_H->getEigVec(_start), 2.0));
				}

				// -----------------------------------------------------------------------------
				
				// entanglement entropy
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif			
				for (long long _start = 0; _start < _Nh; ++_start)
				{
					for (int i = 1; i <= _Ns; i++)
					{
						// calculate the entanglement entropy
						//_entr(_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), _Ns - 1, _Hs);
						//_entr(_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), _Ns, _Hs);
						uint _maskA	= 1 << (i - 1);
						uint _enti	= _Ns - i;
						_ent[_enti](_start, _r) = Entropy::Entanglement::Bipartite::vonNeuman<double>(_H->getEigVec(_start), 1, _Ns, _maskA, DensityMatrix::RHO_METHODS::SCHMIDT, 2);
					}
				}
			}

			// -----------------------------------------------------------------------------
			
			// diagonal
			{
				for (u64 _start = 0; _start < _Nh; ++_start)
				{
					// calculate the diagonal elements
					const auto& _measured = _measure.measureG(_H->getEigVec(_start));
					// save the diagonal elements
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
					for (int i = 0; i < _measured.size(); ++i)
					{
						const auto _elem			= _measured[i];
						_diagElems[i](_start, _r)	= _elem;

						if (_start >= _minIdxDiag && _start < _maxIdxDiag)
						{
							// add element to the histogram
							_histOperatorsDiag[i].append(_elem);

							auto _elem2				= _elem * _elem;
							auto _logElem			= std::log(std::fabs(_elem));
							auto _logElem2			= std::log(_elem * _elem);

							// save the statistics
							// mean
							_diagElemsStat[i](0, _r) += _elem;
							// typical
							_diagElemsStat[i](1, _r) += _logElem;
							// mean2
							_diagElemsStat[i](2, _r) += _elem2;
							// typical2
							_diagElemsStat[i](3, _r) += _logElem2;
							
						}
					}
				}
				
				// finalize statistics
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (int i = 0; i < _ops.size(); ++i)
				{
					for (uint ii = 0; ii < 4; ii++)
					{
						_diagElemsStat[i](ii, _r)		/= _hs_fractions_diag;
					}
					// save the statistics
					_diagElemsStat[i](4, _r) = StatisticalMeasures::gaussianity(_diagElems[i].col(_r));
					_diagElemsStat[i](5, _r) = StatisticalMeasures::kurtosis(_diagElems[i].col(_r));
					_diagElemsStat[i](6, _r) = StatisticalMeasures::binder_cumulant(_diagElems[i].col(_r));
				}

				// additionally, for typical values, calculate the exponential of the mean
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (int i = 0; i < _ops.size(); ++i)
				{
					_diagElemsStat[i](1, _r) = std::exp(_diagElemsStat[i](1, _r));
					_diagElemsStat[i](3, _r) = std::exp(_diagElemsStat[i](3, _r));
				}
			}

			// -----------------------------------------------------------------------------
			
			// offdiagonal
			{
				u64 _constantIterator	= 0;
				u64 _vanishingIterator	= 0;
				u64 _extensiveIterator	= 0;
				u64 _totalIterator		= 0;
				const double _avEn		= _H->getEnAv();
				// go through the whole spectrum (do not save pairs, only one element as it's Hermitian.
				for (u64 i = _minIdxDiag; i < _maxIdxDiag; ++i)
				{
					auto _en_l = _H->getEigVal(i);
					for (u64 j = i + 1; j < _maxIdxDiag; ++j)
					{
						auto _en_r = _H->getEigVal(j);
						
						// check the energy difference
						if (!SystemProperties::hs_fraction_close_mean(_en_l, _en_r, _avEn, this->modP.modEnDiff_))
							continue;

						_totalIterator++;

						// calculate the frequency
						const double w				= std::abs(_en_l - _en_r);

						// calculate the values
						const auto& _measured		= _measure.measureG(_H->getEigVec(i), _H->getEigVec(j));

						// checks for means
						const bool _isvanishing		= w < off_vanishing_threshold;
						const bool _isconstant		= w < off_constant_threshold_up && w > off_constant_threshold_dn;
						const bool _isextensive		= w > off_extensive_threshold;

						_constantIterator			+= _isconstant;
						_vanishingIterator			+= _isvanishing;
						_extensiveIterator			+= _isextensive;

						// save the off-diagonal elements
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
						for (int i = 0; i < _ops.size(); ++i)
						{
							auto _elem				= _measured[i];
							auto _elem2				= _elem * _elem;
							auto _logElem			= std::log(std::abs(_elem));
							auto _logElem2			= std::log(_elem * _elem);

							// mean
							_offdiagElemesStat[i](0, _r) += _elem;
							// typical
							_offdiagElemesStat[i](1, _r) += _logElem;
							// mean2
							_offdiagElemesStat[i](2, _r) += _elem2;
							// typical2
							_offdiagElemesStat[i](3, _r) += _logElem2;
							// mean4
							_offdiagElemesStat[i](4, _r) += _elem2 * _elem2;
							// meanabs
							_offdiagElemesStat[i](5, _r) += std::abs(_elem);

							// check binned statistics
							if (_isextensive)
							{
								// mean
								_offdiagElemesStatExtensive[i](0, _r) += _elem;
								// typical
								_offdiagElemesStatExtensive[i](1, _r) += _logElem;
								// mean2
								_offdiagElemesStatExtensive[i](2, _r) += _elem2;
								// typical2
								_offdiagElemesStatExtensive[i](3, _r) += _logElem2;
								// mean4
								_offdiagElemesStatExtensive[i](4, _r) += _elem2 * _elem2;
								// meanabs
								_offdiagElemesStatExtensive[i](5, _r) += std::abs(_elem);
								//histogram
								_histOperatorsOffdiagExtensive[i].append(_elem);
							}
							if (_isconstant)
							{
								// mean
								_offdiagElemesStatConst[i](0, _r) += _elem;
								// typical
								_offdiagElemesStatConst[i](1, _r) += _logElem;
								// mean2
								_offdiagElemesStatConst[i](2, _r) += _elem2;
								// typical2
								_offdiagElemesStatConst[i](3, _r) += _logElem2;
								// mean4
								_offdiagElemesStatConst[i](4, _r) += _elem2 * _elem2;
								// meanabs
								_offdiagElemesStatConst[i](5, _r) += std::abs(_elem);
								// histogram
								_histOperatorsOffdiagConst[i].append(_elem);
							}
							if (_isvanishing)
							{
								// mean
								_offdiagElemesStatVanish[i](0, _r) += _elem;
								// typical
								_offdiagElemesStatVanish[i](1, _r) += _logElem;
								// mean2
								_offdiagElemesStatVanish[i](2, _r) += _elem2;
								// typical2
								_offdiagElemesStatVanish[i](3, _r) += _logElem2;
								// mean4
								_offdiagElemesStatVanish[i](4, _r) += _elem2 * _elem2;
								// meanabs
								_offdiagElemesStatVanish[i](5, _r) += std::abs(_elem);
								// histogram
								_histOperatorsOffdiagVanishing[i].append(_elem);
							}

							// add to value histogram
							_histOperatorsOffdiag[i].append(_elem);

							// add to the histograms
							_histAv[i].append(w, _elem2);
							_histAvTypical[i].append(w, _logElem2);
						}
					}
				}
				
				// finalize statistics
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (int i = 0; i < _ops.size(); ++i)
				{
					for (uint ii = 0; ii < 6; ii++)
					{
						_offdiagElemesStat[i](ii, _r) /= (long double)_totalIterator;
						_offdiagElemesStatConst[i](ii, _r) /= (long double)_constantIterator;
						_offdiagElemesStatVanish[i](ii, _r) /= (long double)_vanishingIterator;
						_offdiagElemesStatExtensive[i](ii, _r) /= (long double)_extensiveIterator;
					}

					// statistics
					_offdiagElemesStat[i](6, _r) = StatisticalMeasures::gaussianity(_offdiagElemesStat[i](5, _r), _offdiagElemesStat[i](2, _r));
					_offdiagElemesStat[i](7, _r) = StatisticalMeasures::binder_cumulant(_offdiagElemesStat[i](2, _r), _offdiagElemesStat[i](4, _r));

					_offdiagElemesStatConst[i](6, _r) = StatisticalMeasures::gaussianity(_offdiagElemesStatConst[i](5, _r), _offdiagElemesStatConst[i](2, _r));
					_offdiagElemesStatConst[i](7, _r) = StatisticalMeasures::binder_cumulant(_offdiagElemesStatConst[i](2, _r), _offdiagElemesStatConst[i](4, _r));

					_offdiagElemesStatVanish[i](6, _r) = StatisticalMeasures::gaussianity(_offdiagElemesStatVanish[i](5, _r), _offdiagElemesStatVanish[i](2, _r));
					_offdiagElemesStatVanish[i](7, _r) = StatisticalMeasures::binder_cumulant(_offdiagElemesStatVanish[i](2, _r), _offdiagElemesStatVanish[i](4, _r));
				}
				
				// additionally, for typical values, calculate the exponential of the mean
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (int i = 0; i < _ops.size(); ++i)
				{
					for (auto ii : { 1, 3 })
					{
						_offdiagElemesStat[i](ii, _r) = std::exp(_offdiagElemesStat[i](ii, _r));
						_offdiagElemesStatConst[i](ii, _r) = std::exp(_offdiagElemesStatConst[i](ii, _r));
						_offdiagElemesStatVanish[i](ii, _r) = std::exp(_offdiagElemesStatVanish[i](ii, _r));
						_offdiagElemesStatExtensive[i](ii, _r) = std::exp(_offdiagElemesStatExtensive[i](ii, _r));
					}
				}

			}
		}

		// save the checkpoints
		if (_Ns >= 14 && _r % 2 == 0)
		{
			// save the diagonals
			_saver(_r);
		}
		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_saver(this->modP.modRanN_);

	// save the command directly to the file
	{
		std::string dir_in = makeDirs(this->mainDir, "QSM_MAT_ELEM");
		std::ofstream file(dir_in + "qsm_scp_newest.log", std::ios::app);
		file << "scp -3 -r scp://klimak97@ui.wcss.pl:22//" + dir + " ./" << std::endl;
		file.close();
		std::ofstream file_rsync(dir_in + "qsm_rsync.log", std::ios::app);
		file_rsync << "rsync -rv --rsh --ignore-existing -e 'ssh -p 22' klimak97@ui.wcss.pl:mylustre-hpc-maciek/QSolver/DATA_LJUBLJANA_BIG_SWEEP ./";
		file_rsync.close();
	}

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

// ------------------------------------------------