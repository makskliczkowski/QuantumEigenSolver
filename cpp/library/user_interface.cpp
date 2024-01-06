#include "include/user_interface/user_interface.h"
int LASTLVL = 0;

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
		SETOPTIONV(nqsP,	nMcSteps,	"m"	);
		SETOPTIONV(nqsP,	blockSize,	"bs"	);
		SETOPTIONV(nqsP,	nBlocks,		"nb"	);
		SETOPTIONV(nqsP,	nHidden,		"nh"	);
		SETOPTIONV(nqsP,	nFlips,		"nf"	);
		SETOPTIONV(nqsP,	type,			"nqst");
		SETOPTIONV(nqsP,	nTherm,		"nt"	);
		SETOPTION(nqsP,	lr						);
		SETOPTIONV(nqsP,	loadNQS,		"lNQS");
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
		// ------ ising ------
		SETOPTION_STEP(modP, J1);
		SETOPTION_STEP(modP, hx);
		SETOPTION_STEP(modP, hz);
		// ---- heisenberg ---
		{
			this->modP.heiDlt_.resize(Ns);
			this->modP.heiJ_.resize(Ns);
			this->modP.heiHx_.resize(Ns);
			this->modP.heiHz_.resize(Ns);
			SETOPTION(modP, heiJ);
			SETOPTION(modP, heiDlt);
			SETOPTION(modP, heiHz);
			SETOPTION(modP, heiHx);
			SETOPTION_STEP(modP, dlt1);
		}
		// ------- xyz -------
		SETOPTION_STEP(modP, J2);
		SETOPTION_STEP(modP, dlt2);
		SETOPTION_STEP(modP, eta1);
		SETOPTION_STEP(modP, eta2);
		// ------ kitaev -----
		{
			// resize
			this->modP.Kx_.resize(Ns);
			this->modP.Ky_.resize(Ns);
			this->modP.Kz_.resize(Ns);
			// set options
			SETOPTION(modP, Kx);
			SETOPTION(modP, Ky);
			SETOPTION(modP, Kz);
		}

	}
	// --------------- QUARDRATIC ----------------
	{
		SETOPTIONV(modP, modTypQ, "modQ");
		// -- aubry-andres ---
		SETOPTION_STEP(modP, Beta);
		SETOPTION_STEP(modP, Phi);
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
	{
		this->setOption(this->quiet, argv, "q");
		this->setOption(this->threadNum, argv, "th");
		// later function choice
		this->setOption(this->chosenFun, argv, "fun");
	}
	// ---------------- DIRECTORY ----------------
	bool setDir [[maybe_unused]] = this->setOption(this->mainDir, argv, "dir");
	this->mainDir = makeDirsC(fs::current_path().string(), "DATA", this->mainDir);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief chooses the method to be used later based on input -fun argument
*/
void UI::funChoice()
{
	LOGINFO_CH_LVL(0);
	LOGINFO("USING #THREADS=" + STR(this->threadNum), LOG_TYPES::CHOICE, 1);
	this->_timer.reset();
	switch (this->chosenFun)
	{
	case -1:
		// default case of showing the help
		this->exitWithHelp();
		break;
		// ------------------------------- NEURAL QST -------------------------------
	case 11:
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
		this->makeSimQuadratic();
		break;
	case 31:
		// this option makes the quadratic Hamiltonian, takes the state from it and recreates a many body state
		LOGINFO("SIMULATION: QUADRATIC HAMILTONIAN - TRANSFORM TO MANY BODY", LOG_TYPES::CHOICE, 1);;
		this->makeSimQuadraticToManyBody();
		break;
	default:
		// default case of showing the help
		this->exitWithHelp();
		break;
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief defines the models based on the input parameters - interacting
*/
bool UI::defineModels(bool _createLat) {
	BEGIN_CATCH_HANDLER
	{
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
	}
	END_CATCH_HANDLER("Exception in setting the lattices: ", ;);

	// check if is complex
	this->isComplex_ = this->symP.checkComplex(this->latP.lat->get_Ns());
	LOGINFO("Making : " + std::string((this->isComplex_) ? " complex" : " real"), LOG_TYPES::INFO, 3);

	// check if is complex and define the Hamiltonian
	bool _ok;
	if (this->isComplex_ || this->useComplex_)
		_ok = this->defineModel(this->hilComplex, this->hamComplex);
	else
		_ok = this->defineModel(this->hilDouble, this->hamDouble);

	return _ok;

}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief defines the models based on the input parameters - quadratic
*/
bool UI::defineModelsQ(bool _createLat)
{
	// create lattice
	LOGINFO("Making quadratic model!", LOG_TYPES::INFO, 1);

	BEGIN_CATCH_HANDLER
	{
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
	}
	END_CATCH_HANDLER("Exception in setting the lattices: ", ;);

	// check if is complex
	this->isComplex_ = this->modP.checkComplex();
	LOGINFO("Making : " + std::string((this->isComplex_) ? " complex" : " real"), LOG_TYPES::INFO, 2);

	// check if is complex and define the Hamiltonian
	bool _ok;
	if (this->isComplex_ || this->useComplex_)
		_ok = this->defineModelQ<cpx>(this->qhamComplex);
	else
		_ok = this->defineModelQ<double>(this->qhamDouble);

	return _ok;

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
	// reset Hamiltonians - memory release
	if (this->hamComplex)
		this->hamComplex.reset();
	if (this->hamDouble)
		this->hamDouble.reset();
	// define the models
	if (!this->defineModels(true))
		return;
	if (this->isComplex_)
		this->symmetries(this->hamComplex, _diag, _states);
	else
		this->symmetries(this->hamDouble, _diag, _states);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Use the simulation and Haar matrices to calculate the degeneracies and entropies
*/
void UI::makeSimSymmetriesDeg()
{
	// reset Hamiltonians - memory release
	if (this->hamComplex)
		this->hamComplex.reset();
	if (this->hamDouble)
		this->hamDouble.reset();
	// define the models
	this->useComplex_ = true;
	if (!this->defineModels(true))
		return;
	this->symmetriesDeg();
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Use the simulation and CUE matrices to calculate entanglement entropy of multiple degeneracies
*/
void UI::makeSimSymmetriesCreateDeg()
{
	// reset Hamiltonians - memory release
	if (this->hamComplex)
		this->hamComplex.reset();
	if (this->hamDouble)
		this->hamDouble.reset();
	// define the models
	this->useComplex_ = true;
	if (!this->defineModels(true))
		return;
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
	this->defineModels();
	if (this->isComplex_) 
	{
		this->defineNQS<cpx>(this->hamComplex, this->nqsCpx);
		this->nqsSingle(this->nqsCpx);
	}
	else
	{
		this->defineNQS<double>(this->hamDouble, this->nqsDouble);
		this->nqsSingle(this->nqsDouble);
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief make simulation with quadratic states mixing
*/
void UI::makeSimQuadratic()
{
	// reset Hamiltonians - memory release
	if (this->qhamComplex)
		this->qhamComplex.reset();
	if (this->qhamDouble)
		this->qhamDouble.reset();
	// define the models
	if (!this->defineModelsQ(true))
		return;
	if (this->isComplex_)
		this->quadraticStatesMix<cpx>(this->qhamComplex);
	else
		this->quadraticStatesMix<double>(this->qhamDouble);
}

/*
* @brief Transform quadratic to many body
*/
void UI::makeSimQuadraticToManyBody()
{
	// reset Hamiltonians - memory release
	if (this->qhamComplex)
		this->qhamComplex.reset();
	if (this->qhamDouble)
		this->qhamDouble.reset();
	// define the models
	this->useComplex_	= true;
	this->isComplex_	= true;
	if (!this->defineModelsQ(true))
		return;
	this->quadraticStatesToManyBody<cpx>(this->qhamComplex);
}