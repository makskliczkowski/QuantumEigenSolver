#include "../../include/user_interface/user_interface.h"

// ##########################################################################################################################################

// ########################################################## S Y M M E T R I E S ###########################################################

// ##########################################################################################################################################

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
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

// ##########################################################################################################################################

/**
* @brief Computes the Hamiltonian with symmetries and saves the entanglement entropies.
* @param start time of the beginning
* @param _H the shared pointer to the Hamiltonian - for convinience of the usage
* @info used in https://arxiv.org/abs/2303.13577
*/
template<typename _T>
void UI::symmetries(std::shared_ptr<Hamiltonian<_T>> _H, bool _diag, bool _states)
{
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	_timer.reset();
	u64 Nh					=	_H->getHilbertSize();
	// create the directories
	std::string dir			=	makeDirsC(this->mainDir, _H->getType(), this->latP.lat->get_info());

	// use those files 
	std::string modelInfo	=	_H->getInfo();
	std::string logMe		=	"";
	std::string filename	=	dir + modelInfo;
	std::ofstream ofs(dir + "logHilbert.dat", std::ios_base::out | std::ios_base::app);
	auto memory				=	_H->getHamiltonianSizeH();
	strSeparatedS(logMe, ',', modelInfo, Nh, STRP(memory, 5));
	ofs << logMe << EL;
	ofs.close();

	// check Hilbert size or whether we should diagonalize and proceed further
	if (Nh == 0 || !_diag)
	{
		LOGINFO("Skipping creation of the Hamiltonian due to signal or empty Hilbert space.", LOG_TYPES::FINISH, 0);
		return;
	}

	// set the Hamiltonian
	_H->buildHamiltonian();

	// set the parameters
	uint Ns					=	_H->getNs();
	u64 stateNum			=	Nh;
	bool useShiftAndInvert	=	false;

	if (Nh < UI_LIMITS_MAXFULLED) 
	{
		_H->diagH(false);
	}
	else
	{
		LOGINFO("Using S&I", LOG_TYPES::TRACE, 2);
		useShiftAndInvert	=	true;
		stateNum			=	UI_LIMITS_SI_STATENUM;
		_H->diagH(false, (int)stateNum, 0, 1000, 1e-5, "sa");
	}
	LOGINFO(_timer.point(0), "Diagonalization.", 2);

	std::string name		=	VEQ(Nh);
	LOGINFO("Spectrum size: " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states: " + STR(stateNum), LOG_TYPES::TRACE, 2);

	// save .h5
	_H->getEigVal(dir, HAM_SAVE_EXT::h5, false);

	// clear energies
	_H->clearEigVal();
	_H->clearH();

	// iterate through bond cut
	const uint maxBondNum	=	Ns / 2;
	arma::mat ENTROPIES(maxBondNum, stateNum, arma::fill::zeros);

	// get the symmetry rotation matrix
	auto _symRot			=	_H->getSymRot();
	bool usesSym			=	_H->hilbertSpace.checkSym();

	// set which bonds we want to cut in bipartite
	v_1d<uint> _bonds		=	{};
	for (int i = 1; i <= maxBondNum; i++)
		_bonds.push_back(i);
	_bonds					=	{ maxBondNum };

	// go entropies!
	_timer.checkpoint("entropies");
	_H->generateFullMap();
	LOGINFO(_timer.point(1), "Full map generation.", 3);

	pBar pbar(5, stateNum, _timer.point(1));
#pragma omp parallel for num_threads(this->threadNum)
	for (auto idx = 0LL; idx < stateNum; idx++) {
		// get the eigenstate
		arma::Col<_T> state = _H->getEigVec(idx);
		if (usesSym)
			state			= _symRot * state;
		// go through bonds
		for (auto i : _bonds) {
			// iterate through the state
			auto entro		= Entropy::Entanglement::Bipartite::vonNeuman<_T>(state, i, _H->hilbertSpace);
			// save the entropy
			ENTROPIES(i - 1, idx) = entro;
		}
		// update progress
		PROGRESS_UPD(idx, pbar, "PROGRESS");
	}
	LOGINFO(_timer.point(1), "Entropies for all eigenstates using SCHMIDT.", 2);

	// save entropies file
	ENTROPIES.save(arma::hdf5_name(filename + ".h5", "entropy", arma::hdf5_opts::append));

	if (useShiftAndInvert)
		return;

	if (_states)
		_H->getEigVec(dir, stateNum, HAM_SAVE_EXT::h5, true);
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
* @brief Tests the currently implemented symmetries based on specific model parameters
* @param start the timer start
*/
void UI::symmetriesTest()
{
	using symTuple				= std::tuple<int, int, int, int, int, int>;
	v_1d<double> entFULL		= {};
	v_1d<double> entSYMS		= {};
	v_1d<double> enSYMS			= {};
	v_1d<symTuple> _symmetries	= {};
	v_1d<u64> _sizesSyms		= {};
	v_1d<std::pair<symTuple, std::map<int, int>>> _degs = {};
	v_1d<u64> _sizes			= {};
	uint Ns						= 1;
	u64 Nh						= 1;
	u64 NhFull					= 1;
	uint La						= Ns / 2;
	std::string dir				= "";
	std::string infoHFull		= "";
	arma::mat ENTROPIES;
	// ---------------------------------- DIAG HAMILTONIAN WITHOUT SYMMETRIES ---------------------------------
	for (auto bc : { 1, 0 }) {
		this->latP.bc_ = bc == 0 ? BoundaryConditions::PBC : BoundaryConditions::OBC;
		this->symP.S_ = false;
		if (this->hamDouble)
			this->hamDouble.reset();
		if (this->latP.lat)
			this->latP.lat.reset();
		this->defineModels(true);
		this->isComplex_ = false;
		this->defineModel(this->hilDouble, this->hamDouble);
		Nh = this->hamDouble->getHilbertSize();
		NhFull = Nh;
		Ns = this->latP.lat->get_Ns();
		La = Ns / 2;

		LOGINFO("Started building full Hamiltonian", LOG_TYPES::TRACE, 1);
		LOGINFO_CH_LVL(2);
		this->hamDouble->buildHamiltonian();
		dir = this->mainDir + kPS + this->hamDouble->getType() + kPS + VEQ(Ns) + kPS + getSTR_BoundaryConditions(this->latP.bc_) + kPS;
		infoHFull = this->hamDouble->getInfo();
		std::string filename = dir + infoHFull;

		LOGINFO("Finished buiding Hamiltonian" + infoHFull, LOG_TYPES::TRACE, 2);
		LOGINFO("Using standard diagonalization", LOG_TYPES::TRACE, 3);
		this->hamDouble->diagH(false);
		LOGINFO("Finished diagonalizing Hamiltonian", LOG_TYPES::TRACE, 2);

		// Save the energies for calculating full Hamiltonian
		createDir(dir);
		this->hamDouble->getEigVal(dir, HAM_SAVE_EXT::dat, false);
		this->hamDouble->getEigVal(dir, HAM_SAVE_EXT::h5, false);
		if (Ns < 10)
		{
			this->hamDouble->getEigVec(dir, Nh, HAM_SAVE_EXT::dat, false);
			this->hamDouble->getEigVec(dir, Nh, HAM_SAVE_EXT::h5, false);
		}

		LOGINFO("Started entropies for full Hamiltonian", LOG_TYPES::TRACE, 1);
		// Save the entropies 
		ENTROPIES.zeros(La, Nh);
		for (long long idx = 0; idx < Nh; idx++) {
			// get the eigenstate
			arma::Col<double> state = this->hamDouble->getEigVec(idx);
			for (uint i = 1; i <= La; i++) {
				// iterate through the state
				auto entro = Entropy::Entanglement::Bipartite::vonNeuman<double>(state, i, this->hamDouble->hilbertSpace);
				// save the entropy
				ENTROPIES(i - 1, idx) = entro;
			}
		}
		LOGINFO("Finished Entropies for full Hamiltonian", LOG_TYPES::TRACE, 2);
		ENTROPIES.save(arma::hdf5_name(filename + ".h5", "entropy", arma::hdf5_opts::append));
		ENTROPIES.save(filename + ".dat", arma::arma_ascii);
		LOGINFO("--------------------------- FINISHED ---------------------------\n", LOG_TYPES::FINISH, 0);
	};

	std::ofstream file, file2, file3;
	openFile(file, dir + "log_" + STR(Ns) + ".dat");
	openFile(file2, dir + "logCompare_" + STR(Ns) + ".dat");
	openFile(file3, dir + "logDeg_" + STR(Ns) + ".dat");
	std::string infoSym = "";

	// parameters
	v_1d<int> paritiesSz = {};
	v_1d<int> paritiesSy = {};
	v_1d<int> parities = {};
	v_1d<int> spinFlip = {};
	v_1d<int> u1Values = {};

	bool useU1 = (this->modP.modTyp_ == MY_MODELS::XYZ_M) && this->modP.eta1_ == 0 && this->modP.eta2_ == 0;
	bool useSzParity = (this->modP.modTyp_ == MY_MODELS::XYZ_M);// && (Ns % 2 == 0);
	bool useSyParity = false;//(this->modP.modTyp_ == MY_MODELS::XYZ_M) && (Ns % 2 == 0);
	if (useSzParity)			paritiesSz = { -1, 1 }; else paritiesSz = { -INT_MAX };
	if (useSyParity)			paritiesSy = { -1, 1 }; else paritiesSy = { -INT_MAX };
	if (useU1)					for (uint i = 0; i <= Ns; i++) u1Values.push_back(i);
	else						u1Values.push_back(-INT_MAX);

	// save the Hamiltonian without symmetries
	arma::sp_mat H0 = this->hamDouble->getHamiltonian();
	// unitary transformation function
	auto unitaryTransform = [](arma::SpMat<cpx>& U, std::shared_ptr<Hamiltonian<cpx>> _H, arma::sp_cx_mat& H) {
		arma::sp_cx_mat _Hsym = _H->getHamiltonian();
		H += U * _Hsym * U.t();
		};

	// create sparse matrix to compare with the full hamiltonian
	arma::sp_cx_mat H(H0.n_rows, H0.n_cols);
	arma::SpMat<cpx> U;
	u64 calcStates = 0;

	this->symP.S_ = true;
	this->isComplex_ = true;
	// go through all symmetries
	for (auto U1 : u1Values) {
		this->symP.U1_ = U1;
		for (uint k = 0; k < Ns; k++) {
			this->symP.k_ = k;

			// set the parities
			if (k == 0 || (k == int(Ns / 2) && Ns % 2 == 0))
				parities = { -1, 1 };
			else
				parities = { -INT_MAX };

			// check the spin flip sector
			bool includeSpinFlip = (!useU1 && (this->modP.hz_ == 0.0)) || (useU1 && (Ns % 2 == 0) && (this->symP.U1_ == Ns / 2) && (this->modP.hz_ == 0.0) && (this->modP.hx_ == 0.0));
			if (includeSpinFlip)
				spinFlip = { -1, 1 };
			else
				spinFlip = { -INT_MAX };

			// go through all
			LOGINFO("STARTING ALL SECTORS", LOG_TYPES::INFO, 2);
			LOGINFO_CH_LVL(2);
			for (auto flip : spinFlip) {
				this->symP.px_ = flip;
				for (auto reflection : parities) {
					this->symP.x_ = reflection;
					for (auto parZ : paritiesSz) {
						this->symP.pz_ = parZ;
						for (auto parY : paritiesSy) {
							this->symP.py_ = parY;
							// create the models
							if (this->hamComplex)
								this->hamComplex.reset();

							// print sectors
							int kS = k;
							int xS = flip != -INT_MAX ? flip : 0;
							int xSy = parY != -INT_MAX ? parY : 0;
							int xSz = parZ != -INT_MAX ? parZ : 0;
							int pS = reflection != -INT_MAX ? reflection : 0;
							int U1s = U1 != -INT_MAX ? U1 : 0;
							std::string sI = VEQ(kS) + "," + VEQ(pS) + "," + VEQ(xS) + "," + VEQ(xSy) + "," + VEQ(xSz) + "," + VEQ(U1s);
							symTuple _tup = std::make_tuple(kS, pS, xS, xSy, xSz, U1s);

							// define and check
							if (!this->defineModel<cpx>(this->hilComplex, this->hamComplex))
							{
								LOGINFO("EMPTY SECTOR: " + sI, LOG_TYPES::TRACE, 3);
								LOGINFO("-------------------", LOG_TYPES::TRACE, 2);
								file << "\t\t->EMPTY SECTOR : " << sI << EL;
								continue;
							}

							// info me pls
							infoSym = this->hamComplex->getInfo();
							Nh = this->hamComplex->getHilbertSize();
							LOGINFO("DOING: " + infoSym, LOG_TYPES::TRACE, 3);
							file << "\t->" << infoSym << EL;

							this->hamComplex->buildHamiltonian();
							LOGINFO("Finished building Hamiltonian " + sI, LOG_TYPES::TRACE, 4);
							this->hamComplex->diagH(false);
							LOGINFO("Finished diagonalizing Hamiltonian " + sI, LOG_TYPES::TRACE, 4);
							//_degs.push_back(std::make_pair(_tup, this->hamComplex->getDegeneracies()));
							_sizes.push_back(Nh);

							// create rotation matrix
							U = this->hamComplex->hilbertSpace.getSymRot();
							if (!useU1)
								unitaryTransform(U, this->hamComplex, H);

							calcStates += Nh;
							arma::Mat<cpx> tStates;
							if (Ns < 10)
								tStates = arma::Mat<cpx>(NhFull, Nh);
							arma::vec entroInner(Nh);
							for (long long i = 0; i < Nh; i++) {
								// get the energy to push back
								auto En = this->hamComplex->getEigVal(i);
								enSYMS.push_back(En);

								// transform the state
								arma::Col<cpx> state = this->hamComplex->getEigVec(i);
								arma::Col<cpx> transformed_state = U * state;
								if (Ns < 10)
									tStates.col(i) = transformed_state;
								// calculate the entanglement entropy
								auto entropy = Entropy::Entanglement::Bipartite::vonNeuman<cpx>(transformed_state, La, this->hamComplex->hilbertSpace);
								entroInner(i) = entropy;
								entSYMS.push_back(entropy);

								// push back symmetry
								_sizesSyms.push_back(Nh);
								_symmetries.push_back(_tup);
							}
							std::string filename = dir + infoSym + ".h5";
							this->hamComplex->getEigVal(dir, HAM_SAVE_EXT::h5, false);
							if (Ns < 10)
								tStates.save(dir + infoSym + ".dat", arma::raw_ascii);
							entroInner.save(arma::hdf5_name(filename, "entropy", arma::hdf5_opts::append));
							LOGINFO("------------------- Finished: " + sI + "---------------------------\n", LOG_TYPES::FINISH, 3);
						}
					}
				}
			}
		}
	}
	LOGINFO("Finished building symmetries", LOG_TYPES::TRACE, 2);
	LOGINFO(VEQ(this->hamDouble->getHilbertSize()) + "\t" + VEQ(calcStates), LOG_TYPES::TRACE, 2);


	LOGINFO("PRINTING DIFFERENT ELEMENTS: \n", LOG_TYPES::TRACE, 2);
	// --------------- PLOT MATRIX DIFFERENCES ---------------
	if (!useU1) {
		auto N = H0.n_cols;
		arma::sp_mat HH = arma::real(H);
		arma::sp_cx_mat res = arma::sp_cx_mat(HH - H0, arma::imag(H));

		printSeparated(std::cout, '\t', 32, true, "index i", "index j", "difference", "original hamiltonian", "symmetry hamiltonian");
		cpx x = 0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cpx val = res(i, j);
				if (res(i, j).real() > 1e-13 || res(i, j).imag() > 1e-13) {
					x += val;
					printSeparated(std::cout, '\t', 32, true, i, j, res(i, j), H0(i, j), H(i, j));
					printSeparated(std::cout, '\t', 32, true, i, j, res);
				}
			}
		}
		printSeparated(std::cout, '\t', 32, true, "Sum of the suspicious elements: ", x, "\n");
	}

	// --------------- GET THE RIGHT INDICES ---------------
	v_1d<uint> idxs(enSYMS.size(), 0);
	for (int i = 0; i != idxs.size(); i++)
		idxs[i] = i;

	std::sort(idxs.begin(), idxs.end(),
		[&](const int& a, const int& b) {
			return (enSYMS[a] < enSYMS[b]);
		}
	);

	LOGINFO("SORTING AND SO ON.", LOG_TYPES::TRACE, 2);
	// --------------- PRINT OUT THE COMPARISON ---------------
	printSeparated(file2, '\t', 15, true, infoHFull);
	printSeparated(file2, '\t', 15, false, "E_FULL", "E_SYMS", "S_FULL", "S_SYMS");
	printSeparated(file2, ',', 3, true, "T", "R", "PX", "PY", "PZ", "U1", "NH");

	for (long long i = 0; i < this->hamDouble->getHilbertSize(); i++) {
		// get the ed energy
		auto SORTIDX = idxs[i];
		auto EED = this->hamDouble->getEigVal(i);
		auto ENTED = ENTROPIES(La - 1, i);

		// sort
		auto ESYM = enSYMS[SORTIDX];
		const auto& [k, p, x, xY, xZ, u1] = _symmetries[SORTIDX];
		auto ENTSYM = entSYMS[SORTIDX];

		printSeparatedP(file2, '\t', 15, false, 7, EED, ESYM, ENTED, ENTSYM);
		printSeparatedP(file2, ',', 3, true, 7, STR(k), STR(p), STR(x), STR(xY), STR(xZ), STR(u1), STR(_sizesSyms[i]));
	}

	// --------------- PRINT OUT THE DEGENERACIES ---------------
	printSeparated(file3, '\t', 15, true, infoHFull);
	printSeparated(file3, ',', 4, false, "T", "R", "PX", "PY", "PZ", "U1", "NH");
	printSeparated(file3, ',', 15, true, "DEG");
	for (uint i = 0; i < _degs.size(); i++)
	{
		auto& [_syms, _degM] = _degs[i];
		const auto& [k, p, x, xY, xZ, u1] = _syms;
		printSeparatedP(file3, ',', 4, false, 7, STR(k), STR(p), STR(x), STR(xY), STR(xZ), STR(u1), STR(_sizes[i]));

		// degeneracy
		int _degC = 0;
		for (auto it = _degM.begin(); it != _degM.end(); it++)
			_degC += it->first * it->second;
		LOGINFO(VEQ(_degC), LOG_TYPES::TRACE, 4);
		printSeparatedP(file3, ',', 15, true, 5, _degC);
	}

	file.close();
	file2.close();
	file3.close();
	LOGINFO_CH_LVL(0);
	LOGINFO("FINISHED ALL.", LOG_TYPES::TRACE, 0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
* @brief Allows to calculate how do the degeneracies look in case of symmetric Hamiltonians...
* At the same time, it calculates mixing of the states in the degenerate manifold
*/
void UI::symmetriesDeg()
{
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	_timer.reset();
	u64 Nh					= this->hamComplex->getHilbertSize();
	const auto _realizations= this->nqsP.nqs_col_bn_;

	// create the directories
	std::string dir			= makeDirsC(this->mainDir, 
										this->hamComplex->getType(), 
										this->latP.lat->get_info(), "CombinationsManifold_" + STR(_realizations));

	// use those files  
	std::string modelInfo	= this->hamComplex->getInfo();

	// save entropies txt check 
	std::string filename	= dir + modelInfo;

	// check Hilbert size or whether we should diagonalize and proceed further
	HILBERT_EMPTY_CHECK(Nh, return);

	// set the Hamiltonian
	this->hamComplex->buildHamiltonian();

	// set the parameters
	uint Ns					= this->hamComplex->getNs();
	// 50% 
	u64 stateNum			= Nh / 2;
	auto ran				= this->hamComplex->ran_;

	IFELSE_EXCEPTION(Nh < UI_LIMITS_MAXFULLED, this->hamComplex->diagH(false), "Size of the Hilbert space to big to diagonalize");
	LOGINFO(_timer.start(), "Degeneracies - diagonalization", 2);

	std::string name		= VEQ(Nh);
	LOGINFO("Spectrum size: " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states: " + STR(stateNum), LOG_TYPES::TRACE, 3);
	this->hamComplex->getEigVal(dir, HAM_SAVE_EXT::h5, false);

	// clear energies
	this->hamComplex->clearH();

	// degeneracies
	v_2d<u64> degeneracyMap;
	BEGIN_CATCH_HANDLER
	{
		degeneracyMap		= this->hamComplex->getDegeneracies();
	} 
	END_CATCH_HANDLER("Cannot setup the degeneracies...", return;)

	// iterate through bond cut
	const uint maxBondNum	= Ns / 2;

	// get the symmetry rotation matrix
	auto _symRot			= this->hamComplex->getSymRot();
	bool usesSym			= this->hamComplex->hilbertSpace.checkSym();
	_timer.checkpoint("entro");
	this->hamComplex->generateFullMap();

	// append gammas - to be sure that we use the correct ones from the degenerate states
	v_1d<int> _gammas		= { };
	for (auto i = 0; i < degeneracyMap.size(); ++i)
		if (degeneracyMap[i].size() != 0)
			_gammas.push_back(i);

	// create random coeficients
	v_2d<CCOL> _HM			= {};
	for (int i = 0; i < _gammas.size(); ++i)
		if (_gammas[i] < stateNum)
			_HM.push_back(ran.createRanState<cpx>(_gammas[i], _realizations));

	// ------------------------------- go through all degeneracies (start with degeneracy 2) -------------------------------
	// get index of the average energy
	auto _idx	[[maybe_unused]]	= this->hamComplex->getEnAvIdx();
	u64 _minIdx [[maybe_unused]]	= _idx - stateNum / 2;
	u64 _maxIdx [[maybe_unused]]	= _idx + stateNum / 2;

	MAT<double> _entropies(_realizations, 2);

	// go through the degeneracies again
	for (int _ig = 0; _ig < _gammas.size(); ++_ig)
	{
		auto _gamma				= _gammas[_ig];
		// get the number of states in the degenerate manifold
		int _degSize			= degeneracyMap[_gamma].size();
		if (_degSize == 0)
			continue;
		LOGINFO("Doing: " + VEQ(_gamma) + VEQV(". Manifold size", _degSize), LOG_TYPES::TRACE, 1);

		// check the start and end idx from middle spectrum
		auto _idxStart			= _degSize;
		auto _idxEnd			= 0;
		// setup the start index
		for (auto i = 0; i < _degSize; i += _gamma)
		{
			if (degeneracyMap[_gamma][i] >= _minIdx)
			{
				// first state in the range found
				_idxStart = i;
				break;
			}
		}
		// setup the end index
		for (auto i = _degSize - _gamma; i >= 0; i -= _gamma)
		{
			if (degeneracyMap[_gamma][i] <= _maxIdx)
			{
				// first index from the back found
				_idxEnd = i;
				break;
			}
		}
		
		// if the indices are correct
		if (_idxStart >= _idxEnd)
		{
			LOGINFO("For: " + VEQ(_gamma) + ". Cannot find states in the middle of the spectrum", LOG_TYPES::TRACE, 2);
			continue;
		}
		LOGINFO("Choosing from: [" + VEQ(_idxStart) + "(true:" + STR(degeneracyMap[_gamma][_idxStart]) + ")," +
									 VEQ(_idxEnd)   + "(true:" + STR(degeneracyMap[_gamma][_idxEnd])   + ")]", 
									 LOG_TYPES::TRACE, 2);

		// get the range to choose from as there is _gamma which decides on the manifold
		auto _rangeLength	=	(_idxEnd - _idxStart) / _gamma;

		pBar pbar(5, _realizations);
//#pragma omp parallel for num_threads(this->threadNum)
		for (int _r = 0; _r < _realizations; _r++)
		{
			// choose indices from degenerate states in the manifold
			auto idx		=	_idxStart + _gamma * ran.template randomInt<int>(0, _rangeLength);
			// set the true state on that index
			auto idxState	=	degeneracyMap[_gamma][idx];

			CCOL _state		=	_HM[_ig][_r][0] * this->hamComplex->getEigVec(idxState);
			// append states
			for (int id = 1; id < _gamma; id++)
			{
				idxState	=	degeneracyMap[_gamma][idx + id];
				_state		+=	_HM[_ig][_r][id] * this->hamComplex->getEigVec(idxState);
			}

			// normalize state
			_state			=	_state / std::sqrt(arma::cdot(_state, _state));

			// rotate!
			if (usesSym)
				_state			=	_symRot * _state;

			// save the values
			_entropies(_r, 0)	=	this->hamComplex->getEigVal(idxState);
			_entropies(_r, 1)	=	Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_state, maxBondNum, this->hamComplex->hilbertSpace);

			// update progress
			PROGRESS_UPD(_r, pbar, VEQ(_gamma) + " Entropy was : S=" + VEQ(_entropies(_r, 1) / (log(2.0) * maxBondNum)));
		}
		if (!_entropies.save(arma::hdf5_name(filename + ".h5", STR(_gamma), arma::hdf5_opts::append)))
			_entropies.save(arma::hdf5_name(dir + "entropies" + ".h5", STR(_gamma), arma::hdf5_opts::append));
	}
	LOGINFO(_timer.point(1), "Degenerate - Entropies", 2);
}

/**
* @brief Use this function to create a combination of states in the middle of the spectrum and 
* calculate their entanglement entropies 
*/
void UI::symmetriesCreateDeg()
{
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	_timer.reset();
	u64 Nh					=			this->hamComplex->getHilbertSize();
	const auto _realizations=			this->nqsP.nqs_col_bn_;

	// create the directories
	std::string dir			=			makeDirsC(this->mainDir, 
													this->hamComplex->getType(), 
													this->latP.lat->get_info(), 
													"CreateCombinationsNearZero_" + STR(_realizations));
	
	// use those files 
	std::string modelInfo	=			this->hamComplex->getInfo();
	
	// save entropies txt check
	std::string filename	=			dir + modelInfo;
	
	// check Hilbert size or whether we should diagonalize and proceed further
	HILBERT_EMPTY_CHECK(Nh, return);
	
	// set the Hamiltonian
	this->hamComplex->buildHamiltonian();
	
	// set the parameters
	uint Ns					=			this->hamComplex->getNs();
	// 20% of the Hilbert space
	u64 stateNum			=			Nh / 5;
	auto ran				=			this->hamComplex->ran_;
	
	IFELSE_EXCEPTION(Nh < UI_LIMITS_MAXFULLED, this->hamComplex->diagH(false), "Size of the Hilbert space to big to diagonalize");
	LOGINFO(_timer.start(), "Interacting Mix - diagonalization");
	
	std::string name		=			VEQ(Nh);
	LOGINFO("Spectrum size: "			+ STR(Nh)		, LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states: "		+ STR(stateNum)	, LOG_TYPES::TRACE, 2);
	this->hamComplex->getEigVal(dir, HAM_SAVE_EXT::h5, false);
	
	// clear energies
	this->hamComplex->clearH();
	
	// iterate through bond cut
	const uint maxBondNum	=			Ns / 2;
	
	// get the symmetry rotation matrix
	auto _symRot			=			this->hamComplex->getSymRot();
	bool usesSym			=			this->hamComplex->hilbertSpace.checkSym();
	this->_timer.checkpoint("entropy");
	this->hamComplex->generateFullMap();
	
	// information about state combination
	const v_1d<int> _gammas	=			{ 1, 2, 3, 4, 5, 10, (int)Ns, int(2*Ns) };
	v_2d<CCOL> _HM			=			{};
	
	// create random coeficients
	for (int i = 0; i < _gammas.size(); ++i)
		if (_gammas[i] < stateNum)		
			_HM.push_back(ran.createRanState<cpx>(_gammas[i], _realizations));
	
	// get index of the average energy
	auto _idx	[[maybe_unused]]	=	this->hamComplex->getEnAvIdx();
	u64 _minIdx [[maybe_unused]]	=	_idx - stateNum / 2;
	u64 _maxIdx [[maybe_unused]]	=	_idx + stateNum / 2;

	auto toChooseFrom		=			Vectors::vecAtoB(stateNum, _minIdx);

	// as many entropies as realizations
	MAT<double> _entropies(_realizations, 2);

	// go through the gammas
	for (int _ig = 0; _ig < _gammas.size(); ++_ig)
	{
		auto gamma = _gammas[_ig];
	
		// too many states
		if (gamma >= stateNum)
			break;
	
		LOGINFO("Taking gamma: " + STR(gamma), LOG_TYPES::TRACE, 3);
		pBar pbar(5, _realizations);
//#pragma omp parallel for num_threads(this->threadNum)
		for (int _r = 0; _r < _realizations; _r++)
		{
			// choose random indices
			auto idx					=	ran.template randomInt<u64>(_minIdx, _maxIdx - 1 - gamma);
			arma::Col<cpx> _state		=	_HM[_ig][_r](0) * this->hamComplex->getEigVec(idx);

			// append states
			for (int id = 1; id < gamma; id++)
				_state					+=	_HM[_ig][_r](id) * this->hamComplex->getEigVec(idx + id);

			// rotate!
			if (usesSym)
				_state					=	_symRot * _state;

			// normalize state
			_state						=	_state / std::sqrt(arma::cdot(_state, _state));
			_entropies(_r, 0)			=	this->hamComplex->getEigVal(idx);
			_entropies(_r, 1)			=	Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_state, maxBondNum, this->hamComplex->hilbertSpace);
	
			PROGRESS_UPD(_r, pbar, VEQ(gamma) + " Entropy was : S=" + VEQ(_entropies(_r, 1) / (log(2.0) * maxBondNum)));
		}
		if(!_entropies.save(arma::hdf5_name(filename + ".h5", STR(gamma), arma::hdf5_opts::append)))
			_entropies.save(arma::hdf5_name(dir + "entropies" + ".h5", STR(gamma), arma::hdf5_opts::append));
	}
	LOGINFO(_timer.point(1), "Mixing - Entropies", 2);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%% DEFINE THE TEMPLATES %%%%%%%%%%%%%%%%%%%%%
template void UI::symmetries<double>(std::shared_ptr<Hamiltonian<double>> _H, bool _diag, bool _states);
template void UI::symmetries<cpx>(std::shared_ptr<Hamiltonian<cpx>> _H, bool _diag, bool _states);