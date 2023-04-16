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
// ###################### LIMITS ############################
constexpr u64 UI_LIMITS_MAXFULLED								= ULLPOW(16);
constexpr u64 UI_LIMITS_MAXPRINT								= ULLPOW(11);
constexpr u64 UI_LIMITS_SI_STATENUM								= 100;
constexpr u64 UI_LIMITS_MIDDLE_SPEC_STATENUM					= 200;
// ##########################################################
#define UI_CHECK_SYM(val, gen)									if(this->##val##_ != -INT_MAX) syms.push_back(std::make_pair(Operators::SymGenerators::##gen, this->##val##_));

// -------------------------------------------------------- make an USER INTERFACE class --------------------------------------------------------

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

namespace UI_PARAMS {
	
	/*
	* @brief Defines parameters used later for the models
	*/
	struct ModP {
		// ############### TYPE ################
		UI_PARAM_CREATE_DEFAULT(modTyp		, MY_MODELS	, MY_MODELS::ISING_M);

		// ############### ISING ###############
		UI_PARAM_STEP(double				, J			, 1.0				);	// spin exchange
		UI_PARAM_STEP(double				, hz		, 1.0				);	// perpendicular field
		UI_PARAM_STEP(double				, hx		, 1.0				);	// transverse field
		// ############### XYZ #################		
		UI_PARAM_STEP(double				, J2		, 2.0				);	// next nearest neighbors exchange
		UI_PARAM_STEP(double				, eta1		, 0.5				);				
		UI_PARAM_STEP(double				, eta2		, 0.5				);
		UI_PARAM_STEP(double				, dlt1		, 0.3				);
		UI_PARAM_STEP(double				, dlt2		, 0.3				);
		// ############### KITAEV ##############
		UI_PARAM_STEP(double				, kx		, 1.0				);	// spin exchange
		UI_PARAM_STEP(double				, ky		, 1.0				);	// spin exchange
		UI_PARAM_STEP(double				, kz		, 1.0				);	// spin exchange

		void setDefault() {
			UI_PARAM_SET_DEFAULT(modTyp);
			// ising
			UI_PARAM_SET_DEFAULT_STEP(J);
			UI_PARAM_SET_DEFAULT_STEP(hz);
			UI_PARAM_SET_DEFAULT_STEP(hx);
			// xyz
			UI_PARAM_SET_DEFAULT_STEP(J2);
			UI_PARAM_SET_DEFAULT_STEP(eta1);
			UI_PARAM_SET_DEFAULT_STEP(eta2);
			UI_PARAM_SET_DEFAULT_STEP(dlt1);
			UI_PARAM_SET_DEFAULT_STEP(dlt2);
			// kitaev
			UI_PARAM_SET_DEFAULT_STEP(kx);
			UI_PARAM_SET_DEFAULT_STEP(ky);
			UI_PARAM_SET_DEFAULT_STEP(kz);
		}
	};

	/*
	* @brief Defines lattice used later for the models
	*/
	struct LatP {
		UI_PARAM_CREATE_DEFAULT(bc	, BoundaryConditions, BoundaryConditions::PBC	);
		UI_PARAM_CREATE_DEFAULT(typ	, LatticeTypes		, LatticeTypes::SQ			);
		UI_PARAM_CREATE_DEFAULT(Lx	, uint				, 2							);
		UI_PARAM_CREATE_DEFAULT(Ly	, uint				, 1							);
		UI_PARAM_CREATE_DEFAULT(Lz	, uint				, 1							);
		UI_PARAM_CREATE_DEFAULT(dim	, uint				, 1							);

		std::shared_ptr<Lattice> lat;
		
		void setDefault() {
			UI_PARAM_SET_DEFAULT(typ);
			UI_PARAM_SET_DEFAULT(bc);
			UI_PARAM_SET_DEFAULT(Lx);
			UI_PARAM_SET_DEFAULT(Ly);
			UI_PARAM_SET_DEFAULT(Lz);
			UI_PARAM_SET_DEFAULT(dim);
		};
	};

	/*
	* @brief Defines a container for symmetry eigenvalues.
	* @warning By default, the parameters are -maximal integer in order to tell that no symmetry is used
	*/
	struct SymP {
		UI_PARAM_CREATE_DEFAULT(S	, bool	, false		);
		UI_PARAM_CREATE_DEFAULT(k	, int	, -INT_MAX	);
		UI_PARAM_CREATE_DEFAULT(px	, int	, -INT_MAX	);
		UI_PARAM_CREATE_DEFAULT(py	, int	, -INT_MAX	);
		UI_PARAM_CREATE_DEFAULT(pz	, int	, -INT_MAX	);
		UI_PARAM_CREATE_DEFAULT(x	, int	, -INT_MAX	);
		UI_PARAM_CREATE_DEFAULT(U1	, int	, -INT_MAX	);
		
		void setDefault() {
			UI_PARAM_SET_DEFAULT(S);
			UI_PARAM_SET_DEFAULT(k);
			UI_PARAM_SET_DEFAULT(px);
			UI_PARAM_SET_DEFAULT(py);
			UI_PARAM_SET_DEFAULT(pz);
			UI_PARAM_SET_DEFAULT(x);
			UI_PARAM_SET_DEFAULT(U1);
		}

		v_1d<std::pair<Operators::SymGenerators, int>> getLocGenerator() {
			v_1d<std::pair<Operators::SymGenerators, int>> syms = {};
			UI_CHECK_SYM(k, T);
			UI_CHECK_SYM(px, PX);
			UI_CHECK_SYM(py, PY);
			UI_CHECK_SYM(pz, PZ);
			UI_CHECK_SYM(x, R);
			return syms;
		}
	};

	// !TODO 
	// Neural network quantum states params
	struct NqsP {
		//unique_ptr<rbmState<_type, _hamtype>> phi;
		v_1d<u64> layersDim;
		UI_PARAM_CREATE_DEFAULT	(nHidden	, uint	, 1					);
		UI_PARAM_CREATE_DEFAULT	(nVisible	, uint	, 1					);
		UI_PARAM_CREATE_DEFAULT	(nLayers	, uint	, 2					);
		UI_PARAM_CREATE_DEFAULT	(nFlips		, uint	, 1					);
		UI_PARAM_CREATE_DEFAULT	(blockSize	, uint	, 8					);
		UI_PARAM_CREATE_DEFAULT	(nTherm		, uint	, 50				);
		UI_PARAM_CREATE_DEFAULT	(nBlocks	, uint	, 500				);
		UI_PARAM_CREATE_DEFAULT	(nMcSteps	, uint	, 1000				);
		UI_PARAM_CREATE_DEFAULT	(batch		, u64	, 1024				);
		UI_PARAM_CREATE_DEFAULTD(lr			, uint	, 1					);
		
		void setDefault() {
			UI_PARAM_SET_DEFAULT(nHidden);
			UI_PARAM_SET_DEFAULT(nVisible);
			UI_PARAM_SET_DEFAULT(nLayers);
			UI_PARAM_SET_DEFAULT(nFlips);
			UI_PARAM_SET_DEFAULT(blockSize);
			UI_PARAM_SET_DEFAULT(nTherm);
			UI_PARAM_SET_DEFAULT(nBlocks);
			UI_PARAM_SET_DEFAULT(nMcSteps);
			UI_PARAM_SET_DEFAULT(batch);
			UI_PARAM_SET_DEFAULT(lr);
		}
	};
};

/*
* @brief
*/
class UI : public UserInterface {

protected:

	// LATTICE params
	UI_PARAMS::LatP latP;

	// SYMMETRY params
	UI_PARAMS::SymP symP;

	// NQS params
	UI_PARAMS::NqsP nqsP;

	// MODEL params
	UI_PARAMS::ModP modP;

	// define basic models
	bool isComplex_							= false;
	Hilbert::HilbertSpace<double>			hilDouble;
	std::shared_ptr<Hamiltonian<double>>	hamDouble;
	Hilbert::HilbertSpace<cpx>				hilComplex;
	std::shared_ptr<Hamiltonian<cpx>>		hamComplex;

	// heisenberg with classical dots stuff
	//v_1d<int> positions = {};
	//arma::vec phis = arma::vec({});
	//arma::vec thetas = arma::vec({});
	//arma::vec J_dot = { 1.0,0.0,-1.0 };
	//double J0_dot = 0.0;
	//double J_dot_dot = 1.0;														// dot - dot  classical interaction

	// averages from operators
	//avOperators av_op;

	void setDefaultMap()					final override {
		this->defaultParams = {
			UI_OTHER_MAP(m			, this->nqsP.nMcSteps_	, FHANDLE_PARAM_HIGHER0			),			// mcsteps	
			UI_OTHER_MAP(b			, this->nqsP.batch_		, FHANDLE_PARAM_HIGHER0			),			// batch
			UI_OTHER_MAP(nb			, this->nqsP.nBlocks_	, FHANDLE_PARAM_HIGHER0			),			// number of blocks
			UI_OTHER_MAP(bs			, this->nqsP.blockSize_	, FHANDLE_PARAM_HIGHER0			),			// block size

			{			"f"			, std::make_tuple(""	, FHANDLE_PARAM_DEFAULT)		},			// file to read from directory
			// ---------------- lattice parameters ----------------
			UI_OTHER_MAP(d			, this->latP._dim		, FHANDLE_PARAM_BETWEEN(1., 3.)	),	
			UI_OTHER_MAP(bc			, this->latP._bc		, FHANDLE_PARAM_BETWEEN(0., 3.)	),
			UI_OTHER_MAP(l			, this->latP._typ		, FHANDLE_PARAM_BETWEEN(0., 1.)	),
			UI_OTHER_MAP(lx			, this->latP._Lx		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(ly			, this->latP._Ly		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(lz			, this->latP._Lz		, FHANDLE_PARAM_HIGHER0			),
			// ---------------- model parameters ----------------
			UI_OTHER_MAP(mod		, this->modP._modTyp	, FHANDLE_PARAM_BETWEEN(0., 2.)	),
			// -------- ising
			UI_PARAM_MAP(J			, this->modP._J			, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(hx			, this->modP._hx		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(hz			, this->modP._hz		, FHANDLE_PARAM_DEFAULT			),
			// -------- heisenberg		
			UI_PARAM_MAP(dlt1		, this->modP._dlt1		, FHANDLE_PARAM_DEFAULT			),
			// -------- xyz
			UI_PARAM_MAP(J2			, this->modP._J2		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(eta1		, this->modP._eta1		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(eta2		, this->modP._eta2		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(dlt2		, this->modP._dlt2		, FHANDLE_PARAM_DEFAULT			),
			// -------- kitaev --------
			UI_PARAM_MAP(kx			, 0.0					, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(ky			, 0.0					, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(kz			, 0.0					, FHANDLE_PARAM_DEFAULT			),
			// ---------------- symmetries ----------------
			UI_PARAM_MAP(ks			, this->symP._k			, FHANDLE_PARAM_HIGHER0			),
			UI_PARAM_MAP(pxs		, this->symP._px		, FHANDLE_PARAM_BETWEEN()		),
			UI_PARAM_MAP(pys		, this->symP._py		, FHANDLE_PARAM_BETWEEN()		),
			UI_PARAM_MAP(pzs		, this->symP._pz		, FHANDLE_PARAM_BETWEEN()		),
			UI_PARAM_MAP(xs			, this->symP._x			, FHANDLE_PARAM_BETWEEN()		),
			UI_PARAM_MAP(u1s		, this->symP._U1		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(SYM		, this->symP._S			, FHANDLE_PARAM_BETWEEN(0., 1.)	),			// even use symmetries?
			// ---------------- other ----------------
			UI_OTHER_MAP(fun		, -1.					, FHANDLE_PARAM_HIGHERV(-1.0)	),			// choice of the function to be calculated
			UI_OTHER_MAP(th			, 1.0					, FHANDLE_PARAM_HIGHER0			),			// number of threads
			UI_OTHER_MAP(q			, 0.0					, FHANDLE_PARAM_DEFAULT			),			// quiet?
			UI_OTHER_MAP(dir		, "DEFALUT"				, FHANDLE_PARAM_DEFAULT			),
		};
	};

	// -------------------------------------------   		 HELPER FUNCTIONS  		-------------------------------------------
	//void compare_ed(double ground_rbm);
	//void save_operators(clk::time_point start, std::string name, double energy, double energy_error);
	
private:
	// INNER METHODS
	// ####################### SYMMETRIES #######################
	template<typename _T>//, std::enable_if_t<is_complex<_T>{}>* = nullptr>
	void symmetries(clk::time_point start, std::shared_ptr<Hamiltonian<_T>> _H);

	// ####################### CLASSICAL ########################
		
	//void make_mc_classical_angles(double Jdot = 0.0);

	// ####################### KITAEV #######################

	//void make_mc_kitaev(t_3d<double> K);

	void defineModels();
public:
	// -----------------------------------------------        CONSTRUCTORS  		-------------------------------------------
	~UI()									= default;
	UI()									= default;
	UI(int argc, char** argv){
		this->setDefaultMap();
		this->init(argc, argv);
	};

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
	void funChoice()						final override;
	void parseModel(int argc, cmdArg& argv) final override;

	// ----------------------------------------------- 			HELPERS  			-------------------------------------------
	void setDefault()						final override;

	// -----------------------------------------------  	   SIMULATION  		    -------------------------------------------	 


	// #######################		     RBMs               #######################
		
	// ########## CLASSICAL
	//void make_mc_classical();
	//void make_mc_angles_sweep();

	// ########## KITAEV
	//void make_mc_kitaev_sweep();

	// ########## TEST 
	//void make_simulation() override;

	// #######################        SYMMETRIES            #######################
	void makeSimSymmetries();
	//void make_simulation_symmetries_sweep();
	//void make_symmetries_test(int l = -1);

};

template<typename _T>
inline void UI::symmetries(clk::time_point start, std::shared_ptr<Hamiltonian<_T>> _H)
{
	// set the Hamiltonian
	_H->hamiltonian();
	// set the parameters
	u64 Nh = _H->getHilbertSize();
	u64 Ns = _H->getNs();
	u64 stateNum = Nh;
	bool useShiftAndInvert = false;
	std::string infoH = _H->getInfo();

	//stouts("\t->", start);
	LOGINFO("Finished buiding Hamiltonian" + infoH, LOG_TYPES::TRACE, 1);
	if (Nh < UI_LIMITS_MAXFULLED) {
		LOGINFO("Using standard diagonalization", LOG_TYPES::TRACE, 2);
		_H->diagH(false);
	}
	else
	{
		LOGINFO("Using S&I", LOG_TYPES::TRACE, 2);
		useShiftAndInvert = true;
		stateNum = UI_LIMITS_SI_STATENUM;
		_H->diagH(false, stateNum, 0, 1000, 1e-5, "sa");
	}
	LOGINFO("Finished the diagonalization", LOG_TYPES::TRACE, 2);
	LOGINFO(STR(t_ms(start)) + " ms", LOG_TYPES::TIME, 2);

	std::string name = VEQ(Nh);
	LOGINFO("Spectrum size: " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states: " + STR(stateNum), LOG_TYPES::TRACE, 2);

	// --- create the directories ---
	std::string dir = this->mainDir + kPS;
	fs::create_directories(dir);

	// --- use those files --- 
	std::string modelInfo = _H->getInfo();

	// --- save energies txt check ---
	std::string filename = dir + infoH;
	if (Nh < UI_LIMITS_MAXPRINT)
		_H->getEigVal(dir, HAM_SAVE_EXT::dat);
	_H->getEigVal(dir, HAM_SAVE_EXT::h5);

	// clear energies
	_H->clearEigVal();
	_H->clearH();

	// iterate through bond cut
	const uint maxBondNum = Ns / 2;
	arma::mat ENTROPIES(maxBondNum, stateNum, arma::fill::zeros);

	// get the symmetry rotation matrix
	auto _symRot = _H->getSymRot();
	bool usesSym = _H->hilbertSpace.checkSym();

	// set which bonds we want to cut in bipartite
	v_1d<uint> _bonds = { maxBondNum };
	auto beforeEntro = clk::now();
#pragma omp parallel for num_threads(this->threadNum)
	for (u64 idx = 0; idx < stateNum; idx++) {
		// get the eigenstate
		arma::Col<_T> state = _H->getEigVec(idx);
		if (usesSym)
			state = _symRot * state;
		// go through bonds
		for (auto i : _bonds) {
			// iterate through the state
			auto entro = Entropy::Entanglement::Bipartite::vonNeuman<_T>(state, i, _H->hilbertSpace);
			// save the entropy
			ENTROPIES(i - 1, idx) = entro;
		}
		if (stateNum > 10 && idx % int(stateNum / 10) == 0)
			LOGINFO("Finished: " + STR(int(idx * 100.0 / stateNum)) + "%", LOG_TYPES::TRACE, 3);
	}
	LOGINFO("Finished entropies!", LOG_TYPES::TRACE, 2);
	LOGINFO(STR(t_ms(beforeEntro)) + " ms", LOG_TYPES::TIME, 2);

	// save entropies file
	ENTROPIES.save(arma::hdf5_name(filename + ".h5", "entropy", arma::hdf5_opts::append));
	if (Ns <= UI_LIMITS_MAXPRINT)
		ENTROPIES.save(filename + ".dat", arma::arma_ascii);

	if (useShiftAndInvert)
		return;

	// set the average energy index
	const u64 avEnIdx = _H->getEnAvIdx();

	// save states near the mean energy index
	if (Ns == 20)
		_H->getEigVec(dir, UI_LIMITS_MIDDLE_SPEC_STATENUM, HAM_SAVE_EXT::h5);
};
