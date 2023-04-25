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
// definitions											 // #
#define NQS_RBM_ANGLES_UPD							 // #
#define NQS_RBM_USESR									 // #
#define NQS_RBM_PINV									 // #
//#define NQS_RBM_SREG									 // #
#ifndef RBM_H											 // #
#include "../NQS/rbm.h"									 // #
#endif													 // #
// ##########################################################


// ######################### MODELS #########################
#ifndef ISING_H											 // #
#include "../models/ising.h"							 // #
#endif // !ISING_H										 // #
// MODELS												 // #
#ifndef XYZ_H											 // #
#include "../models/XYZ.h"								 // #
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
// ###################### LIMITS ############################
#define UI_ENERGYMEAN_SUBVEC(MCSTEPS, TROUT)					int(TROUT * MCSTEPS), MCSTEPS - int(TROUT * MCSTEPS) - 1
constexpr int UI_LIMITS_NQS_ED									= ULLPOW(16);
constexpr int UI_LIMITS_NQS_LANCZOS_STATENUM					= 100;


constexpr u64 UI_LIMITS_MAXFULLED								= ULLPOW(16);
constexpr u64 UI_LIMITS_MAXPRINT								= ULLPOW(10);
constexpr u64 UI_LIMITS_SI_STATENUM								= 100;
constexpr u64 UI_LIMITS_MIDDLE_SPEC_STATENUM					= 200;
// ##########################################################
#define UI_CHECK_SYM(val, gen)									if(this->val##_ != -INT_MAX) syms.push_back(std::make_pair(Operators::SymGenerators::gen, this->val##_));

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
		UI_PARAM_STEP(double				, J1		, 1.0				);	// spin exchange
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
			UI_PARAM_SET_DEFAULT_STEP(J1);
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
		UI_PARAM_CREATE_DEFAULT	(type		, NQSTYPES	, NQSTYPES::RBM		);
		UI_PARAM_CREATE_DEFAULT	(nHidden	, uint		, 1					);
		UI_PARAM_CREATE_DEFAULT	(nVisible	, uint		, 1					);
		UI_PARAM_CREATE_DEFAULT	(nLayers	, uint		, 2					);
		UI_PARAM_CREATE_DEFAULT	(nFlips		, uint		, 1					);
		UI_PARAM_CREATE_DEFAULT	(blockSize	, uint		, 8					);
		UI_PARAM_CREATE_DEFAULT	(nTherm		, uint		, 50				);
		UI_PARAM_CREATE_DEFAULT	(nBlocks	, uint		, 500				);
		UI_PARAM_CREATE_DEFAULT	(nMcSteps	, uint		, 1000				);
		UI_PARAM_CREATE_DEFAULT	(batch		, u64		, 1024				);
		UI_PARAM_CREATE_DEFAULTD(lr			, double	, 1					);
		
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

	// define the NQS
	std::shared_ptr<NQS<cpx,cpx>>			nqsCpx;
	std::shared_ptr<NQS<double,cpx>>		nqsDouble;

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
			UI_OTHER_MAP(nqs		, this->nqsP.type_		, FHANDLE_PARAM_DEFAULT			),			// type of the NQS state	
			UI_OTHER_MAP(m			, this->nqsP.nMcSteps_	, FHANDLE_PARAM_HIGHER0			),			// mcsteps	
			UI_OTHER_MAP(b			, this->nqsP.batch_		, FHANDLE_PARAM_HIGHER0			),			// batch
			UI_OTHER_MAP(nb			, this->nqsP.nBlocks_	, FHANDLE_PARAM_HIGHER0			),			// number of blocks
			UI_OTHER_MAP(bs			, this->nqsP.blockSize_	, FHANDLE_PARAM_HIGHER0			),			// block size
			UI_OTHER_MAP(nh			, this->nqsP.nHidden_	, FHANDLE_PARAM_HIGHER0			),			// hidden params
			UI_OTHER_MAP(nf			, this->nqsP.nFlips_	, FHANDLE_PARAM_HIGHER0			),			// hidden params

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
			UI_PARAM_MAP(J1			, this->modP._J1		, FHANDLE_PARAM_DEFAULT			),
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
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% I N N E R    M E T H O D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	// ############# S Y M M E T R I E S   E D #############
	template<typename _T>//, std::enable_if_t<is_complex<_T>{}>* = nullptr>
	void symmetries(clk::time_point start, std::shared_ptr<Hamiltonian<_T>> _H);
	void symmetriesTest(clk::time_point start);

	// ####################### N Q S #######################
	template<typename _T>
	void nqsSingle(clk::time_point start, std::shared_ptr<NQS<_T, cpx>> _NQS);
	//void make_mc_classical_angles(double Jdot = 0.0);

	// ###################### KITAEV #######################

	//void make_mc_kitaev(t_3d<double> K);

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% D E F I N I T I O N S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	void defineModels();

	template<typename _T>
	void defineModel(Hilbert::HilbertSpace<_T>&& _Hil, std::shared_ptr<Hamiltonian<_T>>& _H);

	template<typename _T>
	void defineNQS(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<NQS<_T, cpx>>& _NQS);

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

	// ####################### N Q S #######################
	void makeSimNQS();

	// ########## CLASSICAL
	//void make_mc_classical();
	//void make_mc_angles_sweep();

	// ########## KITAEV
	//void make_mc_kitaev_sweep();

	// ########## TEST 
	//void make_simulation() override;

	// #######################        SYMMETRIES            #######################
	void makeSimSymmetries();
	void makeSimSymmetriesSweep();
	//void make_simulation_symmetries_sweep();
	//void make_symmetries_test(int l = -1);

};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<typename _T>
inline void UI::defineModel(Hilbert::HilbertSpace<_T>&& _Hil, std::shared_ptr<Hamiltonian<_T>>& _H)
{
	v_1d<GlobalSyms::GlobalSym> _glbSyms = {};
	v_1d<std::pair<Operators::SymGenerators, int>> _locSyms = {};
	if (this->symP.S_ == true)
	{
		// create Hilbert space
		if (this->symP.k_ != -INT_MAX && !(this->symP.k_ == 0 || this->symP.k_ == this->latP.lat->get_Ns() / 2))
			this->isComplex_ = true;
		// ------ LOCAL ------
		_locSyms = this->symP.getLocGenerator();
		// ------ GLOBAL ------
		// check U1
		if (this->symP.U1_ != -INT_MAX) _glbSyms.push_back(GlobalSyms::getU1Sym(this->latP.lat, this->symP.U1_));
	};

	_Hil = Hilbert::HilbertSpace<_T>(this->latP.lat, _locSyms, _glbSyms);
	switch (this->modP.modTyp_)
	{
	case MY_MODELS::ISING_M:
		_H = std::make_shared<IsingModel<_T>>(std::move(_Hil),
			this->modP.J1_, this->modP.hx_, this->modP.hz_, this->modP.J10_, this->modP.hx0_, this->modP.hz0_);
		break;
	case MY_MODELS::XYZ_M:
		_H = std::make_shared<XYZ<_T>>(std::move(_Hil),
			this->modP.J1_, this->modP.J2_, this->modP.hx_, this->modP.hz_,
			this->modP.dlt1_, this->modP.dlt2_, this->modP.eta1_, this->modP.eta2_,
			this->modP.J10_, this->modP.J20_, this->modP.hx0_, this->modP.hz0_,
			this->modP.dlt10_, this->modP.dlt20_, this->modP.eta10_, this->modP.eta20_,
			false);
		break;
	default:
		_H = std::make_shared<XYZ<_T>>(std::move(_Hil),
			this->modP.J1_, this->modP.J2_, this->modP.hx_, this->modP.hz_,
			this->modP.dlt1_, this->modP.dlt2_, this->modP.eta1_, this->modP.eta2_,
			this->modP.J10_, this->modP.J20_, this->modP.hx0_, this->modP.hz0_,
			this->modP.dlt10_, this->modP.dlt20_, this->modP.eta10_, this->modP.eta20_,
			false);
		break;
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Based on a given type, it creates a NQS
*/
template<typename _T>
inline void UI::defineNQS(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<NQS<_T, cpx>>& _NQS)
{
	switch (this->nqsP.type_)
	{
	case NQSTYPES::RBM:
		_NQS = std::make_shared<RBM_S<_T, cpx>>(_H,
												 this->nqsP.nHidden_,
												 this->nqsP.batch_,
												 this->threadNum,
												 this->nqsP.lr_);
		break;
	default:
		LOGINFO("I don't know any other NQS types :<", LOG_TYPES::INFO, 1);
		break;
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Computes the Hamiltonian with symmetries and saves the entanglement entropies.
* @param start time of the beginning
* @param _H the shared pointer to the Hamiltonian - for convinience of the usage
* @info used in https://arxiv.org/abs/2303.13577
*/
template<typename _T>
inline void UI::symmetries(clk::time_point start, std::shared_ptr<Hamiltonian<_T>> _H)
{
	// set the Hamiltonian
	_H->hamiltonian();
	// set the parameters
	u64 Nh					=			_H->getHilbertSize();
	if (Nh == 0)
		return;
	uint Ns = _H->getNs();
	u64 stateNum = Nh;
	bool useShiftAndInvert = false;
	std::string infoH		=			_H->getInfo();

	//stouts("\t->", start);
	LOGINFO("Finished buiding Hamiltonian" + infoH, LOG_TYPES::TRACE, 1);
	if (Nh < UI_LIMITS_MAXFULLED) {
		LOGINFO("Using standard diagonalization", LOG_TYPES::TRACE, 2);
		_H->diagH(false);
	}
	else
	{
		LOGINFO("Using S&I", LOG_TYPES::TRACE, 2);
		useShiftAndInvert	=			true;
		stateNum = UI_LIMITS_SI_STATENUM;
		_H->diagH(false, (int)stateNum, 0, 1000, 1e-5, "sa");
	}
	LOGINFO("Finished the diagonalization", LOG_TYPES::TRACE, 2);
	LOGINFO(STR(t_ms(start)) + " ms", LOG_TYPES::TIME, 2);

	std::string name		=			VEQ(Nh);
	LOGINFO("Spectrum size: " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states: " + STR(stateNum), LOG_TYPES::TRACE, 2);

	// --- create the directories ---
	std::string dir			=			this->mainDir + kPS + _H->getType() + kPS + this->latP.lat->get_info() + kPS;
	fs::create_directories(dir);

	// --- use those files --- 
	std::string modelInfo	=			_H->getInfo();

	// --- save energies txt check ---
	std::string filename	=			dir + infoH;
	if (Nh < UI_LIMITS_MAXPRINT)
		_H->getEigVal(dir, HAM_SAVE_EXT::dat, false);
	_H->getEigVal(dir, HAM_SAVE_EXT::h5, false);

	// clear energies
	_H->clearEigVal();
	_H->clearH();

	// iterate through bond cut
	const uint maxBondNum	=			Ns / 2;
	arma::mat ENTROPIES(maxBondNum, stateNum, arma::fill::zeros);

	// get the symmetry rotation matrix
	auto _symRot			=			_H->getSymRot();
	bool usesSym			=			_H->hilbertSpace.checkSym();

	// set which bonds we want to cut in bipartite
	v_1d<uint> _bonds		=			{};
	for (int i = 1; i <= maxBondNum; i++)
		_bonds.push_back(i);
	auto beforeEntro		=			clk::now();
#pragma omp parallel for num_threads(this->threadNum)
	for (long long idx = 0; idx < (long long)stateNum; idx++) {
		// get the eigenstate
		arma::Col<_T> state =			_H->getEigVec(idx);
		if (usesSym)
			state			=			_symRot * state;
		// go through bonds
		for (auto i : _bonds) {
			// iterate through the state
			auto entro		=			Entropy::Entanglement::Bipartite::vonNeuman<_T>(state, i, _H->hilbertSpace);
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
	const u64 avEnIdx		=			_H->getEnAvIdx();

	// save states near the mean energy index
	if (Ns == 20)
		_H->getEigVec(dir, UI_LIMITS_MIDDLE_SPEC_STATENUM, HAM_SAVE_EXT::h5, true);
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Tests the currently implemented symmetries based on specific model parameters
* @param start the timer start
*/
inline void UI::symmetriesTest(clk::time_point start)
{	
	using symTuple										=		std::tuple<int, int, int, int, int, int>;
	v_1d<double> entFULL								=		{};
	v_1d<double> entSYMS								=		{};
	v_1d<double> enSYMS									=		{};
	v_1d<symTuple> _symmetries							=		{};
	v_1d<u64> _sizesSyms								=		{};
	v_1d<std::pair<symTuple, std::map<int, int>>> _degs =		{};
	v_1d<u64> _sizes									=		{};
	uint Ns												=		1;
	u64 Nh												=		1;
	uint La												=		Ns / 2;
	std::string dir										=		"";
	std::string infoHFull								=		"";
	arma::mat ENTROPIES;
	// ---------------------------------- DIAG HAMILTONIAN WITHOUT SYMMETRIES ---------------------------------
	for (auto bc: {1, 0}) {
		this->latP.bc_									=		bc == 0 ? BoundaryConditions::PBC : BoundaryConditions::OBC;
		this->symP.S_									=		false;
		if (this->hamDouble)
			this->hamDouble.reset();
		this->defineModels();
		Nh												=		this->hamDouble->getHilbertSize();
		Ns												=		this->latP.lat->get_Ns();
		La												=		Ns / 2;

		LOGINFO("Started building full Hamiltonian", LOG_TYPES::TRACE, 1);
		LASTLVL = 2;
		this->hamDouble->hamiltonian();
		dir = this->mainDir + kPS + this->hamDouble->getType() + kPS + getSTR_BoundaryConditions(this->latP.bc_) + kPS;
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

		LOGINFO("Started entropies for full Hamiltonian", LOG_TYPES::TRACE, 1);
		// Save the entropies 
		ENTROPIES.zeros(La, Nh);
		for (u64 idx = 0; idx < Nh; idx++) {
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
	std::string infoSym			=		"";

	// parameters
	v_1d<int> paritiesSz		=		{};
	v_1d<int> paritiesSy		=		{};
	v_1d<int> parities			=		{};
	v_1d<int> spinFlip			=		{};
	v_1d<int> u1Values			=		{};

	bool useU1					=		(this->modP.modTyp_ == MY_MODELS::XYZ_M) && this->modP.eta1_ == 0 && this->modP.eta2_ == 0;
	bool useSzParity			=		(this->modP.modTyp_ == MY_MODELS::XYZ_M) && (Ns % 2 == 0);
	bool useSyParity			=		(this->modP.modTyp_ == MY_MODELS::XYZ_M) && (Ns % 2 == 0);
	if (useSzParity)			paritiesSz = { -1, 1 }; else paritiesSz = { -INT_MAX };
	if (useSzParity)			paritiesSy = { -1, 1 }; else paritiesSy = { -INT_MAX };
	if (useU1)					for (uint i = 0; i <= Ns; i++) u1Values.push_back(i);
	else						u1Values.push_back(-INT_MAX);

	// save the Hamiltonian without symmetries
	arma::sp_mat H0 = this->hamDouble->getHamiltonian();
	// unitary transformation function
	auto unitaryTransform = [this](arma::SpMat<cpx>& U, std::shared_ptr<Hamiltonian<cpx>> _H, arma::sp_cx_mat& H) {
		arma::sp_cx_mat _Hsym	=		_H->getHamiltonian();
		H						+=		U * _Hsym * U.t();
	};

	// create sparse matrix to compare with the full hamiltonian
	arma::sp_cx_mat H(H0.n_rows, H0.n_cols);
	arma::SpMat<cpx> U;
	u64 calcStates				=		0;
	this->symP.S_				=		true;
	// go through all symmetries
	for (auto U1 : u1Values) {
		this->symP.U1_			=		U1;
		for (uint k = 0; k < Ns; k++) {
			this->symP.k_		=		k;

			// set the parities
			if (k == 0 || (k == int(Ns / 2) && Ns % 2 == 0))
				parities		=		{ -1, 1 };
			else
				parities		=		{ -INT_MAX };

			// check the spin flip sector
			bool includeSpinFlip = (!useU1 && (this->modP.hz_ == 0.0)) || (useU1 && (Ns % 2 == 0) && (this->symP.U1_ == Ns / 2) && (this->modP.hz_ == 0.0) && (this->modP.hx_ == 0.0));
			if (includeSpinFlip)
				spinFlip		=		{ -1, 1 };
			else
				spinFlip		=		{ -INT_MAX };

			// go through all
			LOGINFO("STARTING ALL SECTORS", LOG_TYPES::INFO, 2);
			LASTLVL = 2;
			for (auto flip : spinFlip) {
				this->symP.px_ = flip;
				for (auto reflection : parities) {
					this->symP.x_ = reflection;
					for (auto parZ : paritiesSz){
						this->symP.pz_ = parZ;
						for (auto parY : paritiesSy) {
							this->symP.py_ = parY;
							// create the models
							if (this->hamComplex)
								this->hamComplex.reset();
							this->defineModel<cpx>(std::move(this->hilComplex), this->hamComplex);
							infoSym			=		this->hamComplex->getInfo();
							Nh				=		this->hamComplex->getHilbertSize();
							LOGINFO("DOING: " + infoSym, LOG_TYPES::TRACE, 3);
							file << "\t->" << infoSym << EL;

							// print sectors
							int kS			=		k;
							int xS			=		flip		!= -INT_MAX ? flip			: 0;
							int xSy			=		parY		!= -INT_MAX ? parY			: 0;
							int xSz			=		parZ		!= -INT_MAX ? parZ			: 0;
							int pS			=		reflection	!= -INT_MAX ? reflection	: 0;
							int U1s			=		U1			!= -INT_MAX ? U1			: 0;
							symTuple _tup	=		std::make_tuple(kS, pS, xS, xSy, xSz, U1s);
							std::string sI  =		VEQ(kS) + "," + VEQ(pS) + "," + VEQ(xS) + "," + VEQ(xSy) + "," + VEQ(xSz) + "," + VEQ(U1s);
							if (Nh == 0)
							{
								LOGINFO("EMPTY SECTOR: " + sI, LOG_TYPES::TRACE, 3);
								LOGINFO("-------------------", LOG_TYPES::TRACE, 2);
								file << "\t\t->EMPTY SECTOR : " << sI << EL;
								continue;
							}
							this->hamComplex->hamiltonian();
							LOGINFO("Finished building Hamiltonian " + sI, LOG_TYPES::TRACE, 4);
							this->hamComplex->diagH(false);
							LOGINFO("Finished diagonalizing Hamiltonian " + sI, LOG_TYPES::TRACE, 4);
							_degs.push_back(std::make_pair(_tup, this->hamComplex->getDegeneracies()));
							_sizes.push_back(Nh);

							// create rotation matrix
							U = this->hamComplex->hilbertSpace.getSymRot();
							if (!useU1)
								unitaryTransform(U, this->hamComplex, H);

							calcStates		+=		Nh;

							arma::vec entroInner(Nh);
							for (u64 i = 0; i < Nh; i++) {
								// get the energy to push back
								auto En = this->hamComplex->getEigVal(i);
								enSYMS.push_back(En);

								// transform the state
								arma::Col<cpx> state = this->hamComplex->getEigVec(i);
								arma::Col<cpx> transformed_state = U * state;

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

	for (u64 i = 0; i < this->hamDouble->getHilbertSize(); i++) {
		// get the ed energy
		auto SORTIDX				=		idxs[i];
		auto EED					=		this->hamDouble->getEigVal(i);
		auto ENTED					=		ENTROPIES(La - 1, i);

		// sort
		auto ESYM					=		enSYMS[SORTIDX];
		const auto& [k,p,x,xY,xZ,u1]=		_symmetries[SORTIDX];
		auto ENTSYM					=		entSYMS[SORTIDX];

		printSeparatedP(file2, '\t', 15, false, 7, EED, ESYM, ENTED, ENTSYM);
		printSeparatedP(file2, ',', 3, true, 7, STR(k), STR(p), STR(x), STR(xY), STR(xZ), STR(u1), STR(_sizesSyms[i]));
	}

	// --------------- PRINT OUT THE DEGENERACIES ---------------
	printSeparated(file3, '\t', 15, true, infoHFull);
	printSeparated(file3, ',', 4, false, "T", "R", "PX", "PY", "PZ", "U1", "NH");
	printSeparated(file3, ',', 15, true, "DEG");
	for (uint i = 0; i < _degs.size(); i++)
	{
		auto [_syms, _degM]				=		_degs[i];
		const auto& [k,p,x,xY,xZ,u1]	=		_syms;
		printSeparatedP(file3, ',', 4, false, 7, STR(k), STR(p), STR(x), STR(xY), STR(xZ), STR(u1), STR(_sizes[i]));

		// degeneracy
		int _degC = 0;
		for (auto it = _degM.begin(); it != _degM.end(); it++)
			_degC						+=		it->first * it->second;
		LOGINFO(VEQ(_degC), LOG_TYPES::TRACE, 4);
		printSeparatedP(file3, ',', 15, true, 5, _degC);
	}

	file.close();
	file2.close();
	file3.close();
	LASTLVL = 0;
	LOGINFO("FINISHED ALL.", LOG_TYPES::TRACE, 0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<typename _T>
inline void UI::nqsSingle(clk::time_point start, std::shared_ptr<NQS<_T,cpx>> _NQS)
{
	LOGINFO("Started building the NQS Hamiltonian", LOG_TYPES::TRACE, 1);
	LOGINFO("Using: " + SSTR(getSTR_NQSTYPES(this->nqsP.type_)), LOG_TYPES::TRACE, 2);
	LASTLVL					=		2;
	std::string dir			=		this->mainDir;
	dir						+=		this->latP.lat->get_info() + kPS;

	std::string nqsInfo		=		_NQS->getInfo();
	std::string modelInfo   =		_NQS->H_->getInfo();
	dir						+=		modelInfo + kPS;
	createDir(dir);

	// calculate ED to compare with Lanczos
	u64 Nh					=		_NQS->getHilbertSize();
	if (Nh <= UI_LIMITS_NQS_ED) {
		_NQS->H_->hamiltonian();
		if (UI_LIMITS_NQS_LANCZOS_STATENUM < Nh)
			_NQS->H_->diagH(false, UI_LIMITS_NQS_LANCZOS_STATENUM, 0, 1000, 0.0, "sa");
		else
			_NQS->H_->diagH(false);
		LOGINFOG("Found the ED groundstate to be EED_0 = " + STRP(_NQS->H_->getEigVal(0), 7), 
				 LOG_TYPES::TRACE, 2);
	}

	// start the simulation
	arma::Col<cpx> _ENS		=		_NQS->train(this->nqsP.nMcSteps_,
												this->nqsP.nTherm_,
												this->nqsP.nBlocks_,
												this->nqsP.blockSize_,
												this->nqsP.nFlips_);
	arma::Mat<double> _ENSM(_ENS.size(), 2, arma::fill::zeros);
	_ENSM.col(0)			=		arma::real(_ENS);
	_ENSM.col(1)			=		arma::imag(_ENS);

	double ENQS_0			=		arma::mean(_ENSM.col(0).tail(10));
	LOGINFOG("Found the NQS groundstate to be ENQS_0 = " + STRP(ENQS_0, 7), LOG_TYPES::TRACE, 2);
	_ENSM.save(dir + "en_" + nqsInfo + ".dat", arma::raw_ascii);


}