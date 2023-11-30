#pragma once
#include "../../source/src/UserInterface/ui.h"

#ifdef DEBUG
//#define DEBUG_BINARY
#else
//#define OMP_NUM_THREADS 16;
#include <thread>
#include <mutex>
#endif

// ######################### RBM ############################
// definitions											 // #
#define NQS_RBM_ANGLES_UPD								 // #
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

// ######################## MODELS Q ########################
#ifndef SYK2_M_H										 // #
#include "../models/quadratic/SYK2.h"					 // #
#endif // !SYK2											 // #
#ifndef FF_M_H											 // #
#include "../models/quadratic/FreeFermions.h"			 // #
#endif // !SYK2											 // #
#ifndef AUBRY_ANDRE_M_H									 // #
#include "../models/quadratic/AubryAndre.h"				 // #
#endif // !SYK2											 // #
// ##########################################################

// ###################### LATTICES ##########################
#ifndef SQUARE_H										 // #
#include "../../source/src/Lattices/square.h"			 // #
#endif													 // #
#ifndef HEXAGONAL_H										 // #
#include "../../source/src/Lattices/hexagonal.h"		 // #
#endif													 // #
// ##########################################################

#define UI_HAAR_DIR std::string("G:\\My Drive\\Python\\Colab\\ProjectsData\\2023_Integrable_XYZ_XXZ\\DATA\\Haar_random\\")

// maximal ed size to compare
// ###################### LIMITS ############################
#define UI_ENERGYMEAN_SUBVEC(MCSTEPS, TROUT)					int(TROUT * MCSTEPS), MCSTEPS - int(TROUT * MCSTEPS) - 1
constexpr int UI_LIMITS_NQS_ED									= ULLPOW(16);
constexpr int UI_LIMITS_NQS_LANCZOS_STATENUM					= 100;


constexpr u64 UI_LIMITS_MAXFULLED								= ULLPOW(18);
constexpr u64 UI_LIMITS_MAXPRINT								= ULLPOW(3);
constexpr u64 UI_LIMITS_SI_STATENUM								= 100;
constexpr u64 UI_LIMITS_MIDDLE_SPEC_STATENUM					= 200;
// ##########################################################
#define UI_CHECK_SYM(val, gen)									if(this->val##_ != -INT_MAX) syms.push_back(std::make_pair(Operators::SymGenerators::gen, this->val##_));

// -------------------------------------------------------- make an USER INTERFACE class --------------------------------------------------------

namespace UI_PARAMS {

	/*
	* @brief Defines parameters used later for the models
	*/
	struct ModP {
		// ############### TYPE ################
		UI_PARAM_CREATE_DEFAULT(modTyp, MY_MODELS, MY_MODELS::ISING_M);
		UI_PARAM_CREATE_DEFAULT(modTypQ, MY_MODELS_Q, MY_MODELS_Q::SYK2_M);

		// ############### ISING ###############
		UI_PARAM_STEP(double, J1, 1.0);								// spin exchange
		UI_PARAM_STEP(double, hz, 1.0);								// perpendicular field
		UI_PARAM_STEP(double, hx, 1.0);								// transverse field
		// ############### XYZ #################		
		UI_PARAM_STEP(double, J2, 2.0);								// next nearest neighbors exchange
		UI_PARAM_STEP(double, eta1, 0.5);
		UI_PARAM_STEP(double, eta2, 0.5);
		UI_PARAM_STEP(double, dlt1, 0.3);
		UI_PARAM_STEP(double, dlt2, 0.3);
		// ############### KITAEV ##############
		UI_PARAM_STEP(double, kx, 1.0);								// spin exchange
		UI_PARAM_STEP(double, ky, 1.0);								// spin exchange
		UI_PARAM_STEP(double, kz, 1.0);								// spin exchange
		// ############ AUBRY_ANDRE ############
		UI_PARAM_STEP(double, Beta, (1 + std::sqrt(5)) / 2);	// phase mult
		UI_PARAM_STEP(double, Phi, 1.0);							// phase add


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
			// aubry-andre
			UI_PARAM_SET_DEFAULT_STEP(Beta);
			UI_PARAM_SET_DEFAULT_STEP(Phi);
		}

		/*
		* @brief Check whether the model itself is complex...
		*/
		bool checkComplex() const
		{
			if (this->modTypQ_ == MY_MODELS_Q::FREE_FERMIONS_M)
				return true;
			return false;
		}
	};

	/*
	* @brief Defines lattice used later for the models
	*/
	struct LatP {
		UI_PARAM_CREATE_DEFAULT(bc, BoundaryConditions, BoundaryConditions::PBC);
		UI_PARAM_CREATE_DEFAULT(typ, LatticeTypes, LatticeTypes::SQ);
		UI_PARAM_CREATE_DEFAULT(Lx, uint, 2);
		UI_PARAM_CREATE_DEFAULT(Ly, uint, 1);
		UI_PARAM_CREATE_DEFAULT(Lz, uint, 1);
		UI_PARAM_CREATE_DEFAULT(dim, uint, 1);

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
		UI_PARAM_CREATE_DEFAULT(S, bool, false);
		UI_PARAM_CREATE_DEFAULT(k, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(px, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(py, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(pz, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(x, int, -INT_MAX);
		UI_PARAM_CREATE_DEFAULT(U1, int, -INT_MAX);

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

		/*
		* @brief Checks if the symmetries make the Hamiltonian complex
		* @param Ns lattice size
		*/
		bool checkComplex(int Ns) {
			if (this->k_ == 0 || (this->k_ == Ns / 2 && Ns % 2 == 0) || this->py_ != -INT_MAX)
				return false;
			return true;
		}
	};

	// !TODO 
	// Neural network quantum states params
	struct NqsP {
		//unique_ptr<rbmState<_type, _hamtype>> phi;
		v_1d<u64> layersDim;
		UI_PARAM_CREATE_DEFAULT(type, NQSTYPES, NQSTYPES::RBM);
		UI_PARAM_CREATE_DEFAULT(nHidden, uint, 1);
		UI_PARAM_CREATE_DEFAULT(nVisible, uint, 1);
		UI_PARAM_CREATE_DEFAULT(nLayers, uint, 2);
		UI_PARAM_CREATE_DEFAULT(nFlips, uint, 1);
		UI_PARAM_CREATE_DEFAULT(blockSize, uint, 8);
		UI_PARAM_CREATE_DEFAULT(nTherm, uint, 50);
		UI_PARAM_CREATE_DEFAULT(nBlocks, uint, 500);
		UI_PARAM_CREATE_DEFAULT(nMcSteps, uint, 1000);
		UI_PARAM_CREATE_DEFAULT(batch, u64, 1024);
		UI_PARAM_CREATE_DEFAULTD(lr, double, 1);

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
	bool isComplex_ = false;
	bool useComplex_ = false;

	// ^^^^^^^^^ FOR DOUBLE ^^^^^^^^^
	Hilbert::HilbertSpace<double>					hilDouble;
	std::shared_ptr<Hamiltonian<double>>			hamDouble;
	std::shared_ptr<QuadraticHamiltonian<double>>	qhamDouble;

	// ^^^^^^^^ FOR COMPLEX ^^^^^^^^^
	Hilbert::HilbertSpace<cpx>						hilComplex;
	std::shared_ptr<Hamiltonian<cpx>>				hamComplex;
	std::shared_ptr<QuadraticHamiltonian<cpx>>		qhamComplex;

	// ^^^^^^^^^^^^^ NQS ^^^^^^^^^^^^
	std::shared_ptr<NQS<cpx, cpx>>			nqsCpx;
	std::shared_ptr<NQS<double, cpx>>		nqsDouble;

	// ##############################
	void setDefaultMap()					final override {
		this->defaultParams = {
			UI_OTHER_MAP(nqs		, this->nqsP.type_		, FHANDLE_PARAM_DEFAULT),			// type of the NQS state	
			UI_OTHER_MAP(m			, this->nqsP.nMcSteps_	, FHANDLE_PARAM_HIGHER0),			// mcsteps	
			UI_OTHER_MAP(b			, this->nqsP.batch_		, FHANDLE_PARAM_HIGHER0),			// batch
			UI_OTHER_MAP(nb			, this->nqsP.nBlocks_	, FHANDLE_PARAM_HIGHER0),			// number of blocks
			UI_OTHER_MAP(bs			, this->nqsP.blockSize_	, FHANDLE_PARAM_HIGHER0),			// block size
			UI_OTHER_MAP(nh			, this->nqsP.nHidden_	, FHANDLE_PARAM_HIGHER0),			// hidden params
			UI_OTHER_MAP(nf			, this->nqsP.nFlips_	, FHANDLE_PARAM_HIGHER0),			// hidden params

			{			"f"			, std::make_tuple(""	, FHANDLE_PARAM_DEFAULT)		},	// file to read from directory
			// ---------------- lattice parameters ----------------
			UI_OTHER_MAP(d			, this->latP._dim		, FHANDLE_PARAM_BETWEEN(1., 3.)),
			UI_OTHER_MAP(bc			, this->latP._bc		, FHANDLE_PARAM_BETWEEN(0., 3.)),
			UI_OTHER_MAP(l			, this->latP._typ		, FHANDLE_PARAM_BETWEEN(0., 1.)),
			UI_OTHER_MAP(lx			, this->latP._Lx		, FHANDLE_PARAM_HIGHER0),
			UI_OTHER_MAP(ly			, this->latP._Ly		, FHANDLE_PARAM_HIGHER0),
			UI_OTHER_MAP(lz			, this->latP._Lz		, FHANDLE_PARAM_HIGHER0),
			// ---------------- model parameters ----------------
			UI_OTHER_MAP(mod		, this->modP._modTyp	, FHANDLE_PARAM_BETWEEN(0., 2.)),
			// -------- ising
			UI_PARAM_MAP(J1			, this->modP._J1		, FHANDLE_PARAM_DEFAULT),
			UI_PARAM_MAP(hx			, this->modP._hx		, FHANDLE_PARAM_DEFAULT),
			UI_PARAM_MAP(hz			, this->modP._hz		, FHANDLE_PARAM_DEFAULT),
			// -------- heisenberg		
			UI_PARAM_MAP(dlt1		, this->modP._dlt1		, FHANDLE_PARAM_DEFAULT),
			// -------- xyz
			UI_PARAM_MAP(J2			, this->modP._J2		, FHANDLE_PARAM_DEFAULT),
			UI_PARAM_MAP(eta1		, this->modP._eta1		, FHANDLE_PARAM_DEFAULT),
			UI_PARAM_MAP(eta2		, this->modP._eta2		, FHANDLE_PARAM_DEFAULT),
			UI_PARAM_MAP(dlt2		, this->modP._dlt2		, FHANDLE_PARAM_DEFAULT),
			// -------- kitaev --------
			UI_PARAM_MAP(kx			, 0.0					, FHANDLE_PARAM_DEFAULT),
			UI_PARAM_MAP(ky			, 0.0					, FHANDLE_PARAM_DEFAULT),
			UI_PARAM_MAP(kz			, 0.0					, FHANDLE_PARAM_DEFAULT),
			// ----------- model quadratic parameters ------------
			UI_OTHER_MAP(mod		, this->modP._modTypQ	, FHANDLE_PARAM_BETWEEN(0., 3.)),
			// -------- aubry-andre
			UI_PARAM_MAP(Beta		, this->modP._Beta		, FHANDLE_PARAM_DEFAULT),
			UI_PARAM_MAP(Phi		, this->modP._Phi		, FHANDLE_PARAM_DEFAULT),


			// ---------------- symmetries ----------------
			UI_PARAM_MAP(ks			, this->symP._k			, FHANDLE_PARAM_HIGHER0),
			UI_PARAM_MAP(pxs		, this->symP._px		, FHANDLE_PARAM_BETWEEN()),
			UI_PARAM_MAP(pys		, this->symP._py		, FHANDLE_PARAM_BETWEEN()),
			UI_PARAM_MAP(pzs		, this->symP._pz		, FHANDLE_PARAM_BETWEEN()),
			UI_PARAM_MAP(xs			, this->symP._x			, FHANDLE_PARAM_BETWEEN()),
			UI_PARAM_MAP(u1s		, this->symP._U1		, FHANDLE_PARAM_DEFAULT),
			UI_PARAM_MAP(SYM		, this->symP._S			, FHANDLE_PARAM_BETWEEN(0., 1.)),		// even use symmetries?
			// ---------------- other ----------------
			UI_OTHER_MAP(fun		, -1.					, FHANDLE_PARAM_HIGHERV(-1.0)),			// choice of the function to be calculated
			UI_OTHER_MAP(th			, 1.0					, FHANDLE_PARAM_HIGHER0),				// number of threads
			UI_OTHER_MAP(q			, 0.0					, FHANDLE_PARAM_DEFAULT),				// quiet?
			UI_OTHER_MAP(dir		, "DEFALUT"				, FHANDLE_PARAM_DEFAULT),
		};
	};

private:
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% I N N E R    M E T H O D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	// ############# S Y M M E T R I E S   E D #############

	template<typename _T>
	void symmetries(clk::time_point start, std::shared_ptr<Hamiltonian<_T>> _H, bool _diag = true, bool _states = false);

	void symmetriesDeg(clk::time_point start);
	void symmetriesCreateDeg(clk::time_point start);

	void symmetriesTest(clk::time_point start);
	std::pair<v_1d<GlobalSyms::GlobalSym>, v_1d<std::pair<Operators::SymGenerators, int>>> createSymmetries();

	// ####################### N Q S #######################

	template<typename _T>
	void nqsSingle(clk::time_point start, std::shared_ptr<NQS<_T, cpx>> _NQS);

	// ##################### QUADRATIC #####################
	template<typename _T>
	void quadraticStatesMix(clk::time_point start, std::shared_ptr<QuadraticHamiltonian<_T>> _H);

	template<typename _T>
	void quadraticStatesToManyBody(clk::time_point start, std::shared_ptr<QuadraticHamiltonian<_T>> _H);

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% D E F I N I T I O N S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	bool defineModels(bool _createLat = true);
	bool defineModelsQ(bool _createLat = true);

	template<typename _T>
	bool defineModel(Hilbert::HilbertSpace<_T>& _Hil, std::shared_ptr<Hamiltonian<_T>>& _H);

	template<typename _T>
	bool defineModelQ(std::shared_ptr<QuadraticHamiltonian<_T>>& _H);

	template<typename _T>
	void defineNQS(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<NQS<_T, cpx>>& _NQS);

public:
	// -----------------------------------------------        CONSTRUCTORS  		-------------------------------------------
	~UI() = default;
	UI() = default;
	UI(int argc, char** argv) {
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

	void funChoice()						final override;
	void parseModel(int argc, cmdArg& argv) final override;

	// ----------------------------------------------- 			HELPERS  			-------------------------------------------

	void setDefault()						final override;

	// -----------------------------------------------  	   SIMULATION  		    -------------------------------------------	 

	// ######################### N Q S ##########################

	void makeSimNQS();

	// ####################### SYMMETRIES #######################

	void makeSimSymmetries(bool _diag = true, bool _states = false);

	void makeSimSymmetriesDeg();
	void makeSimSymmetriesCreateDeg();

	void makeSimSymmetriesSweep();
	void makeSimSymmetriesSweepHilbert();

	// ####################### QUADRATIC ########################

	void makeSimQuadratic();
	void makeSimQuadraticToManyBody();

};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# D E F I N E S ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Defines the interacting model based on the input file...
*/
template<typename _T>
inline bool UI::defineModel(Hilbert::HilbertSpace<_T>& _Hil, std::shared_ptr<Hamiltonian<_T>>& _H)
{
	bool _isGood = true;
	// get the symmetries
	auto [_glbSyms, _locSyms] = this->createSymmetries();
	_Hil = Hilbert::HilbertSpace<_T>(this->latP.lat, _locSyms, _glbSyms);
	if (_Hil.getHilbertSize() == 0)
	{
		LOGINFO("No states in the Hilbert space. Not creating model.", LOG_TYPES::INFO, 3);
		_isGood = false;
	}
	else
		LOGINFO(VEQVS(HilbertSize, _Hil.getHilbertSize()), LOG_TYPES::INFO, 3);

	// switch the model types
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
	return _isGood;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Defines the quadratic model based on the input file...
*/
template<typename _T>
inline bool UI::defineModelQ(std::shared_ptr<QuadraticHamiltonian<_T>>& _H)
{
	bool _isGood = true;

	// switch the type of quadratic model
	switch (this->modP.modTypQ_)
	{
	case MY_MODELS_Q::FREE_FERMIONS_M:
		_H = std::make_shared<FreeFermions<_T>>(this->latP.lat, this->modP.J1_, this->modP.J10_, 0.0);
		break;
	case MY_MODELS_Q::AUBRY_ANDRE_M:
		_H = std::make_shared<AubryAndre<_T>>(this->latP.lat, this->modP.J1_, this->modP.dlt1_, this->modP.Beta_, this->modP.Phi_,
			this->modP.J10_, this->modP.dlt10_, this->modP.Beta0_, this->modP.Phi0_, 0.0);
		break;
	case MY_MODELS_Q::SYK2_M:
		_H = std::make_shared<SYK2<_T>>(this->latP.lat, 0.0);
		break;
	default:
		_H = std::make_shared<FreeFermions<_T>>(this->latP.lat, this->modP.J1_, this->modP.J10_, 0.0);
		break;
	}
	return _isGood;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

// ##########################################################################################################################################
// ##########################################################################################################################################
// ########################################################## S Y M M E T R I E S ###########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Computes the Hamiltonian with symmetries and saves the entanglement entropies.
* @param start time of the beginning
* @param _H the shared pointer to the Hamiltonian - for convinience of the usage
* @info used in https://arxiv.org/abs/2303.13577
*/
template<typename _T>
inline void UI::symmetries(clk::time_point start, std::shared_ptr<Hamiltonian<_T>> _H, bool _diag, bool _states)
{
	LOGINFO(LOG_TYPES::TRACE, "", 40, '#', 1);
	u64 Nh = _H->getHilbertSize();
	// --- create the directories ---
	std::string dir = makeDirs(this->mainDir, _H->getType(), this->latP.lat->get_info());
	fs::create_directories(dir);

	// --- use those files --- 
	std::string modelInfo = _H->getInfo();
	std::string logMe = "";
	// --- save energies txt check ---
	std::string filename = dir + modelInfo;
	std::ofstream ofs(dir + "logHilbert.dat", std::ios_base::out | std::ios_base::app);
	auto memory = _H->getHamiltonianSizeH();
	strSeparatedS(logMe, ',', modelInfo, Nh, STRP(memory, 5));
	ofs << logMe << EL;
	ofs.close();

	// check Hilbert size or whether we should diagonalize and proceed further
	if (Nh == 0 || !_diag)
	{
		LOGINFO("Skipping creation of the Hamiltonian due to signal or empty Hilbert space.", LOG_TYPES::FINISH, 2);
		return;
	}

	// set the Hamiltonian
	_H->hamiltonian();

	// set the parameters
	uint Ns = _H->getNs();
	u64 stateNum = Nh;
	bool useShiftAndInvert = false;

	//stouts("\t->", start);
	LOGINFO("Finished buiding Hamiltonian" + modelInfo, LOG_TYPES::TRACE, 1);
	if (Nh < UI_LIMITS_MAXFULLED) {
		LOGINFO("Using standard diagonalization", LOG_TYPES::TRACE, 2);
		_H->diagH(false);
	}
	else
	{
		LOGINFO("Using S&I", LOG_TYPES::TRACE, 2);
		useShiftAndInvert = true;
		stateNum = UI_LIMITS_SI_STATENUM;
		_H->diagH(false, (int)stateNum, 0, 1000, 1e-5, "sa");
	}
	LOGINFO("Finished the diagonalization", LOG_TYPES::TRACE, 2);
	LOGINFO(start, "Diagonalization.", 2);

	std::string name = VEQ(Nh);
	LOGINFO("Spectrum size: " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states: " + STR(stateNum), LOG_TYPES::TRACE, 2);

	// save .h5
	_H->getEigVal(dir, HAM_SAVE_EXT::h5, false);

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
	v_1d<uint> _bonds = {};
	for (int i = 1; i <= maxBondNum; i++)
		_bonds.push_back(i);
	_bonds = { maxBondNum };
	// go entropies!
	auto beforeEntro = clk::now();
	_H->generateFullMap();
#pragma omp parallel for num_threads(this->threadNum)
	for (auto idx = 0LL; idx < stateNum; idx++) {
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
	LOGINFO(beforeEntro, "Entropies for all eigenstates using SCHMIDT.", 2);

	// save entropies file
	ENTROPIES.save(arma::hdf5_name(filename + ".h5", "entropy", arma::hdf5_opts::append));

	if (useShiftAndInvert)
		return;

	if (_states)
		_H->getEigVec(dir, stateNum, HAM_SAVE_EXT::h5, true);
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Allows to calculate how do the degeneracies look in case of symmetric Hamiltonians...
*/
inline void UI::symmetriesDeg(clk::time_point start)
{
	LOGINFO(LOG_TYPES::TRACE, "", 40, '#', 1);
	u64 Nh = this->hamComplex->getHilbertSize();
	// --- create the directories ---
	std::string dir = makeDirs(this->mainDir, this->hamComplex->getType(), this->latP.lat->get_info());
	fs::create_directories(dir);

	// --- use those files --- 
	std::string modelInfo = this->hamComplex->getInfo();

	// --- save entropies txt check ---
	std::string filename = dir + modelInfo;
	std::ofstream ofs(dir + modelInfo + ".dat", std::ios_base::out);

	// check Hilbert size or whether we should diagonalize and proceed further
	if (Nh == 0)
	{
		LOGINFO("Skipping creation of the Hamiltonian due to signal or empty Hilbert space.", LOG_TYPES::FINISH, 2);
		return;
	}

	// set the Hamiltonian
	this->hamComplex->hamiltonian();

	// set the parameters
	uint Ns = this->hamComplex->getNs();
	u64 stateNum = Nh / 2;
	auto ran = this->hamComplex->ran_;
	bool useShiftAndInvert [[maybe_unused]] = false;

	LOGINFO("Finished buiding Hamiltonian" + modelInfo, LOG_TYPES::TRACE, 1);
	if (Nh < UI_LIMITS_MAXFULLED) {
		LOGINFO("Using standard diagonalization", LOG_TYPES::TRACE, 2);
		this->hamComplex->diagH(false);
	}
	else
	{
		LOGINFO("Using S&I", LOG_TYPES::TRACE, 2);
		useShiftAndInvert = true;
		stateNum = UI_LIMITS_SI_STATENUM;
		this->hamComplex->diagH(false, (int)stateNum, 0, 1000, 1e-5, "sa");
	}
	LOGINFO("Finished the diagonalization", LOG_TYPES::TRACE, 2);
	LOGINFO(STR(t_ms(start)) + " ms", LOG_TYPES::TIME, 2);

	std::string name = VEQ(Nh);
	LOGINFO("Spectrum size: " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states: " + STR(stateNum), LOG_TYPES::TRACE, 2);
	this->hamComplex->getEigVal(dir, HAM_SAVE_EXT::h5, false);

	// clear energies
	this->hamComplex->clearH();

	// degeneracies
	v_2d<u64> degeneracyMap = this->hamComplex->getDegeneracies();

	// iterate through bond cut
	const uint maxBondNum = Ns / 2;

	// get the symmetry rotation matrix
	auto _symRot = this->hamComplex->getSymRot();
	bool usesSym = this->hamComplex->hilbertSpace.checkSym();
	auto beforeEntro = clk::now();
	this->hamComplex->generateFullMap();

	// ------------------------------- load the Haar mats -------------------------------
	// information about state combination
	const int _realizations = this->nqsP.nBlocks_;
	v_1d<int> _gammas = { };

	// append gammas - to be sure that we use the correct ones
	for (auto i = 0; i < degeneracyMap.size(); ++i)
		if (degeneracyMap[i].size() != 0)
			_gammas.push_back(i);

	v_2d<arma::Col<cpx>> _HM = {};

	// create random coeficients
	for (int i = 0; i < _gammas.size(); ++i)
	{
		if (_gammas[i] >= stateNum)
			break;

		_HM.push_back({});
		// go through gammas
		for (int j = 0; j < _realizations; ++j)
			if (_gammas[i] > 1)
				_HM[i].push_back(ran.createRanState(_gammas[i]));
			else
				_HM[i].push_back({ 1.0 });
	}

	// ------------------------------- go through all degeneracies (start with degeneracy 2) -------------------------------
	// get index of the average energy
	auto _idx [[maybe_unused]] = this->hamComplex->getEnAvIdx();
	u64 _minIdx = _idx - stateNum / 2;
	u64 _maxIdx = _idx + stateNum / 2;
	// as many entropies as realizations
	v_1d<double> entropies(_realizations);

	// go through the degeneracies again
	for (int _ig = 0; _ig < _gammas.size(); ++_ig)
	{
		auto _gamma = _gammas[_ig];
		int _degSize = degeneracyMap[_gamma].size();
		if (_degSize == 0)
			continue;
		LOGINFO("Doing: " + VEQ(_gamma) + ". Size=" + VEQ(_degSize), LOG_TYPES::TRACE, 1);

		// check the start and end idx from middle spectrum
		auto _idxStart = _degSize;
		auto _idxEnd = 0;
		for (auto i = 0; i < _degSize; i += _gamma)
			if (degeneracyMap[_gamma][i] >= _minIdx)
			{
				_idxStart = i;
				break;
			}
		for (auto i = _degSize - _gamma; i >= 0; i -= _gamma)
			if (degeneracyMap[_gamma][i] <= _maxIdx)
			{
				_idxEnd = i;
				break;
			}
		if (_idxStart >= _idxEnd)
		{
			LOGINFO("for: " + VEQ(_gamma) + ". Cannot find states in the middle of the spectrum", LOG_TYPES::TRACE, 1);
			continue;
		}
		LOGINFO("Choosing from: [" + VEQ(_idxStart) + "," + VEQ(_idxEnd) + "]", LOG_TYPES::TRACE, 1);

		// go through the degeneracies
#pragma omp parallel for num_threads(this->threadNum)
		for (int _r = 0; _r < _realizations; _r++)
		{
			// choose indices from degenerate states in the manifold
			//auto indices				=	(gamma > 1) ? ran.choice(toChooseFrom, gamma) : v_1d<int>({ ran.randomInt<int>(_minIdx, _minIdx + stateNum - 1) });
			auto idx = ran.randomInt<int>(_idxStart, _idxEnd + 1);
			auto idxState = degeneracyMap[_gamma][idx];

			arma::Col<cpx> _state = _HM[_ig][_r][0] * this->hamComplex->getEigVec(idxState);
			// append states
			for (int id = 1; id < _gamma; id++)
			{
				idxState = degeneracyMap[_gamma][idx + id];
				_state += _HM[_ig][_r][id] * this->hamComplex->getEigVec(idxState);
			}

			// rotate!
			if (usesSym)
				_state = _symRot * _state;

			// normalize state
			//_state						=	_state / sqrt(arma::cdot(_state, _state));
			auto entro = Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_state, maxBondNum, this->hamComplex->hilbertSpace);

			entropies[_r] = entro;
			if (_r > 10 && _r % (_realizations / 10) == 0)
				LOGINFO("Finished: " + STR(_r * 100.0 / _realizations) + "%. Entropy was: " + VEQ(entro / (log(2) * maxBondNum)), LOG_TYPES::TRACE, 4);

		}
		// print those guys
		for (int _r = 0; _r < _realizations; _r++)
			ofs << "," << STRP(entropies[_r], 10);
		ofs << "\n";
	}
	LOGINFO("Finished entropies!", LOG_TYPES::TRACE, 2);
	LOGINFO(STR(t_ms(beforeEntro)) + " ms", LOG_TYPES::TIME, 2);
	ofs.close();
}

/*
* @brief Use this function to create a combination of states in the middle of the spectrum and calculate their entanglement entropies
*/
inline void UI::symmetriesCreateDeg(clk::time_point start)
{
	//	LOGINFO(LOG_TYPES::TRACE, "", 40, '#', 1);
	//	u64 Nh					=			this->hamComplex->getHilbertSize();
	//	// --- create the directories ---
	//	std::string dir			=			makeDirs(this->mainDir, this->hamComplex->getType(), this->latP.lat->get_info(), "combinations");
	//	fs::create_directories(dir);
	//
	//	// --- use those files --- 
	//	std::string modelInfo	=			this->hamComplex->getInfo();
	//
	//	// --- save entropies txt check ---
	//	std::string filename	=			dir + modelInfo;
	//	std::ofstream ofs(dir + modelInfo + ".dat", std::ios_base::out);
	//	
	//	// check Hilbert size or whether we should diagonalize and proceed further
	//	if (Nh == 0)
	//	{
	//		LOGINFO("Skipping creation of the Hamiltonian due to signal or empty Hilbert space.", LOG_TYPES::FINISH, 2);
	//		return;
	//	}
	//
	//	// set the Hamiltonian
	//	this->hamComplex->hamiltonian();
	//
	//	// set the parameters
	//	uint Ns					=			this->hamComplex->getNs();
	//	u64 stateNum			=			Nh / 10;
	//	auto ran				=			this->hamComplex->ran_;
	//	bool useShiftAndInvert [[maybe_unused]] = false;
	//
	//	LOGINFO("Finished buiding Hamiltonian" + modelInfo, LOG_TYPES::TRACE, 1);
	//	if (Nh < UI_LIMITS_MAXFULLED) {
	//		LOGINFO("Using standard diagonalization", LOG_TYPES::TRACE, 2);
	//		this->hamComplex->diagH(false);
	//	}
	//	else
	//	{
	//		LOGINFO("Using S&I", LOG_TYPES::TRACE, 2);
	//		useShiftAndInvert	=			true;
	//		stateNum = UI_LIMITS_SI_STATENUM;
	//		this->hamComplex->diagH(false, (int)stateNum, 0, 1000, 1e-5, "sa");
	//	}
	//	LOGINFO("Finished the diagonalization", LOG_TYPES::TRACE, 2);
	//	LOGINFO(STR(t_ms(start)) + " ms", LOG_TYPES::TIME, 2);
	//
	//	std::string name		=			VEQ(Nh);
	//	LOGINFO("Spectrum size: "			+ STR(Nh)		, LOG_TYPES::TRACE, 3);
	//	LOGINFO("Taking num states: "		+ STR(stateNum)	, LOG_TYPES::TRACE, 2);
	//	this->hamComplex->getEigVal(dir, HAM_SAVE_EXT::h5, false);
	//
	//	// clear energies
	//	this->hamComplex->clearH();
	//
	//	// iterate through bond cut
	//	const uint maxBondNum	=			Ns / 2;
	//
	//	// get the symmetry rotation matrix
	//	auto _symRot			=			this->hamComplex->getSymRot();
	//	bool usesSym			=			this->hamComplex->hilbertSpace.checkSym();
	//	auto beforeEntro		=			clk::now();
	//	this->hamComplex->generateFullMap();
	//
	//	// information about state combination
	//	const int _realizations	=			this->nqsP.nBlocks_;
	//	const v_1d<int> _gammas	=			{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50 };
	//	v_2d<arma::Col<cpx>> _HM=			{};
	//
	//	// create random coeficients
	//	for (int i = 0; i < _gammas.size(); ++i)
	//	{
	//		if (_gammas[i] >= stateNum)
	//			break;
	//
	//		_HM.push_back({});
	//		// go through gammas
	//		for (int j = 0; j < _realizations; ++j)
	//			if(_gammas[i] > 1)
	//				_HM[i].push_back(ran.createRanState(_gammas[i]));
	//			else
	//				_HM[i].push_back({ 1.0 });
	//	}
	//
	//	// get index of the average energy
	//	auto _idx [[maybe_unused]]	= this->hamComplex->getEnAvIdx();
	//	u64 _minIdx					= _idx - stateNum / 2;
	//	//u64 _maxIdx					= _idx + stateNum / 2;
	//	v_1d<int> toChooseFrom(stateNum);
	//	std::iota(toChooseFrom.begin(), toChooseFrom.end(), _minIdx);
	//	// as many entropies as realizations
	//	v_1d<double> entropies(_realizations);
	//
	//	// go through the gammas
	//	for (int _ig = 0; _ig < _gammas.size(); ++_ig)
	//	{
	//		auto gamma = _gammas[_ig];
	//
	//		// too many states
	//		if (gamma >= stateNum)
	//			break;
	//
	//		ofs << gamma;
	//		LOGINFO("Taking gamma: " + STR(gamma), LOG_TYPES::TRACE, 3);
	//
	//		// go through the realizations
	//#pragma omp parallel for num_threads(this->threadNum)
	//		for (int _r = 0; _r < _realizations; _r++)
	//		{
	//			// choose random indices
	//			//auto indices				=	(gamma > 1) ? ran.choice(toChooseFrom, gamma) : v_1d<int>({ ran.randomInt<int>(_minIdx, _minIdx + stateNum - 1) });
	//			auto idx					=	ran.randomInt<int>(_minIdx, _minIdx + stateNum - 1 - gamma);
	//			//arma::Col<cpx> _state		=	_HM[_ig][_r][0] * this->hamComplex->getEigVec(indices[0]);
	//			arma::Col<cpx> _state		=	_HM[_ig][_r][0] * this->hamComplex->getEigVec(idx);
	//			// append states
	//			for (int id = 1; id < gamma; id++)
	//				_state					+=	_HM[_ig][_r][id] * this->hamComplex->getEigVec(idx + id);
	//			// rotate!
	//			if (usesSym)
	//				_state					=	_symRot * _state;
	//			// normalize state
	//			//_state						=	_state / sqrt(arma::cdot(_state, _state));
	//			auto entro					=	Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_state, maxBondNum, this->hamComplex->hilbertSpace);
	//
	//			entropies[_r] = entro;
	//			if (_r > 10 && _r % (_realizations / 10) == 0)
	//				LOGINFO("Finished: " + STR(_r * 100.0 / _realizations) + "%. Entropy was: " + VEQ(entro / (log(2) * maxBondNum)), LOG_TYPES::TRACE, 4);
	//
	//		}
	//		for (int _r = 0; _r < _realizations; _r++)
	//			ofs << "," << STRP(entropies[_r], 10);
	//		ofs << "\n";
	//	}
	//	ofs.close();
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Tests the currently implemented symmetries based on specific model parameters
* @param start the timer start
*/
inline void UI::symmetriesTest(clk::time_point start)
{
	using symTuple = std::tuple<int, int, int, int, int, int>;
	v_1d<double> entFULL = {};
	v_1d<double> entSYMS = {};
	v_1d<double> enSYMS = {};
	v_1d<symTuple> _symmetries = {};
	v_1d<u64> _sizesSyms = {};
	v_1d<std::pair<symTuple, std::map<int, int>>> _degs = {};
	v_1d<u64> _sizes = {};
	uint Ns = 1;
	u64 Nh = 1;
	u64 NhFull = 1;
	uint La = Ns / 2;
	std::string dir = "";
	std::string infoHFull = "";
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
		this->hamDouble->hamiltonian();
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

							this->hamComplex->hamiltonian();
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
							for (u64 i = 0; i < Nh; i++) {
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

	for (u64 i = 0; i < this->hamDouble->getHilbertSize(); i++) {
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

// ##########################################################################################################################################
// ##########################################################################################################################################
// ######################################################### V A R I A T I O N A L ##########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

template<typename _T>
inline void UI::nqsSingle(clk::time_point start, std::shared_ptr<NQS<_T, cpx>> _NQS)
{
	LOGINFO("Started building the NQS Hamiltonian", LOG_TYPES::TRACE, 1);
	LOGINFO("Using: " + SSTR(getSTR_NQSTYPES(this->nqsP.type_)), LOG_TYPES::TRACE, 2);
	LOGINFO_CH_LVL(2);
	std::string dir = this->mainDir;
	dir += this->latP.lat->get_info() + kPS;

	std::string nqsInfo = _NQS->getInfo();
	std::string modelInfo = _NQS->H_->getInfo();
	dir += modelInfo + kPS;
	createDir(dir);

	// calculate ED to compare with Lanczos
	u64 Nh = _NQS->getHilbertSize();
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
	arma::Col<cpx> _ENS = _NQS->train(this->nqsP.nMcSteps_,
		this->nqsP.nTherm_,
		this->nqsP.nBlocks_,
		this->nqsP.blockSize_,
		this->nqsP.nFlips_);
	arma::Mat<double> _ENSM(_ENS.size(), 2, arma::fill::zeros);
	_ENSM.col(0) = arma::real(_ENS);
	_ENSM.col(1) = arma::imag(_ENS);

	double ENQS_0 = arma::mean(_ENSM.col(0).tail(10));
	LOGINFOG("Found the NQS groundstate to be ENQS_0 = " + STRP(ENQS_0, 7), LOG_TYPES::TRACE, 2);
	_ENSM.save(dir + "en_" + nqsInfo + ".dat", arma::raw_ascii);


}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ########################################################### Q U A D R A T I C ############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

template<typename _T>
inline void UI::quadraticStatesMix(clk::time_point start, std::shared_ptr<QuadraticHamiltonian<_T>> _H)
{
	LOGINFO(LOG_TYPES::TRACE, "", 40, '#', 1);
	// --- create the directories ---
	std::string dir = makeDirs(this->mainDir, _H->getType(), this->latP.lat->get_info());
	fs::create_directories(dir);

	// ------ use those files -------
	std::string modelInfo = _H->getInfo();

	// set the parameters
	uint Ns = _H->getNs();
	uint Nh = _H->getHilbertSize();
	// how many states to take for calculating the entropies
	u64 stateNum = this->nqsP.nMcSteps_;
	// how many states to take for the average
	u64 realizations = this->nqsP.nBlocks_;
	// number of combinations to take from single particle states
	u64 combinations = this->nqsP.blockSize_;
	auto _type = _H->getTypeI();

	// --- save energies txt check ---
	std::string filename = filenameQuadraticRandom(dir + modelInfo + "_Gamma=" + STR(stateNum),
		_type, _H->ran_);
	if (combinations < stateNum)
	{
		LOGINFO("Bad number of combinations. Must be bigger than the number of states", LOG_TYPES::ERROR, 0);
		throw std::runtime_error("BAD COMBINATIONS");
	}

	// check the model (if necessery to build hamilonian, do it)
	if (_type != (uint)MY_MODELS_Q::FREE_FERMIONS_M)
	{
		_H->hamiltonian();
		LOGINFO("Finished buiding Hamiltonian" + modelInfo, LOG_TYPES::TRACE, 1);
		_H->diagH(false);
		LOGINFO(STR(t_ms(start)) + " ms", LOG_TYPES::TIME, 2);
	}
	// obtain the single particle energies
	arma::Mat<_T> W = _H->getTransMat();
	LOGINFO("Finished creating matrices" + modelInfo, LOG_TYPES::TRACE, 1);

	std::string name = VEQ(Nh);
	LOGINFO("Spectrum size:						  " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states (combinations):	  " + STR(stateNum), LOG_TYPES::TRACE, 2);
	LOGINFO("Taking num realizations (averages): " + STR(realizations), LOG_TYPES::TRACE, 2);

	// save single particle energies
	if (!fs::exists(dir + modelInfo + ".h5"))
		_H->getEigVal(dir, HAM_SAVE_EXT::h5, false);

	// iterate through bond cut
	v_1d<uint> _bonds = { uint(Ns / 4), uint(Ns / 2) };
	arma::mat ENTROPIES(_bonds.size(), realizations, arma::fill::zeros);

	auto beforeEntro = clk::now();

	std::vector<double> energies;
	std::vector<arma::uvec> orbs;
	// get the states to use later
	if (_H->getTypeI() != (uint)MY_MODELS_Q::FREE_FERMIONS_M || Ns % 8 != 0)
		_H->getManyBodyEnergies(Ns / 2, energies, orbs, combinations);
	else
		_H->getManyBodyEnergiesZero(Ns / 2, energies, orbs, combinations);
	LOGINFO("Combinations time: " + STR(t_ms(beforeEntro)) + " ms", LOG_TYPES::TIME, 2);

	// make matrices cut to a specific number of bonds
	std::vector<arma::Mat<_T>> Ws;
	// conjugate transpose it - to be used later
	std::vector<arma::Mat<_T>> WsC;
	for (int i = 0; i < _bonds.size(); ++i)
	{
		arma::Mat<_T> _subM = W.submat(0, 0, W.n_rows - 1, _bonds[i] - 1);
		Ws.push_back(_subM);
		WsC.push_back(_subM.t());
	}

	// indices of orbitals
	std::vector<uint> idxs(orbs.size());
	std::iota(idxs.begin(), idxs.end(), 0);

	pBar pbar(5, realizations);
	// calculate!
#pragma omp parallel for num_threads(this->threadNum)
	for (auto idx = 0LL; idx < realizations; idx++)
	{
		// create vector of orbitals (combine two many-body states from our total number of combinations)
		auto orbitalIndices = _H->ran_.choice(idxs, stateNum);
		v_1d<arma::uvec> orbitals;
		for (auto idxO : orbitalIndices)
			orbitals.push_back(orbs[idxO]);

		// generate coefficients (create random state consisting of stateNum = gamma states)
		auto coeff = _H->ran_.createRanState(stateNum);

		// go through bonds (La)
		uint _bondI = 0;
		for (auto i [[maybe_unused]] : _bonds) {
			// iterate through the state
			auto J = SingleParticle::corrMatrix<_T>(Ns, Ws[_bondI], WsC[_bondI], orbitals, coeff, _H->ran_);
			auto E = Entropy::Entanglement::Bipartite::SingleParticle::vonNeuman(J);
			// save the entropy
			ENTROPIES(_bondI, idx) = E;
			_bondI++;
		}
		try {
			if (idx % pbar.percentageSteps == 0)
				pbar.printWithTime(LOG_LVL3 + SSTR("PROGRESS"));
		}
		catch (std::exception& e)
		{
			LOGINFO("Couldn't print progress: " + SSTR(e.what()), LOG_TYPES::WARNING, 3);
		}
	}
	// save entropies file
	ENTROPIES.save(arma::hdf5_name(filename + ".h5", "entropy"));

	// save means
	try {
		arma::Col<double> means(2, arma::fill::zeros);
		std::ofstream ofs(dir + modelInfo + "_meanE.dat", std::ios_base::app);
		for (int i = 0; i < _bonds.size(); i++)
		{
			auto meanE = arma::mean(ENTROPIES.row(i));
			LOGINFO("Bond: " + STR(_bonds[i]) + "->Entropy: " + STRP(meanE, 12), LOG_TYPES::TRACE, 3);
			LOGINFO(STRP(meanE / (_bonds[i] * std::log(2.0)), 12), LOG_TYPES::TRACE, 4);
			ofs << stateNum << "," << meanE / (_bonds[i] * std::log(2.0)) << ",";
		}
		ofs << EL;
		ofs.close();
	}
	catch (std::exception& e)
	{
		LOGINFO("Couldn't save the averages... No worries...", LOG_TYPES::WARNING, 1);
	}

	// save entropies
	LOGINFO("Finished entropies! " + VEQ(stateNum) + ", " + VEQ(realizations), LOG_TYPES::TRACE, 2);
	LOGINFO(STR(t_ms(beforeEntro)) + " ms", LOG_TYPES::TIME, 2);
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Creates some noninteracting states and transforms them to many body
*/
template<typename _T>
inline void UI::quadraticStatesToManyBody(clk::time_point start, std::shared_ptr<QuadraticHamiltonian<_T>> _H)
{
	LOGINFO(LOG_TYPES::TRACE, "", 40, '#', 1);
	this->_timer.reset();

	// --- create the directories ---
	std::string dir = makeDirs(this->mainDir, _H->getType(), this->latP.lat->get_info());
	fs::create_directories(dir);

	// ------ use those files -------
	std::string modelInfo	= _H->getInfo();
	uint Ns					= _H->getNs();
	uint Nh					= _H->getHilbertSize();
	// how many states to take for calculating the entropies
	u64 stateNum			= this->nqsP.nMcSteps_;
	// how many states to take for the average
	u64 realizations		= this->nqsP.nBlocks_;
	// number of combinations to take from single particle states
	u64 combinations		= this->nqsP.blockSize_;
	auto _type				= _H->getTypeI();

	// --- save energies txt check ---
	std::string filename	= filenameQuadraticRandom(dir + modelInfo + "_Gamma=" + STR(stateNum), _type, _H->ran_);

	// check how many states maximally can one combine
	if (combinations < stateNum)
	{
		LOGINFO("Bad number of combinations. Must be bigger than the number of states", LOG_TYPES::ERROR, 0);
		throw std::runtime_error("BAD COMBINATIONS");
	}

	// check the model (if necessery to build hamilonian, do it)
	if (_type != (uint)MY_MODELS_Q::FREE_FERMIONS_M)
	{
		_H->hamiltonian();
		LOGINFO("Finished buiding Hamiltonian" + modelInfo, LOG_TYPES::TRACE, 1);
		_H->diagH(false);
		LOGINFO(start, __FUNCTION__, 3);
	}
	// obtain the single particle energies
	arma::Mat<_T> W = _H->getTransMat();
	_H->getSPEnMat();

	// go through the information
	LOGINFO("Spectrum size:						  " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states (combinations):	  " + STR(stateNum), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num realizations (averages):  " + STR(realizations), LOG_TYPES::TRACE, 3);

	// save single particle energies
	if (!fs::exists(dir + modelInfo + ".h5"))
		_H->getEigVal(dir, HAM_SAVE_EXT::h5, false);

	// iterate through bond cut
	uint _bonds				= uint(Ns / 2);
	arma::vec ENTROPIES_SP(realizations, arma::fill::zeros);
	arma::vec ENTROPIES_MB(realizations, arma::fill::zeros);

	// set which bonds we want to cut in bipartite
	_timer.checkpoint("entropy");
	std::vector<double> energies;
	std::vector<arma::uvec> orbs;

	// get the states to use later
	_H->getManyBodyEnergies(Ns / 2, energies, orbs, combinations);
	LOGINFO(_timer.point("entropy"), "Combinations time:", 3);

	// make matrices cut to a specific number of bonds
	arma::Mat<_T> Ws		= W.submat(0, 0, W.n_rows - 1, _bonds - 1);;
	// conjugate transpose it - to be used later
	arma::Mat<_T> WsC		= Ws.t();

	// indices of orbitals
	auto idxs				= VEC::vecAtoB(orbs.size());
	// indices of orbitals

	// Hilbert space
	auto _hilbert			= Hilbert::HilbertSpace<_T>(this->latP.lat);

	// calculate!
	pBar pbar(5, realizations, _timer.point(0));
//#pragma omp parallel for num_threads(this->threadNum)
	for (u64 idx = 0; idx < realizations; idx++)
	{
		// create vector of orbitals (combine two many-body states from our total number of combinations)
		auto orbitalIndices = _H->ran_.choice(idxs, stateNum);
		v_1d<arma::uvec> orbitals;
		for (auto& idxO : orbitalIndices)
			orbitals.push_back(orbs[idxO]);

		// generate coefficients (create random state consisting of stateNum = gamma states)
		auto coeff			= _H->ran_.createRanState(stateNum);

		// -------------------------------- CORRELATION --------------------------------

		// iterate through the state
		auto J = SingleParticle::corrMatrix<_T>(Ns, Ws, WsC, orbitals, coeff, _H->ran_);
		auto E = Entropy::Entanglement::Bipartite::SingleParticle::vonNeuman(J);
		// save the entropy
		ENTROPIES_SP(idx)	= E;

		// --------------------------------- MANY BODY ---------------------------------
		arma::Col<_T> _state(_hilbert.getHilbertSize(), arma::fill::zeros);
		for (int i = 0; i < orbitals.size(); ++i)
		{
			std::complex<double> _c = coeff(i);
			_state			+= _c * _H->getManyBodyState(VEC::colToVec(orbitals[i]), _hilbert);
		}
		_state				= _state / arma::cdot(_state, _state);
		E = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_state, _bonds, _hilbert);
		ENTROPIES_MB(idx)	= E;

		// update progress
		PROGRESS_UPD(idx, pbar);
	}
	// save entropies file
	ENTROPIES_MB.save(arma::hdf5_name(filename + "_MB.h5", "entropy"));
	ENTROPIES_SP.save(arma::hdf5_name(filename + "_SP.h5", "entropy"));

	// save entropies
	LOGINFO("Finished entropies! " + VEQ(stateNum) + ", " + VEQ(realizations), LOG_TYPES::TRACE, 2);
	LOGINFO(_timer.point("entropy"), "Entropies time:", 3);
	return;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
