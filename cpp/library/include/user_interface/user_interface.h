#pragma once
/***************************************
* Defines the user interface class based
* on a general UI class. All methods for 
* this software stored.
* DEC 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#include "../../source/src/UserInterface/ui.h"

#ifdef _DEBUG
//	#define DEBUG_BINARY
#else
//	#define OMP_NUM_THREADS 16;
#	include <thread>
#	include <mutex>
#endif

// ######################### NQS ############################
// save the weights?									 // #
#define NQS_SAVE_WEIGHTS								 // #
#ifdef NQS_SAVE_WEIGHTS									 // #
#	define NQS_SAVE_DIR "WEIGHTS" + kPS					 // #
#endif													 // #
														 // #
// use CPU?												// #
#define NQS_USE_CPU										 // #
//#define NQS_USE_GPU									// #
														// #
#ifdef NQS_USE_CPU										 // #
#	define NQS_USE_MULTITHREADING						 // #
//#	define NQS_USE_OMP									 // #
#elif defined NSQ_USE_GPU								 // #
// something											 // #
#endif													 // #
														 // #
// definitions											 // #
#define NQS_ANGLES_UPD									 // #
														 // #
// use vector only?										 // #
#define NQS_USE_VEC_ONLY								 // #
#define NQS_USESR										 // #
#ifdef NQS_USESR										 // #
// how to handle the inverse of the matrix				// #
//#	define NQS_PINV 1e-3								// #
// regularization for the covariance matrix				// #
//#	define NQS_SREG										// #												  
#endif													 // #
#ifndef RBMPP_H											 // #
#	include "../NQS/rbm_pp.h"							 // #
#endif													 // #
#ifndef RBM_H											 // #
#	include "../NQS/rbm.h"								 // #
#endif													 // #
// ##########################################################


// ######################### MODELS #########################
#ifndef ISING_H											 // #
#include "../models/ising.h"							 // #
#endif // !ISING_H										 // #
#ifndef XYZ_H											 // #
#include "../models/XYZ.h"								 // #
#endif // !XYZ_H										 // #
#ifndef HEISENBERG_KITAEV_H								 // #
#include "../models/heisenberg-kitaev.h"				 // #
#endif													 // #
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

// ###################### LIMITS ############################

#define UI_ENERGYMEAN_SUBVEC(MCSTEPS, TROUT)					int(TROUT * MCSTEPS), MCSTEPS - int(TROUT * MCSTEPS) - 1
// --- NQS
constexpr int UI_LIMITS_NQS_ED									= ULLPOW(18);
constexpr int UI_LIMITS_NQS_FULLED								= ULLPOW(12);
constexpr int UI_LIMITS_NQS_LANCZOS_STATENUM					= 100;

// --- ED
constexpr u64 UI_LIMITS_MAXFULLED								= ULLPOW(18);
constexpr u64 UI_LIMITS_MAXPRINT								= ULLPOW(3);
constexpr u64 UI_LIMITS_SI_STATENUM								= 100;
constexpr u64 UI_LIMITS_MIDDLE_SPEC_STATENUM					= 200;
// --- QUADRATIC
constexpr int UI_LIMITS_QUADRATIC_COMBINATIONS					= 24;

// ##########################################################

#define UI_CHECK_SYM(val, gen)									if(this->val##_ != -INT_MAX) syms.push_back(std::make_pair(Operators::SymGenerators::gen, this->val##_));

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################## P A R A M S ###############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

namespace UI_PARAMS {

	// ----------------------------------------------------------------

	/*
	* @brief Defines parameters used later for the models
	*/
	struct ModP {
		// ############### TYPE ################
		UI_PARAM_CREATE_DEFAULT(modTyp, MY_MODELS, MY_MODELS::ISING_M);
		
		// #####################################
		// ####### I N T E R A C T I N G #######
		// #####################################

		// ############## ISING ################
		UI_PARAM_STEP(double, J1, 1.0);								// spin exchange
		UI_PARAM_STEP(double, hz, 1.0);								// perpendicular field
		UI_PARAM_STEP(double, hx, 1.0);								// transverse field
		// ############### XYZ #################		
		UI_PARAM_STEP(double, J2, 2.0);								// next nearest neighbors exchange
		UI_PARAM_STEP(double, eta1, 0.5);
		UI_PARAM_STEP(double, eta2, 0.5);
		UI_PARAM_STEP(double, dlt1, 0.3);
		UI_PARAM_STEP(double, dlt2, 0.3);

		// ############# KITAEV ################

		v_1d<double> Kx_;
		v_1d<double> Ky_;
		v_1d<double> Kz_;
		v_1d<double> heiJ_;
		v_1d<double> heiDlt_;
		v_1d<double> heiHx_;
		v_1d<double> heiHz_;

		// #####################################
		// ######### Q U A D R A T I C #########
		// #####################################

		UI_PARAM_CREATE_DEFAULT(modTypQ, MY_MODELS_Q, MY_MODELS_Q::SYK2_M);
		
		// for simulation
		UI_PARAM_CREATE_DEFAULT(q_gamma, uint, 1);					// mixing quadratic states
		UI_PARAM_CREATE_DEFAULT(q_manifold, bool, false);			// use the degenerate manifold?
		UI_PARAM_CREATE_DEFAULT(q_manybody, bool, true);			// use the many body calculation?
		UI_PARAM_CREATE_DEFAULT(q_randomCombNum, uint, 100);		// number of random combinations for the average (to choose from)
		UI_PARAM_CREATE_DEFAULT(q_realizationNum, uint, 100);		// number of realizations for the average
		UI_PARAM_CREATE_DEFAULT(q_shuffle, bool, true);				// shuffle the states?

		// ########### AUBRY_ANDRE #############

		UI_PARAM_STEP(double, Beta, (1 + std::sqrt(5)) / 2);	// phase mult
		UI_PARAM_STEP(double, Phi, 1.0);							// phase add

		// -------------------------------------
		void setDefault() 
		{
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
			this->Kx_		= v_1d<double>(1, 1.0);
			this->Ky_		= v_1d<double>(1, 1.0);
			this->Kz_		= v_1d<double>(1, 1.0);
			this->heiJ_		= v_1d<double>(1, 1.0);
			this->heiDlt_	= v_1d<double>(1, 1.0);
			this->heiHz_	= v_1d<double>(1, 1.0);
			this->heiHx_	= v_1d<double>(1, 1.0);
			// aubry-andre
			UI_PARAM_SET_DEFAULT_STEP(Beta);
			UI_PARAM_SET_DEFAULT_STEP(Phi);
		}

		// -------------------------------------

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

	// ----------------------------------------------------------------

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

	// ----------------------------------------------------------------

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
		bool checkComplex(int Ns) const {
			if (this->k_ == 0 || (this->k_ == Ns / 2 && Ns % 2 == 0) || this->py_ != -INT_MAX)
				return false;
			return true;
		}
	};

	// ----------------------------------------------------------------

	// !TODO 
	// Neural network quantum states params
	struct NqsP 
	{
		v_1d<u64> layersDim;
		UI_PARAM_CREATE_DEFAULT(type, NQSTYPES, NQSTYPES::RBM_T);
		UI_PARAM_CREATE_DEFAULT(nHidden, uint, 1);
		UI_PARAM_CREATE_DEFAULT(nVisible, uint, 1);
		UI_PARAM_CREATE_DEFAULT(nLayers, uint, 2);
		UI_PARAM_CREATE_DEFAULT(nFlips, uint, 1);
		UI_PARAM_CREATE_DEFAULT(blockSize, uint, 8);
		UI_PARAM_CREATE_DEFAULT(nTherm, uint, 50);
		UI_PARAM_CREATE_DEFAULT(nBlocks, uint, 500);
		UI_PARAM_CREATE_DEFAULT(nMcSteps, uint, 1000);
		// for collecting
		UI_PARAM_CREATE_DEFAULT(nMcSamples, uint, 100);
		UI_PARAM_CREATE_DEFAULT(nSBlocks, uint, 100);
		UI_PARAM_CREATE_DEFAULT(blockSizeS, uint, 8);
		UI_PARAM_CREATE_DEFAULTD(lr, double, 1);
		// weight load directory
		inline static const std::string _loadNQS	= ""; 
		std::string loadNQS_								= "";

		void setDefault() 
		{
			UI_PARAM_SET_DEFAULT(nHidden);
			UI_PARAM_SET_DEFAULT(nVisible);
			UI_PARAM_SET_DEFAULT(nLayers);
			UI_PARAM_SET_DEFAULT(nFlips);
			UI_PARAM_SET_DEFAULT(blockSize);
			UI_PARAM_SET_DEFAULT(nTherm);
			UI_PARAM_SET_DEFAULT(nBlocks);
			UI_PARAM_SET_DEFAULT(nMcSteps);
			UI_PARAM_SET_DEFAULT(lr);
			UI_PARAM_SET_DEFAULT(loadNQS);
			// collection
			UI_PARAM_SET_DEFAULT(nMcSamples);
			UI_PARAM_SET_DEFAULT(nSBlocks);
			UI_PARAM_SET_DEFAULT(blockSizeS);
		}
	};
};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# U I N T E R F ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief User interface class for the QES
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

	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	
	// define basic models
	bool isComplex_		= false;						// checks the complex sector
	bool useComplex_	= false;						// forces complex sector choice

	// ^^^^^^^^^ FOR DOUBLE ^^^^^^^^^					
	Hilbert::HilbertSpace<double>						hilDouble;
	std::shared_ptr<Hamiltonian<double>>				hamDouble;
	std::shared_ptr<QuadraticHamiltonian<double>>		qhamDouble;

	// ^^^^^^^^ FOR COMPLEX ^^^^^^^^^
	Hilbert::HilbertSpace<cpx>							hilComplex;
	std::shared_ptr<Hamiltonian<cpx>>					hamComplex;
	std::shared_ptr<QuadraticHamiltonian<cpx>>			qhamComplex;

	// ^^^^^^^^^^^^ NQS ^^^^^^^^^^^^^
	std::shared_ptr<NQS<2, cpx>>						nqsCpx;
	std::shared_ptr<NQS<2, double>>						nqsDouble;

	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	void setDefaultMap()								final override;

private:
	// reset model
	void resetEd()										{ if (this->hamComplex) this->hamComplex.reset(); if (this->hamDouble) this->hamDouble.reset();		};
	void resetQuadratic()								{ if (this->qhamComplex) this->qhamComplex.reset(); if (this->qhamDouble) this->qhamDouble.reset(); };

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% I N N E R    M E T H O D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	// ############# S Y M M E T R I E S   E D #############

	template<typename _T>
	void symmetries(std::shared_ptr<Hamiltonian<_T>> _H, bool _diag = true, bool _states = false);

	void symmetriesDeg();
	void symmetriesCreateDeg();

	void symmetriesTest();
	std::pair<v_1d<GlobalSyms::GlobalSym>, v_1d<std::pair<Operators::SymGenerators, int>>> createSymmetries();

	// ####################### N Q S #######################

	template<typename _T, uint _spinModes>
	void nqsSingle(std::shared_ptr<NQS<_spinModes, _T>> _NQS);

	// ##################### QUADRATIC #####################

	template<typename _T>
	void quadraticStatesManifold(std::shared_ptr<QuadraticHamiltonian<_T>> _H);


	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% D E F I N I T I O N S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	bool defineLattice();
	bool defineModels(bool _createLat = true);
	bool defineModelsQ(bool _createLat = true);

	template<typename _T>
	bool defineModel(Hilbert::HilbertSpace<_T>& _Hil, std::shared_ptr<Hamiltonian<_T>>& _H);
	template<typename _T>
	bool defineModelQ(std::shared_ptr<QuadraticHamiltonian<_T>>& _H);
	template<typename _T, uint _spinModes = 2>
	void defineNQS(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<NQS<_spinModes, _T>>& _NQS);

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
public:
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% C O N S T R U C T O R S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	~UI()		= default;
	UI()		= default;
	UI(int argc, char** argv)				{ this->setDefaultMap(); this->init(argc, argv); };

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P A R S E R  F O R   H E L P %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	void exitWithHelp()						override final;

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% R E A L   P A R S E R %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	void funChoice()						final override;
	void parseModel(int argc, cmdArg& argv) final override;

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% H E L P E R S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	void setDefault()						final override;

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S I M U L A T I O N %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	// ############################################### N Q S 

	void makeSimNQS();

	// ############################################### S Y M M E T R I E S 

	void makeSimSymmetries(bool _diag = true, bool _states = false);

	void makeSimSymmetriesDeg();
	void makeSimSymmetriesCreateDeg();

	void makeSimSymmetriesSweep();
	void makeSimSymmetriesSweepHilbert();

	// ############################################### Q U A D R A T I C

	void makeSymQuadraticManifold();

};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# D E F A U L T ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Sets the default map for the UI
*/
inline void UI::setDefaultMap()
{
	this->defaultParams = {
		UI_OTHER_MAP(nqs	, this->nqsP.type_			, FHANDLE_PARAM_DEFAULT),			// type of the NQS state	
		UI_OTHER_MAP(m		, this->nqsP.nMcSteps_		, FHANDLE_PARAM_HIGHER0),			// mcsteps	
		UI_OTHER_MAP(nb		, this->nqsP.nBlocks_		, FHANDLE_PARAM_HIGHER0),			// number of blocks
		UI_OTHER_MAP(bs		, this->nqsP.blockSize_		, FHANDLE_PARAM_HIGHER0),			// block size
		UI_OTHER_MAP(nh		, this->nqsP.nHidden_		, FHANDLE_PARAM_HIGHER0),			// hidden params
		UI_OTHER_MAP(nf		, this->nqsP.nFlips_		, FHANDLE_PARAM_HIGHER0),			// flip number
		// for collecting in nqs
		UI_OTHER_MAP(bsS	, this->nqsP.blockSizeS_	, FHANDLE_PARAM_HIGHER0),			// block size samples
		UI_OTHER_MAP(mcS	, this->nqsP.nMcSamples_	, FHANDLE_PARAM_HIGHER0),			// mcsteps samples
		UI_OTHER_MAP(nbS	, this->nqsP.nSBlocks_		, FHANDLE_PARAM_HIGHER0),			// number of blocks - samples
		UI_OTHER_MAP(dirNQS	, this->nqsP.loadNQS_		, FHANDLE_PARAM_DEFAULT),			// directory to load the weights from

		// --------------- directory parameters ---------------
		{"f"				, std::make_tuple(""		, FHANDLE_PARAM_DEFAULT)},			// file to read from directory
		
		// ---------------- lattice parameters ----------------
		UI_OTHER_MAP(d		, this->latP._dim			, FHANDLE_PARAM_BETWEEN(1., 3.)),
		UI_OTHER_MAP(bc		, this->latP._bc			, FHANDLE_PARAM_BETWEEN(0., 3.)),
		UI_OTHER_MAP(l		, this->latP._typ			, FHANDLE_PARAM_BETWEEN(0., 1.)),
		UI_OTHER_MAP(lx		, this->latP._Lx			, FHANDLE_PARAM_HIGHER0),
		UI_OTHER_MAP(ly		, this->latP._Ly			, FHANDLE_PARAM_HIGHER0),
		UI_OTHER_MAP(lz		, this->latP._Lz			, FHANDLE_PARAM_HIGHER0),
		
		// ----------------- model parameters -----------------			
		UI_OTHER_MAP(mod		, this->modP._modTyp, FHANDLE_PARAM_BETWEEN(0., 2.)),
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

		// -------------------- symmetries -------------------
		UI_PARAM_MAP(ks			, this->symP._k			, FHANDLE_PARAM_HIGHER0),
		UI_PARAM_MAP(pxs		, this->symP._px		, FHANDLE_PARAM_BETWEEN()),
		UI_PARAM_MAP(pys		, this->symP._py		, FHANDLE_PARAM_BETWEEN()),
		UI_PARAM_MAP(pzs		, this->symP._pz		, FHANDLE_PARAM_BETWEEN()),
		UI_PARAM_MAP(xs			, this->symP._x			, FHANDLE_PARAM_BETWEEN()),
		UI_PARAM_MAP(u1s		, this->symP._U1		, FHANDLE_PARAM_DEFAULT),
		UI_PARAM_MAP(SYM		, this->symP._S			, FHANDLE_PARAM_BETWEEN(0., 1.)),	// even use symmetries?
		
		// ---------------- other ----------------
		UI_OTHER_MAP(fun		, -1.					, FHANDLE_PARAM_HIGHERV(-2.0)),		// choice of the function to be calculated
		UI_OTHER_MAP(th			, 1.0					, FHANDLE_PARAM_HIGHER0),			// number of threads
		UI_OTHER_MAP(q			, 0.0					, FHANDLE_PARAM_DEFAULT),			// quiet?
		UI_OTHER_MAP(dir		, "DEFALUT"				, FHANDLE_PARAM_DEFAULT),
	};
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
	bool _isGood				= true;
	// get the symmetries
	auto [_glbSyms, _locSyms]	= this->createSymmetries();
	_Hil						= Hilbert::HilbertSpace<_T>(this->latP.lat, _locSyms, _glbSyms);
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
	case MY_MODELS::HEI_KIT_M:
		_H = std::make_shared<HeisenbergKitaev<_T>>(std::move(_Hil), 
			this->modP.Kx_, this->modP.Ky_, this->modP.Kz_,
			this->modP.heiJ_, this->modP.heiDlt_, this->modP.heiHz_, this->modP.heiHx_);
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
* @brief Based on a given type, it creates a NQS. Uses the model provided by the user to get the Hamiltonian.
* @param _H Specific Hamiltonian
* @param _NQS Neural Network Quantum State frameweork
*/
template<typename _T, uint _spinModes>
inline void UI::defineNQS(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<NQS<_spinModes, _T>>& _NQS)
{
	switch (this->nqsP.type_)
	{
	case NQSTYPES::RBM_T:
		_NQS = std::make_shared<RBM_S<_spinModes, _T>>(	_H,
														this->nqsP.nHidden_,
														this->nqsP.lr_,
														this->threadNum);
		break;
	case NQSTYPES::RBMPP_T:
		_NQS = std::make_shared<RBM_PP_S<_spinModes, _T>>(_H,
														this->nqsP.nHidden_,
														this->nqsP.lr_,
														this->threadNum);
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
inline void UI::symmetries(std::shared_ptr<Hamiltonian<_T>> _H, bool _diag, bool _states)
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

/*
* @brief Allows to calculate how do the degeneracies look in case of symmetric Hamiltonians...
* At the same time, it calculates mixing of the states in the degenerate manifold
*/
inline void UI::symmetriesDeg()
{
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	_timer.reset();
	u64 Nh					= this->hamComplex->getHilbertSize();
	const auto _realizations= this->nqsP.nBlocks_;

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

/*
* @brief Use this function to create a combination of states in the middle of the spectrum and 
* calculate their entanglement entropies 
*/
inline void UI::symmetriesCreateDeg()
{
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	_timer.reset();
	u64 Nh					=			this->hamComplex->getHilbertSize();
	const auto _realizations=			this->nqsP.nBlocks_;

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

/*
* @brief Tests the currently implemented symmetries based on specific model parameters
* @param start the timer start
*/
inline void UI::symmetriesTest()
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

template<typename _T, uint _spinModes>
inline void UI::nqsSingle(std::shared_ptr<NQS<_spinModes, _T>> _NQS)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	LOGINFO("Started building the NQS Hamiltonian", LOG_TYPES::TRACE, 1);
	LOGINFO("Using: " + SSTR(getSTR_NQSTYPES(this->nqsP.type_)), LOG_TYPES::TRACE, 2);
	// get info
	std::string nqsInfo		= _NQS->getInfo();
	std::string modelInfo	= _NQS->getHamiltonianInfo();
	std::string dir			=	makeDirsC(this->mainDir, this->latP.lat->get_info(), modelInfo, nqsInfo);

	// calculate ED to compare with Lanczos or Full
	u64 Nh						= _NQS->getHilbertSize();
	arma::Col<_T> _mbs;
	if (Nh <= UI_LIMITS_NQS_ED)
	{
		auto _H = _NQS->getHamiltonian();
		_H->hamiltonian();
		if (Nh <= UI_LIMITS_NQS_FULLED)
		{
			_H->diagH(false);
			_mbs = _H->getEigVec(0);
			if(latP.lat->get_Ns() < 6)
				_H->prettyPrint(stout, _mbs, latP.lat->get_Ns(), 1e-2);
			LOGINFO("Found the ED groundstate to be EED_0 = " + STRP(_NQS->getHamiltonianEigVal(0), 7), LOG_TYPES::TRACE, 2);
		}
		else
		{
			_H->diagH(false, 50, 0, 1000, 0, "lanczos");
			LOGINFO("Found the ED groundstate to be EED_0 = " + STRP(_NQS->getHamiltonianEigVal(0), 7), LOG_TYPES::TRACE, 2);
		}
	}
	if (!this->nqsP.loadNQS_.empty())
		_NQS->setWeights(this->nqsP.loadNQS_, "weights.h5");
	
	// set the operators to save
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>> _opsG = {};
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>> _opsL = {};
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>> _opsC = {};
	// go through the lattice sites
	{
		auto _SzL = std::make_shared<Operators::OperatorNQS<_T, uint>>(Operators::sigmaZ_L<_T>(latP.lat), "sz_l");
		auto _SzC = std::make_shared<Operators::OperatorNQS<_T, uint, uint>>(Operators::sigmaZ_C<_T>(latP.lat), "sz_c");
		_opsL.push_back(_SzL);
		_opsC.push_back(_SzC);
	}
	// create measurement operator
	NQSAv::MeasurementNQS<_T> _meas(this->latP.lat, dir,  _opsG, 
																			_opsL, 
																			_opsC, this->threadNum);

	// start the simulation
	arma::Col<_T> _EN(this->nqsP.nMcSteps_ + this->nqsP.nMcSamples_, arma::fill::zeros);
	_EN.subvec(0, this->nqsP.nMcSteps_ - 1) = _NQS->train(this->nqsP.nMcSteps_,
																			this->nqsP.nTherm_,
																			this->nqsP.nBlocks_,
																			this->nqsP.blockSize_,
																			dir,												
																			this->nqsP.nFlips_,
																			this->quiet,
																			_timer.start(),
																			10);
	_EN.subvec(this->nqsP.nMcSteps_, _EN.size() - 1) = _NQS->collect(this->nqsP.nMcSamples_,
																	0,
																	this->nqsP.nSBlocks_,
																	this->nqsP.blockSizeS_,
																	this->nqsP.nFlips_,
																	this->quiet,
																	_timer.start(),
																	_meas);
	arma::Mat<double> _ENSM(_EN.size(), 2, arma::fill::zeros);
	_ENSM.col(0)	= arma::real(_EN);
	_ENSM.col(1)	= arma::imag(_EN);

	// save energy
	auto perc		= int(this->nqsP.nMcSamples_ / 20);
	perc			= perc == 0 ? 1 : perc;
	auto ENQS_0		= arma::mean(_ENSM.col(0).tail(perc));
	LOGINFOG("Found the NQS groundstate to be ENQS_0 = " + STRP(ENQS_0, 7), LOG_TYPES::TRACE, 2);
	_ENSM.save(dir + "history.dat", arma::raw_ascii);

	// many body
	if (_mbs.size() != 0)
		_meas.measure(_mbs, _NQS->getHilbertSpace());
	// save the measured quantities
	_meas.save();
}

// ##########################################################################################################################################

// ##########################################################################################################################################
// ##########################################################################################################################################
// ########################################################### Q U A D R A T I C ############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

// ##########################################################################################################################################

template<typename _T>
inline void UI::quadraticStatesManifold(std::shared_ptr<QuadraticHamiltonian<_T>> _H)
{
	uint Ns					= _H->getNs();
	uint Nh					= _H->getHilbertSize();
	uint _type				= _H->getTypeI();

	// --- create the directories ---
	bool _manifold			= (_type == (uint)MY_MODELS_Q::FREE_FERMIONS_M) && this->modP.q_manifold_;
	std::string str0		= "QuadraticEntropies" + std::string(this->modP.q_shuffle_ ? "S" : "") + std::string((_manifold) ? "Manifold" : "");
	std::string dir			= makeDirsC(this->mainDir, _H->getType(), this->latP.lat->get_info(), str0);

	// ------ use those files -------
	std::string modelInfo	= _H->getInfo();
	// how many states to take for calculating the entropies
	u64 _gamma				= this->modP.q_gamma_;
	// how many states to take for the average (realizations of the combinations N * \Gamma states)
	u64 _realizations		= this->modP.q_realizationNum_;
	// number of combinations to take from single particle states (is \Gamma)
	u64 _combinations		= this->modP.q_randomCombNum_;

	// --- save energies txt check ---
	std::string filename	= filenameQuadraticRandom(dir + modelInfo + VEQV(_R, _realizations) + VEQV(_C, _combinations) +
							  VEQV(_Gamma, _gamma), _type, _H->ran_);

	IF_EXCEPTION(_combinations < _gamma, "Bad number of combinations. Must be bigger than the number of states");

	// check the model (if necessery to build hamilonian, do it)
	if (_type != (uint)MY_MODELS_Q::FREE_FERMIONS_M)
	{
		_H->buildHamiltonian();
		_H->diagH(false);
		LOGINFO(_timer.start(), "Diagonalization", 3);
	}

	// go through the information
	LOGINFO("Spectrum size:						  " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking states to mix (Gamma):		  " + STR(_gamma), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states (combinations):	  " + STR(_combinations), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num realizations (averages):  " + STR(_realizations), LOG_TYPES::TRACE, 3);

	// save single particle energies
	if (!fs::exists(filename + ".h5"))
		_H->getEigVal(dir, HAM_SAVE_EXT::h5, false);

	// iterate through bond cut
	uint _bonds				= uint(Ns / 2);
	arma::mat ENERGIES(_realizations, _gamma, arma::fill::zeros);
	arma::vec ENTROPIES_SP(_realizations, arma::fill::zeros);
	arma::vec ENTROPIES_MB(_realizations, arma::fill::zeros);

	// many body orbitals (constructed of vectors of indices of single particle states)
	v_1d<double> energies;
	v_2d<uint> orbs;

	// single particle orbital indices (all posibilities to choose from in the combination)
	v_1d<uint> _SPOrbitals  = Vectors::vecAtoB<uint>(Ns);
	
	// get the states to use later
	_timer.checkpoint("combinations");
	
	// calculate many body orbitals to be used
	if (Ns <= UI_LIMITS_QUADRATIC_COMBINATIONS)
		_H->getManyBodyOrbitals(Ns / 2, _SPOrbitals, orbs);
	else
		_H->getManyBodyOrbitals(Ns / 2, _SPOrbitals, orbs, _combinations, this->threadNum);
	if(this->modP.q_shuffle_)
		std::shuffle(orbs.begin(), orbs.end(), this->ran_.eng());
	// obtain the energies
	_H->getManyBodyEnergies(energies, orbs, this->threadNum);

		// obtain the single particle energies
	arma::Mat<_T> W			= _H->getTransMat();
	// make matrices cut to a specific number of bonds
	arma::Mat<_T> Ws		= W.submat(0, 0, W.n_rows - 1, _bonds - 1);
	// conjugate transpose it - to be used later
	arma::Mat<_T> WsC		= Ws.t();
	// Hilbert space
	auto _hilbert			= Hilbert::HilbertSpace<_T>(this->latP.lat);
	auto _Nh			    = _hilbert.getHilbertSize();
	LOGINFO(_timer.point("combinations"), "Combinations time:", 3);

	// -------------------------------- CORRELATION --------------------------------
	auto _appendEntroSP = [&](u64 _idx, const std::vector<std::vector<uint>>& _orbitals, arma::Col<_T>& _coeff)
		{
			// iterate through the state
			auto J				= SingleParticle::CorrelationMatrix::corrMatrix(Ns, Ws, WsC, _orbitals, _coeff, this->ran_);
			ENTROPIES_SP(_idx)	= Entropy::Entanglement::Bipartite::SingleParticle::vonNeuman<cpx>(J);
		};

	// --------------------------------- MANY BODY ---------------------------------
	auto _appendEntroMB = [&](u64 _idx, const std::vector<std::vector<uint>>& _orbitals, arma::Col<_T>& _coeff)
		{
			// use the slater matrix to obtain the many body state
			arma::Mat<_T> _slater(Ns / 2, Ns / 2, arma::fill::zeros);
			arma::Col<cpx> _state(_Nh, arma::fill::zeros);
			for (int i = 0; i < _orbitals.size(); ++i)
				_state += _coeff(i) * _H->getManyBodyState(_orbitals[i], _hilbert, _slater);
			ENTROPIES_MB(_idx) = Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_state, _bonds, _hilbert);;
		};
	// ----------------------------------- SAVER -----------------------------------
	auto _saveEntro = [&](bool _save)
		{
			if (_save)
			{
#ifndef _DEBUG
#	pragma omp critical
#endif
				{
					LOGINFO("Checkpoint", LOG_TYPES::TRACE, 4);
					ENTROPIES_SP.save(arma::hdf5_name(filename + "_SP.h5", "entropy"));
					ENERGIES.save(arma::hdf5_name(filename + "_EN.h5", "energy"));
					if (this->modP.q_manybody_)
						ENTROPIES_MB.save(arma::hdf5_name(filename + "_MB.h5", "entropy"));
				}
			}
		};
	// ------------------------------------ MAIN -----------------------------------
	_timer.checkpoint("entropy");
	pBar pbar(5, _realizations, _timer.point(0));

	// ------------ MANIFOLDS ------------
	// check if one wants to create a combinations at degenerate manifolds
	if (_manifold && _gamma != 1)
	{
		// zip the energies and orbitals together
		auto _zippedEnergies = Containers::zip(energies, orbs);

		// get map with frequencies of specific energies
		auto _frequencies = Vectors::freq<10>(energies, _gamma - 1);
		if (_frequencies.size() == 0)
			throw std::runtime_error("Bad number of frequencies. Must be bigger than $\\Gamma$.");

		// remove the _zippedEnergies based on existing in this map to get the manifolds
		std::erase_if(_zippedEnergies, [&](const auto& elem)
			{
				auto const& [_en, _vec] = elem;
				return !_frequencies.contains(Math::trunc<double, 10>(_en));
			});

		// sort the zipped energies please
		Containers::sort<0>(_zippedEnergies, [](const auto& a, const auto& b) { return a < b; });

		// go through the realizations
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum)
#endif
		for (u64 idx = 0; idx < _realizations; idx++)
		{
			// generate coefficients (create random state consisting of stateNum = \Gamma states)
			auto coeff			= this->ran_.createRanState<_T>(_gamma);

			// get the random state
			long long idxState	= this->ran_.randomInt(0, _zippedEnergies.size() - _gamma);

			// while we cannot take _gamma states out of it, lower the index
			long long _iter		= 1;
			const auto E_idx	= Math::trunc<double, 10>(std::get<0>(_zippedEnergies[idxState]));

			while (((idxState - _iter) > 0) && _iter <= _gamma + 1)
			{
				const auto E_in	= Math::trunc<double, 10>(std::get<0>(_zippedEnergies[idxState - _iter]));
				if (E_in != E_idx)
					break;
				_iter++;
			}
			idxState -= (_iter - 1);

			// take the chosen orbitals
			std::vector<std::vector<uint>> orbitals;
			for (uint i = idxState; i < idxState + _gamma; ++i)
			{
				orbitals.push_back(std::get<1>(_zippedEnergies[i]));
				ENERGIES(idx, i - idxState) = std::get<0>(_zippedEnergies[i]);
			}

			// SP
			_appendEntroSP(idx, orbitals, coeff);
			// MB
			if(this->modP.q_manybody_)
				_appendEntroMB(idx, orbitals, coeff);
			// update progress (checkpoint for saving the entropies)
			PROGRESS_UPD_DO(idx, pbar, "PROGRESS", _saveEntro(idx % pbar.percentageSteps == 0));
		}
	}
	else
	{
		// go through the realizations
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum)
#endif
		for (u64 idx = 0; idx < _realizations; idx++)
		{
			// generate coefficients (create random state consisting of stateNum = \Gamma states)
			auto coeff		= this->ran_.createRanState<_T>(_gamma);
			// get the random state
			auto idxState	= this->ran_.randomInt<uint>(0, energies.size() - _gamma);

			// take the chosen orbitals
			std::vector<std::vector<uint>> orbitals;
			for (uint i = idxState; i < idxState + _gamma; ++i)
			{
				orbitals.push_back(orbs[i]);
				ENERGIES(idx, i - idxState) = energies[i];
			}
			// SP
			_appendEntroSP(idx, orbitals, coeff);
			// MB
			if (this->modP.q_manybody_)
				_appendEntroMB(idx, orbitals, coeff);

			// update progress (checkpoint for saving the entropies)
			PROGRESS_UPD_DO(idx, pbar, "PROGRESS", _saveEntro(idx % pbar.percentageSteps == 0));
		}
	}
	
	// save in the end ;)
	_saveEntro(true);

	// save entropies
	LOGINFO("Finished entropies! " + VEQ(_gamma) + ", " + VEQ(_realizations), LOG_TYPES::TRACE, 2);
	LOGINFO(_timer.point("entropy"), "Entropies time:", 3);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%