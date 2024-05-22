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
#if 1													 // #
#define NQS_SAVE_WEIGHTS								 // #
#ifdef NQS_SAVE_WEIGHTS									 // #
#	define NQS_SAVE_DIR "WEIGHTS" + kPS					 // #
#endif													 // #
														 // #
//#define NQS_USE_GPU									// #
#define NQS_USE_CPU										 // #
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
#endif 													 // #
// ##########################################################


// ######################### MODELS #########################
#if 1
#ifndef ISING_H											 // #
#	include "../models/ising.h"							 // #
#endif // !ISING_H										 // #
#ifndef XYZ_H											 // #
#	include "../models/XYZ.h"							 // #
#endif // !XYZ_H										 // #
#ifndef HEISENBERG_KITAEV_H								 // #
#	include "../models/heisenberg-kitaev.h"				 // #
#endif													 // #
// random Hamiltonians									 // #
#ifndef QSM_H											 // #
#	include "../models/quantum_sun.h"					 // #
#endif													 // #
#ifndef ROSENZWEIG_PORTER_H								 // #
#	include "../models/rosenzweig-porter.h"				 // #
#endif													 // #
#ifndef ULTRAMETRIC_H									 // #
#	include "../models/ultrametric.h"					 // #
#endif													 // #
#endif
// ##########################################################

// ######################## MODELS Q ########################
#if 1
#ifndef POWER_LAW_RANDOM_BANDED_H
#include "../models/quadratic/PowerLawRandomBanded.h"	 // #
#endif
#ifndef SYK2_M_H										 // #
#include "../models/quadratic/SYK2.h"					 // #
#endif // !SYK2											 // #
#ifndef FF_M_H											 // #
#include "../models/quadratic/FreeFermions.h"			 // #
#endif // !SYK2											 // #
#ifndef AUBRY_ANDRE_M_H									 // #
#include "../models/quadratic/AubryAndre.h"				 // #
#endif // !SYK2											 // #
#endif
// ##########################################################

// ###################### LATTICES ##########################
#if 1													 // #
#ifndef SQUARE_H										 // #
#include "../../source/src/Lattices/square.h"			 // #
#endif													 // #
#ifndef HEXAGONAL_H										 // #
#include "../../source/src/Lattices/hexagonal.h"		 // #
#endif													 // #
#endif													 // #
// ##########################################################

// ##################### STATISTICAL ########################
#if 1													 // #
#ifndef STATISTICS_H									 // #
#include "../quantities/statistics.h"					 // #
#endif													 // #
#include "../algebra/quantities/measure.h"				 // #
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

// ############################################################## P A R A M S ###############################################################

// ##########################################################################################################################################

namespace UI_PARAMS 
{

	// ----------------------------------------------------------------

	/*
	* @brief Defines parameters used later for the models
	*/
	struct ModP 
	{
		// ################################## TYPE ##################################
		
		UI_PARAM_CREATE_DEFAULT(modTyp, MY_MODELS, MY_MODELS::ISING_M);
		UI_PARAM_CREATE_DEFAULT(modRanN, uint, 1);			// number of random states
		UI_PARAM_CREATE_DEFAULT(modRanSeed, u64, 0);		// seed for the random number generator
		UI_PARAM_CREATE_DEFAULTD(modMidStates, double, 1.0);// states in the middle of the spectrum
		UI_PARAM_CREATE_DEFAULTD(modEnDiff, double, 1.0);	// tolerance for the energy difference of the states in offdiagonal

		// ##########################################################################
		
		// ########################## I N T E R A C T I N G #########################
		
		// ##########################################################################

		// ############## ISING ################
		
		UI_PARAM_STEP(double, J1, 1.0);			// spin exchange
		UI_PARAM_STEP(double, hz, 1.0);			// perpendicular field
		UI_PARAM_STEP(double, hx, 1.0);			// transverse field

		// ############### XYZ #################		
		
		UI_PARAM_STEP(double, J2, 2.0);			// next nearest neighbors exchange
		UI_PARAM_STEP(double, eta1, 0.5);
		UI_PARAM_STEP(double, eta2, 0.5);
		UI_PARAM_STEP(double, dlt1, 0.3);
		UI_PARAM_STEP(double, dlt2, 0.3);

		// ############# KITAEV ################

		v_1d<double> Kx_;
		v_1d<double> Ky_;
		v_1d<double> Kz_;
		void resizeKitaev(const size_t Ns)
		{
			this->Kx_.resize(Ns);
			this->Ky_.resize(Ns);
			this->Kz_.resize(Ns);
		};

		// ########### HEISENBERG ##############

		v_1d<double> heiJ_;
		v_1d<double> heiDlt_;
		v_1d<double> heiHx_;
		v_1d<double> heiHz_;

		void resizeHeisenberg(const size_t Ns)
		{
			this->heiJ_.resize(Ns);
			this->heiDlt_.resize(Ns);
			this->heiHx_.resize(Ns);
			this->heiHz_.resize(Ns);
		};

		// ############### QSM #################

		struct qsm_t
		{
			UI_PARAM_CREATE_DEFAULTD(qsm_N, size_t, 1);
			UI_PARAM_CREATE_DEFAULTD(qsm_Ntot, size_t, 1);
			UI_PARAM_CREATE_DEFAULTD(qsm_gamma, double, 1.0);
			UI_PARAM_CREATE_DEFAULTD(qsm_g0, double, 1.0);

			UI_PARAM_CREATE_DEFAULTV(qsm_alpha, double);
			UI_PARAM_CREATE_DEFAULTV(qsm_xi, double);
			UI_PARAM_CREATE_DEFAULTV(qsm_h, double);
			void resizeQSM()
			{
				auto _N = this->qsm_Ntot_ - this->qsm_N_;
				if (_N < 0)
					return;
				this->qsm_alpha_r_ = 0;
				this->qsm_alpha_.resize(this->qsm_Ntot_ - this->qsm_N_);
				this->qsm_xi_r_ = 0;
				this->qsm_xi_.resize(this->qsm_Ntot_ - this->qsm_N_);
				this->qsm_h_r_ = 0;
				this->qsm_h_.resize(_N);
			};
		} qsm;

		// ######### ROSENZWEIG PORTER #########

		struct rosenzweig_porter_t
		{
			UI_PARAM_CREATE_DEFAULTV(rp_g, double);
			UI_PARAM_CREATE_DEFAULTD(rp_single_particle, bool, 0);
			UI_PARAM_CREATE_DEFAULTD(rp_be_real, bool, 1);
			UI_PARAM_CREATE_DEFAULTD(rp_g_sweep_n, int, 1);

			void resizeRP()
			{
				this->rp_g_.resize(this->rp_g_sweep_n_);
			};

		} rosenzweig_porter;

		// ############ ULTRAMETRIC ############

		struct ultrametric_t
		{
			UI_PARAM_CREATE_DEFAULTD(um_N, size_t, 1);
			UI_PARAM_CREATE_DEFAULTD(um_Ntot, size_t, 1);
			UI_PARAM_CREATE_DEFAULTV(um_alpha, double);
			UI_PARAM_CREATE_DEFAULTD(um_g, double, 1.0);

			void resizeUM()
			{
				auto _N = this->um_Ntot_ - this->um_N_;
				if (_N < 0)
					return;
				this->um_alpha_r_ = 0;
				this->um_alpha_.resize(_N);
			};
		} ultrametric;

		// #####################################
		// ######### Q U A D R A T I C #########
		// #####################################
		
		// for simulation
		UI_PARAM_CREATE_DEFAULT(q_gamma, uint, 1);					// mixing quadratic states
		UI_PARAM_CREATE_DEFAULT(q_manifold, bool, false);			// use the degenerate manifold?
		UI_PARAM_CREATE_DEFAULT(q_manybody, bool, true);			// use the many body calculation?
		UI_PARAM_CREATE_DEFAULT(q_randomCombNum, uint, 100);		// number of random combinations for the average (to choose from)
		UI_PARAM_CREATE_DEFAULT(q_realizationNum, uint, 100);		// number of realizations for the average
		UI_PARAM_CREATE_DEFAULT(q_shuffle, bool, true);				// shuffle the states?

		// ########### AUBRY_ANDRE ############
		
		struct aubry_andre_t
		{
			UI_PARAM_STEP(double, aa_J, 1.0);						// hopping
			UI_PARAM_STEP(double, aa_lambda, 0.5);					// modulation strength
			UI_PARAM_STEP(double, aa_beta, (1 + std::sqrt(5)) / 2);	// phase multiplication
			UI_PARAM_STEP(double, aa_phi, 1.0);						// phase addition
		} aubry_andre;

		// ############ POWER LAW ##############

		struct power_law_random_bandwidth_t
		{
			UI_PARAM_CREATE_DEFAULTD(plrb_a, double, 1.0);
			UI_PARAM_CREATE_DEFAULTD(plrb_b, double, 1.0);
		} power_law_random_bandwidth;

		// -------------------------------------

		// #####################################

		// -------------------------------------

		void setDefault() 
		{
			UI_PARAM_SET_DEFAULT(modTyp);

			// SPIN
			{
				// ising
				{
					UI_PARAM_SET_DEFAULT_STEP(J1);
					UI_PARAM_SET_DEFAULT_STEP(hz);
					UI_PARAM_SET_DEFAULT_STEP(hx);
				}
				// xyz
				{
					UI_PARAM_SET_DEFAULT_STEP(J2);
					UI_PARAM_SET_DEFAULT_STEP(eta1);
					UI_PARAM_SET_DEFAULT_STEP(eta2);
					UI_PARAM_SET_DEFAULT_STEP(dlt1);
					UI_PARAM_SET_DEFAULT_STEP(dlt2);
				}
				// kitaev
				{
					this->Kx_		= v_1d<double>(1, 1.0);
					this->Ky_		= v_1d<double>(1, 1.0);
					this->Kz_		= v_1d<double>(1, 1.0);
					this->heiJ_		= v_1d<double>(1, 1.0);
					this->heiDlt_	= v_1d<double>(1, 1.0);
					this->heiHz_	= v_1d<double>(1, 1.0);
					this->heiHx_	= v_1d<double>(1, 1.0);
				}
				// QSM
				{
					UI_PARAM_SET_DEFAULT_STRUCT(qsm, qsm_gamma);
					UI_PARAM_SET_DEFAULT_STRUCT(qsm, qsm_g0);
					UI_PARAM_SET_DEFAULT_STRUCT(qsm, qsm_Ntot);
					UI_PARAM_SET_DEFAULT_STRUCT(qsm, qsm_N);
					this->qsm.qsm_alpha_	= v_1d<double>(1, 1.0);
					this->qsm.qsm_xi_		= v_1d<double>(1, 1.0);
					this->qsm.qsm_h_		= v_1d<double>(1, 1.0);
				}
				// Rosenzweig-Porter
				{
					this->rosenzweig_porter.rp_g_ = v_1d<double>(1, 1.0);
				}
			}

			// -------------------------------------
			
			// QUADRATIC
			{
				// aubry-andre
				{
					UI_PARAM_SET_DEFAULT_STRUCT(aubry_andre, aa_J);
					UI_PARAM_SET_DEFAULT_STRUCT(aubry_andre, aa_lambda);
					UI_PARAM_SET_DEFAULT_STRUCT(aubry_andre, aa_beta);
					UI_PARAM_SET_DEFAULT_STRUCT(aubry_andre, aa_phi);
				}
			}
		}

		// -------------------------------------

		/*
		* @brief Check whether the model itself is complex...
		*/
		bool checkComplex() const
		{
			if (this->modTyp_ == MY_MODELS::FREE_FERMIONS_M)
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
		// for the Hamiltonians that only use the total number of particles, not lattices!
		UI_PARAM_CREATE_DEFAULT(Ntot, uint, 1);

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

// ############################################################# U I N T E R F ##############################################################

// ##########################################################################################################################################

/*
* @brief User interface class for the QES
*/
class UI : public UserInterface 
{

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
	// standard elements
	template<typename _T>
	void get_inf_dir_ext_r(std::shared_ptr<Hamiltonian<_T>> _H, std::string& _dir, std::string& modelInfo, std::string& randomStr, std::string& ext);
	template<typename _T>
	void get_inf_dir_ext(std::shared_ptr<Hamiltonian<_T>> _H, std::string& _dir, std::string& modelInfo, std::string& ext);

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

	template<typename _T>
	void quadraticSpectralFunction(std::shared_ptr<QuadraticHamiltonian<_T>> _H);

	// ##################### SPIN MODELS ###################

	std::pair<v_1d<std::shared_ptr<Operators::Operator<double>>>, strVec>
		ui_eth_getoperators(bool _isquadratic = true, bool _ismanybody = true);
	template<typename _T>
	void ui_eth_randomize(std::shared_ptr<Hamiltonian<_T>> _H);
	template<typename _T>
	void checkETH(std::shared_ptr<Hamiltonian<_T>> _H);
	template<typename _T>
	void checkETH_statistics(std::shared_ptr<Hamiltonian<_T>> _H);
	template<typename _T>
	void checkETH_level_prop(std::shared_ptr<Hamiltonian<_T>> _H);
	template<typename _T>
	void checkETH_scaling_offdiag(std::shared_ptr<Hamiltonian<_T>> _H);
	template<typename _T>
	void checkETH_time_evo(std::shared_ptr<Hamiltonian<_T>> _H);

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% D E F I N I T I O N S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	bool defineLattice();
	bool defineLattice(std::shared_ptr<Lattice> _lat, LatticeTypes _typ = LatticeTypes::SQ);

	// models
	bool defineModels(bool _createLat = true, bool _checkSyms = true, bool _useHilbert = true);
	bool defineModelsQ(bool _createLat = true);

	template<typename _T>
	bool defineHilbert(Hilbert::HilbertSpace<_T>& _Hil);

	// many-body
	template<typename _T>
	bool defineModel(Hilbert::HilbertSpace<_T>& _Hil, std::shared_ptr<Hamiltonian<_T>>& _H);
	template<typename _T>
	bool defineModel(std::shared_ptr<Hamiltonian<_T>>& _H, uint _Ns);
	template<typename _T>

	// quadratic
	bool defineModelQ(std::shared_ptr<QuadraticHamiltonian<_T>>& _H, std::shared_ptr<Lattice> _lat);
	template<typename _T>
	bool defineModelQ(std::shared_ptr<QuadraticHamiltonian<_T>>& _H, uint _Ns);

	// NQS
	template<typename _T, uint _spinModes = 2>
	void defineNQS(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<NQS<_spinModes, _T>>& _NQS);

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
public:
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% C O N S T R U C T O R S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	~UI()		= default;
	UI()		= default;
	UI(int argc, char** argv);

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
	void makeSimQuadraticSpectralFunction();

	// ############################################### E T H

	void makeSimETH();
	void makeSimETHSweep();
	
};

// ##########################################################################################################################################

inline UI::UI(int argc, char** argv)
{
	this->setDefaultMap(); 
	this->init(argc, argv);
}

// ##########################################################################################################################################

// ############################################################# D E F A U L T ##############################################################

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
		UI_OTHER_MAP(mod		, this->modP._modTyp, FHANDLE_PARAM_BETWEEN(0., 1000.)),
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

// ############################################################# D E F I N E S ##############################################################

// ##########################################################################################################################################

template<typename _T>
inline bool UI::defineHilbert(Hilbert::HilbertSpace<_T>& _Hil)
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
	return _isGood;
}

// ##########################################################################################################################################

/*
* @brief Defines the interacting model based on the input file...
*/
template<typename _T>
inline bool UI::defineModel(Hilbert::HilbertSpace<_T>& _Hil, std::shared_ptr<Hamiltonian<_T>>& _H)
{
	if (!defineHilbert<_T>(_Hil))
		return false;

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
	case MY_MODELS::QSM_M:
		_H = std::make_shared<QSM<_T>>(std::move(_Hil), 
			this->modP.qsm.qsm_N_, this->modP.qsm.qsm_gamma_, this->modP.qsm.qsm_g0_,
			this->modP.qsm.qsm_alpha_, this->modP.qsm.qsm_h_, this->modP.qsm.qsm_xi_);
		break;
	case MY_MODELS::RP_M:
		_H = std::make_shared<RosenzweigPorter<_T>>(std::move(_Hil), this->modP.rosenzweig_porter.rp_g_[0]);
		break;
	case MY_MODELS::ULTRAMETRIC_M:
		_H = std::make_shared<Ultrametric<_T>>(std::move(_Hil), this->modP.ultrametric.um_N_, this->modP.ultrametric.um_g_, this->modP.ultrametric.um_alpha_);
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
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	return true;
}

template<typename _T>
inline bool UI::defineModel(std::shared_ptr<Hamiltonian<_T>>& _H, uint _Ns)
{
	Hilbert::HilbertSpace<_T> _Hil(_Ns);
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
		// --------------------------- RANDOM MODELS ---------------------------
	case MY_MODELS::QSM_M:
		_H = std::make_shared<QSM<_T>>(std::move(_Hil), 
			this->modP.qsm.qsm_N_, this->modP.qsm.qsm_gamma_, this->modP.qsm.qsm_g0_,
			this->modP.qsm.qsm_alpha_, this->modP.qsm.qsm_h_, this->modP.qsm.qsm_xi_);
		break;
	case MY_MODELS::RP_M:
		_H = std::make_shared<RosenzweigPorter<_T>>(std::move(_Hil), this->modP.rosenzweig_porter.rp_g_[0]);
		break;
	case MY_MODELS::ULTRAMETRIC_M:
		_H = std::make_shared<Ultrametric<_T>>(std::move(_Hil), this->modP.ultrametric.um_N_, this->modP.ultrametric.um_g_, this->modP.ultrametric.um_alpha_);
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
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	return true;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Defines the quadratic model based on the input file...
*/
template<typename _T>
inline bool UI::defineModelQ(std::shared_ptr<QuadraticHamiltonian<_T>>& _H, std::shared_ptr<Lattice> _lat)
{
	bool _isGood = true;

	// switch the type of quadratic model
	switch (this->modP.modTyp_)
	{
	case MY_MODELS::FREE_FERMIONS_M:
		_H = std::make_shared<FreeFermions<_T>>(this->latP.lat, this->modP.J1_, this->modP.J10_, 0.0);
		break;
	case MY_MODELS::AUBRY_ANDRE_M:
		_H = std::make_shared<AubryAndre<_T>>(this->latP.lat, this->modP.aubry_andre.aa_J_, this->modP.aubry_andre.aa_lambda_,
											  this->modP.aubry_andre.aa_beta_, this->modP.aubry_andre.aa_phi_,
											  this->modP.aubry_andre.aa_J0_, this->modP.aubry_andre.aa_lambda0_,
											  this->modP.aubry_andre.aa_beta0_, this->modP.aubry_andre.aa_phi0_, 0.0);
		break;
	case MY_MODELS::SYK2_M:
		_H = std::make_shared<SYK2<_T>>(this->latP.lat, 0.0);
		break;
	case MY_MODELS::POWER_LAW_RANDOM_BANDED_M:
		_H = std::make_shared<PowerLawRandomBanded<_T>>(this->latP.lat, 0.0);
		break;
	default:
		_H = std::make_shared<FreeFermions<_T>>(this->latP.lat, this->modP.J1_, this->modP.J10_, 0.0);
		break;
	}
	return _isGood;
}

template<typename _T>
inline bool UI::defineModelQ(std::shared_ptr<QuadraticHamiltonian<_T>>& _H, uint _Ns)
{
	bool _isGood = true;

	// switch the type of quadratic model
	switch (this->modP.modTyp_)
	{
	case MY_MODELS::FREE_FERMIONS_M:
		_H = std::make_shared<FreeFermions<_T>>(_Ns, this->modP.J1_, this->modP.J10_, 0.0);
		break;
	case MY_MODELS::AUBRY_ANDRE_M:
		_H = std::make_shared<AubryAndre<_T>>(_Ns, this->modP.aubry_andre.aa_J_, this->modP.aubry_andre.aa_lambda_,
													this->modP.aubry_andre.aa_beta_, this->modP.aubry_andre.aa_phi_,
													this->modP.aubry_andre.aa_J0_, this->modP.aubry_andre.aa_lambda0_,
													this->modP.aubry_andre.aa_beta0_, this->modP.aubry_andre.aa_phi0_, 0.0);
		break;
	case MY_MODELS::SYK2_M:
		_H = std::make_shared<SYK2<_T>>(_Ns, 0.0);
		break;
	case MY_MODELS::POWER_LAW_RANDOM_BANDED_M:
		_H = std::make_shared<PowerLawRandomBanded<_T>>(_Ns, 0.0);
		break;
	default:
		_H = std::make_shared<FreeFermions<_T>>(_Ns, this->modP.J1_, this->modP.J10_, 0.0);
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

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// ############################################################### I N F O S ################################################################

// ##########################################################################################################################################

template<typename _T>
inline void UI::get_inf_dir_ext(std::shared_ptr<Hamiltonian<_T>> _H, std::string& _dir, std::string& modelInfo, std::string& ext)
{
	_H->updateInfo();
	modelInfo	= _H->getInfo();
	_dir		= makeDirsC(this->mainDir, _dir, modelInfo);
	ext			= ".h5";
}

template<typename _T>
inline void UI::get_inf_dir_ext_r(std::shared_ptr<Hamiltonian<_T>> _H, std::string& _dir, std::string& modelInfo, std::string& randomStr, std::string& ext)
{
	_H->updateInfo();
	modelInfo	= _H->getInfo();
	randomStr	= FileParser::appWRandom("", _H->ran_);
	_dir		= makeDirsC(this->mainDir, _dir, modelInfo);
	ext			= ".h5";
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<typename _T>
inline void UI::checkETH(std::shared_ptr<Hamiltonian<_T>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

	// check the random field
	auto _rH	= this->modP.qsm.qsm_h_r_;
	auto _rA	= this->modP.qsm.qsm_h_ra_;
	size_t _Ns	= this->modP.qsm.qsm_Ntot_;
	u64 _Nh		= ULLPOW(_Ns);

	// stats in the middle number (diagonal part)
	size_t _Dt	= this->modP.modMidStates_ >= 1.0 ? std::min(u64(this->modP.modMidStates_), _Nh) : std::max(u64(1), u64(_Nh * this->modP.modMidStates_));

	// get info
	std::string modelInfo	= _H->getInfo();
	std::string randomStr	= FileParser::appWRandom("", _H->ran_);
	std::string dir			= makeDirsC(this->mainDir, "QSM_MAT_ELEM", modelInfo, (_rH != 0) ? VEQV(dh, _rA) + "_" + STRP(_rH, 3) : "");
	std::string extension	= ".h5";

	// set seed
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	// set the placeholder for the values to save
	v_1d<arma::Mat<double>> _ent = v_1d<arma::Mat<double>>(_Ns, -1e5 * arma::Mat<double>(_Dt, this->modP.modRanN_, arma::fill::ones));
	arma::Mat<double> _gaps = -1e5 * arma::Col<double>(this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _ipr1 = -1e5 * arma::Mat<double>(_Dt, this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _ipr2 = -1e5 * arma::Mat<double>(_Dt, this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _en	= -1e5 * arma::Mat<double>(_H->getHilbertSize(), this->modP.modRanN_, arma::fill::zeros);

	// choose the random position inside the dot for the correlation operators
	uint _pos = this->ran_.randomInt(0, this->modP.qsm.qsm_N_);

	// create the operators (for them we calculate the offdiagonals)
	auto _sx = Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, _Ns - 1);
	auto _sxc = Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, { _pos, (uint)_Ns - 1 });
	auto _szc = Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { _pos, (uint)_Ns - 1 });

	// for those we calculate the diagonals only (for each site)
	std::vector<Operators::Operator<double>> _sz_is;
	std::vector<std::string> _sz_i_names;
	// create the diagonal operators for spin z at each side
	for (auto i = 0; i < _Ns; ++i)
	{
		_sz_is.push_back(Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, i));
		_sz_i_names.push_back("sz_" + STR(i));
	}

	// create the matrices
	//Operators::OpVec_glb_t _ops			= { _sx, _sxh, _sxc, _sz, _szc };	
	v_1d<Operators::Operator<double>> _ops = { _sx, _sxc, _szc };
	v_1d<std::string> _opsN = { "sx_l", "sx_c", "sz_c" };
	for (auto i = 0; i < _Ns; ++i)
	{
		_ops.push_back(_sz_is[i]);
		_opsN.push_back(_sz_i_names[i]);
	}

	// create the measurement class
	Measurement<_T> _measure(this->modP.qsm.qsm_Ntot_, dir, _ops, _opsN, 1, _Nh);

	// to save the operators (those elements will be stored for each operator separately)
	// a given matrix element <n|O|n> will be stored in i'th column of the i'th operator
	// the n'th row in the column will be the state index
	// the columns corresponds to realizations of disorder
	v_1d<arma::Mat<_T>> _diagElems(_ops.size(), -1e5 * arma::Mat<_T>(_Dt, this->modP.modRanN_, arma::fill::ones));

	// create the saving function
	std::function<void(uint)> _diagSaver = [&](uint _r)
		{
			// save the matrices

			for(int i = 0; i < _Ns; ++i)
			{
				// save the entropies (only append when i > 0)
				saveAlgebraic(dir, "entro" + randomStr + extension, _ent[i], STR(i), i > 0);
			}
			// save the iprs and the energy to the same file
			saveAlgebraic(dir, "stat" + randomStr + extension, _gaps, "gap_ratio", false);
			saveAlgebraic(dir, "stat" + randomStr + extension, _ipr1, "part_entropy_q=1", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _ipr2, "part_entropy_q=2", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _en, "energy", true);

			// diagonal elements (only append when _opi > 0)
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "diag" + randomStr + extension, _diagElems[_opi], _name, _opi > 0);
			}
			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
		};

	// go through realizations
	for (int _r = 0; _r < this->modP.modRanN_; ++_r)
	{
		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
		_timer.checkpoint(STR(_r));

		// -----------------------------------------------------------------------------
		
		LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
		_H->clearH();
		_H->randomize(this->modP.qsm.qsm_h_ra_, _rH, {"h"});

		// -----------------------------------------------------------------------------

		// set the Hamiltonian
		_H->buildHamiltonian();
		_H->diagH(false);
		LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
		
		// -----------------------------------------------------------------------------
		
#if QSM_CHECK_HS_NORM
#	ifdef _DEBUG
		if((_r == 0 || _r == this->modP.modRanN_ - 1) && _Ns < 11)
		{
			_measure.checkG_mat(_H->getEigVec());
		}
#	endif
#endif
		
		// -----------------------------------------------------------------------------
				
		// get the average energy index and the points around it on the diagonal
		u64 _minIdxDiag						= 0; 
		u64 _maxIdxDiag						= 0;
		std::tie(_minIdxDiag, _maxIdxDiag)	= _H->getEnArndAvIdx(_Dt / 2, _Dt / 2);
		const auto _offdiagPairs			= _H->getEnPairsIdx(_minIdxDiag, _maxIdxDiag, this->modP.modEnDiff_);

		// -----------------------------------------------------------------------------

		// save the energies
		{
			_en.col(_r) = _H->getEigVal();
		}

		// -----------------------------------------------------------------------------
		
		// save the offdiagonal elements
		// this will be stored as a matrix with the following structure:
		// the 0 column will be the higher energy
		// the 1 column will be the lower energy
		// the 2 column will be the matrix element
		// save only for sx_l, sx_c, sz_c
		// in the vector we store different operators
		v_1d<arma::Mat<_T>> _offDiag(3, -1e5 * arma::Mat<_T>(_offdiagPairs.size(), 3, arma::fill::ones));

		// -----------------------------------------------------------------------------
		
		// calculator of the properties
		{
			// -----------------------------------------------------------------------------
			
			{
				// calculate the eigenlevel statistics
				_gaps(_r) = SystemProperties::eigenlevel_statistics(_H->getEigVal());
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
				for (long long _start = _minIdxDiag; _start < _maxIdxDiag; ++_start)
				{
					_ipr1(_start - _minIdxDiag, _r) = SystemProperties::information_entropy(_H->getEigVec(_start));
					_ipr2(_start - _minIdxDiag, _r) = std::log(1.0 / SystemProperties::participation_ratio(_H->getEigVec(_start), 2.0));
				}

				// -----------------------------------------------------------------------------
				
				// entanglement entropy
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif			
				for (long long _start = _minIdxDiag; _start < _maxIdxDiag; ++_start)
				{
					for (int i = 1; i <= _Ns; i++)
					{
						// calculate the entanglement entropy
						//_entr(_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), _Ns - 1, _Hs);
						//_entr(_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), _Ns, _Hs);
						uint _maskA	= 1 << (i - 1);
						uint _enti	= _Ns - i;
						_ent[_enti](_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), 1, _Ns, _maskA, DensityMatrix::RHO_METHODS::SCHMIDT, 2);
					}
				}
			}

			// -----------------------------------------------------------------------------
			
			// diagonal
			{
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (long long _start = _minIdxDiag; _start < _maxIdxDiag; ++_start)
				{
					// calculate the diagonal elements
					const auto& _measured	= _measure.measureG(_H->getEigVec(_start));
					const long long j		= _start - _minIdxDiag;
					// save the diagonal elements
					for (uint i = 0; i < _measured.size(); ++i)
					{
						_diagElems[i](j, _r) = _measured[i];
					}
				}
			}

			// -----------------------------------------------------------------------------
			
			// offdiagonal
			{
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (long long _start = 0; _start < _offdiagPairs.size(); ++_start)
				{
					const auto& [w, high, low]	= _offdiagPairs[_start];
					//const auto& _measured		= _measure.measureG(_H->getEigVec(high), _H->getEigVec(low));
					const auto& _measured		= _measure.measureG(_H->getEigVec(low), _H->getEigVec(high), _offDiag.size());

					// save the off-diagonal elements
					for (uint i = 0; i < _offDiag.size(); ++i)
					{
						_offDiag[i](_start, 0) = _H->getEigVal(high);
						_offDiag[i](_start, 1) = _H->getEigVal(low);
						//_offDiag[i](_start, 2) = high;
						//_offDiag[i](_start, 3) = low;
						_offDiag[i](_start, 2) = _measured[i];
					}
				}

			}

		}

		// save the checkpoints
		{
			// save the diagonals
			_diagSaver(_r);

			// save the offdiagonal
			for (uint _opi = 0; _opi < _offDiag.size(); ++_opi)
			{
				// get the name of the operator
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "offdiag_" + _name + randomStr + extension, _offDiag[_opi], STR(_r), _r > 0);
			}

		}
		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_diagSaver(this->modP.modRanN_);

	// save the command directly to the file
	{
		std::string dir_in = makeDirs(this->mainDir, "QSM_MAT_ELEM");
		std::ofstream file(dir_in + "qsm_scp.log", std::ios::app);
		file << "scp -3 -r scp://klimak97@ui.wcss.pl:22//" + dir + " ./" << std::endl;
		file.close();
		std::ofstream file_rsync(dir_in + "qsm_rsync.log", std::ios::app);
		file_rsync << "rsync -rv --rsh --ignore-existing -e 'ssh -p 22' klimak97@ui.wcss.pl:mylustre-hpc-maciek/QSolver/DATA_LJUBLJANA_BIG_SWEEP ./";
		file_rsync.close();
	}

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%