#pragma once
/***************************************
* Defines the user interface class based
* on a general UI class. All methods for 
* this software stored.
* DEC 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#include "./user_interface_par.hpp"

// ##########################################################################################################################################

/*
* @brief User interface class for the QES
*/
class UI : public UserInterface 
{
protected:
	UI_PARAMS::LatP latP;								// LATTICE params
	UI_PARAMS::SymP symP;								// SYMMETRY params
	UI_PARAMS::NqsP nqsP;								// NQS params
	UI_PARAMS::ModP modP;								// MODEL params

	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	bool isComplex_		= false;						// checks the complex sector
	bool useComplex_	= false;						// forces complex sector choice

	// ^^^^^^^^^ FOR DOUBLE ^^^^^^^^^					
	Hilbert::HilbertSpace<double>						hilDouble;
	std::shared_ptr<Hamiltonian<double>>				hamDouble;
	//std::shared_ptr<QuadraticHamiltonian<double>>		qhamDouble;

	// ^^^^^^^^ FOR COMPLEX ^^^^^^^^^
	Hilbert::HilbertSpace<cpx>							hilComplex;
	std::shared_ptr<Hamiltonian<cpx>>					hamComplex;
	//std::shared_ptr<QuadraticHamiltonian<cpx>>			qhamComplex;

	// ^^^^^^^^^^^^ NQS ^^^^^^^^^^^^^
	std::shared_ptr<NQS_NS::NQS<2, cpx>>				nqsCpx;
	std::shared_ptr<NQS_NS::NQS<2, double>>				nqsDouble;

	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	void setDefaultMap()								final override;

private:
	// standard elements
	template<typename _T>
	void get_inf_dir_ext_r(std::shared_ptr<Hamiltonian<_T>> _H, std::string& _dir, std::string& modelInfo, std::string& randomStr, std::string& ext);
	template<typename _T>
	void get_inf_dir_ext(std::shared_ptr<Hamiltonian<_T>> _H, std::string& _dir, std::string& modelInfo, std::string& ext);

	// reset model
	void resetEd()										{ if (this->hamComplex) this->hamComplex.reset(); if (this->hamDouble) this->hamDouble.reset();	};
	void resetQuadratic()								{ if (this->hamComplex) this->hamComplex.reset(); if (this->hamDouble) this->hamDouble.reset(); };

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
	void nqsExcited();

	// ##################### QUADRATIC #####################

	template<typename _T>
	void quadraticStatesManifold(std::shared_ptr<Hamiltonian<_T>>& _H);

	template<typename _T>
	void quadraticSpectralFunction(std::shared_ptr<Hamiltonian<_T>>& _H);

	// ##################### SPIN MODELS ###################

	template <typename _T = double, typename _OpT = Operators::Operator<_T>>
	std::pair<v_sp_t<_OpT>, strVec>
		ui_getoperators(const size_t _Nh, bool _isquadratic = true, bool _ismanybody = true)
			requires std::is_base_of_v<Operators::GeneralOperator<_T, typename _OpT::repType, typename _OpT::repTypeV>, _OpT> &&
			std::is_same_v<typename _OpT::innerType, _T>;

	std::pair<v_1d<std::shared_ptr<Operators::Operator<double>>>, strVec>
		ui_eth_getoperators(const size_t _Nh, bool _isquadratic = true, bool _ismanybody = true);
	template<typename _T>
	void ui_eth_randomize(std::shared_ptr<Hamiltonian<_T>> _H, int _r = 0, uint _spinChanged = 0);
	template<typename _T>
	void checkETH(std::shared_ptr<Hamiltonian<_T>> _H);
	
	template<typename _T>
	void checkETH_statistics(std::shared_ptr<Hamiltonian<_T>> _H);

	template<typename _T>
	std::array<double, 6> checkETH_statistics_mat_elems(
		u64 _start, u64 _end,
		std::atomic<size_t>& _statiter,
		int _th,
		u64 _Nh,
		Hamiltonian<_T>* _H,
		const arma::Mat<_T>& _overlaps,
		HistogramAverage<double>& _histAv,
		HistogramAverage<double>& _histAvTypical,
		arma::Mat<double>* _offdiagElemsOmega,
		arma::Mat<double>* _offdiagElemsOmegaLow,
		VMAT<_T>* _offdiagElems,
		VMAT<_T>* _offdiagElemsLow,
		VMAT<double>* _offdiagElemesStat,
		const double _bandwidth = 2.0,
		const double _energyAt	= 0.0,
		int _opi 				= 0,
		int _r 					= 0
	);

	template<typename _T>
	void checkETH_time_evo(std::shared_ptr<Hamiltonian<_T>> _H);

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% D E F I N I T I O N S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	bool defineLattice();
	bool defineLattice(std::shared_ptr<Lattice>& _lat, LatticeTypes _typ = LatticeTypes::SQ);

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
	bool defineModel(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<Lattice>& _lat);

	// NQS
	template<typename _T, uint _spinModes = 2, typename _Ht = _T, typename _stateType = double>
	void defineNQS(std::shared_ptr<Hamiltonian<_Ht>>& _H, 
		std::shared_ptr<NQS_NS::NQS<_spinModes, _Ht, _T, _stateType>>& _NQS, 
		const v_sp_t<NQS_NS::NQS<_spinModes, _Ht, _T, _stateType>>& _NQSl = {}, 
		const v_1d<double>& _beta = {});

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

	bool remainingSlurmTime(int _r, Timer* _timer, long _single_run_time, long _job_time = -1, std::string _checkpoint = "START");

	// ############################################### N Q S 

	void makeSimNQS();
	void makeSimNQSExcited();

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
		// UI_OTHER_MAP(m		, this->nqsP.nMcSteps_		, FHANDLE_PARAM_HIGHER0),		// mcsteps	
		// UI_OTHER_MAP(nb		, this->nqsP.nBlocks_		, FHANDLE_PARAM_HIGHER0),		// number of blocks
		// UI_OTHER_MAP(bs		, this->nqsP.blockSize_		, FHANDLE_PARAM_HIGHER0),		// block size
		UI_OTHER_MAP(nh		, this->nqsP.nqs_nh_		, FHANDLE_PARAM_HIGHER0),			// hidden params
		UI_OTHER_MAP(nf		, this->nqsP.nFlips_		, FHANDLE_PARAM_HIGHER0),			// flip number
		// for collecting in nqs
		// UI_OTHER_MAP(bsS	, this->nqsP.blockSizeS_	, FHANDLE_PARAM_HIGHER0),			// block size samples
		// UI_OTHER_MAP(mcS	, this->nqsP.nMcSamples_	, FHANDLE_PARAM_HIGHER0),			// mcsteps samples
		// UI_OTHER_MAP(nbS	, this->nqsP.nSBlocks_		, FHANDLE_PARAM_HIGHER0),			// number of blocks - samples
		UI_OTHER_MAP(dirNQS	, this->nqsP.loadNQS_		, FHANDLE_PARAM_DEFAULT),			// directory to load the weights from

		// --------------- directory parameters ---------------
		{"f"				, std::make_tuple(""		, FHANDLE_PARAM_DEFAULT)},			// file to read from directory
		
		// ---------------- lattice parameters ----------------
		UI_OTHER_MAP(d		, this->latP._dim			, FHANDLE_PARAM_BETWEEN(1., 3.)),
		UI_OTHER_MAP(bc		, this->latP._bc			, FHANDLE_PARAM_BETWEEN(0., 3.)),
		UI_OTHER_MAP(l		, this->latP._typ			, FHANDLE_PARAM_BETWEEN(0., 5)),
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

/*
* @brief Based on the input by the user, creates a Hilbert space for future purposes
*/
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
	// if user provides the Hilbert space, use it!
	if (!defineHilbert<_T>(_Hil))
		return false;

	// switch the model types
	switch (this->modP.modTyp_)
	{
	// !!!!!!!!!!!!!!!!!!!!!!! SPIN !!!!!!!!!!!!!!!!!!!!!!!
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
		{
			v_1d<double> _ain(_Hil.getNs() - this->modP.qsm.qsm_N_);
			std::fill(_ain.begin(), _ain.end(), this->modP.qsm.qsm_alpha_[0]);
			_H = std::make_shared<QSM<_T>>(std::move(_Hil), 
				this->modP.qsm.qsm_N_, this->modP.qsm.qsm_gamma_, this->modP.qsm.qsm_g0_,
				_ain, this->modP.qsm.qsm_h_, this->modP.qsm.qsm_xi_);
		}
		break;
	case MY_MODELS::RP_M:
		_H = std::make_shared<RosenzweigPorter<_T>>(std::move(_Hil), 
			this->modP.rosenzweig_porter.rp_g_[0], !this->modP.rosenzweig_porter.rp_single_particle_);
		break;
	case MY_MODELS::ULTRAMETRIC_M:
		_H = std::make_shared<Ultrametric<_T>>(std::move(_Hil), this->modP.ultrametric.um_N_, this->modP.ultrametric.um_g_, this->modP.ultrametric.um_alpha_);
		break;

	// !!!!!!!!!!!!!!!!!!!!!!! QUADRATIC FERMIONS !!!!!!!!!!!!!!!!!!!!!!!
	case MY_MODELS::FREE_FERMIONS_M:
		_H = std::make_shared<FreeFermions<_T>>(std::move(_Hil), this->modP.J1_, this->modP.J10_, 0.0);
		break;
	case MY_MODELS::AUBRY_ANDRE_M:
		_H = std::make_shared<AubryAndre<_T>>(std::move(_Hil), this->modP.aubry_andre.aa_J_, this->modP.aubry_andre.aa_lambda_,
											this->modP.aubry_andre.aa_beta_, this->modP.aubry_andre.aa_phi_,
											this->modP.aubry_andre.aa_J0_, this->modP.aubry_andre.aa_lambda0_,
											this->modP.aubry_andre.aa_beta0_, this->modP.aubry_andre.aa_phi0_, 0.0);
		break;
	case MY_MODELS::SYK2_M:
		_H = std::make_shared<SYK2<_T>>(std::move(_Hil));
		break;
	case MY_MODELS::POWER_LAW_RANDOM_BANDED_M:
		_H = std::make_shared<PowerLawRandomBanded<_T>>(std::move(_Hil), this->modP.power_law_random_bandwidth.plrb_a_[0],
			this->modP.power_law_random_bandwidth.plrb_b_, this->modP.power_law_random_bandwidth.plrb_mb_);
		break;
	default:
		throw std::runtime_error("Model not defined!" 
								"Usage: "
								"MB: 1 - Ising, 2 - XYZ, 3 - Heisenberg-Kitaev, 4 - QSM, 5 - RP, 6 - Ultrametric, " 
								"SP: 100 - Free Fermions, 101 - Aubry-Andre, 102 - SYK2, 103 - Power Law Random Banded");
	}
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	return true;
}

template<typename _T>
inline bool UI::defineModel(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<Lattice>& _lat)
{
	// construct only if necessary
	//if (this->modP.modTyp_ < MY_MODELS::FREE_FERMIONS_M)
	Hilbert::HilbertSpace<_T> _Hil(_lat);
	
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
		{
			v_1d<double> _ain(_lat->get_Ns() - this->modP.qsm.qsm_N_);
			std::fill(_ain.begin(), _ain.end(), this->modP.qsm.qsm_alpha_[0]);
			_H = std::make_shared<QSM<_T>>(std::move(_Hil), 
				this->modP.qsm.qsm_N_, this->modP.qsm.qsm_gamma_, this->modP.qsm.qsm_g0_,
				_ain, this->modP.qsm.qsm_h_, this->modP.qsm.qsm_xi_);
		}
		break;
	case MY_MODELS::RP_M:
		_H = std::make_shared<RosenzweigPorter<_T>>(std::move(_Hil), 
			this->modP.rosenzweig_porter.rp_g_[0], !this->modP.rosenzweig_porter.rp_single_particle_);
		break;
	case MY_MODELS::ULTRAMETRIC_M:
		_H = std::make_shared<Ultrametric<_T>>(std::move(_Hil), this->modP.ultrametric.um_N_, this->modP.ultrametric.um_g_, this->modP.ultrametric.um_alpha_);
		break;
		// --------------------------- QUADRATIC MODELS ---------------------------
	case MY_MODELS::FREE_FERMIONS_M:
		_H = std::make_shared<FreeFermions<_T>>(_lat, this->modP.J1_, this->modP.J10_, 0.0);
		break;
	case MY_MODELS::AUBRY_ANDRE_M:
		_H = std::make_shared<AubryAndre<_T>>(_lat, this->modP.aubry_andre.aa_J_, this->modP.aubry_andre.aa_lambda_,
			this->modP.aubry_andre.aa_beta_, this->modP.aubry_andre.aa_phi_,
			this->modP.aubry_andre.aa_J0_, this->modP.aubry_andre.aa_lambda0_,
			this->modP.aubry_andre.aa_beta0_, this->modP.aubry_andre.aa_phi0_, 0.0);
		break;
	case MY_MODELS::SYK2_M:
		_H = std::make_shared<SYK2<_T>>(_lat, 0.0);
		break;
	case MY_MODELS::POWER_LAW_RANDOM_BANDED_M:
		_H = std::make_shared<PowerLawRandomBanded<_T>>(_lat, this->modP.power_law_random_bandwidth.plrb_a_[0],
			this->modP.power_law_random_bandwidth.plrb_b_, this->modP.power_law_random_bandwidth.plrb_mb_);
		break;
	default:
		throw std::runtime_error("Model not defined!" 
								"Usage: "
								"MB: 1 - Ising, 2 - XYZ, 3 - Heisenberg-Kitaev, 4 - QSM, 5 - RP, 6 - Ultrametric, " 
								"SP: 100 - Free Fermions, 101 - Aubry-Andre, 102 - SYK2, 103 - Power Law Random Banded");
	}
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	return true;
}

template<typename _T>
inline bool UI::defineModel(std::shared_ptr<Hamiltonian<_T>>& _H, uint _Ns)
{
	// construct only if necessary
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
		{
			_H = std::make_shared<QSM<_T>>(std::move(_Hil), 
				this->modP.qsm.qsm_N_, this->modP.qsm.qsm_gamma_, this->modP.qsm.qsm_g0_,
				this->modP.qsm.qsm_alpha_, this->modP.qsm.qsm_h_, this->modP.qsm.qsm_xi_);
		}
		break;
	case MY_MODELS::RP_M:
		_H = std::make_shared<RosenzweigPorter<_T>>(std::move(_Hil), 
			this->modP.rosenzweig_porter.rp_g_[0], !this->modP.rosenzweig_porter.rp_single_particle_);
		break;
	case MY_MODELS::ULTRAMETRIC_M:
		_H = std::make_shared<Ultrametric<_T>>(std::move(_Hil), this->modP.ultrametric.um_N_, this->modP.ultrametric.um_g_, this->modP.ultrametric.um_alpha_);
		break;
		// --------------------------- QUADRATIC MODELS ---------------------------
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
		_H = std::make_shared<PowerLawRandomBanded<_T>>(_Ns, this->modP.power_law_random_bandwidth.plrb_a_[0],
			this->modP.power_law_random_bandwidth.plrb_b_, this->modP.power_law_random_bandwidth.plrb_mb_);
		break;
	default:
		throw std::runtime_error("Model not defined!" 
								"Usage: "
								"MB: 1 - Ising, 2 - XYZ, 3 - Heisenberg-Kitaev, 4 - QSM, 5 - RP, 6 - Ultrametric, " 
								"SP: 100 - Free Fermions, 101 - Aubry-Andre, 102 - SYK2, 103 - Power Law Random Banded");
		break;
	}
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	return true;
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
//	_timer.reset();
//	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);
//
//	// check the random field
//	auto _rH	= this->modP.qsm.qsm_h_r_;
//	auto _rA	= this->modP.qsm.qsm_h_ra_;
//	size_t _Ns	= this->modP.qsm.qsm_Ntot_;
//	u64 _Nh		= ULLPOW(_Ns);
//
//	// stats in the middle number (diagonal part)
//	size_t _Dt	= this->modP.modMidStates_ >= 1.0 ? std::min(u64(this->modP.modMidStates_), _Nh) : std::max(u64(1), u64(_Nh * this->modP.modMidStates_));
//
//	// get info
//	std::string modelInfo	= _H->getInfo();
//	std::string randomStr	= FileParser::appWRandom("", _H->ran_);
//	std::string dir			= makeDirsC(this->mainDir, "QSM_MAT_ELEM", modelInfo, (_rH != 0) ? VEQV(dh, _rA) + "_" + STRP(_rH, 3) : "");
//	std::string extension	= ".h5";
//
//	// set seed
//	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);
//
//	// set the placeholder for the values to save
//	v_1d<arma::Mat<double>> _ent = v_1d<arma::Mat<double>>(_Ns, -1e5 * arma::Mat<double>(_Dt, this->modP.modRanN_, arma::fill::ones));
//	arma::Mat<double> _gaps = -1e5 * arma::Col<double>(this->modP.modRanN_, arma::fill::ones);
//	arma::Mat<double> _ipr1 = -1e5 * arma::Mat<double>(_Dt, this->modP.modRanN_, arma::fill::ones);
//	arma::Mat<double> _ipr2 = -1e5 * arma::Mat<double>(_Dt, this->modP.modRanN_, arma::fill::ones);
//	arma::Mat<double> _en	= -1e5 * arma::Mat<double>(_H->getHilbertSize(), this->modP.modRanN_, arma::fill::zeros);
//
//	// choose the random position inside the dot for the correlation operators
//	uint _pos = this->ran_.randomInt(0, this->modP.qsm.qsm_N_);
//
//	// create the operators (for them we calculate the offdiagonals)
//	auto _sx = Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, _Ns - 1);
//	auto _sxc = Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, { _pos, (uint)_Ns - 1 });
//	auto _szc = Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { _pos, (uint)_Ns - 1 });
//
//	// for those we calculate the diagonals only (for each site)
//	std::vector<Operators::Operator<double>> _sz_is;
//	std::vector<std::string> _sz_i_names;
//	// create the diagonal operators for spin z at each side
//	for (auto i = 0; i < _Ns; ++i)
//	{
//		_sz_is.push_back(Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, i));
//		_sz_i_names.push_back("sz_" + STR(i));
//	}
//
//	// create the matrices
//	//Operators::OpVec_glb_t _ops			= { _sx, _sxh, _sxc, _sz, _szc };	
//	v_1d<Operators::Operator<double>> _ops = { _sx, _sxc, _szc };
//	v_1d<std::string> _opsN = { "sx_l", "sx_c", "sz_c" };
//	for (auto i = 0; i < _Ns; ++i)
//	{
//		_ops.push_back(_sz_is[i]);
//		_opsN.push_back(_sz_i_names[i]);
//	}
//
//	// create the measurement class
//	Measurement<_T> _measure(this->modP.qsm.qsm_Ntot_, dir, _ops, _opsN, 1, _Nh);
//
//	// to save the operators (those elements will be stored for each operator separately)
//	// a given matrix element <n|O|n> will be stored in i'th column of the i'th operator
//	// the n'th row in the column will be the state index
//	// the columns corresponds to realizations of disorder
//	v_1d<arma::Mat<_T>> _diagElems(_ops.size(), -1e5 * arma::Mat<_T>(_Dt, this->modP.modRanN_, arma::fill::ones));
//
//	// create the saving function
//	std::function<void(uint)> _diagSaver = [&](uint _r)
//		{
//			// save the matrices
//
//			for(int i = 0; i < _Ns; ++i)
//			{
//				// save the entropies (only append when i > 0)
//				saveAlgebraic(dir, "entro" + randomStr + extension, _ent[i], STR(i), i > 0);
//			}
//			// save the iprs and the energy to the same file
//			saveAlgebraic(dir, "stat" + randomStr + extension, _gaps, "gap_ratio", false);
//			saveAlgebraic(dir, "stat" + randomStr + extension, _ipr1, "part_entropy_q=1", true);
//			saveAlgebraic(dir, "stat" + randomStr + extension, _ipr2, "part_entropy_q=2", true);
//			saveAlgebraic(dir, "stat" + randomStr + extension, _en, "energy", true);
//
//			// diagonal elements (only append when _opi > 0)
//			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
//			{
//				auto _name = _measure.getOpGN(_opi);
//				saveAlgebraic(dir, "diag" + randomStr + extension, _diagElems[_opi], _name, _opi > 0);
//			}
//			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
//		};
//
//	// go through realizations
//	for (int _r = 0; _r < this->modP.modRanN_; ++_r)
//	{
//		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
//		_timer.checkpoint(STR(_r));
//
//		// -----------------------------------------------------------------------------
//		
//		LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
//		_H->clearH();
//		_H->randomize(this->modP.qsm.qsm_h_ra_, _rH, {"h"});
//
//		// -----------------------------------------------------------------------------
//
//		// set the Hamiltonian
//		_H->buildHamiltonian();
//		_H->diagH(false);
//		LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
//		
//		// -----------------------------------------------------------------------------
//		
//#if QSM_CHECK_HS_NORM
//#	ifdef _DEBUG
//		if((_r == 0 || _r == this->modP.modRanN_ - 1) && _Ns < 11)
//		{
//			_measure.checkG_mat(_H->getEigVec());
//		}
//#	endif
//#endif
//		
//		// -----------------------------------------------------------------------------
//				
//		// get the average energy index and the points around it on the diagonal
//		u64 _minIdxDiag						= 0; 
//		u64 _maxIdxDiag						= 0;
//		std::tie(_minIdxDiag, _maxIdxDiag)	= _H->getEnArndAvIdx(_Dt / 2, _Dt / 2);
//		const auto _offdiagPairs			= _H->getEnPairsIdx(_minIdxDiag, _maxIdxDiag, this->modP.modEnDiff_);
//
//		// -----------------------------------------------------------------------------
//
//		// save the energies
//		{
//			_en.col(_r) = _H->getEigVal();
//		}
//
//		// -----------------------------------------------------------------------------
//		
//		// save the offdiagonal elements
//		// this will be stored as a matrix with the following structure:
//		// the 0 column will be the higher energy
//		// the 1 column will be the lower energy
//		// the 2 column will be the matrix element
//		// save only for sx_l, sx_c, sz_c
//		// in the vector we store different operators
//		v_1d<arma::Mat<_T>> _offDiag(3, -1e5 * arma::Mat<_T>(_offdiagPairs.size(), 3, arma::fill::ones));
//
//		// -----------------------------------------------------------------------------
//		
//		// calculator of the properties
//		{
//			// -----------------------------------------------------------------------------
//			
//			{
//				// calculate the eigenlevel statistics
//				_gaps(_r) = SystemProperties::eigenlevel_statistics(_H->getEigVal());
//				LOGINFO(StrParser::colorize(VEQ(_gaps(_r, 0)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
//				LOGINFO(_timer.point(STR(_r)), "Gap ratios", 1);
//			}
//
//			// -----------------------------------------------------------------------------
//			
//			// other measures
//			{
//				// participation ratios
//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threadNum)
//#endif							
//				for (long long _start = _minIdxDiag; _start < _maxIdxDiag; ++_start)
//				{
//					_ipr1(_start - _minIdxDiag, _r) = SystemProperties::information_entropy(_H->getEigVec(_start));
//					_ipr2(_start - _minIdxDiag, _r) = std::log(1.0 / SystemProperties::participation_ratio(_H->getEigVec(_start), 2.0));
//				}
//
//				// -----------------------------------------------------------------------------
//				
//				// entanglement entropy
//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threadNum)
//#endif			
//				for (long long _start = _minIdxDiag; _start < _maxIdxDiag; ++_start)
//				{
//					for (int i = 1; i <= _Ns; i++)
//					{
//						// calculate the entanglement entropy
//						//_entr(_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), _Ns - 1, _Hs);
//						//_entr(_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), _Ns, _Hs);
//						uint _maskA	= 1 << (i - 1);
//						uint _enti	= _Ns - i;
//						_ent[_enti](_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), 1, _Ns, _maskA, DensityMatrix::RHO_METHODS::SCHMIDT, 2);
//					}
//				}
//			}
//
//			// -----------------------------------------------------------------------------
//			
//			// diagonal
//			{
//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threadNum)
//#endif
//				for (long long _start = _minIdxDiag; _start < _maxIdxDiag; ++_start)
//				{
//					// calculate the diagonal elements
//					const auto& _measured	= _measure.measureG(_H->getEigVec(_start));
//					const long long j		= _start - _minIdxDiag;
//					// save the diagonal elements
//					for (uint i = 0; i < _measured.size(); ++i)
//					{
//						_diagElems[i](j, _r) = _measured[i];
//					}
//				}
//			}
//
//			// -----------------------------------------------------------------------------
//			
//			// offdiagonal
//			{
//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threadNum)
//#endif
//				for (long long _start = 0; _start < _offdiagPairs.size(); ++_start)
//				{
//					const auto& [w, high, low]	= _offdiagPairs[_start];
//					//const auto& _measured		= _measure.measureG(_H->getEigVec(high), _H->getEigVec(low));
//					const auto& _measured		= _measure.measureG(_H->getEigVec(low), _H->getEigVec(high), _offDiag.size());
//
//					// save the off-diagonal elements
//					for (uint i = 0; i < _offDiag.size(); ++i)
//					{
//						_offDiag[i](_start, 0) = _H->getEigVal(high);
//						_offDiag[i](_start, 1) = _H->getEigVal(low);
//						//_offDiag[i](_start, 2) = high;
//						//_offDiag[i](_start, 3) = low;
//						_offDiag[i](_start, 2) = _measured[i];
//					}
//				}
//
//			}
//
//		}
//
//		// save the checkpoints
//		{
//			// save the diagonals
//			_diagSaver(_r);
//
//			// save the offdiagonal
//			for (uint _opi = 0; _opi < _offDiag.size(); ++_opi)
//			{
//				// get the name of the operator
//				auto _name = _measure.getOpGN(_opi);
//				saveAlgebraic(dir, "offdiag_" + _name + randomStr + extension, _offDiag[_opi], STR(_r), _r > 0);
//			}
//
//		}
//		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
//		// -----------------------------------------------------------------------------
//	}
//
//	// save the diagonals
//	_diagSaver(this->modP.modRanN_);
//
//	// save the command directly to the file
//	{
//		std::string dir_in = makeDirs(this->mainDir, "QSM_MAT_ELEM");
//		std::ofstream file(dir_in + "qsm_scp.log", std::ios::app);
//		file << "scp -3 -r scp://klimak97@ui.wcss.pl:22//" + dir + " ./" << std::endl;
//		file.close();
//		std::ofstream file_rsync(dir_in + "qsm_rsync.log", std::ios::app);
//		file_rsync << "rsync -rv --rsh --ignore-existing -e 'ssh -p 22' klimak97@ui.wcss.pl:mylustre-hpc-maciek/QSolver/DATA_LJUBLJANA_BIG_SWEEP ./";
//		file_rsync.close();
//	}
//
//	// bye
//	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

// ##########################################################################################################################################

inline std::string PREFIX(const std::string& name, const std::string& random_tag, const std::string& extension, std::string time_tag = "") {
	if (time_tag.empty()) 
	{
		time_tag = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
	}
	return name + random_tag + "_t=" + time_tag + extension;
}

namespace TimeEvo
{

	// ----------------------------------------------------------------
	// File name prefixes
	// ----------------------------------------------------------------
	inline const std::string KEY_ME                      = "/ME";                        // for mean energy
	inline const std::string KEY_EPS                     = "eps_micro";                  // for a given energy density
	inline const std::string KEY_MEAN_LVL                = "mean_lvl_spacing";           // mean level spacing
	inline const std::string KEY_HEIS_TIME_GAMMA         = "heis_time_gamma";            // Heisenberg time (gamma)
	inline const std::string KEY_HEIS_TIME_AROUND_MEAN   = "heis_time_around_mean";      // Heisenberg time around mean
	inline const std::string KEY_HEIS_TIME_AROUND_MEAN_T = "heis_time_around_mean_typ";  // Heisenberg time around mean (typical)
	inline const std::string KEY_ENERGIES                = "energies";                   // energies
	inline const std::string KEY_MEAN                    = "mean";                       // Tr<psi|H|psi> / N_h
	inline const std::string KEY_MEAN_STATE              = "mean_state";                 // <psi_0|H|psi_0>
	inline const std::string KEY_MEAN_STATE2             = "mean_state2";                // <psi_0|H|psi_0>^2
	inline const std::string KEY_TIME                    = "time";                       // time
	inline const std::string KEY_TIME_SHORT              = "times/short";                // time short
	inline const std::string KEY_TIME_MEDIUM             = "times/medium";               // time medium
	inline const std::string KEY_TIME_MIDDLE             = "times/middle";               // time middle
	inline const std::string KEY_TIME_LONG               = "times/long";                 // time long
	inline const std::string KEY_ENT_ENTROPY_PREFIX      = "entanglement_entropy/";      // entanglement entropy
	inline const std::string KEY_ENT_SITES               = "entanglement_entropy/sites"; // entanglement entropy for sites
	inline const std::string KEY_PARTICIPATION           = "participation_entropy";      // participation entropy S_p = - sum_i (|psi_i|^4)
	inline const std::string KEY_ENERGY_DENSITY          = "energy_density";             // energy density
	inline const std::string KEY_REALIZATION             = "realization";                // number of realizations

	inline const std::string SUFFIX_ZERO                 = "/zero";                      // for zero energy state
	inline const std::string SUFFIX_ME                   = "/ME";                        // for mean energy state
	inline const std::string SUFIX_BIPARTITE             = "/bipartite";               	 // for bipartite entanglement
	inline const std::string SUFFIX_MICRO_ME             = "/micro/ME/1";                // for microcanonical ensemble
	inline const std::string SUFFIX_MICRO_ME2            = "/micro/ME/2";                // for microcanonical ensemble squared
	inline const std::string SUFFIX_DIAGONAL_ME          = "/diagonal/ME";               // for diagonal ensemble

	// For macros/functions that require arguments, use inline constexpr functions:
	inline std::string SUFFIX_EPS(double x)            { return "/eps=" + std::to_string(x); 				}
	inline std::string SUFFIX_MICRO_ED(double x)       { return "/micro/eps=" + std::to_string(x) + "/1"; 	}
	inline std::string SUFFIX_MICRO2_ED(double x)      { return "/micro/eps=" + std::to_string(x) + "/2"; 	}
	inline std::string SUFFIX_DIAGONAL_ED(double x)    { return "/diagonal/eps=" + std::to_string(x); 		}

	// File name prefixes (these require runtime variables, so keep as inline functions)
	inline std::string STAT_PREFIX(const std::string& random_tag, const std::string& extension, const std::string& time_tag) {
		return PREFIX("stat", random_tag, extension, time_tag);
	}
	inline std::string LDOS_PREFIX(const std::string& random_tag, const std::string& extension, const std::string& time_tag) {
		return PREFIX("ldos", random_tag, extension, time_tag);
	}
	inline std::string DIAG_PREFIX(const std::string& random_tag, const std::string& extension, const std::string& time_tag) {
		return PREFIX("diag", random_tag, extension, time_tag);
	}
	inline std::string EDENS_PREFIX(const std::string& random_tag, const std::string& extension, const std::string& time_tag) {
		return PREFIX("energydens", random_tag, extension, time_tag);
	}
	inline std::string EVO_PREFIX(const std::string& random_tag, const std::string& extension, const std::string& time_tag) {
		return PREFIX("evo", random_tag, extension, time_tag);
	}
	inline std::string AVS_PREFIX(const std::string& random_tag, const std::string& extension, const std::string& time_tag) {
		return PREFIX("avs", random_tag, extension, time_tag);
	}

	template <typename T>
	struct TimeEvoParams
	{
		size_t _Ns;
		size_t _Nh;
		size_t _hs_fraction_diag        = 0;
		bool _isQuadratic               = true;
		bool _isManyBody                = true;
	
		long double _bw_est;
		long double _dtau_est;
		long double _heisenberg_time_est;
		long double _thouless_est;
		long long   _ntimes;
		long long   _nrealiz            = 1;
	
		// stores the operators
		std::vector<std::shared_ptr<Operators::Operator<double>>> _ops;
		std::vector<std::string> _opsN;
	
		// stores the model info
		std::string modelInfo;
		std::string dir;
		std::string randomStr;
		std::string extension;
		std::string time_tag;

		// stores the time evolution
		bool _uniform_time;
		arma::Col<double> _timespace;
	
		// stores the check vectors
		v_1d<double> _to_check_microcanonical_eps;
		v_1d<double> _energy_densities;
		v_1d<size_t> _entropies_sites;
	
		// ----------------------------------------------------------------
		//! Functions
		// ----------------------------------------------------------------
	
		void set_timespace();
	
		// ----------------------------------------------------------------
		//! Set Hamiltonian parameters
		// ----------------------------------------------------------------
	
		void set_hamil_params(MY_MODELS _type, std::shared_ptr<Hamiltonian<T>> _H, size_t n_tot, size_t n_real);
	
		// ----------------------------------------------------------------
		//! Containers to hold all results
		// ----------------------------------------------------------------
	
		struct ResultContainers 
		{
			arma::Col<T> initial_state_me;                                  // [nh]
			arma::Mat<T> initial_state_ed;                                  // [nh x n_ops]
	
			// level statistics
			arma::Mat<double> mean_level_spacing;                           // [4 x n_real]
			arma::Mat<double> energies;                                     // [nh x n_real]
	
			// local density of states
			arma::Mat<double> ldos_me;                                      // [nh x n_real]
			std::vector<arma::Mat<double>> ldos_ed;                         // [n_edens][nh x n_real]
	
			// energy densities
			arma::Mat<T> energy_densities_me;                          		// [3 x n_real] - mean energy, <state|H|state>, <state|H2|state>
			std::vector<arma::Mat<T>> energy_densities_ed;             		// [n_edens][3 x n_real] - mean energy, <state|H|state>, <state|H2|state>
	
			// diagonal elements
			std::vector<arma::Mat<T>> diagonals;                            // [n_ops][nh x n_real]
	
			// time evolution
			std::vector<arma::Mat<T>> time_evo_me;                          // [n_ops][n_times x n_real]
			std::vector<arma::Col<T>> time_evo_zero_me;						// [n_ops][n_real]
			std::vector<std::vector<arma::Mat<T>>> time_evo_ed;             // [n_edens][n_ops][n_times x n_real]
			std::vector<std::vector<arma::Col<T>>> time_evo_zero_ed;        // [n_edens][n_ops][n_real]
	
			// entropies
			arma::Mat<double> time_participation_entropy;                   // [n_times x n_real]
			std::vector<arma::Mat<double>> time_ee;                        	// [n_sites][n_times x n_real]
			arma::Mat<double> time_ee_bipartite;                            // [n_times x n_real]
	
			// microcanonical averages
			std::vector<arma::Mat<T>> microcanonical_me;                   	// [n_ops][n_eps x n_real]
			std::vector<arma::Mat<double>> microcanonical_me2;              // [n_ops][n_eps x n_real]
			std::vector<std::vector<arma::Mat<T>>> microcanonical_ed;      	// [n_edens][n_ops][n_eps x n_real]
			std::vector<std::vector<arma::Mat<double>>> microcanonical_ed2; // [n_edens][n_ops][n_eps x n_real]
	
			// diagonal ensembles
			arma::Mat<double> diagonal_me;                                  // [n_ops x n_real]
			std::vector<arma::Mat<double>> diagonal_ed;                     // [n_edens][n_ops x n_real]
	
			// finished realisations
			size_t realisation_count = 0;                                   // number of realizations finished
	
		} _r;
	
		// ----------------------------------------------------------------
	
		void save_for_realization(size_t r);		
	
		// ----------------------------------------------------------------
	
		void allocate_result_containers();
		
		// ----------------------------------------------------------------
	};

	// --------------------------------------------------------------------

	template <typename T>
	void microcanonical_saver(
		size_t r,                                            // realization index
		size_t opi,                                          // operator index
		const std::vector<arma::Mat<T>>& diagonals,          // diagonals[opi] is [nh × n_real]
		const arma::Col<double>& soverlaps,                  // |<ψ|n>|² vector of length nh
		const std::vector<uint64_t>& mins,                   // window start indices per ε - given energy density
		const std::vector<uint64_t>& maxs,                   // window end indices per ε - given energy density
		arma::Mat<double>& diagvals,                         // [n_ops × n_real], to write long‑time avg here (the diagonal ensemble)
		std::vector<arma::Mat<T>>& microvals,                // [n_ops][n_eps × n_real] to save the microcanonical averages here
		std::vector<arma::Mat<double>>& microvals2           // [n_ops][n_eps × n_real] squares of the microcanonical averages
		);
	
	// --------------------------------------------------------------------

	template <typename T>
	void evolve_state(
		size_t                                   r,                      	// realization index
		const arma::Col<T>&                      init_state,             	// initial state vector [nh]
		const std::shared_ptr<Hamiltonian<T>>&   H,                     	// Hamiltonian object
		arma::Mat<double>*                       ldos,                   	// local density of states [nh x n_real] - optional
		arma::Mat<T>*                            energydensities,        	// energy densities [3 x n_real] - optional
		std::vector<arma::Mat<T>>*               micro,                  	// microcanonical averages [n_ops][n_eps x n_real] - optional
		std::vector<arma::Mat<double>>*          micro2,                 	// microcanonical averages squared [n_ops][n_eps x n_real] - optional
		arma::Mat<double>*                       diagvals,               	// diagonal ensemble [n_ops x n_real] - optional
		const std::vector<arma::Mat<T>>&         diagonals,              	// operator diagonals [n_ops][nh x n_real]
		std::vector<arma::Mat<T>>&               time_evo_me,            	// time evolution of operators [n_ops][n_times x n_real]
		std::vector<arma::Col<T>>*               time_zero_me,           	// zero time value of operators [n_ops][n_real] - optional
		std::vector<arma::Mat<double>>*          time_ee,                	// entanglement entropy [n_sites][n_times x n_real] - optional
		arma::Mat<double>*                       time_ee_bipartite,      	// entanglement entropy bipartite [n_times x n_real] - optional
		arma::Mat<double>*                       time_participation_entropy,// participation entropy [n_times x n_real] - optional
		const std::vector<GeneralizedMatrix<double>>& matrices,				// operators [n_ops][nh x nh]
		const arma::Col<double>&                 times,                  	// time grid [n_times]
		size_t                                   Ns                = 0,  	// total number of sites
		const std::vector<size_t>&               entropy_sites     = {},  	// sites for entanglement entropy
		const std::vector<double>&               micro_epsilons    = {},  	// epsilons for microcanonical windows
		bool                                     append            = false, // append values?
		bool                                     use_log           = false, // use log of values?
		bool                                     auto_cor          = false, // enable autocorrelation?
		int                                      thread_num        = 1      // number of threads
	);

	// --------------------------------------------------------------------
};


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%