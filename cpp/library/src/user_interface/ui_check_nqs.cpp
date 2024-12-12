#include "../../include/user_interface/user_interface.h"
#include "armadillo"
#include <memory>
#include <stdexcept>

// ##########################################################################################################################################

constexpr int UI_NQS_PRECISION = 6;

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief A placeholder for making the simulation with NQS. It uses the complex Hamiltonian.
*/
void UI::makeSimNQS()
{
	this->useComplex_ = true;
	this->defineModels(true);
	this->defineNQS<cpx>(this->hamComplex, this->nqsCpx);
}

/**
* @brief Configures and initiates the simulation for the Neural Quantum State (NQS) in an excited state.
*
* This function sets up the necessary parameters and configurations to run a simulation
* for the Neural Quantum State (NQS) in an excited state. It performs the following steps:
* - Enables the use of complex numbers for the simulation.
* - Defines the lattice structure required for the simulation.
* - Initializes the NQS in an excited state with complex numbers and a specified excitation level.
*/
void UI::makeSimNQSExcited()
{
	this->useComplex_ = true;
	this->defineLattice();
	this->nqsExcited<cpx, 2>();
}


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
* @brief Defines a Neural Quantum State (NQS) based on the provided Hamiltonian and parameters.
* 
* This function initializes an NQS object using the specified Hamiltonian and parameters. 
* It supports different types of NQS models, such as Restricted Boltzmann Machine (RBM) 
* and RBM with pre-processing (RBMPP). The function also sets various hyperparameters 
* for the NQS model.
* 
* @tparam _T The data type used in the Hamiltonian and NQS (e.g., double, float).
* @tparam _spinModes The number of spin modes in the NQS.
* 
* @param _H A shared pointer to the Hamiltonian object.
* @param _NQS A shared pointer to the NQS object to be defined.
* @param _NQSl A vector of shared pointers to existing NQS objects.
* @param _beta A vector of doubles representing the beta parameters.
* 
* @throws std::invalid_argument If an unknown NQS type is specified.
*/
template<typename _T, uint _spinModes>
inline void UI::defineNQS(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<NQS<_spinModes, _T>>& _NQS, 
		const v_1d<std::shared_ptr<NQS<_spinModes, _T>>>& _NQSl, const v_1d<double>& _beta)
{
	auto createNQS = [&](auto&&... args) -> std::shared_ptr<NQS<_spinModes, _T>> {
		switch (this->nqsP.type_)
		{
		case NQSTYPES::RBM_T:
			return std::make_shared<RBM_S<_spinModes, _T>>(std::forward<decltype(args)>(args)...);
		case NQSTYPES::RBMPP_T:
			return std::make_shared<RBM_PP_S<_spinModes, _T>>(std::forward<decltype(args)>(args)...);
		default:
			throw std::invalid_argument("Unknown NQS type");
		}
	};

	_NQS = createNQS(_H, this->nqsP.nqs_nh_, this->nqsP.nqs_lr_, this->threadNum, 1, _NQSl, _beta);

	// Set the hyperparameters
#ifdef NQS_USESR_MAT_USED
	_NQS->setPinv(this->nqsP.nqs_tr_pinv_);
#endif
	_NQS->setSolver(this->nqsP.nqs_tr_sol_, this->nqsP.nqs_tr_tol_, this->nqsP.nqs_tr_iter_, this->nqsP.nqs_tr_reg_);
	_NQS->setPreconditioner(this->nqsP.nqs_tr_prec_);
#ifdef NQS_USESR
	_NQS->setSregScheduler(this->nqsP.nqs_tr_regs_, this->nqsP.nqs_tr_reg_, this->nqsP.nqs_tr_regd_, this->nqsP.nqs_tr_epo_, this->nqsP.nqs_tr_regp_);
#endif
	_NQS->setScheduler(this->nqsP.nqs_sch_, this->nqsP.nqs_lr_, this->nqsP.nqs_lrd_, this->nqsP.nqs_tr_epo_, this->nqsP.nqs_lr_pat_);
	_NQS->setEarlyStopping(this->nqsP.nqs_es_pat_, this->nqsP.nqs_es_del_);
}

// ##########################################################################################################################################

// ######################################################### V A R I A T I O N A L ##########################################################
// 
// ##########################################################################################################################################

/**
* @brief Computes the excited states using Neural Quantum States (NQS).
* 
* This function performs the following steps:
* 1. Initializes the Hilbert space and Hamiltonian.
* 2. Defines the NQS states for the excited states.
* 3. Logs the start of the NQS Hamiltonian building process.
* 4. Retrieves and logs information about the NQS and model.
* 5. Sets up training parameters for NQS.
* 6. Calculates the exact diagonalization (ED) to compare with Lanczos or Full diagonalization methods.
* 7. Measures and logs the ground state and excited state energies using ED and Lanczos methods.
* 8. Sets up the energies container for NQS.
* 9. Trains the NQS for each excited state and collects the results.
* 10. Calculates and logs the mean energies for each state.
* 11. Saves the final results to HDF5 files.
* 12. Logs the true energies for each state.
* 
* @note This function assumes that the necessary libraries (e.g., Armadillo, Operators, NQSAv) are included and that the class members (e.g., nqsP, latP, mainDir, threadNum) are properly initialized.
*/
template<typename _T, uint _spinModes>
void UI::nqsExcited()
{
	const int stateNum 	= this->nqsP.nqs_ex_beta_.size() + 1;
	std::shared_ptr<Hamiltonian<_T, _spinModes>> _H;
	Hilbert::HilbertSpace<_T> _hilbert;
	this->defineModel<_T>(_hilbert, _H);
	
	// define the NQS states for the excited states
	arma::Col<_T> _meansNQS(this->nqsP.nqs_ex_beta_.size() + 1, arma::fill::zeros), _meansED(this->nqsP.nqs_ex_beta_.size() + 1, arma::fill::zeros),
					_meansLAN(this->nqsP.nqs_ex_beta_.size() + 1, arma::fill::zeros), _stdsNQS(this->nqsP.nqs_ex_beta_.size() + 1, arma::fill::zeros);
	
	v_1d<std::shared_ptr<NQS<_spinModes, _T>>> _NQS(this->nqsP.nqs_ex_beta_.size() + 1);	// define the NQS states
	this->defineNQS<_T, _spinModes>(_H, _NQS[0]);											// define the first one already here for the ground state
	
	{
		LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
		LOGINFO("Started building the NQS Hamiltonian", LOG_TYPES::TRACE, 1);								// Log the start of the NQS Hamiltonian building process
		LOGINFO("Using NQS type: " + SSTR(getSTR_NQSTYPES(this->nqsP.type_)), LOG_TYPES::TRACE, 2);			// Log the type of NQS being used
		_timer.reset();																						// Reset the timer to measure the duration of the NQS Hamiltonian building process						
	}
	
	std::string nqsInfo		= _NQS[0]->getInfo();
	std::string modelInfo	= _NQS[0]->getHamiltonianInfo();
	std::string dir			= makeDirsC(this->mainDir, this->latP.lat->get_info(), modelInfo, nqsInfo);
	
	MonteCarlo::MCS_train_t _parT(this->nqsP.nqs_tr_epo_, this->nqsP.nqs_tr_th_, this->nqsP.nqs_tr_mc_, this->nqsP.nqs_tr_bs_, this->nqsP.nFlips_, dir);		// Set up training parameters for NQS
	MonteCarlo::MCS_train_t _parC(this->nqsP.nqs_col_mc_, this->nqsP.nqs_col_th_, this->nqsP.nqs_col_bn_, this->nqsP.nqs_col_bs_, this->nqsP.nFlips_, dir);		// Set up collection parameters for NQS
	MonteCarlo::MCS_train_t _parE(this->nqsP.nqs_ex_mc_, this->nqsP.nqs_ex_th_, this->nqsP.nqs_ex_bn_, this->nqsP.nqs_ex_bs_, this->nqsP.nFlips_, dir);			// Set up parameters for the excited states

	u64 Nh					= _NQS[0]->getHilbertSize();									// get the size of the Hilbert space						
	auto Nvis 				= _NQS[0]->getNvis();											// get the number of visible units
	const bool fullED 		= Nh <= UI_LIMITS_NQS_FULLED;
	const bool lanED 		= (this->nqsP.nqs_ed_ && Nh <= ULLPOW(24));
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>> _opsG = {};							// set up the operators to save - global
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>> _opsL = {};						// set up the operators to save - local
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>> _opsC = {};				// set up the operators to save - correlation
	
	// for the time evolution
	const auto _timespace = this->nqsP.nqs_te_tlog_ > 0 ? arma::logspace(this->nqsP.nqs_te_dt_, this->nqsP.nqs_te_tf_, this->nqsP.nqs_te_tlog_) : arma::regspace(this->nqsP.nqs_te_dt_, this->nqsP.nqs_te_dt_, this->nqsP.nqs_te_tf_);
	saveAlgebraic(dir, "measurement.h5", _timespace, "time_evo/time", false);
	// operators
	{
		Operators::Operator<_T, uint> _SzL 			= Operators::SpinOperators::sig_z_l<_T>(Nvis);
		_opsL.push_back(std::make_shared<Operators::OperatorNQS<_T, uint>>(std::move(_SzL)));
		Operators::Operator<_T, uint> _SxL 			= Operators::SpinOperators::sig_x_l<_T>(Nvis);
		_opsL.push_back(std::make_shared<Operators::OperatorNQS<_T, uint>>(std::move(_SxL)));
		Operators::Operator<_T, uint, uint> _SzC 	= Operators::SpinOperators::sig_z_c<_T>(Nvis);
		_opsC.push_back(std::make_shared<Operators::OperatorNQS<_T, uint, uint>>(std::move(_SzC)));
		// special flux operator
		Operators::Operator<_T> _flux 				= Operators::SpinOperators::Flux::sig_f<_T>(Nvis, this->latP.lat->get_flux_sites(0, 0));
		_opsG.push_back(std::make_shared<Operators::OperatorNQS<_T>>(std::move(_flux)));
	}
	// ---------------
	v_1d<NQSAv::MeasurementNQS<_T> > _meas_ED, _meas_LAN, _meas_NQS;
	for (int i = 0; i < stateNum; ++i) 
	{
		_meas_ED.push_back(NQSAv::MeasurementNQS<_T>(this->latP.lat, dir, _opsG, _opsL, _opsC, this->threadNum));
		_meas_LAN.push_back(NQSAv::MeasurementNQS<_T>(this->latP.lat, dir, _opsG, _opsL, _opsC, this->threadNum));
		_meas_NQS.push_back(NQSAv::MeasurementNQS<_T>(this->latP.lat, dir, _opsG, _opsL, _opsC, this->threadNum));
	}

	if (lanED || fullED)
	{
		_H->buildHamiltonian();

		if (fullED)
		{																					// try with the full diagonalization
			_H->diagH(false);																// diagonalize the Hamiltonian
			for (int i = 0; i < stateNum; ++i) 												// save the energies
			{
				const arma::Col<_T>& _mbs = _H->getEigVec(i);
				if(Nh <= ULLPOW(7)) _H->prettyPrint(stout, _mbs, latP.lat->get_Ns(), 1e-3);	// measure the quantities
				_meas_ED[i].measure(_mbs, _hilbert);										// save the measured quantities
				_meansED(i) = _H->getEigVal(i);
				_meas_ED[i].saveMB({".h5"}, "measurement", "measurement", "measurement", "ED/" + STR(i), i > 0 || this->nqsP.nqs_te_);	
				LOGINFO("Found the ED (full) state(" + STR(i) + ") to be E=" + STRPS(_meansED[i], UI_NQS_PRECISION), LOG_TYPES::INFO, 2);

				if (this->nqsP.nqs_te_)
				{
					// Operators::Operator<cpx> _Szk 	= Operators::SpinOperators::sig_k<cpx>(Nvis, 0.0);
					// arma::SpMat<cpx> _SkMat 			= _Szk.template generateMat<false, cpx, arma::SpMat>(Nh);
					// arma::Col<cpx> _mbs1 			= _SkMat * _mbs;
					arma::SpMat<cpx> _Sz0Mat 			= Operators::SpinOperators::sig_z<cpx>(Nvis, 0).template generateMat<false, cpx, arma::SpMat>(Nh);
					arma::Col<double> _vals(_timespace.size(), arma::fill::zeros);
					_H->quenchHamiltonian();
					_H->buildHamiltonian(false);
					_H->diagH(false, false);			// diagonalize the Hamiltonian

					arma::Mat<_T> _eigv 				= _H->getEigVec();
					arma::Col<_T> _ovrl 				= _eigv.t() * _mbs;

#pragma omp parallel for num_threads(this->threadNum)
					for (int j = 0; j < _timespace.size(); ++j)
					{
						arma::Col<cpx> _mbsin 			= SystemProperties::TimeEvolution::time_evo(_eigv, _H->getEigVal(), _ovrl, _timespace(j));
						_vals(j) 						= algebra::cast<double>(Operators::applyOverlap(_mbsin, _Sz0Mat));
					}
					saveAlgebraic(dir, "measurement.h5", _vals, "ED/" + STR(i) + "/time_evo/Sz/0", true);
					_H->quenchHamiltonian();
					_H->buildHamiltonian(false);
					_H->diagH(false, false);			// diagonalize the Hamiltonian
				}
			}
			saveAlgebraic(dir, "history.h5", _meansED, "ED/energy", false); 				// save the results to HDF5 file
		}
		// ---------------
		_H->clearEigVal();
		_H->clearEigVec();
		// ---------------
		{
			_H->diagH(false, 50, 0, 1000, 1e-12, "lanczos");								// get LANCZOS diagonalization
			const auto& _eigvec 	= _H->getEigVec();										// get the eigenvectors in Krylov basis
			const auto& _krylov_mb 	= _H->getKrylov();										// get the Krylov basis
			for (int i = 0; i < stateNum; ++i) 												// save the energies
			{
				const arma::Col<_T> _mbs = LanczosMethod<_T>::trueState(_eigvec, _krylov_mb, i);
				_meas_LAN[i].measure(_mbs, _hilbert);										// save the measured quantities
				_meansLAN(i) = _H->getEigVal(i);											// save the energies to the container
				LOGINFO("Found the ED (Lanczos) state(" + STR(i) + ") to be E=" + STRPS(_meansLAN[i], UI_NQS_PRECISION), LOG_TYPES::INFO, 2);
				_meas_LAN[i].saveMB({".h5"}, "measurement", "measurement", "measurement", "LAN/" + STR(i), fullED || i > 0 || this->nqsP.nqs_te_);
			}
			saveAlgebraic(dir, "history.h5", _meansLAN, "Lanczos/energy", fullED);			// save the results to HDF5 file
		}
		// ---------------
		_H->clear();
		// ---------------
		LOGINFO("", LOG_TYPES::TRACE, 20, '#', 1);
		LOGINFO(2);
	}
	LOGINFO(nqsInfo, LOG_TYPES::TRACE, 2);
	LOGINFO(1);

	v_1d<std::shared_ptr<NQS<_spinModes, _T>>> _NQS_lower = {};								// define the NQS states for the excited states
	for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) 
	{
		_timer.checkpoint(VEQ(i));

		arma::Col<_T> _EN_TRAIN, _EN_TESTS, _EN_STD, _EN_TESTS_STD;							// set up the energies container for NQS

		if (!_NQS[i])
			this->defineNQS<_T, _spinModes>(_H, _NQS[i], _NQS_lower, { this->nqsP.nqs_ex_beta_.begin(), this->nqsP.nqs_ex_beta_.begin() + i });
		
		_NQS[i]->setTrainParExc(_parE);														// set the parameters in the excited states
		{
			std::tie(_EN_TRAIN, _EN_STD) = _NQS[i]->train(_parT, this->quiet, this->nqsP.nqs_tr_rst_, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);
			LOGINFO("", LOG_TYPES::TRACE, 20, '#', 1);
			LOGINFO(1);
			// -------------------------------------
			_NQS[i]->collect(_parC, _meas_NQS[i], &_EN_TESTS, &_EN_TESTS_STD, this->quiet, this->nqsP.nqs_col_rst_, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);			
			// -------------------------------------
			_meansNQS(i) 	= arma::mean(_EN_TESTS);
			_stdsNQS(i) 	= arma::stddev(_EN_TESTS);
			LOGINFOG("Found the NQS state(" + STR(i) + ") to be E=" + STRPS(_meansNQS(i), UI_NQS_PRECISION) + " +- " + STRPS(_stdsNQS(i) / 2.0, UI_NQS_PRECISION), LOG_TYPES::TRACE, 2);
			LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
			LOGINFO(4);
		}

		{
			auto _EN_r 		= algebra::cast<double>(_EN_TRAIN);
			auto _EN_rt 	= algebra::cast<double>(_EN_TESTS);
			auto _EN_std_r 	= algebra::cast<double>(_EN_STD);
			auto _EN_std_rt = algebra::cast<double>(_EN_TESTS_STD);

			// Save final results to HDF5
			saveAlgebraic(dir, "history.h5", _EN_r, "train/history/" + STR(i), lanED || fullED || i > 0);
			saveAlgebraic(dir, "history.h5", _EN_rt, "collect/history/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", _EN_std_r, "train/std/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", _EN_std_rt, "collect/std/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", _meansNQS, "NQS/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", arma::Col<double>(this->nqsP.nqs_ex_beta_), "betas", true);
			_NQS[i]->saveInfo(dir, "history.h5", i);
			_meas_NQS[i].saveNQS({".h5"}, "measurement", "measurement", "measurement", "NQS/" + STR(i), lanED || fullED || i > 0);
		}
		_NQS_lower.push_back(_NQS[i]);
	}
	
	for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) {
		LOGINFO("True energies: EED_" + STR(i) + " = " + STRP(_meansED[i], UI_NQS_PRECISION) + 
				", ELAN_" + STR(i) + " = " + STRP(_meansLAN[i], UI_NQS_PRECISION) + 
				", ENQS_" + STR(i) + " = " + STRP(_meansNQS[i], UI_NQS_PRECISION) + 
				" +- " + STRPS(_stdsNQS(i) / 2.0, UI_NQS_PRECISION), LOG_TYPES::TRACE, 2);
	}

	// try time evolution 
	if (this->nqsP.nqs_te_)
	{
		LOGINFO(3);
		MonteCarlo::MCS_train_t _parTime(this->nqsP.nqs_te_mc_, this->nqsP.nqs_te_th_, this->nqsP.nqs_te_bn_, this->nqsP.nqs_te_bs_, this->nqsP.nFlips_, dir); 
		_H->quenchHamiltonian();
		_parC.MC_sam_ = 1;

		for (int j = 0; j < this->nqsP.nqs_ex_beta_.size() + 1; ++j)
		{
			_timer.checkpoint("Time evolution for NQS state(" + STR(j) + ")");
			LOGINFO("Starting the time evolution for NQS state(" + STR(j) + ")", LOG_TYPES::TRACE, 1);
			arma::Col<_T> _sz0(_parC.MC_sam_ * _parC.nblck_), _En(_parTime.nblck_);		// set up the containers for the time evolution
			arma::Col<double> _sz0_mean(_timespace.size(), arma::fill::zeros);			// set up the container for the mean values
			Operators::OperatorNQS<_T> _Sz0 = Operators::SpinOperators::sig_z<_T>(Nvis, 0);
			
			// reset the NQS
			_NQS[j]->reset(_parTime.nblck_);											// reset the derivatives	
			_NQS[j]->template collect<arma::Col<_T>>(_parC, _Sz0, &_sz0);				// collect the data using ratio method - before the time evolution
			_sz0_mean(0) = algebra::cast<double>(arma::mean(_sz0));						// save the mean value
			for (int i = 0; i < _timespace.size() - 1; ++i)
			{
				double _dt = _timespace(i + 1) - _timespace(i);							// set the time step for the evolution
				_NQS[j]->evolveStep(i, _dt, _parTime, false, this->nqsP.nqs_te_rst_);	// evolve the system
				_NQS[j]->template collect<arma::Col<_T>>(_parC, _Sz0, &_sz0);			// collect the data using ratio method
				_sz0_mean(i + 1) = algebra::cast<double>(arma::mean(_sz0));				// save the mean value
				if (i % 10 == 0)
					LOGINFO("TE(" + STR(i + 1) + "/" + STR(_timespace.size()) + ") Time = " + STRPS(_timespace(i + 1), 3) + ", Sz_0 = " + STRPS(_sz0_mean(i + 1), 3) + ", dt = " + STRPS(_dt, 2), LOG_TYPES::TRACE, 2);
			}
			saveAlgebraic(dir, "measurement.h5", _sz0_mean, "NQS/" + STR(j) + "/time_evo/Sz/0", true);
		}
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template void UI::defineNQS<double, 2>(std::shared_ptr<Hamiltonian<double, 2>>& _H, std::shared_ptr<NQS<2, double>>& _NQS, 
		const v_1d<std::shared_ptr<NQS<2, double>>>& _NQSl, const v_1d<double>& _beta);
template void UI::defineNQS<cpx, 2>(std::shared_ptr<Hamiltonian<cpx, 2>>& _H, std::shared_ptr<NQS<2, cpx>>& _NQS,
		const v_1d<std::shared_ptr<NQS<2, cpx>>>& _NQSl, const v_1d<double>& _beta);	 	

// %%%%%%%%%%%%%%%%%%%%% DEFINE THE TEMPLATES %%%%%%%%%%%%%%%%%%%%%
template void UI::nqsExcited<double, 2>();
template void UI::nqsExcited<cpx, 2>();