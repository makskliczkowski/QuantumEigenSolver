#include "../../include/user_interface/user_interface.h"
#include "armadillo"
#include <memory>
#include <stdexcept>


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief A placeholder for making the simulation with NQS. It uses the complex Hamiltonian.
*/
void UI::makeSimNQS()
{
	this->useComplex_ = true;
	this->defineModels(true);
	this->defineNQS<cpx>(this->hamComplex, this->nqsCpx);
	this->nqsSingle(this->nqsCpx);

}

void UI::makeSimNQSExcited()
{
	this->useComplex_ = true;
	// this->defineModels(true);
	// this->defineNQS<cpx>(this->hamComplex, this->nqsCpx);
	this->defineLattice();
	this->nqsExcited<cpx, 2>();
}


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Based on a given type, it creates a NQS. Uses the model provided by the user to get the Hamiltonian.
* @param _H Specific Hamiltonian
* @param _NQS Neural Network Quantum State frameweork
* @param _NQSl List of NQS to be used for the state estimation (applies when we search for the excited states)
* @param _beta List of beta values to be used for the state estimation (applies when we search for the excited states)
*/
template<typename _T, uint _spinModes>
inline void UI::defineNQS(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<NQS<_spinModes, _T>>& _NQS, 
		const v_1d<std::shared_ptr<NQS<_spinModes, _T>>>& _NQSl, const v_1d<double>& _beta)
{
	// check what type of NQS to use and create it
	switch (this->nqsP.type_)
	{
	case NQSTYPES::RBM_T:
		_NQS = std::make_shared<RBM_S<_spinModes, _T>>(	_H,
														this->nqsP.nqs_nh_,
														this->nqsP.nqs_lr_,
														this->threadNum, 
														1,
														_NQSl,
														_beta);
		break;
	case NQSTYPES::RBMPP_T:
		_NQS = std::make_shared<RBM_PP_S<_spinModes, _T>>(_H,
														this->nqsP.nqs_nh_,
														this->nqsP.nqs_lr_,
														this->threadNum,
														1,
														_NQSl,
														_beta);
		break;
	default:
		throw std::invalid_argument("I don't know any other NQS types :<");
		break;
	}

	// set the hyperparameters
#ifdef NQS_USESR_MAT_USED
	_NQS->setPinv(this->nqsP.nqs_tr_pinv_);
#endif
	// regarding the solver
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

/*
* @brief Based on a given type, it creates a NQS. Uses the model provided by the user to get the Hamiltonian.
* @param _NQS Neural Network Quantum State frameweork
*/
template<typename _T, uint _spinModes>
void UI::nqsSingle(std::shared_ptr<NQS<_spinModes, _T>> _NQS)
{
	// _timer.reset();
	// LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	// LOGINFO("Started building the NQS Hamiltonian", LOG_TYPES::TRACE, 1);
	// LOGINFO("Using: " + SSTR(getSTR_NQSTYPES(this->nqsP.type_)), LOG_TYPES::TRACE, 2);
	
	// // get info
	// std::string nqsInfo		= _NQS->getInfo();
	// std::string modelInfo	= _NQS->getHamiltonianInfo();
	// std::string dir			= makeDirsC(this->mainDir, this->latP.lat->get_info(), modelInfo, nqsInfo);

	// // calculate ED to compare with Lanczos or Full
	// u64 Nh					= _NQS->getHilbertSize();

	// arma::Col<_T> _mbs;
	// if (Nh <= UI_LIMITS_NQS_ED)
	// {
	// 	auto _H = _NQS->getHamiltonian();
	// 	_H->buildHamiltonian();
	// 	if (Nh <= UI_LIMITS_NQS_FULLED)
	// 	{
	// 		_H->diagH(false);
	// 		_mbs = _H->getEigVec(0);
	// 		if(Nh < ULLPOW(7))
	// 			_H->prettyPrint(stout, _mbs, latP.lat->get_Ns(), 1e-3);
	// 		LOGINFO("Found the ED groundstate to be EED_0 = " + STRP(_NQS->getHamiltonianEigVal(0), 7), LOG_TYPES::TRACE, 2);
	// 	}
	// 	else
	// 	{
	// 		_H->diagH(false, 50, 0, 1000, 0, "lanczos");
	// 		LOGINFO("Found the ED groundstate to be EED_0 = " + STRP(_NQS->getHamiltonianEigVal(0), 7), LOG_TYPES::TRACE, 2);
	// 	}
	// }

	// // load the weights
	// if (!this->nqsP.loadNQS_.empty())
	// 	_NQS->setWeights(this->nqsP.loadNQS_, "weights.h5");
	
	// // set the operators to save
	// v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>> _opsG = {};
	// v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>> _opsL = {};
	// v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>> _opsC = {};
	// // go through the lattice sites
	// {
	// 	// auto _SzL = std::make_shared<Operators::OperatorNQS<_T, uint>>(Operators::sigmaZ_L<_T>(latP.lat), "sz_l");
	// 	// auto _SzC = std::make_shared<Operators::OperatorNQS<_T, uint, uint>>(Operators::sigmaZ_C<_T>(latP.lat), "sz_c");
	// 	// _opsL.push_back(_SzL);
	// 	// _opsC.push_back(_SzC);
	// }
	// // create measurement operator
	// NQSAv::MeasurementNQS<_T> _meas(this->latP.lat, dir,  
	// 								_opsG, 
	// 								_opsL, 
	// 								_opsC, this->threadNum);

	// // start the simulation
	// NQS_train_t _parT(this->nqsP.nqs_tr_epo_, this->nqsP.nqs_tr_th_, 
	// 				 this->nqsP.nqs_tr_mc_, this->nqsP.nqs_tr_bs_, 
	// 				 this->nqsP.nFlips_, dir);
	// NQS_train_t _parC(this->nqsP.nqs_col_mc_, this->nqsP.nqs_col_th_, 
	// 				 this->nqsP.nqs_col_bn_, this->nqsP.nqs_col_bs_, 
	// 				 this->nqsP.nFlips_, dir);
					 
	// arma::Col<_T> _EN(_parT.MC_sam_ + _parC.MC_sam_, arma::fill::zeros);
	// auto _out = _NQS->train(_parT, this->quiet, _timer.start(), 10);
	// _EN.subvec(0, _parT.MC_sam_ - 1) = std::get<0>(_out);
	// _EN.subvec(_parT.MC_sam_, _EN.size() - 1) = _NQS->collect(_parC, this->quiet, _timer.start(), _meas);

	// // save the energies
	// arma::Mat<double> _ENSM(_EN.size(), 2, arma::fill::zeros);
	// _ENSM.col(0)	= arma::real(_EN);
	// _ENSM.col(1)	= arma::imag(_EN);

	// // save energy
	// auto perc		= int(_parT.MC_sam_ / 20);
	// perc			= perc == 0 ? 1 : perc;
	// auto ENQS_0		= arma::mean(_ENSM.col(0).tail(perc));
	// LOGINFOG("Found the NQS groundstate to be ENQS_0 = " + STRP(ENQS_0, 7), LOG_TYPES::TRACE, 2);
	// _ENSM.save(dir + "history.dat", arma::raw_ascii);

	// // many body
	// if (_mbs.size() != 0)
	// 	_meas.measure(_mbs, _NQS->getHilbertSpace());
	// // save the measured quantities
	// _meas.save();
}


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<typename _T, uint _spinModes>
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
void UI::nqsExcited()
{
	const int prec 		= 6;															// calculate ED to compare with Lanczos or Full diagonalization methods
	const int stateNum 	= this->nqsP.nqs_ex_beta_.size() + 1;
	Hilbert::HilbertSpace<_T> _hilbert;
	std::shared_ptr<Hamiltonian<_T, _spinModes>> _H;
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
	
	NQS_train_t _parT(this->nqsP.nqs_tr_epo_, this->nqsP.nqs_tr_th_, this->nqsP.nqs_tr_mc_, this->nqsP.nqs_tr_bs_, this->nqsP.nFlips_, dir);		// Set up training parameters for NQS
	NQS_train_t _parC(this->nqsP.nqs_col_mc_, this->nqsP.nqs_col_th_, this->nqsP.nqs_col_bn_, this->nqsP.nqs_col_bs_, this->nqsP.nFlips_, dir);		// Set up collection parameters for NQS
	NQS_train_t _parE(this->nqsP.nqs_ex_mc_, this->nqsP.nqs_ex_th_, this->nqsP.nqs_ex_bn_, this->nqsP.nqs_ex_bs_, this->nqsP.nFlips_, dir);			// Set up parameters for the excited states

	u64 Nh					= _NQS[0]->getHilbertSize();									// get the size of the Hilbert space						
	auto Nvis 				= _NQS[0]->getNvis();											// get the number of visible units
	
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>> _opsG = {};							// set up the operators to save - global
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>> _opsL = {};						// set up the operators to save - local
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>> _opsC = {};				// set up the operators to save - correlation
	{
		Operators::Operator<_T, uint> _SzL = Operators::SpinOperators::sig_z_l<_T>(Nvis);
		_SzL.setNameS("Sz");
		_opsL.push_back(std::make_shared<Operators::OperatorNQS<_T, uint>>(std::move(_SzL)));
	}
	NQSAv::MeasurementNQS<_T> _measGS(this->latP.lat, dir, _opsG, _opsL, _opsC, this->threadNum), _measES = _measGS;
	NQSAv::MeasurementNQS<_T> _measLAN(this->latP.lat, dir, _opsG, _opsL, _opsC, this->threadNum), _measLANES = _measLAN;
	NQSAv::MeasurementNQS<_T> _meas(this->latP.lat, dir, {}, {}, {}, this->threadNum);

	if (this->nqsP.nqs_ed_ && Nh <= ULLPOW(24)) 
	{
		_H->buildHamiltonian();	
		arma::Col<_T> _mbsed, _exsed;														// build the Hamiltonian
		if (Nh <= UI_LIMITS_NQS_FULLED) 
		{																					// try with the full diagonalization
			_H->diagH(false);																// diagonalize the Hamiltonian
			_mbsed = _H->getEigVec(0);
			_exsed = _H->getEigVec(1);

			if(Nh <= ULLPOW(7)) {
				_H->prettyPrint(stout, _mbsed, latP.lat->get_Ns(), 1e-3);
				_H->prettyPrint(stout, _exsed, latP.lat->get_Ns(), 1e-3);
			}
			_measGS.measure(_mbsed, _hilbert);												// save the measured quantities
			_measES.measure(_exsed, _hilbert);

			for (int i = 0; i < stateNum; ++i) 												// save the energies
			{
				_meansED(i) = _H->getEigVal(i);
				LOGINFO("Found the ED (full) state(" + STR(i) + ") to be E=" + STRPS(_meansED[i], prec), LOG_TYPES::INFO, 2);
			}
		}
		// ---------------
		_H->clearEigVal();
		_H->clearEigVec();
		// ---------------
		{
			_H->diagH(false, 50, 0, 1000, 1e-12, "lanczos");								// get LANCZOS diagonalization
			const auto& _eigvec 	= _H->getEigVec();
			const auto& _krylov_mb 	= _H->getKrylov();
			auto _mbs 				= LanczosMethod<_T>::trueState(_eigvec, _krylov_mb, 0);
			auto _exs 				= LanczosMethod<_T>::trueState(_eigvec, _krylov_mb, 1);

			if(Nh <= ULLPOW(7)) {
				_H->prettyPrint(stout, _mbs, latP.lat->get_Ns(), 1e-3);
				_H->prettyPrint(stout, _exs, latP.lat->get_Ns(), 1e-3);
				// print the overlap
				if (!_mbsed.is_empty()) {
					LOGINFO("Overlap between the ED and Lanczos ground state: " + STRPS(arma::cdot(_mbsed, _mbs), prec), LOG_TYPES::INFO, 2);
					LOGINFO("Overlap between the ED and Lanczos excited state: " + STRPS(arma::cdot(_exsed, _exs), prec), LOG_TYPES::INFO, 2);
				}
			}
			_measLAN.measure(_mbs, _hilbert);
			_measLANES.measure(_exs, _hilbert);

			for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) 					// save the energies
			{
				_meansLAN(i) = _H->getEigVal(i);
				LOGINFO("Found the ED (Lanczos) state(" + STR(i) + ") to be E=" + STRPS(_meansLAN[i], prec), LOG_TYPES::INFO, 2);
			}
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
		
		// set the parameters in the excited states
		_NQS[i]->setTrainParExc(_parE);

		{
			std::tie(_EN_TRAIN, _EN_STD) = _NQS[i]->train(_parT, this->quiet, this->nqsP.nqs_tr_rst_, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);
			LOGINFO("", LOG_TYPES::TRACE, 20, '#', 1);
			
			if (i == 0)
				_NQS[i]->collect(_parC, _measGS, 	&_EN_TESTS,  &_EN_TESTS_STD, this->quiet, this->nqsP.nqs_col_rst_, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);
			else if (i == 1)
				_NQS[i]->collect(_parC, _measES, 	&_EN_TESTS,  &_EN_TESTS_STD, this->quiet, this->nqsP.nqs_col_rst_, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);
			else
				_NQS[i]->collect(_parC, _meas, 	&_EN_TESTS,  &_EN_TESTS_STD, this->quiet, this->nqsP.nqs_col_rst_, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);
			
			// ------------------
			_meansNQS(i) 	= arma::mean(_EN_TESTS);
			_stdsNQS(i) 	= arma::stddev(_EN_TESTS);
			LOGINFOG("Found the NQS state(" + STR(i) + ") to be E=" + STRPS(_meansNQS(i), prec), LOG_TYPES::TRACE, 2);
			LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
			LOGINFO(4);
		}

		// Save results
		{
			auto _EN_r 		= algebra::cast<double>(_EN_TRAIN);
			auto _EN_rt 	= algebra::cast<double>(_EN_TESTS);
			auto _EN_std_r 	= algebra::cast<double>(_EN_STD);
			auto _EN_std_rt = algebra::cast<double>(_EN_TESTS_STD);

			// Save final results to HDF5
			saveAlgebraic(dir, "history.h5", _EN_r, "train/history/" + STR(i), i != 0);
			saveAlgebraic(dir, "history.h5", _EN_rt, "collect/history/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", _EN_std_r, "train/std/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", _EN_std_rt, "collect/std/" + STR(i), true);

			// Save true energies and those from NQS
			if (Nh <= UI_LIMITS_NQS_FULLED)
				saveAlgebraic(dir, "history.h5", _meansED, "ED/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", _meansLAN, "EDL/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", _meansNQS, "NQS/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", arma::Col<double>(this->nqsP.nqs_ex_beta_), "betas", true);
		
			_NQS[i]->saveInfo(dir, "history.h5", i);

			if (i <= 1) {
				_measGS.save({".h5"}, "ED_0_G", "ED_0_L", "ED_0_C");
				_measES.save({".h5"}, "ED_1_G", "ED_1_L", "ED_1_C");
				_measLAN.save({".h5"}, "OP_0_G", "OP_0_L", "OP_0_C");
				_measLANES.save({".h5"}, "OP_1_G", "OP_1_L", "OP_1_C");
			}
		}
		_NQS_lower.push_back(_NQS[i]);
	}
	
	for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) {
		LOGINFO("True energies: EED_" + STR(i) + " = " + STRP(_meansED[i], prec) + 
				", ELAN_" + STR(i) + " = " + STRP(_meansLAN[i], prec) + 
				", ENQS_" + STR(i) + " = " + STRP(_meansNQS[i], prec) + 
				" +- " + STRPS(_stdsNQS(i) / 2.0, prec), LOG_TYPES::TRACE, 2);
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template void UI::defineNQS<double, 2>(std::shared_ptr<Hamiltonian<double, 2>>& _H, std::shared_ptr<NQS<2, double>>& _NQS, 
		const v_1d<std::shared_ptr<NQS<2, double>>>& _NQSl, const v_1d<double>& _beta);
template void UI::defineNQS<cpx, 2>(std::shared_ptr<Hamiltonian<cpx, 2>>& _H, std::shared_ptr<NQS<2, cpx>>& _NQS,
		const v_1d<std::shared_ptr<NQS<2, cpx>>>& _NQSl, const v_1d<double>& _beta);	 	

// %%%%%%%%%%%%%%%%%%%%% DEFINE THE TEMPLATES %%%%%%%%%%%%%%%%%%%%%
template void UI::nqsSingle<double, 2>(std::shared_ptr<NQS<2, double>> _NQS);
template void UI::nqsSingle<cpx, 2>(std::shared_ptr<NQS<2, cpx>> _NQS);

template void UI::nqsExcited<double, 2>();
template void UI::nqsExcited<cpx, 2>();