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
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	LOGINFO("Started building the NQS Hamiltonian", LOG_TYPES::TRACE, 1);
	LOGINFO("Using: " + SSTR(getSTR_NQSTYPES(this->nqsP.type_)), LOG_TYPES::TRACE, 2);
	
	// get info
	std::string nqsInfo		= _NQS->getInfo();
	std::string modelInfo	= _NQS->getHamiltonianInfo();
	std::string dir			= makeDirsC(this->mainDir, this->latP.lat->get_info(), modelInfo, nqsInfo);

	// calculate ED to compare with Lanczos or Full
	u64 Nh					= _NQS->getHilbertSize();

	arma::Col<_T> _mbs;
	if (Nh <= UI_LIMITS_NQS_ED)
	{
		auto _H = _NQS->getHamiltonian();
		_H->buildHamiltonian();
		if (Nh <= UI_LIMITS_NQS_FULLED)
		{
			_H->diagH(false);
			_mbs = _H->getEigVec(0);
			if(Nh < ULLPOW(7))
				_H->prettyPrint(stout, _mbs, latP.lat->get_Ns(), 1e-3);
			LOGINFO("Found the ED groundstate to be EED_0 = " + STRP(_NQS->getHamiltonianEigVal(0), 7), LOG_TYPES::TRACE, 2);
		}
		else
		{
			_H->diagH(false, 50, 0, 1000, 0, "lanczos");
			LOGINFO("Found the ED groundstate to be EED_0 = " + STRP(_NQS->getHamiltonianEigVal(0), 7), LOG_TYPES::TRACE, 2);
		}
	}

	// load the weights
	if (!this->nqsP.loadNQS_.empty())
		_NQS->setWeights(this->nqsP.loadNQS_, "weights.h5");
	
	// set the operators to save
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>> _opsG = {};
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>> _opsL = {};
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>> _opsC = {};
	// go through the lattice sites
	{
		// auto _SzL = std::make_shared<Operators::OperatorNQS<_T, uint>>(Operators::sigmaZ_L<_T>(latP.lat), "sz_l");
		// auto _SzC = std::make_shared<Operators::OperatorNQS<_T, uint, uint>>(Operators::sigmaZ_C<_T>(latP.lat), "sz_c");
		// _opsL.push_back(_SzL);
		// _opsC.push_back(_SzC);
	}
	// create measurement operator
	NQSAv::MeasurementNQS<_T> _meas(this->latP.lat, dir,  
									_opsG, 
									_opsL, 
									_opsC, this->threadNum);

	// start the simulation
	NQS_train_t _parT(this->nqsP.nqs_tr_epo_, this->nqsP.nqs_tr_th_, 
					 this->nqsP.nqs_tr_mc_, this->nqsP.nqs_tr_bs_, 
					 this->nqsP.nFlips_, dir);
	NQS_train_t _parC(this->nqsP.nqs_col_mc_, this->nqsP.nqs_col_th_, 
					 this->nqsP.nqs_col_bn_, this->nqsP.nqs_col_bs_, 
					 this->nqsP.nFlips_, dir);
					 
	arma::Col<_T> _EN(_parT.MC_sam_ + _parC.MC_sam_, arma::fill::zeros);
	auto _out = _NQS->train(_parT, this->quiet, _timer.start(), 10);
	_EN.subvec(0, _parT.MC_sam_ - 1) = std::get<0>(_out);
	_EN.subvec(_parT.MC_sam_, _EN.size() - 1) = _NQS->collect(_parC, this->quiet, _timer.start(), _meas);

	// save the energies
	arma::Mat<double> _ENSM(_EN.size(), 2, arma::fill::zeros);
	_ENSM.col(0)	= arma::real(_EN);
	_ENSM.col(1)	= arma::imag(_EN);

	// save energy
	auto perc		= int(_parT.MC_sam_ / 20);
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


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<typename _T, uint _spinModes>
void UI::nqsExcited()
{
	Hilbert::HilbertSpace<_T> _hilbert;
	std::shared_ptr<Hamiltonian<_T, _spinModes>> _H;
	this->defineModel<_T>(_hilbert, _H);
	
	// define the NQS states for the excited states
	arma::Col<_T> _meansNQS(this->nqsP.nqs_ex_beta_.size() + 1, arma::fill::zeros), _meansED(this->nqsP.nqs_ex_beta_.size() + 1, arma::fill::zeros);
	v_1d<std::shared_ptr<NQS<_spinModes, _T>>> _NQS(this->nqsP.nqs_ex_beta_.size() + 1);
	// define the first one already here for the ground state
	this->defineNQS<_T, _spinModes>(_H, _NQS[0]);
	
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	LOGINFO("Started building the NQS Hamiltonian", LOG_TYPES::TRACE, 1);
	LOGINFO("Using: " + SSTR(getSTR_NQSTYPES(this->nqsP.type_)), LOG_TYPES::TRACE, 2);

	// get info
	std::string nqsInfo		= _NQS[0]->getInfo();
	std::string modelInfo	= _NQS[0]->getHamiltonianInfo();
	std::string dir			= makeDirsC(this->mainDir, this->latP.lat->get_info(), modelInfo, nqsInfo);
	
	NQS_train_t _parT(this->nqsP.nqs_tr_epo_, this->nqsP.nqs_tr_th_, 
					this->nqsP.nqs_tr_mc_, this->nqsP.nqs_tr_bs_, this->nqsP.nFlips_, dir);
	NQS_train_t _parC(this->nqsP.nqs_col_mc_, this->nqsP.nqs_col_th_, 
					this->nqsP.nqs_col_bn_, this->nqsP.nqs_col_bs_, this->nqsP.nFlips_, dir);
	NQS_train_t _parE(this->nqsP.nqs_ex_mc_, this->nqsP.nqs_ex_th_, 
					this->nqsP.nqs_ex_bn_, this->nqsP.nqs_ex_bs_, this->nqsP.nFlips_, dir);

	// calculate ED to compare with Lanczos or Full
	const int prec 		 	= 6;
	u64 Nh					= _NQS[0]->getHilbertSize();
	auto Nvis 				= _NQS[0]->getNvis();
	// set the operators to save
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>> _opsG = {};
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>> _opsL = {};
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>> _opsC = {};
	{
		Operators::Operator<_T, uint> _SzL = Operators::SpinOperators::sig_z_l<_T>(Nvis);
		_SzL.setNameS("Sz/L");
		_opsL.push_back(std::make_shared<Operators::OperatorNQS<_T, uint>>(std::move(_SzL)));
	}
	NQSAv::MeasurementNQS<_T> _measGS(this->latP.lat, dir,  
									_opsG, 
									_opsL, 
									_opsC, this->threadNum);
	NQSAv::MeasurementNQS<_T> _meas(this->latP.lat, dir,  
									{}, 
									{}, 
									{}, this->threadNum);

	if (this->nqsP.nqs_ed_) {
		_H->buildHamiltonian();

		// try with the full diagonalization
		if (Nh <= UI_LIMITS_NQS_FULLED) {
			_H->diagH(false);

			// get the ground state and find the measurements
			arma::Col<_T> _mbs = _H->getEigVec(0);
			if(Nh < ULLPOW(7))
				_H->prettyPrint(stout, _mbs, latP.lat->get_Ns(), 1e-3);
			
			// save the measured quantities
			_measGS.measure(_mbs, _hilbert);
		
			for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) {
				_meansED(i) = _H->getEigVal(i);
				LOGINFO("Found the ED (full) state(" + STR(i) + ") to be E=" + STRPS(_meansED[i], prec), LOG_TYPES::INFO, 2);
			}
		}
		// get LANCZOS
		{
			_H->diagH(false, 100, 0, 1000, 1e-10, "lanczos");
			
			for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) {
				_meansED(i) = _H->getEigVal(i);
				LOGINFO("Found the ED (Lanczos) state(" + STR(i) + ") to be E=" + STRPS(_meansED[i], prec), LOG_TYPES::INFO, 2);
			}
		}
		_H->clearEigVal();
		_H->clearEigVec();
		_H->clearH();
		_H->clearKrylov();
		LOGINFO("", LOG_TYPES::TRACE, 20, '#', 1);
		LOGINFO(2);
	}
	LOGINFO(nqsInfo, LOG_TYPES::TRACE, 2);
	LOGINFO(1);

	// setup the energies container

	v_1d<std::shared_ptr<NQS<_spinModes, _T>>> _NQS_lower = {};
	for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) {
		_timer.checkpoint(VEQ(i));

		arma::Col<_T> _EN_TRAIN, _EN_TESTS, _EN_std;

		if (!_NQS[i])
			this->defineNQS<_T, _spinModes>(_H, _NQS[i], _NQS_lower, { this->nqsP.nqs_ex_beta_.begin(), this->nqsP.nqs_ex_beta_.begin() + i });
		
		// set the parameters in the excited states
		_NQS[i]->setTrainParExc(_parE);

		// train
		{
			auto _out			= _NQS[i]->train(_parT, this->quiet, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);
			_EN_TRAIN 			= std::get<0>(_out);
			_EN_std 			= std::get<1>(_out);
			// _timer.checkpoint(VEQ(i) + "collect");
			LOGINFO("", LOG_TYPES::TRACE, 20, '#', 1);
			_EN_TESTS			= _NQS[i]->collect(_parC, this->quiet, _timer.point(VEQ(i)), (i == 0) ? _measGS : _meas, true, nqsP.nqs_tr_pc_);


			// calculate mean energies
			{
				const auto m_perc = int(_EN_TRAIN.n_rows / 20) == 0 ? _EN_TRAIN.n_rows : int(_EN_TRAIN.n_rows / 20);
				int _elemIter 	= 0;
				_meansNQS(i)	= 0.0;
				for (int k = _EN_TRAIN.n_elem - 1; k >= 0; --k) {
					if (!EQP(_EN_TRAIN(k), 0.0, 1e-6)) {
						_meansNQS(i) += _EN_TRAIN(k);
						_elemIter++;
					}
					if (_elemIter == m_perc)
						break;
				}
				_meansNQS(i) /= _elemIter;
			}

			LOGINFOG("Found the NQS state( " + STR(i) + ") to be E=" + STRPS(_meansNQS(i), prec), LOG_TYPES::TRACE, 2);
			LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
			LOGINFO(4);
		}

		// saver
		{
			auto _EN_r 		= algebra::cast<double>(_EN_TRAIN);
			auto _EN_rt 	= algebra::cast<double>(_EN_TESTS);
			auto _EN_std_r 	= algebra::cast<double>(_EN_std);

			// save final results to HDF5
			saveAlgebraic(dir, "history.h5", _EN_r, "train/" + STR(i), i != 0);
			saveAlgebraic(dir, "history.h5", _EN_rt, "test/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", _EN_std_r, "std/" + STR(i), true);

			// sumup true energies and those from NQS
			saveAlgebraic(dir, "history.h5", _meansED, "ED/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", _meansNQS, "NQS/" + STR(i), true);
			saveAlgebraic(dir, "history.h5", arma::Col<double>(this->nqsP.nqs_ex_beta_), "betas", true);

			// save info
			_NQS[i]->saveInfo(dir, "history.h5", i);

			if (i == 0) {
				// save the measured quantities
				_measGS.save();
			}
		}

		// push when the lower states are used for the excited states
		_NQS_lower.push_back(_NQS[i]);
	}
	
	for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) {
		LOGINFO("True energies: EED_" + STR(i) + " = " + STRP(_meansED[i], prec) + " ENQS_" + STR(i) + " = " + STRP(_meansNQS[i], prec), LOG_TYPES::TRACE, 2);
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