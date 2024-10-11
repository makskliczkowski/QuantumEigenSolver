#include "../../include/user_interface/user_interface.h"
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
*/
template<typename _T, uint _spinModes>
inline void UI::defineNQS(std::shared_ptr<Hamiltonian<_T>>& _H, std::shared_ptr<NQS<_spinModes, _T>>& _NQS)
{
	// check what type of NQS to use and create it
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
		throw std::invalid_argument("I don't know any other NQS types :<");
		break;
	}
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
	NQS_train_t _par = { this->nqsP.nMcSteps_, this->nqsP.nTherm_, this->nqsP.nBlocks_, this->nqsP.blockSize_, dir, this->nqsP.nFlips_ };
	arma::Col<_T> _EN(this->nqsP.nMcSteps_ + this->nqsP.nMcSamples_, arma::fill::zeros);
	_EN.subvec(0, this->nqsP.nMcSteps_ - 1) = _NQS->train(_par, this->quiet, _timer.start(), 10);

	_par = { this->nqsP.nMcSamples_, this->nqsP.nTherm_, this->nqsP.nSBlocks_, this->nqsP.blockSizeS_, dir, this->nqsP.nFlips_ };
	_EN.subvec(this->nqsP.nMcSteps_, _EN.size() - 1) = _NQS->collect(_par, this->quiet, _timer.start(), _meas);

	// save the energies
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


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<typename _T, uint _spinModes>
void UI::nqsExcited()
{
	Hilbert::HilbertSpace<_T> _hilbert 				= Hilbert::HilbertSpace<_T, _spinModes>(this->latP.lat);
	std::shared_ptr<Hamiltonian<_T, _spinModes>> _H = std::make_shared<IsingModel<_T>>(std::move(_hilbert), 
													this->modP.J1_, this->modP.hx_, this->modP.hz_);
	std::shared_ptr<NQS<_spinModes, _T>> _NQS_0;
	std::shared_ptr<NQS<_spinModes, _T>> _NQS_1;
	
	if (this->nqsP.type_ == NQSTYPES::RBM_T)
	{
		_NQS_0 = std::make_shared<RBM_S<_spinModes, _T>>(_H, this->nqsP.nHidden_, this->nqsP.lr_, this->threadNum);
	} else {
		_NQS_0 = std::make_shared<RBM_PP_S<_spinModes, _T>>(_H, this->nqsP.nHidden_, this->nqsP.lr_, this->threadNum);
	}

	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
	LOGINFO("Started building the NQS Hamiltonian", LOG_TYPES::TRACE, 1);
	LOGINFO("Using: " + SSTR(getSTR_NQSTYPES(this->nqsP.type_)), LOG_TYPES::TRACE, 2);
	
	// get info
	std::string nqsInfo		= _NQS_0->getInfo();
	std::string modelInfo	= _NQS_0->getHamiltonianInfo();
	std::string dir			= makeDirsC(this->mainDir, this->latP.lat->get_info(), modelInfo, nqsInfo);

	// calculate ED to compare with Lanczos or Full
	u64 Nh					= _NQS_0->getHilbertSize();

	arma::Col<_T> _mbs;
	if (Nh <= UI_LIMITS_NQS_ED)
	{
		_H->buildHamiltonian();
		if (Nh <= UI_LIMITS_NQS_FULLED)
		{
			_H->diagH(false);
			_mbs = _H->getEigVec(0);
			if(Nh < ULLPOW(7))
				_H->prettyPrint(stout, _mbs, latP.lat->get_Ns(), 1e-3);
		}
		else
		{
			_H->diagH(false, 50, 0, 1000, 0, "lanczos");
		}
		LOGINFO("Found the ED groundstate to be EED_0 = " + STRP(_H->getEigVal(0), 7), LOG_TYPES::TRACE, 2);
		LOGINFO("Found the ED groundstate to be EED_1 = " + STRP(_H->getEigVal(1), 7), LOG_TYPES::TRACE, 2);
		LOGINFO("Found the ED groundstate to be EED_2 = " + STRP(_H->getEigVal(2), 7), LOG_TYPES::TRACE, 2);
	}

	arma::Col<_T> _EN0(this->nqsP.nMcSteps_ + this->nqsP.nMcSamples_, arma::fill::zeros);
	arma::Col<_T> _EN1(this->nqsP.nMcSteps_ + this->nqsP.nMcSamples_, arma::fill::zeros);


	// set the operators to save
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T>>> _opsG = {};
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint>>> _opsL = {};
	v_1d<std::shared_ptr<Operators::OperatorNQS<_T, uint, uint>>> _opsC = {};
	NQSAv::MeasurementNQS<_T> _meas(this->latP.lat, dir,  
									_opsG, 
									_opsL, 
									_opsC, this->threadNum);
	// train the ground state
	NQS_train_t _par = { this->nqsP.nMcSteps_, this->nqsP.nTherm_, this->nqsP.nBlocks_, this->nqsP.blockSize_, dir, this->nqsP.nFlips_ };
	_EN0.subvec(0, this->nqsP.nMcSteps_ - 1) = _NQS_0->train(_par, this->quiet, _timer.start(), 25);
	_par = { this->nqsP.nMcSamples_, this->nqsP.nTherm_, this->nqsP.nSBlocks_, this->nqsP.blockSizeS_, dir, this->nqsP.nFlips_ };
	_EN0.subvec(this->nqsP.nMcSteps_, _EN0.size() - 1) = _NQS_0->collect(_par, this->quiet, _timer.start(), _meas, true);

	auto perc		= int(this->nqsP.nMcSamples_ / 20);
	perc			= perc == 0 ? 1 : perc;
	auto ENQS_0		= arma::mean(_EN0.col(0).tail(perc));
	LOGINFOG("Found the NQS groundstate to be ENQS_0 = " + STRP(ENQS_0, 7), LOG_TYPES::TRACE, 2);
	
	// create the excited state
	{
		v_1d<std::shared_ptr<NQS<_spinModes, _T>>> _NQS_0_p 	= { _NQS_0 };
		v_1d<double> _betas 									= { 5e-1 };
		if (this->nqsP.type_ == NQSTYPES::RBM_T)
		{
			_NQS_1 = std::make_shared<RBM_S<_spinModes, _T>>(_H, this->nqsP.nHidden_, this->nqsP.lr_, this->threadNum, 1, _NQS_0_p, _betas);
			// std::reinterpret_pointer_cast<RBM_S<_spinModes, _T>>(_NQS_1)->setWeights(std::reinterpret_pointer_cast<RBM_S<_spinModes, _T>>(_NQS_0));
		} else {
			_NQS_1 = std::make_shared<RBM_PP_S<_spinModes, _T>>(_H, this->nqsP.nHidden_, this->nqsP.lr_, this->threadNum, 1, _NQS_0_p, _betas);
		}

		// train the excited state
		_par = { this->nqsP.nMcSteps_, this->nqsP.nTherm_, this->nqsP.nBlocks_, this->nqsP.blockSize_, dir, this->nqsP.nFlips_ };
		_EN1.subvec(0, this->nqsP.nMcSteps_ - 1) = _NQS_1->train(_par, this->quiet, _timer.start(), 25);
		_par = { this->nqsP.nMcSamples_, this->nqsP.nTherm_, this->nqsP.nSBlocks_, this->nqsP.blockSizeS_, dir, this->nqsP.nFlips_ };
		_EN1.subvec(this->nqsP.nMcSteps_, _EN1.size() - 1) = _NQS_1->collect(_par, this->quiet, _timer.start(), _meas, true);
	}
	auto ENQS_1		= arma::mean(_EN1.col(0).tail(perc));
	LOGINFOG("Found the NQS excited state to be ENQS_1 = " + STRP(ENQS_1, 7), LOG_TYPES::TRACE, 2);

	// sumup true energies and those from NQS
	LOGINFO("True energies: EED_0 = " + STRP(_H->getEigVal(0), 7) + " EED_1 = " + STRP(_H->getEigVal(1), 7), LOG_TYPES::TRACE, 2);
	LOGINFO("NQS energies: ENQS_0 = " + STRP(ENQS_0, 7) + " ENQS_1 = " + STRP(ENQS_1, 7), LOG_TYPES::TRACE, 2);	
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template void UI::defineNQS<double, 2>(std::shared_ptr<Hamiltonian<double, 2>>& _H, std::shared_ptr<NQS<2, double>>& _NQS);
template void UI::defineNQS<cpx, 2>(std::shared_ptr<Hamiltonian<cpx, 2>>& _H, std::shared_ptr<NQS<2, cpx>>& _NQS);

// %%%%%%%%%%%%%%%%%%%%% DEFINE THE TEMPLATES %%%%%%%%%%%%%%%%%%%%%
template void UI::nqsSingle<double, 2>(std::shared_ptr<NQS<2, double>> _NQS);
template void UI::nqsSingle<cpx, 2>(std::shared_ptr<NQS<2, cpx>> _NQS);

template void UI::nqsExcited<double, 2>();
template void UI::nqsExcited<cpx, 2>();