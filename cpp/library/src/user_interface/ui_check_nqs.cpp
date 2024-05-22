#include "../../include/user_interface/user_interface.h"

// ##########################################################################################################################################

// ######################################################### V A R I A T I O N A L ##########################################################
// 
// ##########################################################################################################################################

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
	std::string dir			=	makeDirsC(this->mainDir, this->latP.lat->get_info(), modelInfo, nqsInfo);

	// calculate ED to compare with Lanczos or Full
	u64 Nh						= _NQS->getHilbertSize();
	arma::Col<_T> _mbs;
	if (Nh <= UI_LIMITS_NQS_ED)
	{
		auto _H = _NQS->getHamiltonian();
		_H->buildHamiltonian();
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

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%% DEFINE THE TEMPLATES %%%%%%%%%%%%%%%%%%%%%
template void UI::nqsSingle<double, 2>(std::shared_ptr<NQS<2, double>> _NQS);
template void UI::nqsSingle<cpx, 2>(std::shared_ptr<NQS<2, cpx>> _NQS);