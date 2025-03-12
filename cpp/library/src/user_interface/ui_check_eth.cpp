#include "../../include/user_interface/user_interface.h"
#include "armadillo"
#include <memory>
#include <vector>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
* @brief Executes the ETH (Eigenstate Thermalization Hypothesis) simulation based on the chosen function.
*
* This function sets up the necessary models and determines whether to use complex numbers based on the model type.
* It then defines the models and executes the appropriate ETH-related function based on the value of `chosenFun`.
*
* The function handles the following cases for `chosenFun`:
* - 40: Placeholder for a potential ETH check function (currently commented out).
* - 42, 43: Executes `checkETH_statistics` with either real or complex Hamiltonian.
* - 45, 46: Executes `checkETH_time_evo` with either real or complex Hamiltonian.
*
* If the models cannot be defined successfully, the function returns early.
*
* @note The function uses the macro `RUN_CPX_REAL` to handle the execution of functions with either real or complex Hamiltonians.
*/
void UI::makeSimETH()
{
	// define the models - we don't need complex numbers (only for the RP model)
	if (this->modP.modTyp_ == MY_MODELS::RP_M)
		this->useComplex_ 	= !this->modP.rosenzweig_porter.rp_be_real_;
	else
		this->useComplex_ 	= false;

	// go complex if needed
	const bool _takeComplex = (this->isComplex_ || this->useComplex_);	

	// define the models depending on which to choose
	bool _isok 				= this->defineModels(false, false, false);

	if (!_isok)
		return;
	
	// go through the function choice
	switch (this->chosenFun)
	{
		case 40:
			// this->checkETH(this->hamDouble);
			break;

		case 42:
		case 43:
			RUN_CPX_REAL(_takeComplex, this->checkETH_statistics, this->hamDouble, this->hamComplex);
			break;

		case 45:
		case 46:
			RUN_CPX_REAL(_takeComplex, this->checkETH_time_evo, this->hamDouble, this->hamComplex);
			break;

		default:
			// Handle unexpected values of chosenFun, if necessary
			break;
	}
}

// -----------------------------------------------------------------------------------------------

void UI::makeSimETHSweep()
{
	// steps for alpha
	v_1d<double> _params;
	switch (this->modP.modTyp_)
	{
	case MY_MODELS::QSM_M:
		_params = this->modP.qsm.qsm_alpha_;
		break;
	case MY_MODELS::RP_M:
		_params = this->modP.rosenzweig_porter.rp_g_;
		break;
	case MY_MODELS::ULTRAMETRIC_M:
		_params = this->modP.ultrametric.um_alpha_;
		break;
	case MY_MODELS::POWER_LAW_RANDOM_BANDED_M:
		_params = this->modP.power_law_random_bandwidth.plrb_a_;
		break;
	default:
		_params = { 1.0 };
	}

	for (int _Ns = 0; _Ns < this->latP.Ntots_.size(); _Ns++)
	{
		// change the Ntot
		this->latP.Ntot_ = this->latP.Ntots_[_Ns];

		// go through the parameters
		for (int _pi = 0; _pi < _params.size(); _pi++)
		{
			const auto _param = _params[_pi];

			// set the alpha
			if (this->modP.modTyp_ == MY_MODELS::QSM_M)
			{
				this->modP.qsm.qsm_Ntot_ 	= this->latP.Ntot_;				
				this->modP.qsm.qsm_alpha_ 	= v_1d<double>(this->latP.Ntot_ - this->modP.qsm.qsm_N_, _param);
				this->modP.qsm.qsm_xi_ 		= v_1d<double>(this->latP.Ntot_ - this->modP.qsm.qsm_N_, this->modP.qsm.qsm_xi_[0]);
				this->modP.qsm.qsm_h_ 		= this->ran_.rvector<v_1d<double>>(this->latP.Ntot_ - this->modP.qsm.qsm_N_, 1.0, 0.5);
			}
			else if (this->modP.modTyp_ == MY_MODELS::RP_M)
				this->modP.rosenzweig_porter.rp_g_ 	= v_1d<double>(this->latP.Ntot_, _param);
			else if (this->modP.modTyp_ == MY_MODELS::ULTRAMETRIC_M)
				this->modP.ultrametric.um_alpha_ 	= v_1d<double>(this->latP.Ntot_ - this->modP.ultrametric.um_N_, _param);
			else if (this->modP.modTyp_ == MY_MODELS::POWER_LAW_RANDOM_BANDED_M)
				this->modP.power_law_random_bandwidth.plrb_a_[0] = _param;

			// define the models
			this->resetEd();

			// simulate
			this->makeSimETH();
			this->modP.modRanNIdx_++;
		}
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// ------------------------------------------ ETH ------------------------------------------------

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

constexpr static bool check_saving_size(u64 _Nh, uint _r)
{
	return ((_Nh > ULLPOW(14)) ||(BETWEEN((size_t)std::log2(_Nh), 10, 15) && (_r % 10 == 0)) || ((_Nh <= 1024) && (_r % 50 == 0)));
}

constexpr static bool check_multithread_operator(u64 _Nh)
{
	return (_Nh <= ULLPOW(9));
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
* @brief Based on the model type, obtain both quadratic and many body operators
* @param _Nh: Hilbert space size
* @param _isquadratic: is the model quadratic
* @param _ismanybody: is the model many body
*/
std::pair<v_1d<std::shared_ptr<Operators::Operator<double>>>, strVec> UI::ui_eth_getoperators(const size_t _Nh, bool _isquadratic, bool _ismanybody)
{
	const size_t _Ns = this->latP.Ntot_;

	// create operator parser
	Operators::OperatorNameParser _parser(_Ns, _Nh);
	auto _parsedOps = _parser.createGlobalOperators<double>(this->modP.operators, _ismanybody, _isquadratic, &this->ran_);

	return _parsedOps;
}

// ###############################################################################################

/**
* @brief Randomize the Hamiltonian to get another realization of the system. It allows for creating 
* different realizations of the same system. 
* @param _H: Hamiltonian to randomize
* @param _r: realization number
* @param _spinchanged: which spin to change (if applicable)
*/
template<typename _T>
void UI::ui_eth_randomize(std::shared_ptr<Hamiltonian<_T>> _H, int _r, uint _spinchanged)
{
	bool isQuadratic [[maybe_unused]]	= _H->getIsQuadratic(),
		 isManyBody	 [[maybe_unused]]	= _H->getIsManyBody();

	// clear the Hamiltonian
	_H->clearH();

	// randomize the Hamiltonian
	if (isManyBody)
	{
		if (this->modP.modTyp_ == MY_MODELS::QSM_M)
		{
			if (_spinchanged == 0)
				_H->randomize(this->modP.qsm.qsm_h_ra_, this->modP.qsm.qsm_h_r_, { "h" });
			else
			{
				auto _Ns	= this->latP.Ntot_;
				auto _N		= this->modP.qsm.qsm_N_;

				if (_r % 2 == 0 || (_spinchanged > _Ns - _N))
					_H->randomize(this->modP.qsm.qsm_h_ra_, this->modP.qsm.qsm_h_r_, { "h" });
				else
				{
					// set only the _spinchanged magnetic field to a different spin, which is the same as the one before but with a different sign
					// this is to check the ETH properties when one perturbs the system just slightly. Other magnetic fields are the same.
					std::shared_ptr<QSM<double>> _Hp = std::reinterpret_pointer_cast<QSM<double>>(_H);
					_Hp->setMagnetic(_Ns - 3 - _spinchanged, -_Hp->getMagnetic(_Ns - _N - _spinchanged));
				}
			}
		}
		else if (this->modP.modTyp_ == MY_MODELS::RP_M)
			_H->randomize(0, 1.0, { "g" });
		else if (this->modP.modTyp_ == MY_MODELS::ULTRAMETRIC_M)
			_H->randomize(0, 1.0, {});
		// else if (this->modP.modTyp_ == MY_MODELS::POWER_LAW_RANDOM_BANDED_M)
	}

	// -----------------------------------------------------------------------------

	// set the Hamiltonian
	_H->buildHamiltonian();
	_H->diagH(false);
}

// ###############################################################################################

/**
* @brief Check the ETH statistics for the matrix elements
* @param _startElem: start index
* @param _stopElem: stop index
* @param _Nh: Hilbert space size
* @param _statiter: atomic counter for statistics
* @param _H: Hamiltonian
* @param _overlaps: overlaps
* @param _histAv: histogram average
* @param _histAvTypical: histogram average typical
* @param _offdiagElemsOmega: off-diagonal elements omega
* @param _offdiagElemsOmegaLow: off-diagonal elements omega low
* @param _offdiagElems: off-diagonal elements
* @param _offdiagElemsLow: off-diagonal elements low
* @param _offdiagElemesStat: off-diagonal elements statistics
* @param _bandwidth: bandwidth
* @param _avEn: average energy
* @param _opi: operator index
* @param _r: realization index
* @param _th: thread number
*/
template<typename _T>
std::array<double, 6> UI::checkETH_statistics_mat_elems(
	u64 _startElem, u64 _stopElem, std::atomic<size_t>& _statiter,  int _th,
	u64 _Nh,
	Hamiltonian<_T>* _H,
	const arma::Mat<_T>& _overlaps, 
	HistogramAverage<double>& _histAv, HistogramAverage<double>& _histAvTypical,
	arma::Mat<double>* _offdiagElemsOmega, arma::Mat<double>* _offdiagElemsOmegaLow,
	VMAT<_T>* _offdiagElems, VMAT<_T>* _offdiagElemsLow,
	VMAT<double>* _offdiagElemesStat,
	const double _bandwidth,
	const double _energyAt,
	int _opi, int _r)
{
	// diagonal
	std::array<double, 6> _offdiagElemesStat_local = {0.0};

	// iterators
	size_t _totalIterator_off			= 0;
	size_t _totalIterator_off_low		= 0;
	const size_t _elemThreadedSize		= _stopElem - _startElem;
	const size_t _elemThreadedLowSize	= _stopElem - _startElem;

	// local Histograms		
	HistogramAverage<double> _histAvLocal(_histAv.edges());
	HistogramAverage<double> _histAvTypicalLocal(_histAvTypical.edges());

	// bandwidth limits
	double omega_upper_cut		= 2.0;
	double omega_lower_cut		= 1e-1;
	u64 _start					= _th >= 0 ? _th : 0;
	u64 _stop					= _Nh;
	u64 _iter					= _th >= 0 ? this->threadNum : 1;
	u64 _statiter_local			= 0;

	for (u64 i = _start; i < _stop; i += _iter)
	{
		const double _en_l = _H->getEigVal(i);

		for (u64 j = i + 1; j < _Nh; ++j)
		{
			const double _en_r = _H->getEigVal(j);

			// check the energy difference
			if (!SystemProperties::hs_fraction_close_mean(_en_l, _en_r, _energyAt, this->modP.modEnDiff_ * _bandwidth))
				continue;

			// calculate the frequency
			const double w			= std::abs(_en_l - _en_r);
			// const double w_ov_bw	= w * _bandwidth;
			const double w_ov_bw	= w / _bandwidth;

			// calculate the values
			const auto& _measured	= algebra::cast<double>(_overlaps(i, j));

			// save the off-diagonal elements
			
			{
				auto _elem			= _measured;
				auto _elemabs		= std::abs(_elem);
				auto _elemreal		= algebra::cast<double>(_elem);
				auto _elem2			= _elemabs * _elemabs;
				auto _logElem		= std::log(_elemabs);
				auto _logElem2		= std::log(_elemabs * _elemabs);

				if (_offdiagElemesStat)
				{
					// Accumulate statistics in thread-local storage
					// mean
					_offdiagElemesStat_local[0] += _elemreal;
					// typical
					_offdiagElemesStat_local[1] += _logElem;
					// mean2
					_offdiagElemesStat_local[2] += _elem2;
					// typical2
					_offdiagElemesStat_local[3] += _logElem2;
					// mean4
					_offdiagElemesStat_local[4] += _elem2 * _elem2;
					// meanabs
					_offdiagElemesStat_local[5] += _elemabs;
				}

				// add to the histograms
				_histAvLocal.append(w, _elem2);
				// _histAvLocal.append(w, _elem2);
				_histAvTypicalLocal.append(w, _logElem2);
				// _histAvTypicalLocal.append(w, _logElem2);

				if (_offdiagElems)
				{
					// save the values
					if (_totalIterator_off < _elemThreadedSize && w_ov_bw <= omega_upper_cut && w_ov_bw >= omega_lower_cut)
					{
						const auto _ii					= _startElem + _totalIterator_off;
						(*_offdiagElemsOmega)(_ii, _r)	= std::abs(w);
						(*_offdiagElems).set(_opi, _ii, _r, _elemreal);
						_totalIterator_off++;
					}
				}
				
				if (_offdiagElemsLow)
				{
					if (_totalIterator_off_low < _elemThreadedLowSize && w * _bandwidth < omega_lower_cut)
					{
						const auto _ii						= _startElem + _totalIterator_off_low;
						(*_offdiagElemsOmegaLow)(_ii, _r)	= std::abs(w);
						(*_offdiagElemsLow).set(_opi, _ii, _r, _elemreal);
						_totalIterator_off_low++;
					}
				}

				_statiter_local++;
			}
		}
	}

	// statistics and locks!
	{
		std::lock_guard<std::mutex> lock(this->mtx_);

		//for (int i = 0; i < 6; ++i)
			//_offdiagElemesStat.add(_opi, i, _r, _offdiagElemesStat_local[i]);

		// merge histograms
		_histAv.merge(_histAvLocal);
		_histAvTypical.merge(_histAvTypicalLocal);

		// statistics
		_statiter += _statiter_local;
	}
	return _offdiagElemesStat_local;
}

// -----------------------------------------------------------------------------------------------

/*
* @brief Check the properties of the models complying to ETH based on the Hamiltonian provided.
* It saves the gap ratios, level statistcs, energies, and the operators.
* @param _H: Hamiltonian to check
*/
template<typename _T>
void UI::checkETH_statistics(std::shared_ptr<Hamiltonian<_T>> _H)
{
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);
	LOGINFO("Number of realizations: " + VEQ(this->modP.getRanReal()), LOG_TYPES::INFO, 2);
	_timer.reset();

	// check the random field
	size_t _Ns				= this->latP.Ntot_;
	u64 _Nh					= _H->getHilbertSize();
	bool isQuadratic		= _H->getIsQuadratic(), 
		 isManyBody			= _H->getIsManyBody();

	// do both!, cause why the hell not
	isQuadratic				= true;
	isManyBody				= true;

	// get the operators
	v_1d<std::shared_ptr<Operators::Operator<double>>> _ops;
	strVec _opsN;
	std::tie(_ops, _opsN)	= this->ui_eth_getoperators(_Nh, isQuadratic, isManyBody);

	// get info about the model
	std::string modelInfo 	= "", dir = "ETH_MAT_STAT", randomStr = "", extension = ".h5";
	this->get_inf_dir_ext_r(_H, dir, modelInfo, randomStr, extension);

	// set the placeholder for the values to save (will save only the diagonal elements and other measures)
	arma::Mat<double> _en, _entroHalf, _entroRHalf, _entroFirst, _entroRFirst, _entroLast, _entroRLast, _schmidFirst, _schmidLast;
	if (this->modP.eth_entro_) {
		_en				= UI_DEF_MAT_D(_Nh, this->modP.getRanReal());								// energies
		_entroHalf		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);	// Renyi entropy q=1
		_entroRHalf		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);	// Renyi entropy q=2
		_entroFirst		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);	// Renyi entropy q=1
		_entroRFirst 	= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);	// Renyi entropy q=2
		_entroLast		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);	// Renyi entropy q=1
		_entroRLast		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);	// Renyi entropy q=2
		_schmidFirst	= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);	// schmid gap
		_schmidLast		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);	// schmid gap
	}

	// information entropy and ipr
	v_1d<double> _qs 	= { 0.1, 0.5, 1.0, 2.0 };
	arma::Mat<double> _e_ipr01, _e_ipr05, _e_ipr1, _e_ipr15, _e_ipr2, _e_ipr3;
	if (this->modP.eth_entro_) {
		_e_ipr01		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);
		_e_ipr05		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);
		_e_ipr1 		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);
		_e_ipr2 		= UI_DEF_MAT_D_COND(_Nh, this->modP.getRanReal(), this->modP.eth_entro_);
	}

	// gap ratios
	v_1d<double> _gapsin(_Nh - 2, 0.0);
	arma::Col<double> _gaps, _meanEn, _meanEnIdx, _meanlvl, _bandwidth, _H2;
	arma::Mat<double> _gapsall;
	{
		_gaps		= UI_DEF_COL_D(this->modP.getRanReal());
		_gapsall	= UI_DEF_MAT_D(_Nh - 2, this->modP.getRanReal());
		_meanEn		= UI_DEF_COL_D(this->modP.getRanReal());
		_meanEnIdx	= UI_DEF_COL_D(this->modP.getRanReal());
		_meanlvl	= UI_DEF_COL_D(this->modP.getRanReal());
		_bandwidth	= UI_DEF_COL_D(this->modP.getRanReal());
		_H2			= UI_DEF_COL_D(this->modP.getRanReal());
	}

	// create the measurem_bandwidthent class
	Measurement<double> _measure(this->latP.Ntot_, dir, _ops, _opsN, 1, _Nh);	
	VMAT<_T> _diagElems			= UI_DEF_VMAT(_T, _ops.size(), _Nh, this->modP.getRanReal());
	size_t _offdiagElemsSize	= this->threadNum * _Nh;
	// offdiagonals
	VMAT<_T> _offdiagElems, _offdiagElemsLow;
	arma::Mat<double> _offdiagElemsOmega, _offdiagElemsOmegaLow;
	if (this->modP.eth_offd_)
	{
		_offdiagElems			= UI_DEF_VMAT(_T, _ops.size(), _offdiagElemsSize, this->modP.getRanReal());
		_offdiagElemsLow		= UI_DEF_VMAT(_T, _ops.size(), _offdiagElemsSize, this->modP.getRanReal());
		_offdiagElemsOmega 		= UI_DEF_MAT_D(_offdiagElems.n_rows(0), this->modP.getRanReal()); 
		_offdiagElemsOmegaLow 	= UI_DEF_MAT_D(_offdiagElems.n_rows(0), this->modP.getRanReal()); 
	}

	// due to mobility edges, for the statistics we'll save two sets of data
	u64 _hs_fractions_diag_stat 	= SystemProperties::hs_fraction_diagonal_cut(0.1, _Nh);

	// (mean - 0, typical - 1, mean2 - 2, typical2 - 3, mean4 - 4, meanabs - 5, gaussianity - 6, binder cumulant - 7)
	VMAT<double> _offdiagElemesStat	= UI_DEF_VMAT(double, _ops.size(), 8, this->modP.getRanReal());

	// saves the histograms of the second moments for the offdiagonal elements -- those are the f-functions for the omega dependence
	v_1d<HistogramAverage<double>> _histAv(_ops.size(), HistogramAverage<double>(1));
	v_1d<HistogramAverage<double>> _histAvTypical(_ops.size(), HistogramAverage<double>(1));

	// histograms for other epsilons - for all 
	v_2d<HistogramAverage<double>> _histAvEps(std::max(this->modP.eth_end_.size(), (size_t)1), v_1d<HistogramAverage<double>>(_ops.size(), HistogramAverage<double>(1)));
	v_2d<HistogramAverage<double>> _histAvTypicalEps(std::max(this->modP.eth_end_.size(), (size_t)1), v_1d<HistogramAverage<double>>(_ops.size(), HistogramAverage<double>(1)));
	auto _fidelitySusceptibility 	= UI_DEF_VMAT_COND(_T, _ops.size(), _Nh, this->modP.getRanReal(), this->modP.eth_susc_);	// with regularization
	auto _fidelitySusceptibilityZ 	= UI_DEF_VMAT_COND(_T, _ops.size(), _Nh, this->modP.getRanReal(), this->modP.eth_susc_);	// without regularization
	
	// nbins operators 
	const size_t _nbinOperators 	= (size_t)(20 * std::log2(_Nh));
	v_1d<Histogram> _histOperatorsDiag(_ops.size(), Histogram(_nbinOperators));
	v_1d<Histogram> _histOperatorsOffdiag(_ops.size(), Histogram(_nbinOperators));

	// create the saving function
	std::function<void(uint)> _saver = [&](uint _r)
		{
			if (_r == 0)
				return;
			const bool _use_subview = _r < modP.getRanReal();

			auto saveAny = [&](const std::string& fname, const std::string& name, const auto& data, bool append = true) {
				saveAlgebraic(dir, fname + randomStr + extension, data, name, append);
			};

			auto saveStat = [&](const std::string& name, const auto& data, bool append = true) {
				saveAny("stat", name, data, append);
			};

			auto saveEntro = [&](const std::string& name, const auto& data, bool append = true) {
				saveAny("entro", name, data, append);
			};

			auto saveHist = [&](const std::string& name, const auto& data, bool append = true) {
				saveAny("hist", name, data, append);
			};

			auto saveDist = [&](const std::string& name, const auto& data, bool append = true) {
				saveAlgebraic(dir, "dist" + randomStr + extension, data, name, append);
			};

			saveStat("gap_ratio", _use_subview ? _gaps.subvec(0, _r) : _gaps, false);
			saveStat("gap_ratios", _use_subview ? _gapsall.rows(0, _r) : _gapsall);
			saveStat("mean_energy", _use_subview ? _meanEn.subvec(0, _r) : _meanEn);
			saveStat("mean_energy_index", _use_subview ? _meanEnIdx.subvec(0, _r) : _meanEnIdx);
			saveStat("mean_level_spacing", _use_subview ? _meanlvl.subvec(0, _r) : _meanlvl);
			saveStat("bandwidth", _use_subview ? _bandwidth.subvec(0, _r) : _bandwidth);
			saveStat("H2", _use_subview ? _H2.subvec(0, _r) : _H2);
			saveStat("energy", _use_subview ? _en.cols(0, _r) : _en);

			if (this->modP.eth_entro_)
			{
				saveEntro("vN/half", _use_subview ? _entroHalf.cols(0, _r) : _entroHalf, false);
				saveEntro("vN/first", _use_subview ? _entroFirst.cols(0, _r) : _entroFirst);
				saveEntro("vN/last", _use_subview ? _entroLast.cols(0, _r) : _entroLast);
				saveEntro("renyi/2.0/first", _use_subview ? _entroRFirst.cols(0, _r) : _entroRFirst);
				saveEntro("renyi/2.0/half", _use_subview ? _entroRHalf.cols(0, _r) : _entroRHalf);
				saveEntro("renyi/2.0/last", _use_subview ? _entroRLast.cols(0, _r) : _entroRLast);
				saveEntro("schmid/first", _use_subview ? _schmidFirst.cols(0, _r) : _schmidFirst);
				saveEntro("schmid/last", _use_subview ? _schmidLast.cols(0, _r) : _schmidLast);
			}

			if (this->modP.eth_susc_)
			{
				for (uint _opi = 0; _opi < _ops.size(); ++_opi)
				{
					auto _name = _measure.getOpGN(_opi);
					saveStat(_name + "/fidelity_susceptibility/mu", _use_subview ? _fidelitySusceptibility[_opi].cols(0, _r) : _fidelitySusceptibility[_opi]);
					saveStat(_name + "/fidelity_susceptibility/0", _use_subview ? _fidelitySusceptibilityZ[_opi].cols(0, _r) : _fidelitySusceptibilityZ[_opi]);
				}
			}

			if (this->modP.eth_ipr_)
			{
				saveAny("ipr", "info/0.1", _use_subview ? _e_ipr01.cols(0, _r) : _e_ipr01, false);
				saveAny("ipr", "info/0.5", _use_subview ? _e_ipr05.cols(0, _r) : _e_ipr05);
				saveAny("ipr", "info/1.0", _use_subview ? _e_ipr1.cols(0, _r) : _e_ipr1);
				saveAny("ipr", "info/2.0", _use_subview ? _e_ipr2.cols(0, _r) : _e_ipr2);
			}

			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				if (_use_subview)
					saveAny("diag", _measure.getOpGN(_opi), _diagElems[_opi].cols(0, _r), _opi > 0);
				else
					saveAny("diag", _measure.getOpGN(_opi), _diagElems[_opi], _opi > 0);
			}

			if (this->modP.eth_offd_)
			{
				for (uint _opi = 0; _opi < _ops.size(); ++_opi)
				{
					auto _name = _measure.getOpGN(_opi);
					saveStat(_name + "/offdiag", _use_subview ? (_offdiagElems[_opi].cols(0, _r)) : (_offdiagElems[_opi]), _opi > 0 || this->modP.eth_susc_);
					saveStat(_name + "/offdiag", _use_subview ? (_offdiagElemsLow[_opi].cols(0, _r)) : (_offdiagElemsLow[_opi]), _opi > 0 || this->modP.eth_susc_);
				}
				saveStat("omega", _use_subview ? _offdiagElemsOmega.cols(0, _r) : _offdiagElemsOmega);
				saveStat("omega", _use_subview ? _offdiagElemsOmegaLow.cols(0, _r) : _offdiagElemsOmegaLow);
			}

			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveStat("operators/" + _name + "/mean", _offdiagElemesStat[_opi].row(0));
				saveStat("operators/" + _name + "/typical", _offdiagElemesStat[_opi].row(1));
				saveStat("operators/" + _name + "/mean2", _offdiagElemesStat[_opi].row(2));
				saveStat("operators/" + _name + "/typical2", _offdiagElemesStat[_opi].row(3));
				saveStat("operators/" + _name + "/mean4", _offdiagElemesStat[_opi].row(4));
				saveStat("operators/" + _name + "/meanabs", _offdiagElemesStat[_opi].row(5));
				saveStat("operators/" + _name + "/gaussianity", _offdiagElemesStat[_opi].row(6));
				saveStat("operators/" + _name + "/binder_cumulant", _offdiagElemesStat[_opi].row(7));
			}

			saveHist("omegas", _histAv[0].edgesCol(), false);
			if (this->modP.eth_end_.size() > 0)
			{
				for (uint _epi = 0; _epi < _histAvEps.size(); ++_epi)
				{
					auto e = this->modP.eth_end_[_epi];
					saveHist(VEQP(e, 3) + "/omegas", _histAvEps[_epi][0].edgesCol());
				}
			}

			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveHist(_name + "_mean", _histAv[_opi].averages_av());
				saveHist(_name + "_typical", _histAvTypical[_opi].averages_av(true));

				if (_histAvEps.size() > 0)
				{
					for (uint _epi = 0; _epi < _histAvEps.size(); ++_epi)
					{
						auto e = this->modP.eth_end_[_epi];
						saveHist(VEQP(e, 3) + "/" + _name + "_mean", _histAvEps[_epi][_opi].averages_av());
						saveHist(VEQP(e, 3) + "/" + _name + "_typ", _histAvTypicalEps[_epi][_opi].averages_av(true));
					}
				}
			}

			if (_histAvEps.size() > 0)
			{
				for (uint _epi = 0; _epi < _histAvEps.size(); ++_epi)
				{
					auto e = this->modP.eth_end_[_epi];
					saveHist(VEQP(e, 3) + "/_counts", _histAvEps[_epi][0].edgesCol());
				}
			}
			saveHist("_counts", _histAv[0].countsCol());

			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				const auto _name = _measure.getOpGN(_opi);
				saveDist(_name + "_diag_edges", _histOperatorsDiag[_opi].edgesCol(), _opi > 0);
				saveDist(_name + "_offdiag_edges", _histOperatorsOffdiag[_opi].edgesCol());
				saveDist(_name + "_diag_counts", _histOperatorsDiag[_opi].countsCol());
				saveDist(_name + "_offdiag_counts", _histOperatorsOffdiag[_opi].countsCol());
			}

			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
		};

	// ---------------------------------------------------------------
	
	_timer.checkpoint("START");
	long _single_run_time 		= -1;
	long _remaining_time	= Slurm::get_remaining_time();
	LOGINFO("Remaining time: " + STR(_remaining_time) + " seconds", LOG_TYPES::INFO, 0);
	// ---------------------------------------------------------------------------------
	for (int _r = 0; _r < this->modP.getRanReal(); ++_r)
	{
		// ----------------------------------------------------------------------------
		
		// checkpoints etc
		{
			{
				LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
				LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
				_timer.checkpoint(STR(_r));
			}

			this->ui_eth_randomize(_H);
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
		}

		// -----------------------------------------------------------------------------
		
		// energy concerned
		{
			// gap ratios
			BEGIN_CATCH_HANDLER
			{
				const auto& _energies = _H->getEigVal();
				
				// get the average energy index and the points around it on the diagonal
				u64 _minIdxDiag_cut			= 0;
				u64 _maxIdxDiag_cut			= _Nh;

				// set
				std::tie(_minIdxDiag_cut, _maxIdxDiag_cut) = _H->getEnArndAvIdx(_hs_fractions_diag_stat / 2, _hs_fractions_diag_stat / 2);
				if(_minIdxDiag_cut == 0 || _maxIdxDiag_cut == 0)
					throw std::runtime_error("Energy indices are zero!");
				else if(_minIdxDiag_cut == _maxIdxDiag_cut)
					throw std::runtime_error("Energy indices are the same!");
				else if (_minIdxDiag_cut >= _maxIdxDiag_cut)
					throw std::runtime_error("Energy indices are wrong!");

				// -----------------------------------------------------------------------------

				const arma::Col<double> _energies_cut = _energies.subvec(_minIdxDiag_cut, _maxIdxDiag_cut - 1).as_col();
				_gaps(_r)			=	SystemProperties::eigenlevel_statistics(_energies_cut);							// calculate the eigenlevel statistics
				SystemProperties::eigenlevel_statistics(_energies.begin(), _energies.end(), _gapsin);
				_gapsall.col(_r)	=	arma::Col<double>(_gapsin);
				LOGINFO(StrParser::colorize(VEQ(_gaps(_r)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);

				// -----------------------------------------------------------------------------

				_en.col(_r) 	= _energies;
				_meanEn(_r) 	= _H->getEnAv();
				_meanEnIdx(_r) 	= _H->getEnAvIdx();

				// -----------------------------------------------------------------------------
				
				// ipr etc.
				{
					#pragma omp parallel for num_threads(this->threadNum)
					for(size_t _idx = 0; _idx < _Nh; ++_idx)
					{
						// get the entanglement
						const arma::Col<_T>& _st = _H->getEigVec(_idx);

						// get the entropies
						_e_ipr01(_idx, _r)	= SystemProperties::participation_entropy(_st, 0.1);
						_e_ipr05(_idx, _r)	= SystemProperties::participation_entropy(_st, 0.5);
						_e_ipr1(_idx, _r)	= SystemProperties::information_entropy(_st);
						_e_ipr2(_idx, _r)	= SystemProperties::participation_entropy(_st, 2.0);
					}
				}

				// -----------------------------------------------------------------------------
			}
			END_CATCH_HANDLER("Gap ratios and/or IPR failed:", break;)
			

			// -----------------------------------------------------------------------------

			// mean level spacing
			{
				_meanlvl(_r)	= _H->getMeanLevelSpacing();
				LOGINFO(StrParser::colorize(VEQP(_meanlvl(_r, 0), 10), StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
			}

			// bandwidth
			{
				_bandwidth(_r)	= _H->getBandwidth();
				_H2(_r)			= _H->getEnergyWidth();
				LOGINFO(StrParser::colorize(VEQ(_bandwidth(_r, 0)), StrParser::StrColors::blue), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_H2(_r, 0)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
			}

			// -----------------------------------------------------------------------------
		}

		// -----------------------------------------------------------------------------

		// set the uniform distribution of frequencies in logspace for the f-functions!!!
		if (_r == 0)
		{
			double _bwIn 		= _bandwidth(0);
			if (modP.modTyp_ == MY_MODELS::ULTRAMETRIC_M)
				_bwIn			= Ultrametric_types::UM_default::getBandwidth(std::reinterpret_pointer_cast<Ultrametric<_T>>(_H)->get_alpha(), (int)std::log2(_Nh));
			else if (modP.modTyp_ == MY_MODELS::QSM_M)
				_bwIn			= Ultrametric_types::UM_default::getBandwidth(std::reinterpret_pointer_cast<QSM<_T>>(_H)->get_alpha(), (int)std::log2(_Nh));
			else if (modP.modTyp_ == MY_MODELS::POWER_LAW_RANDOM_BANDED_M)
				_bwIn			= PRLB_types::PRLB_default::getBandwidth(std::reinterpret_pointer_cast<PowerLawRandomBanded<_T>>(_H)->get_a(), (int)std::log2(_Nh));
			else if (modP.modTyp_ == MY_MODELS::RP_M)
				_bwIn			= RP_types::RP_default::getBandwidth(std::reinterpret_pointer_cast<RosenzweigPorter<_T>>(_H)->get_gamma(), (int)std::log2(_Nh));
			
			LOGINFO("Setting bandwidth to: ", LOG_TYPES::TRACE, 2);
			LOGINFO(StrParser::colorize(VEQ(_bwIn), StrParser::StrColors::green), LOG_TYPES::TRACE, 3);
			LOGINFO(StrParser::colorize(VEQ((int)std::log2(_Nh)), StrParser::StrColors::green), LOG_TYPES::TRACE, 3);

			// values that are the limits when the 
			// double oMax			= 2.0 * _bandwidth(0);
			const double oMax	= _bwIn * 3.0;
			const double oMin	= 0.1 / (long double)_Nh;
			//double oMax			= std::abs(_H->getEigVal(_maxIdxDiag) - _H->getEigVal(_minIdxDiag)) * 2;
			//double oMin			= _Nh <= UI_LIMITS_MAXFULLED ? 1.0 / _Nh : 1e-3;

			// set the histograms
			for (auto iHist = 0; iHist < _ops.size(); ++iHist)
			{
				_histAv[iHist].reset(_nbinOperators);
				_histAv[iHist].uniformLog(oMax, oMin);

				_histAvTypical[iHist].reset(_nbinOperators);
				_histAvTypical[iHist].uniformLog(oMax, oMin);

				// set other
				if (!this->modP.eth_susc_)
					continue;

				for (auto iEps = 0; iEps < this->modP.eth_end_.size(); ++iEps)
				{
					_histAvEps[iEps][iHist].reset(_nbinOperators);
					_histAvEps[iEps][iHist].uniformLog(oMax, oMin);

					_histAvTypicalEps[iEps][iHist].reset(_nbinOperators);
					_histAvTypicalEps[iEps][iHist].uniformLog(oMax, oMin);
				}
			}
		}

		// -----------------------------------------------------------------------------
	
		// entanglement entropies
		if (this->modP.eth_entro_ && _Ns < 20) {
			v_1d<int> _sites   = { 0 };
			uint _lastSiteMask = Binary::prepareMask<int, v_1d<int>, false>(_sites, _Ns - 1);
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif				
			for(size_t _idx = 0; _idx < _Nh; ++_idx) 
			{
				const auto& _state			= _H->getEigVecCol(_idx);

				// half of the system
				{
					auto _rho_v 			= DensityMatrix::Values::redDensMat_v(_state, uint(_Ns / 2), uint(_Ns));
					// calculate the von Neumann entropy
					_entroHalf(_idx, _r) 	= Entropy::Entanglement::Bipartite::vonNeuman(_rho_v);
					_entroRHalf(_idx, _r) 	= Entropy::Entanglement::Bipartite::Renyi::renyi(_rho_v, 2.0);
				}

				// first site
				{
					auto _rho_v 			= DensityMatrix::Values::redDensMat_v(_state, 1, uint(_Ns));
					// calculate the von Neumann entropy
					_entroLast(_idx, _r) 	= Entropy::Entanglement::Bipartite::vonNeuman(_rho_v);
					_entroRLast(_idx, _r) 	= Entropy::Entanglement::Bipartite::Renyi::renyi(_rho_v, 2.0);

					// schmid gap
					_schmidLast(_idx, _r)	= _rho_v(0) - _rho_v(1);
				}

				// last site 
				{
					auto _rho_v 			= DensityMatrix::Values::redDensMat_v(_state, 
																				1, 
																				uint(_Ns),
																				_lastSiteMask,
																				Entropy::Entanglement::Bipartite::RHO_METHODS::SCHMIDT,
																				2);
					// calculate the von Neumann entropy
					_entroFirst(_idx, _r) 	= Entropy::Entanglement::Bipartite::vonNeuman(_rho_v);
					_entroRFirst(_idx, _r) 	= Entropy::Entanglement::Bipartite::Renyi::renyi(_rho_v, 2.0);

					// schmid gap
					_schmidFirst(_idx, _r)	= _rho_v(0) - _rho_v(1);
				}
			}
		}

		// -----------------------------------------------------------------------------

		// calculator of the properties
		{
			// all elements together
			BEGIN_CATCH_HANDLER
			{
				// get matrices
				const auto& _matrices	= _measure.getOpG_mat();
				const auto& _eigVec		= _H->getEigVec();
				const auto& _eigVal 	= _H->getEigVal();
				const double _avEn		= _H->getEnAv();
				const double _bw		= _bandwidth(_r);

				// go through the operators
#ifndef _DEBUG
#pragma omp parallel for num_threads(check_multithread_operator(_Nh) ? this->threadNum : 1)
#endif
				for (int _opi = 0; _opi < _matrices.size(); _opi++)
				{
					// LOGINFO("Doing operator: " + _opsN[_opi], LOG_TYPES::TRACE, 2);
					arma::Mat<_T> _overlaps = Operators::applyOverlapMat(_eigVec, _matrices[_opi]);
					std::atomic<size_t> _totalIteratorIn(0);

					// get histograms
					{
						v_1d<std::array<double, 6>> _out = Threading::createFutures<UI, std::array<double, 6>>(this, _totalIteratorIn, this->threadNum, 
																	!check_multithread_operator(_Nh) && this->threadNum != 1, 
																	_offdiagElemsSize, &UI::checkETH_statistics_mat_elems<_T>, 
																	_Nh, _H.get(),
																	std::ref(_overlaps), std::ref(_histAv[_opi]), std::ref(_histAvTypical[_opi]),
																	&_offdiagElemsOmega, &_offdiagElemsOmegaLow,
																	&_offdiagElems, &_offdiagElemsLow,
																	&_offdiagElemesStat,
																	_bw, _avEn, _opi, _r);
						for (const auto& _o : _out)
							for (int i = 0; i < 6; ++i)
								_offdiagElemesStat.add(_opi, i, _r, _o[i]);
					}

					// get histograms for the epsilons
					if (this->modP.eth_end_.size() > 0)
					{	
						for (uint _epi = 0; _epi < this->modP.eth_end_.size(); ++_epi)
						{
							std::atomic<size_t> _totalIteratorIn2(0);
							auto _energyIn = _eigVal(0) + this->modP.eth_end_[_epi] * _bw;
							v_1d<std::array<double, 6>> _out = Threading::createFutures<UI, std::array<double, 6>>(this, _totalIteratorIn2, this->threadNum, 
																	(!check_multithread_operator(_Nh) && this->threadNum != 1), 
																	_offdiagElemsSize, &UI::checkETH_statistics_mat_elems<_T>, 
																	_Nh, _H.get(),
																	std::ref(_overlaps), std::ref(_histAvEps[_epi][_opi]), std::ref(_histAvTypicalEps[_epi][_opi]),
																	nullptr, nullptr,
																	nullptr, nullptr,
																	nullptr,
																	_bw, _energyIn, _opi, _r);
						}
					}


					// -----------------------------------------------------------------

					// fidelity susceptability
					if (this->modP.eth_susc_)
					{
						auto _fidelitySusceptibilityIn = _fidelitySusceptibility[_opi].col(_r);
						SystemProperties::AGP::fidelity_susceptability_tot(_eigVal, _overlaps, _bw / (unsigned long)_Nh, _fidelitySusceptibilityIn);
						auto _fidelitySusceptibilityZIn = _fidelitySusceptibilityZ[_opi].col(_r);
						SystemProperties::AGP::fidelity_susceptability_tot(_eigVal, _overlaps, 0.0, _fidelitySusceptibilityZIn);
					}

					// ############## finalize statistics ##############

					// offdiagonal
					{
						for (uint ii = 0; ii < 6; ii++)
							_offdiagElemesStat.divide(_opi, ii, _r, (long double)_totalIteratorIn);

						// statistics
						_offdiagElemesStat.set(_opi, 6, _r, StatisticalMeasures::gaussianity(_offdiagElemesStat.get(_opi, 5, _r), _offdiagElemesStat.get(_opi, 2, _r)));
						_offdiagElemesStat.set(_opi, 7, _r, StatisticalMeasures::binder_cumulant(_offdiagElemesStat.get(_opi, 2, _r), _offdiagElemesStat.get(_opi, 4, _r)));

						// additionally, for typical values, calculate the exponential of the mean
						for (auto ii : { 1, 3 })
							_offdiagElemesStat.set(_opi, ii, _r, std::exp(_offdiagElemesStat.get(_opi, ii, _r)));
					}
					_diagElems[_opi].col(_r) = _overlaps.diag().as_col();

					// save the histograms of the diagonals and offdiagonals!
					_histOperatorsDiag[_opi].setHistogramCounts(_diagElems[_opi].col(_r), _r == 0);
					_histOperatorsOffdiag[_opi].setHistogramCounts(_offdiagElems[_opi].col(_r), _r == 0);
				}
			}
			END_CATCH_HANDLER("Operators failed:", break;)
		}

		// -----------------------------------------------------------------------------

		// get single runtime
		_single_run_time = _timer.elapsed<long>(STR(_r), Timer::TimePrecision::SECONDS);
		LOGINFO("Single run time: " + STR(_single_run_time) + " seconds", LOG_TYPES::TRACE, 1);

		// save the checkpoints
		if (check_saving_size(_Nh, _r) && symP.checkpoint_)
			_saver(_r);

		if (this->remainingSlurmTime(_r, &_timer, _single_run_time, _remaining_time))
			break;
			
		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '%', 1);
		// -----------------------------------------------------------------------------
	}
	_saver(this->modP.getRanReal());
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<typename _T>
void UI::checkETH_time_evo(std::shared_ptr<Hamiltonian<_T>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

	// check the random field
	size_t _Ns				= this->latP.Ntot_;
	u64 _Nh					= _H->getHilbertSize();
	bool isQuadratic		= _H->getIsQuadratic(), 
		isManyBody			= _H->getIsManyBody();

	// do both!, cause why the hell not
	isQuadratic				= true;
	isManyBody				= true;

	// get the operators
	v_1d<std::shared_ptr<Operators::Operator<double>>> _ops;
	strVec _opsN;
	std::tie(_ops, _opsN)	= this->ui_eth_getoperators(_Nh, isQuadratic, isManyBody);

	// get info
	std::string modelInfo, dir = "ETH_MAT_TIME_EVO", randomStr, extension;
	this->get_inf_dir_ext_r(_H, dir, modelInfo, randomStr, extension);

	// create the measurement class
	Measurement<double> _measure(this->latP.Ntot_, dir, _ops, _opsN, 1, _Nh);

	// set the placeholder for the values to save (will save only the diagonal elements and other measures)
	arma::Mat<double> _meanlvl 			= UI_DEF_MAT_D(4, this->modP.getRanReal());
	u64 _hs_fractions_diag				= SystemProperties::hs_fraction_diagonal_cut(0.5, _Nh);

	// time evolution saved here
	long double _heisenberg_time_est	= _Nh;
	const bool _uniform_time			= true;
	arma::Col<double> _timespace;
	if (!_uniform_time)
		_timespace = arma::logspace(-2, std::log10(_heisenberg_time_est * 1000), 5000);
	else
	{
		// Create a uniform distribution of times at long times (10*_heisenberg_time_est to 1000*_heisenberg_time_est)
		arma::Col<double> _long_times 	= arma::linspace(10 * _heisenberg_time_est, 1000 * _heisenberg_time_est, 1000);
		
		// Create uniform distributions for short and medium times
		arma::Col<double> _short_times 	= arma::linspace(1e-2, 1e-1, 1000);
		arma::Col<double> _medium_times = arma::linspace(1e-1, 1e0, 1000);
		
		// Create middle times from logspace (more points in the transition region)
		double _t_th 		= _heisenberg_time_est;
		if (modP.modTyp_ == MY_MODELS::ULTRAMETRIC_M)
			_t_th			= 1.0 / Ultrametric_types::UM_default::getThouless(std::reinterpret_pointer_cast<Ultrametric<_T>>(_H)->get_alpha(), (int)std::log2(_Nh));
		// else if (modP.modTyp_ == MY_MODELS::QSM_M)
			// _bwIn			= Ultrametric_types::UM_default::getBandwidth(std::reinterpret_pointer_cast<QSM<_T>>(_H)->get_alpha(), (int)std::log2(_Nh));
		// else if (modP.modTyp_ == MY_MODELS::POWER_LAW_RANDOM_BANDED_M)
			// _bwIn			= PRLB_types::PRLB_default::getBandwidth(std::reinterpret_pointer_cast<PowerLawRandomBanded<_T>>(_H)->get_a(), (int)std::log2(_Nh));
		// else if (modP.modTyp_ == MY_MODELS::RP_M)
			// _bwIn			= RP_types::RP_default::getBandwidth(std::reinterpret_pointer_cast<RosenzweigPorter<_T>>(_H)->get_gamma(), (int)std::log2(_Nh));
		
		arma::Col<double> _middle_times = arma::linspace(std::log10(_t_th * 0.5) , std::log10(5 * _t_th), 2000);
		
		// Combine all the time ranges
		_timespace.set_size(5000);
		_timespace.rows(0, 999) 		= _short_times;
		_timespace.rows(1000, 1999) 	= _medium_times;
		_timespace.rows(2000, 3999) 	= _middle_times;
		_timespace.rows(4000, 4999) 	= _long_times;
	}
	// create initial states for the quench
	arma::Col<_T> _initial_state_me		= arma::Col<_T>(_Nh, arma::fill::zeros);

	arma::Mat<double> _energies 		= UI_DEF_MAT_D(_Nh, this->modP.getRanReal());
	arma::Mat<double> _ldos_me 			= UI_DEF_MAT_D(_Nh, this->modP.getRanReal());
	// to save the energy densities (mean energy[system], mean energy[state], <state|H2|state>)
	arma::Mat<_T> _energydensitiesME(3, this->modP.getRanReal());
	VMAT<_T> _diagonals					= UI_DEF_VMAT(_T, _ops.size(), _Nh, this->modP.getRanReal());
	// save the time evolution here
	VMAT<_T> _timeEvolutionME			= UI_DEF_VMAT(_T, _ops.size(), _timespace.size(), this->modP.getRanReal());
	// VMAT<_T> _timeEvolutionME_AV, _timeEvolutionME_TP;		
	// size_t _timeEvolutionME_AV_num		= 20;

	// add the average over states
	// if (_Nh < ULLPOW(13)) {
	// 	_timeEvolutionME_AV				= UI_DEF_VMAT(_T, _ops.size(), _timespace.size(), this->modP.getRanReal());
	// 	_timeEvolutionME_TP				= UI_DEF_VMAT(_T, _ops.size(), _timespace.size(), this->modP.getRanReal());
	// }
	arma::Mat<double> _timePEntro		= UI_DEF_MAT_D(_timespace.size(), this->modP.getRanReal());
	// entropies to take
	v_1d<int> _entropiesSites			= {1, int(_Ns / 2), (int)_Ns};
	VMAT<double> _timeEntropyME			= UI_DEF_VMAT(double, _entropiesSites.size(), _timespace.size(), this->modP.getRanReal());
	arma::Mat<double> _timeEntropyBipartiteME(_timespace.size(), this->modP.getRanReal(), arma::fill::zeros);
	v_1d<arma::Col<_T>> _timeZeroME(_ops.size(), arma::Col<_T>(this->modP.getRanReal(), arma::fill::zeros));

	// ------------------------- MICROCANONICAL AVERAGES -------------------------

	const v_1d<double> _toCheckEps		= { 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4 };

	// to save the diagonal ensambles 
	arma::Mat<double> _diagonalME		= UI_DEF_MAT_D(_ops.size(), this->modP.getRanReal());

	// to save the microcanonical averages
	VMAT<_T> _microcanonicalME			= UI_DEF_VMAT(_T, _ops.size(), _toCheckEps.size(), this->modP.getRanReal());
	VMAT<double> _microcanonical2ME		= UI_DEF_VMAT(double, _ops.size(), _toCheckEps.size(), this->modP.getRanReal());

	// -------------------------------- SAVER ------------------------------------

	auto _microcanonical_saver	= [&](	uint _r,
										uint _opi, 
										VMAT<_T>& _microvals,
										VMAT<double>& _microvals2,
										arma::Mat<double>& _diagvals,
										const arma::Col<double>& _soverlaps, 
										const v_1d<u64>& _mins,
										const v_1d<u64>& _maxs)
		{
			// get the diagonal elements for a given realization and operator _opi
			const auto& _diagonal	= _diagonals[_opi].col(_r);
			// long time average (but for all states times the overlaps)
			_diagvals(_opi, _r)		= algebra::cast<double>(arma::dot(_diagonal, _soverlaps));

			uint _size				= _toCheckEps.size();
			u64 _Nh_minus_1			= _Nh - 1;
			for (int _ifrac = 0; _ifrac < _size; _ifrac++)
			{
				u64 _minin			= _mins[_ifrac];
				u64 _maxin			= _maxs[_ifrac];

				_minin				= std::max(_minin, 0ull);
				_maxin				= std::min(_maxin, _Nh_minus_1);

				if (_minin >= _maxin) 
				{
					_minin = _minin > 0 ? _minin - 1 : 0;
					_maxin = _maxin < _Nh_minus_1 ? _maxin + 1 : _Nh_minus_1;
				}

				// around the energy (\sum c_n^2 a_nn) (in the energy window - this is the diagonal ensemble within the energy window)
				//_diagvals[_opi](1 + _ifrac, _r)	= algebra::cast<double>(arma::dot(_diagonal.subvec(_minin, _maxin), _soverlaps.subvec(_minin, _maxin)));
				// around the energy, just the microcanonical average
				auto sub_diagonal					= _diagonal.subvec(_minin, _maxin);
				_microvals[_opi](_ifrac, _r)		= arma::mean(sub_diagonal);
				_microvals2[_opi](_ifrac, _r)		= arma::mean(arma::square(arma::abs(sub_diagonal)));
			}
		};

	// ----------------------------- EVOLVE STATE -------------------------------

	auto _evolveState = [&](uint _r, 
							const arma::Col<_T>& _initial_state,
							arma::Mat<double>* _ldoses,
							arma::Mat<_T>* _energydensities,
							VMAT<_T>* _microvals,
							VMAT<double>* _microvals2,
							arma::Mat<double>* _diagvals,
							VMAT<_T>* _timeEvolution,
							v_1d<arma::Col<_T>>* _timeZero,
							const v_1d<GeneralizedMatrix<double>>& _matrices,
							bool _entropyCalculate = true, 
							bool _append = false,
							bool _uselog = false,
							bool _autocor= false)
		{
			if (_timeEvolution == nullptr) {
				LOGINFO("Empty state evolution container... returning", LOG_TYPES::TRACE, 3);
				return;
			}

			// calculate the overlaps of the initial state with the eigenvectors 
			// (states are columns and vector is column as well, so we need to have the transpose)
			const auto& _eigvecs				= _H->getEigVec();
			const auto& _eigvals				= _H->getEigVal();
			const arma::Col<_T> _overlaps		= _eigvecs.t() * _initial_state;
			const arma::Col<double> _soverlaps	= arma::square(arma::abs(_overlaps));

			// calculate the average energy index
			double _meanE						= _H->getEnAv();

			// save the energies
			if (_ldoses)
				_ldoses->col(_r)				= SystemProperties::calculate_LDOS(_eigvals, _overlaps);
			if (_energydensities)
			{
				(*_energydensities)(0, _r)			= _meanE;
				// apply the Hamiltonian to the initial state
				const arma::Col<_T> _init_stat_H	= _H->getHamiltonian() * _initial_state;
				const auto _E						= arma::cdot(_initial_state, _init_stat_H);
				const auto _E2						= arma::cdot(_init_stat_H, _init_stat_H);
				u64 _Eidx							= _H->calcEnIdx(algebra::cast<double>(_E));
				LOGINFO(VEQ(_Eidx), LOG_TYPES::TRACE, 4);
				LOGINFO(VEQP(_E, 5), LOG_TYPES::TRACE, 4);
				LOGINFO(VEQP(_E2, 5), LOG_TYPES::TRACE, 4);
				(*_energydensities)(1, _r)			= _E;
				(*_energydensities)(2, _r)			= _E2;
				// calculate the bounds
				std::vector<u64> _mins(_toCheckEps.size()), _maxs(_toCheckEps.size());
				for (int _ifrac = 0; _ifrac < _toCheckEps.size(); _ifrac++)
					std::tie(_mins[_ifrac], _maxs[_ifrac]) = _H->getEnArndEnEps(_Eidx, _toCheckEps[_ifrac]);
				
				if (_microvals && _microvals2 && _diagvals)
				{
		#pragma omp parallel for num_threads(this->threadNum)
					for (int _opi = 0; _opi < _ops.size(); ++_opi)
					{
						_microcanonical_saver(_r, _opi, *_microvals, *_microvals2, *_diagvals, _soverlaps, _mins, _maxs);
					}
				}
			}

			// -----------------------------------------------------------------------------
			v_1d<_T> _zerovalues(_ops.size());
#pragma omp parallel for num_threads(this->threadNum)
			for (uint _opi = 0; _opi < _ops.size(); ++_opi) {
				_zerovalues[_opi] = arma::as_scalar(arma::cdot(_initial_state, (_matrices[_opi] * _initial_state)));
				if (_timeZero)
					(*_timeZero)[_opi](_r) = _zerovalues[_opi];
			}

			// evolution
#pragma omp parallel for num_threads(this->threadNum) schedule(dynamic)
			for (int _ti = 0; _ti < _timespace.size(); _ti++)
			{
				const auto _time					= _timespace(_ti);
				const arma::Col<cpx> _st			= SystemProperties::TimeEvolution::time_evo(_eigvecs, _eigvals, _overlaps, _time);

				// for each operator we can now apply the expectation value		
				for (uint _opi = 0; _opi < _ops.size(); ++_opi)
				{
					const cpx _rt					= arma::as_scalar(arma::cdot(_st, (_matrices[_opi] * _st))) * (_autocor ? _zerovalues[_opi] : 1.0);
					auto& _time_evo 				= (*_timeEvolution)[_opi](_ti, _r);
					if (_append) {
						_time_evo += _uselog ? std::log(std::abs(algebra::cast<_T>(_rt))) : algebra::cast<_T>(_rt);
					} else {
						_time_evo = _uselog ? std::log(std::abs(algebra::cast<_T>(_rt))) : algebra::cast<_T>(_rt);
					}
				}

				// say the time
				if (_ti % (std::max(1, static_cast<int>(_timespace.size() / 3))) == 0)
					LOGINFO(VEQ(_ti) + "/" + STR(_timespace.size()), LOG_TYPES::TRACE, 3);
				

				// calculate the entanglement entropy for each site
				if (this->modP.eth_entro_ && _entropyCalculate)
				{
					//for (int i = 1; i <= _Ns; i++)
					auto _iter = 0;
					for (const auto i: _entropiesSites)
					{
						// calculate the entanglement entropy
						uint _maskA							= 1 << (i - 1);
						_timeEntropyME[_iter++](_ti, _r)	= Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_st, 1, _Ns, _maskA, DensityMatrix::RHO_METHODS::SCHMIDT, 2);
					}
					if(_Nh <= UI_LIMITS_MAXFULLED / 4)
						_timeEntropyBipartiteME(_ti, _r)	= Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_st, int(_Ns / 2), _Ns, (ULLPOW((int(_Ns / 2)))) - 1);
					// calculate the participation entropy
					{
						_timePEntro(_ti, _r)				= SystemProperties::information_entropy(_st);
					}
				}
			}

		};


	// -------------------------------- SAVER ------------------------------------

	// create the saving function
	std::function<void(uint)> _saver = [&](uint _r)
		{
			// variance in th Hamiltonian
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(0).as_col()), "mean_lvl_spacing", false);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(1).as_col()), "heis_time_gamma", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(2).as_col()), "heis_time_around_mean", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(3).as_col()), "heis_time_around_mean_typ", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _energies, "energies", true);

			// save the ldos's
			saveAlgebraic(dir, "ldos" + randomStr + extension, _ldos_me, "ME", false);

			// save the energy densities
			saveAlgebraic(dir, "energydens" + randomStr + extension, arma::Col<_T>(_energydensitiesME.row(0).as_col()), "mean", false);
			saveAlgebraic(dir, "energydens" + randomStr + extension, arma::Col<_T>(_energydensitiesME.row(1).as_col()), "mean_state", true);
			saveAlgebraic(dir, "energydens" + randomStr + extension, arma::Col<_T>(_energydensitiesME.row(2).as_col()), "mean_state2", true);
			
			// save the matrices for time evolution
			saveAlgebraic(dir, "evo" + randomStr + extension, _timespace, "time", false);
			//for(int i = 0; i < _Ns; i++)
			for(int i = 0; i < _entropiesSites.size(); ++i)
				saveAlgebraic(dir, "evo" + randomStr + extension, _timeEntropyME[i], "entanglement_entropy/ME/" + STR((_entropiesSites[i])), true);
			saveAlgebraic(dir, "evo" + randomStr + extension, _timeEntropyBipartiteME, "entanglement_entropy/ME/bipartite", true);
			saveAlgebraic(dir, "evo" + randomStr + extension, _timePEntro, "participation_entropy/ME", true);

			// save the averages epsilon
			saveAlgebraic(dir, "avs" + randomStr + extension, arma::vec(_toCheckEps), "eps", false);
			
			// go through the operators
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);

				// diagonal
				saveAlgebraic(dir, "diag" + randomStr + extension, _diagonals[_opi], _name, _opi > 0);

				// evolution
				saveAlgebraic(dir, "evo" + randomStr + extension, _timeEvolutionME[_opi], _name + "/ME", true);

				// if (_Nh < ULLPOW(13)) {
				// 	saveAlgebraic(dir, "evo" + randomStr + extension, _timeEvolutionME_AV[_opi], _name + "/mean", true);
				// 	saveAlgebraic(dir, "evo" + randomStr + extension, _timeEvolutionME_TP[_opi], _name + "/typ", true);
				// }

				// at zero
				saveAlgebraic(dir, "evo" + randomStr + extension, _timeZeroME[_opi], _name + "/zero/ME", true);

				// diagonal ensemble
				saveAlgebraic(dir, "avs" + randomStr + extension, _microcanonicalME[_opi], _name + "/micro/ME", true);
				saveAlgebraic(dir, "avs" + randomStr + extension, _microcanonical2ME[_opi], _name + "/micro2/ME", true);

				// long time average
				saveAlgebraic(dir, "avs" + randomStr + extension, arma::vec(_diagonalME.row(_opi).as_col()), _name + "/diagonal/ME", true);
			}

			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
		};

	// -----------------------
	long _single_run_time 	= -1;
	long _remaining_time	= Slurm::get_remaining_time();
	LOGINFO("Remaining time: " + STR(_remaining_time) + " seconds", LOG_TYPES::INFO, 0);
	_timer.checkpoint("START");
	// -----------------------

	// go through realizations
	for (int _r = 0; _r < this->modP.getRanReal(); ++_r)
	{
		arma::Col<_T> _diagonal;
		
		// checkpoints etc
		{
			// -----------------------------------------------------------------------------
			LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
			LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
			_timer.checkpoint(STR(_r));

			this->ui_eth_randomize(_H, _r);
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
			// get the diagonal
			_diagonal 			= _H->getDiag();
			_initial_state_me 	= SystemProperties::TimeEvolution::create_initial_quench_state<_T>(SystemProperties::TimeEvolution::QuenchTypes::SEEK, _Nh, _Ns, _H->getEnAv(), _diagonal);
		}

		// -----------------------------------------------------------------------------

		// calculator of the properties
		{
			// mean level spacing
			{
				long double _h_freq	= 1.0 / _Nh;
				_energies.col(_r)	= _H->getEigVal();
				_meanlvl(0, _r)		= _H->getMeanLevelSpacing();
				_meanlvl(1, _r)		= SystemProperties::mean_lvl_heis_time(_meanlvl(0, _r), _Ns);

				// get the average energy index and the points around it on the diagonal
				const auto [_min, _max] = _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
				auto _E					= _H->getEigVal().subvec(_min, _max);
				_meanlvl(2, _r)			= 1.0 / SystemProperties::mean_lvl_spacing(_E);
				// get the Heisenberg frequency from the mean level spacing
				_h_freq					= SystemProperties::mean_lvl_spacing_typ(_E);
				_meanlvl(3, _r)			= 1.0 / _h_freq;
				LOGINFO(StrParser::colorize(VEQP(_meanlvl(0, _r), 10) + ": mean level spacing", StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(1, _r)) + ": mean level Heisenberg time", StrParser::StrColors::blue), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(2, _r)) + ": mean level Heisenberg time around energy " + VEQP(_min, 3), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(3, _r)) + ": mean level Heisenberg time around energy - typical", StrParser::StrColors::yellow), LOG_TYPES::TRACE, 1);
			}

			// -----------------------------------------------------------------------------

			// other measures
			{
				// calculate the diagonals
				const auto& _matrices				= _measure.getOpG_mat();
				const auto& _eigvec					= _H->getEigVec();

#pragma omp parallel for num_threads(_Nh < ULLPOW(14) ? this->threadNum : 2)
				for (int _opi = 0; _opi < _ops.size(); ++_opi)
				{
					_diagonals[_opi].col(_r) = Operators::applyOverlapMat(_eigvec, _matrices[_opi]).diag();
				}

				_timer.checkpoint(STR(_r) + ": time evolution");

				// evolve the states
				_evolveState(_r, _initial_state_me, &_ldos_me, &_energydensitiesME,  
					&_microcanonicalME, &_microcanonical2ME, &_diagonalME, &_timeEvolutionME, &_timeZeroME, _matrices, false, false, false);

				// if (_Nh < ULLPOW(13))
				// {
				// 	for (int ii = 0; ii < _timeEvolutionME_AV_num; ++ii)
				// 	{
				// 		LOGINFO(VEQ(ii), LOG_TYPES::TRACE, 2);
				// 		_initial_state_me = SystemProperties::TimeEvolution::create_initial_quench_state<_T>(SystemProperties::TimeEvolution::QuenchTypes::RANDP, _Nh, _Ns, _H->getEnAv(), _diagonal);
				// 		_evolveState(_r, _initial_state_me, nullptr, nullptr, nullptr, nullptr, nullptr, &_timeEvolutionME_AV, nullptr, _matrices, false, ii != 0, false, true);
				// 		_evolveState(_r, _initial_state_me, nullptr, nullptr, nullptr, nullptr, nullptr, &_timeEvolutionME_TP, nullptr, _matrices, false, ii != 0, true, true);
				// 	}
				// 	for (uint _opi = 0; _opi < _ops.size(); ++_opi)
				// 	{
				// 		_timeEvolutionME_AV[_opi].col(_r) /= _timeEvolutionME_AV_num;
				// 		_timeEvolutionME_TP[_opi].col(_r) =	arma::exp(_timeEvolutionME_TP[_opi].col(_r) / _timeEvolutionME_AV_num);
				// 	}
				// }
				LOGINFO(_timer.point(STR(_r) + ": time evolution"), "Time evolution: " + STR(_r), 3);
			}
		}
		// -----------------------------------------------------------------------------

		// get single runtime
		_single_run_time = _timer.elapsed<long>(STR(_r), Timer::TimePrecision::SECONDS);
		LOGINFO("Single run time: " + STR(_single_run_time) + " seconds", LOG_TYPES::TRACE, 1);

		// save the checkpoints
		if (check_saving_size(_Nh, _r) && symP.checkpoint_)
			_saver(_r);

		if (this->remainingSlurmTime(_r, &_timer, _single_run_time, _remaining_time))
			break;

		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '%', 1);
		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_saver(this->modP.getRanReal());

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%% DEFINE THE TEMPLATES %%%%%%%%%%%%%%%%%%%%%

template void UI::checkETH_statistics<double>(std::shared_ptr<Hamiltonian<double>> _H);
template void UI::checkETH_statistics<cpx>(std::shared_ptr<Hamiltonian<cpx>> _H);

template std::array<double, 6> UI::checkETH_statistics_mat_elems<double>(u64 _start, u64 _end,	std::atomic<size_t>& _statiter, int _th, u64 _Nh,
	Hamiltonian<double>* _H,
	const arma::Mat<double>& _overlaps,
	HistogramAverage<double>& _histAv,
	HistogramAverage<double>& _histAvTypical,
	arma::Mat<double>* _offdiagElemsOmega,
	arma::Mat<double>* _offdiagElemsOmegaLow,
	VMAT<double>* _offdiagElems,
	VMAT<double>* _offdiagElemsLow,
	VMAT<double>* _offdiagElemesStat,
	const double _bandwidth,
	const double _avEn,
	int _opi,
	int _r
	);
template std::array<double, 6> UI::checkETH_statistics_mat_elems<cpx>(u64 _start, u64 _end, std::atomic<size_t>& _statiter, int _th, u64 _Nh,
	Hamiltonian<cpx>* _H,
	const arma::Mat<cpx>& _overlaps,
	HistogramAverage<double>& _histAv,
	HistogramAverage<double>& _histAvTypical,
	arma::Mat<double>* _offdiagElemsOmega,
	arma::Mat<double>* _offdiagElemsOmegaLow,
	VMAT<cpx>* _offdiagElems,
	VMAT<cpx>* _offdiagElemsLow,
	VMAT<double>* _offdiagElemesStat,
	const double _bandwidth,
	const double _avEn,
	int _opi,
	int _r
	);

template void UI::checkETH_time_evo<double>(std::shared_ptr<Hamiltonian<double>> _H);
template void UI::checkETH_time_evo<cpx>(std::shared_ptr<Hamiltonian<cpx>> _H);