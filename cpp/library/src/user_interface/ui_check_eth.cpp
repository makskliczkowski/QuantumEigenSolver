#include "../../include/user_interface/user_interface.h"

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::makeSimETH()
{
	// define the models
	this->resetEd();
	// force complex Hamiltonian

	if(this->modP.modTyp_ == MY_MODELS::QSM_M)
		this->useComplex_ = false;
	else if(this->modP.modTyp_ == MY_MODELS::RP_M)
		this->useComplex_ = true;

	// simulate
	if (this->defineModels(false, false, false))
	{
		if (this->chosenFun == 40)
			this->checkETH(this->hamDouble);
		else if (this->chosenFun == 42)
			this->checkETH_statistics(this->hamDouble);
		else if (this->chosenFun == 44)
			this->checkETH_scaling_offdiag(this->hamDouble);
		else if (this->chosenFun == 45)
			this->checkETH_time_evo(this->hamDouble);
	}
}

// -----------------------------------------------------------------------------------------------

void UI::makeSimETHSweep()
{
	// steps for alpha
	v_1d<double> _params;
	if (this->modP.modTyp_ == MY_MODELS::QSM_M)
		_params = this->modP.qsm.qsm_alpha_;
	else if (this->modP.modTyp_ == MY_MODELS::RP_M)
		_params = this->modP.rosenzweig_porter.rp_g_;

	// get the random seed for this realization
	auto seed [[maybe_unused]]	= std::random_device()();

	for(int _pi = 0; _pi < _params.size(); _pi++)
	{
		const auto _param = _params[_pi];

		// set the alpha
		if (this->modP.modTyp_ == MY_MODELS::QSM_M)
		{
			for (int i = 0; i < this->modP.qsm.qsm_alpha_.size(); i++)
				this->modP.qsm.qsm_alpha_[i] = _param;
			this->useComplex_ = false;
		}
		else if (this->modP.modTyp_ == MY_MODELS::RP_M)
		{
			this->modP.rosenzweig_porter.rp_g_[0] = this->modP.rosenzweig_porter.rp_g_[_pi];
			this->useComplex_ = true;
		}

		// define the models
		this->resetEd();

		// simulate
		if (this->defineModels(false, false, false))
		{
			if (this->chosenFun == 41)
				this->checkETH(this->hamDouble);
			else if (this->chosenFun == 43)
				this->checkETH_statistics(this->hamDouble);
		}
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::checkETH_scaling_offdiag(std::shared_ptr<Hamiltonian<double>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

	// check the random field
	auto _rH	= this->modP.qsm.qsm_h_r_;
	auto _rA	= this->modP.qsm.qsm_h_ra_;
	size_t _Ns	= this->modP.qsm.qsm_Ntot_;
	u64 _Nh		= ULLPOW(_Ns);

	// get info
	std::string modelInfo	= _H->getInfo();
	std::string randomStr	= FileParser::appWRandom("", _H->ran_);
	std::string dir			= makeDirsC(this->mainDir, "QSM_MAT_ELEM_SCALING_OFFD", modelInfo, (_rH != 0) ? VEQV(dh, _rA) + "_" + STRP(_rH, 3) : "");
	std::string extension	= ".h5";
	v_1d<Operators::Operator<double>> _ops;
	strVec _opsN;

	// set seed
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	u64 _hs_fractions_diag			= SystemProperties::hs_fraction_diagonal_cut(this->modP.modMidStates_, _Nh);

	// set the placeholder for the values to save (will save only the diagonal elements and other measures)
	// GAMMA (see 2022, Suntajs, PRL)
	// gamma0, h_gamma, h_mean_lvl_mean, h_mean_lvl_typ, t_th_est
	arma::Mat<double> _meanlvl		= -1e5 * arma::Mat<double>(5, this->modP.modRanN_, arma::fill::ones);

	// create the operators
	{
		// couplings to calculate the Hybrydization condition!
		auto _sxc0						= Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, { 0, (uint)(_Ns - 1)});

		// couplings for the sz!
		auto _szc0						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { 0, (uint)(_Ns - 1)});

		// couplings outside the grain
		auto _szl						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { (uint) (_Ns - 1)});
		auto _sz3						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { 3});

		// create the matrices
		_ops							= { _sxc0, _szc0, _szl, _sz3 };
		_opsN							= { "sxc/0", "szc/0", "sz/l", "sz/3" };
	}

	// create the measurement class
	Measurement<double> _measure(this->modP.qsm.qsm_Ntot_, dir, _ops, _opsN);
	_measure.initializeMatrices(_Nh);

	// (mean, typical, mean2, typical2, mean4, meanabs, gaussianity, binder cumulant)
	size_t _offdiag_elem_num			= 1000;
	uint _nbinOperators					= 15 * _Ns;
	VMAT<double> _offdiagElemesStat_low(_ops.size(), 8, this->modP.modRanN_, arma::fill::zeros);
	VMAT<double> _offdiagElements_low(_ops.size(), this->modP.modRanN_, _offdiag_elem_num, arma::fill::ones, -1e5);
	VMAT<double> _diagonals(_ops.size(), _Nh, this->modP.modRanN_, arma::fill::zeros);

	// save the energies
	arma::Mat<double> _energies			= arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::zeros);

	// ----------------------- nbins operators -----------------------
	v_1d<Histogram> _histOperatorsOffdiag_low(_ops.size(), Histogram());

	// create the histograms for the operators
	for (uint _opi = 0; _opi < _ops.size(); ++_opi) 
	{
		// offdiagonal
		double _offdiagLimit = 0.5 - 0.025 * _Ns;
		_histOperatorsOffdiag_low[_opi].reset(_nbinOperators);
		_histOperatorsOffdiag_low[_opi].uniform(_offdiagLimit, -_offdiagLimit);	
	}

	// create the saving function
	std::function<void(uint)> _saver	= [&](uint _r)
		{
			// variance in th Hamiltonian
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(0).as_col()), "mean_level_gamma", false);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(1).as_col()), "heis_time_gamma", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(2).as_col()), "1_over_mean_level_spacing", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(3).as_col()), "1_over_mean_level_spacing_typ", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(4).as_col()), "th_freq", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _energies, "energies", true);

			// append statistics from the diagonal elements
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);

				// diagonal
				saveAlgebraic(dir, "diag" + randomStr + extension, _diagonals[_opi], _name, _opi > 0);

				// offdiagonal elements
				saveAlgebraic(dir, "offdiagval" + randomStr + extension, _offdiagElements_low[_opi], "offdiag/low/" + _name, true);

				// offdiagonal elements
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStat_low[_opi], "offdiag/low/" + _name, true);

				// save the means!
				if (_r == this->modP.modRanN_)
				{
					arma::Col<double> _meanOffdiag_low = arma::mean(_offdiagElemesStat_low[_opi], 1);
					saveAlgebraic(dir, "meanstat" + randomStr + extension, _meanOffdiag_low, "mean/offdiag/low/" + _name, true);
				}
			}

			// save the distributions of the operators - histograms for the values
			saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag_low[0].edgesCol(), "offdiag/edges/low/", true);
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag_low[_opi].countsCol(), _name + "/offdiag/counts/low", true);
			}

			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
		};
	double _alpha	= std::reinterpret_pointer_cast<QSM<double>>(_H)->get_alpha();
	double _g0		= std::reinterpret_pointer_cast<QSM<double>>(_H)->get_g0();

	// go through realizations
	for (int _r = 0; _r < this->modP.modRanN_; ++_r)
	{
		// ----------------------------------------------------------------------------
		
		// checkpoints etc
		{
			LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
			_timer.checkpoint(STR(_r));

			// -----------------------------------------------------------------------------

			LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
			_H->clearH();
			_H->randomize(this->modP.qsm.qsm_h_ra_, _rH, { "h" });

			// -----------------------------------------------------------------------------

			// set the Hamiltonian
			_H->buildHamiltonian();
			_H->diagH(false);
			_energies.col(_r) = _H->getEigVal();
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
		}

		// -----------------------------------------------------------------------------
				
		// get the average energy index and the points around it on the diagonal
		u64 _minIdxDiag, _maxIdxDiag = 0; 

		// set
		{
			std::tie(_minIdxDiag, _maxIdxDiag) = _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
		}

		// -----------------------------------------------------------------------------
		
		//double _th_freq		= QSM<double>::get_thouless_freq_est(_alpha, _Ns - this->modP.qsm.qsm_N_, _g0);
		long double _th_freq	= QSM<double>::get_thouless_freq_est(_alpha, _g0, _Ns);
		long double _h_freq		= 1.0 / _Nh;
		
		// calculator of the properties
		{
			// -----------------------------------------------------------------------------

			// mean level spacing
			{
				_meanlvl(0, _r)		= SystemProperties::mean_lvl_gamma(_H->getHamiltonian());
				_meanlvl(1, _r)		= SystemProperties::mean_lvl_heis_time(_meanlvl(0, _r), _Ns);

				// get the average energy index and the points around it on the diagonal
				u64 _hs_fractions_diagi = SystemProperties::hs_fraction_diagonal_cut(0.5, _Nh);
				const auto [_min, _max] = _H->getEnArndAvIdx(_hs_fractions_diagi / 2, _hs_fractions_diagi / 2);
				auto _E					= _H->getEigVal().subvec(_min, _max);
				_meanlvl(2, _r)			= 1.0 / SystemProperties::mean_lvl_spacing(_E);
				// get the Heisenberg frequency from the mean level spacing
				_h_freq					= SystemProperties::mean_lvl_spacing_typ(_E);
				_meanlvl(3, _r)			= 1.0 / _h_freq;
				_meanlvl(4, _r)			= _th_freq;
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(0, _r)), StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(1, _r)), StrParser::StrColors::blue), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(2, _r)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(3, _r)), StrParser::StrColors::yellow), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(4, _r)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
			}

			// -----------------------------------------------------------------------------

			// other measures
			{
				// get the matrices
				const auto& _matrices			= _measure.getOpG_mat();

				// -----------------------------------------------------------------------------

				double _lowbound				= std::sqrt(_h_freq * _th_freq);
				LOGINFO(StrParser::colorize(VEQ(_lowbound), StrParser::StrColors::green), LOG_TYPES::TRACE, 1);

				// offdiagonal
#ifndef _DEBUG
#pragma omp parallel for num_threads(_Ns > 14 ? 3 : this->threadNum)
#endif
				for(int _opi = 0; _opi < _matrices.size(); _opi++)
				{
					const auto& _mat			= _matrices[_opi];
					arma::Mat<double> _overlaps	= Operators::applyOverlapMat(_H->getEigVec(), _mat);
					_diagonals[_opi].col(_r)	= _overlaps.diag();

					// save the iterators
					u64 _totalIterator_l		= 0;
					const double _avEn			= _H->getEnAv();

					// go through the whole spectrum (do not save pairs, only one element as it's Hermitian.
					for (u64 i = 0; i < _Nh; ++i)
					{
						const auto _en_l		= _H->getEigVal(i);

						for (u64 j = i + 1; j < _Nh; ++j)
						{
							const auto _en_r	= _H->getEigVal(j);

							// check the energy difference
							if (!SystemProperties::hs_fraction_close_mean(_en_l, _en_r, _avEn, this->modP.modEnDiff_))
								continue;

							bool _isAroundLow		= SystemProperties::hs_fraction_diff_between(_en_l, _en_r, _lowbound / 5.0, _lowbound * 5.0);

							// check the frequency
							if(!_isAroundLow)
								continue;

							// calculate the frequency
							const double w			= std::abs(_en_l - _en_r);

							// calculate the values
							const auto& _measured	= _overlaps(i, j);

							// save the off-diagonal elements

							auto _elem				= _measured;
							auto _elem2				= _elem * _elem;
							auto _logElem			= std::log(std::abs(_elem));
							auto _logElem2			= std::log(_elem * _elem);

							if (_isAroundLow)
							{
								// mean
								_offdiagElemesStat_low[_opi](0, _r) += _elem;
								// typical
								_offdiagElemesStat_low[_opi](1, _r) += _logElem;
								// mean2
								_offdiagElemesStat_low[_opi](2, _r) += _elem2;
								// typical2
								_offdiagElemesStat_low[_opi](3, _r) += _logElem2;
								// mean4
								_offdiagElemesStat_low[_opi](4, _r) += _elem2 * _elem2;
								// meanabs
								_offdiagElemesStat_low[_opi](5, _r) += std::abs(_elem);

								// add to value histogram
								_histOperatorsOffdiag_low[_opi].append(_elem);

								// save the values
								if(_totalIterator_l < _offdiag_elem_num)
									_offdiagElements_low[_opi](_r, _totalIterator_l) = _elem;

								_totalIterator_l++;
							}
						}
					}

					// finalize statistics
					{
						for (uint ii = 0; ii < 6; ii++)
						{
							_offdiagElemesStat_low[_opi](ii, _r) /= (long double)_totalIterator_l;
						}
						// statistics
						_offdiagElemesStat_low[_opi](6, _r)		= StatisticalMeasures::gaussianity(_offdiagElemesStat_low[_opi](5, _r), _offdiagElemesStat_low[_opi](2, _r));
						_offdiagElemesStat_low[_opi](7, _r)		= StatisticalMeasures::binder_cumulant(_offdiagElemesStat_low[_opi](2, _r), _offdiagElemesStat_low[_opi](4, _r));
					}

					// additionally, for typical values, calculate the exponential of the mean
					{
						for (auto ii : { 1, 3 })
						{
							_offdiagElemesStat_low[_opi](ii, _r)	= std::exp(_offdiagElemesStat_low[_opi](ii, _r));
						}
					}
				}
			}
		}
		
		// save the checkpoints
		if ((_Ns >= 14 && (_r % 4 == 0)) || (_Ns < 14 && (_r % 25 == 0)))
		{
			// save the diagonals
			_saver(_r);
		}
		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_saver(this->modP.modRanN_);

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::checkETH_statistics(std::shared_ptr<Hamiltonian<double>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

	// check the random field
	auto _rH				= this->modP.qsm.qsm_h_r_;
	auto _rA				= this->modP.qsm.qsm_h_ra_;
	size_t _Ns				= this->modP.qsm.qsm_Ntot_;
	u64 _Nh					= ULLPOW(_Ns);
	v_1d<Operators::Operator<double>> _ops;
	strVec _opsN;

	// get info
	std::string modelInfo	= _H->getInfo();
	std::string randomStr	= FileParser::appWRandom("", _H->ran_);
	std::string dir			= makeDirsC(this->mainDir, "QSM_MAT_ELEM_STATISTICS_COR", modelInfo, (_rH != 0) ? VEQV(dh, _rA) + "_" + STRP(_rH, 3) : "");
	std::string extension	= ".h5";

	// set seed
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	// set the placeholder for the values to save (will save only the diagonal elements and other measures)
	arma::Mat<double> _en			= -1e5 * arma::Mat<double>(_H->getHilbertSize(), this->modP.modRanN_, arma::fill::zeros);
	arma::Col<double> _gaps			= -1e5 * arma::Col<double>(this->modP.modRanN_, arma::fill::ones);
	// GAMMA (see 2022, Suntajs, PRL)
	arma::Col<double> _meanlvl		= -1e5 * arma::Col<double>(this->modP.modRanN_, arma::fill::ones);

	// operators!
	{
		auto _szcom1					= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { (uint)_Ns - 2, (uint)_Ns - 1});
		auto _szcoh						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { (uint)(_Ns / 2), (uint)_Ns - 1});
		auto _szl						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { (uint)_Ns - 1});
		auto _sz3						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { 3 });
		auto _sxsx						= Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, { 0, (uint)_Ns - 1 });

		// create the matrices
		 _ops							= { _szcom1, _szcoh, _szl, _sz3, _sxsx };
		_opsN							= { "szco_lm1", "szco_l2_l", "sz_l", "sz_3", "sxc_l" };
	}

	// create the measurement class
	Measurement<double> _measure(this->modP.qsm.qsm_Ntot_, dir, _ops, _opsN);
	_measure.initializeMatrices(_Nh);

	// to save the operators (those elements will be stored for each operator separately)
	// a given matrix element <n|O|n> will be stored in i'th column of the i'th operator
	// the n'th row in the column will be the state index
	// the columns corresponds to realizations of disorder
	VMAT<double> _diagElems(_ops.size(), _Nh, this->modP.modRanN_, arma::fill::ones, -1e5);

	// (mean, typical, mean2, typical2, gaussianity, kurtosis, binder cumulant)
	// the columns will correspond to realizations
	u64 _hs_fractions_diag		= SystemProperties::hs_fraction_diagonal_cut(this->modP.modMidStates_, _Nh);
	// due to mobility edges, for the statistics we'll save two sets of data
	u64 _hs_fractions_diag_stat = SystemProperties::hs_fraction_diagonal_cut(0.1, _Nh);

	// mean, typical, mean2, typical2, gaussianity, kurtosis, binder cumulant
	VMAT<double> _diagElemsStat(_ops.size(), 7, this->modP.modRanN_, arma::fill::zeros);
	VMAT<double> _diagElemsStat_cut(_ops.size(), 7, this->modP.modRanN_, arma::fill::zeros);

	// (mean, typical, mean2, typical2, mean4, meanabs, gaussianity, binder cumulant)
	VMAT<double> _offdiagElemesStat(_ops.size(), 8, this->modP.modRanN_, arma::fill::zeros);

	// saves the histograms of the second moments for the offdiagonal elements -- those are the f-functions for the omega dependence
	v_1d<HistogramAverage<double>> _histAv(_ops.size(), HistogramAverage<double>());
	v_1d<HistogramAverage<double>> _histAvTypical(_ops.size(), HistogramAverage<double>());
	
	// ----------------------- nbins operators -----------------------
	v_1d<Histogram> _histOperatorsDiag(_ops.size(), Histogram());
	v_1d<Histogram> _histOperatorsOffdiag(_ops.size(), Histogram());
	uint _nbinOperators			= 15 * _Ns;

	// create the histograms for the operators
	for (uint _opi = 0; _opi < _ops.size(); ++_opi) 
	{
		// diagonal
		_histOperatorsDiag[_opi].reset(_nbinOperators);
		_histOperatorsDiag[_opi].uniform(0.5, -0.5); 

		// offdiagonal
		double _offdiagLimit	= 0.5 - 0.025 * _Ns;
		_histOperatorsOffdiag[_opi].reset(_nbinOperators);
		_histOperatorsOffdiag[_opi].uniform(_offdiagLimit, -_offdiagLimit);
	}

	// create the saving function
	std::function<void(uint)> _saver = [&](uint _r)
		{
			// save the iprs and the energy to the same file
			saveAlgebraic(dir, "stat" + randomStr + extension, _gaps, "gap_ratio", false);
			// variance in th Hamiltonian
			saveAlgebraic(dir, "stat" + randomStr + extension, _meanlvl, "mean_level_gamma", true);
			// energy
			saveAlgebraic(dir, "stat" + randomStr + extension, _en, "energy", true);

			// append statistics from the diagonal elements
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "stat" + randomStr + extension, _diagElemsStat[_opi], "diag_" + _name, true);
				saveAlgebraic(dir, "stat" + randomStr + extension, _diagElemsStat_cut[_opi], "diag_" + _name + "_0.1", true);

				// offdiagonal elements
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStat[_opi], "offdiag_" + _name, true);

				// save the means!
				if (_r == this->modP.modRanN_)
				{
					arma::Col<double> _meanDiag = arma::mean(_diagElemsStat[_opi], 1);
					saveAlgebraic(dir, "stat" + randomStr + extension, _meanDiag, "mean_diag_" + _name, true);
					arma::Col<double> _meanDiag_cut = arma::mean(_diagElemsStat_cut[_opi], 1);
					saveAlgebraic(dir, "stat" + randomStr + extension, _meanDiag_cut, "mean_diag_" + _name + "_0.1", true);
					arma::Col<double> _meanOffdiag = arma::mean(_offdiagElemesStat[_opi], 1);
					saveAlgebraic(dir, "stat" + randomStr + extension, _meanOffdiag, "mean_offdiag_" + _name, true);
				}
			}

			// diagonal operators saved (only append when _opi > 0)
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "diag" + randomStr + extension, _diagElems[_opi], _name, _opi > 0);
			}

			// save the histograms of the operators for the f functions
			saveAlgebraic(dir, "hist" + randomStr + extension, _histAv[0].edgesCol(), "omegas", false);
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "hist" + randomStr + extension, _histAv[_opi].averages_av(), _name + "_mean", true);
				saveAlgebraic(dir, "hist" + randomStr + extension, _histAvTypical[_opi].averages_av(true), _name + "_typical", true);
			}
			saveAlgebraic(dir, "hist" + randomStr + extension, _histAv[0].countsCol(), "_counts", true);

			// save the distributions of the operators - histograms for the values
			saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsDiag[0].edgesCol(), "diag_edges", false);
			saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag[0].edgesCol(), "offdiag_edges", true);
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsDiag[_opi].countsCol(), _name + "_diag_counts", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag[_opi].countsCol(), _name + "_offdiag_counts", true);
			}

			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
		};

	// go through realizations
	for (int _r = 0; _r < this->modP.modRanN_; ++_r)
	{
		// ----------------------------------------------------------------------------
		
		// checkpoints etc
		{
			LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
			_timer.checkpoint(STR(_r));

			// -----------------------------------------------------------------------------

			LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
			_H->clearH();
			_H->randomize(this->modP.qsm.qsm_h_ra_, _rH, { "h" });

			// -----------------------------------------------------------------------------

			// set the Hamiltonian
			_H->buildHamiltonian();
			_H->diagH(false);
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
		}

		// -----------------------------------------------------------------------------
				
		// get the average energy index and the points around it on the diagonal
		u64 _minIdxDiag, _maxIdxDiag, _minIdxDiag_cut, _maxIdxDiag_cut = 0; 

		// set
		{
			std::tie(_minIdxDiag, _maxIdxDiag)		= _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
			std::tie(_minIdxDiag_cut, _maxIdxDiag_cut) = _H->getEnArndAvIdx(_hs_fractions_diag_stat / 2, _hs_fractions_diag_stat / 2);
		}

		// -----------------------------------------------------------------------------
		
		// set the uniform distribution of frequencies in logspace for the f-functions!!!
		if (_r == 0)
		{

			double oMax			= std::abs(_H->getEigVal(_maxIdxDiag) - _H->getEigVal(_minIdxDiag)) * 2.5;
			double oMin			= 1.0l / _Nh / 10;
			u64 _nFrequencies	= 12 * _Ns;

			// set the histograms
			for (auto iHist = 0; iHist < _ops.size(); ++iHist)
			{
				_histAv[iHist].reset(_nFrequencies);
				_histAv[iHist].uniformLog(oMax, oMin);

				_histAvTypical[iHist].reset(_nFrequencies);
				_histAvTypical[iHist].uniformLog(oMax, oMin);
			}
		}

		// -----------------------------------------------------------------------------

		// save the energies
		{
			_en.col(_r) = _H->getEigVal();
		}

		// -----------------------------------------------------------------------------
		
		// calculator of the properties
		{
			// -----------------------------------------------------------------------------

			// gap ratios
			{
				// calculate the eigenlevel statistics
				_gaps(_r) = SystemProperties::eigenlevel_statistics(arma::Col<double>(_H->getEigVal().subvec(_minIdxDiag, _maxIdxDiag - 1)));
				LOGINFO(StrParser::colorize(VEQ(_gaps(_r, 0)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
				LOGINFO(_timer.point(STR(_r)), "Gap ratios", 1);
			}

			// -----------------------------------------------------------------------------

			// mean level spacing
			{
				_meanlvl(_r) = SystemProperties::mean_lvl_gamma(_H->getHamiltonian());
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(_r, 0)), StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
			}

			// -----------------------------------------------------------------------------

			// other measures
			{

				// -----------------------------------------------------------------------------

				// all elements together
				{
					// get matrices
					const auto& _matrices = _measure.getOpG_mat();

					// go through the operators
#ifndef _DEBUG
#pragma omp parallel for num_threads(4)
#endif
					for (int _opi = 0; _opi < _matrices.size(); _opi++)
					{
						const auto& _mat			= _matrices[_opi];
						arma::Mat<double> _overlaps = Operators::applyOverlapMat(_H->getEigVec(), _mat);

						// save the iterators
						u64 _totalIterator_off		= 0;
						const double _avEn			= _H->getEnAv();

						// go through the whole spectrum (do not save pairs, only one element as it's Hermitian.
						for (u64 i = 0; i < _Nh; ++i)
						{
							const auto _en_l = _H->getEigVal(i);

							// get diagonal statistics
							{
								const auto& _elem = _overlaps(i, i);

								if (i >= _minIdxDiag && i <= _maxIdxDiag)
								{
									// add element to the histogram
									_histOperatorsDiag[i].append(_elem);

									const auto _elem2		= _elem * _elem;
									const auto _logElem		= std::log(std::fabs(_elem));
									const auto _logElem2	= std::log(_elem * _elem);

									// save the statistics
									// mean
									_diagElemsStat[i](0, _r) += _elem;
									// typical
									_diagElemsStat[i](1, _r) += _logElem;
									// mean2
									_diagElemsStat[i](2, _r) += _elem2;
									// typical2
									_diagElemsStat[i](3, _r) += _logElem2;

									if (i >= _minIdxDiag_cut && i <= _maxIdxDiag_cut)
									{
										// mean
										_diagElemsStat_cut[i](0, _r) += _elem;
										// typical
										_diagElemsStat_cut[i](1, _r) += _logElem;
										// mean2
										_diagElemsStat_cut[i](2, _r) += _elem2;
										// typical2
										_diagElemsStat_cut[i](3, _r) += _logElem2;
									}
								}
							}

							for (u64 j = i + 1; j < _Nh; ++j)
							{
								const auto _en_r = _H->getEigVal(j);

								// check the energy difference
								if (!SystemProperties::hs_fraction_close_mean(_en_l, _en_r, _avEn, this->modP.modEnDiff_))
									continue;

								_totalIterator_off++;

								// calculate the frequency
								const double w			= std::abs(_en_l - _en_r);

								// calculate the values
								const auto& _measured	= _overlaps(i, j);

								// save the off-diagonal elements

								{
									auto _elem			= _measured;
									auto _elem2			= _elem * _elem;
									auto _logElem		= std::log(std::abs(_elem));
									auto _logElem2		= std::log(_elem * _elem);

									// mean
									_offdiagElemesStat[i](0, _r) += _elem;
									// typical
									_offdiagElemesStat[i](1, _r) += _logElem;
									// mean2
									_offdiagElemesStat[i](2, _r) += _elem2;
									// typical2
									_offdiagElemesStat[i](3, _r) += _logElem2;
									// mean4
									_offdiagElemesStat[i](4, _r) += _elem2 * _elem2;
									// meanabs
									_offdiagElemesStat[i](5, _r) += std::abs(_elem);

									// add to value histogram
									_histOperatorsOffdiag[i].append(_elem);

									// add to the histograms
									_histAv[i].append(w, _elem2);
									_histAvTypical[i].append(w, _logElem2);
								}
							}
						}

						// ############## finalize statistics ##############

						// diagonal
						{
							{
								for (uint ii = 0; ii < 4; ii++)
								{
									_diagElemsStat[_opi](ii, _r)		/= _hs_fractions_diag;
									_diagElemsStat_cut[_opi](ii, _r)	/= _hs_fractions_diag_stat;
								}
								// save the statistics
								_diagElemsStat[_opi](4, _r) = StatisticalMeasures::gaussianity(_diagElems[_opi].col(_r));
								_diagElemsStat[_opi](5, _r) = StatisticalMeasures::kurtosis(_diagElems[_opi].col(_r));
								_diagElemsStat[_opi](6, _r) = StatisticalMeasures::binder_cumulant(_diagElems[_opi].col(_r));
								// save the statistics
								_diagElemsStat_cut[_opi](4, _r) = StatisticalMeasures::gaussianity(_diagElemsStat_cut[_opi].col(_r));
								_diagElemsStat_cut[_opi](5, _r) = StatisticalMeasures::kurtosis(_diagElemsStat_cut[_opi].col(_r));
								_diagElemsStat_cut[_opi](6, _r) = StatisticalMeasures::binder_cumulant(_diagElemsStat_cut[_opi].col(_r));
							}

							// additionally, for typical values, calculate the exponential of the mean
							{
								_diagElemsStat[_opi](1, _r)		= std::exp(_diagElemsStat[_opi](1, _r));
								_diagElemsStat[_opi](3, _r)		= std::exp(_diagElemsStat[_opi](3, _r));
								_diagElemsStat_cut[_opi](1, _r) = std::exp(_diagElemsStat_cut[_opi](1, _r));
								_diagElemsStat_cut[_opi](3, _r) = std::exp(_diagElemsStat_cut[_opi](3, _r));
							}
						}

						// offdiagonal
						{

							{
								for (uint ii = 0; ii < 6; ii++)
									_offdiagElemesStat[_opi](ii, _r) /= (long double)_totalIterator_off;

								// statistics
								_offdiagElemesStat[_opi](6, _r) = StatisticalMeasures::gaussianity(_offdiagElemesStat[_opi](5, _r), _offdiagElemesStat[_opi](2, _r));
								_offdiagElemesStat[_opi](7, _r) = StatisticalMeasures::binder_cumulant(_offdiagElemesStat[_opi](2, _r), _offdiagElemesStat[_opi](4, _r));
							}

							// additionally, for typical values, calculate the exponential of the mean
							{
								for (auto ii : { 1, 3 })
								{
									_offdiagElemesStat[_opi](ii, _r) = std::exp(_offdiagElemesStat[_opi](ii, _r));
								}
							}
						}

						// save the diagonal part
						_diagElems[_opi].col(_r)	= _overlaps.diag();
					}

				}
			}
		}

		// save the checkpoints
		if ((_Ns >= 14 && _r % 2 == 0) || (_Ns < 14 && _r % 20 == 0))
			_saver(_r);

		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);

		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_saver(this->modP.modRanN_);

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::checkETH_level_prop(std::shared_ptr<Hamiltonian<double>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

	// check the random field
	auto _rH	= this->modP.qsm.qsm_h_r_;
	auto _rA	= this->modP.qsm.qsm_h_ra_;
	size_t _Ns	= this->modP.qsm.qsm_Ntot_;
	u64 _Nh		= ULLPOW(_Ns);

	// get info
	std::string modelInfo	= _H->getInfo();
	std::string randomStr	= FileParser::appWRandom("", _H->ran_);
	std::string dir			= makeDirsC(this->mainDir, "QSM_MAT_ELEM_STAT", modelInfo, (_rH != 0) ? VEQV(dh, _rA) + "_" + STRP(_rH, 3) : "");
	std::string extension	= ".h5";

	// set seed
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	// set the placeholder for the values to save (will save only the diagonal elements and other measures)
	v_1d<arma::Mat<double>> _ent	= v_1d<arma::Mat<double>>(_Ns, -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones));
	arma::Mat<double> _en			= -1e5 * arma::Mat<double>(_H->getHilbertSize(), this->modP.modRanN_, arma::fill::zeros);
	arma::Mat<double> _gaps			= -1e5 * arma::Col<double>(this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _ipr1			= -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _ipr2			= -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones);

	// choose the random position inside the dot for the correlation operators
	uint _pos						= this->ran_.randomInt(0, this->modP.qsm.qsm_N_);

	// create the operators
	auto _sx						= Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, _Ns - 1);
	auto _sxc						= Operators::SpinOperators::sig_x(this->modP.qsm.qsm_Ntot_, { _pos, (uint)_Ns - 1 });
	auto _szc						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { _pos, (uint)_Ns - 1 });
	//auto _sp_sm

	// for each site
	std::vector<Operators::Operator<double>> _sz_is;
	std::vector<std::string> _sz_i_names;

	// create the diagonal operators for spin z at each side
	for (auto i = 0; i < _Ns; ++i)
	{
		_sz_is.push_back(Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, i));
		_sz_i_names.push_back("sz_" + STR(i));
	}

	// create the matrices
	v_1d<Operators::Operator<double>> _ops	= { _sx, _sxc, _szc			};
	v_1d<std::string> _opsN					= { "sx_l", "sx_c", "sz_c"	};

	// matrices
	for (auto i = 0; i < _Ns; ++i)
	{
		_ops.push_back(_sz_is[i]);
		_opsN.push_back(_sz_i_names[i]);
	}

	// create the measurement class
	Measurement<double> _measure(this->modP.qsm.qsm_Ntot_, dir, _ops, _opsN);
	_measure.initializeMatrices(_Nh);

	// to save the operators (those elements will be stored for each operator separately)
	// a given matrix element <n|O|n> will be stored in i'th column of the i'th operator
	// the n'th row in the column will be the state index
	// the columns corresponds to realizations of disorder
	v_1d<arma::Mat<double>> _diagElems(_ops.size(), -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones));

	// (mean, typical, mean2, typical2, gaussianity, kurtosis, binder cumulant)
	// the columns will correspond to realizations
	u64 _hs_fractions_diag		= SystemProperties::hs_fraction_diagonal_cut(this->modP.modMidStates_, _Nh);
	// due to mobility edges, for the statistics we'll save two sets of data
	v_1d<arma::Mat<double>> _diagElemsStat(_ops.size(), arma::Mat<double>(7, this->modP.modRanN_, arma::fill::zeros));
	v_1d<arma::Mat<double>> _diagElemsStat_cut(_ops.size(), arma::Mat<double>(7, this->modP.modRanN_, arma::fill::zeros));
	
	const double off_extensive_threshold	= 1e-1;
	const double off_constant_threshold_up	= off_extensive_threshold;
	const double off_constant_threshold_dn	= 1e-2;
	const double off_vanishing_threshold	= std::min(5e-1 / _Ns, off_constant_threshold_dn);

	// (mean, typical, mean2, typical2, mean4, meanabs, gaussianity, binder cumulant)
	v_1d<arma::Mat<double>> _offdiagElemesStat(_ops.size(), arma::Mat<double>(8, this->modP.modRanN_, arma::fill::zeros));
	v_1d<arma::Mat<double>> _offdiagElemesStatConst(_ops.size(), arma::Mat<double>(8, this->modP.modRanN_, arma::fill::zeros));
	v_1d<arma::Mat<double>> _offdiagElemesStatVanish(_ops.size(), arma::Mat<double>(8, this->modP.modRanN_, arma::fill::zeros));
	v_1d<arma::Mat<double>> _offdiagElemesStatExtensive(_ops.size(), arma::Mat<double>(8, this->modP.modRanN_, arma::fill::zeros));

	// saves the histograms of the second moments for the offdiagonal elements 
	v_1d<HistogramAverage<double>> _histAv(_ops.size(), HistogramAverage<double>());
	v_1d<HistogramAverage<double>> _histAvTypical(_ops.size(), HistogramAverage<double>());
	
	// ----------------------- nbins operators -----------------------
	v_1d<Histogram> _histOperatorsDiag(_ops.size(), Histogram());
	v_1d<Histogram> _histOperatorsOffdiag(_ops.size(), Histogram());
	v_1d<Histogram> _histOperatorsOffdiagVanishing(_ops.size(), Histogram());
	v_1d<Histogram> _histOperatorsOffdiagConst(_ops.size(), Histogram());
	v_1d<Histogram> _histOperatorsOffdiagExtensive(_ops.size(), Histogram());
	uint _nbinOperators = 15 * _Ns;
	for (uint _opi = 0; _opi < _ops.size(); ++_opi) 
	{
		// diagonal
		_histOperatorsDiag[_opi].reset(_nbinOperators);
		_histOperatorsDiag[_opi].uniform(0.5, -0.5); 
		// offdiagonal
		_histOperatorsOffdiag[_opi].reset(_nbinOperators);
		_histOperatorsOffdiagConst[_opi].reset(_nbinOperators);
		_histOperatorsOffdiagVanishing[_opi].reset(_nbinOperators);
		_histOperatorsOffdiagExtensive[_opi].reset(_nbinOperators);
		double _offdiagLimit	= 0.5 - 0.025 * _Ns;
		_histOperatorsOffdiag[_opi].uniform(_offdiagLimit, -_offdiagLimit);
		_histOperatorsOffdiagExtensive[_opi].uniform(_offdiagLimit, -_offdiagLimit);
		_offdiagLimit			*= 0.8;
		_histOperatorsOffdiagConst[_opi].uniform(_offdiagLimit, -_offdiagLimit);
		_histOperatorsOffdiagVanishing[_opi].uniform(_offdiagLimit, -_offdiagLimit);

	}

	// create the saving function
	std::function<void(uint)> _saver = [&](uint _r)
		{
			// save the matrices
			for(int i = 0; i < _Ns; ++i)
			{
				// save the entropies (only append when i > 0)
				saveAlgebraic(dir, "entro" + randomStr + extension, _ent[i], STR(i), i > 0);
			}

			// save the iprs and the energy to the same file
			saveAlgebraic(dir, "stat" + randomStr + extension, _gaps, "gap_ratio", false);
			// participation ratio
			saveAlgebraic(dir, "stat" + randomStr + extension, _ipr1, "part_entropy_q=1", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _ipr2, "part_entropy_q=2", true);
			// energy
			saveAlgebraic(dir, "stat" + randomStr + extension, _en, "energy", true);

			// append statistics from the diagonal elements
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "stat" + randomStr + extension, _diagElemsStat[_opi], "diag_" + _name, true);

				// offdiagonal elements
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStat[_opi], "offdiag_" + _name, true);
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStatConst[_opi], "offdiag_" + _name + "_const", true);
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStatVanish[_opi], "offdiag_" + _name + "_vanish", true);
				saveAlgebraic(dir, "stat" + randomStr + extension, _offdiagElemesStatExtensive[_opi], "offdiag_" + _name + "_extensive", true);
			}

			// diagonal elements (only append when _opi > 0)
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "diag" + randomStr + extension, _diagElems[_opi], _name, _opi > 0);
			}

			// save the histograms
			saveAlgebraic(dir, "hist" + randomStr + extension, _histAv[0].edgesCol(), "omegas", false);
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "hist" + randomStr + extension, _histAv[_opi].averages_av(), _name + "_mean", true);
				saveAlgebraic(dir, "hist" + randomStr + extension, _histAvTypical[_opi].averages_av(true), _name + "_typical", true);
			}
			saveAlgebraic(dir, "hist" + randomStr + extension, _histAv[0].countsCol(), "_counts", true);

			// save the distributions of the operators
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsDiag[_opi].edgesCol(), _name + "_diag_edges", _opi > 0);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsDiag[_opi].countsCol(), _name + "_diag_counts", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag[_opi].edgesCol(), _name + "_offdiag_edges", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag[_opi].countsCol(), _name + "_offdiag_counts", true);
				// additional
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag[_opi].edgesCol(), _name + "_offdiag_edges_vanish", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiagConst[_opi].countsCol(), _name + "_offdiag_counts_const", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiagVanishing[_opi].countsCol(), _name + "_offdiag_counts_vanish", true);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiagExtensive[_opi].countsCol(), _name + "_offdiag_counts_extensive", true);
			}

			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
		};

	// go through realizations
	for (int _r = 0; _r < this->modP.modRanN_; ++_r)
	{
		// ----------------------------------------------------------------------------
		
		// checkpoints etc
		{
			LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
			_timer.checkpoint(STR(_r));

			// -----------------------------------------------------------------------------

			LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
			_H->clearH();
			_H->randomize(this->modP.qsm.qsm_h_ra_, _rH, { "h" });

			// -----------------------------------------------------------------------------

			// set the Hamiltonian
			_H->buildHamiltonian();
			_H->diagH(false);
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
		}

		// -----------------------------------------------------------------------------
				
		// get the average energy index and the points around it on the diagonal
		u64 _minIdxDiag, _maxIdxDiag = 0; 

		// set
		{
			std::tie(_minIdxDiag, _maxIdxDiag)		= _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
		}

		// -----------------------------------------------------------------------------
		
		// set the uniform distribution of frequencies in logspace!!!
		if (_r == 0)
		{

			double oMax			= std::abs(_H->getEigVal(_maxIdxDiag) - _H->getEigVal(_minIdxDiag)) * 2.5;
			double oMin			= 1.0l / _Nh / 10;
			u64 _nFrequencies	= 12 * _Ns;

			// set the histograms
			for (auto iHist = 0; iHist < _ops.size(); ++iHist)
			{
				_histAv[iHist].reset(_nFrequencies);
				_histAv[iHist].uniformLog(oMax, oMin);

				_histAvTypical[iHist].reset(_nFrequencies);
				_histAvTypical[iHist].uniformLog(oMax, oMin);
			}
		}

		// -----------------------------------------------------------------------------

		// save the energies
		{
			_en.col(_r) = _H->getEigVal();
		}

		// -----------------------------------------------------------------------------
		
		// calculator of the properties
		{
			// -----------------------------------------------------------------------------
			
			// gap ratios
			{
				// calculate the eigenlevel statistics
				_gaps(_r)					= SystemProperties::eigenlevel_statistics(arma::Col<double>(_H->getEigVal().subvec(_minIdxDiag, _maxIdxDiag - 1)));
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
				for (long long _start = 0; _start < _Nh; ++_start)
				{
					_ipr1(_start, _r) = SystemProperties::information_entropy(_H->getEigVec(_start));
					_ipr2(_start, _r) = std::log(1.0 / SystemProperties::participation_ratio(_H->getEigVec(_start), 2.0));
				}

				// -----------------------------------------------------------------------------
				
				// entanglement entropy
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif			
				for (long long _start = 0; _start < _Nh; ++_start)
				{
					for (int i = 1; i <= _Ns; i++)
					{
						// calculate the entanglement entropy
						//_entr(_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), _Ns - 1, _Hs);
						//_entr(_start - _minIdxDiag, _r) = Entropy::Entanglement::Bipartite::vonNeuman<_T>(_H->getEigVec(_start), _Ns, _Hs);
						uint _maskA	= 1 << (i - 1);
						uint _enti	= _Ns - i;
						_ent[_enti](_start, _r) = Entropy::Entanglement::Bipartite::vonNeuman<double>(_H->getEigVec(_start), 1, _Ns, _maskA, DensityMatrix::RHO_METHODS::SCHMIDT, 2);
					}
				}
			}

			// -----------------------------------------------------------------------------
			
			// diagonal
			{
				for (u64 _start = 0; _start < _Nh; ++_start)
				{
					// calculate the diagonal elements
					const auto& _measured = _measure.measureG(_H->getEigVec(_start));
					// save the diagonal elements
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
					for (int i = 0; i < _measured.size(); ++i)
					{
						const auto _elem			= _measured[i];
						_diagElems[i](_start, _r)	= _elem;

						if (_start >= _minIdxDiag && _start < _maxIdxDiag)
						{
							// add element to the histogram
							_histOperatorsDiag[i].append(_elem);

							auto _elem2				= _elem * _elem;
							auto _logElem			= std::log(std::fabs(_elem));
							auto _logElem2			= std::log(_elem * _elem);

							// save the statistics
							// mean
							_diagElemsStat[i](0, _r) += _elem;
							// typical
							_diagElemsStat[i](1, _r) += _logElem;
							// mean2
							_diagElemsStat[i](2, _r) += _elem2;
							// typical2
							_diagElemsStat[i](3, _r) += _logElem2;
							
						}
					}
				}
				
				// finalize statistics
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (int i = 0; i < _ops.size(); ++i)
				{
					for (uint ii = 0; ii < 4; ii++)
					{
						_diagElemsStat[i](ii, _r)		/= _hs_fractions_diag;
					}
					// save the statistics
					_diagElemsStat[i](4, _r) = StatisticalMeasures::gaussianity(_diagElems[i].col(_r));
					_diagElemsStat[i](5, _r) = StatisticalMeasures::kurtosis(_diagElems[i].col(_r));
					_diagElemsStat[i](6, _r) = StatisticalMeasures::binder_cumulant(_diagElems[i].col(_r));
				}

				// additionally, for typical values, calculate the exponential of the mean
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (int i = 0; i < _ops.size(); ++i)
				{
					_diagElemsStat[i](1, _r) = std::exp(_diagElemsStat[i](1, _r));
					_diagElemsStat[i](3, _r) = std::exp(_diagElemsStat[i](3, _r));
				}
			}

			// -----------------------------------------------------------------------------
			
			// offdiagonal
			{
				u64 _constantIterator	= 0;
				u64 _vanishingIterator	= 0;
				u64 _extensiveIterator	= 0;
				u64 _totalIterator		= 0;
				const double _avEn		= _H->getEnAv();
				// go through the whole spectrum (do not save pairs, only one element as it's Hermitian.
				for (u64 i = _minIdxDiag; i < _maxIdxDiag; ++i)
				{
					auto _en_l = _H->getEigVal(i);
					for (u64 j = i + 1; j < _maxIdxDiag; ++j)
					{
						auto _en_r = _H->getEigVal(j);
						
						// check the energy difference
						if (!SystemProperties::hs_fraction_close_mean(_en_l, _en_r, _avEn, this->modP.modEnDiff_))
							continue;

						_totalIterator++;

						// calculate the frequency
						const double w				= std::abs(_en_l - _en_r);

						// calculate the values
						const auto& _measured		= _measure.measureG(_H->getEigVec(i), _H->getEigVec(j));

						// checks for means
						const bool _isvanishing		= w < off_vanishing_threshold;
						const bool _isconstant		= w < off_constant_threshold_up && w > off_constant_threshold_dn;
						const bool _isextensive		= w > off_extensive_threshold;

						_constantIterator			+= _isconstant;
						_vanishingIterator			+= _isvanishing;
						_extensiveIterator			+= _isextensive;

						// save the off-diagonal elements
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
						for (int i = 0; i < _ops.size(); ++i)
						{
							auto _elem				= _measured[i];
							auto _elem2				= _elem * _elem;
							auto _logElem			= std::log(std::abs(_elem));
							auto _logElem2			= std::log(_elem * _elem);

							// mean
							_offdiagElemesStat[i](0, _r) += _elem;
							// typical
							_offdiagElemesStat[i](1, _r) += _logElem;
							// mean2
							_offdiagElemesStat[i](2, _r) += _elem2;
							// typical2
							_offdiagElemesStat[i](3, _r) += _logElem2;
							// mean4
							_offdiagElemesStat[i](4, _r) += _elem2 * _elem2;
							// meanabs
							_offdiagElemesStat[i](5, _r) += std::abs(_elem);

							// check binned statistics
							if (_isextensive)
							{
								// mean
								_offdiagElemesStatExtensive[i](0, _r) += _elem;
								// typical
								_offdiagElemesStatExtensive[i](1, _r) += _logElem;
								// mean2
								_offdiagElemesStatExtensive[i](2, _r) += _elem2;
								// typical2
								_offdiagElemesStatExtensive[i](3, _r) += _logElem2;
								// mean4
								_offdiagElemesStatExtensive[i](4, _r) += _elem2 * _elem2;
								// meanabs
								_offdiagElemesStatExtensive[i](5, _r) += std::abs(_elem);
								//histogram
								_histOperatorsOffdiagExtensive[i].append(_elem);
							}
							if (_isconstant)
							{
								// mean
								_offdiagElemesStatConst[i](0, _r) += _elem;
								// typical
								_offdiagElemesStatConst[i](1, _r) += _logElem;
								// mean2
								_offdiagElemesStatConst[i](2, _r) += _elem2;
								// typical2
								_offdiagElemesStatConst[i](3, _r) += _logElem2;
								// mean4
								_offdiagElemesStatConst[i](4, _r) += _elem2 * _elem2;
								// meanabs
								_offdiagElemesStatConst[i](5, _r) += std::abs(_elem);
								// histogram
								_histOperatorsOffdiagConst[i].append(_elem);
							}
							if (_isvanishing)
							{
								// mean
								_offdiagElemesStatVanish[i](0, _r) += _elem;
								// typical
								_offdiagElemesStatVanish[i](1, _r) += _logElem;
								// mean2
								_offdiagElemesStatVanish[i](2, _r) += _elem2;
								// typical2
								_offdiagElemesStatVanish[i](3, _r) += _logElem2;
								// mean4
								_offdiagElemesStatVanish[i](4, _r) += _elem2 * _elem2;
								// meanabs
								_offdiagElemesStatVanish[i](5, _r) += std::abs(_elem);
								// histogram
								_histOperatorsOffdiagVanishing[i].append(_elem);
							}

							// add to value histogram
							_histOperatorsOffdiag[i].append(_elem);

							// add to the histograms
							_histAv[i].append(w, _elem2);
							_histAvTypical[i].append(w, _logElem2);
						}
					}
				}
				
				// finalize statistics
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (int i = 0; i < _ops.size(); ++i)
				{
					for (uint ii = 0; ii < 6; ii++)
					{
						_offdiagElemesStat[i](ii, _r) /= (long double)_totalIterator;
						_offdiagElemesStatConst[i](ii, _r) /= (long double)_constantIterator;
						_offdiagElemesStatVanish[i](ii, _r) /= (long double)_vanishingIterator;
						_offdiagElemesStatExtensive[i](ii, _r) /= (long double)_extensiveIterator;
					}

					// statistics
					_offdiagElemesStat[i](6, _r) = StatisticalMeasures::gaussianity(_offdiagElemesStat[i](5, _r), _offdiagElemesStat[i](2, _r));
					_offdiagElemesStat[i](7, _r) = StatisticalMeasures::binder_cumulant(_offdiagElemesStat[i](2, _r), _offdiagElemesStat[i](4, _r));

					_offdiagElemesStatConst[i](6, _r) = StatisticalMeasures::gaussianity(_offdiagElemesStatConst[i](5, _r), _offdiagElemesStatConst[i](2, _r));
					_offdiagElemesStatConst[i](7, _r) = StatisticalMeasures::binder_cumulant(_offdiagElemesStatConst[i](2, _r), _offdiagElemesStatConst[i](4, _r));

					_offdiagElemesStatVanish[i](6, _r) = StatisticalMeasures::gaussianity(_offdiagElemesStatVanish[i](5, _r), _offdiagElemesStatVanish[i](2, _r));
					_offdiagElemesStatVanish[i](7, _r) = StatisticalMeasures::binder_cumulant(_offdiagElemesStatVanish[i](2, _r), _offdiagElemesStatVanish[i](4, _r));
				}
				
				// additionally, for typical values, calculate the exponential of the mean
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
				for (int i = 0; i < _ops.size(); ++i)
				{
					for (auto ii : { 1, 3 })
					{
						_offdiagElemesStat[i](ii, _r) = std::exp(_offdiagElemesStat[i](ii, _r));
						_offdiagElemesStatConst[i](ii, _r) = std::exp(_offdiagElemesStatConst[i](ii, _r));
						_offdiagElemesStatVanish[i](ii, _r) = std::exp(_offdiagElemesStatVanish[i](ii, _r));
						_offdiagElemesStatExtensive[i](ii, _r) = std::exp(_offdiagElemesStatExtensive[i](ii, _r));
					}
				}

			}
		}

		// save the checkpoints
		if (_Ns >= 14 && _r % 2 == 0)
		{
			// save the diagonals
			_saver(_r);
		}
		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_saver(this->modP.modRanN_);

	// save the command directly to the file
	{
		std::string dir_in = makeDirs(this->mainDir, "QSM_MAT_ELEM");
		std::ofstream file(dir_in + "qsm_scp_newest.log", std::ios::app);
		file << "scp -3 -r scp://klimak97@ui.wcss.pl:22//" + dir + " ./" << std::endl;
		file.close();
		std::ofstream file_rsync(dir_in + "qsm_rsync.log", std::ios::app);
		file_rsync << "rsync -rv --rsh --ignore-existing -e 'ssh -p 22' klimak97@ui.wcss.pl:mylustre-hpc-maciek/QSolver/DATA_LJUBLJANA_BIG_SWEEP ./";
		file_rsync.close();
	}

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::checkETH_time_evo(std::shared_ptr<Hamiltonian<double>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

	// check the random field
	auto _rH	= this->modP.qsm.qsm_h_r_;
	auto _rA	= this->modP.qsm.qsm_h_ra_;
	size_t _Ns	= this->modP.qsm.qsm_Ntot_;
	u64 _Nh		= ULLPOW(_Ns);
	v_1d<Operators::Operator<double>> _ops;
	v_1d<std::string> _opsN;

	// get info
	std::string modelInfo	= _H->getInfo();
	std::string randomStr	= FileParser::appWRandom("", _H->ran_);
	std::string dir			= makeDirsC(this->mainDir, "QSM_MAT_ELEM_TIME_EVOLUTION", modelInfo, (_rH != 0) ? VEQV(dh, _rA) + "_" + STRP(_rH, 3) : "");
	std::string extension	= ".h5";

	// set seed
	if (this->modP.modRanSeed_ != 0) _H->setSeed(this->modP.modRanSeed_);

	// couplings for the sz!
	{
		auto _szc_lm1					= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { ((uint)_Ns - 2), ((uint)_Ns - 1) });
		auto _szc_h						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { ((uint)(_Ns / 2)), ((uint)_Ns - 1)});
		auto _sz_l						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { ((uint)_Ns - 1) });
		auto _sz_lm1					= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { ((uint)_Ns - 2) });
		auto _sz_h						= Operators::SpinOperators::sig_z(this->modP.qsm.qsm_Ntot_, { ((uint)(_Ns / 2)) });

		// create the matrices
		_ops							= { _szc_lm1, _szc_h, _sz_l, _sz_lm1, _sz_h };
		_opsN							= { "szc/l_lm1", "szc/l2_l", "sz/l", "sz/lm1", "sz/l2"};
	}

	// create the measurement class
	Measurement<double> _measure(this->modP.qsm.qsm_Ntot_, dir, _ops, _opsN);
	_measure.initializeMatrices(_Nh);

	// set the placeholder for the values to save (will save only the diagonal elements and other measures)
	// GAMMA (see 2022, Suntajs, PRL)
	// gamma0_2, t_heisenberg_gamma, _t_heisenberg_mean, _t_heisenberg_typ
	arma::Mat<double> _meanlvl			= -1e5 * arma::Mat<double>(5, this->modP.modRanN_, arma::fill::ones);
	u64 _hs_fractions_diag				= SystemProperties::hs_fraction_diagonal_cut(0.5, _Nh);

	// time evolution saved here
	long double _heisenberg_time_est	= ULLPOW((_Ns));
	arma::Col<double> _timespace		= arma::logspace(-2, std::log10(_heisenberg_time_est * 100), 2500);
	// create initial states for the quench
	arma::Col<double> _initial_state_me;

	// saves the energies and the LDOSs of the initial states
	arma::Mat<double> _energies			= arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::zeros);
	// save the LDOS
	arma::Mat<double> _ldos_me			= arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::zeros);
	// to save the energy densities (mean energy[system], mean energy state, <state|H2|state>, mobility_edge_point)
	arma::Mat<double> _energydensitiesME(4, this->modP.modRanN_);
	// to save the diagonal elements
	VMAT<double> _diagonals(_ops.size(), _Nh, this->modP.modRanN_, arma::fill::zeros);

	// save the time evolution here
	VMAT<double> _timeEvolutionME(_ops.size(), _timespace.size(), this->modP.modRanN_, arma::fill::zeros);
	VMAT<double> _timeEntropyME(_Ns, _timespace.size(), this->modP.modRanN_, arma::fill::zeros);
	v_1d<arma::Col<double>> _timeZeroME(_ops.size(), arma::Col<double>(this->modP.modRanN_, arma::fill::zeros));

	// ------------------------- MICROCANONICAL AVERAGES -------------------------

	const v_1d<double> _toCheckEps = { 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4 };

	// to save the microcanonical averages
	VMAT<double> _microcanonicalME(_ops.size(), 1 + _toCheckEps.size(), this->modP.modRanN_, arma::fill::zeros);

	// to save the diagonal ensemble averages
	VMAT<double> _diagonalME(_ops.size(), _toCheckEps.size(), this->modP.modRanN_, arma::fill::zeros);

	// -------------------------------- SAVER ------------------------------------

	auto _microcanonical_saver	= [&](	uint _r,
										uint _opi, 
										VMAT<double>& _diagonalvals,
										VMAT<double>& _microvals,
										const arma::Col<double>& _soverlaps, 
										const v_1d<u64>& _mins,
										const v_1d<u64>& _maxs)
		{
			const auto& _diagonal	= _diagonals[_opi].col(_r);
			// long time average (all states)
			_microvals[_opi](0, _r)	= arma::dot(_diagonal, _soverlaps);

			uint _size				= _toCheckEps.size();
			for (int _ifrac = 0; _ifrac < _size; _ifrac++)
			{
				u64 _minin			= _mins[_ifrac];
				u64 _maxin			= _maxs[_ifrac];
				if (_minin <= 0)
					_minin = 0;
				if (_maxin >= _Nh - 1)
					_maxin = _Nh - 1;

				// check the bounds
				while(_minin >= _maxin)
				{
					if(_minin > 0)
						_minin--;
					if(_maxin < _Nh - 1)
						_maxin++;
				}
				// around the energy (\sum c_n^2 a_nn) (in the energy window)
				_microvals[_opi](1 + _ifrac, _r)= arma::dot(_diagonal.subvec(_minin, _maxin), _soverlaps.subvec(_minin, _maxin));
				// around the energy, just ann
				_diagonalvals[_opi](_ifrac, _r)	= arma::mean(_diagonal.subvec(_minin, _maxin));
			}
		};

	// ----------------------------- EVOLVE STATE -------------------------------

	auto _evolveState = [&](uint _r, 
							const arma::Col<double>& _initial_state,
							arma::Mat<double>& _ldoses,
							arma::Mat<double>& _energydensities,
							VMAT<double>& _diagonalvals,
							VMAT<double>& _microvals,
							VMAT<double>& _timeEvolution,
							v_1d<arma::Col<double>>& _timeZero,
							const v_1d<arma::SpMat<double>>& _matrices)
		{
			// calculate the overlaps of the initial state with the eigenvectors
			const arma::Col<double> _overlaps	= _H->getEigVec().t() * arma::real(_initial_state);
			const arma::Col<double> _soverlaps	= arma::square(arma::abs(_overlaps));
			// calculate the average energy index
			double _meanE						= _H->getEnAv();

			// save the energies
			_ldoses.col(_r)						= SystemProperties::calculate_LDOS(_H->getEigVal(), _overlaps);
			_energydensities(0, _r)				= _meanE;

			// apply the Hamiltonian to the initial state
			const arma::Col<double> _init_stat_H= _H->getHamiltonian() * _initial_state;
			const double _E						= arma::cdot(_initial_state, _init_stat_H);
			const double _E2					= arma::cdot(_init_stat_H, _init_stat_H);
			u64 _Eidx							= _H->calcEnIdx(_E);
			LOGINFO(VEQ(_Eidx), LOG_TYPES::TRACE, 1);
			_energydensities(1, _r)				= _E;
			_energydensities(2, _r)				= _E2;
			_energydensities(3, _r)				= std::reinterpret_pointer_cast<QSM<double>>(_H)->get_mobility_edge(_E);

			// calculate the bounds
			std::vector<u64> _mins(_toCheckEps.size()), _maxs(_toCheckEps.size());

			// around the given energy _E
			for (int _ifrac = 0; _ifrac < _toCheckEps.size(); _ifrac++)
				std::tie(_mins[_ifrac], _maxs[_ifrac]) = _H->getEnArndEnEps(_Eidx, _toCheckEps[_ifrac]);

			// microcanonical and diagonal ensembles
#pragma omp parallel for num_threads(_ops.size())
			for (int _opi = 0; _opi < _ops.size(); ++_opi)
			{
				// ferromagnetic
				_microcanonical_saver(_r, _opi, _diagonalvals, _microvals, _soverlaps, _mins, _maxs);
			}

			// -----------------------------------------------------------------------------
			
			// save zero time value
			{
	#pragma omp parallel for num_threads(_ops.size())
				for (uint _opi = 0; _opi < _ops.size(); ++_opi)
					_timeZero[_opi](_r) = arma::as_scalar(arma::cdot(_initial_state, _matrices[_opi] * _initial_state));
			}

			// evolution
#pragma omp parallel for num_threads(this->threadNum)
			for (int _ti = 0; _ti < _timespace.size(); _ti++)
			{
				const auto _time = _timespace(_ti);
				const auto _st	 = SystemProperties::TimeEvolution::time_evo(_H->getEigVec(), _H->getEigVal(), _overlaps, _time);

				// for each operator we can now apply the expectation value		
				for (uint _opi = 0; _opi < _ops.size(); ++_opi)
				{
					const cpx _rt					= arma::as_scalar(arma::cdot(_st, arma::Col<cpx>(_matrices[_opi] * _st)));
					_timeEvolution[_opi](_ti, _r)	= std::real(_rt);
				}

				// calculate the entanglement entropy for each site
				{
					for (int i = 1; i <= _Ns; i++)
					{
						// calculate the entanglement entropy
						uint _maskA						= 1 << (i - 1);
						uint _enti						= _Ns - i;
						_timeEntropyME[_enti](_ti, _r)	= Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_st, 1, _Ns, _maskA, DensityMatrix::RHO_METHODS::SCHMIDT, 2);
					}
				}
			}

		};


	// -------------------------------- SAVER ------------------------------------

	// create the saving function
	std::function<void(uint)> _saver = [&](uint _r)
		{
			// variance in th Hamiltonian
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(0).as_col()), "mean_level_gamma", false);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(1).as_col()), "heis_time_gamma", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(2).as_col()), "1_over_mean_level_spacing", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(3).as_col()), "1_over_mean_level_spacing_typ", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, arma::vec(_meanlvl.row(4).as_col()), "th_time_est", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _energies, "energies", true);

			// save the ldos's
			saveAlgebraic(dir, "ldos" + randomStr + extension, _ldos_me, "ME", false);

			// save the energy densities
			saveAlgebraic(dir, "energydens" + randomStr + extension, arma::vec(_energydensitiesME.row(0).as_col() / _Ns), "mean", false);
			saveAlgebraic(dir, "energydens" + randomStr + extension, arma::mat(_energydensitiesME.rows(1, 3) / _Ns), "ME", true);
			
			// save the matrices for time evolution
			saveAlgebraic(dir, "evo" + randomStr + extension, _timespace, "time", false);
			for(int i = 0; i < _Ns; i++)
				saveAlgebraic(dir, "evo" + randomStr + extension, _timeEntropyME[i], "entanglement_entropy/ME/" + STR((i + 1)), true);
			// save the averages epsilon
			saveAlgebraic(dir, "avs" + randomStr + extension, arma::vec(_toCheckEps), "eps", false);

			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);

				// diagonal
				saveAlgebraic(dir, "diag" + randomStr + extension, _diagonals[_opi], _name, _opi > 0);

				// evolution
				saveAlgebraic(dir, "evo" + randomStr + extension, _timeEvolutionME[_opi], _name + "/ME", true);

				// at zero
				saveAlgebraic(dir, "evo" + randomStr + extension, _timeZeroME[_opi], _name + "/zero/ME", true);

				// diagonal ensemble
				saveAlgebraic(dir, "avs" + randomStr + extension, _diagonalME[_opi], _name + "/diag/ME", true);

				// microcanonical
				saveAlgebraic(dir, "avs" + randomStr + extension, arma::mat(_microcanonicalME[_opi].rows(1, _toCheckEps.size())), _name + "/micro/ME", true);

				// long time average
				saveAlgebraic(dir, "avs" + randomStr + extension, arma::vec(_microcanonicalME[_opi].row(0).as_col()), _name + "/long/ME", true);
			}

			LOGINFO("Checkpoint:" + STR(_r), LOG_TYPES::TRACE, 4);
		};

	double _alpha	= std::reinterpret_pointer_cast<QSM<double>>(_H)->get_alpha();
	double _g0		= std::reinterpret_pointer_cast<QSM<double>>(_H)->get_g0();

	// go through realizations
	for (int _r = 0; _r < this->modP.modRanN_; ++_r)
	{
		// ----------------------------------------------------------------------------
		
		// checkpoints etc
		{
			LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
			_timer.checkpoint(STR(_r));

			// -----------------------------------------------------------------------------

			LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
			_H->clearH();
			_H->randomize(this->modP.qsm.qsm_h_ra_, _rH, { "h" });

			// -----------------------------------------------------------------------------

			// set the Hamiltonian
			_H->buildHamiltonian();
			_H->diagH(false);
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);

			// -----------------------------------------------------------------------------

			// set the initial state (seek the quench near the average energy)
			_initial_state_me = SystemProperties::TimeEvolution::create_initial_quench_state<double>(SystemProperties::TimeEvolution::QuenchTypes::SEEK, _Nh, _Ns, _H->getEnAv(), _H->getHamiltonian().diag());
		}

		// -----------------------------------------------------------------------------

		// calculator of the properties
		{
			// -----------------------------------------------------------------------------

			// mean level spacing
			{
				long double _th_freq	= QSM<double>::get_thouless_freq_est(_alpha, _g0, _Ns);
				long double _h_freq		= 1.0 / _Nh;
				// energies
				_energies.col(_r)	= _H->getEigVal();
				// mean levels (gamma, heisenberg)
				_meanlvl(0, _r)		= SystemProperties::mean_lvl_gamma(_H->getHamiltonian());
				_meanlvl(1, _r)		= SystemProperties::mean_lvl_heis_time(_meanlvl(0, _r), _Ns);

				// get the average energy index and the points around it on the diagonal
				const auto [_min, _max] = _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
				auto _E					= _H->getEigVal().subvec(_min, _max);
				_meanlvl(2, _r)			= 1.0 / SystemProperties::mean_lvl_spacing(_E);
				// get the Heisenberg frequency from the mean level spacing
				_h_freq					= SystemProperties::mean_lvl_spacing_typ(_E);
				_meanlvl(3, _r)			= 1.0 / _h_freq;
				_meanlvl(4, _r)			= 1.0 / _th_freq;
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(0, _r)), StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(1, _r)), StrParser::StrColors::blue), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(2, _r)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(3, _r)), StrParser::StrColors::yellow), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(4, _r)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
			}

			// -----------------------------------------------------------------------------

			// other measures
			{
				// calculate the diagonals
				const auto& _matrices				= _measure.getOpG_mat();

#pragma omp parallel for num_threads(_ops.size())
				for (int _opi = 0; _opi < _ops.size(); ++_opi)
				{
					for (int i = 0; i < _Nh; ++i)
						_diagonals[_opi](i, _r)		= Operators::applyOverlap(_H->getEigVec(i), _matrices[_opi]);
				}

				// evolve the states
				_evolveState(_r, _initial_state_me, _ldos_me, _energydensitiesME, _diagonalME, _microcanonicalME, _timeEvolutionME, _timeZeroME, _matrices);
			}
		}
		// save the checkpoints
		if ((_Ns >= 14 && (_r % 4 == 0)) || (_Ns < 14 && (_r % 25 == 0)))
		{
			// save the diagonals
			_saver(_r);
		}
		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_saver(this->modP.modRanN_);

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}
