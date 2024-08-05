#include "../../include/user_interface/user_interface.h"

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::makeSimETH()
{
	bool _isquadratic [[maybe_unused]] = check_noninteracting(this->modP.modTyp_);

	// define the models
	if (this->modP.modTyp_ == MY_MODELS::RP_M)
		this->useComplex_ = !this->modP.rosenzweig_porter.rp_be_real_;
	else
		this->useComplex_ = false;

	// go complex if needed
	bool _takeComplex = (this->isComplex_ || this->useComplex_);	

	// define the models depending on which to choose
	bool _isok = false;
	if (check_noninteracting(this->modP.modTyp_))
		_isok = this->defineModelsQ(false);
	else
		_isok = this->defineModels(false, false, false);
	
	// go through the function choice
	if (_isok)
	{
		if (this->chosenFun == 40)
		{
			//this->checkETH(this->hamDouble);
		}
		else if (this->chosenFun == 42 || this->chosenFun == 43)
		{
			if (_takeComplex)
			{
				this->checkETH_statistics(this->hamComplex);
			}
			else
			{
				this->checkETH_statistics(this->hamDouble);
			}
		}
		else if (this->chosenFun == 44)
		{
			if (_takeComplex)
			{
				this->checkETH_scaling_offdiag(this->hamComplex);
			}
			else
			{
				this->checkETH_scaling_offdiag(this->hamDouble);
			}
		}
		else if (this->chosenFun == 46 || this->chosenFun == 45)
		{
			if (_takeComplex)
			{
				this->checkETH_time_evo(this->hamComplex);
			}
			else
			{
				this->checkETH_time_evo(this->hamDouble);
			}
		}
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
		_params = { this->modP.power_law_random_bandwidth.plrb_a_ };
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
				this->modP.qsm.qsm_alpha_ = v_1d<double>(this->latP.Ntot_, _param);
			else if (this->modP.modTyp_ == MY_MODELS::RP_M)
				this->modP.rosenzweig_porter.rp_g_ = v_1d<double>(this->latP.Ntot_, _param);
			else if (this->modP.modTyp_ == MY_MODELS::ULTRAMETRIC_M)
				this->modP.ultrametric.um_alpha_ = v_1d<double>(this->latP.Ntot_ - this->modP.ultrametric.um_N_, _param);
			else if (this->modP.modTyp_ == MY_MODELS::POWER_LAW_RANDOM_BANDED_M)
				this->modP.power_law_random_bandwidth.plrb_a_ = _param;

			// define the models
			this->resetEd();

			// simulate
			this->makeSimETH();
		}
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// ------------------------------------------ ETH ------------------------------------------------

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

constexpr static bool check_saving_size(u64 _Nh, uint _r)
{
	return ((_Nh > ULLPOW(14)) || (BETWEEN(_Nh, 10, 15) && (_r % 10 == 0)) || ((_Nh <= 1024) && (_r % 50 == 0)));
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Based on the model type, obtain both quadratic and many body operators
* @param _Nh: Hilbert space size
* @param _isquadratic: is the model quadratic
* @param _ismanybody: is the model many body
*/
std::pair<v_1d<std::shared_ptr<Operators::Operator<double>>>, strVec> UI::ui_eth_getoperators(const size_t _Nh, bool _isquadratic, bool _ismanybody)
{
	const size_t _Ns = this->latP.Ntot_;
	v_1d<std::shared_ptr<Operators::Operator<double>>> _ops;
	strVec _opsN;

	// operators!
	{
		// add many body operators if applicable
		if (_ismanybody)
		{
			auto _type				= "mb";
			// add z spins 
			v_1d<size_t> _toTake	= v_1d<size_t>({ 0, (size_t)(_Ns / 2), _Ns - 1});
			//v_1d<size_t> _toTake = v_1d<size_t>({ _Ns - 1});

			for (uint i : _toTake)
			//for (uint i = 0; i < _Ns; ++i)
			{
				auto _op	= std::make_shared<Operators::Operator<double>>(Operators::SpinOperators::sig_z(this->latP.Ntot_, i));

				// push it
				_ops.push_back(_op);
				_opsN.push_back(Operators::createOperatorName(_type, "sz", STR(i)));
			}

			// add other operators
			_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::SpinOperators::sig_z(this->latP.Ntot_, { (uint)(_Ns - 2), (uint)(_Ns - 1) })));
			_opsN.push_back(Operators::createOperatorName(_type, "szc", STR(_Ns - 2), STR(_Ns - 1)));
			_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::SpinOperators::sig_z(this->latP.Ntot_, { (uint)(0), (uint)(1) })));
			_opsN.push_back(Operators::createOperatorName(_type, "szc", STR(0), STR(1)));
			_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::SpinOperators::sig_x(this->latP.Ntot_, { 0, (uint)_Ns - 1 })));
			_opsN.push_back(Operators::createOperatorName(_type, "sxc", STR(0), STR(_Ns - 1)));
		}

		// add quadratic operators if applicable
		if (_isquadratic)
		{
			auto _type				= "sp";

			// add other operators
			if (_Nh <= ULLPOW(16))
			{
				_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::QuadraticOperators::quasimomentum_occupation(_Nh)));
				_opsN.push_back(Operators::createOperatorName(_type, "quasimomentum_occupation", "0"));
			}

			// correlation
			_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::QuadraticOperators::nn_correlation(_Nh, 0, _Nh - 1)));
			_opsN.push_back(Operators::createOperatorName(_type, "nn_correlation", "0", STR(_Nh - 1)));
			_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::QuadraticOperators::nn_correlation(_Nh, 0, 1)));
			_opsN.push_back(Operators::createOperatorName(_type, "nn_correlation", "0", "1"));

			v_1d<size_t> _toTake = v_1d<size_t>({ 0, (size_t)(_Nh / 2), _Nh - 1});
			for (auto i: _toTake)
			{
				_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::QuadraticOperators::site_occupation(_Nh, i)));
				_opsN.push_back(Operators::createOperatorName(_type, "site_occupation", STR(i)));
			}

			// random superposition (all of 'em)
			v_1d<double> _rcoefs; 
			_rcoefs = v_1d<double>(_Nh, 1.0);
			for(int i = 0; i < _Nh; i++)
				_rcoefs[i] = this->ran_.template random<double>(-1.0, 1.0);
			_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::QuadraticOperators::site_occupation_r(_Nh, _rcoefs)));
			_opsN.push_back(Operators::createOperatorName(_type, "site_occupation_r"));

			// nq - pi
			_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::QuadraticOperators::site_nq(_Nh, PI)));
			_opsN.push_back(Operators::createOperatorName(_type, "nq", "pi"));

			// nq - pi / 4
			_ops.push_back(std::make_shared<Operators::Operator<double>>(Operators::QuadraticOperators::site_nq(_Nh, PI / 4.0)));
			_opsN.push_back(Operators::createOperatorName(_type, "nq", "pi" OPERATOR_SEP_DIV "4"));
		}
	}

	return std::make_pair(_ops, _opsN);
}

// ###############################################################################################

/*
* @brief Randomize the Hamiltonian to get another realization of the system. 
* @param _H: Hamiltonian to randomize
* @param _r: realization number
* @param _spinchanged: which spin to change (if applicable)
*/
template<typename _T>
void UI::ui_eth_randomize(std::shared_ptr<Hamiltonian<_T>> _H, int _r, uint _spinchanged)
{
	bool isQuadratic [[maybe_unused]]	= _H->getIsQuadratic(),
		 isManyBody	 [[maybe_unused]]	= _H->getIsManyBody();

	_H->clearH();

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
					auto _h = _Hp->getMagnetic(_Ns - _N - _spinchanged);
					_Hp->setMagnetic(_Ns - 3 - _spinchanged, -_h);
				}
			}
		}
		else if (this->modP.modTyp_ == MY_MODELS::RP_M)
			_H->randomize(0, 1.0, { "g" });
		else if (this->modP.modTyp_ == MY_MODELS::ULTRAMETRIC_M)
			_H->randomize(0, 1.0, {});
	}

	// -----------------------------------------------------------------------------

	// set the Hamiltonian
	_H->buildHamiltonian();
	_H->diagH(false);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<typename _T>
void UI::checkETH_scaling_offdiag(std::shared_ptr<Hamiltonian<_T>> _H)
{
	_timer.reset();
	LOGINFO("", LOG_TYPES::TRACE, 40, '#', 0);

	// check the random field
	size_t _Ns	= this->modP.qsm.qsm_Ntot_;

	// many body Hilbert space size
	u64 _Nh		= ULLPOW(_Ns);

	// get info
	std::string modelInfo, dir = "ETH_MAT_OFFD_SCALING", randomStr, extension;
	this->get_inf_dir_ext_r(_H, dir, modelInfo, randomStr, extension);

	bool isQuadratic		= _H->getIsQuadratic(), 
		isManyBody			= _H->getIsManyBody();

	v_1d<std::shared_ptr<Operators::Operator<double>>> _ops;
	strVec _opsN;
	std::tie(_ops, _opsN)	= this->ui_eth_getoperators(_Nh, isQuadratic, isManyBody);

	// get the diagonal fraction
	u64 _hs_fractions_diag	= SystemProperties::hs_fraction_diagonal_cut(this->modP.modMidStates_, _Nh);

	// set the placeholder for the values to save (will save only the diagonal elements and other measures)
	// GAMMA (see 2022, Suntajs, PRL)
	// gamma0, h_gamma, h_mean_lvl_mean, h_mean_lvl_typ, t_th_est
	arma::Mat<double> _meanlvl		= -1e5 * arma::Mat<double>(5, this->modP.modRanN_, arma::fill::ones);

	// create the measurement class
	Measurement<double> _measure(this->modP.qsm.qsm_Ntot_, dir, _ops, _opsN, 1, _Nh);

	// (mean, typical, mean2, typical2, mean4, meanabs, gaussianity, binder cumulant)
	size_t _offdiag_elem_num			= 1000;
	uint _nbinOperators					= 15 * _Ns;
	VMAT<double> _offdiagElemesStat_low(_ops.size(), 8, this->modP.modRanN_, arma::fill::zeros);
	VMAT<_T> _offdiagElements_low(_ops.size(), this->modP.modRanN_, _offdiag_elem_num, arma::fill::ones, -1e5);
	VMAT<_T> _diagonals(_ops.size(), _Nh, this->modP.modRanN_, arma::fill::zeros);

	// save the energies
	arma::Mat<double> _energies			= arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::zeros);

	// ----------------------- nbins operators -----------------------
	v_1d<Histogram> _histOperatorsOffdiag_low(_ops.size(), Histogram());

	// create the histograms for the operators
	for (uint _opi = 0; _opi < _ops.size(); ++_opi) 
	{
		// offdiagonal
		const double _offdiagLimit = 0.5 - 0.025 * _Ns;
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
				saveAlgebraic(dir, "offdiagval" + randomStr + extension, _offdiagElements_low[_opi], "offdiag/low/" + _name, _opi > 0);

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
			if (this->modP.modTyp_ == MY_MODELS::QSM_M)
				_H->randomize(this->modP.qsm.qsm_h_ra_, this->modP.qsm.qsm_h_r_, { "h" });
			else if(this->modP.modTyp_ == MY_MODELS::RP_M)
				_H->randomize(0, 1.0, { "g" });

			// -----------------------------------------------------------------------------

			// set the Hamiltonian
			_H->buildHamiltonian();
			_H->diagH(false);
			_energies.col(_r) = _H->getEigVal();
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
		}

		// -----------------------------------------------------------------------------
				
		// get the average energy index and the points around it on the diagonal
		u64 _minIdxDiag = 0, _maxIdxDiag = 0; 

		// set
		{
			std::tie(_minIdxDiag, _maxIdxDiag) = _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
		}

		// -----------------------------------------------------------------------------
		
		//double _th_freq		= QSM<double>::get_thouless_freq_est(_alpha, _Ns - this->modP.qsm.qsm_N_, _g0);
		long double _th_freq	= QSM<double>::get_thouless_freq_est(0.9, 1.0, _Ns);
		long double _h_freq		= 1.0 / _Nh;
		
		// calculator of the properties
		{
			// -----------------------------------------------------------------------------

			// mean level spacing
			{
				_meanlvl(0, _r)		= _H->getMeanLevelSpacing();
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
#pragma omp parallel for num_threads(_Ns > 14 ? 2 : this->threadNum)
#endif
				for(int _opi = 0; _opi < _matrices.size(); _opi++)
				{
					arma::Mat<_T> _overlaps	= Operators::applyOverlapMat(_H->getEigVec(), _matrices[_opi]);
					_diagonals[_opi].col(_r)	= _overlaps.diag();

					// save the iterators
					u64 _totalIterator_l		= 0;
					const double _avEn			= _H->getEnAv();

					// go through the whole spectrum (do not save pairs, only one element as it's Hermitian.
					for (u64 i = 0; i < _Nh; ++i)
					{
						const double _en_l		= _H->getEigVal(i);

						for (u64 j = i + 1; j < _Nh; ++j)
						{
							const double _en_r	= _H->getEigVal(j);

							// check the energy difference
							if (!SystemProperties::hs_fraction_close_mean(_en_l, _en_r, _avEn, this->modP.modEnDiff_))
								continue;

							bool _isAroundLow		= SystemProperties::hs_fraction_diff_between(_en_l, _en_r, _lowbound / 5.0, _lowbound * 5.0);

							// check the frequency
							if(!_isAroundLow)
								continue;

							// calculate the frequency
							//const double w			= std::abs(_en_l - _en_r);

							// calculate the values
							const auto& _measured	= _overlaps(i, j);

							// save the off-diagonal elements

							auto _elem				= _measured;
							auto _elemabs			= std::abs(_elem);
							auto _elemreal			= algebra::cast<double>(_elem);
							auto _elem2				= _elemabs * _elemabs;
							auto _logElem			= std::log(_elemabs);
							auto _logElem2			= std::log(_elemabs * _elemabs);

							if (_isAroundLow)
							{
								// mean
								_offdiagElemesStat_low(_opi, 0, _r) += _elemreal;
								// typical
								_offdiagElemesStat_low(_opi, 1, _r) += _logElem;
								// mean2
								_offdiagElemesStat_low(_opi, 2, _r) += _elem2;
								// typical2
								_offdiagElemesStat_low(_opi, 3, _r) += _logElem2;
								// mean4
								_offdiagElemesStat_low(_opi, 4, _r) += _elem2 * _elem2;
								// meanabs
								_offdiagElemesStat_low(_opi, 5, _r) += _elemabs;

								// add to value histogram
								_histOperatorsOffdiag_low[_opi].append(_elemreal);

								// save the values
								if(_totalIterator_l < _offdiag_elem_num)
									_offdiagElements_low(_opi, _r, _totalIterator_l) = _elemreal;

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
						_offdiagElemesStat_low(_opi, 6, _r)		= StatisticalMeasures::gaussianity(_offdiagElemesStat_low(_opi, 5, _r), _offdiagElemesStat_low(_opi, 2, _r));
						_offdiagElemesStat_low(_opi, 7, _r)		= StatisticalMeasures::binder_cumulant(_offdiagElemesStat_low(_opi, 2, _r), _offdiagElemesStat_low(_opi, 4, _r));
					}

					// additionally, for typical values, calculate the exponential of the mean
					{
						for (auto ii : { 1, 3 })
						{
							_offdiagElemesStat_low(_opi, ii, _r)= std::exp(_offdiagElemesStat_low(_opi, ii, _r));
						}
					}
				}
			}
		}
		
		// save the checkpoints
		if (check_saving_size(_Nh, _r))
			_saver(_r);
		LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
		// -----------------------------------------------------------------------------
	}

	// save the diagonals
	_saver(this->modP.modRanN_);

	// bye
	LOGINFO(_timer.start(), "ETH CALCULATOR", 0);
}

// ###############################################################################################

/*
* @brief Check the properties of the models complying to ETH based on the Hamiltonian provided.
* It saves the gap ratios, level statistcs, energies, and the operators.
* @param _H: Hamiltonian to check
*/
template<typename _T>
void UI::checkETH_statistics(std::shared_ptr<Hamiltonian<_T>> _H)
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

	// get info about the model
	std::string modelInfo, dir = "ETH_MAT_STAT", randomStr, extension;
	this->get_inf_dir_ext_r(_H, dir, modelInfo, randomStr, extension);

	// set the placeholder for the values to save (will save only the diagonal elements and other measures)
	arma::Mat<double> _en			= -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _entroHalf	= -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _entroFirst	= -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _entroLast	= -1e5 * arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::ones);

	// gap ratios
	v_1d<double> _gapsin			= v_1d<double>(_Nh - 2, 0.0);
	arma::Col<double> _gaps			= -1e5 * arma::Col<double>(this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _gapsall		= -1e5 * arma::Mat<double>(_Nh - 2, this->modP.modRanN_, arma::fill::ones);
	// mean lvl
	arma::Col<double> _meanlvl		= -1e5 * arma::Col<double>(this->modP.modRanN_, arma::fill::ones);

	// create the measurement class
	Measurement<double> _measure(this->latP.Ntot_, dir, _ops, _opsN, 1, _Nh);

	// to save the operators (those elements will be stored for each operator separately)
	// a given matrix element <n|O|n> will be stored in i'th column of the i'th operator
	// the n'th row in the column will be the state index
	// the columns corresponds to realizations of disorder
	VMAT<_T> _diagElems(_ops.size(), _Nh, this->modP.modRanN_, arma::fill::ones, -1e5);
	// constraint the offdiagonals also to _Nh elements only
	VMAT<_T> _offdiagElems(_ops.size(), _Nh, this->modP.modRanN_, arma::fill::ones, -1e5);
	VMAT<_T> _offdiagElemsLow(_ops.size(), _Nh, this->modP.modRanN_, arma::fill::ones, -1e5);
	arma::Mat<double> _offdiagElemsOmega = -1e5 * arma::Mat<double>(_offdiagElems.n_rows(0), this->modP.modRanN_, arma::fill::ones);
	arma::Mat<double> _offdiagElemsOmegaLow = -1e5 * arma::Mat<double>(_offdiagElemsLow.n_rows(0), this->modP.modRanN_, arma::fill::ones);

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
	uint _nbinOperators = std::min((size_t)(12 * _Ns), (size_t)300);
	v_1d<Histogram> _histOperatorsDiag(_ops.size(), Histogram(_nbinOperators));
	v_1d<Histogram> _histOperatorsOffdiag(_ops.size(), Histogram(_nbinOperators));

	// create the saving function
	std::function<void(uint)> _saver = [&](uint _r)
		{
			// save the iprs and the energy to the same file
			saveAlgebraic(dir, "stat" + randomStr + extension, _gaps, "gap_ratio", false);
			// variance in th Hamiltonian
			saveAlgebraic(dir, "stat" + randomStr + extension, _meanlvl, "mean_level_gamma", true);
			// energy
			saveAlgebraic(dir, "stat" + randomStr + extension, _en, "energy", true);
			// entropy
			saveAlgebraic(dir, "stat" + randomStr + extension, _entroHalf, "entropy_half", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _entroFirst, "entropy_first", true);
			saveAlgebraic(dir, "stat" + randomStr + extension, _entroLast, "entropy_last", true);
			// gap ratios
			saveAlgebraic(dir, "stat" + randomStr + extension, _gapsall, "gap_ratios", true);

			// diagonal operators saved (only append when _opi > 0)
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "diag" + randomStr + extension, algebra::cast<double>(_diagElems[_opi]), _name, _opi > 0);
			}

			// offdiagonal operators saved (only append when _opi > 0)
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "offdiag" + randomStr + extension, algebra::cast<double>(_offdiagElems[_opi]), _name, _opi > 0);
				saveAlgebraic(dir, "offdiag" + randomStr + extension, algebra::cast<double>(_offdiagElemsLow[_opi]), _name, true);
			}
			saveAlgebraic(dir, "offdiag" + randomStr + extension, _offdiagElemsOmega, "omega_high", true);
			saveAlgebraic(dir, "offdiag" + randomStr + extension, _offdiagElemsOmegaLow, "omega_low", true);

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
			for (uint _opi = 0; _opi < _ops.size(); ++_opi)
			{
				auto _name = _measure.getOpGN(_opi);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsDiag[_opi].edgesCol(), _name + "_diag_edges", _opi > 0);
				saveAlgebraic(dir, "dist" + randomStr + extension, _histOperatorsOffdiag[_opi].edgesCol(), _name + "_offdiag_edges", true);
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
			LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
			_timer.checkpoint(STR(_r));
			this->ui_eth_randomize(_H);
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);
		}

		// -----------------------------------------------------------------------------
				
		// get the average energy index and the points around it on the diagonal
		u64 _minIdxDiag = 0, _maxIdxDiag = 0, _minIdxDiag_cut = 0, _maxIdxDiag_cut = 0; 

		// set
		{
			std::tie(_minIdxDiag, _maxIdxDiag)			= _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
			std::tie(_minIdxDiag_cut, _maxIdxDiag_cut)	= _H->getEnArndAvIdx(_hs_fractions_diag_stat / 2, _hs_fractions_diag_stat / 2);
		}

		// -----------------------------------------------------------------------------
		
		{
			// -----------------------------------------------------------------------------

			// save the energies
			{
				_en.col(_r) = _H->getEigVal();
			}

			// -----------------------------------------------------------------------------

			// gap ratios
			{
				// calculate the eigenlevel statistics
				_gaps(_r) = SystemProperties::eigenlevel_statistics(arma::Col<double>(_H->getEigVal().subvec(_minIdxDiag, _maxIdxDiag - 1)));

				SystemProperties::eigenlevel_statistics(_H->getEigVal().begin(), _H->getEigVal().end(), _gapsin);
				_gapsall.col(_r) = arma::Col<double>(_gapsin);

				LOGINFO(StrParser::colorize(VEQ(_gaps(_r, 0)), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
				LOGINFO(_timer.point(STR(_r)), "Gap ratios", 1);
			}

			// -----------------------------------------------------------------------------

			// mean level spacing
			{
				_meanlvl(_r) = _H->getMeanLevelSpacing();
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(_r, 0)), StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
			}

			// -----------------------------------------------------------------------------
		}

		// -----------------------------------------------------------------------------

		// set the uniform distribution of frequencies in logspace for the f-functions!!!
		if (_r == 0)
		{

			double oMax			= std::abs(_H->getEigVal(_maxIdxDiag) - _H->getEigVal(_minIdxDiag)) * 2;
			double oMin			= _Nh <= UI_LIMITS_MAXFULLED ? 1.0 / _Nh : 1e-3;

			// set the histograms
			for (auto iHist = 0; iHist < _ops.size(); ++iHist)
			{
				_histAv[iHist].reset(_nbinOperators);
				_histAv[iHist].uniformLog(oMax, oMin);

				_histAvTypical[iHist].reset(_nbinOperators);
				_histAvTypical[iHist].uniformLog(oMax, oMin);
			}
		}

		// -----------------------------------------------------------------------------
		
		// calculator of the properties
		{

			// other measures
			{

				// entanglement
				if (_Ns <= std::log2(UI_LIMITS_MAXFULLED))
				{
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif				
					for(size_t _idx = 0; _idx < _Nh; ++_idx)
					{
						// get the entanglement
						const auto& _state					= _H->getEigVec(_idx);
						_entroHalf(_idx, _r)	= Entropy::Entanglement::Bipartite::vonNeuman<_T>(_state, uint(_Ns / 2), uint(_Ns));
						_entroFirst(_idx, _r)	= Entropy::Entanglement::Bipartite::vonNeuman<_T>(_state, 1, uint(_Ns));
						_entroLast(_idx, _r)	= Entropy::Entanglement::Bipartite::vonNeuman<_T>(_state, 1, uint(_Ns), 1, Entropy::Entanglement::Bipartite::RHO_METHODS::SCHMIDT, 2);
					}

				}

				// -----------------------------------------------------------------------------

				// all elements together
				{
					// get matrices
					const auto& _matrices	= _measure.getOpG_mat();
					const double _avEn		= _H->getEnAv();

					// go through the operators
#ifndef _DEBUG
#pragma omp parallel for num_threads(_Ns < 13 ? this->threadNum : 1)
#endif
					for (int _opi = 0; _opi < _matrices.size(); _opi++)
					{
						LOGINFO("Doing operator: " + _opsN[_opi], LOG_TYPES::TRACE, 2);
						arma::Mat<_T> _overlaps						= Operators::applyOverlapMat(_H->getEigVec(), _matrices[_opi]);

						// save the iterators
						u64 _totalIterator_off [[maybe_unused]]		= 0;
						u64 _totalIterator_off_low [[maybe_unused]] = 0;
						double _limlow	= std::sqrt(1.0 / _Nh * std::abs(_H->getEigVal(_maxIdxDiag) - _H->getEigVal(_minIdxDiag))) * 5;
						double _limhigh = 0.5;

						// go through the whole spectrum (do not save pairs, only one element as it's Hermitian conjugate.
						for (u64 i = 0; i < _Nh; ++i)
						{
							const auto _en_l = _H->getEigVal(i);

							// get diagonal statistics
							{
								const auto& _elem = _overlaps(i, i);

								if (i >= _minIdxDiag && i <= _maxIdxDiag)
								{

									const auto _elemabs		= std::abs(_elem);
									const auto _elemreal	= algebra::cast<double>(_elem);
									const auto _elem2		= _elemabs * _elemabs;
									const auto _logElem		= std::log(_elemabs);
									const auto _logElem2	= std::log(_elemabs * _elemabs);

									// save the statistics
									// mean
									_diagElemsStat[_opi](0, _r) += _elemreal;
									// typical
									_diagElemsStat[_opi](1, _r) += _logElem;
									// mean2
									_diagElemsStat[_opi](2, _r) += _elem2;
									// typical2
									_diagElemsStat[_opi](3, _r) += _logElem2;

									if (i >= _minIdxDiag_cut && i <= _maxIdxDiag_cut)
									{
										// mean
										_diagElemsStat_cut[_opi](0, _r) += _elemreal;
										// typical
										_diagElemsStat_cut[_opi](1, _r) += _logElem;
										// mean2
										_diagElemsStat_cut[_opi](2, _r) += _elem2;
										// typical2
										_diagElemsStat_cut[_opi](3, _r) += _logElem2;
									}
								}
							}

							for (u64 j = i + 1; j < _Nh; ++j)
							{
								const auto _en_r = _H->getEigVal(j);

								// check the energy difference
								if (!SystemProperties::hs_fraction_close_mean(_en_l, _en_r, _avEn, this->modP.modEnDiff_))
									continue;


								// calculate the frequency
								const double w			= std::abs(_en_l - _en_r);

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

									// mean
									_offdiagElemesStat[_opi](0, _r) += _elemreal;
									// typical
									_offdiagElemesStat[_opi](1, _r) += _logElem;
									// mean2
									_offdiagElemesStat[_opi](2, _r) += _elem2;
									// typical2
									_offdiagElemesStat[_opi](3, _r) += _logElem2;
									// mean4
									_offdiagElemesStat[_opi](4, _r) += _elem2 * _elem2;
									// meanabs
									_offdiagElemesStat[_opi](5, _r) += _elemabs;

									// add to the histograms
									_histAv[_opi].append(w, _elem2);
									_histAvTypical[_opi].append(w, _logElem2);

									// save the values
									if ((_totalIterator_off < _offdiagElems.n_rows(_opi) && w > _limhigh))
									{
										_totalIterator_off++;
										_offdiagElems.set(_opi, _totalIterator_off - 1, _r, _elemreal);
										if (_opi == 0)
											_offdiagElemsOmega(_totalIterator_off - 1, _r) = std::abs(w);
									}
									if (_totalIterator_off_low < _offdiagElemsLow.n_rows(_opi) && w <= _limlow)
									{
										_totalIterator_off_low++;
										_offdiagElemsLow.set(_opi, _totalIterator_off_low - 1, _r, _elemreal);
										if (_opi == 0)
											_offdiagElemsOmegaLow(_totalIterator_off_low - 1, _r) = std::abs(w);
									}
								}
							}
						}

						// ############## finalize statistics ##############

						LOGINFO("Finalizing statistics for operator: " + _opsN[_opi], LOG_TYPES::TRACE, 3);

						// diagonal
						//{
						//	{
						//		for (uint ii = 0; ii < 4; ii++)
						//		{
						//			_diagElemsStat[_opi](ii, _r)		/= _hs_fractions_diag;
						//			_diagElemsStat_cut[_opi](ii, _r)	/= _hs_fractions_diag_stat;
						//		}
						//		// save the statistics
						//		_diagElemsStat[_opi](4, _r) = StatisticalMeasures::gaussianity<_T>(_diagElems[_opi].col(_r));
						//		_diagElemsStat[_opi](5, _r) = StatisticalMeasures::kurtosis<_T>(_diagElems[_opi].col(_r));
						//		_diagElemsStat[_opi](6, _r) = StatisticalMeasures::binder_cumulant<_T>(_diagElems[_opi].col(_r));
						//		// save the statistics
						//		_diagElemsStat_cut[_opi](4, _r) = StatisticalMeasures::gaussianity<double>(_diagElemsStat_cut[_opi].col(_r));
						//		_diagElemsStat_cut[_opi](5, _r) = StatisticalMeasures::kurtosis<double>(_diagElemsStat_cut[_opi].col(_r));
						//		_diagElemsStat_cut[_opi](6, _r) = StatisticalMeasures::binder_cumulant<double>(_diagElemsStat_cut[_opi].col(_r));
						//	}

						//	// additionally, for typical values, calculate the exponential of the mean
						//	{
						//		_diagElemsStat[_opi](1, _r)		= std::exp(_diagElemsStat[_opi](1, _r));
						//		_diagElemsStat[_opi](3, _r)		= std::exp(_diagElemsStat[_opi](3, _r));
						//		_diagElemsStat_cut[_opi](1, _r) = std::exp(_diagElemsStat_cut[_opi](1, _r));
						//		_diagElemsStat_cut[_opi](3, _r) = std::exp(_diagElemsStat_cut[_opi](3, _r));
						//	}
						//}

						// LOGINFO("Finished the diagonal statistics for: " + _opsN[_opi], LOG_TYPES::TRACE, 3);

						// offdiagonal
						//{

						//	{
						//		for (uint ii = 0; ii < 6; ii++)
						//			_offdiagElemesStat[_opi](ii, _r) /= (long double)_totalIterator_off;

						//		// statistics
						//		_offdiagElemesStat[_opi](6, _r) = StatisticalMeasures::gaussianity(_offdiagElemesStat[_opi](5, _r), _offdiagElemesStat[_opi](2, _r));
						//		_offdiagElemesStat[_opi](7, _r) = StatisticalMeasures::binder_cumulant(_offdiagElemesStat[_opi](2, _r), _offdiagElemesStat[_opi](4, _r));
						//	}

						//	// additionally, for typical values, calculate the exponential of the mean
						//	{
						//		for (auto ii : { 1, 3 })
						//		{
						//			_offdiagElemesStat[_opi](ii, _r) = std::exp(_offdiagElemesStat[_opi](ii, _r));
						//		}
						//	}
						//}

						//LOGINFO("Finished the offdiagonal statistics for: " + _opsN[_opi], LOG_TYPES::TRACE, 3);

						// save the diagonal part
						_diagElems[_opi].col(_r) = _overlaps.diag();

						// save the histograms of the diagonals and offdiagonals!
						_histOperatorsDiag[_opi].setHistogramCounts(_diagElems[_opi].col(_r), _r == 0);
						_histOperatorsOffdiag[_opi].setHistogramCounts(_offdiagElems[_opi].col(_r), _r == 0);

					}

				}
			}
		}

		// save the checkpoints
		if (check_saving_size(_Nh, _r))
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
	isQuadratic				= false;
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
	arma::Mat<double> _meanlvl			= -1e5 * arma::Mat<double>(4, this->modP.modRanN_, arma::fill::ones);
	u64 _hs_fractions_diag				= SystemProperties::hs_fraction_diagonal_cut(0.5, _Nh);

	// time evolution saved here
	long double _heisenberg_time_est	= _Nh;
	arma::Col<double> _timespace		= arma::logspace(-2, std::log10(_heisenberg_time_est * 100), 2500);
	// create initial states for the quench
	arma::Col<_T> _initial_state_me		= arma::Col<_T>(_Nh, arma::fill::zeros);

	// saves the energies and the LDOSs of the initial states
	arma::Mat<double> _energies			= arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::zeros);
	// save the LDOS
	arma::Mat<double> _ldos_me			= arma::Mat<double>(_Nh, this->modP.modRanN_, arma::fill::zeros);
	// to save the energy densities (mean energy[system], mean energy[state], <state|H2|state>)
	arma::Mat<_T> _energydensitiesME(3, this->modP.modRanN_);
	// to save the diagonal elements
	VMAT<_T> _diagonals(_ops.size(), _Nh, this->modP.modRanN_, arma::fill::zeros);

	// save the time evolution here
	VMAT<_T> _timeEvolutionME(_ops.size(), _timespace.size(), this->modP.modRanN_, arma::fill::zeros);
	// entropies to take
	v_1d<int> _entropiesSites			= {1, int(_Ns / 2), _Ns - 1, _Ns};
	VMAT<double> _timeEntropyME(4, _timespace.size(), this->modP.modRanN_, arma::fill::zeros);
	arma::Mat<double> _timeEntropyBipartiteME(_timespace.size(), this->modP.modRanN_, arma::fill::zeros);
	v_1d<arma::Col<_T>> _timeZeroME(_ops.size(), arma::Col<_T>(this->modP.modRanN_, arma::fill::zeros));

	// ------------------------- MICROCANONICAL AVERAGES -------------------------

	const v_1d<double> _toCheckEps		= { 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4 };

	// to save the microcanonical averages
	arma::Mat<double> _diagonalME(_ops.size(), this->modP.modRanN_, arma::fill::zeros);

	// to save the diagonal ensemble averages (might be useful for comparison to the microcanonical averages)
	VMAT<_T> _microcanonicalME(_ops.size(), _toCheckEps.size(), this->modP.modRanN_, arma::fill::zeros);
	VMAT<double> _microcanonical2ME(_ops.size(), _toCheckEps.size(), this->modP.modRanN_, arma::fill::zeros);

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
							arma::Mat<double>& _ldoses,
							arma::Mat<_T>& _energydensities,
							VMAT<_T>& _microvals,
							VMAT<double>& _microvals2,
							arma::Mat<double>& _diagvals,
							VMAT<_T>& _timeEvolution,
							v_1d<arma::Col<_T>>& _timeZero,
							const v_1d<GeneralizedMatrix<double>>& _matrices)
		{
			// calculate the overlaps of the initial state with the eigenvectors 
			// (states are columns and vector is column as well, so we need to have the transpose)
			const auto& _eigvecs				= _H->getEigVec();
			const auto& _eigvals				= _H->getEigVal();
			const arma::Col<_T> _overlaps		= _eigvecs.st() * _initial_state;
			const arma::Col<double> _soverlaps	= arma::square(arma::abs(_overlaps));

			// calculate the average energy index
			double _meanE						= _H->getEnAv();

			// save the energies
			_ldoses.col(_r)						= SystemProperties::calculate_LDOS(_eigvals, _overlaps);
			_energydensities(0, _r)				= _meanE;

			// apply the Hamiltonian to the initial state
			const arma::Col<_T> _init_stat_H	= _H->getHamiltonian() * _initial_state;
			const auto _E						= (arma::cdot(_initial_state, _init_stat_H));
			const auto _E2						= (arma::cdot(_init_stat_H, _init_stat_H));
			u64 _Eidx							= _H->calcEnIdx(algebra::cast<double>(_E));
			LOGINFO(VEQ(_Eidx), LOG_TYPES::TRACE, 4);
			LOGINFO(VEQP(_E, 5), LOG_TYPES::TRACE, 4);
			LOGINFO(VEQP(_E2, 5), LOG_TYPES::TRACE, 4);
			_energydensities(1, _r)				= _E;
			_energydensities(2, _r)				= _E2;

			// calculate the bounds
			std::vector<u64> _mins(_toCheckEps.size()), _maxs(_toCheckEps.size());
			for (int _ifrac = 0; _ifrac < _toCheckEps.size(); _ifrac++)
				std::tie(_mins[_ifrac], _maxs[_ifrac]) = _H->getEnArndEnEps(_Eidx, _toCheckEps[_ifrac]);

			// microcanonical and diagonal ensembles
#pragma omp parallel for num_threads(this->threadNum)
			for (int _opi = 0; _opi < _ops.size(); ++_opi)
			{
				// ferromagnetic
				_microcanonical_saver(_r, _opi, _microvals, _microvals2, _diagvals, _soverlaps, _mins, _maxs);
			}

			// -----------------------------------------------------------------------------
			
			// save zero time value
			{
	#pragma omp parallel for num_threads(this->threadNum)
				for (uint _opi = 0; _opi < _ops.size(); ++_opi)
					_timeZero[_opi](_r) = arma::as_scalar(arma::cdot(_initial_state, (_matrices[_opi] * _initial_state)));
			}

			// evolution
#pragma omp parallel for num_threads(this->threadNum)
			for (int _ti = 0; _ti < _timespace.size(); _ti++)
			{
				const auto _time							= _timespace(_ti);
				const arma::Col<std::complex<double>> _st	= SystemProperties::TimeEvolution::time_evo(_eigvecs, _eigvals, _overlaps, _time);

				// for each operator we can now apply the expectation value		
				for (uint _opi = 0; _opi < _ops.size(); ++_opi)
				{
					const cpx _rt					= arma::as_scalar(arma::cdot(_st, (_matrices[_opi] * _st)));
					_timeEvolution[_opi](_ti, _r)	= algebra::cast<_T>(_rt);
				}

				// say the time
				if (_ti % 100 == 0)
					LOGINFO(VEQ(_ti) + "/" + STR(_timespace.size()), LOG_TYPES::TRACE, 3);

				// calculate the entanglement entropy for each site
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

	// go through realizations
	for (int _r = 0; _r < this->modP.modRanN_; ++_r)
	{
		// ----------------------------------------------------------------------------
		
		// checkpoints etc
		{
			// -----------------------------------------------------------------------------
			LOGINFO(VEQ(_r), LOG_TYPES::TRACE, 30, '#', 1);
			LOGINFO("Doing: " + STR(_r), LOG_TYPES::TRACE, 0);
			_timer.checkpoint(STR(_r));

			this->ui_eth_randomize(_H, _r);
			LOGINFO(_timer.point(STR(_r)), "Diagonalization", 1);

			// create the initial state
			_initial_state_me	= SystemProperties::TimeEvolution::create_initial_quench_state<_T>(SystemProperties::TimeEvolution::QuenchTypes::SEEK, _Nh, _Ns, _H->getEnAv(), _H->getDiag());
		}

		// -----------------------------------------------------------------------------

		// calculator of the properties
		{
			// -----------------------------------------------------------------------------

			// mean level spacing
			{
				long double _h_freq		= 1.0 / _Nh;
				// energies
				_energies.col(_r)	= _H->getEigVal();
				// mean levels (gamma, heisenberg)
				_meanlvl(0, _r)		= _H->getMeanLevelSpacing();
				_meanlvl(1, _r)		= SystemProperties::mean_lvl_heis_time(_meanlvl(0, _r), _Ns);

				// get the average energy index and the points around it on the diagonal
				const auto [_min, _max] = _H->getEnArndAvIdx(_hs_fractions_diag / 2, _hs_fractions_diag / 2);
				auto _E					= _H->getEigVal().subvec(_min, _max);
				_meanlvl(2, _r)			= 1.0 / SystemProperties::mean_lvl_spacing(_E);
				// get the Heisenberg frequency from the mean level spacing
				_h_freq					= SystemProperties::mean_lvl_spacing_typ(_E);
				_meanlvl(3, _r)			= 1.0 / _h_freq;
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(0, _r)) + ": mean level spacing", StrParser::StrColors::green), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(1, _r)) + ": mean level Heisenberg time", StrParser::StrColors::blue), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(2, _r)) + ": mean level spacing around energy " + VEQP(_min, 3), StrParser::StrColors::red), LOG_TYPES::TRACE, 1);
				LOGINFO(StrParser::colorize(VEQ(_meanlvl(3, _r)) + ": mean level spacing aroung energy - typical", StrParser::StrColors::yellow), LOG_TYPES::TRACE, 1);
			}

			// -----------------------------------------------------------------------------

			// other measures
			{
				// calculate the diagonals
				const auto& _matrices				= _measure.getOpG_mat();
				const auto& _eigvec					= _H->getEigVec();

#pragma omp parallel for num_threads(_Ns < 14 ? this->threadNum : 2)
				for (int _opi = 0; _opi < _ops.size(); ++_opi)
				{
					_diagonals[_opi].col(_r) = Operators::applyOverlapMat(_eigvec, _matrices[_opi]).diag();
				}

				// evolve the states
				_evolveState(_r, _initial_state_me, _ldos_me, _energydensitiesME,  
					_microcanonicalME, _microcanonical2ME, _diagonalME, _timeEvolutionME, _timeZeroME, _matrices);
			}
		}
		// save the checkpoints
		if (check_saving_size(_Nh, _r))
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

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%% DEFINE THE TEMPLATES %%%%%%%%%%%%%%%%%%%%%
template void UI::checkETH_scaling_offdiag<double>(std::shared_ptr<Hamiltonian<double>> _H);
template void UI::checkETH_scaling_offdiag<cpx>(std::shared_ptr<Hamiltonian<cpx>> _H);

template void UI::checkETH_statistics<double>(std::shared_ptr<Hamiltonian<double>> _H);
template void UI::checkETH_statistics<cpx>(std::shared_ptr<Hamiltonian<cpx>> _H);

template void UI::checkETH_time_evo<double>(std::shared_ptr<Hamiltonian<double>> _H);
template void UI::checkETH_time_evo<cpx>(std::shared_ptr<Hamiltonian<cpx>> _H);