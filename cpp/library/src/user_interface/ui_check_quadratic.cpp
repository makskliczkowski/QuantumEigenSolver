#include "../../include/user_interface/user_interface.h"

// ##########################################################################################################################################

// ########################################################### Q U A D R A T I C ############################################################

// ##########################################################################################################################################

template<typename _T>
void UI::quadraticStatesManifold(std::shared_ptr<QuadraticHamiltonian<_T>> _H)
{
	uint Ns					= _H->getNs();
	uint Nh					= _H->getHilbertSize();
	uint _type				= _H->getTypeI();

	// --- create the directories ---
	bool _manifold			= (_type == (uint)MY_MODELS::FREE_FERMIONS_M) && this->modP.q_manifold_;
	std::string str0		= "QuadraticEntropies" + std::string(this->modP.q_shuffle_ ? "S" : "") + std::string((_manifold) ? "Manifold" : "");
	std::string dir			= makeDirsC(this->mainDir, _H->getType(), this->latP.lat->get_info(), str0);

	// ------ use those files -------
	std::string modelInfo	= _H->getInfo();
	// how many states to take for calculating the entropies
	u64 _gamma				= this->modP.q_gamma_;
	// how many states to take for the average (realizations of the combinations N * \Gamma states)
	u64 _realizations		= this->modP.q_realizationNum_;
	// number of combinations to take from single particle states (is \Gamma)
	u64 _combinations		= this->modP.q_randomCombNum_;

	// --- save energies txt check ---
	std::string filename	= filenameQuadraticRandom(dir + modelInfo + VEQV(_R, _realizations) + VEQV(_C, _combinations) +
							  VEQV(_Gamma, _gamma), _type, _H->ran_);

	IF_EXCEPTION(_combinations < _gamma, "Bad number of combinations. Must be bigger than the number of states");

	// check the model (if necessery to build hamilonian, do it)
	if (_type != (uint)MY_MODELS::FREE_FERMIONS_M)
	{
		_H->buildHamiltonian();
		_H->diagH(false);
		LOGINFO(_timer.start(), "Diagonalization", 3);
	}

	// go through the information
	LOGINFO("Spectrum size:						  " + STR(Nh), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking states to mix (Gamma):		  " + STR(_gamma), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num states (combinations):	  " + STR(_combinations), LOG_TYPES::TRACE, 3);
	LOGINFO("Taking num realizations (averages):  " + STR(_realizations), LOG_TYPES::TRACE, 3);

	// save single particle energies
	if (!fs::exists(filename + ".h5"))
		_H->getEigVal(dir, HAM_SAVE_EXT::h5, false);

	// iterate through bond cut
	uint _bonds				= uint(Ns / 2);
	arma::mat ENERGIES(_realizations, _gamma, arma::fill::zeros);
	arma::vec ENTROPIES_SP(_realizations, arma::fill::zeros);
	arma::vec ENTROPIES_MB(_realizations, arma::fill::zeros);

	// many body orbitals (constructed of vectors of indices of single particle states)
	v_1d<double> energies;
	v_2d<uint> orbs;

	// single particle orbital indices (all posibilities to choose from in the combination)
	v_1d<uint> _SPOrbitals  = Vectors::vecAtoB<uint>(Ns);
	
	// get the states to use later
	_timer.checkpoint("combinations");
	
	// calculate many body orbitals to be used
	if (Ns <= UI_LIMITS_QUADRATIC_COMBINATIONS)
		_H->getManyBodyOrbitals(Ns / 2, _SPOrbitals, orbs);
	else
		_H->getManyBodyOrbitals(Ns / 2, _SPOrbitals, orbs, _combinations, this->threadNum);
	if(this->modP.q_shuffle_)
		std::shuffle(orbs.begin(), orbs.end(), this->ran_.eng());
	// obtain the energies
	_H->getManyBodyEnergies(energies, orbs, this->threadNum);

		// obtain the single particle energies
	arma::Mat<_T> W			= _H->getTransMat();
	// make matrices cut to a specific number of bonds
	arma::Mat<_T> Ws		= W.submat(0, 0, W.n_rows - 1, _bonds - 1);
	// conjugate transpose it - to be used later
	arma::Mat<_T> WsC		= Ws.t();
	// Hilbert space
	auto _hilbert			= Hilbert::HilbertSpace<_T>(this->latP.lat);
	auto _Nh			    = _hilbert.getHilbertSize();
	LOGINFO(_timer.point("combinations"), "Combinations time:", 3);

	// -------------------------------- CORRELATION --------------------------------
	auto _appendEntroSP = [&](u64 _idx, const std::vector<std::vector<uint>>& _orbitals, arma::Col<_T>& _coeff)
		{
			// iterate through the state
			auto J				= SingleParticle::CorrelationMatrix::corrMatrix(Ns, Ws, WsC, _orbitals, _coeff, this->ran_);
			ENTROPIES_SP(_idx)	= Entropy::Entanglement::Bipartite::SingleParticle::vonNeuman<cpx>(J);
		};

	// --------------------------------- MANY BODY ---------------------------------
	auto _appendEntroMB = [&](u64 _idx, const std::vector<std::vector<uint>>& _orbitals, arma::Col<_T>& _coeff)
		{
			// use the slater matrix to obtain the many body state
			arma::Mat<_T> _slater(Ns / 2, Ns / 2, arma::fill::zeros);
			arma::Col<cpx> _state(_Nh, arma::fill::zeros);
			for (int i = 0; i < _orbitals.size(); ++i)
				_state += _coeff(i) * _H->getManyBodyState(_orbitals[i], _hilbert, _slater);
			ENTROPIES_MB(_idx) = Entropy::Entanglement::Bipartite::vonNeuman<cpx>(_state, _bonds, _hilbert);;
		};
	// ----------------------------------- SAVER -----------------------------------
	auto _saveEntro = [&](bool _save)
		{
			if (_save)
			{
#ifndef _DEBUG
#	pragma omp critical
#endif
				{
					LOGINFO("Checkpoint", LOG_TYPES::TRACE, 4);
					ENTROPIES_SP.save(arma::hdf5_name(filename + "_SP.h5", "entropy"));
					ENERGIES.save(arma::hdf5_name(filename + "_EN.h5", "energy"));
					if (this->modP.q_manybody_)
						ENTROPIES_MB.save(arma::hdf5_name(filename + "_MB.h5", "entropy"));
				}
			}
		};
	// ------------------------------------ MAIN -----------------------------------
	_timer.checkpoint("entropy");
	pBar pbar(5, _realizations, _timer.point(0));

	// ------------ MANIFOLDS ------------
	// check if one wants to create a combinations at degenerate manifolds
	if (_manifold && _gamma != 1)
	{
		// zip the energies and orbitals together
		auto _zippedEnergies = Containers::zip(energies, orbs);

		// get map with frequencies of specific energies
		auto _frequencies = Vectors::freq<10>(energies, _gamma - 1);
		if (_frequencies.size() == 0)
			throw std::runtime_error("Bad number of frequencies. Must be bigger than $\\Gamma$.");

		// remove the _zippedEnergies based on existing in this map to get the manifolds
		std::erase_if(_zippedEnergies, [&](const auto& elem)
			{
				auto const& [_en, _vec] = elem;
				return !_frequencies.contains(Math::trunc<double, 10>(_en));
			});

		// sort the zipped energies please
		Containers::sort<0>(_zippedEnergies, [](const auto& a, const auto& b) { return a < b; });

		// go through the realizations
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum)
#endif
		for (long long idx = 0; idx < _realizations; idx++)
		{
			// generate coefficients (create random state consisting of stateNum = \Gamma states)
			auto coeff			= this->ran_.createRanState<_T>(_gamma);

			// get the random state
			long long idxState	= this->ran_.randomInt(0, _zippedEnergies.size() - _gamma);

			// while we cannot take _gamma states out of it, lower the index
			long long _iter		= 1;
			const auto E_idx	= Math::trunc<double, 10>(std::get<0>(_zippedEnergies[idxState]));

			while (((idxState - _iter) > 0) && _iter <= _gamma + 1)
			{
				const auto E_in	= Math::trunc<double, 10>(std::get<0>(_zippedEnergies[idxState - _iter]));
				if (E_in != E_idx)
					break;
				_iter++;
			}
			idxState -= (_iter - 1);

			// take the chosen orbitals
			std::vector<std::vector<uint>> orbitals;
			for (uint i = idxState; i < idxState + _gamma; ++i)
			{
				orbitals.push_back(std::get<1>(_zippedEnergies[i]));
				ENERGIES(idx, i - idxState) = std::get<0>(_zippedEnergies[i]);
			}

			// SP
			_appendEntroSP(idx, orbitals, coeff);
			// MB
			if(this->modP.q_manybody_)
				_appendEntroMB(idx, orbitals, coeff);
			// update progress (checkpoint for saving the entropies)
			PROGRESS_UPD_DO(idx, pbar, "PROGRESS", _saveEntro(idx % pbar.percentageSteps == 0));
		}
	}
	else
	{
		// go through the realizations
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum)
#endif
		for (long long idx = 0; idx < _realizations; idx++)
		{
			// generate coefficients (create random state consisting of stateNum = \Gamma states)
			auto coeff		= this->ran_.createRanState<_T>(_gamma);
			// get the random state
			auto idxState	= this->ran_.randomInt<uint>(0, energies.size() - _gamma);

			// take the chosen orbitals
			std::vector<std::vector<uint>> orbitals;
			for (uint i = idxState; i < idxState + _gamma; ++i)
			{
				orbitals.push_back(orbs[i]);
				ENERGIES(idx, i - idxState) = energies[i];
			}
			// SP
			_appendEntroSP(idx, orbitals, coeff);
			// MB
			if (this->modP.q_manybody_)
				_appendEntroMB(idx, orbitals, coeff);

			// update progress (checkpoint for saving the entropies)
			PROGRESS_UPD_DO(idx, pbar, "PROGRESS", _saveEntro(idx % pbar.percentageSteps == 0));
		}
	}
	
	// save in the end ;)
	_saveEntro(true);

	// save entropies
	LOGINFO("Finished entropies! " + VEQ(_gamma) + ", " + VEQ(_realizations), LOG_TYPES::TRACE, 2);
	LOGINFO(_timer.point("entropy"), "Entropies time:", 3);
}

template<typename _T>
void UI::quadraticSpectralFunction(std::shared_ptr<QuadraticHamiltonian<_T>> _H)
{
	uint Ns					= _H->getNs();
	//uint Nh					= _H->getHilbertSize();
	uint _type				= _H->getTypeI();

	// --- create the directories ---
	std::string dir			= makeDirsC(this->mainDir, _H->getType(), this->latP.lat->get_info(), "QuadraticSpectral");

	// ------ use those files -------
	std::string modelInfo	= _H->getInfo();

	// --- save energies txt check ---
	std::string filename	= filenameQuadraticRandom(modelInfo, _type, _H->ran_);

	// check the model (if necessery to build hamilonian, do it)
	
	{
		_H->buildHamiltonian();
		_H->diagH(false);
		LOGINFO(_timer.start(), "Diagonalization", 3);
	}

	// save single particle energies
	if (!fs::exists(filename + ".h5"))
		_H->getEigVal(dir, HAM_SAVE_EXT::h5, false);

	arma::Col<double> _D	= _H->getEigVal();
	arma::Mat<_T> _Hmat		= arma::Mat<_T>(_H->getHamiltonian());
	arma::Mat<_T> _Umat		= _H->getEigVec();
	arma::Mat<cpx>_Umatcpx  = algebra::cast<cpx>(_Umat);

	// _minimal energy
	uint _oNum		= 1000;
	double _Egs		= -10;//2 * _D(0);
	double _Ees		= 10;//2 * _D(Ns - 1);
	auto _omegas	= arma::linspace(_Egs, _Ees, arma::uword(_oNum));
	auto _dw		= _omegas(1) - _omegas(0);

	// calculate the DFT matrix
	std::shared_ptr<Lattice> _lat = _H->getLat();
	auto _exps		= _lat->calculate_dft_vectors();
	auto _expst		= _exps.t();
	//_lat->calculate_dft_matrix_vec(true);

	// output
	arma::Mat<double> _outspectrals(_omegas.n_elem, Ns, arma::fill::zeros);
	arma::Mat<double> _outspectrals_r(_omegas.n_elem, Ns, arma::fill::zeros);
	arma::Col<double> _Dos_r(_omegas.n_elem, arma::fill::zeros);
	arma::Col<double> _Dos_k(_omegas.n_elem, arma::fill::zeros);


	// go through omegas
#pragma omp parallel for num_threads(this->threadNum)
	for(int _omega = 0; _omega < _omegas.size(); ++_omega)
	{
		auto _tstart = NOW;
		//auto _spectr = SystemProperties::Spectral::Noninteracting::time_resolved_greens_function(_omegas(_omega), _Hmat, this->modP.q_broad_);
		auto _spectr = SystemProperties::Spectral::Noninteracting::time_resolved_greens_function(_omegas(_omega), _D, _Umatcpx, this->modP.q_broad_);
		if (_omega % 10 == 0)
			LOGINFO(STRP(DURATIONMS(NOW, _tstart), 6) + "ms - Time for omega (exact - SF): " + STR(_omega) + "/" + STR(_omegas.size()), LOG_TYPES::CHOICE, 3);

		//LOGINFO(STRP(DURATIONMS(_tstart, NOW), 6) + "Time for omega (not exact): " + STR(_omega) + "/" + STR(_omegas.size()), LOG_TYPES::CHOICE, 3);

		// calculate the total matrix
		arma::Col<cpx> _spectkdft	= SystemProperties::function_fourier_diag_k(_spectr, _exps, _expst);

		if (_omega % 10 == 0)
			LOGINFO(STRP(DURATIONMS(NOW, _tstart), 6) + "ms - Fourier transform time for omega (dft - SF): " + STR(_omega) + "/" + STR(_omegas.size()), LOG_TYPES::CHOICE, 4);

		auto _spect_k				= SystemProperties::Spectral::Noninteracting::spectral_function(_spectkdft);
		auto _spect_r				= SystemProperties::Spectral::Noninteracting::spectral_function(_spectr);
		// calculate the k-space matrix

		// save the first row
		_outspectrals.row(_omega)	= _spect_k.as_row();
		_outspectrals_r.row(_omega) = _spect_r.diag().as_row();
		_Dos_r(_omega)				+= arma::trace(_spect_r);
		_Dos_k(_omega)				+= arma::as_scalar(arma::sum(_spect_k));
		
		// printer
		if (_omega % 10 == 0)
			LOGINFO(_tstart, "Time for omega: " + STR(_omega) + "/" + STR(_omegas.size()), 2);
	}

	// save me!
	{
		// normalize the DOS from the spectral function
		{
			double _integral	= arma::as_scalar(arma::trapz(_omegas, _Dos_r));
			_Dos_r				/= _integral;
			_integral			= arma::as_scalar(arma::trapz(_omegas, _Dos_k));
			_Dos_k				/= _integral;
		}

		// normalize the spectral functions
		for (int k = 0; k < Ns; ++k)
		{
			double _integral = arma::as_scalar(arma::trapz(_omegas, _outspectrals.col(k)));
			_outspectrals.col(k) /= _integral;
		}

		saveAlgebraic(dir, filename + "_spectral.h5", _outspectrals, "spectrals_k", false);
		saveAlgebraic(dir, filename + "_spectral.h5", _outspectrals_r, "spectrals_r", true);
		saveAlgebraic(dir, filename + "_spectral.h5", _Dos_r, "dos_r", true);
		saveAlgebraic(dir, filename + "_spectral.h5", _Dos_k, "dos_k", true);
		saveAlgebraic(dir, filename + "_spectral.h5", _omegas, "omegas", true);

		// eDOS from the energies
		{
			auto _Dos			= SystemProperties::Spectral::Noninteracting::dos_gauss(_omegas, _D, 1e-1);
			double _integral	= arma::as_scalar(arma::trapz(_omegas, _Dos));
			_Dos				/= _integral;
			saveAlgebraic(dir, filename + "_spectral.h5", _Dos, "edos", true);
		}

	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%% DEFINE THE TEMPLATES %%%%%%%%%%%%%%%%%%%%%
//template void UI::quadraticStatesManifold<double>(std::shared_ptr<QuadraticHamiltonian<double>> _H);
template void UI::quadraticStatesManifold<cpx>(std::shared_ptr<QuadraticHamiltonian<cpx>> _H);

template void UI::quadraticSpectralFunction<double>(std::shared_ptr<QuadraticHamiltonian<double>> _H);
template void UI::quadraticSpectralFunction<cpx>(std::shared_ptr<QuadraticHamiltonian<cpx>> _H);