#include "../../include/user_interface/user_interface.h"
#include "armadillo"
#include <memory>
#include <stdexcept>

// ##########################################################################################################################################

constexpr int UI_NQS_PRECISION = 6;

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
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

template <typename _T>
using v_sp_t_NQS_g = v_sp_t<Operators::OperatorNQS<_T>>;
template <typename _T>
using v_sp_t_NQS_l = v_sp_t<Operators::OperatorNQS<_T, uint>>;
template <typename _T>
using v_sp_t_NQS_c = v_sp_t<Operators::OperatorNQS<_T, uint, uint>>;

template <typename _T>
static void get_NQS_operators_test(	v_sp_t<Operators::OperatorNQS<_T>>& _opsG, 
									v_sp_t<Operators::OperatorNQS<_T, uint>>& _opsL, 
									v_sp_t<Operators::OperatorNQS<_T, uint, uint>>& _opsC,
									uint _Nvis,
									std::shared_ptr<Lattice>& _lat)
{
	// clear the operators
	_opsC.clear();
	_opsL.clear();
	_opsG.clear();
	// set up the operators to save - local	
	Operators::Operator<_T, uint> _SzL 			= Operators::SpinOperators::sig_z_l<_T>(_Nvis);
	_opsL.push_back(std::make_shared<Operators::OperatorNQS<_T, uint>>(std::move(_SzL)));
	Operators::Operator<_T, uint> _SxL 			= Operators::SpinOperators::sig_x_l<_T>(_Nvis);
	_opsL.push_back(std::make_shared<Operators::OperatorNQS<_T, uint>>(std::move(_SxL)));
	// set up the operators to save - correlation
	Operators::Operator<_T, uint, uint> _SzC 	= Operators::SpinOperators::sig_z_c<_T>(_Nvis);
	_opsC.push_back(std::make_shared<Operators::OperatorNQS<_T, uint, uint>>(std::move(_SzC)));
	Operators::Operator<_T, uint, uint> _SyC 	= Operators::SpinOperators::sig_y_c<_T>(_Nvis);
	_opsC.push_back(std::make_shared<Operators::OperatorNQS<_T, uint, uint>>(std::move(_SyC)));
	Operators::Operator<_T, uint, uint> _SxC 	= Operators::SpinOperators::sig_x_c<_T>(_Nvis);
	_opsC.push_back(std::make_shared<Operators::OperatorNQS<_T, uint, uint>>(std::move(_SxC)));
	// special flux operator - as global
	if (_lat && (_lat->get_Type() == LatticeTypes::HON ||  _lat->get_Type() == LatticeTypes::HEX) && _Nvis >= 10)
	{
		Operators::Operator<_T> _flux 			= Operators::SpinOperators::Flux::sig_f<_T>(_Nvis, _lat->get_flux_sites(1, 0));
		_opsG.push_back(std::make_shared<Operators::OperatorNQS<_T>>(std::move(_flux)));
	}
}
// template instantiation
template void get_NQS_operators_test<double>(v_sp_t<Operators::OperatorNQS<double>>& _opsG, 
										v_sp_t<Operators::OperatorNQS<double, uint>>& _opsL, 
										v_sp_t<Operators::OperatorNQS<double, uint, uint>>& _opsC,
										uint _Nvis,
										std::shared_ptr<Lattice>& _lat);
template void get_NQS_operators_test<cpx>(v_sp_t<Operators::OperatorNQS<cpx>>& _opsG,
										v_sp_t<Operators::OperatorNQS<cpx, uint>>& _opsL,
										v_sp_t<Operators::OperatorNQS<cpx, uint, uint>>& _opsC,
										uint _Nvis,
										std::shared_ptr<Lattice>& _lat);	

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
* @brief Perform time evolution using exact diagonalization (ED) for a given state.
* 
* This function applies a quench operator to an initial state, normalizes it, and then
* performs time evolution using the eigenvectors and eigenvalues of the Hamiltonian.
* The time evolution of the operator is measured at each time step and the results
* are saved to a file.
* 
* @tparam _T Type of the elements in the state vector and Hamiltonian.
* @tparam _spinModes Number of spin modes (default is 2).
* @param _mbs Initial state vector.
* @param _H Pointer to the Hamiltonian object.
* @param _QMat Sparse matrix representing the quench operator.
* @param _QOMat Vector of sparse matrices representing the operators to measure.
* @param _timespace Vector of time points at which to measure the evolution.
* @param dir Directory where the results will be saved.
* @param stateIndex Index of the state being evolved.
* @param _quenchOpMeasure Vector of quench operators to measure.
* @param method String representing the method used for the evolution.
* @param threadNum Number of threads to use for parallel computation.
*/
template<typename _T, uint _spinModes = 2>
void nqs_perform_time_evo_ed(const arma::Col<_T>& _mbs,
						Hamiltonian<_T, _spinModes>* _H, 
						arma::SpMat<_T>& _QMat, v_1d<arma::SpMat<_T>>& _QOMat, 
						const arma::vec& _timespace, 
						const std::string& dir, 
						int stateIndex, 
						const v_sp_t<Operators::OperatorComb<_T>>& _quenchOpMeasure, 
						const std::string& method, 
						int threadNum) 
{
	arma::Mat<double> _vals(_timespace.size() + 1, _QOMat.size(), arma::fill::zeros);

	arma::Col<_T> _mbsIn 		= _QMat * _H->getEigVec(stateIndex); 	// apply the quench operator to the state (i'th)
	_mbsIn 						= _mbsIn / arma::norm(_mbsIn); 	 		// normalize the state - be sure
	const arma::Mat<_T>& _eigv 	= _H->getEigVec();  					// get the eigenvectors
	arma::Col<_T> _ovrl 		= _eigv.t() * _mbsIn; 	 				// overlap with the initial state

	// measure the time evolution of the operator - first time step
	for (int j = 0; j < _QOMat.size(); ++j)
		_vals(0, j) = algebra::cast<double>(Operators::applyOverlap(_mbsIn, _QOMat[j]));
	
	// measure the time evolution of the operator - other time steps
#pragma omp parallel for num_threads(threadNum)
	for (int j = 0; j < _timespace.size(); ++j) 
	{
		if (j % int(_timespace.size() / 10) == 0)
			LOGINFO(std::format("Time evolution step: {} of {}", j, _timespace.size()), LOG_TYPES::INFO, 3);

		arma::Col<cpx> _mb_te = SystemProperties::TimeEvolution::time_evo(_eigv, _H->getEigVal(), _ovrl, _timespace(j));
		for (int k = 0; k < _QOMat.size(); ++k)
			_vals(j + 1, k) = algebra::cast<double>(Operators::applyOverlap(_mb_te, _QOMat[k]));
	}

	// save the results
	for (int j = 0; j < _QOMat.size(); ++j)
		saveAlgebraic(dir, "measurement.h5", _vals.col(j), std::format("{}/{}/time_evo/{}", method, stateIndex, _quenchOpMeasure[j]->getNameS()), true);
}
// template instantiation
template void nqs_perform_time_evo_ed<double>(const arma::Col<double>& _mbs,
										Hamiltonian<double, 2>* _H,
										arma::SpMat<double>& _QMat, v_1d<arma::SpMat<double>>& _QOMat,
										const arma::vec& _timespace,
										const std::string& dir,
										int stateIndex,
										const v_sp_t<Operators::OperatorComb<double>>& _quenchOpMeasure,
										const std::string& method,
										int threadNum);
template void nqs_perform_time_evo_ed<cpx>(const arma::Col<cpx>& _mbs,
										Hamiltonian<cpx, 2>* _H,
										arma::SpMat<cpx>& _QMat, v_1d<arma::SpMat<cpx>>& _QOMat,
										const arma::vec& _timespace,
										const std::string& dir,
										int stateIndex,
										const v_sp_t<Operators::OperatorComb<cpx>>& _quenchOpMeasure,
										const std::string& method,
										int threadNum);

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
* @brief Perform full exact diagonalization (ED) on a given Hamiltonian and measure the results.
* 
* @tparam _T The data type used for the Hamiltonian and other matrices (e.g., double, complex).
* @tparam _spinModes The number of spin modes (default is 2).
* @param stateNum The number of states to be considered for diagonalization.
* @param _H Pointer to the Hamiltonian object.
* @param _hilbert The Hilbert space associated with the Hamiltonian.
* @param _quenchOp Shared pointer to the quench operator.
* @param _quenchOpMeasure Vector of shared pointers to the quench operators for measurement.
* @param _QMat Sparse matrix for quench operations.
* @param _QOMat Vector of sparse matrices for quench operations.
* @param _timespace Vector of time points for time evolution.
* @param dir Directory path for saving results.
* @param _meas_ED Vector of measurement objects for ED.
* @param _meansED Column vector to store the mean values of the ED results.
* @param time_evo Boolean flag indicating whether to perform time evolution.
* @param _saved Boolean reference indicating whether the results have been saved.
* @param threadNum The number of threads to be used for parallel processing.
*/
template<typename _T, uint _spinModes = 2>
void nqs_perform_full_ed(int stateNum, Hamiltonian<_T, _spinModes>* _H, 
						const Hilbert::HilbertSpace<_T, _spinModes>& _hilbert,
                        const std::shared_ptr<Operators::OperatorComb<_T>>& _quenchOp,
                        const std::vector<std::shared_ptr<Operators::OperatorComb<_T>>>& _quenchOpMeasure,
                        arma::SpMat<_T>& _QMat, v_1d<arma::SpMat<_T>>& _QOMat, 
                        const arma::vec& _timespace, 
						const std::string& dir,
                        v_1d<NQSAv::MeasurementNQS<_T>>& _meas_ED,
                        arma::Col<_T>& _meansED, 
						bool time_evo, 
						bool& _saved, 
						int threadNum) 
{
    LOGINFO("Started full diagonalization", LOG_TYPES::TRACE, 3);
    _H->diagH(false);
    for (int i = 0; i < stateNum; ++i) {
        const arma::Col<_T>& _mbs = _H->getEigVec(i);
        _meansED(i) = _H->getEigVal(i);
        LOGINFO(std::format("Found ED (full) state({}) with E={}", i, STRPS(_meansED[i], UI_NQS_PRECISION)), LOG_TYPES::INFO, 2);

        // Measure and save
        _meas_ED[i].measure(_mbs, _hilbert);
        _meas_ED[i].saveMB({".h5"}, "measurement", "measurement", "measurement", "ED/" + STR(i), i > 0 || time_evo);

        if (time_evo) 
			nqs_perform_time_evo_ed(_mbs, _H, _QMat, _QOMat, _timespace, dir, i, _quenchOpMeasure, "ED", threadNum);
	}
	saveAlgebraic(dir, "history.h5", _meansED, "ED/energy", _saved);
	_saved = true;
}
// template instantiation
template void nqs_perform_full_ed<double, 2>(int stateNum, Hamiltonian<double, 2>* _H, 
										const Hilbert::HilbertSpace<double, 2>& _hilbert,
										const std::shared_ptr<Operators::OperatorComb<double>>& _quenchOp,
										const std::vector<std::shared_ptr<Operators::OperatorComb<double>>>& _quenchOpMeasure,
										arma::SpMat<double>& _QMat, v_1d<arma::SpMat<double>>& _QOMat,
										const arma::vec& _timespace,
										const std::string& dir,
										v_1d<NQSAv::MeasurementNQS<double>>& _meas_ED,
										arma::Col<double>& _meansED,
										bool time_evo,
										bool& _saved,
										int threadNum);
template void nqs_perform_full_ed<cpx, 2>(int stateNum, Hamiltonian<cpx, 2>* _H,
										const Hilbert::HilbertSpace<cpx, 2>& _hilbert,
										const std::shared_ptr<Operators::OperatorComb<cpx>>& _quenchOp,
										const std::vector<std::shared_ptr<Operators::OperatorComb<cpx>>>& _quenchOpMeasure,
										arma::SpMat<cpx>& _QMat, v_1d<arma::SpMat<cpx>>& _QOMat,
										const arma::vec& _timespace,
										const std::string& dir,
										v_1d<NQSAv::MeasurementNQS<cpx>>& _meas_ED,
										arma::Col<cpx>& _meansED,
										bool time_evo,
										bool& _saved,
										int threadNum);

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
* @brief Perform Lanczos exact diagonalization (ED) and optionally time evolution.
* 
* This function performs Lanczos diagonalization on the given Hamiltonian to find the eigenvalues and eigenvectors.
* It then measures and saves the results, and optionally performs time evolution.
* 
* @tparam _T The data type for the Hamiltonian and operators.
* @tparam _spinModes The number of spin modes (default is 2).
* @param stateNum The number of states to compute.
* @param _H Pointer to the Hamiltonian object.
* @param _hilbert The Hilbert space object.
* @param _quenchOp Shared pointer to the quench operator.
* @param _quenchOpMeasure Vector of shared pointers to the quench operators for measurement.
* @param _QMat Sparse matrix for the quench operator.
* @param _QOMat Vector of sparse matrices for the quench operators.
* @param _timespace Vector of time points for time evolution.
* @param dir Directory to save the results.
* @param _meas_LAN Vector of measurement objects for the Lanczos method.
* @param _meansLAN Column vector to store the mean values of the Lanczos method.
* @param time_evo Boolean flag to indicate if time evolution should be performed.
* @param _saved Boolean reference to indicate if the results have been saved.
* @param threadNum The number of threads to use for parallel processing.
*/
template<typename _T, uint _spinModes = 2>
void nqs_perform_lanczos_ed(int stateNum, Hamiltonian<_T, _spinModes>* _H, 
							const Hilbert::HilbertSpace<_T, _spinModes>& _hilbert,
							const std::shared_ptr<Operators::OperatorComb<_T>>& _quenchOp,
							const std::vector<std::shared_ptr<Operators::OperatorComb<_T>>>& _quenchOpMeasure,
							arma::SpMat<_T>& _QMat, v_1d<arma::SpMat<_T>>& _QOMat, 
							const arma::vec& _timespace, 
							const std::string& dir,
							v_1d<NQSAv::MeasurementNQS<_T>>& _meas_LAN,
							arma::Col<_T>& _meansLAN, 
							bool time_evo, 
							bool& _saved, 
							int threadNum) 
{
	LOGINFO("Started Lanczos diagonalization", LOG_TYPES::TRACE, 3);
	_H->diagH(false, std::min(128, (int)(_hilbert.getFullHilbertSize() / 2)), 0, 1000, 1e-13, "lanczos");
	const auto& _eigvec			= _H->getEigVec();
	const auto& _krylov_mb 		= _H->getKrylov();

	if (time_evo) {
		_QMat = algebra::change_basis_matrix(_krylov_mb, _QMat, true);
		for (auto& _QOMat_ : _QOMat)
			_QOMat_ = algebra::change_basis_matrix(_krylov_mb, _QOMat_, true);
	}

	// Measure and save
	for (int i = 0; i < stateNum; ++i) 
	{
		const arma::Col<_T> _mbs 				= LanczosMethod<_T>::trueState(_eigvec, _krylov_mb, i);
		_meansLAN(i)							= _H->getEigVal(i);
		LOGINFO(std::format("Found ED (Lanczos) state({}) with E={}", i, STRPS(_meansLAN[i], UI_NQS_PRECISION)), LOG_TYPES::INFO, 2);
		_meas_LAN[i].measure(_mbs, _hilbert);
		_meas_LAN[i].saveMB({".h5"}, "measurement", "measurement", "measurement", std::format("LAN/{}", i), i > 0 || time_evo);
		if (time_evo) 
		{
			nqs_perform_time_evo_ed(_mbs, _H, _QMat, _QOMat, _timespace, dir, i, _quenchOpMeasure, "LAN", threadNum);
		}
	}
	saveAlgebraic(dir, "history.h5", _meansLAN, "Lanczos/energy", _saved);
	_saved = true;
}
// template instantiation
template void nqs_perform_lanczos_ed<double, 2>(int stateNum, Hamiltonian<double, 2>* _H,
										const Hilbert::HilbertSpace<double, 2>& _hilbert,
										const std::shared_ptr<Operators::OperatorComb<double>>& _quenchOp,
										const std::vector<std::shared_ptr<Operators::OperatorComb<double>>>& _quenchOpMeasure,
										arma::SpMat<double>& _QMat, v_1d<arma::SpMat<double>>& _QOMat,
										const arma::vec& _timespace,
										const std::string& dir,
										v_1d<NQSAv::MeasurementNQS<double>>& _meas_LAN,
										arma::Col<double>& _meansLAN,
										bool time_evo,
										bool& _saved,
										int threadNum);
template void nqs_perform_lanczos_ed<cpx, 2>(int stateNum, Hamiltonian<cpx, 2>* _H,
										const Hilbert::HilbertSpace<cpx, 2>& _hilbert,
										const std::shared_ptr<Operators::OperatorComb<cpx>>& _quenchOp,
										const std::vector<std::shared_ptr<Operators::OperatorComb<cpx>>>& _quenchOpMeasure,
										arma::SpMat<cpx>& _QMat, v_1d<arma::SpMat<cpx>>& _QOMat,
										const arma::vec& _timespace,
										const std::string& dir,
										v_1d<NQSAv::MeasurementNQS<cpx>>& _meas_LAN,
										arma::Col<cpx>& _meansLAN,
										bool time_evo,
										bool& _saved,
										int threadNum);

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<typename _T, uint _spinModes = 2>
static std::pair<arma::Col<_T>, arma::Col<_T>> 
nqs_perform_diag(int stateNum,
                bool time_evo,
                bool fullED,
                bool& _saved,
                Hamiltonian<_T, _spinModes>* _H,
                v_1d<NQSAv::MeasurementNQS<_T>>& _meas_ED,
                v_1d<NQSAv::MeasurementNQS<_T>>& _meas_LAN,
                const Hilbert::HilbertSpace<_T, _spinModes>& _hilbert,
                const std::shared_ptr<Operators::OperatorComb<_T>>& _quenchOp,
                const std::vector<std::shared_ptr<Operators::OperatorComb<_T>>>& _quenchOpMeasure,
                const arma::vec& _timespace,
                const std::string& dir,
                int threadNum) 
{
    // Validate inputs
    if (!_H) {
		throw std::invalid_argument("Hamiltonian is not defined");
	}
    if (_meas_ED.size() != stateNum || _meas_LAN.size() != stateNum) {
        throw std::invalid_argument("Measurement vectors have incorrect size");
	}

	// Hilbert
	const size_t Nh = _H->getHilbertSize();

    // Initialize results and build Hamiltonian
    arma::Col<_T> _meansED(stateNum, arma::fill::zeros), _meansLAN(stateNum, arma::fill::zeros);
    _H->buildHamiltonian();

    // Generate quench operators if time evolution is enabled
    arma::SpMat<_T> _QMat;
    v_1d<arma::SpMat<_T>> _QOMat;
    if (time_evo) 
	{
		_QMat = _quenchOp->template generateMat<false, _T, arma::SpMat>(Nh);
		for (auto& _quenchOpMeasure_ : _quenchOpMeasure)
			_QOMat.push_back(_quenchOpMeasure_->template generateMat<false, _T, arma::SpMat>(Nh));
    }

    // Full diagonalization
    if (fullED) {
        nqs_perform_full_ed(stateNum, _H, _hilbert, _quenchOp, _quenchOpMeasure, 
                                _QMat, _QOMat, _timespace, dir, _meas_ED, _meansED, time_evo, _saved, threadNum);
		// ---------------
		_H->clearEigVal();
		_H->clearEigVec();
		// ---------------
    }

    // Lanczos diagonalization
    nqs_perform_lanczos_ed(stateNum, _H, _hilbert, _quenchOp, _quenchOpMeasure, 
                                _QMat, _QOMat, _timespace, dir, _meas_LAN, _meansLAN, time_evo, _saved, threadNum);
	// ---------------
	_H->clear();
	// ---------------
    return std::make_pair(_meansED, _meansLAN);
}
// template instantiation
template std::pair<arma::Col<double>, arma::Col<double>> nqs_perform_diag<double, 2>(int stateNum, bool time_evo, bool fullED, bool& _saved, Hamiltonian<double, 2>* _H, v_1d<NQSAv::MeasurementNQS<double>>& _meas_ED, v_1d<NQSAv::MeasurementNQS<double>>& _meas_LAN, const Hilbert::HilbertSpace<double, 2>& _hilbert, const std::shared_ptr<Operators::OperatorComb<double>>& _quenchOp, const std::vector<std::shared_ptr<Operators::OperatorComb<double>>>& _quenchOpMeasure, const arma::vec& _timespace, const std::string& dir, int threadNum);
template std::pair<arma::Col<cpx>, arma::Col<cpx>> nqs_perform_diag<cpx, 2>(int stateNum, bool time_evo, bool fullED, bool& _saved, Hamiltonian<cpx, 2>* _H, v_1d<NQSAv::MeasurementNQS<cpx>>& _meas_ED, v_1d<NQSAv::MeasurementNQS<cpx>>& _meas_LAN, const Hilbert::HilbertSpace<cpx, 2>& _hilbert, const std::shared_ptr<Operators::OperatorComb<cpx>>& _quenchOp, const std::vector<std::shared_ptr<Operators::OperatorComb<cpx>>>& _quenchOpMeasure, const arma::vec& _timespace, const std::string& dir, int threadNum);

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
			return std::make_shared<NQS_PP_S<_spinModes, _T, _T, double, RBM_S<_spinModes, _T>>>(std::forward<decltype(args)>(args)...);
		default:
			throw std::invalid_argument("Unknown NQS type");
		}
	};
	NQS_Const_par_t<_spinModes, _T> _parNQS;
	_parNQS.nHid_ 		= { this->nqsP.nqs_nh_ };
	_parNQS.lr_ 		= { this->nqsP.nqs_lr_ };
	_parNQS.threadNum_ 	= this->threadNum;
	_parNQS.H_ 			= _H;
	// create the NQS
	_NQS 				= createNQS(_parNQS, _NQSl, _beta);
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
	const int stateNum 	= static_cast<int>(this->nqsP.nqs_ex_beta_.size()) + 1;
	std::shared_ptr<Hamiltonian<_T, _spinModes>> _H;
	Hilbert::HilbertSpace<_T> _hilbert;
	this->defineModel<_T>(_hilbert, _H);
	
	// define the NQS states for the excited states
	arma::Col<_T>	_meansNQS(this->nqsP.nqs_ex_beta_.size() + 1, arma::fill::zeros), _stdsNQS(this->nqsP.nqs_ex_beta_.size() + 1, arma::fill::zeros);
	
	v_sp_t<NQS<_spinModes, _T>> _NQS(this->nqsP.nqs_ex_beta_.size() + 1);					// define the NQS states
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
	const bool fullED 		= this->nqsP.nqs_ed_ && Nh <= UI_LIMITS_NQS_ED;					// use the full diagonalization
	const bool lanED 		= this->nqsP.nqs_ed_ && Nh <= ULLPOW(24);						// use the Lanczos method for the ED
	v_sp_t<Operators::OperatorNQS<_T>> _opsG;												// set up the operators to save - global
	v_sp_t<Operators::OperatorNQS<_T, uint>> _opsL;											// set up the operators to save - local
	v_sp_t<Operators::OperatorNQS<_T, uint, uint>> _opsC;									// set up the operators to save - correlation
	get_NQS_operators_test(_opsG, _opsL, _opsC, Nvis, this->latP.lat);						// get the operators for the NQS

	// ---------------
	v_1d<NQSAv::MeasurementNQS<_T>> _meas_ED, _meas_LAN, _meas_NQS;
	for (int i = 0; i < stateNum; ++i) 
	{
		_meas_ED.push_back(NQSAv::MeasurementNQS<_T>(this->latP.lat, dir, _opsG, _opsL, _opsC, this->threadNum));
		_meas_LAN.push_back(NQSAv::MeasurementNQS<_T>(this->latP.lat, dir, _opsG, _opsL, _opsC, this->threadNum));
		_meas_NQS.push_back(NQSAv::MeasurementNQS<_T>(this->latP.lat, dir, _opsG, _opsL, _opsC, this->threadNum));
	}

	// ---------------
	arma::vec _timespace;
	std::shared_ptr<Operators::OperatorComb<_T>> _quenchOp;
	v_sp_t<Operators::OperatorComb<_T>> _quenchOpMeasure;
	if (this->nqsP.nqs_te_) 
	{
		_quenchOp 			= std::make_shared<Operators::OperatorComb<_T>>(Operators::SpinOperators::sig_z<_T>(Nvis, 0));
		_quenchOpMeasure.push_back(std::make_shared<Operators::OperatorComb<_T>>(Operators::SpinOperators::sig_z<_T>(Nvis, 0)));
		_quenchOpMeasure.push_back(std::make_shared<Operators::OperatorComb<_T>>(Operators::SpinOperators::sig_z<_T>(Nvis, Nvis - 1)));
		_quenchOpMeasure.push_back(std::make_shared<Operators::OperatorComb<_T>>(Operators::SpinOperators::sig_z<_T>(Nvis, {1, Nvis - 1})));
		_timespace 			= time_space_nqs(this->nqsP.nqs_te_tlog_, this->nqsP.nqs_te_dt_, this->nqsP.nqs_te_tf_);
		saveAlgebraic(dir, "measurement.h5", _timespace, "time_evo/time", false);
	}

	// save lattice information
	bool _saved = Lattice::save_bonds(this->latP.lat, dir, "history.h5");

	// check ED
	arma::Col<_T> _meansED(stateNum, arma::fill::zeros), _meansLAN(stateNum, arma::fill::zeros);
	if (lanED || fullED)
	{ 	
		std::tie(_meansED, _meansLAN) = nqs_perform_diag<_T, _spinModes>(stateNum, this->nqsP.nqs_te_, 
					fullED, _saved, 
					_H.get(), _meas_ED, _meas_LAN, _hilbert, _quenchOp, _quenchOpMeasure, _timespace, dir, this->threadNum);
		LOGINFO("", LOG_TYPES::TRACE, 20, '#', 1);
		LOGINFO(2);
	}
	LOGINFO(nqsInfo, LOG_TYPES::TRACE, 2);
	LOGINFO(1);

	v_1d<std::shared_ptr<NQS<_spinModes, _T>>> _NQS_lower = {};									// define the NQS states for the excited states
	for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) 
	{
		_timer.checkpoint(VEQ(i));
		arma::Col<_T> _EN_TRAIN, _EN_TESTS, _EN_STD, _EN_TESTS_STD;								// set up the energies container for NQS

		// define the NQS states for the excited states
		if (!_NQS[i]) 
		{
			this->defineNQS<_T, _spinModes>(_H, _NQS[i], _NQS_lower, 
				{ this->nqsP.nqs_ex_beta_.begin(), this->nqsP.nqs_ex_beta_.begin() + i });	
		}
		_NQS[i]->setTrainParExc(_parE);															// set the parameters in the excited states

		if (this->nqsP.nqs_tr_pt_ > 0)
		{
			LOGINFO("Using parallel tempering for training", LOG_TYPES::TRACE, 2);
			LOGINFO(1);
			
			typename MonteCarlo::ParallelTempering<_T>::Solver_p _mcs = _NQS[i];
			auto _pt = std::make_shared<MonteCarlo::ParallelTempering<_T>>(_mcs, this->nqsP.nqs_tr_pt_, MonteCarlo::BetaSpacing::LOGARITHMIC, 0.1, 1.0);
			
			_pt->train(_parT, this->quiet, this->nqsP.nqs_tr_rst_, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);
			auto [_best_idx, _best_acc_idx] = _pt->getBestInfo();

			// get the best loss
			_EN_TRAIN 	= _pt->getBestLosses();
			_EN_STD 	= _pt->getStdLosses(_best_idx);

			// get the best solver
			if (_best_idx > 0)
			{
				auto bestSolver = _pt->getBestSolver();
				if (bestSolver)
					_NQS[i] = std::dynamic_pointer_cast<NQS<_spinModes, _T>>(bestSolver);
			}
			
		}
		else {
			std::tie(_EN_TRAIN, _EN_STD) = _NQS[i]->train(_parT, this->quiet, this->nqsP.nqs_tr_rst_, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);
		}
		LOGINFO("", LOG_TYPES::TRACE, 20, '#', 1);
		LOGINFO(1);

		// collect the data
		{
			// -------------------------------------
			_NQS[i]->collect(_parC, _meas_NQS[i], &_EN_TESTS, &_EN_TESTS_STD, this->quiet, this->nqsP.nqs_col_rst_, _timer.point(VEQ(i)), nqsP.nqs_tr_pc_);			
			// -------------------------------------
LOGINFO("", LOG_TYPES::TRACE, 40, '#', 1);
			LOGINFO(4);
		}

		{
			const bool _append = lanED || fullED || i > 0 || _saved;
			_NQS[i]->save_history(dir, _EN_TRAIN, _EN_TESTS, _EN_STD, _EN_TESTS_STD, 
										this->nqsP.nqs_ex_beta_, 
										_meansNQS, _stdsNQS, i, _append);
			_NQS[i]->saveInfo(dir, "history.h5", i);
			_meas_NQS[i].saveNQS({".h5"}, "measurement", "measurement", "measurement", "NQS/" + STR(i), _append);
		}
		_NQS_lower.push_back(_NQS[i]);
	}
	
	for (int i = 0; i < this->nqsP.nqs_ex_beta_.size() + 1; ++i) {
		LOGINFO(std::format("True energies: EED_{} = {}, ELAN_{} = {}, ENQS_{} = {} +- {}", 
				i, STRP(_meansED[i], UI_NQS_PRECISION), 
				i, STRP(_meansLAN[i], UI_NQS_PRECISION), 
				i, STRP(_meansNQS[i], UI_NQS_PRECISION), 
				STRPS(_stdsNQS(i) / 2.0, UI_NQS_PRECISION)), LOG_TYPES::TRACE, 2);
	}

	// try time evolution 
	if (this->nqsP.nqs_te_)
	{

		LOGINFO(3);
		MonteCarlo::MCS_train_t _parTime(this->nqsP.nqs_te_mc_, this->nqsP.nqs_te_th_, this->nqsP.nqs_te_bn_, this->nqsP.nqs_te_bs_, this->nqsP.nFlips_, dir); 
		// _H->quenchHamiltonian();
		_parC.MC_sam_ 	= 1;
		auto _RKsolver 	= algebra::ODE::createRKsolver<_T>(static_cast<algebra::ODE::ODE_Solvers>(this->nqsP.nqs_te_rk_));	// create the Runge-Kutta solver

		for (int j = 0; j < this->nqsP.nqs_ex_beta_.size() + 1; ++j)
		{
			// Set the hyperparameters
			{
	#ifdef NQS_USESR_MAT_USED
				_NQS->setPinv(this->nqsP.nqs_tr_pinv_);
	#endif
				_NQS[j]->setSolver(this->nqsP.nqs_tr_sol_, this->nqsP.nqs_tr_tol_, this->nqsP.nqs_tr_iter_, this->nqsP.nqs_tr_reg_);
				_NQS[j]->setPreconditioner(this->nqsP.nqs_tr_prec_);
	#ifdef NQS_USESR
				_NQS[j]->setSregScheduler(this->nqsP.nqs_tr_regs_, this->nqsP.nqs_tr_reg_, this->nqsP.nqs_tr_regd_, this->nqsP.nqs_tr_epo_, this->nqsP.nqs_tr_regp_);
	#endif
				_NQS[j]->setScheduler(this->nqsP.nqs_sch_, this->nqsP.nqs_lr_, this->nqsP.nqs_lrd_, this->nqsP.nqs_tr_epo_, this->nqsP.nqs_lr_pat_);
				_NQS[j]->setEarlyStopping(this->nqsP.nqs_es_pat_, this->nqsP.nqs_es_del_);
			}
			LOGINFO("Starting the time evolution for NQS state(" + STR(j) + ")", LOG_TYPES::TRACE, 1);
			v_1d<arma::Col<_T>> _vals(_quenchOpMeasure.size(), arma::Col<_T>(_parC.MC_sam_ * _parC.nblck_));
			arma::Col<_T> _En(_parTime.nblck_);											// set up the containers for the time evolution
			arma::Mat<double> _vals_mean(_timespace.size() + 1, _quenchOpMeasure.size(), arma::fill::zeros); // set up the container for the mean values

			// set up the operators for the time evolution
			v_1d<Operators::OperatorNQS<_T>> _QOpM_v;
			for (auto& _quenchOpMeasure_ : _quenchOpMeasure)
				_QOpM_v.push_back(*_quenchOpMeasure_.get());

			_NQS[j]->reset(_parTime.nblck_);												// reset the derivatives	
			_NQS[j]->setModifier(_quenchOp);												// set the quench operator
			_NQS[j]->evolveSet(_parTime, this->quiet, false);								// set the evolution function
			_NQS[j]->template collect<arma::Col<_T>>(_parC, _QOpM_v, _vals);				// collect the data using ratio method - before the time evolution
			for (int xx = 0; xx < _quenchOpMeasure.size(); ++xx)
				_vals_mean(0, xx) = algebra::cast<double>(arma::mean(_vals[xx]));			// save the mean value

			LOGINFO("TE(" + STR(0) + "/" + STR(_timespace.size()) + ") Time = 0", LOG_TYPES::TRACE, 2);
			for (int xx = 0; xx < _quenchOpMeasure.size(); ++xx)
				LOGINFO(_quenchOpMeasure[xx]->getNameS() + STRPS(_vals_mean(0, xx), 3), LOG_TYPES::TRACE, 3);
			
			double _dt = 0;
			for (int i = 0; i < _timespace.size(); ++i)
			{
				if (i < _timespace.size() - 1)
				 	_dt = _timespace(i + 1) - _timespace(i);								// set the time step for the evolution
				_NQS[j]->evolveStepSet(i, _dt, _RKsolver);							// evolve the NQS
				_NQS[j]->template collect<arma::Col<_T>>(_parC, _QOpM_v, _vals);			// collect the data using ratio method
				for (int xx = 0; xx < _quenchOpMeasure.size(); ++xx)						
					_vals_mean(i + 1, xx) = algebra::cast<double>(arma::mean(_vals[xx]));	// save the mean value
				
				if (i % 10 == 0)
				{
					LOGINFO("TE(" + STR(i + 1) + "/" + STR(_timespace.size()) + ") Time = " + STRPS(_timespace(i), 3), LOG_TYPES::TRACE, 2);
					for (int xx = 0; xx < _quenchOpMeasure.size(); ++xx)
						LOGINFO(_quenchOpMeasure[xx]->getNameS() + " : " + STRPS(_vals_mean(i + 1, xx), 3), LOG_TYPES::TRACE, 3);
				}
			}			
			for (int xx = 0; xx < _quenchOpMeasure.size(); ++xx)
				saveAlgebraic(dir, "measurement.h5", _vals_mean.col(xx), std::format("NQS/{}/time_evo/{}", j, _quenchOpMeasure[xx]->getNameS()), true);
		}

		if (_RKsolver) {
			delete _RKsolver;
			_RKsolver = nullptr;
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