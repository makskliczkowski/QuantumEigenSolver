#include "../../include/NQS/nqs_final.hpp"
#include <numeric>

// ##########################################################################################################################################

// template instantiation of the class NQS_lower_t for <spins, double, double, double>
template struct NQS_lower_t<2u, double, double, double>;
template struct NQS_lower_t<3u, double, double, double>;
template struct NQS_lower_t<4u, double, double, double>;
// template instantiation of the class NQS_lower_t for <spins, double, complex, double>
template struct NQS_lower_t<2u, double, std::complex<double>, double>;
template struct NQS_lower_t<3u, double, std::complex<double>, double>;
template struct NQS_lower_t<4u, double, std::complex<double>, double>;
// template instantiation of the class NQS_lower_t for <spins, complex, double, double>
template struct NQS_lower_t<2u, std::complex<double>, double, double>;
template struct NQS_lower_t<3u, std::complex<double>, double, double>;
template struct NQS_lower_t<4u, std::complex<double>, double, double>;
// template instantiation of the class NQS_lower_t for <spins, complex, complex, double>
template struct NQS_lower_t<2u, std::complex<double>, std::complex<double>, double>;
template struct NQS_lower_t<3u, std::complex<double>, std::complex<double>, double>;
template struct NQS_lower_t<4u, std::complex<double>, std::complex<double>, double>;

// ##########################################################################################################################################

/**
* @brief Sets the size of the containers for the derivatives of the lower states - for the gradient estimation of the excited state.
* @param _mcsexcited size of the Monte Carlo steps for the excited state - ratios sampled in the excited state
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS_lower_t<_spinModes, _Ht, _T, _stateType>::setDerivContSize(size_t _mcsexcited)
{
    if (this->f_lower_size_ == 0)
        return;
    
    // Initialize the ratios containers for lower and excited states
    this->ratios_lower_.resize(this->f_lower_size_, arma::Col<_T>(this->train_lower_.nblck_, arma::fill::zeros));
    this->ratios_excited_.resize(this->f_lower_size_, arma::Col<_T>(_mcsexcited, arma::fill::zeros));
}

// template instantiation of function above 
NQS_INST_L_CMB(double, double, setDerivContSize, void, (size_t));
NQS_INST_L_CMB(double, std::complex<double>, setDerivContSize, void, (size_t));
NQS_INST_L_CMB(std::complex<double>, double, setDerivContSize, void, (size_t));
NQS_INST_L_CMB(std::complex<double>, std::complex<double>, setDerivContSize, void, (size_t));

// ##########################################################################################################################################

/**
* @brief Sets the projection operator for the NQS (Neural Quantum State) lower class.
*
* This function creates and sets the projection operator for the current excited state.
* The projection operator is used to project to the current state and calculate the 
* probability ratio for the excited state using the vector representation.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type.
* @tparam _stateType The state type.
* @param _current_exc_state The current excited state operator.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS_lower_t<_spinModes, _Ht, _T, _stateType>::setProjector(Operators::_OP_V_T_CR _current_exc_state)
{
    // create the projection operator
    this->enP_ = Operators::GeneralOperators::projectorSumComb(Ns_, 
                    _current_exc_state,     // project to current state <s|psi_w>
                    this->exc_ratio_);      // calculate the probability ratio (for the excited state) using the vector representation \psi _w(s') / \psi _w(s)
}

// template instantiation of function above
NQS_INST_L_CMB(double, double, setProjector, void, (Operators::_OP_V_T_CR));
NQS_INST_L_CMB(double, std::complex<double>, setProjector, void, (Operators::_OP_V_T_CR));
NQS_INST_L_CMB(std::complex<double>, double, setProjector, void, (Operators::_OP_V_T_CR));
NQS_INST_L_CMB(std::complex<double>, std::complex<double>, setProjector, void, (Operators::_OP_V_T_CR));

// ##########################################################################################################################################

/**
* @brief Collects the additional energy contribution from the overlap with the lower states for the excited state energy estimation.
* 
* This function calculates the mean value of the energy contribution from the overlap with the lower states. It collects the energy 
* estimation for the lower states and computes the mean value of the energy contributions.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type.
* @tparam _stateType The state type.
* @param i The index of the lower state.
* @returns The mean value of the energy estimation from the lower states overlap.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS_lower_t<_spinModes, _Ht, _T, _stateType>::collectLowerEnergy(uint i)
{
    if (this->f_lower_size_ == 0)
        return _T(0.0);
    
    // Collect the energy estimation for the lower states
    this->f_lower[i]->template collect<std::vector<_T>>(this->train_lower_, this->enP_, &this->containerP_[i], nullptr, false);
    
    _T _mean = 0.0;
    MonteCarlo::blockmean(this->containerP_[i], std::max(this->train_lower_.bsize_, uint(8)), &_mean);

    // Calculate the mean value of the energy contributions
    // _T _mean = std::accumulate(this->containerP_[i].begin(), this->containerP_[i].end(), _T(0.0)) / _T(this->containerP_[i].size());
    
    // Return the weighted mean value
    return this->f_lower_b_[i] * _mean;
}

// template instantiation of function above
NQS_INST_L_CMB(double, double, collectLowerEnergy, double, (uint));
NQS_INST_L_CMB(double, std::complex<double>, collectLowerEnergy, std::complex<double>, (uint));
NQS_INST_L_CMB(std::complex<double>, double, collectLowerEnergy, double, (uint));
NQS_INST_L_CMB(std::complex<double>, std::complex<double>, collectLowerEnergy, std::complex<double>, (uint));

// ##########################################################################################################################################

/**
* @brief Collects the ratios for the lower states for the gradient estimation of the excited state.
* This ratio is given by the \psi _w(s') / \psi _w_j(s) at each Monte Carlo step (average in the lower states).
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type.
* @tparam _stateType The state type.
* @param i The index of the lower state.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS_lower_t<_spinModes, _Ht, _T, _stateType>::collectLowerRatios(uint i)
{
    if (this->f_lower_size_ == 0)
        return;

    this->f_lower[i]->collect_ratio(this->train_lower_, this->nqs_exc_, this->ratios_lower_[i]); // Collect the ratio for the lower states
}

// template instantiation of function above
NQS_INST_L_CMB(double, double, collectLowerRatios, void, (uint));
NQS_INST_L_CMB(double, std::complex<double>, collectLowerRatios, void, (uint));
NQS_INST_L_CMB(std::complex<double>, double, collectLowerRatios, void, (uint));
NQS_INST_L_CMB(std::complex<double>, std::complex<double>, collectLowerRatios, void, (uint));

// ##########################################################################################################################################

/**
* @brief Collects the ratios of excited states for a given index and current excited state.
*
* This function calculates the ratio of the ansatz for the excited state at a given index.
* If the size of the lower state is zero, it returns 0.0. Otherwise, it uses the ansatz
* ratio from the NQS excited state object.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for the return value.
* @tparam _stateType The type of the state.
* @param i The index for which the ratio is to be collected.
* @param _current_exc_state The current excited state operator.
* @return The ratio of the ansatz for the excited state at the given index.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T NQS_lower_t<_spinModes, _Ht, _T, _stateType>::collectExcitedRatios(uint i, Operators::_OP_V_T_CR _current_exc_state)
{
    if (this->f_lower_size_ == 0)
        return _T(0.0);
    return this->nqs_exc_->ansatz_ratio(_current_exc_state, this->f_lower[i].get());
}

// template instantiation of function above
NQS_INST_L_CMB(double, double, collectExcitedRatios, double, (uint, Operators::_OP_V_T_CR));
NQS_INST_L_CMB(double, std::complex<double>, collectExcitedRatios, std::complex<double>, (uint, Operators::_OP_V_T_CR));
NQS_INST_L_CMB(std::complex<double>, double, collectExcitedRatios, double, (uint, Operators::_OP_V_T_CR));
NQS_INST_L_CMB(std::complex<double>, std::complex<double>, collectExcitedRatios, std::complex<double>, (uint, Operators::_OP_V_T_CR));

// ##########################################################################################################################################

/**
* @brief Collects the ratio of function values over ansatz values for a given number of blocks.
* 
* This function performs Monte Carlo sampling to collect the ratio of a given function applied to 
* the neural quantum state (NQS) over the ansatz values. The sampling is done in blocks to reduce 
* autocorrelations and thermalize the state. 
* The ratio is calculated as f(s) / \psi(s) for each sampled state, where \psi is the NQS ansatz sampled at this time.
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for the calculations.
* @tparam _stateType The type of the state.
* @param _par The training parameters, including the number of flips, block size, and number of blocks.
* @param _f A function that takes the NQS state and returns a value of type _T.
* @param _container An Armadillo column vector to store the collected ratios. !TODO - change to a more general container.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::collect_ratio(const MonteCarlo::MCS_train_t& _par, std::function<_T(const NQSS&)> _f, arma::Col<_T>& _container)
{
	this->setRandomFlipNum(_par.nFlip);						// set the random state at the begining
	if (_par.MC_th_ > 0) 
		this->setRandomState();								// set the random state at the begining
	this->blockSample(_par.MC_th_, NQS_STATE, false);		// remove autocorrelations and thermalizes

	for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 	// iterate blocks - allows to collect samples outside of the block
	{
		this->blockSample(_par.bsize_, NQS_STATE, false);	// sample them!
		
		const _T _top 		= _f(NQS_STATE);				// calculate f(s) / \psi(s)
#ifdef NQS_LOWER_RATIO_LOGDIFF
		const _T _bottom 	= this->ansatzlog(NQS_STATE);	// calculate the log of the lower ansatz
		_container(_taken) 	= std::exp(_top - _bottom);		// collect samples
#else 
		const _T _bottom 	= this->ansatz(NQS_STATE);
		_container(_taken) 	+= _top / _bottom;
#endif
	}											
}

// template instantiation of function above
NQS_INST_CMB(double, double, collect_ratio, void, (const MonteCarlo::MCS_train_t&, std::function<double(const NQSS&)>, arma::Col<double>&));
NQS_INST_CMB(double, std::complex<double>, collect_ratio, void, (const MonteCarlo::MCS_train_t&, std::function<std::complex<double>(const NQSS&)>, arma::Col<std::complex<double>>&));
NQS_INST_CMB(std::complex<double>, double, collect_ratio, void, (const MonteCarlo::MCS_train_t&, std::function<double(const NQSS&)>, arma::Col<double>&));
NQS_INST_CMB(std::complex<double>, std::complex<double>, collect_ratio, void, (const MonteCarlo::MCS_train_t&, std::function<std::complex<double>(const NQSS&)>, arma::Col<std::complex<double>>&));

// ##########################################################################################################################################

/**
* @brief Collects the ratio of the ansatz function values between the current NQS and another NQS.
*
* This function collects the ratio of the ansatz function values between the current NQS and another provided NQS
* and stores the results in the provided container. It performs block sampling to remove autocorrelations and 
* thermalizes the system before collecting the samples.
*
* The ratio collected is given by:
* \f[
* \text{ratio} = \frac{\psi_{\text{other}}(s)}{\psi_{\text{current}}(s)}
* \f]
* where \(\psi_{\text{current}}(s)\) is the ansatz function value of the current NQS and \(\psi_{\text{other}}(s)\) 
* is the ansatz function value of the other NQS for the sampled state \(s\). Reffer to the NQS::ansatz_ratio() function for more details.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for the calculations.
* @tparam _stateType The state type.
* @param _par The training parameters for the NQS.
* @param other Pointer to another NQS object to compare against.
* @param _container The container to store the collected ratio values.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::collect_ratio(const MonteCarlo::MCS_train_t& _par, NQS<_spinModes, _Ht, _T, _stateType>* other, arma::Col<_T>& _container)
{
	if (!other)													// if the other NQS is not provided, return
		return;

	this->setRandomFlipNum(_par.nFlip);							// set the random state at the begining
	if (_par.MC_th_ > 0) 
			this->setRandomState();								// set the random state at the begining
    this->blockSample(_par.MC_th_, NQS_STATE, false);		    // remove autocorrelations and thermalizes

	for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 		// iterate blocks - allows to collect samples outside of the block
	{
		this->blockSample(_par.bsize_, NQS_STATE, false);		// sample them! - remove autocorrelations and thermalizes
		const auto _val 	= this->ansatz_ratio(NQS_STATE, other);
		_container(_taken) 	= _val;								// store the value of f(s) / \psi(s)
	}									
}

// template instantiation of function above
template void NQS<2u, double, double, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<2u, double, double, double>*, arma::Col<double>&);
template void NQS<2u, double, std::complex<double>, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<2u, double, std::complex<double>, double>*, arma::Col<std::complex<double>>&);
template void NQS<2u, std::complex<double>, double, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<2u, std::complex<double>, double, double>*, arma::Col<double>&);
template void NQS<2u, std::complex<double>, std::complex<double>, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<2u, std::complex<double>, std::complex<double>, double>*, arma::Col<std::complex<double>>&);
template void NQS<3u, double, double, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<3u, double, double, double>*, arma::Col<double>&);
template void NQS<3u, double, std::complex<double>, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<3u, double, std::complex<double>, double>*, arma::Col<std::complex<double>>&);
template void NQS<3u, std::complex<double>, double, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<3u, std::complex<double>, double, double>*, arma::Col<double>&);
template void NQS<3u, std::complex<double>, std::complex<double>, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<3u, std::complex<double>, std::complex<double>, double>*, arma::Col<std::complex<double>>&);
template void NQS<4u, double, double, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<4u, double, double, double>*, arma::Col<double>&);
template void NQS<4u, double, std::complex<double>, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<4u, double, std::complex<double>, double>*, arma::Col<std::complex<double>>&);
template void NQS<4u, std::complex<double>, double, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<4u, std::complex<double>, double, double>*, arma::Col<double>&);
template void NQS<4u, std::complex<double>, std::complex<double>, double>::collect_ratio(const MonteCarlo::MCS_train_t&, NQS<4u, std::complex<double>, std::complex<double>, double>*, arma::Col<std::complex<double>>&);
// ##########################################################################################################################################

