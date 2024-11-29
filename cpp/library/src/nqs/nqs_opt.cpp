#include "../../include/NQS/nqs_final.hpp"

// ##########################################################################################################################################

// ############################################################ T R A I N I N G #############################################################

// ##########################################################################################################################################

#include <cmath>
#include <utility>
#include <functional>
#include <complex>


// ##########################################################################################################################################

/**
* @brief Checks the stopping condition for the training and eventually stops the training. Also, saves the weights if needed.
* @note This function is called at the end of each iteration in the training loop.
* @param i current iteration
* @param _par parameters for the training
* @param _currLoss current loss value
* @param _quiet quiet mode
* @returns whether the training should be stopped
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
bool NQS<_spinModes, _Ht, _T, _stateType>::trainStop(size_t i, const NQS_train_t& _par, _T _currLoss, _T _currstd, bool _quiet)
{
	auto best 				= this->info_p_.best();
	
	const std::string _prog = "Iteration " + STR(i) + "/" + STR(_par.MC_sam_) +
								", Loss: " + STRPS(_currLoss, 4) + " ± " + STRPS(_currstd / 2.0, 3) +
								", Best: " + VEQPS(best, 4) + 
								", Acceptance: " + STR(this->accepted_) + "/" + STR(this->total_) + " (" + STRP((double)this->accepted_ / this->total_ * 100, 2) + "%)" +
								", LR: " + STRPS(this->info_p_.lr_, 4) +
								", Reg: " + STRPS(this->info_p_.sreg_, 4);

	PROGRESS_UPD_Q(i, this->pBar_, _prog, !_quiet);
	
	this->updateWeights_ 	= !this->info_p_.stop(i, _currLoss) && this->updateWeights_;

#ifdef NQS_SAVE_WEIGHTS
	if (i % this->pBar_.percentageSteps == 0 || !this->updateWeights_)  
		this->saveWeights(_par.dir + NQS_SAVE_DIR, "weights_" + STR(this->lower_states_.f_lower_size_) + ".h5");
#endif

	if (!this->updateWeights_) {
		LOGINFO("Stopping at iteration " + STR(i) + " with last loss value: " + STRPS(_currLoss, 4), LOG_TYPES::WARNING, 1);
		return true;
	}
	return false;
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, trainStop, bool, (size_t, const NQS_train_t&, double, double, bool));
NQS_INST_CMB(double, std::complex<double>, trainStop, bool, (size_t, const NQS_train_t&, std::complex<double>, std::complex<double>, bool));
NQS_INST_CMB(std::complex<double>, double, trainStop, bool, (size_t, const NQS_train_t&, double, double, bool));
NQS_INST_CMB(std::complex<double>, std::complex<double>, trainStop, bool, (size_t, const NQS_train_t&, std::complex<double>, std::complex<double>, bool));

// ##########################################################################################################################################

/**
* @brief Perform single training step of the NQS. This function is called in the training loop. It samples the states, calculates the gradient, and updates the weights.
* @param i current iteration 
* @param En vector of local energies
* @param meanEn vector of mean energies
* @param stdEn vector of standard deviations
* @param _par parameters for the training
* @param quiet quiet mode
* @param randomStart random start
* @param _timer timer for the training
* @returns whether the training should be stopped
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
bool NQS<_spinModes, _Ht, _T, _stateType>::trainStep(size_t i, 
													arma::Col<_T>& En,
													arma::Col<_T>& meanEn, 
													arma::Col<_T>& stdEn, 
													const NQS_train_t& _par, 
                                                    const bool quiet, 
                                                    const bool randomStart,
                                                    Timer& _timer)
{
	this->total_ 	= 0;										// reset the total number of flips
	this->accepted_ = 0;										// reset the number of accepted flips
	if (randomStart && _par.MC_th_ > 0) 
		this->setRandomState();									// set the random state at the begining
	this->blockSample(_par.MC_th_, NQS_STATE, !randomStart);	// thermalize the system - burn-in

	for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 		// iterate blocks - this ensures the calculation of a stochastic gradient constructed within the block
	{		
		this->blockSample(_par.bsize_, NQS_STATE, false);		// sample them using the local Metropolis sampling
		this->grad(NQS_STATE, _taken);							// calculate the gradient at each point of the iteration! - this is implementation specific!!!
		En(_taken) = this->locEnKernel();						// local energy - stored at each point within the estimation of the gradient (stochastic)

		if (this->lower_states_.f_lower_size_ > 0)				// calculate the excited states overlaps for the gradient - if used
		{
#ifndef _DEBUG 
# 	pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
		for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++) // calculate the excited states overlaps for the gradient - if used
			this->lower_states_.ratios_excited_[_low](_taken) = this->lower_states_.collectExcitedRatios(_low, NQS_STATE);
		}
	}

	if (this->lower_states_.f_lower_size_ > 0)					// collect the average for the lower states and collect the same for the lower states with this ansatz - for the gradient calculation
	{
#ifndef _DEBUG 
# 	pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
		for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++)  											// collect the average for the lower states and collect the same for the lower states with this ansatz - for the gradient calculation
			this->lower_states_.collectLowerRatios(_low);
	}
	
	MonteCarlo::blockmean(En, std::max((size_t)_par.bsize_, (size_t)8), &meanEn(i - 1), &stdEn(i - 1)); 				// save the mean energy
	TIMER_START_MEASURE(this->gradFinal(En, i, meanEn(i - 1)), (i % this->pBar_.percentageSteps == 0), _timer, STR(i)); // calculate the final update vector - either use the stochastic reconfiguration or the standard gradient descent

	if (this->updateWeights_)
		this->updateWeights(); 									// finally, update the weights with the calculated gradient (force) [can be done with the stochastic reconfiguration or the standard gradient descent] - implementation specific!!!

	if (this->trainStop(i, _par, meanEn(i - 1), stdEn(i - 1), quiet))
		return true;
	return false;
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, trainStep, bool, (size_t, arma::Col<double>&, arma::Col<double>&, arma::Col<double>&, const NQS_train_t&, const bool, const bool, Timer&));
NQS_INST_CMB(double, std::complex<double>, trainStep, bool, (size_t, arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&, const NQS_train_t&, const bool, const bool, Timer&));
NQS_INST_CMB(std::complex<double>, double, trainStep, bool, (size_t, arma::Col<double>&, arma::Col<double>&, arma::Col<double>&, const NQS_train_t&, const bool, const bool, Timer&));
NQS_INST_CMB(std::complex<double>, std::complex<double>, trainStep, bool, (size_t, arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&, const NQS_train_t&, const bool, const bool, Timer&));

// ##########################################################################################################################################

template <typename _T>
using NQS_TRAIN_PAIR = std::pair<arma::Col<_T>, arma::Col<_T>>;

/**
* @brief Perform single training of the NQS.
* @param mcSteps Monte Carlo steps to be used in the training - this is an outer loop for the training (number of iterations)
* @param nThrm Monte Carlo steps spent for thermalizing the system - burn-in
* @param nBlck	number of blocks to be used for calculating the covariance matrix - number of iterations for the gradient calculation (single iteration step)
* @param bSize number of update steps within a single block - size of the block (number of flip trials (global))
* @param nFlip number of flips for a single update of a state (default is 1)
* @param quiet wanna talk? (default is false)
* @param randomStart random start (default is false)
* @param _t timepoint for timestamping the training
* @param progPrc progress percentage to be displayed in the progress bar
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS_TRAIN_PAIR<_T> NQS<_spinModes, _Ht, _T, _stateType>::train(const NQS_train_t& _par, 
                                                                                            bool quiet, 
                                                                                            bool randomStart, 
                                                                                            clk::time_point _t, 
                                                                                            uint progPrc)
{
	{
		this->pBar_ = pBar(progPrc, _par.MC_sam_);							// set the progress bar		
		_par.hi();															// set the info about training
		this->derivativesReset(_par.nblck_);								// reset the derivatives				
		this->lower_states_.setDerivContSize(_par.nblck_);					// set the size of the containers for the lower states
	}

	Timer _timer;															// timer for the training
	arma::Col<_T> meanEn(_par.MC_sam_, arma::fill::zeros);					// here we save the mean energy
	arma::Col<_T> stdEn(_par.MC_sam_, arma::fill::zeros);					// here we save the standard deviation of the energy
	arma::Col<_T> En(_par.nblck_, arma::fill::zeros);						// history of energies (for given weights) - here we save the local energies at each block
	this->setRandomState();													// set the random state at the begining and the number of flips
	this->setRandomFlipNum(_par.nFlip);										// set the random state at the begining and the number of flips

	// go through the Monte Carlo steps
	uint i = 1;
	for (i = 1; i <= _par.MC_sam_; ++i) {
		if (this->trainStep(i, En, meanEn, stdEn, _par, quiet, randomStart, _timer)) // perform the training step
			break;
	}

	LOGINFO(_t, "NQS_EQ_" + STR(this->lower_states_.f_lower_size_), 1);
	if (i > 1) {
		if (i <= _par.MC_sam_) 
			return std::make_pair(meanEn.subvec(0, i - 1), stdEn.subvec(0, i - 1));
		else
			return std::make_pair(meanEn, stdEn);
	}
	return std::make_pair(meanEn.subvec(0, 0), stdEn.subvec(0, 0));
}

// template instantiation of function above for <spins, double and complex, double and complex, double>

NQS_INST_CMB(double, double, train, NQS_TRAIN_PAIR<double>, (const NQS_train_t&, bool, bool, clk::time_point, uint));
NQS_INST_CMB(double, std::complex<double>, train, NQS_TRAIN_PAIR<std::complex<double>>, (const NQS_train_t&, bool, bool, clk::time_point, uint));
NQS_INST_CMB(std::complex<double>, double, train, NQS_TRAIN_PAIR<double>, (const NQS_train_t&, bool, bool, clk::time_point, uint));
NQS_INST_CMB(std::complex<double>, std::complex<double>, train, NQS_TRAIN_PAIR<std::complex<double>>, (const NQS_train_t&, bool, bool, clk::time_point, uint));

// ##########################################################################################################################################

// ########################################################## C O L L E C T I O N ###########################################################

// ##########################################################################################################################################

/**
* @brief Collects a step in the Neural Quantum State (NQS) optimization process.
*
* This function performs a single step of the NQS optimization by sampling the system,
* measuring the operators, and optionally collecting the energy values.
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for numerical values.
* @tparam _stateType Type of the state.
* @param i Current step index.
* @param _par Training parameters for NQS.
* @param _meas Measurement object for NQS.
* @param _E Pointer to a column vector to store energy values (optional).
* @param _EM Pointer to a column vector to store mean energy values (optional).
* @param _ES Pointer to a column vector to store energy standard deviation values (optional).
* @param quiet Flag to suppress progress updates.
* @param randomStart Flag to initialize with a random state.
* @param _timer Timer object for measuring execution time.
* @return True if the step was successfully collected, false otherwise.
*
* @details
* The function performs the following steps:
* 1. Determines the start element for energy collection.
* 2. Checks if energy collection is enabled and validates the size of the energy vector.
* 3. Optionally sets a random state if `randomStart` is true.
* 4. Thermalizes the system using block sampling.
* 5. Iterates over blocks to sample the states and measure the operators.
* 6. Collects local energy values if energy collection is enabled.
* 7. Normalizes the measurements.
* 8. Computes the mean energy and standard deviation if energy collection is enabled.
* 9. Updates the progress bar if `quiet` is false.
*
* @note
* - Ensure that the size of the energy vector `_E` is sufficient to store the collected values.
* - The function uses assertions to validate input parameters, which may cause the program to terminate if the assertions fail.
* - The function catches and logs any exceptions that occur during the process.
*
* @potential_fixes
* - Ensure that the input parameters are correctly initialized and valid.
* - Check the size of the energy vector `_E` before calling the function.
* - Handle exceptions appropriately to avoid unexpected termination.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
bool NQS<_spinModes, _Ht, _T, _stateType>::collectStep(size_t i, const NQS_train_t& _par, 
                                                        NQSAv::MeasurementNQS<_T>& _meas,
                                                        arma::Col<_T>* _E, 
                                                        arma::Col<_T>* _EM,           
                                                        arma::Col<_T>* _ES, 
                                                        const bool quiet,
                                                        const bool randomStart,
                                                        Timer& _timer)
{
    try
    {
        const size_t _startElem = (_par.nblck_ * (i - 1));          // start element for the energy
        const bool _collectEn   = _E != nullptr;
        if (_collectEn)
            assert(_E->size() >= _startElem + _par.nblck_);         

        if (randomStart && _par.MC_th_ > 0) 
            this->setRandomState();									// set the random state at the begining
        this->blockSample(_par.MC_th_, NQS_STATE, !randomStart);	// thermalize the system - burn-in
        
        for (uint _taken = 0; _taken < _par.nblck_; ++_taken)       // iterate blocks - allows to collect samples outside of the block
        {
            this->blockSample(_par.bsize_, NQS_STATE, false);       // sample them using the local Metropolis sampling - sample the states

            if (_collectEn)
                (*_E)(_startElem + _taken) = this->locEnKernel();   // local energy - stored at each point within the estimation of the gradient (stochastic)
            _meas.measure(NQS_STATE, this->pRatioFunc_);            // measure the operators...
            _meas.normalize(1);							            // normalize the measurements - save each sample individually
        }
        if (_collectEn)                                             // save the mean energy and the standard deviation
			MonteCarlo::blockmean(_E->subvec(_startElem, _startElem + _par.nblck_ - 1), _par.bsize_, &(*_EM)(i - 1), &(*_ES)(i - 1));
        PROGRESS_UPD_Q(i, this->pBar_, "PROGRESS NQS", !quiet); 	// update the progress bar
	}
    catch (const std::exception& e) {
        LOGINFO("Error in the NQS collection step: " + std::string(e.what()), LOG_TYPES::ERROR, 1);
        return false;
    }
    return true;
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, collectStep, bool, (size_t, const NQS_train_t&, NQSAv::MeasurementNQS<double>&, arma::Col<double>*, arma::Col<double>*, arma::Col<double>*, const bool, const bool, Timer&));
NQS_INST_CMB(double, std::complex<double>, collectStep, bool, (size_t, const NQS_train_t&, NQSAv::MeasurementNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, const bool, const bool, Timer&));
NQS_INST_CMB(std::complex<double>, double, collectStep, bool, (size_t, const NQS_train_t&, NQSAv::MeasurementNQS<double>&, arma::Col<double>*, arma::Col<double>*, arma::Col<double>*, const bool, const bool, Timer&));
NQS_INST_CMB(std::complex<double>, std::complex<double>, collectStep, bool, (size_t, const NQS_train_t&, NQSAv::MeasurementNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, const bool, const bool, Timer&));

// ##########################################################################################################################################

/**
* @brief Collects the samples for the given number of Monte Carlo steps. The samples are collected in the form of the measurements
* of the operator. The operator is given as a function that calculates the probability of the operator.
* @param _par parameters for the training
* @param _meas measurement object to store the measurements
* @param _collectEn collect the energy as well
* @param _t timepoint for timestamping the training
* @param quiet wanna talk? (default is false)
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::collect(const NQS_train_t& _par,
																	NQSAv::MeasurementNQS<_T>& _meas,
																	arma::Col<_T>* _energies,
																	arma::Col<_T>* _energiesStd,
																	bool quiet,
                                                                    bool randomStart,
																	clk::time_point _t,
																	uint progPrc)
{									
	const bool _collectEn = _energies != nullptr;
	arma::Col<_T> meanEn;
	{
		this->pBar_	= pBar(progPrc, _par.MC_sam_);
		_par.hi("Collect: ");

		if (_collectEn) 
        {
            if (_energies)
                _energies->resize(_par.MC_sam_ * _par.nblck_);
            if (_energiesStd)
                _energiesStd->resize(_par.MC_sam_);
			meanEn 	= arma::Col<_T>(_par.MC_sam_, arma::fill::zeros);
		}
	}
    Timer _timer;
	this->setRandomFlipNum(_par.nFlip);                             // set the random state at the begining
    this->total_    = 0;                                            // reset the total number of flips
    this->accepted_ = 0;                                            // reset the number of accepted flips
	
    for (uint i = 1; i <= _par.MC_sam_; ++i)
	{
        if (!this->collectStep(i, _par, _meas, _energies, &meanEn, _energiesStd, quiet, randomStart, _timer))
            break;        
    }

	LOGINFO(_t, "NQS_COLECTION_" + STR(this->lower_states_.f_lower_size_), 1);
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, collect, void, (const NQS_train_t&, NQSAv::MeasurementNQS<double>&, arma::Col<double>*, arma::Col<double>*, bool, bool, clk::time_point, uint));
NQS_INST_CMB(double, std::complex<double>, collect, void, (const NQS_train_t&, NQSAv::MeasurementNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, bool, bool, clk::time_point, uint));
NQS_INST_CMB(std::complex<double>, double, collect, void, (const NQS_train_t&, NQSAv::MeasurementNQS<double>&, arma::Col<double>*, arma::Col<double>*, bool, bool, clk::time_point, uint));
NQS_INST_CMB(std::complex<double>, std::complex<double>, collect, void, (const NQS_train_t&, NQSAv::MeasurementNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, bool, bool, clk::time_point, uint));

// ##########################################################################################################################################

/**
* @brief Collects a step in the NQS training process.
* 
* This function performs a collection step during the training of a Neural Quantum State (NQS). 
* It thermalizes the system, samples states using local Metropolis sampling, and calculates 
* local energies and operator values.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type.
* @tparam _stateType State type.
* @tparam _CT Container type for operator values and energies.
* 
* @param i Current step index.
* @param _par Training parameters.
* @param _opG Operator for NQS.
* @param _opvals Pointer to container for operator values.
* @param _energies Pointer to container for energies.
* 
* @return True if the step was successfully collected, false otherwise.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
template <typename _CT>
bool NQS<_spinModes, _Ht, _T, _stateType>::collectStep(size_t i, const NQS_train_t& _par, Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies)
{
	try
    {
        const size_t _startElem = (_par.nblck_ * (i - 1));          // start element for the containers
#ifdef _DEBUG
        if (_energies) 
		{
			if constexpr (std::is_same_v<_CT, v_1d<_T>>)
				assert(_energies->size() >= _startElem + _par.nblck_);
			else if constexpr (std::is_same_v<_CT, arma::Col<_T>>)
				assert(_energies->n_elem >= _startElem + _par.nblck_);
		}

		if (_opvals)
		{
			if constexpr (std::is_same_v<_CT, v_1d<_T>>)
				assert(_opvals->size() >= _startElem + _par.nblck_);
			else if constexpr (std::is_same_v<_CT, arma::Col<_T>>)
				assert(_opvals->n_elem >= _startElem + _par.nblck_);
		}
#endif

		// -------------------------------------------------------- // train phase
		
        this->blockSample(_par.MC_th_, NQS_STATE, true);			// thermalize the system - burn-in
        
        for (uint _taken = 0; _taken < _par.nblck_; ++_taken)       // iterate blocks - allows to collect samples outside of the block
        {
            this->blockSample(_par.bsize_, NQS_STATE, false);       // sample them using the local Metropolis sampling - sample the states

            if (_energies) {
				if constexpr (std::is_same_v<_CT, v_1d<_T>>)
					(*_energies)[_startElem + _taken] = this->locEnKernel();   // local energy - stored at each point within the estimation of the gradient (stochastic)
				else if constexpr (std::is_same_v<_CT, arma::Col<_T>>)
					(*_energies)(_startElem + _taken) = this->locEnKernel();   // local energy - stored at each point within the estimation of the gradient (stochastic)
			}

			const auto _val = _opG(NQS_STATE, this->pRatioFunc_);	// calculate the operator value
			if constexpr (std::is_same_v<_CT, v_1d<_T>>)
				(*_opvals)[_startElem + _taken] = _val;				// calculate the operator value
			else if constexpr (std::is_same_v<_CT, arma::Col<_T>>)
				(*_opvals)(_startElem + _taken) = _val;				// calculate the operator value
        }
	}
    catch (const std::exception& e) {
        LOGINFO("Error in the NQS collection step: " + std::string(e.what()), LOG_TYPES::ERROR, 1);
        return false;
    }
    return true;
}

// template instantiation of function above for <spins, double and complex, double and complex, double> and v_1d - container has the type of the nqs data type
// v_1d
template bool NQS<2u, double, double, double>::collectStep(size_t, const NQS_train_t&, Operators::OperatorNQS<double>&, v_1d<double>*, v_1d<double>*);
template bool NQS<2u, double, std::complex<double>, double>::collectStep(size_t, const NQS_train_t&, Operators::OperatorNQS<std::complex<double>>&, v_1d<std::complex<double>>*, v_1d<std::complex<double>>*);
template bool NQS<2u, std::complex<double>, double, double>::collectStep(size_t, const NQS_train_t&, Operators::OperatorNQS<double>&, v_1d<double>*, v_1d<double>*);
template bool NQS<2u, std::complex<double>, std::complex<double>, double>::collectStep(size_t, const NQS_train_t&, Operators::OperatorNQS<std::complex<double>>&, v_1d<std::complex<double>>*, v_1d<std::complex<double>>*);
// arma::Col
template bool NQS<2u, double, double, double>::collectStep(size_t, const NQS_train_t&, Operators::OperatorNQS<double>&, arma::Col<double>*, arma::Col<double>*);
template bool NQS<2u, double, std::complex<double>, double>::collectStep(size_t, const NQS_train_t&, Operators::OperatorNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*);
template bool NQS<2u, std::complex<double>, double, double>::collectStep(size_t, const NQS_train_t&, Operators::OperatorNQS<double>&, arma::Col<double>*, arma::Col<double>*);
template bool NQS<2u, std::complex<double>, std::complex<double>, double>::collectStep(size_t, const NQS_train_t&, Operators::OperatorNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*);

// ##########################################################################################################################################

/**
* @brief Collects data for the Neural Quantum State (NQS) optimization process.
*
* This function performs the collection of operator values and energies during the 
* Monte Carlo sampling process for the NQS optimization. It can optionally reset 
* the state to a random configuration before starting the collection.
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the NQS.
* @tparam _stateType State type for the NQS.
* @tparam _CT Container type for operator values and energies.
* 
* @param _par Parameters for the NQS training process.
* @param _opG Operator for the NQS.
* @param _opvals Pointer to the container for storing operator values.
* @param _energies Pointer to the container for storing energies.
* @param reset Flag indicating whether to reset the state to a random configuration.
* 
* @throws std::runtime_error If _opvals is not initialized.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
template <typename _CT>
void NQS<_spinModes, _Ht, _T, _stateType>::collect(const NQS_train_t& _par, Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies, bool reset)
{
	if (!_opvals)
		throw std::runtime_error("NQS::collect: _opvals is not initialized!");
	
	if (reset) {
		this->setRandomFlipNum(_par.nFlip);                        	// set the random state at the begining
		this->setRandomState();										// set the random state at the begining and the number of flips
	}

#ifdef _DEBUG
	if (_energies) 
	{
		if constexpr (std::is_same_v<_CT, v_1d<_T>>)
		{
			if (_energies->size() != _par.MC_sam_ * _par.nblck_)
				_energies->resize(_par.MC_sam_ * _par.nblck_);
		}
		else if constexpr (std::is_same_v<_CT, arma::Col<_T>>)
		{
			if (_energies->n_elem != _par.MC_sam_)
				_energies->resize(_par.MC_sam_);
		}
	}
	if (_opvals) 
	{
		if constexpr (std::is_same_v<_CT, v_1d<_T>>)
		{
			if (_opvals->size() != _par.MC_sam_ * _par.nblck_)
				_opvals->resize(_par.MC_sam_ * _par.nblck_);
		}
		else if constexpr (std::is_same_v<_CT, arma::Col<_T>>)
		{
			if (_opvals->n_elem != _par.MC_sam_)
				_opvals->resize(_par.MC_sam_);
		}
	}
#endif

	for (uint i = 1; i <= _par.MC_sam_; ++i)
	{
		if (!this->collectStep(i, _par, _opG, _opvals, _energies))
			break;
	}
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
// v_1d
NQS_INST_CMB(double, double, collect, void, (const NQS_train_t&, Operators::OperatorNQS<double>&, v_1d<double>*, v_1d<double>*, bool));
NQS_INST_CMB(double, std::complex<double>, collect, void, (const NQS_train_t&, Operators::OperatorNQS<std::complex<double>>&, v_1d<std::complex<double>>*, v_1d<std::complex<double>>*, bool));
NQS_INST_CMB(std::complex<double>, double, collect, void, (const NQS_train_t&, Operators::OperatorNQS<double>&, v_1d<double>*, v_1d<double>*, bool));
NQS_INST_CMB(std::complex<double>, std::complex<double>, collect, void, (const NQS_train_t&, Operators::OperatorNQS<std::complex<double>>&, v_1d<std::complex<double>>*, v_1d<std::complex<double>>*, bool));
// arma::Col
NQS_INST_CMB(double, double, collect, void, (const NQS_train_t&, Operators::OperatorNQS<double>&, arma::Col<double>*, arma::Col<double>*, bool));
NQS_INST_CMB(double, std::complex<double>, collect, void, (const NQS_train_t&, Operators::OperatorNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, bool));
NQS_INST_CMB(std::complex<double>, double, collect, void, (const NQS_train_t&, Operators::OperatorNQS<double>&, arma::Col<double>*, arma::Col<double>*, bool));
NQS_INST_CMB(std::complex<double>, std::complex<double>, collect, void, (const NQS_train_t&, Operators::OperatorNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, bool));

// ##########################################################################################################################################
