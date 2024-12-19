#include "../../include/NQS/nqs_final.hpp"

// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

// ##########################################################################################################################################

/**
* @brief Computes the gradient of the loss function with respect to the network parameters.
*
* This function calculates the gradient of the loss function using the covariance derivatives
* and updates the learning rate based on the scheduler. It also accounts for the derivatives
* of the lower states if they are used.
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Numeric type for calculations.
* @tparam _stateType Type of the state.
* @param _energies Container holding the energy values.
* @param _step Current step in the optimization process.
* @param _currLoss Current loss value.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::gradF(const Container_t& _energies, int _step,  _T _currLoss)
{
	this->info_p_.lr_ = this->info_p_.lr(_step, algebra::real(_currLoss));										// calculate current learning rate based on the scheduler
	const _T _samples = static_cast<_T>(_energies.n_elem);
	{
		// calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}> 
		// [+ sum_i ^{n-1} \beta _i <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> <Psi _W/Psi_W(i)>] 
		// - for the excited states, the derivatives are appe	
		// calculate the centered derivatives
		this->derivativesMean_ 			= arma::mean(this->derivatives_, 0);									// calculate the mean of the derivatives
		this->derivativesCentered_ 		= this->derivatives_.each_row() - this->derivativesMean_;				// calculate the centered derivatives
		this->derivativesCenteredH_		= arma::trans(this->derivativesCentered_);								// calculate the transposed centered derivatives
		this->F_						= this->derivativesCenteredH_ * ((_energies - _currLoss) / _samples);	// calculate the covariance vector for the gradient 

#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threads_.threadNum_)
#endif
		for (int _low = 0; _low < this->lower_states_.f_lower_size_; _low++)			// append with the lower states derivatives - if the lower states are used
		{
			// Calculate <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> 
			const auto& ratios_excited	= this->lower_states_.ratios_excited_[_low];	// <Psi_W_j / Psi_W> evaluated at W - column vector
			const auto& ratios_lower 	= this->lower_states_.ratios_lower_[_low];		// <Psi_W / Psi_W_j> evaluated at W_j - column vector
			const auto& f_lower_b 		= this->lower_states_.f_lower_b_[_low];			// penalty for the lower states 
			const _T _meanLower 		= arma::mean(ratios_lower);						// mean of the ratios in the lower state
			const _T _meanExcited 		= arma::mean(ratios_excited);					// mean of the ratios in the excited state
			this->F_ 					+= this->derivativesCenteredH_ * ((ratios_excited - _meanExcited) * (f_lower_b * _meanLower / _samples));
		}
	}
}
// template instantiation of function above for <spins, double and complex, double and complex, double>
template void NQS<2u, double, double, double>::gradF(const arma::Col<double>&, int, double);
template void NQS<2u, std::complex<double>, std::complex<double>, double>::gradF(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<2u, std::complex<double>, double, double>::gradF(const arma::Col<double>&, int, double);
template void NQS<2u, double, std::complex<double>, double>::gradF(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<3u, double, double, double>::gradF(const arma::Col<double>&, int, double);
template void NQS<3u, std::complex<double>, std::complex<double>, double>::gradF(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<3u, std::complex<double>, double, double>::gradF(const arma::Col<double>&, int, double);
template void NQS<3u, double, std::complex<double>, double>::gradF(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<4u, double, double, double>::gradF(const arma::Col<double>&, int, double);
template void NQS<4u, std::complex<double>, std::complex<double>, double>::gradF(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<4u, std::complex<double>, double, double>::gradF(const arma::Col<double>&, int, double);
template void NQS<4u, double, std::complex<double>, double>::gradF(const arma::Col<cpx>&, int, std::complex<double>);

// ##########################################################################################################################################

#ifdef NQS_USESR
/**
* @brief Calculates the update parameters for weights using the stochastic reconfiguration (SR) method.
* 
* This function updates the weights using a second-order optimization method, where the update rule is:
* weights[new] <- weights[old] - lr * S^{-1} * F
* Here, S is the geometric tensor, calculated as S = <\Delta _k* \Delta _k> - <\Delta _k*><\Delta _k>.
* The geometric tensor acts as a regularizer for the gradient descent method.
* 
* The method can use either the pseudoinverse or direct inversion of the matrix S. If the stochastic reconfiguration
* is skipped, the standard gradient descent is used instead. The function can also run without explicitly calculating
* the geometric tensor S.
* 
* @param step Current step of updating, used for regularization purposes.
* @param _currLoss Current loss value, used for regularization purposes.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::gradSR(uint step, _T _currLoss)
{
	bool _inversionSuccess 		= false;
	if (this->info_p_.sreg_ > 0) 
		this->covMatrixReg(step);

#ifdef NQS_USESR_MAT_USED
	{
		// regularize the covariance matrix before inverting it (if needed and set)

		// calculate the pseudoinverse
		int _attempts 			= 0;
		double _regFactor		= this->info_p_.sreg_ > 0 ? this->info_p_.sreg_ : 1e-5;

		while (!_inversionSuccess && _attempts < NQS_SREG_ATTEMPTS)
		{
			try {
				if (this->info_p_.pinv_ > 0.0)
					this->dF_ = this->info_p_.lr_ * (arma::pinv(this->S_, this->info_p_.pinv_) * this->F_);
				else if (this->info_p_.pinv_ == 0.0)
					this->dF_ = this->info_p_.lr_ * (arma::pinv(this->S_) * this->F_);
				else 
					this->dF_ = this->info_p_.lr_ * arma::solve(this->S_, this->F_, arma::solve_opts::likely_sympd);
				
				// if the inversion was successful, set the flag
				_inversionSuccess = true;
			} catch (std::exception& e) {
                // Increase regularization factor and apply to diagonal
				LOGINFO("Inverse calculation failed, attempt " + STR(_attempts + 1) + ". E: " + e.what(), LOG_TYPES::ERROR, 1);
				if (!arma::is_finite(this->S_)) {
					LOGINFO("Non-finite values in the diagonal of the covariance matrix. Stopping the training.", LOG_TYPES::ERROR, 1);
					break;
				}

                this->S_.diag() += _regFactor;  	// Apply regularization to diagonal
                _regFactor 		*= 10;  			// Increase regularization factor for next attempt
                ++_attempts;
			}
		}
	}
#else
	if (this->precond_ != nullptr)
		this->precond_->set(this->derivativesCenteredH_, this->derivativesCentered_, -1.0);
	
	if (this->solver_ != nullptr) 
	{
		this->solver_->setReg(this->info_p_.sreg_);											// set the regularization						
		this->solver_->solve(this->derivativesCentered_, this->derivativesCenteredH_, 		// S and S+ matrices
							this->F_, 														// b
							nullptr, //step <= 1 ? nullptr : &this->dF_, 								// x0
							this->precond_);												// preconditioner
		_inversionSuccess = this->solver_->isConverged();
		this->dF_ = this->info_p_.lr_ * this->solver_->moveSolution();						// get the solution
	} 
	else {
		// !DEPRECATED - use solver class instead
		this->dF_ = this->info_p_.lr_ * algebra::Solvers::FisherMatrix::solve<_T>(
										this->info_p_.solver_,													// choose the solver type 
										this->derivativesCentered_,     										// Ensure this matches the type expected by _gramMatrix
										this->derivativesCenteredH_,											// This should also match arma::Col<_T>
										this->F_,               												// This should be of type arma::Col<_T>
										nullptr, //step <= 1 ? nullptr : &this->dF_,										// This should also match arma::Col<_T>
										this->precond_ ? this->precond_ : nullptr, 								// Preconditioner
										this->info_p_.tol_,                  		 							// Tolerance
										std::min(size_t(5 * this->F_.n_elem), size_t(this->info_p_.maxIter_)),	// Max iterations,
										&_inversionSuccess,														// Convergence flag						
										this->info_p_.sreg_ //this->precond_ ? -1.0 : this->info_p_.sreg_								// Set the regularization only if no preconditioner is used 
										);
	}

#endif
    this->updateWeights_ = _inversionSuccess;
}
// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, gradSR, void, (uint, double));
NQS_INST_CMB(std::complex<double>, std::complex<double>, gradSR, void, (uint, std::complex<double>));
NQS_INST_CMB(std::complex<double>, double, gradSR, void, (uint, double));
NQS_INST_CMB(double, std::complex<double>, gradSR, void, (uint, std::complex<double>));
#endif

// ##########################################################################################################################################

/**
* @brief Computes the final gradient for the Neural Quantum State (NQS) optimization.
*
* This function calculates the gradient for the NQS optimization process, taking into account
* the current learning rate, the centered derivatives, and the covariance vector. It also
* includes the derivatives for the lower states if they are used. The function can optionally
* apply stochastic reconfiguration with or without matrix calculation.
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the calculations.
* @tparam _stateType State type.
* @param _energies Energies of the NQS.
* @param _step Current optimization step.
* @param _currLoss Current loss value.
 */
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::gradFinal(const Container_t& _energies, int _step, _T _currLoss)
{
	this->gradF(_energies, _step, _currLoss);					// calculate the gradient of the loss function
#ifdef NQS_USESR_MAT_USED										// ---- STOCHASTIC RECONFIGURATION WITH MATRIX CALCULATION ----
	// update model by recalculating the gradient (applying the stochastic reconfiguration)
	// this->S_ = arma::cov(this->derivativesC_, this->derivatives_, 1);
	this->S_ = this->derivativesCenteredH_ * this->derivativesCentered_ / _samples;
	
	// check the norm of the gradient and normalize it if needed
	// if (auto gradNorm = arma::norm(this->F_); gradNorm > NQS_SREG_GRAD_NORM_THRESHOLD)
	// {
	// 	std::cerr << "Gradient norm is too large: " << gradNorm << std::endl;
	// 	this->F_ *= NQS_SREG_GRAD_NORM_THRESHOLD / gradNorm;
	// }
#endif
#if defined NQS_USESR											// ---- STOCHASTIC RECONFIGURATION POSSIBLY WITHOUT MATRIX CALCULATION ----
	return this->gradSR(_step, _currLoss);
#else
	// standard updater with the gradient only!
	this->updateWeights_ = true;
	this->dF_ = this->info_p_.lr_ * this->F_;
#endif
}
// template instantiation of function above for <spins, double and complex, double and complex, double>
template void NQS<2u, double, double, double>::gradFinal(const arma::Col<double>&, int, double);
template void NQS<2u, std::complex<double>, std::complex<double>, double>::gradFinal(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<2u, std::complex<double>, double, double>::gradFinal(const arma::Col<double>&, int, double);
template void NQS<2u, double, std::complex<double>, double>::gradFinal(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<3u, double, double, double>::gradFinal(const arma::Col<double>&, int, double);
template void NQS<3u, std::complex<double>, std::complex<double>, double>::gradFinal(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<3u, std::complex<double>, double, double>::gradFinal(const arma::Col<double>&, int, double);
template void NQS<3u, double, std::complex<double>, double>::gradFinal(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<4u, double, double, double>::gradFinal(const arma::Col<double>&, int, double);
template void NQS<4u, std::complex<double>, std::complex<double>, double>::gradFinal(const arma::Col<cpx>&, int, std::complex<double>);
template void NQS<4u, std::complex<double>, double, double>::gradFinal(const arma::Col<double>&, int, double);
template void NQS<4u, double, std::complex<double>, double>::gradFinal(const arma::Col<cpx>&, int, std::complex<double>);

// ##########################################################################################################################################

// ############################################################ T R A I N I N G #############################################################

// ##########################################################################################################################################b

#include <cmath>
#include <utility>
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
bool NQS<_spinModes, _Ht, _T, _stateType>::trainStop(size_t i, const MonteCarlo::MCS_train_t& _par, _T _currLoss, _T _currstd, bool _quiet)
{
	const auto best			= this->info_p_.best();
	const double acceptance = (double)this->accepted_ / this->total_ * 100.0;

	double coolingRate 		= 1.001; 	// Cooling rate (how fast to decrease beta)

	// If acceptance rate is too low, adjust beta
	if (acceptance < 5e-2) 
	{
		// this->setRandomState(); 		// Reset the state
		this->beta_ *= coolingRate;
		LOGINFO("Acceptance rate is too low: " + STR(acceptance) + "%. Changing beta: " + STRP(this->beta_, 3), LOG_TYPES::DEBUG, 3);
	}

	const std::string _prog = "Iteration " + STR(i) + "/" + STR(_par.MC_sam_) +
								", Loss: " + STRPS(_currLoss, 4) + " ± " + STRPS(_currstd / 2.0, 3) +
								", Best: " + VEQPS(best, 4) + 
								", Acceptance: " + STR(this->accepted_) + "/" + STR(this->total_) + " (" + STRP(acceptance, 2) + "%)" +
								", LR: " + STRPS(this->info_p_.lr_, 4) +
								", Reg: " + STRPS(this->info_p_.sreg_, 4) + 
								", Beta: " + STRPS(this->beta_, 4);

	PROGRESS_UPD_Q(i, (*this->pBar_), _prog, !_quiet);
	
	this->updateWeights_ 	= !this->info_p_.stop(i, _currLoss) && this->updateWeights_;
#ifdef NQS_SAVE_WEIGHTS
	if ((i % this->pBar_->percentageSteps == 0) || !this->updateWeights_) 
		this->saveWeights(_par.dir + NQS_SAVE_DIR, "weights_" + STR(this->lower_states_.f_lower_size_) + ".h5");
#endif

	if (!this->updateWeights_) {
		LOGINFO("Stopping at iteration " + STR(i) + " with last loss value: " + STRPS(_currLoss, 4) + "+-" + STRPS(_currstd/2.0, 4), LOG_TYPES::WARNING, 1);
		return true;
	}
	return false;
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, trainStop, bool, (size_t, const MonteCarlo::MCS_train_t&, double, double, bool));
NQS_INST_CMB(double, std::complex<double>, trainStop, bool, (size_t, const MonteCarlo::MCS_train_t&, std::complex<double>, std::complex<double>, bool));
NQS_INST_CMB(std::complex<double>, double, trainStop, bool, (size_t, const MonteCarlo::MCS_train_t&, double, double, bool));
NQS_INST_CMB(std::complex<double>, std::complex<double>, trainStop, bool, (size_t, const MonteCarlo::MCS_train_t&, std::complex<double>, std::complex<double>, bool));

// ##########################################################################################################################################

/**
* @brief Performs a single training step for the Neural Quantum State (NQS) optimization.
* 
* This function executes one iteration of the training process, which includes thermalization,
* sampling, gradient calculation, and weight updates. It also handles the calculation of local
* energies and overlaps with excited states if applicable.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for calculations (e.g., float, double).
* @tparam _stateType State type.
* 
* @param i Current iteration index.
* @param En Vector to store local energies for each block.
* @param meanEn Vector to store mean energies for each iteration.
* @param stdEn Vector to store standard deviations of energies for each iteration.
* @param _par Training parameters.
* @param quiet Flag to suppress output if true.
* @param randomStart Flag to initialize with a random state if true.
* @param _timer Timer object for measuring execution time.
* 
* @return True if the training should stop based on the stopping criteria, false otherwise.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
bool NQS<_spinModes, _Ht, _T, _stateType>::trainStep(size_t i, 
													Container_t& En,
													Container_t& meanEn, 
													Container_t& stdEn, 
													const MonteCarlo::MCS_train_t& _par, 
                                                    const bool quiet, 
                                                    const bool randomStart,
                                                    Timer& _timer)
{
	this->total_ 	= 0;										// reset the total number of flips
	this->accepted_ = 0;										// reset the number of accepted flips
	if (randomStart && _par.MC_th_ > 0) {
		this->setRandomState();									// set the random state at the begining
		this->blockSample<false>(_par.MC_th_, NQS_STATE);		// thermalize the system - burn-in
	} else {
		this->blockSample<true>(_par.MC_th_, NQS_STATE);		// thermalize the system - burn-in
	}															// only if the random start is used and the thermalization is used. Otherwise, the random state is set at the beginning of the training

	for (uint _taken = 0; _taken < _par.nblck_; ++_taken) 		// iterate blocks - this ensures the calculation of a stochastic gradient constructed within the block
	{		
		this->blockSample<false>(_par.bsize_, NQS_STATE);		// sample them using the local Metropolis sampling
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
	
	MonteCarlo::blockmean(En, std::max((size_t)_par.bsize_, (size_t)8), &meanEn(i - 1), &stdEn(i - 1)); 					// save the mean energy
	TIMER_START_MEASURE(this->gradFinal(En, i, meanEn(i - 1)), (i % this->pBar_->percentageSteps == 0), _timer, STR(i)); 	// calculate the final update vector - either use the stochastic reconfiguration or the standard gradient descent

	if (this->updateWeights_)
		this->updateWeights(); 									// finally, update the weights with the calculated gradient (force) [can be done with the stochastic reconfiguration or the standard gradient descent] - implementation specific!!!

	if (this->trainStop(i, _par, meanEn(i - 1), stdEn(i - 1), quiet))
		return true;
	return false;
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, trainStep, bool, (size_t, arma::Col<double>&, arma::Col<double>&, arma::Col<double>&, const MonteCarlo::MCS_train_t&, const bool, const bool, Timer&));
NQS_INST_CMB(double, std::complex<double>, trainStep, bool, (size_t, arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&, const MonteCarlo::MCS_train_t&, const bool, const bool, Timer&));
NQS_INST_CMB(std::complex<double>, double, trainStep, bool, (size_t, arma::Col<double>&, arma::Col<double>&, arma::Col<double>&, const MonteCarlo::MCS_train_t&, const bool, const bool, Timer&));
NQS_INST_CMB(std::complex<double>, std::complex<double>, trainStep, bool, (size_t, arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&, const MonteCarlo::MCS_train_t&, const bool, const bool, Timer&));

// ##########################################################################################################################################

template <typename _T>
using NQS_TRAIN_PAIR = std::pair<arma::Col<_T>, arma::Col<_T>>;

/**
* @brief Trains the Neural Quantum State (NQS) using Monte Carlo sampling.
* 
* This function performs the training of the NQS model based on the provided training parameters.
* It initializes necessary components, performs Monte Carlo sampling, and updates the model's
* parameters accordingly. The training process can be monitored using a progress bar and can be
* configured to start from a random state each time.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for numerical operations.
* @tparam _stateType Type of the state representation.
* 
* @param _par Training parameters encapsulated in MonteCarlo::MCS_train_t.
* @param quiet If true, suppresses output during training.
* @param randomStart If true, initializes the training with a random state.
* @param _t Time point for logging purposes.
* @param progPrc Progress percentage for the progress bar.
* 
* @return A pair of arma::Col<_T> vectors containing the mean energy and standard deviation of the energy
*         over the Monte Carlo samples.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS<_spinModes, _Ht, _T, _stateType>::Container_pair_t NQS<_spinModes, _Ht, _T, _stateType>::train(const MonteCarlo::MCS_train_t& _par, 
                                                                                            bool quiet, 
                                                                                            bool randomStart, 
                                                                                            clk::time_point _t, 
                                                                                            uint progPrc)
{
	{
		if (this->pBar_ != nullptr) 
			delete this->pBar_;												// delete the progress bar if it exists
		this->pBar_ = new pBar(progPrc, _par.MC_sam_);						// set the progress bar		
		_par.hi();															// set the info about training
		this->reset(_par.nblck_);											// reset the derivatives				
	}

	Timer _timer;															// timer for the training
	arma::Col<_T> meanEn(_par.MC_sam_, arma::fill::zeros);					// here we save the mean energy
	arma::Col<_T> stdEn(_par.MC_sam_, arma::fill::zeros);					// here we save the standard deviation of the energy
	this->E_ = arma::Col<_T>(_par.nblck_, arma::fill::zeros);				// history of energies (for given weights) - here we save the local energies at each block
	this->setRandomState();													// set the random state at the begining and the number of flips
	this->setRandomFlipNum(_par.nFlip);										// set the random state at the begining and the number of flips

	uint i = 1;
	for (i = 1; i <= _par.MC_sam_; ++i) {									// go through the Monte Carlo steps
		if (this->trainStep(i, this->E_ , meanEn, stdEn, _par, quiet, randomStart, _timer)) // perform the training step
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

NQS_INST_CMB(double, double, train, NQS_TRAIN_PAIR<double>, (const MonteCarlo::MCS_train_t&, bool, bool, clk::time_point, uint));
NQS_INST_CMB(double, std::complex<double>, train, NQS_TRAIN_PAIR<std::complex<double>>, (const MonteCarlo::MCS_train_t&, bool, bool, clk::time_point, uint));
NQS_INST_CMB(std::complex<double>, double, train, NQS_TRAIN_PAIR<double>, (const MonteCarlo::MCS_train_t&, bool, bool, clk::time_point, uint));
NQS_INST_CMB(std::complex<double>, std::complex<double>, train, NQS_TRAIN_PAIR<std::complex<double>>, (const MonteCarlo::MCS_train_t&, bool, bool, clk::time_point, uint));

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
* - The function returns true if the step was successfully collected, false otherwise.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
bool NQS<_spinModes, _Ht, _T, _stateType>::collectStep(size_t i, const MonteCarlo::MCS_train_t& _par, 
                                                        NQSAv::MeasurementNQS<_T>& _meas,
                                                        Container_t* _E, 
                                                        Container_t* _EM,           
                                                        Container_t* _ES, 
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

        if (randomStart && _par.MC_th_ > 0) {
            this->setRandomState();									// set the random state at the begining
			this->blockSample<false>(_par.MC_th_, NQS_STATE);		// thermalize the system - burn-in
		} else {
        	this->blockSample<true>(_par.MC_th_, NQS_STATE);		// thermalize the system - burn-in
		}
        
        for (uint _taken = 0; _taken < _par.nblck_; ++_taken)       // iterate blocks - allows to collect samples outside of the block
        {
            this->blockSample<false>(_par.bsize_, NQS_STATE);       // sample them using the local Metropolis sampling - sample the states

            if (_collectEn)
                (*_E)(_startElem + _taken) = this->locEnKernel();   // local energy - stored at each point within the estimation of the gradient (stochastic)
            _meas.measure(NQS_STATE, this->pRatioFunc_);            // measure the operators...
            _meas.normalize(1);							            // normalize the measurements - save each sample individually
        }
        if (_collectEn) {                                           // save the mean energy and the standard deviation
			if (_EM && _ES)
				MonteCarlo::blockmean(_E->subvec(_startElem, _startElem + _par.nblck_ - 1), std::max(_par.bsize_, uint(8)), &(*_EM)(i - 1), &(*_ES)(i - 1));
			else if (_EM && !_ES)
				MonteCarlo::blockmean(_E->subvec(_startElem, _startElem + _par.nblck_ - 1), std::max(_par.bsize_, uint(8)), &(*_EM)(i - 1));
		}
        PROGRESS_UPD_Q(i, (*this->pBar_), "PROGRESS NQS", !quiet); 	// update the progress bar
	}
    catch (const std::exception& e) {
        LOGINFO("Error in the NQS collection step: " + std::string(e.what()), LOG_TYPES::ERROR, 1);
        return false;
    }
    return true;
}

// template instantiation of function above for <spins, double and complex, double and complex, double>
NQS_INST_CMB(double, double, collectStep, bool, (size_t, const MonteCarlo::MCS_train_t&, NQSAv::MeasurementNQS<double>&, arma::Col<double>*, arma::Col<double>*, arma::Col<double>*, const bool, const bool, Timer&));
NQS_INST_CMB(double, std::complex<double>, collectStep, bool, (size_t, const MonteCarlo::MCS_train_t&, NQSAv::MeasurementNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, const bool, const bool, Timer&));
NQS_INST_CMB(std::complex<double>, double, collectStep, bool, (size_t, const MonteCarlo::MCS_train_t&, NQSAv::MeasurementNQS<double>&, arma::Col<double>*, arma::Col<double>*, arma::Col<double>*, const bool, const bool, Timer&));
NQS_INST_CMB(std::complex<double>, std::complex<double>, collectStep, bool, (size_t, const MonteCarlo::MCS_train_t&, NQSAv::MeasurementNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, const bool, const bool, Timer&));

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
void NQS<_spinModes, _Ht, _T, _stateType>::collect(const MonteCarlo::MCS_train_t& _par,
																	NQSAv::MeasurementNQS<_T>& _meas,
																	Container_t* _energies,
																	Container_t* _energiesStd,
																	bool quiet,
                                                                    bool randomStart,
																	clk::time_point _t,
																	uint progPrc)
{									
	const bool _collectEn = _energies != nullptr;
	arma::Col<_T> meanEn;
	{
		if (this->pBar_ != nullptr) 
			delete this->pBar_;
		this->pBar_	= new pBar(progPrc, _par.MC_sam_);
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
NQS_INST_CMB(double, double, collect, void, (const MonteCarlo::MCS_train_t&, NQSAv::MeasurementNQS<double>&, arma::Col<double>*, arma::Col<double>*, bool, bool, clk::time_point, uint));
NQS_INST_CMB(double, std::complex<double>, collect, void, (const MonteCarlo::MCS_train_t&, NQSAv::MeasurementNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, bool, bool, clk::time_point, uint));
NQS_INST_CMB(std::complex<double>, double, collect, void, (const MonteCarlo::MCS_train_t&, NQSAv::MeasurementNQS<double>&, arma::Col<double>*, arma::Col<double>*, bool, bool, clk::time_point, uint));
NQS_INST_CMB(std::complex<double>, std::complex<double>, collect, void, (const MonteCarlo::MCS_train_t&, NQSAv::MeasurementNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, bool, bool, clk::time_point, uint));

// ##########################################################################################################################################

/**
* @brief Collects a step in the NQS process. This allows for acting on the state with a single global operator (just a single value returned for 
* the expectation value of the operator).
* 
* This function performs a collection step (also during the training of a Neural Quantum State). 
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
bool NQS<_spinModes, _Ht, _T, _stateType>::collectStep(size_t i, const MonteCarlo::MCS_train_t& _par, Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies)
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

        this->blockSample<false>(_par.MC_th_, NQS_STATE);			// thermalize the system - burn-in - does not need to set the state at the beginning
        for (uint _taken = 0; _taken < _par.nblck_; ++_taken)       // iterate blocks - allows to collect samples outside of the block
        {
            this->blockSample<false>(_par.bsize_, NQS_STATE);       // sample them using the local Metropolis sampling - sample the states

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
template bool NQS<2u, double, double, double>::collectStep(size_t, const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<double>&, v_1d<double>*, v_1d<double>*);
template bool NQS<2u, double, std::complex<double>, double>::collectStep(size_t, const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<std::complex<double>>&, v_1d<std::complex<double>>*, v_1d<std::complex<double>>*);
template bool NQS<2u, std::complex<double>, double, double>::collectStep(size_t, const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<double>&, v_1d<double>*, v_1d<double>*);
template bool NQS<2u, std::complex<double>, std::complex<double>, double>::collectStep(size_t, const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<std::complex<double>>&, v_1d<std::complex<double>>*, v_1d<std::complex<double>>*);
// arma::Col
template bool NQS<2u, double, double, double>::collectStep(size_t, const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<double>&, arma::Col<double>*, arma::Col<double>*);
template bool NQS<2u, double, std::complex<double>, double>::collectStep(size_t, const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*);
template bool NQS<2u, std::complex<double>, double, double>::collectStep(size_t, const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<double>&, arma::Col<double>*, arma::Col<double>*);
template bool NQS<2u, std::complex<double>, std::complex<double>, double>::collectStep(size_t, const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*);

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
void NQS<_spinModes, _Ht, _T, _stateType>::collect(const MonteCarlo::MCS_train_t& _par, Operators::OperatorNQS<_T>& _opG, _CT* _opvals, _CT* _energies, bool reset)
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
			if (_opvals->n_elem != _par.MC_sam_ * _par.nblck_)
				_opvals->resize(_par.MC_sam_ * _par.nblck_);
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
NQS_INST_CMB(double, double, collect, void, (const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<double>&, v_1d<double>*, v_1d<double>*, bool));
NQS_INST_CMB(double, std::complex<double>, collect, void, (const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<std::complex<double>>&, v_1d<std::complex<double>>*, v_1d<std::complex<double>>*, bool));
NQS_INST_CMB(std::complex<double>, double, collect, void, (const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<double>&, v_1d<double>*, v_1d<double>*, bool));
NQS_INST_CMB(std::complex<double>, std::complex<double>, collect, void, (const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<std::complex<double>>&, v_1d<std::complex<double>>*, v_1d<std::complex<double>>*, bool));
// arma::Col
NQS_INST_CMB(double, double, collect, void, (const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<double>&, arma::Col<double>*, arma::Col<double>*, bool));
NQS_INST_CMB(double, std::complex<double>, collect, void, (const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, bool));
NQS_INST_CMB(std::complex<double>, double, collect, void, (const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<double>&, arma::Col<double>*, arma::Col<double>*, bool));
NQS_INST_CMB(std::complex<double>, std::complex<double>, collect, void, (const MonteCarlo::MCS_train_t&, Operators::OperatorNQS<std::complex<double>>&, arma::Col<std::complex<double>>*, arma::Col<std::complex<double>>*, bool));

// ##########################################################################################################################################