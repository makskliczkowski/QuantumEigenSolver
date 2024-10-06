#include "../include/nqs.h"

// ##########################################################################################################################################

/*
* @brief Say hello to the NQS solver.
*/
void NQS_train_t::hi(const std::string& _in) const
{
    std::string outstr	= "";
    strSeparatedP(outstr, '\t', 2,
                VEQV(Monte Carlo Steps, this->mcSteps),
                VEQV(Thermalization Steps, this->nThrm),
                VEQV(Block Number, this->nBlck),
                VEQV(Size of the single block, this->bSize),
                VEQV(Number of flips taken at each step, this->nFlip));
    LOGINFOG(_in + outstr, LOG_TYPES::TRACE, 1);
}

// // ##########################################################################################################################################

// // ############################################################ S A M P L I N G #############################################################

// // ##########################################################################################################################################

// /*
// * @brief Block updates the current state according to Metropolis-Hastings algorithm. The block size is chosen so that
// * it gives the given statistics of the averages found.
// * @param _bSize the size of the correlation block; how many flips one needs to do in order to get rid of the MC correlations - bin size 
// * @param _start the state to start from
// * @param _nFlip number of flips in a single step
// */
// template<uint _spinModes, typename _Ht, typename _T, class _stateType>
// void NQS<_spinModes, _Ht, _T, _stateType>::blockSample(uint _bSize, NQS_STATE_T _start, bool _therm)
// {
// 	// check whether we should set a state again or thermalize the whole process (applies on integer state)
// 	// Set state based on whether thermalization is required or _start differs from current state
// 	if (_therm 
// #ifndef NQS_USE_VEC_ONLY
// 		|| _start != this->curState_
// #endif
// 		) this->setState(_start, _therm);

// 	// set the temporary state - the vectors are usefull so set them accordingly
// 	this->tmpVec_ = this->curVec_;

// 	for (uint bStep = 0; bStep < _bSize; ++bStep) // go through each block step
// 	{
// 		this->chooseRandomFlips(); 	// set the random flip sites - it depends on a given implementation of the NQS
// 		this->applyFlipsT();		// flip the vector - use temporary vector tmpVec to store the flipped vector

// 		// check the probability (choose to use the iterative update of presaved weights [the angles previously updated] or calculate ratio from scratch)
// #ifndef NQS_ANGLES_UPD
// 		double proba = std::abs(this->pRatio(this->curVec_, this->tmpVec_));
// #else
// 		double proba = std::abs(this->pRatio(this->nFlip_));
// #endif
// 		// we need to take into account the probability comming from the ratio of states (after and before the flip)
// 		if (this->ran_.template random<float>() <= proba * proba)
// 		{
// 			// update current state and vector when the flip has been accepted (the probability is higher than the random number)
// 			this->applyFlipsC();
// 			// update angles if needed
// 			this->update(this->nFlip_);
// 		}
// 		// set the vector back to normal (unflip)
// 		else
// 		{
// 			this->unupdate();
// 			this->unapplyFlipsT();
// 		}
// 	}

// 	// set the current integer state
// #ifndef NQS_USE_VEC_ONLY
// 	this->curState_ = BASE_TO_INT<u64>(this->curVec_, discVal_);
// #endif

// 	// Set the state again if angles update is disabled
// #ifndef NQS_ANGLES_UPD
// 	this->setState(NQS_STATE, true);
// #endif
// }

// ///////////////////////////////////////////////////////////////////////

// #ifdef NQS_NOT_OMP_MT
// /*
// * @brief Allows to run a thread pool based on the condition that all threads wait for a mutex to further run the program.
// * The threads are killed when the program is finished. The function calculates the local energy for a given thread.
// * @param _start starting site for a given thread
// * @param _end ending site for a given thread
// * @param _threadNum number of the thread currently run
// */
// template<uint _spinModes, typename _Ht, typename _T, class _stateType>
// void NQS<_spinModes, _Ht, _T, _stateType>::locEnKernel(uint _start, uint _end, uint _threadNum)
// {
// 	while (!this->kernels_[_threadNum].flagThreadKill_)	// does not go in if the simulation is finished
// 	{
// 		// wait for the lock to end
// 		{
// 			// aquire mutex lock as required by condition variable
// 			std::unique_lock<std::mutex> lock(this->kernels_[_threadNum].mutex);	
// 			// thread will suspend here and release the lock if the expression does not return true
// 			this->kernels_[_threadNum].cv.wait(lock, [this, _threadNum] { return this->kernels_[_threadNum].flagThreadRun_; });	

// 			if (this->kernels_[_threadNum].flagThreadKill_)
// 			{
// 				this->kernels_[_threadNum].end_ = true;
// 				break;
// 			}
// 		}

// 		// Process the work
// 		this->kernels_[_threadNum].kernelValue_ = 0.0;
// 		for (auto site = _start; site < _end; ++site)
// 		{
// 			this->kernels_[_threadNum].kernelValue_ += algebra::cast<_T>(this->H_->locEnergy(NQS_STATE,
// 																							site, 
// 																							std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
// 																							this,
// 																							std::placeholders::_1,
// 																							std::placeholders::_2)));
// 		}
// 		// lock again
// 		{
// 			std::lock_guard<std::mutex> lock(this->kernels_[_threadNum].mutex);
// 			this->kernels_[_threadNum].flagThreadRun_	= false;
// 			this->kernels_[_threadNum].end_				= true; 
// 		}
// 		this->kernels_[_threadNum].cv.notify_one(); // Notify waiting threads if needed
// 	}
// }

// /*
// * @brief Allows to run a thread pool based on the condition that all threads 
// wait for a mutex to further run the program. This is meant for the excited states.
// * The threads are killed when the program is finished. The function calculates the local energy for a given thread.
// * @param _start starting site for a given thread
// * @param _end ending site for a given thread
// * @param _threadNum number of the thread currently run
// */
// template<uint _spinModes, typename _Ht, typename _T, class _stateType>
// void NQS<_spinModes, _Ht, _T, _stateType>::locEnKernelExcited(uint _start, uint _end, uint _threadNum)
// {
// 	while (!this->kernels_[_threadNum].flagThreadKill_)	// does not go in if the simulation is finished
// 	{
// 		// wait for the lock to end
// 		{
// 			// aquire mutex lock as required by condition variable
// 			std::unique_lock<std::mutex> lock(this->kernels_[_threadNum].mutex);	
// 			// thread will suspend here and release the lock if the expression does not return true
// 			this->kernels_[_threadNum].cv.wait(lock, [this, _threadNum] { return this->kernels_[_threadNum].flagThreadRun_; });	

// 			if (this->kernels_[_threadNum].flagThreadKill_)
// 			{
// 				this->kernels_[_threadNum].end_ = true;
// 				break;
// 			}
// 		}

// 		// Process the work
// 		this->kernels_[_threadNum].kernelValue_ = 0.0;
// 		for (auto site = _start; site < _end; ++site)
// 		{
// 			this->kernels_[_threadNum].kernelValue_ += algebra::cast<_T>(this->H_->locEnergy(NQS_STATE,
// 																							site, 
// 																							std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::pKernel,
// 																							this,
// 																							std::placeholders::_1,
// 																							std::placeholders::_2)));
// 			// go through the lower energy states and obtain additional values

// 		}
// 		// lock again
// 		{
// 			std::lock_guard<std::mutex> lock(this->kernels_[_threadNum].mutex);
// 			this->kernels_[_threadNum].flagThreadRun_	= false;
// 			this->kernels_[_threadNum].end_				= true; 
// 		}
// 		this->kernels_[_threadNum].cv.notify_one(); // Notify waiting threads if needed
// 	}
// }
// #endif

// // ##########################################################################################################################################

// // ############################################################ G R A D I E N T #############################################################

// // ##########################################################################################################################################

// // -----------------------------------------------------------------------------------------------------------------------------------
// #ifdef NQS_USESR

// 	#ifdef NQS_USESR_NOMAT
// // -----------------------------------------------------------------------------------------------------------------------------------

// /*
// * @brief Calculates the matrix element of the geometric tensor S_ij = <\Delta _i* \Delta _j> - <\Delta _i*><\Delta _j> 
// * without constructing the matrix explicitly. This is done by calculating the matrix element directly. 
// * @param i row index - the matrix is symmetric
// * @param j column index - the matrix is symmetric
// * @returns the matrix element of the geometric tensor S_ij
// */
// template<uint _spinModes, typename _Ht, typename _T, class _stateType>
// _T NQS<_spinModes, _Ht, _T, _stateType>::getSRMatrixElement(size_t i, size_t j)
// {
// 	// get the subviews of the columns (each column has multiple observations now)
// 	auto _colO_i 		= this->derivativesC_.col(i);
// 	auto _colO_j 		= this->derivatives_.col(j);

// 	// compute <\Delta _i* \Delta _j>
// 	_T avg_Oi_Oj 		= arma::dot(_colO_i, _colO_j) / static_cast<double>(_colO_i.n_elem);

// 	// compute <\Delta _i*><\Delta _j>
// 	_T avg_Oi_avg_Oj 	= algebra::conjugate(this->derivativesM_(i)) * this->derivativesM_(j);

// 		#ifdef NQS_SREG
// 		{
// 		// regularize the covariance matrix before inverting it (if needed and set) by adding the regularization term
// 		if (i == j)
// 			avg_Oi_Oj += this->covMatrixRegStart * avg_Oi_avg_Oj;
// 		}
// 		#endif
// 	return avg_Oi_Oj - avg_Oi_avg_Oj;
// }

// // -----------------------------------------------------------------------------------------------------------------------------------

// /*
// * @brief Calculates the matrix-vector multiplication using the elements obtained from the getSRMatrixElement. 
// * This is done without constructing the matrix explicitly. Once the matrix is known, the multiplication is done
// * by calculating the matrix-vector multiplication directly.
// * @param x vector to be multiplied with the matrix
// * @param y resulting vector from the multiplication
// */
// template <uint _spinModes, typename _Ht, typename _T, class _stateType>
// void NQS<_spinModes, _Ht, _T, _stateType>::getSRMatVec(const arma::Col<_T>& x, arma::Col<_T>& y, size_t n)
// {	
// 	// implement the matrix-vector multiplication using the elements obtained from the getSRMatrixElement
// 	// #pragma omp parallel for num_threads(this->threadNum_)
// 	for (size_t i = 0; i < n; ++i)
// 	{
// 		// initialize the y vector
// 		y(i) = 0.0;

// 		// y_i = \sum _j S_ij x_j
// 		// Compute the matrix-vector multiplication using only upper triangular matrix and reflection
// 		// as the matrix is square and symmetric
// 		for (size_t j = 0; j < n; j++)
// 			y(i) += this->getSRMatrixElement(i, j) * x(j);
// // 		for (size_t j = i; j < n; j++) {
// //             const _T S_ij = this->getSRMatrixElement(i, j);
// //             y(i) += S_ij * x(j);

// //             // Since S is symmetric, reflect the result to y(j)
// //             if (i != j) {
// // // #pragma omp atomic
// //                 y(j) += S_ij * x(i);
// //             }
// //         }
// 	}

// }
// // ----------------------------------------------------------------------------------------------------------------------------------- 
// 	#endif 

// /*
// * @brief Calculates the update parameters for weights to be updated in a form:
// * weights[new] <- weights[old] - lr * S^{-1} * F --> second order optimization method (stochastic reconfiguration)
// * It uses the matrix S = <\Delta _k* \Delta _k> - <\Delta _k*><\Delta _k> to calculate the update. This matrix is 
// * called the geometric tensor. It is used as a regularizer for the gradient descent method.
// * This method can be either calculated with the pseudoinverse or with the direct inversion of the matrix.
// * Otherwise, one can skip the stochastic reconfiguration and use the standard gradient descent.
// * @note The method can be run without calculating the geometric tensor S explicitly.
// * @param step current step of updating - for the regularization purpose
// */
// template<uint _spinModes, typename _Ht, typename _T, class _stateType>
// void NQS<_spinModes, _Ht, _T, _stateType>::gradSR(uint step)
// {
// 	#ifndef NQS_USESR_NOMAT
// 	{
// 		// regularize the covariance matrix before inverting it (if needed and set)
// 		{
// 		#ifdef NQS_SREG
// 		this->covMatrixReg();
// 		#endif
// 		}
		
// 		#ifdef NQS_PINV
// 		{
// 			// calculate the pseudoinverse
// 			if (NQS_PINV > 0)
// 				this->F_ = this->lr_ * (arma::pinv(this->S_, NQS_PINV) * this->F_);
// 			else
// 				this->F_ = this->lr_ * (arma::pinv(this->S_) * this->F_);
// 			return;
// 		}
// 		#else 
// 		// solve normally
// 		//this->F_ = this->lr_ * (arma::inv(this->S_) * this->F_);
// 		this->F_ = this->lr_ * arma::solve(this->S_, this->F_, arma::solve_opts::likely_sympd);
// 		#endif 
// 	}
// 	#else
// 	{
// 		std::function<void(const arma::Col<_T>&, arma::Col<_T>&, size_t)> _Fun = std::bind(&NQS<_spinModes, _Ht, _T, _stateType>::getSRMatVec, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

// 		arma::Col<_T> _F = this->F_;

// 		// solve the system manually
// 		algebra::Solvers::solve(this->F_, _Fun, _F, algebra::Solvers::SolverType::MY_CONJ_GRAD, 1.0e-5);
// 		// algebra::Solvers::ConjugateGradient::solve_my_conj_grad<_T>(this->F_, _Fun, this->x_, this->r_, this->p_, this->Ap_, 1e-6);

// 		_F *= this->lr_;
// 		// exchange the vectors
// 		this->F_ = std::move(_F);
// 		// this->F_ = this->lr_ * this->x_;
// 	}
// 	#endif
// }
// // -----------------------------------------------------------------------------------------------------------------------------------
// #endif

// ///////////////////////////////////////////////////////////////////////

// /*
// * @brief	Calculate the final step for the weights to be updated.
// * First of all, it calculates the generalized forces
// * 
// * @warning Uses the forces vector (member F_) to store the forces that update the weights
// * 
// */
// template<uint _spinModes, typename _Ht, typename _T, class _stateType>
// void NQS<_spinModes, _Ht, _T, _stateType>::gradFinal(const NQSB& _energies)
// {
// 	// calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}> [+ sum_i ^{n-1} \beta _i <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> <Psi _W/Psi_W(i)>]
// 	this->derivativesC_ = arma::conj(this->derivatives_);				// precalculate the conjugate of the derivatives
// 	this->F_			= arma::cov(this->derivativesC_, _energies, 1);	// calculate the covariance vector for the gradient !TODO modify this for excited states! 
// #ifdef NQS_USESR
// 	{
// 	#ifndef NQS_USESR_NOMAT
// 		// form the covariance matrix explicitly
// 		this->S_		= arma::cov(this->derivativesC_, this->derivatives_, 1);
// 	#else 
// 		{
// 		arma::Mat<_T> S1 	= arma::cov(this->derivativesC_, this->derivatives_, 0);
// 		std::cout << "Covariance matrix: " << S1.n_rows << ", " << S1.n_cols << std::endl;


// 		// calculate the matrix without forming it explicitly and use the mean of the derivatives
// 		this->derivativesM_ = arma::mean(this->derivatives_, 0).as_col();
// 		std::cout << "Mean of the derivatives: " << this->derivativesM_.n_elem << std::endl;

// 		arma::Mat<_T> S2	= S1;
// 		S2.zeros();

// 		for (size_t i = 0; i < this->derivativesC_.n_cols; ++i)
// 			for (size_t j = 0; j < this->derivativesC_.n_cols; ++j)
// 				S2(i, j) = this->getSRMatrixElement(i, j);
// 		// check the difference
// 		arma::Mat<double> diff = arma::abs(S1 - S2);
// 		diff.print("Difference: ");
// 		}
// 	#endif
// 		// update model by recalculating the gradient (applying the stochastic reconfiguration)
// 		this->gradSR(0);
// 	}
// #else
// 	{
// 		// standard updater with the gradient only!
// 		this->F_ *= this->lr_;
// 	}
// #endif
// }

// // ##########################################################################################################################################

// // TEMPLATES

// // ##########################################################################################################################################

// // two spin modes
// template class NQS<2, double, double, double>;
// template class NQS<2, std::complex<double>, double, double>;
// template class NQS<2, double, std::complex<double>, double>;
// template class NQS<2, std::complex<double>, std::complex<double>, double>;

// // // four spin modes
// template class NQS<4, double, double, double>;
// template class NQS<4, std::complex<double>, double, double>;
// template class NQS<4, double, std::complex<double>, double>;
// template class NQS<4, std::complex<double>, std::complex<double>, double>;

// // local energy kernel
// template void NQS<2, double, double, double>::locEnKernel(uint, uint, uint);
// template void NQS<2, std::complex<double>, double, double>::locEnKernel(uint, uint, uint);
// template void NQS<2, double, std::complex<double>, double>::locEnKernel(uint, uint, uint);
// template void NQS<2, std::complex<double>, std::complex<double>, double>::locEnKernel(uint, uint, uint);

// template void NQS<4, double, double, double>::locEnKernel(uint, uint, uint);
// template void NQS<4, std::complex<double>, double, double>::locEnKernel(uint, uint, uint);
// template void NQS<4, double, std::complex<double>, double>::locEnKernel(uint, uint, uint);
// template void NQS<4, std::complex<double>, std::complex<double>, double>::locEnKernel(uint, uint, uint);

// // local energy kernel excited
// template void NQS<2, double, double, double>::locEnKernelExcited(uint, uint, uint);
// template void NQS<2, std::complex<double>, double, double>::locEnKernelExcited(uint, uint, uint);
// template void NQS<2, double, std::complex<double>, double>::locEnKernelExcited(uint, uint, uint);
// template void NQS<2, std::complex<double>, std::complex<double>, double>::locEnKernelExcited(uint, uint, uint);

// template void NQS<4, double, double, double>::locEnKernelExcited(uint, uint, uint);
// template void NQS<4, std::complex<double>, double, double>::locEnKernelExcited(uint, uint, uint);
// template void NQS<4, double, std::complex<double>, double>::locEnKernelExcited(uint, uint, uint);
// template void NQS<4, std::complex<double>, std::complex<double>, double>::locEnKernelExcited(uint, uint, uint);

// // block sampling
// template void NQS<2, double, double, double>::blockSample(uint, NQS_STATE_T, bool);
// template void NQS<2, std::complex<double>, double, double>::blockSample(uint, NQS_STATE_T, bool);
// template void NQS<2, double, std::complex<double>, double>::blockSample(uint, NQS_STATE_T, bool);
// template void NQS<2, std::complex<double>, std::complex<double>, double>::blockSample(uint, NQS_STATE_T, bool);

// template void NQS<4, double, double, double>::blockSample(uint, NQS_STATE_T, bool);
// template void NQS<4, std::complex<double>, double, double>::blockSample(uint, NQS_STATE_T, bool);
// template void NQS<4, double, std::complex<double>, double>::blockSample(uint, NQS_STATE_T, bool);
// template void NQS<4, std::complex<double>, std::complex<double>, double>::blockSample(uint, NQS_STATE_T, bool);

// // gradient with stochastic reconfiguration
// template void NQS<2, double, double, double>::gradSR(uint);
// template void NQS<2, std::complex<double>, double, double>::gradSR(uint);
// template void NQS<2, double, std::complex<double>, double>::gradSR(uint);
// template void NQS<2, std::complex<double>, std::complex<double>, double>::gradSR(uint);

// template void NQS<4, double, double, double>::gradSR(uint);
// template void NQS<4, std::complex<double>, double, double>::gradSR(uint);
// template void NQS<4, double, std::complex<double>, double>::gradSR(uint);
// template void NQS<4, std::complex<double>, std::complex<double>, double>::gradSR(uint);

// // gradient final
// template void NQS<2, double, double, double>::gradFinal(const NQSB&);
// template void NQS<2, std::complex<double>, double, double>::gradFinal(const NQSB&);
// template void NQS<2, double, std::complex<double>, double>::gradFinal(const NQSB&);
// template void NQS<2, std::complex<double>, std::complex<double>, double>::gradFinal(const NQSB&);

// template void NQS<4, double, double, double>::gradFinal(const NQSB&);
// template void NQS<4, std::complex<double>, double, double>::gradFinal(const NQSB&);
// template void NQS<4, double, std::complex<double>, double>::gradFinal(const NQSB&);
// template void NQS<4, std::complex<double>, std::complex<double>, double>::gradFinal(const NQSB&);

// // ##########################################################################################################################################