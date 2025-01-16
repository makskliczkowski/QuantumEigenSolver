#include "./nqs_sr.tpp"

// ##########################################################################################################################################

/**
* @brief Sets the preconditioner for the NQS optimization.
*
* This function sets the preconditioner based on the provided integer value.
* If the value is not zero, it chooses the appropriate preconditioner and logs the choice.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type used for computations.
* @tparam _stateType The type representing the state.
* @param _pre An integer representing the preconditioner choice. If _pre is zero, no preconditioner is set.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setPreconditioner(int _pre) 									
{ 
	if (this->precond_)
		delete this->precond_;

	if (_pre >= 0) 
	{ 
		// set the new preconditioner
		this->precond_ = algebra::Solvers::Preconditioners::choose<_T>(_pre);
		LOGINFO("Using preconditioner: " + algebra::Solvers::Preconditioners::name(_pre), LOG_TYPES::CHOICE, 3);
	}
};

/**
* @brief Sets the solver for the NQS optimization process.
* 
* This function configures the solver to be used in the optimization process of the 
* Neural Quantum State (NQS). It sets the solver parameters including the solver type, 
* tolerance, maximum iterations, and regularization parameter.
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type used in the solver.
* @tparam _stateType The state type.
* 
* @param _sol The solver type identifier.
* @param _tol The tolerance for the solver.
* @param _maxiter The maximum number of iterations for the solver.
* @param _reg The regularization parameter for the solver.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setSolver(int _sol, double _tol, int _maxiter, double _reg)						
{ 
	this->info_p_.setSolver(_sol, _maxiter, _tol); 
	// delete the old solver
	if (this->solver_)
		delete this->solver_;	
	// set the new solver
	this->solver_ = algebra::Solvers::General::choose<_T, true>(_sol, this->info_p_.fullSize_, _tol, _maxiter, _reg);
	LOGINFO("Using solver: " + algebra::Solvers::General::name(_sol) + " with tolerance: " + VEQPS(_tol, 3) + " and iterations: " + STR(_maxiter), LOG_TYPES::CHOICE, 3); 
};

// ##########################################################################################################################################