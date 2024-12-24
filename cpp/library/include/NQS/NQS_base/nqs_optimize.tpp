#include "./nqs_sr.tpp"

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setPreconditioner(int _pre) 									
{ 
	if (_pre != 0) 
	{ 
		this->precond_ = algebra::Solvers::Preconditioners::choose<_T>(_pre);
		LOGINFO("Using preconditioner: " + algebra::Solvers::Preconditioners::name(_pre), LOG_TYPES::CHOICE, 3);
	}
};

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setSolver(int _sol, double _tol, int _maxiter, double _reg)						
{ 
	this->info_p_.setSolver(_sol, _maxiter, _tol); 
	this->solver_ = algebra::Solvers::General::choose<_T, true>(_sol, this->info_p_.fullSize_, _tol, _maxiter, _reg);
	LOGINFO("Using solver: " + algebra::Solvers::General::name(_sol) + " with tolerance: " + VEQPS(_tol, 3) + " and iterations: " + STR(_maxiter), LOG_TYPES::CHOICE, 3); 
};

// ##########################################################################################################################################