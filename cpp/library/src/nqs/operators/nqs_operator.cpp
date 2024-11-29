#include "../../../include/NQS/nqs_operator.h"

// ##########################################################################################################################################

namespace Operators 
{
    template class Operators::OperatorNQS<double>;
    template class Operators::OperatorNQS<std::complex<double>>;
    template class Operators::OperatorNQS<double, uint>;
    template class Operators::OperatorNQS<double, int>;
    template class Operators::OperatorNQS<std::complex<double>, uint>;
    template class Operators::OperatorNQS<std::complex<double>, int>;
    template class Operators::OperatorNQS<double, uint, uint>;
    template class Operators::OperatorNQS<double, int, int>;
    template class Operators::OperatorNQS<std::complex<double>, uint, uint>;
    template class Operators::OperatorNQS<std::complex<double>, int, int>;
    // no more exotics

    // ##########################################################################################################################################

    /**
    * @brief Apply the operators with a value change (with pRatio). The function is used to calculate the 
    * probability ratio for the given state and the operator.
    * @param s base state to apply the operators to
    * @param _fun pRatio function from the NQS
    * @param ...a additional parameters to the operators
    * @returns value of the operator acting on the state with the probability ratio applied
    */
	template<typename _T, typename ..._Ts>
	_T OperatorNQS<_T, _Ts...>::operator()(u64 s, NQSFunCol _fun, _Ts ...a)
	{
		_T _valTotal = 0.0;
		for (const auto& [s2, _val] : this->operator()(s, a...)) {            // go through operator acting on the state
			Binary::int2base(s2, this->state_, _SPIN_RBM);              // set the state
			_valTotal += _val * algebra::cast<_T>(_fun(this->state_));  // calculate the probability ratio
		}
		return algebra::cast<_T>(_valTotal);
	}

    // template instantiation
    template double OperatorNQS<double>::operator()(u64, NQSFunCol);					
    template double OperatorNQS<double, uint>::operator()(u64, NQSFunCol, uint);		
    template double OperatorNQS<double, int>::operator()(u64, NQSFunCol, int);		
    template double OperatorNQS<double, uint, uint>::operator()(u64, NQSFunCol, uint, uint);		
    template double OperatorNQS<double, int, int>::operator()(u64, NQSFunCol, int, int);		

    template cpx OperatorNQS<cpx>::operator()(u64, NQSFunCol);					
    template cpx OperatorNQS<cpx, uint>::operator()(u64, NQSFunCol, uint);		
    template cpx OperatorNQS<cpx, int>::operator()(u64, NQSFunCol, int);		
    template cpx OperatorNQS<cpx, uint, uint>::operator()(u64, NQSFunCol, uint, uint);		
    template cpx OperatorNQS<cpx, int, int>::operator()(u64, NQSFunCol, int, int);		

	////////////////////////////////////////////////////////////////////////////

	/**
	* @brief Apply the operators with a value change (!with pRatio). The function is used to calculate the
	* probability ratio for the given state and the operator.
	* @param s base state to apply the operators to - vector version
	* @param _fun pRatio function from the NQS
	* @param ...a additional parameters to the operators
	* @returns value of the operator acting on the state with the probability ratio applied
	*/
	template<typename _T, typename ..._Ts>
	_T Operators::OperatorNQS<_T, _Ts...>::operator()(_OP_V_T_CR s, NQSFunCol _fun, _Ts ...a) const
	{
		_T _valTotal = 0.0;
		for (auto& [s2, _val] : this->operator()(s, a...))
		{
			_T _functionVal = CAST<_T>(_fun(s2));                   // calculate the probability ratio
			_valTotal 		= _valTotal + _functionVal * _val;      // calculate the value
		}
		return _valTotal;
	}

    // template instantiation
    template double OperatorNQS<double>::operator()(_OP_V_T_CR, NQSFunCol) const;					
    template double OperatorNQS<double, uint>::operator()(_OP_V_T_CR, NQSFunCol, uint) const;		
    template double OperatorNQS<double, int>::operator()(_OP_V_T_CR, NQSFunCol, int) const;		
    template double OperatorNQS<double, uint, uint>::operator()(_OP_V_T_CR, NQSFunCol, uint, uint) const;		
    template double OperatorNQS<double, int, int>::operator()(_OP_V_T_CR, NQSFunCol, int, int) const;		

    template cpx OperatorNQS<cpx>::operator()(_OP_V_T_CR, NQSFunCol) const;					
    template cpx OperatorNQS<cpx, uint>::operator()(_OP_V_T_CR, NQSFunCol, uint) const;		
    template cpx OperatorNQS<cpx, int>::operator()(_OP_V_T_CR, NQSFunCol, int) const;		
    template cpx OperatorNQS<cpx, uint, uint>::operator()(_OP_V_T_CR, NQSFunCol, uint, uint) const;		
    template cpx OperatorNQS<cpx, int, int>::operator()(_OP_V_T_CR, NQSFunCol, int, int) const;		

	// ##########################################################################################################################################

};

// ##########################################################################################################################################
