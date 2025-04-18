#include "./nqs_pp_prob.tpp"

namespace NQS_NS
{
// ##########################################################################################################################

/**
* @brief Computes the updated Pfaffian for a given row.
*
* This function calculates the updated Pfaffian value for a specified row
* using the Cayley's formula if the NQS_USE_ARMA macro is defined.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for the Pfaffian.
* @tparam _stateType The type representing the state.
* @tparam _CorrState The type representing the correlated state.
* @param _row The row index for which the updated Pfaffian is to be computed.
* @return The updated Pfaffian value of type _T.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::getUpdatedPfaffian(uint _row) const
{
#ifdef NQS_REF_PP_USE_PFAFFIAN_UPDATE
    #ifdef NQS_USE_ARMA
        return ::algebra::Pfaffian::cayleys(this->pp_weights_.pfaffian_, this->pp_weights_.X_inv.row(_row), this->pp_weights_.X_upd_[0]);
    #else
        !TODO : implement more use cases for the update
    #endif
    #else 

#endif
}

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::getUpdatedPfaffian(uint _row, _T _pfaffian) const
{
#ifdef NQS_REF_PP_USE_PFAFFIAN_UPDATE
    #ifdef NQS_USE_ARMA
        return ::algebra::Pfaffian::cayleys(_pfaffian, this->pp_weights_.X_inv.row(_row), this->pp_weights_.X_upd_[_row]);
    #else

    #endif
    #else
#endif
}

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::getUpdatedPfaffian(uint _row, _T _pfaffian, const NQSB& _Xrow) const
{
#ifdef NQS_REF_PP_USE_PFAFFIAN_UPDATE
    #ifdef NQS_USE_ARMA
        return ::algebra::Pfaffian::cayleys(_pfaffian, this->pp_weights_.X_inv.row(_row), _Xrow);
    #else
    #endif
    #else
#endif
}

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::getUpdatedPfaffian(uint _row, _T _pfaffian, const NQSW& _X) const
{
#ifdef NQS_REF_PP_USE_PFAFFIAN_UPDATE
    #ifdef NQS_USE_ARMA
        return ::algebra::Pfaffian::cayleys(_pfaffian, this->pp_weights_.X_inv.row(_row), _X.row(_row));
    #else
    #endif
#else
#endif
}
// ##########################################################################################################################################

/**
* @brief Updates the X matrix inverse for a given row corresponding to the change in the X matrix.
* @param _row row which has been changed
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::updateXInv(uint _row)
{
    this->pp_weights_.X_inv = ::algebra::scherman_morrison_skew(this->pp_weights_.X_inv, _row, this->pp_weights_.X_.row(_row));
}
// ##########################################################################################################################################

}; // namespace NQS_NS

// ##########################################################################################################################################