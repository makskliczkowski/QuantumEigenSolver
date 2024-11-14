#include "rbm_pp_setup.hpp"

#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE

/*
* @brief Updates the Pfaffian matrix for a given row corresponding to the change in the X matrix.
* This needs the inverse of the X matrix to be updated accordingly as well.
* @param _row row which has been changed
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::updatePfaffian(uint _row)
{
	this->pfaffian_ = algebra::Pfaffian::pfaffian_upd_row_n_col(this->pfaffian_, this->Xinv_.row(_row), this->X_.row(_row));
}

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::updatePfaffian(uint _row, _T& _pfaffian)
{
	_pfaffian = algebra::Pfaffian::pfaffian_upd_row_n_col(_pfaffian, this->Xinv_.row(_row), this->X_.row(_row));
}

template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::updatePfaffian(uint _row, _T& _pfaffian, const arma::Mat<_T>& _X)
{
	_pfaffian = algebra::Pfaffian::pfaffian_upd_row_n_col(_pfaffian, this->Xinv_.row(_row), _X.row(_row));
}

/*
* @brief Updates the Pfaffian matrix for a given row corresponding to the change in the Xnew matrix.
* This needs the inverse of the Xmatrix to be updated accordingly as well.
* @param _row row which has been changed
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::updatePfaffian_C(uint _row)
{
	this->pfaffianNew_ = algebra::Pfaffian::pfaffian_upd_row_n_col(this->pfaffian_, this->Xinv_.row(_row), this->Xnew_.row(_row));
}


// ##########################################################################################################################################

/*
* @brief Updates the X matrix inverse for a given row corresponding to the change in the X matrix.
* @param _row row which has been changed
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::updateXInv(uint _row)
{
	this->Xinv_ = algebra::scherman_morrison_skew(this->Xinv_, _row, this->X_.row(_row));
}

/*
* @brief Updates the X matrix inverse for a given row corresponding to the change in the Xnew matrix.
* @param _row row which has been changed
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::updateXInv_C(uint _row)
{
	this->Xinv_ = algebra::scherman_morrison_skew(this->Xinv_, _row, this->Xnew_.row(_row));
}

#endif