#ifndef USER_INTERFACE_PARAMS_H
#define USER_INTERFACE_PARAMS_H

// ##########################################################################################################################################

#include "./user_interface_nqsp.hpp"

// ###################### LIMITS ############################

#define UI_ENERGYMEAN_SUBVEC(MCSTEPS, TROUT)					int(TROUT * MCSTEPS), MCSTEPS - int(TROUT * MCSTEPS) - 1
// --- NQS
constexpr int UI_LIMITS_NQS_ED									= ULLPOW(13);
constexpr int UI_LIMITS_NQS_FULLED								= ULLPOW(12);
constexpr int UI_LIMITS_LANCZOS 								= ULLPOW(24);
constexpr int UI_LIMITS_NQS_LANCZOS_STATENUM					= 100;

// --- QUADRATIC
constexpr int UI_LIMITS_QUADRATIC_COMBINATIONS					= 20;
constexpr int UI_LIMITS_QUADRATIC_STATEFULL						= 32;

// ##########################################################################################################################################

// ############################################################# U I N T E R F ##############################################################

// ##########################################################################################################################################

// for running the complex Hamiltonian or the real one
#define RUN_CPX_REAL(TAKE_COMPLEX, _F, _MREAL, _MCPX) if (TAKE_COMPLEX) _F<std::complex<double>>(_MCPX); else _F<double>(_MREAL);
// default containers
#define UI_DEF_VMAT(Type, _sizex, _sizey, _sizez) VMAT<Type>(_sizex, _sizey, _sizez, arma::fill::ones, -1e5)
#define UI_DEF_VMAT_COND(Type, _sizex, _sizey, _sizez, _cond) _cond ? VMAT<Type>(_sizex, _sizey, _sizez, arma::fill::ones, -1e5) : VMAT<Type>(0, 0, 0, arma::fill::ones, -1e5)
#define UI_DEF_MAT_D(sizex, sizey) -1e5 * arma::ones<arma::Mat<double>>(sizex, sizey)
#define UI_DEF_MAT_D_COND(sizex, sizey, cond) cond ? arma::Mat<double>(sizex, sizey, arma::fill::ones) : arma::Mat<double>()
#define UI_DEF_MAT_D_CONDT(sizex, sizey, cond, T) cond ? arma::Mat<T>(sizex, sizey, arma::fill::ones) : arma::Mat<T>()
#define UI_DEF_COL_D(size) -1e5 * arma::ones<arma::Col<double>>(size)

#endif // USER_INTERFACE_PARAMS_H