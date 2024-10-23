/*
* @brief Definitions for the lower states in the NQS class
* @file nqs_definitions_lower.h
* Uses those definitions for the training and the overlaps when one is looking for the excited states.
*/

#pragma once

/////////////////////////////////////////////////////////////
#include "armadillo"
#include "nqs_definitions_base.h"
#include <functional>
#include <memory>
#include <vector>
#ifndef NQS_OPERATOR_H
#	include "../nqs_operator.h"
#endif
/////////////////////////////////////////////////////////////

// forward declarations
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
class NQS;

#define NQS_LOWER_RATIO_LOGDIFF 

// ##########################################################################################################################################

/*
* @brief Structure for storing the lower states information - for the training and the overlaps 
* when one is looking for the excited states - the lower states are needed for the energy estimation and gradients
* @see Adv. Physics Res. 2024, 3, 2300078 for more information

* 1) Energy estimation for the excited states - change Hamiltonian H' = H + \sum_i \beta_i H_i, where 
* H_i are the projectors to the lower states H_i = |f_i><f_i| / (<f_i|f_i>) and \beta_i are the penalties for the excited states
* This means that for a given state <s| the estimation of the additional part needs to be done. 
* This is achieved by performing the estimates of the projectors |s><s| multiplied by the probability 
* ratio of the given state and the new proposed state from the lower states.

* 2) Gradients for the excited states - the gradients are calculated 
* as <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}> 
[+ sum_i ^{n-1} \beta _i <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> <Psi _W/Psi_W(i)>]
* This structure intends that for each given state <s| that is used for the estimation of 
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
struct NQS_lower_t
{
    NQS_train_t train_lower_;                                       // training information for the lower states                       
    NQS<_spinModes, _Ht, _T, _stateType>* nqs_exc_;			    	// pointer to the excited state NQS instance
    using NQSLS_p					= 		std::vector<std::shared_ptr<NQS<_spinModes, _Ht, _T, _stateType>>>;

    // for the excited states 
    size_t Ns_                      =       0;                      // number of the states in the basis
    bool isSet_						= 		false;
    uint f_lower_size_				=		0;						// number of the lower states
    NQSLS_p f_lower					=		{};						// lower states (for the training and looking for the overlaps)
    std::vector<double> f_lower_b_	=		{};						// pentalties for the lower states - for the excited states calculation

    // ##########################################################################################################################################

    NQS_lower_t() : containerP_(Operators::Containers::OperatorContainer<_T>(1)) {};
    NQS_lower_t(size_t _Ns, NQSLS_p _f_lower, std::vector<double> _f_lower_b, NQS<_spinModes, _Ht, _T, _stateType>* _nqs_exc);
    
    // ##########################################################################################################################################
    
    // for the energy estimation
    Operators::Containers::OperatorContainer<_T> containerP_;       // container for the projectors  
    Operators::OperatorNQS<_T> enP_;                                // operator for the energy estimation - it is a combination of the projectors to the basis state currently used in the excited state estimation

    // for the gradient ratio estimation
    std::vector<arma::Col<_T>> ratios_lower_;                       // calculate this->ansatz(s) / \psi _wj(s) at each MC step (average in the lower states)
    std::vector<arma::Col<_T>> ratios_excited_;                     // calculate \psi _wj(s) / this->ansatz(s) at each MC step

    // ##########################################################################################################################################
    
    std::function<_T(Operators::_OP_V_T_CR)> exc_ratio_;            // set later
    void setDerivContSize(size_t _mcslower, size_t _mcsexcited);
    void setProjector(Operators::_OP_V_T_CR _current_exc_state);
    
    // ##########################################################################################################################################

    _T collectLowerEnergy(uint i);
    _T ansatz(Operators::_OP_V_T_CR _v, int _low = 0) const         { return this->f_lower[_low]->ansatz(_v); };
    _T ansatzlog(Operators::_OP_V_T_CR _v, int _low = 0) const      { return this->f_lower[_low]->ansatzlog(_v); };

    // ##########################################################################################################################################

    std::function<_T(Operators::_OP_V_T_CR)> exc_ansatz_;
    void collectLowerRatios(uint i);
    _T collectExcitedRatios(uint i, Operators::_OP_V_T_CR _current_exc_state);
};

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline NQS_lower_t<_spinModes, _Ht, _T, _stateType>::NQS_lower_t(size_t _Ns, 
                        NQSLS_p _f_lower, std::vector<double> _f_lower_b, NQS<_spinModes, _Ht, _T, _stateType>* _nqs_exc)
    : nqs_exc_(_nqs_exc), 
    Ns_(_Ns),
    isSet_(!_f_lower.empty()),
    f_lower_size_(_f_lower.size()),
    f_lower(_f_lower),
    f_lower_b_(_f_lower_b)    
{
    // keep empty
    if (!_nqs_exc)
    {
        LOGINFO("NQS_lower_t: No excited state NQS instance has been set...", LOG_TYPES::WARNING, 3);
        return;
    }

    this->containerP_ = Operators::Containers::OperatorContainer<_T>(_Ns);
    this->containerP_.decideSize();
}

// ##########################################################################################################################################

/*
* @brief Sets the size of the containers for the derivatives of the lower states - for the gradient estimation of the excited state.
* @param _mcslower size of the Monte Carlo steps for the lower states - ratios sampled in the lower states
* @param _mcsexcited size of the Monte Carlo steps for the excited state - ratios sampled in the excited state
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS_lower_t<_spinModes, _Ht, _T, _stateType>::setDerivContSize(size_t _mcslower, size_t _mcsexcited)
{
    if (this->f_lower_size_ == 0)
        return;
    
    this->ratios_lower_             = std::vector<arma::Col<_T>>(this->f_lower_size_, arma::Col<_T>(_mcslower, arma::fill::zeros));
    this->ratios_excited_           = std::vector<arma::Col<_T>>(this->f_lower_size_, arma::Col<_T>(_mcsexcited, arma::fill::zeros));
}

// ##########################################################################################################################################

/*
* @brief Sets the projector for the lower states - for the energy estimation of the excited state. 
* The measurement class is reset and new projector operator is being set. 
* @param _Ns number of states in the basis vector
* @param _current_exc_state current excited state vector
* @param _exc_state_pratio function for calculating the probability ratio for the excited state
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS_lower_t<_spinModes, _Ht, _T, _stateType>::setProjector(Operators::_OP_V_T_CR _current_exc_state)
{
    // create the projection operator
    this->enP_ = Operators::GeneralOperators::projectorSumComb(Ns_, 
                    _current_exc_state,     // project to current state <s|psi_w>
                    this->exc_ratio_);      // calculate the probability ratio (for the excited state) using the vector representation \psi _w(s') / \psi _w(s)
}

// ##########################################################################################################################################

/*
* @brief Collect the addition to the energy coming from the overlap with the lower states - for the excited state energy estimation
* @param i index of the lower state
* @returns the mean value of the energy estimation coming from the lower states overlap
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T NQS_lower_t<_spinModes, _Ht, _T, _stateType>::collectLowerEnergy(uint i)
{
    if (this->f_lower_size_ == 0)
        return _T(0.0);

    this->containerP_.reset();
    this->f_lower[i]->collect(this->train_lower_, this->enP_, this->containerP_);
    // get the mean value
    return this->f_lower_b_[i] * this->containerP_.template mean<_T>()(0, 0);
}

// ##########################################################################################################################################

/*
* @brief Collect the ratios for the lower states - for the gradient estimation of the excited state. This ratio is given by the
* \psi _w(s') / \psi _w_j(s) at each MC step (average in the lower states)
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS_lower_t<_spinModes, _Ht, _T, _stateType>::collectLowerRatios(uint i)
{
    if (this->f_lower_size_ == 0)
        return;
    this->f_lower[i]->collect_ratio(this->train_lower_, this->exc_ansatz_, this->ratios_lower_[i]);
}

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T NQS_lower_t<_spinModes, _Ht, _T, _stateType>::collectExcitedRatios(uint i, Operators::_OP_V_T_CR _current_exc_state)
{
    if (this->f_lower_size_ == 0)
        return _T(0.0);
#ifdef NQS_LOWER_RATIO_LOGDIFF
    return std::exp(this->ansatzlog(_current_exc_state, i) - this->exc_ansatz_(_current_exc_state));
#else
    return this->ansatz(_current_exc_state, i) / this->exc_ansatz_(_current_exc_state);
#endif
}

// ##########################################################################################################################################

// ######### NQS TYPES #############
BEGIN_ENUM(NQSTYPES)			// #
{								// #
	DECL_ENUM_ELEMENT(RBM),		// #
	DECL_ENUM_ELEMENT(RBMPP)	// #
}								// #
END_ENUM(NQSTYPES)				// #
// #################################