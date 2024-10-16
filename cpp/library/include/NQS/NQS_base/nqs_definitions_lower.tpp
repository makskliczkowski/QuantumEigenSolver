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
    // NQS<_spinModes, _Ht, _T, _stateType>* nqs_;			    	// pointer to the excited state NQS instance
    using NQSLS_p					= 		std::vector<std::shared_ptr<NQS<_spinModes, _Ht, _T, _stateType>>>;

    // constant functions
    std::function<_T(Operators::_OP_V_T_CR)> const_funV_;
    std::function<_T(u64)> const_fun_;

    // for the excited states 
    bool isSet_						= 		false;
    uint f_lower_size_				=		0;						// number of the lower states
    NQSLS_p f_lower					=		{};						// lower states (for the training and looking for the overlaps)
    std::vector<double> f_lower_b_	=		{};						// pentalties for the lower states - for the excited states calculation

    // ##########################################################################################################################################

    NQS_lower_t() : containerP_(Operators::Containers::OperatorContainer<_T>(1)) {};
    NQS_lower_t(size_t _Ns, NQSLS_p _f_lower, std::vector<double> _f_lower_b);
    
    // ##########################################################################################################################################
    
    // for the energy estimation
    Operators::Containers::OperatorContainer<_T> containerP_;       // container for the projectors  
    Operators::OperatorNQS<_T> enP_;                                // operator for the energy estimation - it is a combination of the projectors to the basis state currently used in the excited state estimation

    // for the gradient ratio estimation
    std::vector<arma::Col<_T>> ratios_lower_;                       // calculate this->ansatz(s) / \psi _wj(s) at each MC step (average in the lower states)
    std::vector<arma::Col<_T>> ratios_excited_;                     // calculate \psi _wj(s) / this->ansatz(s) at each MC step

    // ##########################################################################################################################################
    
    void setDerivContSize(size_t _mcslower, size_t _mcsexcited);
    void setProjector(size_t _Ns, Operators::_OP_V_T_CR _current_exc_state, std::function<_T(Operators::_OP_V_T_CR _v)> _exc_state_pratio_v);
    
    // ##########################################################################################################################################

    _T collectLowerEnergy(uint i);
    _T ansatz(Operators::_OP_V_T_CR _v, int _low = 0) const         { return this->f_lower[_low]->ansatz(_v); };
    _T ansatzlog(Operators::_OP_V_T_CR _v, int _low = 0) const      { return this->f_lower[_low]->ansatzlog(_v); };
   
    // ##########################################################################################################################################

    void collectRatiosLower(uint i, std::function<_T(Operators::_OP_V_T_CR)> _f);
};

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline NQS_lower_t<_spinModes, _Ht, _T, _stateType>::NQS_lower_t(size_t _Ns, NQSLS_p _f_lower, std::vector<double> _f_lower_b)
    : isSet_(!_f_lower.empty()),
    f_lower_size_(_f_lower.size()),
    f_lower(_f_lower),
    f_lower_b_(_f_lower_b)    
{
    this->containerP_ = Operators::Containers::OperatorContainer<_T>(_Ns);
    this->containerP_.decideSize();

    // set constant functions
    this->const_fun_    = [](u64 _s)                    -> _T { return _T(1.0); };
    this->const_funV_   = [](Operators::_OP_V_T_CR _v)  -> _T { return _T(1.0); };
}

// ##########################################################################################################################################

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
inline void NQS_lower_t<_spinModes, _Ht, _T, _stateType>::setProjector(size_t _Ns, Operators::_OP_V_T_CR _current_exc_state, 
                                std::function<_T(Operators::_OP_V_T_CR _v)> _exc_state_pratio_v)
{
    // create the projection operator
    this->enP_ = Operators::GeneralOperators::projectorSumComb(_Ns, 
                                                                _current_exc_state,     // project to current state
                                                                this->const_fun_,       // calculate the probability ratio (for the excited state)
                                                                _exc_state_pratio_v);   // calculate the probability ratio (for the excited state) using the vector representation
}

// ##########################################################################################################################################

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

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS_lower_t<_spinModes, _Ht, _T, _stateType>::collectRatiosLower(uint i, std::function<_T(Operators::_OP_V_T_CR)> _f)
{
    if (this->f_lower_size_ == 0)
        return;

    this->f_lower[i]->collect_ratio(this->train_lower_, _f, this->ratios_lower_[i]);
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