/*
* @brief Definitions for the lower states in the NQS class
* @file nqs_definitions_lower.h
* Uses those definitions for the training and the overlaps when one is looking for the excited states.
*/

#pragma once

/////////////////////////////////////////////////////////////
#include "armadillo"
#include "nqs_definitions_base.h"
#include <cmath>
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

#define NQS_INST_L_CMB(_Ht, _T, FUN, FUNRET, ARGS) 							        \
					template FUNRET  NQS_lower_t<2u, _Ht, _T, double>::FUN ARGS; 	\
					template FUNRET  NQS_lower_t<3u, _Ht, _T, double>::FUN ARGS; 	\
					template FUNRET  NQS_lower_t<4u, _Ht, _T, double>::FUN ARGS; 	

// ##########################################################################################################################################

/**
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
template <uint _spinModes, typename _Ht, typename _T, class _stateType>	// clone

struct NQS_lower_t
{
    using NQSLS_p					= 		std::vector<std::shared_ptr<NQS<_spinModes, _Ht, _T, _stateType>>>;
    MonteCarlo::MCS_train_t train_lower_;                           // training information for the lower states                       
    NQS<_spinModes, _Ht, _T, _stateType>* nqs_exc_;			    	// pointer to the excited state NQS instance
    size_t Ns_                      =       0;                      // number of the states in the basis
    bool isSet_						= 		false;
    uint f_lower_size_				=		0;						// number of the lower states
    std::vector<double> f_lower_b_	=		{};						// pentalties for the lower states - for the excited states calculation
    NQSLS_p f_lower					=		{};						// lower states (for the training and looking for the overlaps)

    // ##########################################################################################################################################

    NQS_lower_t() : containerP_({}) {};
    NQS_lower_t(size_t _Ns, NQSLS_p _f_lower, std::vector<double> _f_lower_b, NQS<_spinModes, _Ht, _T, _stateType>* _nqs_exc);
    
    // ##########################################################################################################################################
    
    // for the energy estimation
    v_2d<_T> containerP_;                                           // containers for the projectors  
    Operators::OperatorNQS<_T> enP_;                                // operator for the energy estimation - it is a combination of the projectors to the basis state currently used in the excited state estimation

    // for the gradient ratio estimation
    std::vector<arma::Col<_T>> ratios_lower_;                       // calculate this->ansatz(s) / \psi _wj(s) at each MC step (average in the lower states)
    std::vector<arma::Col<_T>> ratios_excited_;                     // calculate \psi _wj(s) / this->ansatz(s) at each MC step

    // ##########################################################################################################################################
    
    std::function<_T(Operators::_OP_V_T_CR)> exc_ratio_;            // set later
    void setDerivContSize(size_t _mcsexcited);
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
    : nqs_exc_(_nqs_exc), Ns_(_Ns), isSet_(!_f_lower.empty()), f_lower_size_(_f_lower.size()), f_lower_b_(_f_lower_b), f_lower(_f_lower)
{
    LOGINFO("NQS_lower_t: Initializing with Ns = " + std::to_string(_Ns), LOG_TYPES::DEBUG, 1);
    LOGINFO("NQS_lower_t: Penalties for lower states set.", LOG_TYPES::DEBUG, 1);
    LOGINFO("NQS_lower_t: Number of lower states = " + std::to_string(f_lower_size_), LOG_TYPES::INFO, 3);

    if (!_nqs_exc)
    {
        LOGINFO("NQS_lower_t: No excited state NQS instance has been set...", LOG_TYPES::WARNING, 3);
        return;
    }

    this->containerP_ = v_2d<_T>(this->f_lower_size_, v_1d<_T>(this->train_lower_.nblck_ * this->train_lower_.MC_sam_, _T(0.0)));
    LOGINFO("NQS_lower_t: Container for projectors initialized.", LOG_TYPES::DEBUG, 1);
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

// ##########################################################################################################################################