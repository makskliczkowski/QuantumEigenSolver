/*
* @brief Definitions for the lower states in the NQS class
* @file nqs_definitions_lower.h
* Uses those definitions for the training and the overlaps when one is looking for the excited states.
*/

#pragma once

/////////////////////////////////////////////////////////////
#include "nqs_definitions_base.h"
#include <memory>
#ifndef NQS_OPERATOR_H
#	include "nqs_operator.h"
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
    NQSLS_p f_lower					=		{};						// lower states (for the training and looking for the overlaps)
    std::vector<double> f_lower_b	=		{};						// pentalties for the lower states - for the excited states calculation

    // ##########################################################################################################################################

    NQS_lower_t()
        : containerP_(Operators::Containers::OperatorContainer<_T>(1)) {};
    NQS_lower_t(size_t _Ns)
        : containerP_(Operators::Containers::OperatorContainer<_T>(_Ns))
    {
        this->containerP_.template decideSize<_T>();

        // set constant functions
        this->const_fun_    = [](u64 _s)                    -> _T { return _T(1.0); };
        this->const_funV_   = [](Operators::_OP_V_T_CR _v)  -> _T { return _T(1.0); };
    }
    
    // ##########################################################################################################################################
    
    // for the energy estimation
    Operators::Containers::OperatorContainer<_T> containerP_;       // container for the projectors  
    Operators::OperatorNQS<_T> enP_;                                // operator for the energy estimation - it is a combination of the projectors to the basis state currently used in the excited state estimation

    // ##########################################################################################################################################

    /*
    * @brief Sets the projector for the lower states - for the energy estimation of the excited state. 
    * The measurement class is reset and new projector operator is being set. 
    * @param _Ns number of states in the basis vector
    * @param _current_exc_state current excited state vector
    * @param _exc_state_pratio function for calculating the probability ratio for the excited state
    */
    void setProjector(size_t _Ns, Operators::_OP_V_T_CR _current_exc_state, 
                                    std::function<_T(Operators::_OP_V_T_CR _v)> _exc_state_pratio_v)
    {
        // create the projection operator
        this->enP_ = Operators::GeneralOperators::projectorSumComb(_Ns, 
                                                                    _current_exc_state,     // project to current state
                                                                    this->const_fun_,       // calculate the probability ratio (for the excited state)
                                                                    _exc_state_pratio_v);   // calculate the probability ratio (for the excited state) using the vector representation
    }

    // ##########################################################################################################################################

    _T collectLowerEnergy(uint i)
    {
        this->containerP_.reset();
        this->f_lower[i]->collect(this->train_lower_, this->enP_, this->containerP_);
        // get the mean value
        return this->f_lower_b[i] * this->containerP_.template mean<_T>()(0, 0);
    }

};

// ##########################################################################################################################################

// ######### NQS TYPES #############
BEGIN_ENUM(NQSTYPES)			// #
{								// #
	DECL_ENUM_ELEMENT(RBM),		// #
	DECL_ENUM_ELEMENT(RBMPP)	// #
}								// #
END_ENUM(NQSTYPES)				// #
// #################################