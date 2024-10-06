/*
* @brief Definitions for the lower states in the NQS class
* @file nqs_definitions_lower.h
* Uses those definitions for the training and the overlaps when one is looking for the excited states.
*/

#pragma once

/////////////////////////////////////////////////////////////
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
    using NQSLS_p					= 		std::vector<std::shared_ptr<NQS<_spinModes, _Ht, _T, _stateType>>>;

    // for the excited states 
    bool isSet_						= 		false;
    NQSLS_p f_lower					=		{};						// lower states (for the training and looking for the overlaps)
    std::vector<double> f_lower_b	=		{};						// pentalties for the excited states
	NQSAv::MeasurementNQS<_T> measureProjectors_;					// measurement projectors for the lower states energy estimation (see )

    // ##########################################################################################################################################

    /*
    * @brief Resets the projectors for the lower states estimation. 
    */
    void resetProjector()
    {
        for (auto& _projector : this->measureProjectors_)
        {
            _projector->reset();
        }
    }

    // ##########################################################################################################################################

    /*
    * 
    */
    void setProjector(size_t _Ns, Operators::_OP_V_T_CR _current_exc_state, 
                std::function<_T(Operators::_OP_V_T_CR _v)> _exc_state_pratio)
    {
    
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