
#ifndef NQS_REF_STATE_H
#define NQS_REF_STATE_H

// include the base class
#ifndef NQS_H
#   include "../nqs_final.hpp"
#endif // !NQS_H

// ***************************************************************************************************************************

/**
* @class NQS_ref
* @brief A class that represents a reference implementation of a Neural Quantum State (NQS).
* 
* This class inherits from _CorrState and provides type definitions and static assertions
* to ensure proper inheritance and type usage for the Monte Carlo solver.
* 
* @tparam _spinModes The spin modes used in the NQS.
* @tparam _Ht The Hamiltonian type used in the NQS.
* @tparam _T The data type used for numerical computations.
* @tparam _stateType The type of the state representation.
* 
* @note This class uses static assertions to ensure that _CorrState is derived from NQS_S.
* @important The class will be designed as for the reference state |\psi > = \sum _x F(x) <x|phi>_{ref} |x>,
* where F(x) is the variational part taking into account the correlations between the particles and <x|phi>_{ref} is 
* overlap with a reference state. For simplicity, the reference state will be taken as the product of the single-particle states.
* @note The class will be designed as for the reference state |\psi > = \sum _x F(x) <x|phi>_{ref} |x>,
*/
template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double, 
            class _CorrState = NQS_S<_spinModes, _Ht, _T, _stateType>>
class NQS_ref : public _CorrState 
{
    // **********************************************************************************************************************
    static_assert(std::is_base_of<NQS_S<_spinModes, _Ht, _T, _stateType>, _CorrState>::value, "_CorrState must derive from NQS_S");
    // **********************************************************************************************************************
    NQS_PUBLIC_TYPES(_T, _stateType);
    MCS_PUBLIC_TYPES(_T, _stateType, arma::Col);                                // type definitions for the Monte Carlo solver
    using NQSLS_p = typename NQS_S<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
    // **********************************************************************************************************************
public:
    virtual auto clone()                    const -> MC_t_p override    =   0;
    virtual auto clone(MC_t_p _n)           -> void override;
    // **********************************************************************************************************************
    virtual ~NQS_ref()                      { DESTRUCTOR_CALL; };
    // **********************************************************************************************************************
};

#endif  
// ##########################################################################################################################################
