#ifndef NQS_REF_STATE_H
#define NQS_REF_STATE_H

// include the base class
#ifndef NQS_H
#   include "../nqs_final.hpp"
#endif // !NQS_H

// ***************************************************************************************************************************

namespace NQS_NS
{
    /**
    * @class NQS_ref
    * @brief A class representing a Neural Quantum State (NQS) reference.
    * 
    * This class inherits from _CorrState and provides functionality for cloning 
    * and managing NQS references. It includes type definitions for Monte Carlo 
    * solvers and Hamiltonians, and provides constructors for initializing NQS_ref 
    * objects with various parameters.
    * 
    * @tparam _spinModes The spin modes used in the NQS.
    * @tparam _Ht The Hamiltonian type.
    * @tparam _T The data type used in the NQS.
    * @tparam _stateType The state type used in the NQS.
    * 
    */
    template <uint _spinModes, typename _Ht, typename _T = _Ht, class _stateType = double, class _CorrState = NQS_S<_spinModes, _Ht, _T, _stateType>>
    requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
    class NQS_ref : public _CorrState
    {
        // **********************************************************************************************************************   
        NQS_PUBLIC_TYPES(_T, _stateType);
        MCS_PUBLIC_TYPES(_T, _stateType, arma::Col);                                // type definitions for the Monte Carlo solver
        NQS_HAMIL_TYPES(_Ht, _spinModes);                                           // type definitions for the Hamiltonian
        using NQSLS_p = typename NQS_S<_spinModes, _Ht, _T, _stateType>::NQSLS_p;
        using NQS_t   = NQS_S<_spinModes, _Ht, _T, _stateType>;
        using NQS_p_t = typename ::std::shared_ptr<NQS_t>;      
        // **********************************************************************************************************************
    public:
        virtual auto clone()                    const -> MC_t_p override            = 0;
        virtual auto clone(MC_t_p _n)           -> void override                    { _CorrState::clone(_n); };
        // **********************************************************************************************************************
        virtual ~NQS_ref()                      { DESTRUCTOR_CALL; };
        // **********************************************************************************************************************
        NQS_ref()                               = default;
        // **********************************************************************************************************************
        NQS_ref(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p)
            : _CorrState(_p)
        { 
            this->setInfo();
        };
        // **********************************************************************************************************************
        NQS_ref(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p, 
                const NQSLS_p& _lower, const v_1d<double>& _beta)
            : _CorrState(_p, _lower, _beta)
        { 
            this->setInfo(); 
        };
        // **********************************************************************************************************************
        NQS_ref(const NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>& _other)
            : _CorrState(_other)
        { 
            this->setInfo();
        };
        // **********************************************************************************************************************
        NQS_ref(NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>&& _other)
            : _CorrState(::std::move(_other))
        { 
            this->setInfo();
        };
        // **********************************************************************************************************************
    	public:													// -------------------- P R O B A B I L I T Y --------------------
		virtual auto logPRatio(uint fP, float fV)				            -> _T override { return _CorrState::logPRatio(fP, fV);              };    
		virtual auto logPRatio(uint fP1, uint fP2, float fV1, float fV2)	-> _T override { return _CorrState::logPRatio(fP1, fP2, fV1, fV2);  };
		virtual auto logPRatio(uint nFlips)						            -> _T override { return _CorrState::logPRatio(nFlips);              };
		virtual auto logPRatio(Config_cr_t _v1)                             -> _T override { return _CorrState::logPRatio(NQS_STATE, _v1);      };
		virtual auto logPRatio(Config_cr_t _v1, Config_cr_t _v2)            -> _T override { return _CorrState::logPRatio(_v1, _v2);            };
		virtual auto logPRatio(int_ini_t fP, dbl_ini_t fV) 		            -> _T override { return _CorrState::logPRatio(fP, fV);              };
        // **********************************************************************************************************************
        virtual auto size()                                     const -> size_t override { return _CorrState::size(); }   
        // **********************************************************************************************************************
    };
};

#endif  
// ##########################################################################################################################################
