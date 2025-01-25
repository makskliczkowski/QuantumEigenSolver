#ifndef NQSPP_H										   
#	include "./NQS_ref_base/nqs_ref_final.hpp"
#endif												
#ifndef RBM_H										
#	include "./rbm_final.hpp"						 
#endif												

#ifndef NQS_INST_H
#define NQS_INST_H

// ##########################################################################################################################################

// NQS - INSTANTIATIONS

// ##########################################################################################################################################

#define NQS_INST_SINGLE_EXT(_Ht, _type, _stateType, _CorrState) 	        \
    extern template class _CorrState<2u, _Ht, _type, _stateType>;           \
    extern template class _CorrState<4u, _Ht, _type, _stateType>;
#define NQS_INST_ALL_EXT(_CorrState)                                        \
    NQS_INST_SINGLE_EXT(double, double, double, _CorrState) 	            \
    NQS_INST_SINGLE_EXT(double, std::complex<double>, double, _CorrState) 	\
    NQS_INST_SINGLE_EXT(std::complex<double>, double, double, _CorrState) 	\
    NQS_INST_SINGLE_EXT(std::complex<double>, std::complex<double>, double, _CorrState)

// standard NQS
// NQS_INST_ALL_EXT(NQS_NS::NQS);
NQS_INST_ALL_EXT(NQS_NS::NQS_S);
// standard RBM
// NQS_INST_ALL_EXT(NQS_NS::RBM);
NQS_INST_ALL_EXT(NQS_NS::RBM_S);

// ##########################################################################################################################################

#define NQS_INST_SINGLE(_Ht, _type, _stateType, _CorrState) 	            \
    template class _CorrState<2u, _Ht, _type, _stateType>;                  \
    template class _CorrState<4u, _Ht, _type, _stateType>;

#define NQS_INST_ALL(_CorrState)                                            \
    NQS_INST_SINGLE(double, double, double, _CorrState) 	                \
    NQS_INST_SINGLE(double, std::complex<double>, double, _CorrState) 	    \
    NQS_INST_SINGLE(std::complex<double>, double, double, _CorrState) 	    \
    NQS_INST_SINGLE(std::complex<double>, std::complex<double>, double, _CorrState)

// ##########################################################################################################################################

// REFERENCE STATES - INSTANTIATIONS

// ##########################################################################################################################################
#define NQS_REF_INST_EXT(_Ht, _type, _stateType, _CorrState, _RefState) 	                                    \
    extern template class _RefState<2u, _Ht, _type, _stateType, _CorrState<2u, _Ht, _type, _stateType>>;  \
    extern template class _RefState<4u, _Ht, _type, _stateType, _CorrState<4u, _Ht, _type, _stateType>>;
#define NQS_REF_INST_ALL_EXT(_CorrState, _RefState)                                     \
    NQS_REF_INST_EXT(double, double, double, _CorrState, _RefState) 	                \
    NQS_REF_INST_EXT(double, std::complex<double>, double, _CorrState, _RefState) 	    \
    NQS_REF_INST_EXT(std::complex<double>, double, double, _CorrState, _RefState) 	    \
    NQS_REF_INST_EXT(std::complex<double>, std::complex<double>, double, _CorrState, _RefState)
#define NQS_REF_INST_ALL_AV_EXT(_CorrState)                                             \
    NQS_REF_INST_ALL_EXT(_CorrState, NQS_NS::NQS_ref)                                   \
    NQS_REF_INST_ALL_EXT(_CorrState, NQS_NS::NQS_PP) 
// ##########################################################################################################################################

// template instantiation of the reference NQS
// NQS_REF_INST_ALL_AV_EXT(NQS_NS::NQS_S);
NQS_REF_INST_ALL_AV_EXT(NQS_NS::RBM_S);

// ##########################################################################################################################################

#define NQS_REF_INST(_Ht, _type, _stateType, _CorrState, _RefState)                                \
    template class _RefState<2u, _Ht, _type, _stateType, _CorrState<2u, _Ht, _type, _stateType>>;  \
    template class _RefState<4u, _Ht, _type, _stateType, _CorrState<4u, _Ht, _type, _stateType>>;
#define NQS_REF_INST_ALL(_CorrState, _RefState)                                                     \
    NQS_REF_INST(double, double, double, _CorrState, _RefState) 	                                \
    NQS_REF_INST(double, std::complex<double>, double, _CorrState, _RefState) 	                    \
    NQS_REF_INST(std::complex<double>, double, double, _CorrState, _RefState) 	                    \
    NQS_REF_INST(std::complex<double>, std::complex<double>, double, _CorrState, _RefState)
#define NQS_REF_INST_ALL_AV(_CorrState)                                             \
    NQS_REF_INST_ALL(_CorrState, NQS_NS::NQS_ref)                                   \
    NQS_REF_INST_ALL(_CorrState, NQS_NS::NQS_PP)

// ##########################################################################################################################################

#endif // !NQS_INST_H