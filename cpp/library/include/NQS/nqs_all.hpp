#include "./nqs_final.hpp"
#ifndef RBM_H											 // #
#	include "./rbm_final.hpp"						 // #
#endif													 // #
#ifndef NQSPP_H										     // #
#	include "./nqs_ref_final.hpp"                   // #
#endif													 // #

// ##########################################################################################################################################
#define NQS_REF_INST(_Ht, _type, _stateType, _CorrState) 	                                    \
    template class NQS_ref<2u, _Ht, _type, _stateType, _CorrState<2u, _Ht, _type, _stateType>>; \
    template class NQS_ref<4u, _Ht, _type, _stateType, _CorrState<4u, _Ht, _type, _stateType>>;
#define NQS_REF_INST_ALL(_CorrState)                                \
    NQS_REF_INST(double, double, double, _CorrState) 	            \
    NQS_REF_INST(double, std::complex<double>, double, _CorrState) 	\
    NQS_REF_INST(std::complex<double>, double, double, _CorrState) 	\
    NQS_REF_INST(std::complex<double>, std::complex<double>, double, _CorrState)
// ##########################################################################################################################################

// template instantiation of the reference NQS
NQS_REF_INST_ALL(NQS_S);
NQS_REF_INST_ALL(NQS_PP_S);
NQS_REF_INST_ALL(RBM_S);

// ##########################################################################################################################################