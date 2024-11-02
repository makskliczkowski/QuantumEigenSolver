#include "../include/NQS/nqs_final.hpp"

// ##########################################################################################################################################

/*
* @brief Say hello to the NQS solver.
*/
void NQS_train_t::hi(const std::string& _in) const
{
    std::string outstr	= "";
    strSeparatedP(outstr, '\t', 2,
                VEQV(Monte Carlo Samples, this->MC_sam_),
                VEQV(Thermalization Steps, this->MC_th_),
                VEQV(Number of blocks (single sample), this->nblck_),
                VEQV(Size of the single block, this->bsize_),
                VEQV(Number of flips taken at each step, this->nFlip));
    LOGINFOG(_in + outstr, LOG_TYPES::TRACE, 1);
}

// ##########################################################################################################################################
