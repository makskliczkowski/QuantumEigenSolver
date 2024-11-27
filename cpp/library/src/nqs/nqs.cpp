#include "../../include/NQS/nqs_final.hpp"
#include <filesystem>
#include <string>

// ##########################################################################################################################################

/*
* @brief Say hello to the NQS solver.
*/
/**
 * @brief Logs the configuration details of the NQS training process.
 *
 * This function constructs a formatted string containing various parameters
 * related to the Neural Quantum State (NQS) training process and logs it
 * with a specified log level and type.
 *
 * @param _in A string to be prefixed to the log message.
 *
 * The logged information includes:
 * - Monte Carlo Samples: The number of Monte Carlo samples used in the training.
 * - Thermalization Steps: The number of thermalization steps performed.
 * - Number of blocks (single sample): The number of blocks in a single sample.
 * - Size of the single block: The size of each block.
 * - Number of flips taken at each step: The number of flips performed at each step.
 * - Learning Rate: The learning rate used during training.
 * - Regularization Parameter: The regularization parameter applied.
 * - Number of Visible Units: The number of visible units in the NQS.
 * - Full Size: The full size of the NQS.
 */
void NQS_train_t::hi(const std::string& _in) const
{
    std::string outstr = "";
    strSeparatedP(outstr, ',', 2,
                  VEQV("Monte Carlo Samples", this->MC_sam_),
                  VEQV("Thermalization Steps", this->MC_th_),
                  VEQV("Number of blocks (single sample)", this->nblck_),
                  VEQV("Size of the single block", this->bsize_),
                  VEQV("Number of flips taken at each step", this->nFlip));
    LOGINFOG(_in + outstr, LOG_TYPES::TRACE, 1);
}

// ##########################################################################################################################################

void NQS_info_t::saveInfo(const std::string& _dir, const std::string& _name, int i) const
{
    if (_name.ends_with(".h5")) {
        LOGINFO("Saving the NQS information to the file: " + _name, LOG_TYPES::INFO, 2);
        const bool _append = std::filesystem::exists(_dir + _name);
        const std::string _namePar = "parameters/" + STR(i) + "/";

        // pseudoinverse 
#ifdef NQS_USESR_MAT_USED
        saveAlgebraic(_dir, _name, arma::vec({ this->pinv_ }), _namePar + "pinv", _append);
#else 
        saveAlgebraic(_dir, _name, arma::vec({ 0.0 }), _namePar + "pinv", _append);
#endif
        // regularization history
        saveAlgebraic(_dir, _name, arma::vec(this->s_ ? this->s_->hist() : v_1d<double>({ this->sreg_ })), _namePar + "regularization", true);

        // save the parameters
        saveAlgebraic(_dir, _name, arma::vec({ double(this->nVis_) }), _namePar + "visible", true);
        saveAlgebraic(_dir, _name, arma::vec({ double(this->fullSize_) }), _namePar + "full", true);

        // learning rates history
        saveAlgebraic(_dir, _name, arma::vec(this->p_ ? this->p_->hist() : v_1d<double>({ this->lr_ })), _namePar + "learning_rate", true);

        // 
    }
}

// ##########################################################################################################################################

NQS_info_t::~NQS_info_t()
{
    if (this->p_) {
        delete p_;
        this->p_ = nullptr;
    }

    if (this->s_) {
        delete s_;
        this->s_ = nullptr;
    }
}       

// ##########################################################################################################################################

//! specialize the NQS class for all the types <HAMILTONIAN, NQS PARAMS, STATE VALUES>

// 1) all real types - double up to 4 spin modes
template class NQS<2u, double, double, double>;
template class NQS<3u, double, double, double>;
template class NQS<4u, double, double, double>;
// 2) all complex types - cpx up to 4 spin modes
template class NQS<2u, cpx, cpx, double>;
template class NQS<3u, cpx, cpx, double>;
template class NQS<4u, cpx, cpx, double>;
// 3) complex hamiltonian with real NQS - cpx up to 4 spin modes
template class NQS<2u, cpx, double, double>;
template class NQS<3u, cpx, double, double>;
template class NQS<4u, cpx, double, double>;
// 4) real hamiltonian with complex NQS - double up to 4 spin modes
template class NQS<2u, double, cpx, double>;
template class NQS<3u, double, cpx, double>;
template class NQS<4u, double, cpx, double>;

// ##########################################################################################################################################
