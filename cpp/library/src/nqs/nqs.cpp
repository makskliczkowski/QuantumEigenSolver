/**
* @file nqs.cpp
* @brief Implementation of Neural Quantum State (NQS) solver functions and template specializations.
*
* This file contains the implementation of various functions and template specializations
* for the Neural Quantum State (NQS) solver. The NQS solver is used for training and
* saving the configuration details of the NQS training process.
*
* The file includes:
* - Function to log the configuration details of the NQS training process.
* - Function to save the NQS information to a file.
* - Destructor for the NQS_info_t class to clean up resources.
* - Template specializations for the NQS class for different types of Hamiltonians, NQS parameters, and state values.
*
* The template specializations cover:
* 1. All real types (double) up to 4 spin modes.
* 2. All complex types (cpx) up to 4 spin modes.
* 3. Complex Hamiltonian with real NQS (cpx and double) up to 4 spin modes.
* 4. Real Hamiltonian with complex NQS (double and cpx) up to 4 spin modes.
*/

#include "../../include/NQS/nqs_final.hpp"
#include <filesystem>
#include <string>

// ##########################################################################################################################################

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
*/
void NQS_train_t::hi(const std::string& _in) const
{
    std::string outstr = "";
    strSeparatedP(outstr, ',', 2,
                VEQV(Monte Carlo Samples, this->MC_sam_),
                VEQV(Thermalization Steps, this->MC_th_),
                VEQV(Number of blocks (single sample), this->nblck_),
                VEQV(Size of the single block, this->bsize_),
                VEQV(Number of flips taken at each step, this->nFlip));
    LOGINFOG(_in + outstr, LOG_TYPES::TRACE, 1);
}

// ##########################################################################################################################################

/**
* @brief Saves the NQS (Neural Quantum State) information to a specified file.
* 
* This function saves various parameters and histories related to the NQS to an HDF5 file.
* The file can be appended to if it already exists.
* 
* @param _dir The directory where the file will be saved.
* @param _name The name of the file to save the information to. Must end with ".h5".
* @param i An integer used to index the parameters in the file.
* 
* The following information is saved:
* - Pseudoinverse (if NQS_USESR_MAT_USED is defined)
* - Regularization history
* - Number of visible units
* - Full size of the NQS
* - Learning rates history
*/
void NQS_info_t::saveInfo(const std::string& _dir, const std::string& _name, int i) const
{
    if (_name.ends_with(".h5")) {
        LOGINFO("Saving the NQS information to the file: " + _name, LOG_TYPES::INFO, 2);
        const bool _append = std::filesystem::exists(_dir + _name);
        const std::string _namePar = "parameters/" + STR(i) + "/";

        // Save pseudoinverse information
#ifdef NQS_USESR_MAT_USED
        saveAlgebraic(_dir, _name, arma::vec({ this->pinv_ }), _namePar + "pinv", _append);
#else 
        saveAlgebraic(_dir, _name, arma::vec({ 0.0 }), _namePar + "pinv", _append);
#endif

        // Save regularization history
        saveAlgebraic(_dir, _name, arma::vec(this->s_ ? this->s_->hist() : v_1d<double>({ this->sreg_ })), _namePar + "regularization", true);

        // Save the number of visible units
        saveAlgebraic(_dir, _name, arma::vec({ double(this->nVis_) }), _namePar + "visible_units", true);

        // Save the full size of the NQS
        saveAlgebraic(_dir, _name, arma::vec({ double(this->fullSize_) }), _namePar + "full_size", true);

        // Save learning rates history
        saveAlgebraic(_dir, _name, arma::vec(this->p_ ? this->p_->hist() : v_1d<double>({ this->lr_ })), _namePar + "learning_rate_history", true);
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
