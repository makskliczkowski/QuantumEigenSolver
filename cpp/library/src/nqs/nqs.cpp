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
#include <string_view>

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

/**
* @brief Resets the NQS object with a specified size.
*
* This function resets the derivatives and sets the size of the containers
* for the lower states to the specified size.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type.
* @tparam _stateType The state type.
* @param _n The new size for the derivatives and lower states containers.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::reset(size_t _n)
{
    this->derivatives_.reset(this->info_p_.fullSize_, _n);  // reset the derivatives - set the size of the containers
	this->lower_states_.setDerivContSize(_n);	            // set the size of the containers for the lower states
}
// template instantiation of the function above
NQS_INST_CMB_ALL(reset, void, (size_t));

// ##########################################################################################################################################

/**
* @brief Sets the state of the Neural Quantum State (NQS) based on the initialization type.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type.
* @tparam _stateType State type.
* @param _st Initialization state type.
* 
* This function sets the state of the NQS object based on the provided initialization state type.
* The possible initialization states are:
* - RANDOM: Sets the state to a random configuration.
* - NO_INIT: Does not initialize the state.
* - FERRO: Sets the state to a ferromagnetic configuration if the appropriate preprocessor directives are defined.
* - ANTI_FERRO: Sets the state to an antiferromagnetic configuration if the appropriate preprocessor directives are defined.
* 
* If the initialization state type is not recognized, the state is set to a random configuration.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void  NQS<_spinModes, _Ht, _T, _stateType>::setState(NQS_init_st_t _st)
{
    switch (_st)
    {
    case NQS_init_st_t::RANDOM:
        return this->setRandomState();
        break;
    case NQS_init_st_t::NO_INIT:
        break;
    case NQS_init_st_t::FERRO:
    {
#ifdef NQS_USE_ARMA
#   ifdef SPIN
#       ifdef NQS_USE_VEC_ONLY
        this->curVec_ = arma::ones<Config_t>(this->info_p_.nVis_) * Operators::_SPIN;
#       else
        // !TODO
#       endif
#   else
    // !TODO
#   endif 
#else
#endif
        break;
    }
    case NQS_init_st_t::ANTI_FERRO:
    {
#ifdef NQS_USE_ARMA
#   ifdef SPIN
#       ifdef NQS_USE_VEC_ONLY
        this->curVec_ = arma::ones<Config_t>(this->info_p_.nVis_) * Operators::_SPIN;
        this->curVec_(arma::span(1, this->info_p_.nVis_ - 1)) *= -1;
#       else
        // !TODO 
#       endif
#   else
    // !TODO
#   endif
#else
#endif
        break;
    }
    default:
        return this->setRandomState(true);
    } 
    this->setState(this->curVec_, true);
}
// template instantiation of the function above
NQS_INST_CMB_ALL(setState, void, (NQS_init_st_t));

// ##########################################################################################################################################

/**
* @brief Swaps the configuration of the current NQS object with another NQS object.
*
* This function exchanges the last configuration of the current NQS object with the last configuration
* of another NQS object passed as a parameter. It retrieves the last configuration of the other NQS object,
* sets the other NQS object to the current configuration, and then sets the current NQS object to the retrieved
* configuration from the other NQS object.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type.
* @tparam _stateType The state type.
* @param _other A pointer to another NQS object whose configuration will be swapped with the current object.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::swapConfig(NQS<_spinModes, _Ht, _T, _stateType>::MC_t_p _other)
{
    auto _st_other = _other->getLastConfig();           // get the last configuration of the other solver
    _other->setConfig(NQS_STATE);                       // swap the configurations
    this->setConfig(_st_other);                         // swap the configurations
}
NQS_INST_CMB_ALL(swapConfig, void, (MC_t_p));

// ##########################################################################################################################################

/**
* @brief Swaps the weights between the current NQS instance and another instance.
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for weights.
* @tparam _stateType The state type.
* @param _other A pointer to another NQS instance with which to swap weights.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::swapWeights(NQS<_spinModes, _Ht, _T, _stateType>::MC_t_p _other)
{
    // pass
}

// ##########################################################################################################################################

// CONSTRUCTORS

// ##########################################################################################################################################

/**
* @brief Copy constructor for the NQS class.
* 
* This constructor initializes a new instance of the NQS class by copying the state from another instance.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type.
* @tparam _stateType State type.
* @param _n The NQS instance to copy from.
* If the NQS_NOT_OMP_MT macro is defined, the number of threads is also initialized.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS<_spinModes, _Ht, _T, _stateType>::NQS(const NQS<_spinModes, _Ht, _T, _stateType>& _n)
    : MonteCarlo::MonteCarloSolver<_T, _stateType, arma::Col<_stateType>>(_n),
    info_p_(_n.info_p_), H_(_n.H_),
    nFlip_(_n.nFlip_), flipPlaces_(_n.flipPlaces_), flipVals_(_n.flipVals_)
{
    const int _Ns               =           _n.info_p_.nSites_;                 // get the number of sites
    // Ns and functions
	this->pRatioFunc_			= 			[this](const Config_t& _v)         { return this->pRatio(_v); };
	this->pKernelFunc_			= 			[this](int_ini_t fP, dbl_ini_t fV) { return this->pRatio(fP, fV); };
	this->logPKernelFunc_		= 			[this](int_ini_t fP, dbl_ini_t fV) { return this->logPRatio(fP, fV); };
    this->logPRatioFuncFlips_   =           [this](uint nFlips)                { return this->logPRatio(nFlips); };

    // copy the lower states info
    this->lower_states_			= 			NQS_lower_t<_spinModes, _Ht, _T, _stateType>(_Ns, _n.lower_states_.f_lower, _n.lower_states_.f_lower_b_, this);
this->lower_states_.exc_ratio_  = 		    [this](const Config_t& _v)         { return this->pRatio(_v); };
#ifdef NQS_LOWER_RATIO_LOGDIFF
	this->lower_states_.exc_ansatz_ = 		[&](const Config_t& _v)         { return this->ansatzlog(_v); };
#else
	this->lower_states_.exc_ansatz_ = 		[&](const Config_t& _v)         { return this->ansatz(_v); };
#endif
    // initialize the information 
    this->info_p_  				=           _n.info_p_;
    // copy the weights
    this->derivatives_ 			=           _n.derivatives_;
    this->dF_ 					=           _n.dF_;
    this->F_ 					=           _n.F_;
    this->Weights_ 				=           _n.Weights_;
    // this->init();
#ifdef NQS_NOT_OMP_MT
    this->initThreads(_n.threads_.threadNum_);
#endif
    {
        // reset the solver and preconditioner
        if (_n.solver_ != nullptr) 
        {
            this->solver_ = _n.solver_->clone();
        }
        // reset the preconditioner
        if (_n.precond_ != nullptr) 
        {
            this->precond_ = _n.precond_->clone();
        }
    }
}

// template instantiation of the function above
template NQS<2u, double, double, double>::NQS(const NQS<2u, double, double, double>&);
template NQS<3u, double, double, double>::NQS(const NQS<3u, double, double, double>&);
template NQS<4u, double, double, double>::NQS(const NQS<4u, double, double, double>&);
template NQS<2u, cpx, cpx, double>::NQS(const NQS<2u, cpx, cpx, double>&);
template NQS<3u, cpx, cpx, double>::NQS(const NQS<3u, cpx, cpx, double>&);
template NQS<4u, cpx, cpx, double>::NQS(const NQS<4u, cpx, cpx, double>&);
template NQS<2u, cpx, double, double>::NQS(const NQS<2u, cpx, double, double>&);
template NQS<3u, cpx, double, double>::NQS(const NQS<3u, cpx, double, double>&);
template NQS<4u, cpx, double, double>::NQS(const NQS<4u, cpx, double, double>&);
template NQS<2u, double, cpx, double>::NQS(const NQS<2u, double, cpx, double>&);
template NQS<3u, double, cpx, double>::NQS(const NQS<3u, double, cpx, double>&);
template NQS<4u, double, cpx, double>::NQS(const NQS<4u, double, cpx, double>&);

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS<_spinModes, _Ht, _T, _stateType>::NQS(NQS<_spinModes, _Ht, _T, _stateType>&& _n)
    : MonteCarlo::MonteCarloSolver<_T, _stateType, arma::Col<_stateType>>(std::move(_n)),
    info_p_(std::move(_n.info_p_)),
    H_(std::move(_n.H_)), nFlip_(_n.nFlip_), 
    flipPlaces_(std::move(_n.flipPlaces_)), flipVals_(std::move(_n.flipVals_))
{
    this->accepted_             = _n.accepted_;
    this->total_ 				= _n.total_;
    this->info_ 				= _n.info_;
    // copy the weights
    this->derivatives_ 			= std::move(_n.derivatives_);
    this->dF_ 					= std::move(_n.dF_);
    this->F_ 					= std::move(_n.F_);
    this->Weights_ 				= std::move(_n.Weights_);
    // setup the functions as in the constructor
    this->pRatioFunc_			= [this](const Config_t& _v)         { return this->pRatio(_v); };
    this->pKernelFunc_			= [this](int_ini_t fP, dbl_ini_t fV) { return this->pRatio(fP, fV); };
    this->logPKernelFunc_		= [this](int_ini_t fP, dbl_ini_t fV) { return this->logPRatio(fP, fV); };
    this->logPRatioFuncFlips_   = [this](uint nFlips)                { return this->logPRatio(nFlips); };

    // copy the lower states info
    this->lower_states_			    = std::move(_n.lower_states_);
    this->lower_states_.exc_ratio_  = [this](const Config_t& _v)     { return this->pRatio(_v); };
#ifdef NQS_LOWER_RATIO_LOGDIFF
    this->lower_states_.exc_ansatz_ = [&](const Config_t& _v)        { return this->ansatzlog(_v); };
#else
    this->lower_states_.exc_ansatz_ = 		[&](const Config_t& _v)         { return this->ansatz(_v); };
#endif
#ifdef NQS_NOT_OMP_MT
    this->initThreads(_n.threads_.threadNum_);
#endif
    // move the preconditioners and solvers
    {
        // reset the solver and preconditioner
        if (_n.solver_ != nullptr) 
        {
            this->solver_ = _n.solver_->move();
        }
        // reset the preconditioner
        if (_n.precond_ != nullptr) 
        {
            this->precond_ = _n.precond_->move();
        }
    }
}

// template instantiation of the function above
template NQS<2u, double, double, double>::NQS(NQS<2u, double, double, double>&&);
template NQS<3u, double, double, double>::NQS(NQS<3u, double, double, double>&&);
template NQS<4u, double, double, double>::NQS(NQS<4u, double, double, double>&&);
template NQS<2u, cpx, cpx, double>::NQS(NQS<2u, cpx, cpx, double>&&);
template NQS<3u, cpx, cpx, double>::NQS(NQS<3u, cpx, cpx, double>&&);
template NQS<4u, cpx, cpx, double>::NQS(NQS<4u, cpx, cpx, double>&&);
template NQS<2u, cpx, double, double>::NQS(NQS<2u, cpx, double, double>&&);
template NQS<3u, cpx, double, double>::NQS(NQS<3u, cpx, double, double>&&);
template NQS<4u, cpx, double, double>::NQS(NQS<4u, cpx, double, double>&&);
template NQS<2u, double, cpx, double>::NQS(NQS<2u, double, cpx, double>&&);
template NQS<3u, double, cpx, double>::NQS(NQS<3u, double, cpx, double>&&);
template NQS<4u, double, cpx, double>::NQS(NQS<4u, double, cpx, double>&&);

// ##########################################################################################################################################

/**
* @brief Assignment operator for the NQS class.
* 
* This operator assigns the values from another NQS object to the current object.
* It performs a deep copy of all member variables from the source object to the destination object.
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The type of the Hamiltonian.
* @tparam _T The type of the parameters.
* @tparam _stateType The type of the state.
* @param _n The source NQS object to copy from.
* @return NQS<_spinModes, _Ht, _T, _stateType>& A reference to the current object after assignment.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS<_spinModes, _Ht, _T, _stateType>& NQS<_spinModes, _Ht, _T, _stateType>::operator=(const NQS<_spinModes, _Ht, _T, _stateType>& _n)
{
    if (this != &_n) 
    {
        this->H_                    = _n.H_;
        this->info_p_               = _n.info_p_;
        this->nFlip_                = _n.nFlip_;
        this->flipPlaces_           = _n.flipPlaces_;
        this->flipVals_             = _n.flipVals_;
        this->accepted_             = _n.accepted_;
        this->total_                = _n.total_;
        this->info_                 = _n.info_;
        this->lower_states_         = _n.lower_states_;
        this->derivatives_          = _n.derivatives_;
        this->dF_                   = _n.dF_;
        this->F_                    = _n.F_;
        this->Weights_              = _n.Weights_;
        // setup the functions as in the constructor
        this->pRatioFunc_           = [this](const Config_t& _v)         { return this->pRatio(_v); };
        this->pKernelFunc_          = [this](int_ini_t fP, dbl_ini_t fV) { return this->pRatio(fP, fV); };
        this->logPKernelFunc_       = [this](int_ini_t fP, dbl_ini_t fV) { return this->logPRatio(fP, fV); };
        this->logPRatioFuncFlips_   = [this](uint nFlips)                { return this->logPRatio(nFlips); };
        this->lower_states_.exc_ratio_ = [this](const Config_t& _v)      { return this->pRatio(_v); };
#ifdef NQS_LOWER_RATIO_LOGDIFF
        this->lower_states_.exc_ansatz_ = [&](const Config_t& _v)        { return this->ansatzlog(_v); };
#else
        this->lower_states_.exc_ansatz_ = [&](const Config_t& _v)        { return this->ansatz(_v); };
#endif
#ifdef NQS_NOT_OMP_MT
        this->initThreads(_n.threads_.threadNum_);
#endif
        {
            // reset the solver and preconditioner
            if (_n.solver_ != nullptr) 
            {
                this->solver_ = _n.solver_->clone();
            }
            // reset the preconditioner
            if (_n.precond_ != nullptr) 
            {
                this->precond_ = _n.precond_->clone();
            }
        }
    }
    return *this;
}

// template instantiation of the function above
template NQS<2u, double, double, double>& NQS<2u, double, double, double>::operator=(const NQS<2u, double, double, double>&);
template NQS<3u, double, double, double>& NQS<3u, double, double, double>::operator=(const NQS<3u, double, double, double>&);
template NQS<4u, double, double, double>& NQS<4u, double, double, double>::operator=(const NQS<4u, double, double, double>&);
template NQS<2u, cpx, cpx, double>& NQS<2u, cpx, cpx, double>::operator=(const NQS<2u, cpx, cpx, double>&);
template NQS<3u, cpx, cpx, double>& NQS<3u, cpx, cpx, double>::operator=(const NQS<3u, cpx, cpx, double>&);
template NQS<4u, cpx, cpx, double>& NQS<4u, cpx, cpx, double>::operator=(const NQS<4u, cpx, cpx, double>&);
template NQS<2u, cpx, double, double>& NQS<2u, cpx, double, double>::operator=(const NQS<2u, cpx, double, double>&);
template NQS<3u, cpx, double, double>& NQS<3u, cpx, double, double>::operator=(const NQS<3u, cpx, double, double>&);
template NQS<4u, cpx, double, double>& NQS<4u, cpx, double, double>::operator=(const NQS<4u, cpx, double, double>&);
template NQS<2u, double, cpx, double>& NQS<2u, double, cpx, double>::operator=(const NQS<2u, double, cpx, double>&);
template NQS<3u, double, cpx, double>& NQS<3u, double, cpx, double>::operator=(const NQS<3u, double, cpx, double>&);
template NQS<4u, double, cpx, double>& NQS<4u, double, cpx, double>::operator=(const NQS<4u, double, cpx, double>&);

// ##########################################################################################################################################

/**
* @brief Clones the state of another NQS object into this one.
*
* This function attempts to cast the provided object to an NQS object of the same template parameters.
* If the cast is successful, it copies the internal state of the other NQS object into this one.
* If the cast fails, it logs an error message.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type.
* @tparam _stateType The state type.
* @param _other A shared pointer to the Monte Carlo object to clone from.
*
* @throws std::bad_cast If the dynamic cast fails.
* @throws std::exception If any other exception occurs during the cloning process.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::clone(MC_t_p _other)
{
    try 
    {
        // cast the other object to the NQS type
        auto _n = std::dynamic_pointer_cast<NQS<_spinModes, _Ht, _T, _stateType>>(_other);

        // check if the cast was successful
        if (_n) 
        {
            this->info_p_               = _n->info_p_;
            this->H_                    = _n->H_;
            this->nFlip_                = _n->nFlip_;
            this->flipPlaces_           = _n->flipPlaces_;
            this->flipVals_             = _n->flipVals_;
            this->accepted_             = _n->accepted_;
            this->total_                = _n->total_;
            this->info_                 = _n->info_;
            this->lower_states_         = _n->lower_states_;
            this->derivatives_          = _n->derivatives_;
            this->dF_                   = _n->dF_;
            this->F_                    = _n->F_;
            this->Weights_              = _n->Weights_;
            this->pRatioFunc_           = [this](const Config_t& _v)         { return this->pRatio(_v); };
            this->pKernelFunc_          = [this](int_ini_t fP, dbl_ini_t fV) { return this->pRatio(fP, fV); };
            this->logPKernelFunc_       = [this](int_ini_t fP, dbl_ini_t fV) { return this->logPRatio(fP, fV); };
            this->logPRatioFuncFlips_   = [this](uint nFlips)                { return this->logPRatio(nFlips); };
            this->lower_states_.exc_ratio_ = [this](const Config_t& _v)      { return this->pRatio(_v); };
#ifdef NQS_LOWER_RATIO_LOGDIFF
            this->lower_states_.exc_ansatz_ = [&](const Config_t& _v)        { return this->ansatzlog(_v); };
#else
            this->lower_states_.exc_ansatz_ = [&](const Config_t& _v)        { return this->ansatz(_v); };
#endif
#ifdef NQS_NOT_OMP_MT
            this->initThreads(_n->threads_.threadNum_);
#endif
            {
                // reset the solver and preconditioner
                if (_n->solver_ != nullptr) 
                {
                    this->solver_ = _n->solver_->clone();
                }
                // reset the preconditioner
                if (_n->precond_ != nullptr) 
                {
                    this->precond_ = _n->precond_->clone();
                }
            }
        }
    }
    catch (const std::bad_cast& e) 
    {
        LOGINFO("Error in cloning the NQS object: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
    }
    catch (const std::exception& e) 
    {
        LOGINFO("Error in cloning the NQS object: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
    }
}
NQS_INST_CMB_ALL(clone, void, (MC_t_p)); 

// ##########################################################################################################################################

/**
* @brief Move assignment operator for the NQS class.
* 
* This operator allows for the move assignment of an NQS object. It transfers the ownership of resources
* from the source object (_n) to the current object (*this). The move assignment operator ensures that 
* the current object takes over the resources of the source object, leaving the source object in a valid 
* but unspecified state.
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type.
* @tparam _stateType The state type.
* @param _n The source NQS object to be moved.
* @return NQS<_spinModes, _Ht, _T, _stateType>& A reference to the current object after the move.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS<_spinModes, _Ht, _T, _stateType>& NQS<_spinModes, _Ht, _T, _stateType>::operator=(NQS<_spinModes, _Ht, _T, _stateType>&& _n)
{
    if (this != &_n) 
    {
        // move the information
        this->H_ 					= std::move(_n.H_);
        this->nFlip_ 				= _n.nFlip_;
        this->flipPlaces_ 			= std::move(_n.flipPlaces_);
        this->flipVals_ 			= std::move(_n.flipVals_);
        this->info_p_ 				= std::move(_n.info_p_);
        this->lower_states_ 		= std::move(_n.lower_states_);
        this->derivatives_ 			= std::move(_n.derivatives_);
        this->dF_ 					= std::move(_n.dF_);
        this->F_ 					= std::move(_n.F_);
        this->Weights_ 				= std::move(_n.Weights_);
#ifdef NQS_NOT_OMP_MT
        this->initThreads(_n.threads_.threadNum_);
#endif
        {
            // move the solver and preconditioner
            this->solver_ = std::move(_n.solver_);
            this->precond_ = std::move(_n.precond_);
        }
    }
    return *this;
}

// template instantiation of the function above
template NQS<2u, double, double, double>& NQS<2u, double, double, double>::operator=(NQS<2u, double, double, double>&&);
template NQS<3u, double, double, double>& NQS<3u, double, double, double>::operator=(NQS<3u, double, double, double>&&);
template NQS<4u, double, double, double>& NQS<4u, double, double, double>::operator=(NQS<4u, double, double, double>&&);
template NQS<2u, cpx, cpx, double>& NQS<2u, cpx, cpx, double>::operator=(NQS<2u, cpx, cpx, double>&&);
template NQS<3u, cpx, cpx, double>& NQS<3u, cpx, cpx, double>::operator=(NQS<3u, cpx, cpx, double>&&);
template NQS<4u, cpx, cpx, double>& NQS<4u, cpx, cpx, double>::operator=(NQS<4u, cpx, cpx, double>&&);
template NQS<2u, cpx, double, double>& NQS<2u, cpx, double, double>::operator=(NQS<2u, cpx, double, double>&&);
template NQS<3u, cpx, double, double>& NQS<3u, cpx, double, double>::operator=(NQS<3u, cpx, double, double>&&);
template NQS<4u, cpx, double, double>& NQS<4u, cpx, double, double>::operator=(NQS<4u, cpx, double, double>&&);
template NQS<2u, double, cpx, double>& NQS<2u, double, cpx, double>::operator=(NQS<2u, double, cpx, double>&&);
template NQS<3u, double, cpx, double>& NQS<3u, double, cpx, double>::operator=(NQS<3u, double, cpx, double>&&);
template NQS<4u, double, cpx, double>& NQS<4u, double, cpx, double>::operator=(NQS<4u, double, cpx, double>&&);

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS<_spinModes, _Ht, _T, _stateType>::NQS(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p)
    : MonteCarlo::MonteCarloSolver<_T, _stateType, arma::Col<_stateType>>(), H_(_p.H_)
{
    const size_t _Ns            =           H_->getNs();
    const size_t _Nh            =           H_->getHilbertSize();
    // set the functions
    this->pRatioFunc_			= 			[this](const Config_t& _v)         { return this->pRatio(_v); };
    this->pKernelFunc_			= 			[this](int_ini_t fP, dbl_ini_t fV) { return this->pRatio(fP, fV); };
    this->logPKernelFunc_		= 			[this](int_ini_t fP, dbl_ini_t fV) { return this->logPRatio(fP, fV); };
    this->logPRatioFuncFlips_   =           [this](uint nFlips)                { return this->logPRatio(nFlips); };

	// set the visible layer (for hardcore-bosons we have the same number as sites but fermions introduce twice the complication)
    this->info_p_               =           NQS_info_t(_Ns, _Ns, 
                                            (_p.nPart_ < 0 || this->spinModes_ == 2) ? this->info_p_.nSites_ : (uint)_p.nPart_, _Nh);
#ifdef NQS_NOT_OMP_MT
	this->initThreads(_p.threadNum_);
#endif
	LOGINFO("Constructed the general NQS class", LOG_TYPES::TRACE, 2);
}
// template instantiation of the function above
template NQS<2u, double, double, double>::NQS(const NQS_Const_par_t<2u, double, double, double>&);
template NQS<3u, double, double, double>::NQS(const NQS_Const_par_t<3u, double, double, double>&);
template NQS<4u, double, double, double>::NQS(const NQS_Const_par_t<4u, double, double, double>&);
template NQS<2u, cpx, cpx, double>::NQS(const NQS_Const_par_t<2u, cpx, cpx, double>&);
template NQS<3u, cpx, cpx, double>::NQS(const NQS_Const_par_t<3u, cpx, cpx, double>&);
template NQS<4u, cpx, cpx, double>::NQS(const NQS_Const_par_t<4u, cpx, cpx, double>&);
template NQS<2u, cpx, double, double>::NQS(const NQS_Const_par_t<2u, cpx, double, double>&);
template NQS<3u, cpx, double, double>::NQS(const NQS_Const_par_t<3u, cpx, double, double>&);
template NQS<4u, cpx, double, double>::NQS(const NQS_Const_par_t<4u, cpx, double, double>&);
// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS<_spinModes, _Ht, _T, _stateType>::NQS(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
    : NQS<_spinModes, _Ht, _T, _stateType>(_p)
{
    const size_t _Ns                =       H_->getNs();
    // copy the lower states info
	this->lower_states_			    = 		NQS_lower_t<_spinModes, _Ht, _T, _stateType>(_Ns, _lower, _beta, this);
    this->lower_states_.exc_ratio_  = 		[this](const Config_t& _v)         { return this->pRatio(_v); };
#ifdef NQS_LOWER_RATIO_LOGDIFF
	this->lower_states_.exc_ansatz_ = 		[&](const Config_t& _v)         { return this->ansatzlog(_v); };
#else
	this->lower_states_.exc_ansatz_ = 		[&](const Config_t& _v)         { return this->ansatz(_v); };
#endif
}
// template instantiation of the function above
template NQS<2u, double, double, double>::NQS(const NQS_Const_par_t<2u, double, double, double>&, const NQSLS_p&, const std::vector<double>&);
template NQS<3u, double, double, double>::NQS(const NQS_Const_par_t<3u, double, double, double>&, const NQSLS_p&, const std::vector<double>&);
template NQS<4u, double, double, double>::NQS(const NQS_Const_par_t<4u, double, double, double>&, const NQSLS_p&, const std::vector<double>&);
template NQS<2u, cpx, cpx, double>::NQS(const NQS_Const_par_t<2u, cpx, cpx, double>&, const NQSLS_p&, const std::vector<double>&);
template NQS<3u, cpx, cpx, double>::NQS(const NQS_Const_par_t<3u, cpx, cpx, double>&, const NQSLS_p&, const std::vector<double>&);
template NQS<4u, cpx, cpx, double>::NQS(const NQS_Const_par_t<4u, cpx, cpx, double>&, const NQSLS_p&, const std::vector<double>&);
template NQS<2u, cpx, double, double>::NQS(const NQS_Const_par_t<2u, cpx, double, double>&, const NQSLS_p&, const std::vector<double>&);
template NQS<3u, cpx, double, double>::NQS(const NQS_Const_par_t<3u, cpx, double, double>&, const NQSLS_p&, const std::vector<double>&);
template NQS<4u, cpx, double, double>::NQS(const NQS_Const_par_t<4u, cpx, double, double>&, const NQSLS_p&, const std::vector<double>&);
// ##########################################################################################################################################

/**
* @brief General destructor of the NQS class.
* 
* This destructor ensures proper cleanup of resources used by the NQS object.
* It waits for any running threads to finish if multithreading is enabled.
* It also deletes dynamically allocated resources to prevent memory leaks.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS<_spinModes, _Ht, _T, _stateType>::~NQS()
{
    DESTRUCTOR_CALL;
#if defined NQS_NOT_OMP_MT
    for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++)
    {
        std::unique_lock<std::mutex> lock(this->threads_.kernels_[_thread].mutex);
        this->threads_.kernels_[_thread].flagThreadKill_    = true;
        this->threads_.kernels_[_thread].end_               = true;
        this->threads_.kernels_[_thread].flagThreadRun_     = 1;
        this->threads_.kernels_[_thread].cv.notify_all();
    }
    for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++)
        if (this->threads_.threads_[_thread].joinable())
            this->threads_.threads_[_thread].join();
#endif    
    // ######################################################################################################################################
}
NQS_INST_CMB_ALL(~NQS, void, ());

// ##########################################################################################################################################

// ############################################################# S E T T E R S ##############################################################

// ##########################################################################################################################################

/**
* @brief Updates the current state processed by the NQS.
* @param _st Column vector to be set as a new state
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::setState(Config_cr_t _st)
{
    NQS_STATE = _st;
#ifndef NQS_USE_VEC_ONLY
    this->curState_ = BASE_TO_INT<u64>(_st, this->discVal_);
#endif
    if (this->a_mod_p_.modified_)
        this->a_mod_p_.logAMod_ = this->logAnsatzModifier(NQS_STATE);
}
NQS_INST_CMB_ALL(setState, void, (const Config_t&));

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief Sets the current state of the NQS object.
* 
* This function sets the current state of the NQS (Neural Quantum State) object
* using the provided state value. It updates the internal state representation
* and converts the state to a vector representation.
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type.
* @tparam _stateType The state type.
* @param _st The state value to set.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::setState(u64 _st)
{
#ifndef NQS_USE_VEC_ONLY
	this->curState_ = _st;
#endif
	INT_TO_BASE(_st, this->curVec_, this->discVal_);
    if (this->a_mod_p_.modified_)
        this->a_mod_p_.logAMod_ = this->logAnsatzModifier(NQS_STATE);
}
NQS_INST_CMB_ALL(setState, void, (u64));

// ##########################################################################################################################################

/**
* @brief Allocates the main gradient parameters and the temporary and current vectors.
* The vectors are set to ones for the start.  
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::allocate()
{
	// allocate gradients
	this->F_.resize(this->info_p_.fullSize_);
    this->dF_.resize(this->info_p_.fullSize_);
    if (this->Weights_.size() != this->info_p_.fullSize_)
        this->Weights_.resize(this->info_p_.fullSize_);
#ifdef NQS_USESR
#	ifndef NQS_USESR_NOMAT
	this->S_.resize(this->info_p_.fullSize_, this->info_p_.fullSize_);
#	endif
#endif
	this->curVec_ = arma::ones(this->info_p_.nVis_);
	this->tmpVec_ = arma::ones(this->info_p_.nVis_);
}
NQS_INST_CMB_ALL(allocate, void, ());

// ##########################################################################################################################################

// ############################################################# W E I G H T S ##############################################################

// ##########################################################################################################################################

/**
* @brief Saves the weights of the neural quantum state (NQS) to a specified file.
*
* This function attempts to save the weights to a given path and file name. If the save operation fails
* and the specified file name is not "weights.h5", it will attempt to save the weights to a default file
* named "weights.h5".
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for the weights.
* @tparam _stateType The state type.
* @param _path The directory path where the weights file will be saved.
* @param _file The name of the file to save the weights to.
* @return True if the weights were saved successfully, false otherwise.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
bool NQS<_spinModes, _Ht, _T, _stateType>::saveWeights(std::string _path, std::string _file)
{
	LOGINFO("Saving the checkpoint configuration.", LOG_TYPES::INFO, 2, '#');
	auto _isSaved = saveAlgebraic(_path, STR(this->lower_states_.f_lower_size_) + _file, this->Weights_, "weights/" + STRP(this->beta_, 5));	// save the weights to a given path
	if (!_isSaved && (_file != "weights.h5"))													    // if not saved properly
	{
		LOGINFO("Couldn't save the weights to the given path.", LOG_TYPES::ERROR, 3);
		LOGINFO("Saving to default... ", LOG_TYPES::ERROR, 3);
		return this->saveWeights(_path, STR(this->lower_states_.f_lower_size_) + "_weights.h5");
	}
	return _isSaved;
}
NQS_INST_CMB_ALL(saveWeights, bool, (std::string, std::string));

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief Sets the weights of the Neural Quantum State (NQS) from a specified file.
*
* This function loads the checkpoint weights from a given file and path, and sets them
* to the current instance of the NQS. It uses the `loadAlgebraic` function to perform
* the loading operation.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for the weights.
* @tparam _stateType The state type of the NQS.
* 
* @param _path The path to the directory containing the weights file.
* @param _file The name of the file containing the weights.
* @return true if the weights were successfully loaded, false otherwise.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool NQS<_spinModes, _Ht, _T, _stateType>::setWeights(std::string _path, std::string _file)
{
	LOGINFO("Loading the checkpoint weights:", LOG_TYPES::INFO, 2);
	return loadAlgebraic(_path, STR(this->lower_states_.f_lower_size_) + _file, this->Weights_, "weights/" + STRP(this->beta_, 5));
}
NQS_INST_CMB_ALL(setWeights, bool, (std::string, std::string));


/**
* @brief Sets the weights of the Neural Quantum State (NQS) object.
* 
* This function sets the weights of the current NQS object to the weights of the provided NQS object.
* It then calls the setWeights() function to apply these weights.
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for weights.
* @tparam _stateType The state type.
* @param _nqs A shared pointer to another NQS object from which to copy the weights.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setWeights(std::shared_ptr<NQS<_spinModes, _Ht, _T, _stateType>> _nqs)
{
    this->Weights_ = _nqs->Weights_;
    this->setWeights();
}
template void NQS<2u, double, double, double>::setWeights(std::shared_ptr<NQS<2u, double, double, double>>);
template void NQS<3u, double, double, double>::setWeights(std::shared_ptr<NQS<3u, double, double, double>>);
template void NQS<4u, double, double, double>::setWeights(std::shared_ptr<NQS<4u, double, double, double>>);
template void NQS<2u, cpx, cpx, double>::setWeights(std::shared_ptr<NQS<2u, cpx, cpx, double>>);
template void NQS<3u, cpx, cpx, double>::setWeights(std::shared_ptr<NQS<3u, cpx, cpx, double>>);
template void NQS<4u, cpx, cpx, double>::setWeights(std::shared_ptr<NQS<4u, cpx, cpx, double>>);
template void NQS<2u, cpx, double, double>::setWeights(std::shared_ptr<NQS<2u, cpx, double, double>>);
template void NQS<3u, cpx, double, double>::setWeights(std::shared_ptr<NQS<3u, cpx, double, double>>);
template void NQS<4u, cpx, double, double>::setWeights(std::shared_ptr<NQS<4u, cpx, double, double>>);
template void NQS<2u, double, cpx, double>::setWeights(std::shared_ptr<NQS<2u, double, cpx, double>>);
template void NQS<3u, double, cpx, double>::setWeights(std::shared_ptr<NQS<3u, double, cpx, double>>);
template void NQS<4u, double, cpx, double>::setWeights(std::shared_ptr<NQS<4u, double, cpx, double>>);

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::update(uint nFlips)
{
    if (this->a_mod_p_.modified_)
        this->a_mod_p_.logAMod_ = this->a_mod_p_.logTmpAMod_;   // update the logarithm of the ansatz modifier
}
NQS_INST_CMB_ALL(update, void, (uint));
// ##########################################################################################################################################

/**
* @brief Updates the Neural Quantum State (NQS) with the given configuration and number of flips.
*
* This function updates the NQS based on the provided configuration and the number of spin flips.
* If the NQS has been modified, it updates the logarithm of the ansatz modifier based on the given modifier.
*
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type.
* @tparam _stateType State type.
* @param _v The configuration to update the NQS with.
* @param nFlips The number of spin flips to consider for the update.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::update(Config_cr_t _v, uint nFlips)
{
    if (this->modified()) {
        // what to do with this vector?
        this->a_mod_p_.logAMod_ = this->logAnsatzModifier(nFlips);
    }
}
NQS_INST_CMB_ALL(update, void, (Config_cr_t, uint));


/**
* @brief Sets the weights of the Neural Quantum State
* 
* This method assigns new weights to the Neural Quantum State (NQS).
* The weights determine the parameters of the neural network representation
* of the quantum state.
* 
* @param _w New weights to be assigned (NQSB type)
* 
* @note The dimensions of _w should match the architecture of the NQS
* 
* @see NQSB
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setWeights(const NQSB& _w)
{
    this->Weights_ = _w;
    this->setWeights();
}
// template instantiation of the function above
template void NQS<2u, double, double, double>::setWeights(const arma::Col<double>&);
template void NQS<3u, double, double, double>::setWeights(const arma::Col<double>&);
template void NQS<4u, double, double, double>::setWeights(const arma::Col<double>&);
template void NQS<2u, cpx, cpx, double>::setWeights(const arma::Col<cpx>&);
template void NQS<3u, cpx, cpx, double>::setWeights(const arma::Col<cpx>&);
template void NQS<4u, cpx, cpx, double>::setWeights(const arma::Col<cpx>&);
template void NQS<2u, cpx, double, double>::setWeights(const arma::Col<double>&);
template void NQS<3u, cpx, double, double>::setWeights(const arma::Col<double>&);
template void NQS<4u, cpx, double, double>::setWeights(const arma::Col<double>&);
template void NQS<2u, double, cpx, double>::setWeights(const arma::Col<cpx>&);
template void NQS<3u, double, cpx, double>::setWeights(const arma::Col<cpx>&);
template void NQS<4u, double, cpx, double>::setWeights(const arma::Col<cpx>&);
// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::setWeights(NQSB&& _w)
{
    this->Weights_ = std::move(_w);
    this->setWeights();
}
// template instantiation of the function above
template void NQS<2u, double, double, double>::setWeights(arma::Col<double>&&);
template void NQS<3u, double, double, double>::setWeights(arma::Col<double>&&);
template void NQS<4u, double, double, double>::setWeights(arma::Col<double>&&);
template void NQS<2u, cpx, cpx, double>::setWeights(arma::Col<cpx>&&);
template void NQS<3u, cpx, cpx, double>::setWeights(arma::Col<cpx>&&);
template void NQS<4u, cpx, cpx, double>::setWeights(arma::Col<cpx>&&);
template void NQS<2u, cpx, double, double>::setWeights(arma::Col<double>&&);
template void NQS<3u, cpx, double, double>::setWeights(arma::Col<double>&&);
template void NQS<4u, cpx, double, double>::setWeights(arma::Col<double>&&);
template void NQS<2u, double, cpx, double>::setWeights(arma::Col<cpx>&&);
template void NQS<3u, double, cpx, double>::setWeights(arma::Col<cpx>&&);
template void NQS<4u, double, cpx, double>::setWeights(arma::Col<cpx>&&);

// ##########################################################################################################################################

/**
* @brief Updates the weights of the neural network by adding the gradient to the current weights
* 
* This method performs a simple update step where the gradient (dF_) is added to
* the current weights (F_) of the neural network.
* 
* @tparam _spinModes Number of spin modes in the system
* @tparam _Ht Type of the Hamiltonian
* @tparam _T Numeric data type (typically float or double)
* @tparam _stateType Type representing the quantum state
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void NQS<_spinModes, _Ht, _T, _stateType>::updateWeights()
{
    this->Weights_ -= this->dF_;
}
NQS_INST_CMB_ALL(updateWeights, void, ());

// ##########################################################################################################################################

// ##########################################################################################################################################

/**
* @brief Sets a random state for the Neural Quantum State (NQS) object.
*
* This function sets a random state for the NQS object based on the value of `discVal_`.
* If `discVal_` is 0.5, the state is set to a random binary state with values -0.5 or 0.5.
* If `discVal_` is 1, the state is set to a random binary state with values 0 or 1.
* Otherwise, the state is set to a random integer state.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type.
* @tparam _stateType The state type.
* @param _upd A boolean flag indicating whether to update the state.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::setRandomState(bool _upd)
{
    constexpr double DISC_VAL_HALF  = 0.5;
    constexpr double DISC_VAL_ONE   = 1.0;

    NQSS randomState;

    if (this->discVal_ == DISC_VAL_HALF) 
    {
#ifdef SPIN
        randomState = arma::randi<NQSS>(this->info_p_.nVis_, arma::distr_param(0, 1)) - 0.5;
#else
        randomState = arma::randi<NQSS>(this->info_p_.nVis_, arma::distr_param(0, 1)) * 0.5;
#endif
    } 
    else if (this->discVal_ == DISC_VAL_ONE) 
        randomState = arma::randi<NQSS>(this->info_p_.nVis_, arma::distr_param(0, 1));
    else 
    {
        // Validate discVal_
        if (this->discVal_ < 0 || this->discVal_ > this->info_p_.Nh_) {
            throw std::invalid_argument("Invalid discVal_ value");
        }
        this->setState(this->ran_->template randomInt<u64>(0, this->info_p_.Nh_ - 1), _upd);
        return; // Exit early after setting state
    }

    this->setState(randomState, _upd);
}
NQS_INST_CMB_ALL(setRandomState, void, (bool));

// ##########################################################################################################################################

/**
* @brief Initializes the threads for the Neural Quantum State (NQS) computation.
* 
* This function sets up the number of threads to be used for parallel computation.
* If multithreading is enabled, it initializes the threads and assigns the computation
* tasks to them. If OpenMP is enabled, it sets the number of OpenMP threads. Otherwise,
* it manually creates and manages the threads.
* 
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type used for computations.
* @tparam _stateType The type representing the state.
* @param _threadNum The number of threads to be initialized.
* @return true if the threads are successfully initialized, false otherwise.
* 
* @note If NQS_USE_MULTITHREADING is not defined, only one thread will be used.
* @note If NQS_USE_OMP is defined, OpenMP will be used for thread management.
* @note If an exception occurs during thread initialization, an error message is logged
*       and the function returns false.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
bool NQS<_spinModes, _Ht, _T, _stateType>::initThreads(uint _threadNum)
{
#ifndef NQS_USE_MULTITHREADING
    this->threads_.threadNum_		=	1;
#else
    this->threads_.threadNum_		=	std::min(_threadNum, this->info_p_.nSites_);
    this->threads_.threadsNumLeft_	=	std::max(_threadNum - this->threads_.threadNum_, (uint)1);
#endif
#if defined NQS_USE_MULTITHREADING      // Use threads for all consecutive parallel regions
    try 
    {
        // reset the threads if they are already initialized
        {
            // wait for the threads to finish their work
            for (int _thread = 0; _thread < this->threads_.kernels_.size(); _thread++)
            {
                std::unique_lock<std::mutex> lock(this->threads_.kernels_[_thread].mutex);
                this->threads_.kernels_[_thread].flagThreadKill_    = true;
                this->threads_.kernels_[_thread].end_               = true;
                this->threads_.kernels_[_thread].flagThreadRun_     = 1;
                this->threads_.kernels_[_thread].cv.notify_all();
            }
            {
                std::unique_lock<std::mutex> lock(this->threads_.mutex);
                this->threads_.threads_.clear();
                this->threads_.kernels_.clear();
            }
            {
                std::unique_lock<std::mutex> lock(this->threads_.mutex);
                this->threads_.threads_.reserve(this->threads_.threadNum_);
                this->threads_.kernels_	        =   v_1d<CondVarKernel<_T>>(this->threads_.threadNum_);
                // set the flags back to false
                for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++)
                {
                    this->threads_.kernels_[_thread].flagThreadKill_    = false;
                    this->threads_.kernels_[_thread].end_               = false;
                    this->threads_.kernels_[_thread].flagThreadRun_     = 0;
                }
            }
        }
#ifdef NQS_USE_OMP
		omp_set_num_threads(this->threadNum_);   
#else
		// calculate how many sites goes to one thread
		uint _siteStep                  =   std::ceil(this->info_p_.nSites_ / 1.0 / this->threads_.threadNum_);

		// start the threads that calculate the energy with the local energy kernel function
		// this function waits for the specific energy calculation to be ready on each thread
		// this is handled through "flagThreadRun_" member
		for (uint i = 0; i < this->threads_.threadNum_; i++)
		{
			std::function<void()> lambda = [this, i, _siteStep]() 
				{ 
					this->locEnKernel(i * _siteStep, std::min((i + 1) * _siteStep, this->info_p_.nSites_), i); 
				};
			this->threads_.threads_.emplace_back(std::thread(lambda));
		}
#endif
    }
    catch (std::exception& e) 
    {
        LOGINFO("Error in initializing the threads: " + std::string(e.what()), LOG_TYPES::ERROR, 3);
        return false;
    }
#elif NQS_USE_GPU
    // !TODO: Implement GPU support

#endif
	return true;
}
NQS_INST_CMB_ALL(initThreads, bool, (uint));

// ##########################################################################################################################################

/**
* @brief Initializes the NQS object by allocating necessary resources and setting a random state.
*
* This function performs two main tasks:
* 1. Allocates the necessary resources for the NQS object.
* 2. Sets the initial state of the NQS object to a random state.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type used for computations.
* @tparam _stateType The type representing the state of the system.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::init()
{
    this->allocate();
    // this->setRandomState();
}
NQS_INST_CMB_ALL(init, void, ());

// ##########################################################################################################################################

/**
* @brief Saves the history of the Neural Quantum State (NQS) training and testing process.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type for the energy values.
* @tparam _stateType State type.
* @param _dir Directory where the history files will be saved.
* @param _EN_TRAIN Column vector of training energy values.
* @param _EN_TESTS Column vector of testing energy values.
* @param _EN_STD Column vector of standard deviations of training energy values.
* @param _EN_TESTS_STD Column vector of standard deviations of testing energy values.
* @param _betas Column vector of beta values used for training excited states.
* @param _meansNQS Column vector to store the means of the NQS states.
* @param _stdsNQS Column vector to store the standard deviations of the NQS states.
* @param i Index of the current state.
* @param _append Boolean flag indicating whether to append to the file.
* @param _name Name of the history file. If empty, defaults to "history.h5".
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::save_history(const std::string& _dir, 							
                                                        const arma::Col<_T>& _EN_TRAIN,				// training
                                                        const arma::Col<_T>& _EN_TESTS,				// test
                                                        const arma::Col<_T>& _EN_STD,				// standard deviations - training
                                                        const arma::Col<_T>& _EN_TESTS_STD,			// standard deviations - test
                                                        const arma::Col<double>& _betas,			// betas - for the training of the excited states
                                                        arma::Col<_T>& _meansNQS,					// means of the NQS states
                                                        arma::Col<_T>& _stdsNQS,					// standard deviations of the NQS states
                                                        const int i,								// the state number
                                                        const bool _append,                         // append to the file
                                                        const std::string& _name)
{
    auto _EN_r      = algebra::cast<double>(_EN_TRAIN);
    auto _EN_rt     = algebra::cast<double>(_EN_TESTS);
    auto _EN_std_r  = algebra::cast<double>(_EN_STD);
    auto _EN_std_rt = algebra::cast<double>(_EN_TESTS_STD);

    // calculate means
    _meansNQS(i) 	= arma::mean(_EN_TESTS);
    _stdsNQS(i) 	= arma::stddev(_EN_TESTS);

    LOGINFOG("Found the NQS state(" + STR(i) + ") to be E=" + STRPS(_meansNQS(i), 6) + " +- " + STRPS(_stdsNQS(i) / 2.0, 6), LOG_TYPES::TRACE, 2);

    // save the results
    const std::string history_file = _name.empty() ? "history.h5" : _name;
    saveAlgebraic(_dir, history_file, _EN_r, "train/history/" + STR(i), _append);
    saveAlgebraic(_dir, history_file, _EN_rt, "collect/history/" + STR(i), true);
    saveAlgebraic(_dir, history_file, _EN_std_r, "train/std/" + STR(i), true);
    saveAlgebraic(_dir, history_file, _EN_std_rt, "collect/std/" + STR(i), true);
    saveAlgebraic(_dir, history_file, _betas, "betas", true);

    // save imaginary part if _T is complex
    if constexpr (std::is_same_v<_T, cpx>)
    {
        arma::vec _EN_i      = arma::imag(_EN_TRAIN);
        arma::vec _EN_it     = arma::imag(_EN_TESTS);
        arma::vec _EN_std_i  = arma::imag(_EN_STD);
        arma::vec _EN_std_it = arma::imag(_EN_TESTS_STD);

        saveAlgebraic(_dir, history_file, _EN_i, "train/history/im/" + STR(i), true);
        saveAlgebraic(_dir, history_file, _EN_it, "collect/history/im/" + STR(i), true);
        saveAlgebraic(_dir, history_file, _EN_std_i, "train/std/im/" + STR(i), true);
        saveAlgebraic(_dir, history_file, _EN_std_it, "collect/std/im/" + STR(i), true);
    }
}

#define NQS_INSTANTIATE_SAVE_HISTORY(Ht, T, stateType) \
template void NQS<2u, Ht, T, stateType>::save_history(const std::string&, const arma::Col<T>&, const arma::Col<T>&, const arma::Col<T>&, const arma::Col<T>&, const arma::Col<double>&, arma::Col<T>&, arma::Col<T>&, const int, const bool, const std::string&);   \
template void NQS<3u, Ht, T, stateType>::save_history(const std::string&, const arma::Col<T>&, const arma::Col<T>&, const arma::Col<T>&, const arma::Col<T>&, const arma::Col<double>&, arma::Col<T>&, arma::Col<T>&, const int, const bool, const std::string&);   \
template void NQS<4u, Ht, T, stateType>::save_history(const std::string&, const arma::Col<T>&, const arma::Col<T>&, const arma::Col<T>&, const arma::Col<T>&, const arma::Col<double>&, arma::Col<T>&, arma::Col<T>&, const int, const bool, const std::string&);

NQS_INSTANTIATE_SAVE_HISTORY(double, double, double)
NQS_INSTANTIATE_SAVE_HISTORY(cpx, cpx, double)
NQS_INSTANTIATE_SAVE_HISTORY(cpx, double, double)
NQS_INSTANTIATE_SAVE_HISTORY(double, cpx, double)

// #################################################################################################≠≠≠≠≠≠#########################################