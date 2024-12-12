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
    this->derivativesReset(_n); 				// reset the derivatives
	this->lower_states_.setDerivContSize(_n);	// set the size of the containers for the lower states
}
// template instantiation of the function above
NQS_INST_CMB_ALL(reset, void, (size_t));

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
// template instantiation of the function above
NQS_INST_CMB_ALL(swapConfig, void, (MC_t_p));

// ##########################################################################################################################################

/**
* @brief Resets the derivatives for the Neural Quantum State (NQS) object.
*
* This function initializes the derivatives, centered derivatives, and 
* centered Hamiltonian derivatives to zero matrices with the specified 
* number of blocks.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type used for calculations.
* @tparam _stateType The type of the state.
* @param nBlocks The number of blocks to initialize the derivatives with.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::derivativesReset(size_t nBlocks)
{
    if (nBlocks != this->derivatives_.n_rows)
    {
        this->derivatives_          = NQSW(nBlocks, this->info_p_.fullSize_, arma::fill::zeros);
        this->derivativesCentered_  = this->derivatives_; 
        this->derivativesCenteredH_ = this->derivatives_.t();  
    }
}
// template instantiation of the function above
NQS_INST_CMB_ALL(derivativesReset, void, (size_t));

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
    : info_p_(_n.info_p_), H_(_n.H_), nFlip_(_n.nFlip_), flipPlaces_(_n.flipPlaces_), flipVals_(_n.flipVals_)
{
    this->accepted_ 			= _n.accepted_;
    this->total_ 				= _n.total_;
    this->info_ 				= _n.info_;
    this->ran_					= _n.ran_;
    this->pBar_					= _n.pBar_;
    // initialize the information 
    this->info_p_  				= _n.info_p_;
    this->lower_states_ 		= _n.lower_states_;
    // copy the weights
    this->derivatives_ 			= _n.derivatives_;
    this->derivativesMean_ 		= _n.derivativesMean_;
    this->derivativesCentered_ 	= _n.derivativesCentered_;
    this->derivativesCenteredH_ = _n.derivativesCenteredH_;
    this->dF_ 					= _n.dF_;
    this->F_ 					= _n.F_;
    // this->init();
#ifdef NQS_NOT_OMP_MT
    this->initThreads(_n.threads_.threadNum_);
#endif
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
    : info_p_(std::move(_n.info_p_)), H_(std::move(_n.H_)), nFlip_(_n.nFlip_), flipPlaces_(std::move(_n.flipPlaces_)), flipVals_(std::move(_n.flipVals_))
{
    this->accepted_ 			= _n.accepted_;
    this->total_ 				= _n.total_;
    this->info_ 				= _n.info_;
    this->pBar_					= _n.pBar_;
    this->ran_					= _n.ran_;
    // initialize the information
    this->info_p_ 				= std::move(_n.info_p_);
    this->lower_states_ 		= std::move(_n.lower_states_);
    // copy the weights
    this->derivatives_ 			= std::move(_n.derivatives_);
    this->derivativesMean_ 		= std::move(_n.derivativesMean_);
    this->derivativesCentered_ 	= std::move(_n.derivativesCentered_);
    this->derivativesCenteredH_ = std::move(_n.derivativesCenteredH_);
    this->dF_ 					= std::move(_n.dF_);
    this->F_ 					= std::move(_n.F_);
    // this->init();
#ifdef NQS_NOT_OMP_MT
    this->initThreads(_n.threads_.threadNum_);
#endif
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
        this->H_ 					= _n.H_;
        this->info_ 				= _n.info_;
        this->pBar_ 				= _n.pBar_;
        this->ran_ 				= _n.ran_;
        this->nFlip_ 				= _n.nFlip_;
        this->flipPlaces_ 			= _n.flipPlaces_;
        this->flipVals_ 			= _n.flipVals_;
        // initialize the information
        this->info_p_ 				= _n.info_p_;
        this->lower_states_ 		= _n.lower_states_;
        // copy the weights
        this->derivatives_ 			= _n.derivatives_;
        this->derivativesMean_ 		= _n.derivativesMean_;
        this->derivativesCentered_ 	= _n.derivativesCentered_;
        this->derivativesCenteredH_ = _n.derivativesCenteredH_;
        this->dF_ 					= _n.dF_;
        this->F_ 					= _n.F_;
        // this->init();
#ifdef NQS_NOT_OMP_MT
        this->initThreads(_n.threads_.threadNum_);
#endif
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
        this->H_ 					= _n.H_;
        this->info_ 				= _n.info_;
        this->pBar_ 				= _n.pBar_;
        this->ran_ 				    = _n.ran_;
        this->nFlip_ 				= _n.nFlip_;
        this->flipPlaces_ 			= _n.flipPlaces_;
        this->flipVals_ 			= _n.flipVals_;
        // initialize the information
        this->info_p_ 				= std::move(_n.info_p_);
        this->lower_states_ 		= std::move(_n.lower_states_);
        // copy the weights
        this->derivatives_ 			= std::move(_n.derivatives_);
        this->derivativesMean_ 		= std::move(_n.derivativesMean_);
        this->derivativesCentered_ 	= std::move(_n.derivativesCentered_);
        this->derivativesCenteredH_ = std::move(_n.derivativesCenteredH_);
        this->dF_ 					= std::move(_n.dF_);
        this->F_ 					= std::move(_n.F_);
        // this->init();
#ifdef NQS_NOT_OMP_MT
        this->initThreads(_n.threads_.threadNum_);
#endif
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

/**
* @brief Constructs a Neural Quantum State (NQS) object.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Hamiltonian type.
* @tparam _T Data type.
* @tparam _stateType State type.
* 
* @param _H Pointer to the Hamiltonian object.
* @param _lr Learning rate.
* @param _threadNum Number of threads to use.
* @param _nParticles Number of particles. If negative or spin modes equal to 2, it will be set to half-filling.
* @param _lower Pointer to the lower state object.
* @param _beta Vector of beta values.
* 
* Initializes various member functions and variables, including probability ratio functions, lower states, 
* learning rate, number of visible units, number of sites, number of particles, Hilbert space size, 
* and random number generator. Optionally initializes threads if multi-threading is enabled.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
NQS<_spinModes, _Ht, _T, _stateType>::NQS(NQS<_spinModes, _Ht, _T, _stateType>::Hamil_t_p _H, 
													double _lr, 
													uint _threadNum, 
													int _nParticles,
													const NQSLS_p& _lower, 
													const std::vector<double>& _beta)
	: H_(_H)
{	
	const size_t _Ns			= 			_H->getNs();	
	this->pRatioFunc_			= 			[&](const Config_t& _v)         { return this->pRatio(_v); };
	this->pKernelFunc_			= 			[&](int_ini_t fP, dbl_ini_t fV) { return this->pRatio(fP, fV); };
	this->logPKernelFunc_		= 			[&](int_ini_t fP, dbl_ini_t fV) { return this->logPRatio(fP, fV); };

	this->lower_states_			= 			NQS_lower_t<_spinModes, _Ht, _T, _stateType>(_Ns, _lower, _beta, this);
	this->lower_states_.exc_ratio_ = 		[&](const Config_t& _v)         { return this->pRatio(_v); };
#ifdef NQS_LOWER_RATIO_LOGDIFF
	this->lower_states_.exc_ansatz_ = 		[&](const Config_t& _v)         { return this->ansatzlog(_v); };
#else
	this->lower_states_.exc_ansatz_ = 		[&](const Config_t& _v)         { return this->ansatz(_v); };
#endif
	this->info_p_.lr_			= 			_lr;

	// set the number of particles
	// set the visible layer (for hardcore-bosons we have the same number as sites but fermions introduce twice the complication)
	this->info_p_.nVis_ 		= 			_Ns * (this->spinModes_ / 2);
	this->info_p_.nSites_		=			_Ns;

	// make it half filling if necessary
	this->info_p_.nParticles_	=			(_nParticles < 0 || this->spinModes_ == 2) ? this->info_p_.nSites_ : (uint)_nParticles;
	this->info_p_.Nh_			=			_H->getHilbertSize();           // check the Hilbert space
	this->ran_					=			&_H->ran_;                      // set the random number generator
#ifdef NQS_NOT_OMP_MT
	this->initThreads(_threadNum);
#endif
	LOGINFO("Constructed the general NQS class", LOG_TYPES::TRACE, 2);
};

// template instantiation of the function above
template NQS<2u, double, double, double>::NQS(NQS<2u, double, double, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<3u, double, double, double>::NQS(NQS<3u, double, double, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<4u, double, double, double>::NQS(NQS<4u, double, double, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<2u, cpx, cpx, double>::NQS(NQS<2u, cpx, cpx, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<3u, cpx, cpx, double>::NQS(NQS<3u, cpx, cpx, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<4u, cpx, cpx, double>::NQS(NQS<4u, cpx, cpx, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<2u, cpx, double, double>::NQS(NQS<2u, cpx, double, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<3u, cpx, double, double>::NQS(NQS<3u, cpx, double, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<4u, cpx, double, double>::NQS(NQS<4u, cpx, double, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<2u, double, cpx, double>::NQS(NQS<2u, double, cpx, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<3u, double, cpx, double>::NQS(NQS<3u, double, cpx, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);
template NQS<4u, double, cpx, double>::NQS(NQS<4u, double, cpx, double>::Hamil_t_p, double, uint, int, const NQSLS_p&, const std::vector<double>&);

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

    // don't delete the random number generator as it is shared with the Hamiltonian
    
    // ######################################################################################################################################
    if (this->pBar_ != nullptr) {
        delete this->pBar_;
        this->pBar_ = nullptr;
    }

    // ######################################################################################################################################
    if (this->precond_ != nullptr) {
        delete this->precond_;
        this->precond_ = nullptr;
    }

    if (this->solver_ != nullptr) {
        delete this->solver_;
        this->solver_ = nullptr;
    }
    // ######################################################################################################################################
}
// template instantiation of the function above
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
    this->curVec_	= _st;
#ifndef NQS_USE_VEC_ONLY
    this->curState_ = BASE_TO_INT<u64>(_st, this->discVal_);
#endif
}
// template instantiation of the function above
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
}
// template instantiation of the function above
NQS_INST_CMB_ALL(setState, void, (u64));

// ##########################################################################################################################################

/*
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
// template instantiation of the function above
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
	auto _isSaved = saveAlgebraic(_path, _file, this->Weights_, "weights/" + STRP(this->beta_, 5));	// save the weights to a given path
	if (!_isSaved && (_file != "weights.h5"))													    // if not saved properly
	{
		LOGINFO("Couldn't save the weights to the given path.", LOG_TYPES::ERROR, 3);
		LOGINFO("Saving to default... ", LOG_TYPES::ERROR, 3);
		return this->saveWeights(_path, "weights.h5");
	}
	return _isSaved;
}
// template instantiation of the function above
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
	return loadAlgebraic(_path, _file, this->Weights_, "weights/" + STRP(this->beta_, 5));
}
// template instantiation of the function above
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
// template instantiation of the function above
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
// template instantiation of the function above
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
// template instantiation of the function above
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
#ifdef NQS_USE_OMP
		omp_set_num_threads(this->threadNum_);   
#else
		this->threads_.threads_.reserve(this->threads_.threadNum_);
		this->threads_.kernels_	        =   v_1d<CondVarKernel<_T>>(this->threads_.threadNum_);
		
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
// template instantiation of the function above
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
// template instantiation of the function above
NQS_INST_CMB_ALL(init, void, ());

// ##########################################################################################################################################

