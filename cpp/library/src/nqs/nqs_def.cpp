#include "../../include/NQS/nqs_final.hpp"
#include <cstddef>

namespace NQS_NS
{

    // ##########################################################################################################################################

    // REGULARIZATION

    // ##########################################################################################################################################

    /**
    * @brief Computes the regularization term for a given epoch and metric.
    *
    * This function returns the regularization term based on the provided epoch and metric.
    * If the pointer `p_` is not null, it calls the function pointed to by `p_` with the
    * given epoch and metric. Otherwise, it returns the default regularization value `reg_`.
    *
    * @param epoch The current epoch for which the regularization term is computed.
    * @param _metric The metric value used in the computation of the regularization term.
    * @return The computed regularization term.
    */
    const double NQS_reg_t::reg(size_t epoch, double _metric) const
    {
        return this->p_ ? (*this->p_)(epoch, _metric) : this->reg_;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
    * @brief Saves the NQS regularization data to a file.
    * 
    * This function saves the regularization data of the NQS (Neural Quantum State) to a specified file.
    * The file can be in HDF5 format if the filename ends with ".h5".
    * 
    * @param _dir The directory where the file will be saved.
    * @param _name The name of the file to save the data to.
    * @param i An integer parameter (purpose not specified in the provided code).
    * @param _namepar A string parameter to be appended to the regularization data name.
    * @param _append A boolean flag indicating whether to append to the file if it exists.
    */
    void NQS_reg_t::save(const std::string& _dir, 
                        const std::string& _name, 
                        int i, 
                        const std::string& _namepar,
                        bool _append) const
    {
        if (_name.ends_with(".h5")) 
            saveAlgebraic(_dir, _name,
                    arma::vec(this->p_ ? this->p_->hist() : v_1d<double>({ this->reg_ })), 
                    _namepar + "regularization", _append);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    NQS_reg_t::NQS_reg_t(const NQS_reg_t& other)
            : reg_(other.reg_),
            p_(other.p_ ? other.p_->clone() : nullptr)
    {}
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    NQS_reg_t::NQS_reg_t(NQS_reg_t&& other) noexcept
        : reg_(std::exchange(other.reg_, 1e-7)),
        p_(std::move(other.p_))
    {}
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    NQS_reg_t& NQS_reg_t::operator=(NQS_reg_t&& other) noexcept
    {
        this->reg_ = std::exchange(other.reg_, 1e-7);
        this->p_ = std::move(other.p_);
        return *this;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    NQS_reg_t& NQS_reg_t::operator=(const NQS_reg_t& other) 
    {
        this->reg_ = other.reg_;
        if (other.p_) 
            this->p_ = other.p_->clone();
        return *this;
    }

    // ##########################################################################################################################################

    // SCHEDULER

    // ##########################################################################################################################################

    void NQS_scheduler_t::save(const std::string& _dir, 
                                const std::string& _name, 
                                int i, 
                                const std::string& _namepar,
                                bool _append) const
    {
        if (_name.ends_with(".h5")) 
            saveAlgebraic(_dir, _name,
                    arma::vec(this->p_ ? this->p_->hist() : v_1d<double>({ this->lr_ })), 
                    _namepar + "learning_rate", _append);
    }

    NQS_scheduler_t::NQS_scheduler_t(const NQS_scheduler_t& other)
        : best_(other.best_),
        lr_(other.lr_),
        p_(other.p_ ? other.p_->clone() : nullptr)
    {}

    NQS_scheduler_t::NQS_scheduler_t(NQS_scheduler_t&& other) noexcept
        : best_(std::exchange(other.best_, 0.0)),
        lr_(std::exchange(other.lr_, 1e-3)),
        p_(std::move(other.p_))
    {}

    NQS_scheduler_t& NQS_scheduler_t::operator=(const NQS_scheduler_t& other)
    {
        if (this != &other)
        {
            best_   = other.best_;
            lr_     = other.lr_;
            p_      = other.p_ ? other.p_->clone() : nullptr;
        }
        return *this;
    }

    NQS_scheduler_t& NQS_scheduler_t::operator=(NQS_scheduler_t&& other) noexcept
    {
        if (this != &other)
        {
            best_   = std::exchange(other.best_, 0.0);
            lr_     = std::exchange(other.lr_, 1e-3);
            p_      = std::move(other.p_);
        }
        return *this;
    }

    // ##########################################################################################################################################

    // ##########################################################################################################################################

    /**
    * @brief Determines whether the training process should stop based on the given epoch and metric.
    * 
    * @tparam _T The type of the metric.
    * @param epoch The current epoch of the training process.
    * @param _metric The metric value used to decide whether to stop the training.
    * @return true if the training process should stop, false otherwise.
    */
    template <typename _T>
    bool NQS_scheduler_t::stop(size_t epoch, _T _metric)
    {
        if (this->p_)
            return this->p_->stop(epoch, algebra::real(_metric)); 
        return false;
    }
    // template instantiation of the function above
    template bool NQS_scheduler_t::stop(size_t, double);
    template bool NQS_scheduler_t::stop(size_t, cpx);

    // ##########################################################################################################################################

    // INFO

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
        if (_name.ends_with(".h5")) 
        {
            LOGINFO("Saving the NQS information to the file: " + _name, LOG_TYPES::INFO, 2);
            const bool _append          = std::filesystem::exists(_dir + _name);
            const std::string _namePar  = std::format("parameters/{}/", STR(i));

            // Save pseudoinverse information
    #ifdef NQS_USESR_MAT_USED
            saveAlgebraic(_dir, _name, arma::vec({ this->pinv_ }), _namePar + "pinv", _append);
    #else 
            saveAlgebraic(_dir, _name, arma::vec({ 0.0 }), _namePar + "pinv", _append);
    #endif
            // Save regularization history
            this->reg_.save(_dir, _name, i, _namePar, _append);

            // Save learning rate history
            this->sched_.save(_dir, _name, i, _namePar, _append);

            // Save the number of visible units
            saveAlgebraic(_dir, _name, arma::vec({ double(this->nVis_) }), _namePar + "visible_units", true);

            // Save the full size of the NQS
            saveAlgebraic(_dir, _name, arma::vec({ double(this->fullSize_) }), _namePar + "full_size", true);
        }
    }

    // ##########################################################################################################################################

    /**
    * @brief Copy constructor for NQS_info_t.
    *
    * This constructor creates a new instance of NQS_info_t by copying the data
    * from another instance.
    *
    * @param other The instance of NQS_info_t to copy from.
    */
    NQS_info_t::NQS_info_t(const NQS_info_t& other)
            : solver_(other.solver_),
            reg_(other.reg_),
            sched_(other.sched_),
            nVis_(other.nVis_),
            nSites_(other.nSites_),
            nSitesSquared_(other.nSitesSquared_),
            fullSize_(other.fullSize_),
            Nh_(other.Nh_),
            nParticles_(other.nParticles_),
            nParticlesSquared_(other.nParticlesSquared_),
            conservesParticles_(other.conservesParticles_)
    {}

    // ##########################################################################################################################################

    /**
    * @brief Move constructor for NQS_info_t.

    * This constructor initializes an NQS_info_t object by moving data from another NQS_info_t object.
    * It uses std::exchange to transfer ownership of resources and reset the original object's members.
    *
    * @param other The NQS_info_t object to move from.
    */
    NQS_info_t::NQS_info_t(NQS_info_t&& other) noexcept
        : solver_(std::move(other.solver_)),
        reg_(std::move(other.reg_)),
        sched_(std::move(other.sched_)),
        nVis_(std::exchange(other.nVis_, 1)),
        nSites_(std::exchange(other.nSites_, 1)),
        nSitesSquared_(std::exchange(other.nSitesSquared_, 1)),
        fullSize_(std::exchange(other.fullSize_, 1)),
        Nh_(std::exchange(other.Nh_, 1)),
        nParticles_(std::exchange(other.nParticles_, 1)),
        nParticlesSquared_(std::exchange(other.nParticlesSquared_, 1)),
        conservesParticles_(std::exchange(other.conservesParticles_, true))
    {
    }

    // ##########################################################################################################################################

    /**
    * @brief Assignment operator for NQS_info_t.
    *
    * This operator performs a deep copy of the given NQS_info_t object into the current object.
    * It checks for self-assignment and then copies each member variable from the source object.
    * If the source object contains pointers to schedulers, it clones these schedulers to ensure
    * that the current object has its own copies.
    *
    * @param other The NQS_info_t object to be copied.
    * @return A reference to the current NQS_info_t object.
    */
    NQS_info_t& NQS_info_t::operator=(const NQS_info_t& other) 
    {
        if (this != &other) 
        {
            solver_             = other.solver_;
            reg_                = other.reg_;
            sched_              = other.sched_;
            nVis_               = other.nVis_;
            nSites_             = other.nSites_;
            nSitesSquared_      = other.nSitesSquared_;
            fullSize_           = other.fullSize_;
            Nh_                 = other.Nh_;
            nParticles_         = other.nParticles_;
            nParticlesSquared_  = other.nParticlesSquared_;
            conservesParticles_ = other.conservesParticles_;
        }
        return *this;
    }

    // ##########################################################################################################################################

    /**
    * @brief Move assignment operator for NQS_info_t.
    *
    * This operator moves the contents of another NQS_info_t object to this object.
    * It uses std::exchange to transfer the values of the member variables from the
    * source object to the destination object, ensuring that the source object is left
    * in a valid but unspecified state.
    *
    * @param other The NQS_info_t object to move from.
    * @return A reference to this NQS_info_t object.
    */
    NQS_info_t& NQS_info_t::operator=(NQS_info_t&& other) noexcept
    {
        if (this != &other) 
        {
            solver_             = std::move(other.solver_);
            reg_                = std::move(other.reg_);
            sched_              = std::move(other.sched_);
            nVis_               = std::exchange(other.nVis_, 1);
            nSites_             = std::exchange(other.nSites_, 1);
            nSitesSquared_      = std::exchange(other.nSitesSquared_, 1);
            fullSize_           = std::exchange(other.fullSize_, 1);
            Nh_                 = std::exchange(other.Nh_, 1);
            nParticles_         = std::exchange(other.nParticles_, 1);
            nParticlesSquared_  = std::exchange(other.nParticlesSquared_, 1);
            conservesParticles_ = std::exchange(other.conservesParticles_, true);
        }
        return *this;
    }

    // ##########################################################################################################################################

    // DERIVATIVES

    // ##########################################################################################################################################

    // class template instantiation
    template struct NQS_deriv<double, double>;
    template struct NQS_deriv<cpx, double>;
    template struct NQS_deriv<double, cpx>;
    template struct NQS_deriv<cpx, cpx>;

    // ##########################################################################################################################################

    template <typename _stateType, typename _type>
    NQS_deriv<_stateType, _type>::NQS_deriv(size_t _fullSize, size_t _nBlocks)
    {
        this->derivatives.set_size(_nBlocks, _fullSize);
        this->derivativesCentered.set_size(_nBlocks, _fullSize);
        this->derivativesCenteredH.set_size(_fullSize, _nBlocks);
        this->energiesCentered.set_size(_nBlocks);
    }
    // template instantiation of the function above
    NQS_DERIV_INST_TYPES(, NQS_deriv, (size_t, size_t));
    // ##########################################################################################################################################

    /**
    * @brief Copy constructor for the NQS_deriv class.
    *
    * This constructor creates a new instance of the NQS_deriv class by copying the
    * data from another instance.
    *
    * @param other The instance of NQS_deriv to copy from.
    */
    template <typename _stateType, typename _type>
    NQS_deriv<_stateType, _type>::NQS_deriv(const NQS_deriv& other)
        : derivativesMean(other.derivativesMean),
        energiesCentered(other.energiesCentered),
        derivatives(other.derivatives),
        derivativesCentered(other.derivativesCentered),
        derivativesCenteredH(other.derivativesCenteredH)
    {}
    // template instantiation of the function above
    template NQS_deriv<double, double>::NQS_deriv(const NQS_deriv<double, double>&);
    template NQS_deriv<cpx, double>::NQS_deriv(const NQS_deriv<cpx, double>&);
    template NQS_deriv<double, cpx>::NQS_deriv(const NQS_deriv<double, cpx>&);
    template NQS_deriv<cpx, cpx>::NQS_deriv(const NQS_deriv<cpx, cpx>&);

    /**
    * @brief Move constructor for the NQS_deriv class.
    * 
    * This constructor initializes a new NQS_deriv object by transferring the resources
    * from another NQS_deriv object using move semantics. This is useful for efficiently
    * transferring ownership of resources without copying.
    * 
    * @tparam _stateType The type representing the state.
    * @tparam _type The type representing the value.
    * @param other The NQS_deriv object to move from.
    */
    template <typename _stateType, typename _type>
    NQS_deriv<_stateType, _type>::NQS_deriv(NQS_deriv&& other) noexcept
        : derivativesMean(std::move(other.derivativesMean)),
        energiesCentered(std::move(other.energiesCentered)),
        derivatives(std::move(other.derivatives)),
        derivativesCentered(std::move(other.derivativesCentered)),
        derivativesCenteredH(std::move(other.derivativesCenteredH))
    {}
    // template instantiation of the functions above
    template NQS_deriv<double, double>::NQS_deriv(NQS_deriv<double, double>&&);
    template NQS_deriv<cpx, double>::NQS_deriv(NQS_deriv<cpx, double>&&);
    template NQS_deriv<double, cpx>::NQS_deriv(NQS_deriv<double, cpx>&&);
    template NQS_deriv<cpx, cpx>::NQS_deriv(NQS_deriv<cpx, cpx>&&);

    /**
    * @brief Assignment operator for the NQS_deriv class.
    * 
    * This operator assigns the values from another NQS_deriv object to the current object.
    * It performs a deep copy of the member variables to ensure that the current object
    * has the same state as the other object.
    * 
    * @tparam _stateType The type representing the state.
    * @tparam _type The type representing the values in the derivatives.
    * @param other The other NQS_deriv object to copy from.
    * @return A reference to the current NQS_deriv object.
    */
    template <typename _stateType, typename _type>
    NQS_deriv<_stateType, _type>& NQS_deriv<_stateType, _type>::operator=(const NQS_deriv& other)
    {
        if (this != &other)
        {
            this->derivatives           = other.derivatives;
            this->derivativesMean       = other.derivativesMean;
            this->energiesCentered      = other.energiesCentered;
            this->derivativesCentered   = other.derivativesCentered;
            this->derivativesCenteredH  = other.derivativesCenteredH;
        }
        return *this;
    }
    // template instantiation of the function above
    template NQS_deriv<double, double>& NQS_deriv<double, double>::operator=(const NQS_deriv<double, double>&);
    template NQS_deriv<cpx, double>& NQS_deriv<cpx, double>::operator=(const NQS_deriv<cpx, double>&);
    template NQS_deriv<double, cpx>& NQS_deriv<double, cpx>::operator=(const NQS_deriv<double, cpx>&);
    template NQS_deriv<cpx, cpx>& NQS_deriv<cpx, cpx>::operator=(const NQS_deriv<cpx, cpx>&);

    /**
    * @brief Move assignment operator for NQS_deriv class.
    *
    * This operator moves the contents of the given NQS_deriv object to the current object.
    * It ensures that the current object is not the same as the given object before performing the move.
    *
    * @tparam _stateType The type representing the state.
    * @tparam _type The type representing the value.
    * @param other The NQS_deriv object to move from.
    * @return A reference to the current NQS_deriv object.
    */
    template <typename _stateType, typename _type>
    NQS_deriv<_stateType, _type>& NQS_deriv<_stateType, _type>::operator=(NQS_deriv&& other) noexcept
    {
        if (this != &other)
        {
            this->derivatives           = std::move(other.derivatives);
            this->derivativesMean       = std::move(other.derivativesMean);
            this->energiesCentered      = std::move(other.energiesCentered);
            this->derivativesCentered   = std::move(other.derivativesCentered);
            this->derivativesCenteredH  = std::move(other.derivativesCenteredH);
        }
        return *this;
    }
    // template instantiation of the function above
    template NQS_deriv<double, double>& NQS_deriv<double, double>::operator=(NQS_deriv<double, double>&&);
    template NQS_deriv<cpx, double>& NQS_deriv<cpx, double>::operator=(NQS_deriv<cpx, double>&&);
    template NQS_deriv<double, cpx>& NQS_deriv<double, cpx>::operator=(NQS_deriv<double, cpx>&&);
    template NQS_deriv<cpx, cpx>& NQS_deriv<cpx, cpx>::operator=(NQS_deriv<cpx, cpx>&&);

    // ##########################################################################################################################################

    /**
    * @brief Sets the centered derivatives and energies for the NQS_deriv object.
    * 
    * This function calculates the centered derivatives and energies based on the provided
    * energies, mean loss, and number of samples. The centered derivatives are computed
    * - using armadillo if NQS_USE_ARMA is defined
    * - using std::vector if NQS_USE_STDVEC is defined
    * 
    * @tparam _stateType The type representing the state.
    * @tparam _type The type representing the numerical values.
    * @tparam _CT The type of the energies container.
    * 
    * @param _energies A container holding the energy values.
    * @param _meanLoss The mean loss value.
    * @param _samples The number of samples.
    */
    template <typename _stateType, typename _type>
    template <typename _CT>
    void NQS_deriv<_stateType, _type>::set_centered(const _CT& _energies, _type _meanLoss, const _type _samples)
    {
    #ifdef NQS_USE_ARMA
        this->derivativesCentered 	            = this->derivatives.each_row() - this->derivativesMean;
        this->derivativesCenteredH 				= this->derivativesCentered.t();
    #else
    #endif
        this->energiesCentered					= (_energies - _meanLoss) / _samples;
    }
    // template instantiation of the function above
    #ifdef NQS_USE_ARMA
    template void NQS_deriv<double, cpx>::set_centered(const arma::Col<cpx>& _energies, cpx _meanLoss, const cpx _samples);
    template void NQS_deriv<cpx, double>::set_centered(const arma::Col<double>& _energies, double _meanLoss, const double _samples);
    template void NQS_deriv<cpx, cpx>::set_centered(const arma::Col<cpx>& _energies, cpx _meanLoss, const cpx _samples);
    template void NQS_deriv<double, double>::set_centered(const arma::Col<double>& _energies, double _meanLoss, const double _samples);
    #else
    #endif
    // ##########################################################################################################################################

    /**
    * @brief Computes the covariance vector for the gradient.
    *
    * This function calculates the covariance vector for the gradient F using the formula:
    * F = <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}>
    * where \Delta _k* represents the centered derivatives and E_{loc} represents the centered energies.
    *
    * @tparam _stateType The type representing the state.
    * @tparam _type The type representing the numerical values.
    * @return NQSB The covariance vector for the gradient.
    */
    template <typename _stateType, typename _type>
    inline NQS_deriv<_stateType, _type>::NQSB NQS_deriv<_stateType, _type>::getF() const
    {
        return this->derivativesCenteredH * this->energiesCentered;
    }
    // template instantiation of the function above
    #ifdef NQS_USE_ARMA
    template arma::Col<cpx> NQS_deriv<double, cpx>::getF() const;
    template arma::Col<double> NQS_deriv<cpx, double>::getF() const;
    template arma::Col<cpx> NQS_deriv<cpx, cpx>::getF() const;
    template arma::Col<double> NQS_deriv<double, double>::getF() const;
    #else
    #endif

    // ##########################################################################################################################################

    /**
    * @brief Computes the final derivatives for the Neural Quantum State (NQS) model.
    * 
    * This function calculates the covariance derivatives of the local energy with respect to the parameters of the NQS model.
    * It also computes the centered derivatives and appends the derivatives for the excited states.
    * 
    *   calculate the covariance derivatives <\Delta _k* E_{loc}> - <\Delta _k*><E_{loc}> 
    *   [+ sum_i ^{n-1} \beta _i <(Psi_W(i) / Psi_W - <Psi_W(i)/Psi>) \Delta _k*> <Psi _W/Psi_W(i)>] 
    *   - for the excited states, the derivatives are appended
    *   calculate the centered derivatives
    *
    * @tparam _stateType The type representing the state of the system.
    * @tparam _type The type used for numerical values (e.g., float, double).
    * @tparam _CT The type of the container holding the energies.
    * 
    * @param _energies A container holding the local energies.
    * @param _step The current step in the optimization process.
    * @param _meanLoss The mean loss value.
    * @param _samples The number of samples used in the calculation.
    */
    template <typename _stateType, typename _type>
    template <typename _CT>
    void NQS_deriv<_stateType, _type>::finalF(const _CT& _energies, int _step, _type _meanLoss, const _type _samples)
    {
    #ifdef NQS_USE_ARMA
        this->derivativesMean = arma::mean(this->derivatives, 0).as_row();
    #elif NQS_USE_STDVEC
    #else
    #endif
        this->set_centered(_energies, _meanLoss, _samples);
    }
    // template instantiation of the function above
    #ifdef NQS_USE_ARMA
    template void NQS_deriv<double, cpx>::finalF(const arma::Col<cpx>&, int, cpx, const cpx);
    template void NQS_deriv<cpx, double>::finalF(const arma::Col<double>&, int, double, const double);
    template void NQS_deriv<cpx, cpx>::finalF(const arma::Col<cpx>&, int, cpx, const cpx);
    template void NQS_deriv<double, double>::finalF(const arma::Col<double>&, int, double, const double);
    #else
    #endif

    // ##########################################################################################################################################

    /**
    * @brief Resets the NQS_deriv object with new sizes for derivatives and centered values.
    * 
    * This function reinitializes the derivatives, energiesCentered, derivativesCentered, 
    * and derivativesCenteredH matrices with the specified sizes. If the number of blocks 
    * (nBlocks) is different from the current number of rows in the derivatives matrix, 
    * the matrices are resized and filled with zeros.
    * 
    * @tparam _stateType The type representing the state.
    * @tparam _type The type used for numerical values.
    * @param fullSize The full size of the derivatives matrix.
    * @param nBlocks The number of blocks (rows) in the derivatives matrix.
    */
    template <typename _stateType, typename _type>
    void NQS_deriv<_stateType, _type>::reset(size_t fullSize, size_t nBlocks)
    {
        if (nBlocks != this->derivatives.n_rows)
        {
            this->derivatives            = NQSW(nBlocks, fullSize, arma::fill::zeros);
            this->energiesCentered       = NQS_COL_T(nBlocks, arma::fill::zeros);
            this->derivativesCentered    = this->derivatives; 
            this->derivativesCenteredH   = this->derivatives.t();  
        }
    }
    // template instantiation of the function above
    NQS_DERIV_INST_TYPES(void, reset, (size_t, size_t));

    // ##########################################################################################################################################

    /**
    * @brief Prints the shapes of various matrices and vectors used in the NQS_deriv class.
    * 
    * This function outputs the dimensions of the following members to the standard output:
    * - derivatives: The number of rows and columns.
    * - derivativesMean: The number of elements.
    * - energiesCentered: The number of elements.
    * - derivativesCentered: The number of rows and columns.
    * - derivativesCenteredH (Transposed): The number of rows and columns.
    */
    template <typename _stateType, typename _type>
    void NQS_deriv<_stateType, _type>::printState() const
    {
        std::cout << "Derivatives shape: " << derivatives.n_rows << " x " << derivatives.n_cols << std::endl;
        std::cout << "Derivatives Mean shape: " << derivativesMean.n_elem << std::endl;
        std::cout << "Energies Centered shape: " << energiesCentered.n_elem << std::endl;
        std::cout << "Derivatives Centered shape: " << derivativesCentered.n_rows << " x " << derivativesCentered.n_cols << std::endl;
        std::cout << "Derivatives Centered (Transposed) shape: " << derivativesCenteredH.n_rows << " x " << derivativesCenteredH.n_cols << std::endl;
    }
    // template instantiation of the function above
    NQS_DERIV_INST_TYPES(void, printState, () const);

};

// ##########################################################################################################################################