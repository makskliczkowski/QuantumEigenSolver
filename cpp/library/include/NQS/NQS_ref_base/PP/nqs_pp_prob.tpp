#include "./nqs_pp_update.tpp"

namespace NQS_NS
{
    // ##########################################################################################################################################

    // ######################################################### P R O B A B I L I T Y ##########################################################

    // ##########################################################################################################################################

    // %%%%%%%%%%%%%%%%%%% S I N G L E   F L I P %%%%%%%%%%%%%%%%%%%

    /**
    * @brief Computes the logarithm of the probability ratio for a given state update.
    *
    * This function updates the Pfaffian candidate matrix and its corresponding value,
    * then calculates the logarithm of the probability ratio based on the updated Pfaffian.
    *
    * @tparam _spinModes Number of spin modes.
    * @tparam _Ht Hamiltonian type.
    * @tparam _T Return type.
    * @tparam _stateType State type.
    * @tparam _CorrState Correlation state type.
    * @param fP Index of the state to be updated.
    * @param fV New value for the state update.
    * @return The logarithm of the probability ratio after the state update.
    */
    template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
    requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
    inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::logPRatio(uint fP, float fV)
    {
        // use the columns of the matrix to seek for the updated values - this enables to calculate the Pfaffian and others
        this->setX_changed(fP, fV);                                 // update pfaffian candidate matrix and its corresponding value
        // ! this does not modify the X matrix, only the X_upd_ matrix [0]. Also, this is thread local! - static
        // ! TODO: check if one can do this with thread local storage

        // set the new Pfaffian candidate with the Cayley's formula
        this->pp_weights_.pfaffianNew_      = getUpdatedPfaffian(fP);  
        this->pp_weights_.pfaffianNewLog_   = std::log(this->pp_weights_.pfaffianNew_);

        // calculate the probability ratio from the Pfaffian and the correlation part
        auto left   = NQS_ref_t::logPRatio(fP, fV);
        auto right  = this->pp_weights_.pfaffianNewLog_ - this->pp_weights_.pfaffianLog_;
        return left + right;
    }

    // ##########################################################################################################################################

    // %%%%%%%%%%%%%%%% M U L T I P L E   F L I P S %%%%%%%%%%%%%%%%

    /**
    * @brief Calculates the probability ratio whenever we use multiple flips.
    * Uses the flips stored within the NQS class (flipPlaces_, flipVals_)
    * If multiple flips are used, one should use calculate the Xinv and pfaffian from scratch
    * @param nFlips number of flips to be used
    * @returns probability ratio for a given ansatz based on the current state
    */
    template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
    requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
    
    inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::logPRatio(uint nFlips)
    {
        if (nFlips == 1)
            return this->logPRatio(this->flipPlaces_[0], this->flipVals_[0]);
        else if (nFlips == 2)
            return this->logPRatio(this->flipPlaces_[0], this->flipPlaces_[1], this->flipVals_[0], this->flipVals_[1]);
        else if (nFlips == 0)
            return 1.0;    

        // update pfaffian candidate matrix and its corresponding value - does it really update the X matrix correctly?
        // !TODO check if this is the correct way to update the X matrix - this is thread local! - static
        for (auto i = 0; i < nFlips; ++i)
            this->setX_changed(this->flipPlaces_[i], this->flipVals_[i]);

        // update the Pfaffian candidate
    #ifdef NQS_REF_PP_USE_PFAFFIAN_UPDATE
        for (auto i = 0; i < nFlips; ++i)
            this->pp_weights_.pfaffianNew_ = getUpdatedPfaffian(this->flipPlaces_[i], this->pp_weights_.pfaffianNew_);
    #else
        // update the matrix X
        auto _state = NQS_STATE;
        for (auto i = 0; i < nFlips; ++i)
            _state = Binary::flip(_state, this->flipPlaces_[i]);
        this->pp_weights_.pfaffianNew_ = algebra::Pfaffian::pfaffian(this->calculateX(_state));
    #endif
        this->pp_weights_.pfaffianLog_ = std::log(this->pp_weights_.pfaffianNew_);
        // return the probability ratio
        const auto left     = NQS_ref_t::logPRatio(nFlips);
        const auto right    = this->pp_weights_.pfaffianNewLog_ - this->pp_weights_.pfaffianLog_;
        return left + right;
    }

    // ##########################################################################################################################################

    // %%%%%%%%%%%%%%%%% U S I N G   V E C T O R S %%%%%%%%%%%%%%%%%

    /**
    * @brief Computes the logarithm of the probability ratio for a given state update using vectors.
    *
    * This function updates the Pfaffian candidate matrix and its corresponding value,
    * then calculates the logarithm of the probability ratio based on the updated Pfaffian.
    *
    * @tparam _spinModes Number of spin modes.
    * @tparam _Ht Hamiltonian type.
    * @tparam _T Return type.
    * @tparam _stateType State type.
    * @param _v1 Initial state configuration.
    * @param _v2 Updated state configuration.
    * @return The logarithm of the probability ratio after the state update.
    */
    template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
    requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
    
    inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::logPRatio(Config_cr_t _v1, Config_cr_t _v2)
    {
        auto left   = NQS_ref_t::logPRatio(_v1, _v2);
        auto right  = this->getPfaffianLog(_v2) - this->getPfaffianLog(_v1);
        return left + right;
    }

    // ##########################################################################################################################################

    // %%%%%%%%%%%%% U S I N G   I N I T I A L I Z E R %%%%%%%%%%%%%

    /**
    * @brief Calculates the probability ratio whenever we use multiple flips ariving from the external vectors.
    * Uses the flips stored within the NQS class (fP, fV)
    * @param fP flip places to be used
    * @param fV flip values to be used
    * @returns probability ratio for a given ansatz based on the current state
    */
    template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
    
    inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::logPRatio(std::initializer_list<int> fP, std::initializer_list<double> fV)
    {
    #ifdef NQS_NOT_OMP_MT
        this->setX_changed(fP, fV, this->pp_weights_.X_upd_);   // update the X matrix (obviously just elements of the X_upd_ matrix)

        // check if the _fP.size() is the same as the _X_upd_.size() and resize if necessary by updating the X matrix
    #ifdef NQS_REF_PP_USE_PFAFFIAN_UPDATE
        this->pp_weights_.pfaffianNew_ = this->pp_weights_.pfaffian_;

        // update the Pfaffian candidate 
        // !TODO: check if the Cayley's formula is faster - this probably does not work
        // !TODO: check if this is the correct way to update the Pfaffian
        for(const auto& _row: fP)
            this->pp_weights_.pfaffianNew_ = this->getUpdatedPfaffian(_row, this->pp_weights_.pfaffianNew_);
    #else
        auto _state         = NQS_STATE;
        // flip the values for the spins
        for (const auto& _row: fP)
            _state = Binary::flip(_state, _row);
        auto _Xnew          = this->calculateX(_state);
        auto _pfaffian		= algebra::Pfaffian::pfaffian(_Xnew);
    #endif
    #else
        this->XTmp_			= this->X_;
        // update
        this->updFPP_F(fP, fV, this->XTmp_);
        auto _pfaffian		= this->getPfaffian(this->XTmp_);
    #endif
        this->pp_weights_.pfaffianNewLog_ = std::log(this->pp_weights_.pfaffianNew_);
        // return the probability ratio
        auto _left          = NQS_ref_t::logPRatio(fP, fV);
        auto _right         = this->pp_weights_.pfaffianNewLog_ - this->pp_weights_.pfaffianLog_;
        return _left + _right;
    }

    // ##########################################################################################################################################

    // %%%%%% DOUBLE SPINS %%%%%%

    template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
    requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
    inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::logPRatio(uint fP1, uint fP2, float fV1, float fV2)
    {
        // update the Pfaffian candidate matrix and its corresponding value
        this->setX_changed(fP1, fV1);
        this->setX_changed(fP2, fV2);

        // update the Pfaffian candidate
        this->pp_weights_.pfaffianNew_ = getUpdatedPfaffian(fP1, this->pp_weights_.pfaffianNew_);
        this->pp_weights_.pfaffianNew_ = getUpdatedPfaffian(fP2, this->pp_weights_.pfaffianNew_);

        // calculate the probability ratio from the Pfaffian and the correlation part
        auto left   = NQS_ref_t::logPRatio(fP1, fV1, fP2, fV2);
        auto right  = this->pp_weights_.pfaffianNewLog_ - this->pp_weights_.pfaffianLog_;
        return left + right;
    }

    // ##########################################################################################################################################

};