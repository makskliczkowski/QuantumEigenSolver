#include "../../include/NQS/nqs_final.hpp"

// ##########################################################################################################################################

// MODIFIED NQS BY OPERATOR

// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::setModifier(std::shared_ptr<Operators::OperatorComb<_T>> _op)
{
    this->a_mod_p_.modifier_ = _op;                                                                         // set the modifier operator
    this->a_mod_p_.modtype_  = _op->getNameS();                                                             // set the modifier type
    this->a_mod_p_.modified_ = true;                                                                        // set the modifier type
    LOGINFO("Set the modifier operator: " + this->a_mod_p_.modtype_, LOG_TYPES::INFO, 2);
    // *****************************************************************************************************
    this->a_mod_p_.logAMod_ = this->logAnsatzModifier(NQS_STATE);                                           // get the log of the modified states
    this->a_mod_p_.logTmpAMod_ = this->a_mod_p_.logAMod_;                                                   // set the temporary log of the modified states
    // *****************************************************************************************************
    this->pRatioFunc_        = [this](const Config_t& _v)          { return this->pRatioMod(_v); };         // change the function pointer
    this->pKernelFunc_       = [this](int_ini_t fP, dbl_ini_t fV)  { return this->pRatioMod(fP, fV); };     // change the function pointer  
    this->logPKernelFunc_    = [this](int_ini_t fP, dbl_ini_t fV)  { return this->logPRatioMod(fP, fV); };  // change the function pointer
    this->logPRatioFuncFlips_= [this](uint nFlips)                 { return this->logPRatioMod(nFlips); };
    this->lower_states_.exc_ratio_ = [this](const Config_t& _v)    { return this->pRatioMod(_v); };         // change the function pointer
    
}
// template instantiation of the function above
template void NQS<2u, double, double, double>::setModifier(std::shared_ptr<Operators::OperatorComb<double>>);
template void NQS<3u, double, double, double>::setModifier(std::shared_ptr<Operators::OperatorComb<double>>);
template void NQS<4u, double, double, double>::setModifier(std::shared_ptr<Operators::OperatorComb<double>>);
template void NQS<2u, cpx, cpx, double>::setModifier(std::shared_ptr<Operators::OperatorComb<cpx>>);
template void NQS<3u, cpx, cpx, double>::setModifier(std::shared_ptr<Operators::OperatorComb<cpx>>);
template void NQS<4u, cpx, cpx, double>::setModifier(std::shared_ptr<Operators::OperatorComb<cpx>>);
template void NQS<2u, cpx, double, double>::setModifier(std::shared_ptr<Operators::OperatorComb<double>>);
template void NQS<3u, cpx, double, double>::setModifier(std::shared_ptr<Operators::OperatorComb<double>>);
template void NQS<4u, cpx, double, double>::setModifier(std::shared_ptr<Operators::OperatorComb<double>>);
template void NQS<2u, double, cpx, double>::setModifier(std::shared_ptr<Operators::OperatorComb<cpx>>);
template void NQS<3u, double, cpx, double>::setModifier(std::shared_ptr<Operators::OperatorComb<cpx>>);
template void NQS<4u, double, cpx, double>::setModifier(std::shared_ptr<Operators::OperatorComb<cpx>>);
// #################################################################################################≠≠≠≠≠≠#########################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void NQS<_spinModes, _Ht, _T, _stateType>::unsetModifier()
{
    this->a_mod_p_.modifier_ = nullptr;                                                                     // set the modifier operator
    this->a_mod_p_.modtype_  = "";                                                                          // set the modifier type
    this->a_mod_p_.modified_ = false;                                                                       // set the modifier type
    LOGINFO("Unset the modifier operator.", LOG_TYPES::INFO, 2);
    // *****************************************************************************************************
    this->a_mod_p_.logAMod_  = 0.0;
    this->a_mod_p_.logTmpAMod_ = 0.0;
    // *****************************************************************************************************
    this->pRatioFunc_        = [this](const Config_t& _v)           { return this->pRatio(_v); };           // change the function pointer
    this->pKernelFunc_       = [this](int_ini_t fP, dbl_ini_t fV)   { return this->pRatio(fP, fV); };       // change the function pointer  
    this->logPKernelFunc_    = [this](int_ini_t fP, dbl_ini_t fV)   { return this->logPRatio(fP, fV); };    // change the function pointer
    this->logPRatioFuncFlips_= [this](uint nFlips)                  { return this->logPRatio(nFlips); };
    this->lower_states_.exc_ratio_ = [this](const Config_t& _v)     { return this->pRatio(_v); };           // change the function pointer
}
NQS_INST_CMB_ALL(unsetModifier, void, ());

// #################################################################################################≠≠≠≠≠≠#########################################

/**
* @brief Computes the logarithm of the modified ansatz.
* 
* This function calculates the logarithm of the modified ansatz based on the provided configuration.
* It first checks if the ansatz modifier is enabled and valid. If not, it returns 0.0.
* Otherwise, it applies the modifier to the configuration and sums up the results.
* Finally, it computes the logarithm of the conjugate of the negative sum.
* 
* !IMPORTANT: This function is only available when the state is not modified.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type for the result.
* @tparam _stateType The state type.
* @param _v The configuration for which the modified ansatz is computed.
* @return _T The logarithm of the modified ansatz.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logAnsatzModifier(Config_cr_t _v)
{
    if (!this->a_mod_p_.modified_ || !this->a_mod_p_.modifier_)
        return 0.0;
    if (this->a_mod_p_.modifies_state_)
        return 0.0;

    _T _out     = 0.0;
    auto _ret   = this->a_mod_p_.modifier_->operator()(_v);                     // get the modified states 
    for (const auto& [_, val]: _ret)                                            // sum up the modified states
        _out += val; 
    _out = std::log(algebra::conjugate(_out));                                 // get the log of the modified states
    return _out;
}
NQS_INST_CMB(double, double, logAnsatzModifier, double, (Config_cr_t));
NQS_INST_CMB(cpx, cpx, logAnsatzModifier, cpx, (Config_cr_t));
NQS_INST_CMB(double, cpx, logAnsatzModifier, cpx, (Config_cr_t));
NQS_INST_CMB(cpx, double, logAnsatzModifier, double, (Config_cr_t));

// #################################################################################################≠≠≠≠≠≠#########################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logAnsatzModifier(uint nFlips)
{
    if (!this->a_mod_p_.modified_ || !this->a_mod_p_.modifier_)
        return 0.0;

    // !TODO: Implement the logAnsatzModifier function if the state is modified
    if (this->a_mod_p_.modifies_state_)
        return 0.0;    

    auto _state = NQS_STATE;
    for (int i = 0; i < nFlips; ++i)
        flip(_state, this->flipPlaces_[i], 0, this->discVal_);
    // //
    // stout << "Flips: " << nFlips << " | ";
    // for (int i = 0; i < nFlips; ++i)
    //     stout << this->flipPlaces_[i] << " ";
    // stout << " | ";
    // // 
    // stout << NQS_STATE << " | " << _state << " | " << std::endl;
    // // 
    return logAnsatzModifier(_state);
}
NQS_INST_CMB(double, double, logAnsatzModifier, double, (uint));
NQS_INST_CMB(cpx, cpx, logAnsatzModifier, cpx, (uint));
NQS_INST_CMB(double, cpx, logAnsatzModifier, cpx, (uint));
NQS_INST_CMB(cpx, double, logAnsatzModifier, double, (uint));

// #################################################################################################≠≠≠≠≠≠#########################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logAnsatzModifier(uint fP, float fV)
{
    if (!this->a_mod_p_.modified_ || !this->a_mod_p_.modifier_)
        return 0.0;
    if (this->a_mod_p_.modifies_state_)
        return 0.0;

    auto _state = NQS_STATE;
    flip(_state, fP, 0, this->discVal_);
    return logAnsatzModifier(_state);
}
NQS_INST_CMB(double, double, logAnsatzModifier, double, (uint, float));
NQS_INST_CMB(cpx, cpx, logAnsatzModifier, cpx, (uint, float));
NQS_INST_CMB(double, cpx, logAnsatzModifier, cpx, (uint, float));
NQS_INST_CMB(cpx, double, logAnsatzModifier, double, (uint, float));

// #################################################################################################≠≠≠≠≠≠#########################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logAnsatzModifier(uint f, uint g, float v, float w)
{
    if (!this->a_mod_p_.modified_ || !this->a_mod_p_.modifier_)
        return 0.0;
    if (this->a_mod_p_.modifies_state_)
        return 0.0;
    auto _state = NQS_STATE;
    flip(_state, f, 0, this->discVal_);
    flip(_state, g, 0, this->discVal_);
    return logAnsatzModifier(_state);
}
NQS_INST_CMB(double, double, logAnsatzModifier, double, (uint, uint, float, float));
NQS_INST_CMB(cpx, cpx, logAnsatzModifier, cpx, (uint, uint, float, float));
NQS_INST_CMB(double, cpx, logAnsatzModifier, cpx, (uint, uint, float, float));
NQS_INST_CMB(cpx, double, logAnsatzModifier, double, (uint, uint, float, float));

// #################################################################################################≠≠≠≠≠≠#########################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logAnsatzModifier(int_ini_t f, dbl_ini_t w)
{
    if (!this->a_mod_p_.modified_ || !this->a_mod_p_.modifier_)
        return 0.0;
    if (this->a_mod_p_.modifies_state_)
        return 0.0; 
    
    auto _state = NQS_STATE;
    for (auto& fp : f)
        flip(_state, fp, 0, this->discVal_);
    return logAnsatzModifier(_state);
}
NQS_INST_CMB(double, double, logAnsatzModifier, double, (int_ini_t f, dbl_ini_t w));
NQS_INST_CMB(cpx, cpx, logAnsatzModifier, cpx, (int_ini_t f, dbl_ini_t w));
NQS_INST_CMB(double, cpx, logAnsatzModifier, cpx, (int_ini_t f, dbl_ini_t w));
NQS_INST_CMB(cpx, double, logAnsatzModifier, double, (int_ini_t f, dbl_ini_t w));

// #################################################################################################≠≠≠≠≠≠#########################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logPRatioMod(Config_cr_t _v)
{
    if (this->a_mod_p_.modified_)
    {
        // !TODO: Implement the logPRatioMod function
        if (this->a_mod_p_.modifies_state_)
        {
            throw std::runtime_error("The state is modified by the operator. This is not yet implemented.");
        }
        else    // the state is not modified by the operator - it only includes a multiplicative factor 
        {
            this->a_mod_p_.logTmpAMod_ = this->logAnsatzModifier(_v);
            return this->logPRatio(_v) + this->a_mod_p_.logTmpAMod_ - this->a_mod_p_.logAMod_;
        }
    }
    return this->logPRatio(_v);
}
NQS_INST_CMB(double, double, logPRatioMod, double, (Config_cr_t));
NQS_INST_CMB(cpx, cpx, logPRatioMod, cpx, (Config_cr_t));
NQS_INST_CMB(double, cpx, logPRatioMod, cpx, (Config_cr_t));
NQS_INST_CMB(cpx, double, logPRatioMod, double, (Config_cr_t));

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logPRatioMod(uint nFlips)
{
    if (this->a_mod_p_.modified_)
    {
        if (this->a_mod_p_.modifies_state_)
        {
            throw std::runtime_error("The state is modified by the operator. This is not yet implemented.");
        }
        else    // the state is not modified by the operator - it only includes a multiplicative factor 
        {
            this->a_mod_p_.logTmpAMod_ = this->logAnsatzModifier(nFlips);
            // stout << "Old: " << _oldF << " | New: " << _newF << " | " << "Diff: " << (_newF - _oldF) << std::endl;
            return this->logPRatio(nFlips) + (this->a_mod_p_.logTmpAMod_ - this->a_mod_p_.logAMod_);
            // return this->logPRatio(nFlips) + (this->logAnsatzModifier(nFlips) - this->a_mod_p_.logAMod_);
        }
    }
    return this->logPRatio(nFlips);
}
NQS_INST_CMB(double, double, logPRatioMod, double, (uint));
NQS_INST_CMB(cpx, cpx, logPRatioMod, cpx, (uint));
NQS_INST_CMB(double, cpx, logPRatioMod, cpx, (uint));
NQS_INST_CMB(cpx, double, logPRatioMod, double, (uint));

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logPRatioMod(uint fP, float fV)
{
    if (this->a_mod_p_.modified_)
    {
        if (this->a_mod_p_.modifies_state_)
        {
            throw std::runtime_error("The state is modified by the operator. This is not yet implemented.");
        }
        else    // the state is not modified by the operator - it only includes a multiplicative factor 
            return this->logPRatio(fP, fV) + (this->logAnsatzModifier(fP, fV) - this->a_mod_p_.logAMod_);
    }
    return this->logPRatio(fP, fV);
}
NQS_INST_CMB(double, double, logPRatioMod, double, (uint, float));
NQS_INST_CMB(cpx, cpx, logPRatioMod, cpx, (uint, float));
NQS_INST_CMB(double, cpx, logPRatioMod, cpx, (uint, float));
NQS_INST_CMB(cpx, double, logPRatioMod, double, (uint, float));

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logPRatioMod(uint f, uint g, float v, float w)
{
    if (this->a_mod_p_.modified_)
    {
        if (this->a_mod_p_.modifies_state_)
        {
            throw std::runtime_error("The state is modified by the operator. This is not yet implemented.");
        }
        else    // the state is not modified by the operator - it only includes a multiplicative factor 
            return this->logPRatio(f, g, v, w) + (this->logAnsatzModifier(f, g, v, w) - this->a_mod_p_.logAMod_);
    }
    return this->logPRatio(f, g, v, w);
}
NQS_INST_CMB(double, double, logPRatioMod, double, (uint, uint, float, float));
NQS_INST_CMB(cpx, cpx, logPRatioMod, cpx, (uint, uint, float, float));
NQS_INST_CMB(double, cpx, logPRatioMod, cpx, (uint, uint, float, float));
NQS_INST_CMB(cpx, double, logPRatioMod, double, (uint, uint, float, float));

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logPRatioMod(int_ini_t f, dbl_ini_t w)
{
    if (this->a_mod_p_.modified_)
    {
        if (this->a_mod_p_.modifies_state_)
        {
            throw std::runtime_error("The state is modified by the operator. This is not yet implemented.");
        }
        else    // the state is not modified by the operator - it only includes a multiplicative factor 
            return this->logPRatio(f, w) + (this->logAnsatzModifier(f, w) - this->a_mod_p_.logAMod_);
    }
    return this->logPRatio(f, w);
}
NQS_INST_CMB(double, double, logPRatioMod, double, (int_ini_t, dbl_ini_t));
NQS_INST_CMB(cpx, cpx, logPRatioMod, cpx, (int_ini_t, dbl_ini_t));
NQS_INST_CMB(double, cpx, logPRatioMod, cpx, (int_ini_t, dbl_ini_t));
NQS_INST_CMB(cpx, double, logPRatioMod, double, (int_ini_t, dbl_ini_t));

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
_T NQS<_spinModes, _Ht, _T, _stateType>::logPRatioMod(Config_cr_t v, Config_cr_t w)
{
    if (this->a_mod_p_.modified_)
    {
        if (this->a_mod_p_.modifies_state_)
        {
            throw std::runtime_error("The state is modified by the operator. This is not yet implemented.");
        }
        else    // the state is not modified by the operator - it only includes a multiplicative factor 
            return this->logPRatio(v, w) + (this->logAnsatzModifier(v) - this->logAnsatzModifier(w));
    }
    return this->logPRatio(v, w);
}
NQS_INST_CMB(double, double, logPRatioMod, double, (Config_cr_t, Config_cr_t));
NQS_INST_CMB(cpx, cpx, logPRatioMod, cpx, (Config_cr_t, Config_cr_t));
NQS_INST_CMB(double, cpx, logPRatioMod, cpx, (Config_cr_t, Config_cr_t));
NQS_INST_CMB(cpx, double, logPRatioMod, double, (Config_cr_t, Config_cr_t));

// #################################################################################################≠≠≠≠≠≠#########################################
