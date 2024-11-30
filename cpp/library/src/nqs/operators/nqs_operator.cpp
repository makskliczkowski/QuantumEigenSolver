#include "../../../include/NQS/nqs_operator.h"
#include <string>

// ##########################################################################################################################################

namespace Operators 
{
    template class Operators::OperatorNQS<double>;
    template class Operators::OperatorNQS<std::complex<double>>;
    template class Operators::OperatorNQS<double, uint>;
    template class Operators::OperatorNQS<double, int>;
    template class Operators::OperatorNQS<std::complex<double>, uint>;
    template class Operators::OperatorNQS<std::complex<double>, int>;
    template class Operators::OperatorNQS<double, uint, uint>;
    template class Operators::OperatorNQS<double, int, int>;
    template class Operators::OperatorNQS<std::complex<double>, uint, uint>;
    template class Operators::OperatorNQS<std::complex<double>, int, int>;
    // no more exotics

    // ##########################################################################################################################################

    /**
    * @brief Apply the operators with a value change (with pRatio). The function is used to calculate the 
    * probability ratio for the given state and the operator.
    * @param s base state to apply the operators to
    * @param _fun pRatio function from the NQS
    * @param ...a additional parameters to the operators
    * @returns value of the operator acting on the state with the probability ratio applied
    */
	template<typename _T, typename ..._Ts>
	_T OperatorNQS<_T, _Ts...>::operator()(u64 s, NQSFunCol _fun, _Ts ...a)
	{
		_T _valTotal = 0.0;
		for (const auto& [s2, _val] : this->operator()(s, a...)) {            // go through operator acting on the state
			Binary::int2base(s2, this->state_, _SPIN_RBM);              // set the state
			_valTotal += _val * algebra::cast<_T>(_fun(this->state_));  // calculate the probability ratio
		}
		return algebra::cast<_T>(_valTotal);
	}

    // template instantiation
    template double OperatorNQS<double>::operator()(u64, NQSFunCol);					
    template double OperatorNQS<double, uint>::operator()(u64, NQSFunCol, uint);		
    template double OperatorNQS<double, int>::operator()(u64, NQSFunCol, int);		
    template double OperatorNQS<double, uint, uint>::operator()(u64, NQSFunCol, uint, uint);		
    template double OperatorNQS<double, int, int>::operator()(u64, NQSFunCol, int, int);		

    template cpx OperatorNQS<cpx>::operator()(u64, NQSFunCol);					
    template cpx OperatorNQS<cpx, uint>::operator()(u64, NQSFunCol, uint);		
    template cpx OperatorNQS<cpx, int>::operator()(u64, NQSFunCol, int);		
    template cpx OperatorNQS<cpx, uint, uint>::operator()(u64, NQSFunCol, uint, uint);		
    template cpx OperatorNQS<cpx, int, int>::operator()(u64, NQSFunCol, int, int);		

	////////////////////////////////////////////////////////////////////////////

	/**
	* @brief Apply the operators with a value change (!with pRatio). The function is used to calculate the
	* probability ratio for the given state and the operator.
	* @param s base state to apply the operators to - vector version
	* @param _fun pRatio function from the NQS
	* @param ...a additional parameters to the operators
	* @returns value of the operator acting on the state with the probability ratio applied
	*/
	template<typename _T, typename ..._Ts>
	_T Operators::OperatorNQS<_T, _Ts...>::operator()(_OP_V_T_CR s, NQSFunCol _fun, _Ts ...a) const
	{
		_T _valTotal = 0.0;
		for (auto& [s2, _val] : this->operator()(s, a...))
		{
			_T _functionVal = CAST<_T>(_fun(s2));                   // calculate the probability ratio
			_valTotal 		= _valTotal + _functionVal * _val;      // calculate the value
		}
		return _valTotal;
	}

    // template instantiation
    template double OperatorNQS<double>::operator()(_OP_V_T_CR, NQSFunCol) const;					
    template double OperatorNQS<double, uint>::operator()(_OP_V_T_CR, NQSFunCol, uint) const;		
    template double OperatorNQS<double, int>::operator()(_OP_V_T_CR, NQSFunCol, int) const;		
    template double OperatorNQS<double, uint, uint>::operator()(_OP_V_T_CR, NQSFunCol, uint, uint) const;		
    template double OperatorNQS<double, int, int>::operator()(_OP_V_T_CR, NQSFunCol, int, int) const;		

    template cpx OperatorNQS<cpx>::operator()(_OP_V_T_CR, NQSFunCol) const;					
    template cpx OperatorNQS<cpx, uint>::operator()(_OP_V_T_CR, NQSFunCol, uint) const;		
    template cpx OperatorNQS<cpx, int>::operator()(_OP_V_T_CR, NQSFunCol, int) const;		
    template cpx OperatorNQS<cpx, uint, uint>::operator()(_OP_V_T_CR, NQSFunCol, uint, uint) const;		
    template cpx OperatorNQS<cpx, int, int>::operator()(_OP_V_T_CR, NQSFunCol, int, int) const;		

	// ##########################################################################################################################################

};

// ##########################################################################################################################################

// MEASUREMENTS OF THE OPERATORS

// ##########################################################################################################################################

namespace NQSAv
{
    // ##########################################################################################################################################

	/**
	* @brief Measures the NQS (Neural Quantum State) function for a given state.
	*
	* This function performs a measurement of the NQS function for a specified state.
	*
	* @tparam _T The type of the NQS function.
	* @param s The state for which the measurement is performed.
	* @param _fun The NQS function to be measured.
	*/
	template <typename _T>
	void NQSAv::MeasurementNQS<_T>::measure(u64 s, NQSFunCol _fun)
	{
		BEGIN_CATCH_HANDLER
		{
			for (int i = 0; i < this->opG_.size(); ++i)
			{
				auto& _op 	= this->opG_[i];
				auto val 	= _op->operator()(s, _fun);
				this->containersG_[i].updCurrent(val);				// update the container
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			for (int i = 0; i < this->opL_.size(); ++i)
			{
				auto& _op 	= this->opL_[i];
				for (auto j = 0; j < this->Ns_; ++j)				// go through the locals
				{
					auto val = _op->operator()(s, _fun, j);
					this->containersL_[i].updCurrent(val, j);
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of local operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			for (int k = 0; k < this->opC_.size(); ++k)
			{
				auto& _op = this->opC_[k];
				for (auto i = 0; i < this->Ns_; ++i)
				{
					for (auto j = 0; j < this->Ns_; ++j)
					{
						auto val = _op->operator()(s, _fun, i, j);
						this->containersC_[k].updCurrent(val, i, j);
					}
				}
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);
	}
    // template instantiation
    template void MeasurementNQS<double>::measure(u64, NQSFunCol);
    template void MeasurementNQS<cpx>::measure(u64, NQSFunCol);

    // ##########################################################################################################################################

    /**
    * @brief Measures the NQS (Neural Quantum State) function for a given state (vector version).
    *
    * This function performs a measurement of the NQS function for a specified state.
    *
    * @tparam _T The type of the NQS function.
    * @param s The state for which the measurement is performed.
    * @param _fun The NQS function to be measured.
    */
    template <typename _T>
    void NQSAv::MeasurementNQS<_T>::measure(Operators::_OP_V_T_CR s, NQSFunCol _fun)
    {
        BEGIN_CATCH_HANDLER
        {
            for (int i = 0; i < this->opG_.size(); ++i)
            {
                auto& _op 	= this->opG_[i];
                auto val 	= _op->operator()(s, _fun);
                this->containersG_[i].updCurrent(val);				// update the container
            }
        }
        END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

        BEGIN_CATCH_HANDLER
        {
            for (int i = 0; i < this->opL_.size(); ++i)
            {
                auto& _op 	= this->opL_[i];
                for (auto j = 0; j < this->Ns_; ++j)				// go through the locals
                {
                    auto val = _op->operator()(s, _fun, j);
                    this->containersL_[i].updCurrent(val, j);
                }
            }
        }
        END_CATCH_HANDLER("Problem in the measurement of local operators.", ;);

        BEGIN_CATCH_HANDLER
        {
            for (int k = 0; k < this->opC_.size(); ++k)
            {
                auto& _op = this->opC_[k];
                for (auto i = 0; i < this->Ns_; ++i)
                {
                    for (auto j = 0; j < this->Ns_; ++j)
                    {
                        auto val = _op->operator()(s, _fun, i, j);
                        this->containersC_[k].updCurrent(val, i, j);
                    }
                }
            }
        }
        END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);
    }
    // template instantiation
    template void MeasurementNQS<double>::measure(Operators::_OP_V_T_CR, NQSFunCol);
    template void MeasurementNQS<cpx>::measure(Operators::_OP_V_T_CR, NQSFunCol);

    // ##########################################################################################################################################

    /**
    * @brief Measure the operators for the given state using the operator representation acting on 
    * the state in a full Hilbert space. This requires providing the Hilbert space and the state.
    * 
    * This function performs a measurement of the NQS (Neural Quantum State) function for a specified state
    * in the full Hilbert space. It updates the containers with the measured values.
    *
    * @tparam _T The type of the NQS function.
    * @param _state The state to measure the operators for.
    * @param _H The Hilbert space in which to measure the operators.
    */
	template<typename _T>
	void NQSAv::MeasurementNQS<_T>::measure(const arma::Col<_T>& _state, const Hilbert::HilbertSpace<_T>& _H)
	{
		BEGIN_CATCH_HANDLER
		{
			// measure global
			for (int i = 0; i < this->opG_.size(); ++i)
			{
				auto& _op 	= this->opG_[i];
				auto& _cont = this->containersG_[i];
				_cont.resetMB();
				_cont.setManyBodyMat(_H, _op.get());                                // set the many body matrix
				auto _val 	= Operators::applyOverlap(_state, _cont.mbmat());   
				_cont.setManyBodyVal(_val);                                         // set the many body value
				_cont.resetMBMat();                                                 // reset the many body matrix
			}	
		}
		END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure local
			for (int i = 0; i < this->opL_.size(); ++i)
			{
				auto& _op 	= this->opL_[i];
				auto& _cont = this->containersL_[i];
				// reset
				_cont.resetMB();

				// go through the local operators
				for (auto j = 0; j < _op->getNs(); ++j)
				{
					_cont.setManyBodyMat(_H, _op.get(), (uint)j);					// set the many body matrix for the operator
					auto _val = Operators::applyOverlap(_state, _cont.mbmat());
					_cont.setManyBodyVal(_val, (uint)j);					        // set the many body value
				}
				_cont.resetMBMat();                                                 // reset the many body matrix
                this->usedMB_ = true;
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of local operators.", ;);

		BEGIN_CATCH_HANDLER
		{
			// measure correlation
			for (int k = 0; k < this->opC_.size(); ++k)
			{
				auto& _op = this->opC_[k];
				auto& _cont = this->containersC_[k];
				_cont.resetMB();

				for (auto i = 0; i < _op->getNs(); ++i)
				{
					for (auto j = 0; j < _op->getNs(); ++j)
					{
						_cont.setManyBodyMat(_H, _op.get(), (uint)i, (uint)j);
						auto _val = Operators::applyOverlap(_state, _cont.mbmat());
						_cont.setManyBodyVal(_val, (uint)i, (uint)j);
					}
				}
				_cont.resetMBMat();
                this->usedMB_ = true;
			}
		}
		END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);
	}

    // template instantiation
    template void MeasurementNQS<double>::measure(const arma::Col<double>&, const Hilbert::HilbertSpace<double>&);
    template void MeasurementNQS<cpx>::measure(const arma::Col<cpx>&, const Hilbert::HilbertSpace<cpx>&);

    // ##########################################################################################################################################

    // SAVE

    // ##########################################################################################################################################

    template <typename _T>
    void MeasurementNQS<_T>::saveMB(const strVec& _ext, std::string _nameGlobal, std::string _nameLocal, std::string _nameCorr, std::string _appName, bool app)
    {
        _nameGlobal = _nameGlobal.size() == 0 ? "op_global" : _nameGlobal;
        _nameLocal  = _nameLocal.size() == 0 ? "op_local" : _nameLocal;
        _nameCorr   = _nameCorr.size() == 0 ? "op_corr" : _nameCorr;
        _appName    = _appName.size() == 0 ? "MB/" : _appName;
        BEGIN_CATCH_HANDLER
        {
            // save global
            for (int i = 0; i < this->opG_.size(); ++i)
            {
                auto& _cont     = this->containersG_[i];
                auto& _op       = this->opG_[i];
                auto _name      = _op->getNameS();
                _name           = _name.size() == 0 ? "OP" + std::to_string(i) : _name;
                // many body
                {
                    const arma::Mat<_T>& M = _cont.mbval();
                    if (M.size() != 0)
                        for (const auto& ext : _ext)
                            saveAlgebraic(dir_, _nameGlobal + ext, M, _appName + "/glob/" + _name, i > 0 || app);
                }
            }
        }
        END_CATCH_HANDLER("Problem in the saving of global operators.", ;);

        BEGIN_CATCH_HANDLER
        {
            // save local
            for (int i = 0; i < this->opL_.size(); ++i)
            {
                auto& _cont = this->containersL_[i];
                auto& _op   = this->opL_[i];
                auto _name  = _op->getNameS();
                _name       = _name.size() == 0 ? "OP" + std::to_string(i) : _name;
                // many body
                {
                    const arma::Mat<_T>& M = _cont.mbval();
                    if (M.size() != 0)
                        for (const auto& ext : _ext)
                            saveAlgebraic(dir_, _nameLocal + ext, M, _appName + "/loc/" + _name, i > 0 || app);
                }
            }
        }
        END_CATCH_HANDLER("Problem in the saving of local operators.", ;);

        BEGIN_CATCH_HANDLER
        {
            // save correlation
            for (int i = 0; i < this->opC_.size(); ++i)
            {
                auto& _cont     = this->containersC_[i];
                auto& _op       = this->opC_[i];
                auto _name      = _op->getNameS();
                _name           = _name.size() == 0 ? "OP" + std::to_string(i) : _name;
                // many body
                {
                    const arma::Mat<_T>& M = _cont.mbval();
                    if (M.size() != 0)
                        for (const auto& ext : _ext)
                            saveAlgebraic(dir_, _nameCorr + ext, M, _appName + "/corr/" + _name, i > 0 || app);
                }
            }
        }
        END_CATCH_HANDLER("Problem in the saving of correlation operators.", ;);
    }

    // template instantiation
    template void MeasurementNQS<double>::saveMB(const strVec&, std::string, std::string, std::string, std::string, bool);
    template void MeasurementNQS<cpx>::saveMB(const strVec&, std::string, std::string, std::string, std::string, bool);

    // ##########################################################################################################################################

    template <typename _T>
    void MeasurementNQS<_T>::saveNQS(const strVec& _ext, std::string _nameGlobal, std::string _nameLocal, std::string _nameCorr, std::string _appName, bool app)
    {
        _nameGlobal = _nameGlobal.size() == 0 ? "op_global" : _nameGlobal;
        _nameLocal  = _nameLocal.size() == 0 ? "op_local" : _nameLocal;
        _nameCorr   = _nameCorr.size() == 0 ? "op_corr" : _nameCorr;
        _appName    = _appName.size() == 0 ? "NQS/" : _appName;

        BEGIN_CATCH_HANDLER
        {
            // save global
            for (int i = 0; i < this->opG_.size(); ++i)
            {
                auto& _cont     = this->containersG_[i];
                auto& _op       = this->opG_[i];
                auto _name      = _op->getNameS();
                _name           = _name.size() == 0 ? "OP" + std::to_string(i) : _name;
				// nqs
				{
					const auto M = _cont.template mean<cpx>();
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, _nameGlobal + ext, M, _appName + "/glob/" + _name, i > 0 || app);
				}
            }
        }
        END_CATCH_HANDLER("Problem in the saving of global operators.", ;);

        BEGIN_CATCH_HANDLER
        {
            // save local
            for (int i = 0; i < this->opL_.size(); ++i)
            {
                auto& _cont = this->containersL_[i];
                auto& _op   = this->opL_[i];
                auto _name  = _op->getNameS();
                _name       = _name.size() == 0 ? "OP" + std::to_string(i) : _name;
                // nqs
				{
					const auto M = _cont.template mean<cpx>();
					// save!
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, _nameLocal + ext, M, _appName + "/loc/" + _name, i > 0 || app);
				}
            }
        }
        END_CATCH_HANDLER("Problem in the saving of local operators.", ;);

        BEGIN_CATCH_HANDLER
        {
            // save correlation
            for (int i = 0; i < this->opC_.size(); ++i)
            {
                auto& _cont     = this->containersC_[i];
                auto& _op       = this->opC_[i];
                auto _name      = _op->getNameS();
                _name           = _name.size() == 0 ? "OP" + std::to_string(i) : _name;
                // nqs
				{
					const auto M = _cont.template mean<cpx>();
					// save!
					if (M.size() != 0)
						for (const auto& ext : _ext)
							saveAlgebraic(dir_, _nameCorr + ext, M, _appName + "/corr/" + _name, i > 0 || app);
				}
            }
        }
        END_CATCH_HANDLER("Problem in the saving of correlation operators.", ;);
    }

    // template instantiation
    template void MeasurementNQS<double>::saveNQS(const strVec&, std::string, std::string, std::string, std::string, bool);
    template void MeasurementNQS<cpx>::saveNQS(const strVec&, std::string, std::string, std::string, std::string, bool);

    // ##########################################################################################################################################

	/**
	* @brief Saves the measurement data of the NQS (Neural Quantum State) to files.
	* 
	* This function saves the measurement data to files specified by the given names.
	* It first checks if there are any containers to save. If no containers are present,
	* it logs a warning message and returns. If the measurement data uses the MB (Measurement Basis),
	* it saves the MB data first. Then, it saves the NQS data.
	* 
	* @tparam _T The type of the measurement data.
	* @param _ext A vector of strings representing the file extensions.
	* @param _nameGlobal The name of the global measurement file.
	* @param _nameLocal The name of the local measurement file.
	* @param _nameCorr The name of the correlation measurement file.
	* @param _app A boolean flag indicating whether to append the data to the existing files.
	*/
	template<typename _T>
	void NQSAv::MeasurementNQS<_T>::save(const strVec& _ext, std::string _nameGlobal, std::string _nameLocal, std::string _nameCorr, std::string _appName, bool _app)
	{
		if (this->containersL_.empty() && this->containersC_.empty() && this->containersG_.empty()) {
			LOGINFO("No containers to save.", LOG_TYPES::WARNING, 3);
			return;
		}
		if (this->usedMB_)
			this->saveMB(_ext, _nameGlobal, _nameLocal, _nameCorr, _appName, _app);
		this->saveNQS(_ext, _nameGlobal, _nameLocal, _nameCorr, _appName, _app);
	}

    // template instantiation
    template void MeasurementNQS<double>::save(const strVec&, std::string, std::string, std::string, std::string, bool);
    template void MeasurementNQS<cpx>::save(const strVec&, std::string, std::string, std::string, std::string, bool);
    // ##########################################################################################################################################

};