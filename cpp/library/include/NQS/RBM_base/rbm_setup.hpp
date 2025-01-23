#include "rbm.h"
// ##########################################################################################################################################

namespace NQS_NS
{
	// ##########################################################################################################################################

	// ########################################################### C O N S T R U C T ############################################################

	// ##########################################################################################################################################

	template <uint _spinModes, typename _Ht, typename _T, class _stateType>
	RBM<_spinModes, _Ht, _T, _stateType>::RBM(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p)
											// const NQSLS_p& _lower, 
											// const std::vector<double>& _beta)
		: NQS_S<_spinModes, _Ht, _T, _stateType>(_p)
	{
		this->nHid_ 			= _p.nHid_[0];
		this->rbmSize_  		= this->nHid_ + this->info_p_.nVis_ + this->nHid_ * this->info_p_.nVis_;
		this->info_p_.fullSize_ = this->rbmSize_;
		this->allocate();
		this->setInfo();
	}

	// ##########################################################################################################################################

	template <uint _spinModes, typename _Ht, typename _T, class _stateType>
	RBM<_spinModes, _Ht, _T, _stateType>::RBM(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
		: NQS_S<_spinModes, _Ht, _T, _stateType>(_p, _lower, _beta)
	{
		this->nHid_ 			= _p.nHid_[0];
		this->rbmSize_  		= this->nHid_ + this->info_p_.nVis_ + this->nHid_ * this->info_p_.nVis_;
		this->info_p_.fullSize_ = this->rbmSize_;
		this->allocate();
		this->setInfo();
	}

	// ##########################################################################################################################################

	// ######################################################## I N I T I A L I Z E R S #########################################################

	// ##########################################################################################################################################

	/**
	* @brief Allocates memory for the weights and other necessary vectors in the RBM class.
	* 
	* This function checks if the weight vectors and other necessary vectors are empty and resizes them accordingly.
	* It also allocates memory for temporary vectors used in multi-threading if the appropriate flags are defined.
	* Finally, it calls the allocate function of the base class to allocate the rest of the necessary resources.
	* 
	* @tparam _spinModes Number of spin modes.
	* @tparam _Ht Hamiltonian type.
	* @tparam _T Data type for the weights.
	* @tparam _stateType State type.
	*/
	template<uint _spinModes, typename _Ht, typename _T, class _stateType>
	inline void RBM<_spinModes, _Ht, _T, _stateType>::allocate()
	{
		// allocate weights
		if (this->bV_.is_empty()) 			this->bV_.resize(this->info_p_.nVis_);
		if (this->bH_.is_empty()) 			this->bH_.resize(this->nHid_);
		if (this->theta_.is_empty()) 		this->theta_.resize(this->nHid_);
		if (this->thetaCOSH_.is_empty()) 	this->thetaCOSH_.resize(this->nHid_);
		if (this->W_.is_empty()) 			this->W_.resize(this->nHid_, this->info_p_.nVis_);
		this->thetaTmpCol_ 	= NQSB(this->nHid_);
		NQS_S<_spinModes, _Ht, _T, _stateType>::allocate();						// allocate the rest
	}

	// ##########################################################################################################################################

	/**
	* @brief Sets the info about the RBM model NQS.
	*/
	template<uint _spinModes, typename _Ht, typename _T, class _stateType>
	inline void RBM<_spinModes, _Ht, _T, _stateType>::setInfo()
	{
		this->info_ = "";
		strSeparatedS(this->info_, ',', "RBM", 
			VEQV(nv, this->info_p_.nVis_), 
			VEQV(nh, this->nHid_), 
			VEQV(nS, this->spinModes_),
			VEQVS(lr, this->info_p_.lr()));
	}

	// ##########################################################################################################################################

	// ############################################################# S E T T E R S ##############################################################

	// ##########################################################################################################################################

	/**
	* @brief Sets the new state according to a given vector. Updates the value if NQS_ANGLES_UPD is set.
	* @param _st Column vector to be set as a new state
	* @param _set if one wants to additionaly set the angles
	*/
	template<uint _spinModes, typename _Ht, typename _T, class _stateType>
	inline void RBM<_spinModes, _Ht, _T, _stateType>::setState(const Config_t& _st, bool _set)
	{
		NQS_S<_spinModes, _Ht, _T, _stateType>::setState(_st);
	#ifdef NQS_ANGLES_UPD
		if (_set) this->setTheta();
	#endif
	}

	////////////////////////////////////////////////////////////////////////////

	/**
	* @brief Sets the new state according to a given integer state. Updates the value if NQS_ANGLES_UPD is set.
	* @param _st Column vector to be set as a new state
	* @param _set if one wants to additionaly set the angles
	*/
	template<uint _spinModes, typename _Ht, typename _T, class _stateType>
	inline void RBM<_spinModes, _Ht, _T, _stateType>::setState(u64 _st, bool _set)
	{
		NQS_S<_spinModes, _Ht, _T, _stateType>::setState(_st);
	#ifdef NQS_ANGLES_UPD
		if (_set) this->setTheta();
	#endif
	}

	// ##########################################################################################################################################

	// ############################################################# W E I G H T S ##############################################################

	// ##########################################################################################################################################

	/**
	* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
	* @param _path folder for the weights to be saved onto
	* @param _file name of the file to save the weights onto
	* @returns whether the load has been successful
	*/
	template<uint _spinModes, typename _Ht, typename _T, class _stateType>
	inline bool RBM<_spinModes, _Ht, _T, _stateType>::setWeights(std::string _path, std::string _file)
	{
		BEGIN_CATCH_HANDLER
		{
			// set the forces vector for the weights
			if(!NQS_S<_spinModes, _Ht, _T, _stateType>::setWeights(_path, _file))
				return false;
			this->bV_	= this->F_.subvec(0, this->info_p_.nVis_ - 1);
			this->bH_	= this->F_.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1);
			this->W_	= arma::reshape(this->F_.subvec(this->info_p_.nVis_ + this->nHid_, this->info_p_.nVis_ + this->nHid_ + this->W_.n_rows * this->W_.n_cols - 1),
										this->W_.n_rows, this->W_.n_cols);
		}
		END_CATCH_HANDLER("Couldn't set the weights for the RBM NQS...", return false);
		return true;
	}

	////////////////////////////////////////////////////////////////////////////

	/**
	* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
	* @param _path folder for the weights to be saved onto
	* @param _file name of the file to save the weights onto
	* @returns whether the load has been successful
	*/
	template<uint _spinModes, typename _Ht, typename _T, class _stateType>
	inline bool RBM<_spinModes, _Ht, _T, _stateType>::saveWeights(std::string _path, std::string _file)
	{
		BEGIN_CATCH_HANDLER
		{
			this->Weights_.subvec(0, this->info_p_.nVis_ - 1) = this->bV_;
			this->Weights_.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1) = this->bH_;
			this->Weights_.subvec(this->info_p_.nVis_ + this->nHid_, this->rbmSize_ - 1) = this->W_.as_col();
			
			// set the forces vector for the weights
			if(!NQS_S<_spinModes, _Ht, _T, _stateType>::saveWeights(_path, _file))
				return false;
		}
		END_CATCH_HANDLER("Couldn't save the weights for the RBM NQS...", return false);
		return true;
	}
	// ##########################################################################################################################################

};	// namespace NQS_NS

// ##########################################################################################################################################