#include "rbm.h"

// ##########################################################################################################################################

// ########################################################### C O N S T R U C T ############################################################

// ##########################################################################################################################################

#include <memory>
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM<_spinModes, _Ht, _T, _stateType>::RBM(std::shared_ptr<Hamiltonian<_Ht, _spinModes>>& _H, uint _nHid, 
										double _lr, uint _threadNum, int _nPart,
										const NQSLS_p& _lower, 
										const std::vector<double>& _beta)
	: NQS_S<_spinModes, _Ht, _T, _stateType>(_H, _lr, _threadNum, _nPart, _lower, _beta)
{
	this->nHid_ 			= _nHid;
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

#if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP	
	for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++)	// allocate the vector for using it in the RBM
		this->thetaTMP_ = NQSB(this->nHid_);
#endif
	this->thetaTmpCol_ = NQSB(this->nHid_);
	NQS_S<_spinModes, _Ht, _T, _stateType>::allocate();						// allocate the rest
}

////////////////////////////////////////////////////////////////////////////

/**
* @brief Initializes the weights and biases for the RBM NQS model
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM<_spinModes, _Ht, _T, _stateType>::init()
{
	// initialize biases visible
// #ifndef _DEBUG
// #pragma omp parallel for num_threads(this->threads_.threadNum_)
// #endif
	NQS_S<_spinModes, _Ht, _T, _stateType>::init();
	const double stddev = std::max(std::sqrt(2.0 / (this->info_p_.nVis_ + this->nHid_)), 0.01);

	// Initialize visible biases
	for (int i = 0; i < this->info_p_.nVis_; i++) {
		this->bV_(i) = algebra::cast<_T>(this->ran_->template randomNormal<double>(0.0, stddev) + I * this->ran_->template randomNormal<double>(0.0, stddev));
	}

	// Initialize hidden biases
	for (int i = 0; i < this->nHid_; i++) {
		this->bH_(i) = algebra::cast<_T>(this->ran_->template randomNormal<double>(0.0, stddev) + I * this->ran_->template randomNormal<double>(0.0, stddev));
	}

	// Initialize weights matrix using Xavier Initialization
	for (int i = 0; i < this->W_.n_rows; i++) {
		for (uint j = 0; j < this->W_.n_cols; j++) {
			this->W_(i, j) = algebra::cast<_T>(this->ran_->template randomNormal<double>(0.0, stddev) + I * this->ran_->template randomNormal<double>(0.0, stddev));
		}
	}
	this->Weights_.subvec(0, this->info_p_.nVis_ - 1) 									= this->bV_;
	this->Weights_.subvec(this->info_p_.nVis_, this->info_p_.nVis_ + this->nHid_ - 1) 	= this->bH_;
	this->Weights_.subvec(this->info_p_.nVis_ + this->nHid_, this->rbmSize_ - 1) 		= this->W_.as_col();
}

////////////////////////////////////////////////////////////////////////////

/*
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
		VEQVS(lr, this->info_p_.lr_));
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

// template <uint _spinModes, typename _Ht, typename _T, class _stateType>
// inline void RBM<_spinModes, _Ht, _T, _stateType>::setWeights(const NQSW& _W, const NQSB& _bV, const NQSB& _bH)
// {
// 	this->W_	= _W;
// 	this->bV_	= _bV;
// 	this->bH_	= _bH;
// }

/*
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