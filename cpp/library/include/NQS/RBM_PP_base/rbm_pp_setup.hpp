#include "rbm_pp.h"

// ##########################################################################################################################################

// ############################################################## A N S A T Z ###############################################################

// ##########################################################################################################################################

/*
* @brief reproduces the RBM-PP NQS ANSATZ.
* @param _in the state for which the ansatz shall be calculated
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::ansatz(Config_cr_t _in) const
{
	// set the Jacobian before!
	return RBM_S<_spinModes, _Ht, _T, _stateType>::ansatz(_in) * this->getPfaffian(_in); //* std::pow(2.0, this->n_hidden)
};

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::ansatzlog(Config_cr_t _in) const
{
	return RBM_S<_spinModes, _Ht, _T, _stateType>::ansatzlog(_in) + std::log(this->getPfaffian(_in));
}

/*
* @brief calculates the ratio of the two RBM-PP states - used for calculating the excited states (_other->ansatz / this->ansatz)
* @param _in vector to calculate the ratio for
* @param _other pointer to the other NQS to calculate the ratio with
* @return ratio of the two states (other / this) for a given state _in (vector)
* @note the ratio is calculated as: _other->ansatz / this->ansatz * _other->getPfaffian(_in) / this->getPfaffian(_in)
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
inline _T RBM_PP<_spinModes, _Ht, _T, _stateType>::ansatz_ratiolog(Config_cr_t _in, NQS<_spinModes, _Ht, _T, _stateType>* _other) const
{
	auto _rbm_pp_other = dynamic_cast<RBM_PP<_spinModes, _Ht, _T, _stateType>*>(_other);
	// return RBM_S<_spinModes, _Ht, _T, _stateType>::ansatz_ratio(_in, _other) * _rbm_pp_other->getPfaffian(_in) / this->getPfaffian(_in);
	return RBM_S<_spinModes, _Ht, _T, _stateType>::ansatz_ratiolog(_in, _other) + std::log(_rbm_pp_other->getPfaffian(_in)) - std::log(this->getPfaffian(_in));
}


// ##########################################################################################################################################

// ######################################################## I N I T I A L I Z E R S #########################################################

// ##########################################################################################################################################

/*
* @brief Based on the spin of the particle at r_i position and the particle at r_j position, returns the index of 
* the coefficient in the flattened F matrix.
* @param _spini spin of the particle at r_i position
* @param _spinj spin of the particle at r_j position
* @param ri position of the particle in the lattice
* @param rj position of the particle in the lattice
* @returns index of the coefficient in the flattened F matrix
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline u64 RBM_PP<_spinModes, _Ht, _T, _stateType>::getFPPIndex(bool _spini, bool _spinj, uint ri, uint rj) const
{
	if (_spini)
	{
		if (_spinj)
			return ri * this->info_p_.nSites_ + rj;
		else
			return this->nSites2_ + ri * this->info_p_.nSites_ + rj;
	}
	else
	{
		if (_spinj)
			return 2 * this->nSites2_ + ri * this->info_p_.nSites_ + rj;
		else
			return 3 * this->nSites2_ + ri * this->info_p_.nSites_ + rj;
	}
}

////////////////////////////////////////////////////////////////////////

/*
* @brief Allocate the memory for the NQS parameters. 
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::allocate()
{
	// !TODO implement changable number of fermions
	// allocate weights
	// matrix for each step
	RBM_S<_spinModes, _Ht, _T, _stateType>::allocate();
	this->X_		= NQSW(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);
	this->Xinv_		= NQSW(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);
#if defined NQS_USE_MULTITHREADING && not defined NQS_USE_OMP	// allocate the vector for using it in the RBM
	for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++)
		this->XTmp_[this->threads_.threads_[_thread].get_id()] = NQSW(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);
#endif

	// allocate the weights themselves !TODO - make this symmetric? 
	// square matrix with spin changes F_{ij}^{\\sigma, \\sigma '}
	this->Fpp_		= NQSB(this->nPP_);
}

////////////////////////////////////////////////////////////////////////

/*
* @brief Initializes the weights and biases for the RBM-PP NQS model
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::init()
{
	RBM<_spinModes, _Ht, _T, _stateType>::init();
	const double std_dev = 1.0 / std::sqrt(this->nPP_);
	// matrix for the PP wave function - contains all the necessary weights
	// is initialized according to the distance between the sites
	this->Fpp_	= NQSB(this->nPP_, arma::fill::zeros);
	auto _lat	= this->H_->getLat();
	for (uint i = 0; i < this->info_p_.nSites_; i++)	// go through the lattice
	{
		for (uint j = 0; j < this->info_p_.nSites_; j++)	// go through the lattice
		{
			// get the distance between the sites
			auto distance = _lat ? _lat->getSiteDistance(i, j) : std::abs<int>(i - j);
			for (const auto& spinSec: this->spinSectors_)
			{
				// make the weights proportional to the distance
				this->Fpp_(this->getFPPIndex(spinSec[0], spinSec[1], i, j)) = algebra::cast<_T>(this->ran_->template randomNormal<double>(0.0, std_dev) + I * this->ran_->template randomNormal<double>(0.0, std_dev));
				if (distance != 0)
					this->Fpp_(this->getFPPIndex(spinSec[0], spinSec[1], i, j)) /= distance * distance;
			}
		}
	}
	// !TODO¿
	this->Weights_.subvec(this->rbmSize_, this->rbmPPSize_ - 1) = this->Fpp_;
}

////////////////////////////////////////////////////////////////////////

/*
* @brief Sets the info about the RBM-PP model NQS.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::setInfo()
{
	this->info_ = "";
	strSeparatedS(	this->info_, ',', "RBMPP", 
					VEQV(mod, this->H_->getType()), 
					VEQV(nv, this->info_p_.nVis_), 
					VEQV(nh, this->nHid_), 
					VEQV(npp, this->nPP_),
					VEQV(nS, this->spinModes_),
					VEQVS(lr, this->info_p_.lr_));
}

// ##########################################################################################################################################

// ############################################################# S E T T E R S ##############################################################

// ##########################################################################################################################################

/**
* @brief Sets the state and the corresponding PP state as well.
* Updates the pfaffian matrix.
* @param _st column state to be set
* @param _set set the matrices?
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::setState(const Config_t& _st, bool _set)
{
	RBM_S<_spinModes, _Ht, _T, _stateType>::setState(_st, _set);
#ifdef NQS_ANGLES_UPD
	if (_set)
	{
		// set PP matrix
		this->setFPP();
		// set Pfaffian value for newly set matrix
		this->setPfaffian();
		// set candidate to be the same for the angles update
		this->Xnew_			= this->X_;
		this->Xinv_			= arma::inv(this->X_);
		// set its value to be the same as well
		this->pfaffianNew_	= this->pfaffian_;
	}
#endif
}

////////////////////////////////////////////////////////////////////////

/**
* @brief Sets the state and the corresponding PP state as well.
* Updates the pfaffian matrix.
* @param _st integer state to be set
* @param _set set the matrices?
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline void RBM_PP<_spinModes, _Ht, _T, _stateType>::setState(u64 _st, bool _set)
{
	RBM_S<_spinModes, _Ht, _T, _stateType>::setState(_st, _set);
#ifdef NQS_ANGLES_UPD
	if (_set)
	{
		// set PP matrix and its corresponding Pfaffian value
		this->setFPP();
		// set Pfaffian value for newly set matrix
		this->setPfaffian();
		// set candidate to be the same for the angles update
		this->Xnew_			= this->X_;
		this->Xinv_			= arma::inv(this->X_);
		// set its value to be the same as well
		this->pfaffianNew_	= this->pfaffian_;
	}
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
inline bool RBM_PP<_spinModes, _Ht, _T, _stateType>::setWeights(std::string _path, std::string _file)
{
	// set for RBM
	if(!RBM_S<_spinModes, _Ht, _T, _stateType>::setWeights(_path, _file))
		return false;

	BEGIN_CATCH_HANDLER
	{
		// Fmat is a vector
		this->Fpp_ = this->Weights_.subvec(this->rbmSize_, this->rbmPPSize_ - 1);
	}
	END_CATCH_HANDLER("Couldn't set the weights for the RBM PP NQS...", return false);
	return true;
}

////////////////////////////////////////////////////////////////////////

/**
* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
* @param _path folder for the weights to be saved onto
* @param _file name of the file to save the weights onto
* @link https://arxiv.org/pdf/1102.3440.pdf - for arxiv reference
* @link https://math.stackexchange.com/questions/4426574/pfaffian-skew-symmetric-using-armadillo - for stackexchange
* @returns whether the load has been successful
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
inline bool RBM_PP<_spinModes, _Ht, _T, _stateType>::saveWeights(std::string _path, std::string _file)
{
	BEGIN_CATCH_HANDLER
	{
		// Fmat is a vector
		this->Weights_.subvec(this->rbmSize_, this->rbmPPSize_ - 1) = this->Fpp_;
		// set the forces vector for the weights
		if(!RBM_S<_spinModes, _Ht, _T, _stateType>::saveWeights(_path, _file))
			return false;
	}
	END_CATCH_HANDLER("Couldn't save the weights for the RBM NQS...", return false);
	return true;
}

// ##########################################################################################################################################