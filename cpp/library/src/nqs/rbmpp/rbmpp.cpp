#include "../../../include/NQS/rbm_pp_final.hpp"

template class RBM_PP<2u, double, double, double>;
template class RBM_PP<2u, cpx, cpx, double>;
template class RBM_PP<2u, double, cpx, double>;
template class RBM_PP<2u, cpx, double, double>;

// template class RBM_PP<3u, double, double, double>;
// template class RBM_PP<3u, cpx, cpx, double>;
// template class RBM_PP<3u, double, cpx, double>;
// template class RBM_PP<3u, cpx, double, double>;

// template class RBM_PP<4u, double, double, double>;
// template class RBM_PP<4u, cpx, cpx, double>;
// template class RBM_PP<4u, double, cpx, double>;
// template class RBM_PP<4u, cpx, double, double>;

// ##########################################################################################################################################

/**
* @brief Constructor for the RBM_PP class.
* 
* This constructor initializes an instance of the RBM_PP class, which is a derived class of RBM_S. 
* It sets up the spin sectors, calculates various sizes related to the problem, and allocates necessary resources.
* 
* @tparam _spinModes Number of spin modes.
* @tparam _Ht Type of the Hamiltonian.
* @tparam _T Data type used in the calculations.
* @tparam _stateType Type of the state.
* 
* @param _H Shared pointer to the Hamiltonian object.
* @param _nHid Number of hidden units.
* @param _lr Learning rate.
* @param _threadNum Number of threads to be used.
* @param _nPart Number of particles.
* @param _lower Lower bound for the NQSLS_p object.
* @param _beta Vector of beta values.
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM_PP<_spinModes, _Ht, _T, _stateType>::RBM_PP(std::shared_ptr<Hamiltonian<_Ht, _spinModes>> _H, uint _nHid, 
							double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta)
	: RBM_S<_spinModes, _Ht, _T, _stateType>(_H, _nHid, _lr, _threadNum, _nPart, _lower, _beta)
{
	// create the spin sectors
	spinSectors_.push_back({ 1, 1 });
	spinSectors_.push_back({ 1, 0 });
	spinSectors_.push_back({ 0, 1 });
	spinSectors_.push_back({ 0, 0 });

	// !TODO make this changable
	this->nSites2_				= this->info_p_.nSites_ * this->info_p_.nSites_;
	this->nParticles2_			= this->info_p_.nParticles_ * this->info_p_.nParticles_;
	this->nPP_					= this->spinSectors_.size() * this->nSites2_; // for both spin channels
	this->rbmPPSize_			= this->rbmSize_ + this->nPP_;
	this->info_p_.fullSize_		= this->rbmPPSize_;
	this->allocate();
	this->setInfo();
}
// template instantiation of the function above
template RBM_PP<2u, double, double, double>::RBM_PP(std::shared_ptr<Hamiltonian<double, 2u>> _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template RBM_PP<2u, cpx, cpx, double>::RBM_PP(std::shared_ptr<Hamiltonian<cpx, 2u>> _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template RBM_PP<2u, double, cpx, double>::RBM_PP(std::shared_ptr<Hamiltonian<double, 2u>> _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template RBM_PP<2u, cpx, double, double>::RBM_PP(std::shared_ptr<Hamiltonian<cpx, 2u>> _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template RBM_PP<4u, double, double, double>::RBM_PP(std::shared_ptr<Hamiltonian<double, 4u>> _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template RBM_PP<4u, cpx, cpx, double>::RBM_PP(std::shared_ptr<Hamiltonian<cpx, 4u>> _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template RBM_PP<4u, double, cpx, double>::RBM_PP(std::shared_ptr<Hamiltonian<double, 4u>> _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template RBM_PP<4u, cpx, double, double>::RBM_PP(std::shared_ptr<Hamiltonian<cpx, 4u>> _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM_PP<_spinModes, _Ht, _T, _stateType>::RBM_PP(const RBM_PP<_spinModes, _Ht, _T, _stateType>& _other)
	: RBM_S<_spinModes, _Ht, _T, _stateType>(_other) 
{
	this->X_			= _other.X_;
	this->Xinv_			= _other.Xinv_;
	this->XinvSkew_		= _other.XinvSkew_;
	this->Xnew_			= _other.Xnew_;
	this->pfaffian_		= _other.pfaffian_;
	this->pfaffianNew_	= _other.pfaffianNew_;
	this->rbmPPSize_	= _other.rbmPPSize_;
	this->nPP_			= _other.nPP_;
	this->nParticles2_	= _other.nParticles2_;
	this->spinSectors_	= _other.spinSectors_;
}
// template instantiation of the function above
template RBM_PP<2u, double, double, double>::RBM_PP(const RBM_PP<2u, double, double, double>&);
template RBM_PP<2u, cpx, cpx, double>::RBM_PP(const RBM_PP<2u, cpx, cpx, double>&);
template RBM_PP<2u, double, cpx, double>::RBM_PP(const RBM_PP<2u, double, cpx, double>&);
template RBM_PP<2u, cpx, double, double>::RBM_PP(const RBM_PP<2u, cpx, double, double>&);
template RBM_PP<4u, double, double, double>::RBM_PP(const RBM_PP<4u, double, double, double>&);
template RBM_PP<4u, cpx, cpx, double>::RBM_PP(const RBM_PP<4u, cpx, cpx, double>&);
template RBM_PP<4u, double, cpx, double>::RBM_PP(const RBM_PP<4u, double, cpx, double>&);
template RBM_PP<4u, cpx, double, double>::RBM_PP(const RBM_PP<4u, cpx, double, double>&);
// ##########################################################################################################################################

template <uint _spinModes, typename _Ht, typename _T, class _stateType>
RBM_PP<_spinModes, _Ht, _T, _stateType>::RBM_PP(RBM_PP<_spinModes, _Ht, _T, _stateType>&& _other)
	: RBM_S<_spinModes, _Ht, _T, _stateType>(std::move(_other))
{
	this->X_			= std::move(_other.X_);
	this->Xinv_			= std::move(_other.Xinv_);
	this->XinvSkew_		= std::move(_other.XinvSkew_);
	this->Xnew_			= std::move(_other.Xnew_);
	this->pfaffian_		= std::move(_other.pfaffian_);
	this->pfaffianNew_	= std::move(_other.pfaffianNew_);
	this->rbmPPSize_	= _other.rbmPPSize_;
	this->nPP_			= _other.nPP_;
	this->nParticles2_	= _other.nParticles2_;
	this->spinSectors_	= std::move(_other.spinSectors_);
}
// template instantiation of the function above
template RBM_PP<2u, double, double, double>::RBM_PP(RBM_PP<2u, double, double, double>&&);
template RBM_PP<2u, cpx, cpx, double>::RBM_PP(RBM_PP<2u, cpx, cpx, double>&&);
template RBM_PP<2u, double, cpx, double>::RBM_PP(RBM_PP<2u, double, cpx, double>&&);
template RBM_PP<2u, cpx, double, double>::RBM_PP(RBM_PP<2u, cpx, double, double>&&);
template RBM_PP<4u, double, double, double>::RBM_PP(RBM_PP<4u, double, double, double>&&);
template RBM_PP<4u, cpx, cpx, double>::RBM_PP(RBM_PP<4u, cpx, cpx, double>&&);
template RBM_PP<4u, double, cpx, double>::RBM_PP(RBM_PP<4u, double, cpx, double>&&);
template RBM_PP<4u, cpx, double, double>::RBM_PP(RBM_PP<4u, cpx, double, double>&&);
// ##########################################################################################################################################

/**
* @brief Clones the current RBM_PP object from another instance.
*
* This function attempts to clone the current RBM_PP object by copying the 
* internal state from another instance of the same type. If the dynamic cast 
* to the appropriate type fails, an error message is printed to the standard 
* error stream.
*
* @tparam _spinModes The number of spin modes.
* @tparam _Ht The Hamiltonian type.
* @tparam _T The data type used for computations.
* @tparam _stateType The type representing the state.
* @param _other A shared pointer to another Monte Carlo object to clone from.
*
* @throws std::exception If an error occurs during the cloning process.
* @throws std::bad_cast If the dynamic cast to the appropriate type fails.
*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM_PP<_spinModes, _Ht, _T, _stateType>::clone(MC_t_p _other)
{
	try
	{
		auto _n = std::dynamic_pointer_cast<RBM_PP<_spinModes, _Ht, _T, _stateType>>(_other);
		if (_n)
		{
			this->X_			= _n->X_;
			this->Xinv_			= _n->Xinv_;
			this->XinvSkew_		= _n->XinvSkew_;
			this->Xnew_			= _n->Xnew_;
			this->pfaffian_		= _n->pfaffian_;
			this->pfaffianNew_	= _n->pfaffianNew_;
			this->rbmPPSize_	= _n->rbmPPSize_;
			this->nPP_			= _n->nPP_;
			this->nParticles2_	= _n->nParticles2_;
			this->spinSectors_	= _n->spinSectors_;
		}
	}
	catch (std::bad_cast & e)
	{
		std::cerr << "Error in cloning the RBM PP object: " << e.what() << std::endl;
	}
	catch (std::exception& e)
	{
		std::cerr << "Error in cloning the RBM PP object: " << e.what() << std::endl;
	}

	// clone the base class
	RBM<_spinModes, _Ht, _T, _stateType>::clone(_other);
}
// template instantiation of the function above
RBM_PP_INST_CMB_ALL(clone, void, (MC_t_p), );

// ##########################################################################################################################################

// ############################################################ G R A D I E N T #############################################################

// ##########################################################################################################################################

/**
* @brief At each step calculates the variational derivatives and stores them in the _derivatives matrix.
* @param _v vector to calculate the derivatives for
* @param _plc row at which to store the derivatives
* !TODO - not half filling / not spins
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM_PP<_spinModes, _Ht, _T, _stateType>::grad(Config_cr_t _v, uint _plc)
{
	// calculate the RBM derivative
	RBM_S<_spinModes, _Ht, _T, _stateType>::grad(_v, _plc);
	// calculate the derivative of the Pffafian 
	// as 1/Pf(X) * \partial Pf(X) / \partial x = 1/2 Tr(X^{-1} \partial X / \partial x)
#ifndef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	// calculate the inverse only if necessary (without the updates at each step).
	this->Xinv_				= arma::inv(this->X_);
#endif // !NQS_RBM_PP_USE_PFAFFIAN_UPDATE

	// not multiplying by pfaffian, as it's dividied by it later in the definition of the derivative
	this->XinvSkew_			= (this->Xinv_.st() - this->Xinv_);
	//this->XinvSkew_			= (this->Xinv_ - this->Xinv_.st());
	auto _currDerivative	= this->derivatives_.row(_plc).subvec(this->rbmSize_, this->rbmPPSize_ - 1);

	// find the correct indices on the values that can be updated
	_currDerivative.zeros();

	int _spinIter [[maybe_unused]] = 0;
//#ifndef _DEBUG
//#pragma omp parallel for num_threads(this->threads_.threadNum_)
//#endif
	for (const auto& s: this->spinSectors_)
	{
		for (uint i = 0; i < this->info_p_.nParticles_; ++i)
		{
			// if left spin differs from the current left spin, just continue
			if (checkBit(this->curVec_, i) != s[0])
				continue;

			for (uint j = 0; j < this->info_p_.nParticles_; ++j)
			{
				// skip the diagonal part
				if (j == i)
					continue;

				// if right spin differs from the current right spin, just continue
				if (checkBit(this->curVec_, j) != s[1])
					continue;

				_currDerivative(this->getFPPIndex(s[0], s[1], i, j)) = this->Xinv_(i, j);
			}
		}
		_spinIter += this->nParticles2_;
	}
}

// template instantiation of the function above
RBM_PP_INST_CMB_ALL(grad, void, (Config_cr_t, uint), );

// ##########################################################################################################################################

// ############################################################ U P D A T E R S #############################################################

// ##########################################################################################################################################

////////////////////////////////////////////////////////////////////////
#ifdef NQS_ANGLES_UPD

/**
* @brief Update angles with the flipped spin (spins)
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM_PP<_spinModes, _Ht, _T, _stateType>::update(uint nFlips)
{
	// update the angles in the RBM
	RBM_S<_spinModes, _Ht, _T, _stateType>::update(nFlips);
	//for (uint i = 0; i < nFlips; i++)
	//{
	//	// update the Pffafian matrix for the candidate
	//	const auto fP = this->flipPlaces_[i];
	//	const auto fV = this->flipVals_[i];
	//	this->updFPP(fP, fV);
	//}
	// update the Pffafian matrix
	// as the candidate pfaffian shall be already updated, use it instead of calculating everything all the time (probably not as efficient)
	// replace updating the pfaffian back
#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	for(uint i = 0; i < nFlips; i++)
		this->Xinv_	= algebra::scherman_morrison_skew(this->Xinv_, this->flipPlaces_[i], this->Xnew_.row(this->flipPlaces_[i]));
#endif
	this->X_		= this->Xnew_;
	this->pfaffian_ = this->pfaffianNew_;
}

// template instantiation of the function above
RBM_PP_INST_CMB_ALL(update, void, (uint), );

////////////////////////////////////////////////////////////////////////

/**
* @brief If a new pfaffian state has not been accepted, unupdate the values
* @param nFlips number of flips to be used
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM_PP<_spinModes, _Ht, _T, _stateType>::unupdate(uint nFlips)
{
	RBM_S<_spinModes, _Ht, _T, _stateType>::unupdate(nFlips);
	// unupdate the Pffafian matrix
	// as the candidate pfaffian shall be already updated, use it instead of calculating everything all the time (probably not as efficient)
	// replace updating the pfaffian back
	this->Xnew_			= this->X_;
	this->pfaffianNew_	= this->pfaffian_;
}

// template instantiation of the function above
RBM_PP_INST_CMB_ALL(unupdate, void, (uint), );

////////////////////////////////////////////////////////////////////////

/**
* @brief Update angles with the flipped spin (spins)
* @param vector after the flips has been done
* @param nFlips number of flips to be used
* @warning values are stored in flipVals_ before the flip, hence "-" is taken
*/
template<uint _spinModes, typename _Ht, typename _T, class _stateType>
void RBM_PP<_spinModes, _Ht, _T, _stateType>::update(Config_cr_t v, uint nFlips)
{
	// update the angles in the RBM
	RBM_S<_spinModes, _Ht, _T, _stateType>::update(v, nFlips);
	//for (uint i = 0; i < nFlips; i++)
	//{
	//	// update the Pffafian matrix for the candidate
	//	const auto fP = this->flipPlaces_[i];
	//	this->updFPP(fP, v(fP));
	//}
#ifdef NQS_RBM_PP_USE_PFAFFIAN_UPDATE
	for(uint i = 0; i < nFlips; i++)
		this->Xinv_	= algebra::scherman_morrison_skew(this->Xinv_, this->flipPlaces_[i], this->Xnew_.row(this->flipPlaces_[i]));
#endif
	this->X_		= this->Xnew_;
	this->pfaffian_ = this->pfaffianNew_;
}
#endif

// template instantiation of the class above
RBM_PP_INST_CMB_ALL(update, void, (Config_cr_t, uint), );

////////////////////////////////////////////////////////////////////////////////////