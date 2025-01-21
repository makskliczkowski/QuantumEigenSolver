#include "../../../../include/NQS/NQS_ref_base/PP/nqs_pp.h"

// ##########################################################################################################################################
template class NQS_PP<2u, double, double, double>;
template class NQS_PP<3u, double, double, double>;
template class NQS_PP<4u, double, double, double>;

template class NQS_PP<2u, std::complex<double>, std::complex<double>, double>;
template class NQS_PP<3u, std::complex<double>, std::complex<double>, double>;
template class NQS_PP<4u, std::complex<double>, std::complex<double>, double>;

template class NQS_PP<2u, double, std::complex<double>, double>;
template class NQS_PP<3u, double, std::complex<double>, double>;
template class NQS_PP<4u, double, std::complex<double>, double>;

template class NQS_PP<2u, std::complex<double>, double, double>;
template class NQS_PP<3u, std::complex<double>, double, double>;
template class NQS_PP<4u, std::complex<double>, double, double>;
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
template<uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::NQS_PP(Hamil_t_p _H,
            uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta)
	: NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>(_H, _nHid, _lr, _threadNum, _nPart, _lower, _beta)
{
	this->nPP_					= this->spinSectors_.size() * this->info_p_.nSitesSquared_;                     // number of particles squared
	this->PPsize_			    = this->nPP_;
	this->info_p_.fullSize_		= NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::size() + this->PPsize_; // full size of the NQS
	this->allocate();
	this->setInfo();
}
// template instantiation of the function above
template NQS_PP<2u, double, double, double>::NQS_PP(Hamil_t_p _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template NQS_PP<2u, std::complex<double>, std::complex<double>, double>::NQS_PP(Hamil_t_p _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template NQS_PP<2u, double, std::complex<double>, double>::NQS_PP(Hamil_t_p _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template NQS_PP<2u, std::complex<double>, double, double>::NQS_PP(Hamil_t_p _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template NQS_PP<4u, double, double, double>::NQS_PP(Hamil_t_p _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template NQS_PP<4u, std::complex<double>, std::complex<double>, double>::NQS_PP(Hamil_t_p _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template NQS_PP<4u, double, std::complex<double>, double>::NQS_PP(Hamil_t_p _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
template NQS_PP<4u, std::complex<double>, double, double>::NQS_PP(Hamil_t_p _H, uint _nHid, double _lr, uint _threadNum, int _nPart, const NQSLS_p& _lower, std::vector<double> _beta);
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
