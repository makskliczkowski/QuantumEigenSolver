#include "./nqs_pp.h"

namespace NQS_NS
{

	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
	requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::NQS_PP(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p)
		: NQS_ref_t(_p)	
	{
		this->init();
		this->nPP_					= this->spinSectors_.size() * this->info_p_.nSitesSquared_;
		this->PPsize_			    = this->nPP_;
		this->info_p_.fullSize_		= NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::size() + this->PPsize_;
		this->allocate();
		this->setInfo();
	}

	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::NQS_PP(const NQS_Const_par_t<_spinModes, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
		: NQS_ref_t(_p, _lower, _beta)	
	{ 
		this->init();
		this->nPP_					= this->spinSectors_.size() * this->info_p_.nSitesSquared_;
		this->PPsize_			    = this->nPP_;
		this->info_p_.fullSize_		= NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::size() + this->PPsize_;
		this->allocate();
		this->setInfo();
	};

	// ##########################################################################################################################################

	/**
	* @brief Copy constructor for the NQS_PP class.
	*
	* This constructor initializes a new instance of the NQS_PP class by copying 
	* the data from an existing instance.
	*
	* @tparam _spinModes Number of spin modes.
	* @tparam _Ht Hamiltonian type.
	* @tparam _T Data type.
	* @tparam _stateType State type.
	* @tparam _CorrState Correlation state type.
	* @param _n The instance of NQS_PP to copy from.
	*/
	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
	requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::NQS_PP(const NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>& _n)
		: NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>(_n)
	{
		this->pp_weights_	= _n.pp_weights_;
		this->PPsize_		= _n.PPsize_;
		this->nPP_			= _n.nPP_;
		this->spinSectors_	= _n.spinSectors_;
	}

	// ##########################################################################################################################################

	/**
	* @brief Move constructor for the NQS_PP class.
	*
	* This constructor initializes an NQS_PP object by moving the resources from another
	* NQS_PP object. It calls the move constructor of the base class NQS_ref and then
	* moves the member variables specific to NQS_PP.
	*
	* @tparam _spinModes The number of spin modes.
	* @tparam _Ht The Hamiltonian type.
	* @tparam _T The data type for the weights.
	* @tparam _stateType The type of the state.
	* @tparam _CorrState The type of the correlated state.
	* 
	* @param _other The NQS_PP object to move from.
	*/
	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::NQS_PP(NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>&& _other)
		: NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>(std::move(_other))
	{
		this->pp_weights_	= std::move(_other.pp_weights_);
		this->PPsize_		= _other.PPsize_;
		this->nPP_			= _other.nPP_;
		this->spinSectors_	= std::move(_other.spinSectors_);
	}

	// ##########################################################################################################################################

	/**
	* @brief Clones the current NQS_PP object from another instance.
	*
	* This function attempts to clone the current NQS_PP object from another instance
	* of the same type. It uses dynamic_pointer_cast to ensure the type safety of the
	* cast. If the cast is successful, it copies the internal state from the other
	* instance. If the cast fails, it catches the bad_cast exception and prints an
	* error message. It also catches any other standard exceptions and prints an error
	* message.
	*
	* @tparam _spinModes The number of spin modes.
	* @tparam _Ht The Hamiltonian type.
	* @tparam _T The data type.
	* @tparam _stateType The state type.
	* @tparam _CorrState The correlated state type.
	* @param _other A shared pointer to the other instance to clone from.
	*/
	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::clone(MC_t_p _other)
	{
		try
		{
			auto _n = std::dynamic_pointer_cast<NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>>(_other);
			if (_n)
			{
				this->pp_weights_	= _n->pp_weights_;
				this->PPsize_		= _n->PPsize_;
				this->nPP_			= _n->nPP_;
				this->spinSectors_	= _n->spinSectors_;
			}
		}
		catch (std::bad_cast & e)
		{
			std::cerr << "Error in cloning the NQS PP object: " << e.what() << std::endl;
		}
		catch (std::exception& e)
		{
			std::cerr << "Error in cloning the NQS PP object: " << e.what() << std::endl;
		}
		NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::clone(_other);
	}

	// ##########################################################################################################################################

	/**
	* @brief Sets the information string for the NQS_PP object.
	*
	* This function formats and sets the `info_` member variable of the NQS_PP object.
	* The formatted string includes the information from the base class NQS_ref and
	* the number of PP (nPP) specific to this object.
	*
	* @tparam _spinModes The number of spin modes.
	* @tparam _Ht The Hamiltonian type.
	* @tparam _T The data type.
	* @tparam _stateType The state type.
	* @tparam _CorrState The correlated state type.
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::setInfo()
	{
		NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::setInfo();
		this->info_ += std::format("nPP={}", this->nPP_);	
	}

	// ##########################################################################################################################################

	/**
	* @brief Allocates memory for various matrices and weights used in the NQS_PP class.
	* 
	* This function performs the following allocations:
	* - Calls the allocate function of the base class NQS_ref to allocate base class weights.
	* - Allocates the X matrices (X_ and X_inv) with dimensions based on the number of particles.
	* - Allocates the F_r1r2_s1s2_ matrix with dimensions based on the number of particle pairs.
	* 
	* @tparam _spinModes Number of spin modes.
	* @tparam _Ht Hamiltonian type.
	* @tparam _T Data type.
	* @tparam _stateType State type.
	* @tparam _CorrState Correlation state type.
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::allocate()
	{
		// !TODO implement changable number of fermions
		// allocate weights
		// matrix for each step
		NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::allocate();

		// allocate the X matices
		this->pp_weights_.X_			= NQSW(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);
		this->pp_weights_.X_inv			= NQSW(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);
		// !TODO get rid of zeros? 
		// !TODO make this matrix symmetric?
		this->pp_weights_.F_r1r2_s1s2_ 	= NQSB(this->nPP_, arma::fill::zeros);
	// #ifdef NQS_NOT_OMP_MT
		// this->XTmp_		= NQSW(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);
		// for (int _thread = 0; _thread < this->threads_.threadNum_; _thread++)
			// this->XTmp_(this->threads_.threads_[_thread].get_id()) = NQSW(this->info_p_.nParticles_, this->info_p_.nParticles_, arma::fill::zeros);
	// #endif
	}

	// ##########################################################################################################################################

	/**
	* @brief Macro to instantiate the allocate function for all combinations of template parameters.
	* 
	* This macro is used to generate the necessary instantiations of the allocate function for all 
	* possible combinations of template parameters. It ensures that the allocate function is available 
	* for different configurations of the NQS_PP class template.
	* 
	* @param allocate The name of the function to be instantiated.
	* @param void The return type of the function.
	* @param () The parameter list of the function.
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::init()
	{
		// ######################################################################################################################################
		NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::init();
		// ######################################################################################################################################
		const double std_dev = 1.0 / std::sqrt(this->nPP_);
		// ######################################################################################################################################
		
		// matrix for the PP wave function - contains all the necessary weights
		// is initialized according to the distance between the sites
		this->pp_weights_.F_r1r2_s1s2_ 	= NQSB(this->nPP_, arma::fill::zeros);
		auto _lat 						= this->H_->getLat();

		// go through the lattice
		for (uint i = 0; i < this->info_p_.nSites_; i++)
		{
			// go through the lattice
			for (uint j = 0; j < this->info_p_.nSites_; j++)	
			{
				// get the distance between the sites
				const auto distance = _lat ? _lat->getSiteDistance(i, j) : std::abs<int>(i - j);

				// go through the spin sectors and initialize the weights
				for (const auto& spinSec : this->spinSectors_)
				{
					const auto _index 						= this->getFPPIndex(spinSec[0], spinSec[1], i, j);
					const auto _value 						= algebra::cast<_T>(this->ran_->template randomNormal<double>(0.0, std_dev) + I * this->ran_->template randomNormal<double>(0.0, std_dev));
					this->pp_weights_.F_r1r2_s1s2_(_index) 	= _value;
					if (distance != 0)
						this->pp_weights_.F_r1r2_s1s2_(_index) /= distance * distance;
				}
			}
		}
		// !TODO: Implement this in a more general way - set the weights in the Weights_ vector to the PP weights
		// set the weights in the Weights_ vector to the PP weights 
		this->Weights_.subvec(NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::size(), this->PPsize_ - 1) = this->pp_weights_.F_r1r2_s1s2_;

		// initialize local variables
		this->pp_weights_.pfaffian_ 		= this->getPfaffian();
		// this->pp_weights_.pfaffianNew_		= this->pp_weights_.pfaffian_;
		// this->pp_weights_.pfaffianNewLog_ 	= this->pp_weights_.pfaffianLog_;
		// other static variables
		// this->pp_weights_.X_upd_			= {};
		// this->pp_weights_.states_upd_		= {};
		this->setX();
	}

	// ##########################################################################################################################################

	// ############################################################## A N S A T Z ###############################################################

	// ##########################################################################################################################################

	/**
	* @brief Computes the ansatz for the given configuration.
	*
	* This function computes the ansatz by combining the correlation part from the base class
	* and the Pfaffian of the given configuration.
	*
	* @tparam _spinModes Number of spin modes.
	* @tparam _Ht Hamiltonian type.
	* @tparam _T Return type.
	* @tparam _stateType State type.
	* @tparam _CorrState Correlation state type.
	* @param _in The input configuration.
	* @return The computed ansatz as the product of the correlation part and the Pfaffian.
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::ansatz(Config_cr_t _in) const
	{
		const auto _FX = NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::ansatz(_in); // get the ansatz from the base class - correlation part
		const auto _R  = this->getPfaffian(_in); 											// get the Pfaffian
		return _FX * _R; 																	// return the product of the two
	};

	// ##########################################################################################################################################

	/**
	* @brief Computes the logarithm of the ansatz for a given configuration.
	*
	* This function calculates the logarithm of the ansatz by combining the correlation part
	* from the base class and the logarithm of the Pfaffian of the input configuration.
	*
	* @tparam _spinModes The number of spin modes.
	* @tparam _Ht The Hamiltonian type.
	* @tparam _T The data type used for calculations.
	* @tparam _stateType The type representing the state.
	* @tparam _CorrState The type representing the correlated state.
	* @param _in The input configuration for which the ansatz logarithm is computed.
	* @return The logarithm of the ansatz for the given configuration.
	*/
	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::ansatzlog(Config_cr_t _in) const
	{
		const auto _FX = NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::ansatzlog(_in); 	// get the ansatz from the base class - correlation part
		const auto _R  = this->getPfaffianLog(_in); 									 		// get the Pfaffian logarithm
		return _FX + _R; 																		// return the sum of the two
	}

	// ##########################################################################################################################################

	/**
	* @brief Calculates the ratio of the two NQS-PP states - used for calculating the excited states (_other->ansatz / this->ansatz)
	* @param _in Vector to calculate the ratio for
	* @param _other Pointer to the other NQS to calculate the ratio with
	* @return Ratio of the two states (other / this) for a given state _in (vector)
	* @note The ratio is calculated as: _other->ansatz / this->ansatz * _other->getPfaffian(_in) / this->getPfaffian(_in)
	*/
	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::ansatz_ratiolog(Config_cr_t _in, NQS<_spinModes, _Ht, _T, _stateType>* _other) const
	{
		auto _nqs_pp_other 	= dynamic_cast<NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>*>(_other);
		auto _FX_ratio 		= NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::ansatz_ratiolog(_in, _other); 	// get the ratio of the correlation part
		auto _R_ratio 		= _nqs_pp_other->getPfaffianLog(_in) - this->getPfaffianLog(_in); 						// get the ratio of the Pfaffian
		return _FX_ratio + _R_ratio; 																				// return the sum of the two
	}

	// ##########################################################################################################################################

	/**
	* @brief Computes the index for the F matrix based on spin modes and site indices.
	*
	* This function calculates the index for the F matrix (FPP) based on the given spin indices
	* (_spini and _spinj) and site indices (ri and rj). The calculation is dependent on the number
	* of spin modes (_spinModes). Currently, the function supports 2 spin modes.
	*
	* @tparam _spinModes Number of spin modes (currently supports 2).
	* @tparam _Ht Hamiltonian type.
	* @tparam _T Type of the elements in the state.
	* @tparam _stateType Type of the state.
	* @tparam _CorrState Type of the correlated state.
	* @param _spini Spin index for the first site.
	* @param _spinj Spin index for the second site.
	* @param ri Row index for the first site.
	* @param rj Row index for the second site.
	* @return The computed index for the F matrix (FPP).
	*
	* @note The function currently supports only 2 spin modes. For other spin modes, the function
	*       returns 0 and needs to be implemented.
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	u64 NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::getFPPIndex(int _spini, int _spinj, uint ri, uint rj) const
	{
		if constexpr (_spinModes == 2)
		{
			if (_spini)
			{
				if (_spinj)
					return ri * this->info_p_.nSites_ + rj;
				else
					return this->info_p_.nSitesSquared_ + ri * this->info_p_.nSitesSquared_ + rj;
			}
			else
			{
				if (_spinj)
					return 2 * this->info_p_.nSitesSquared_ + ri * this->info_p_.nSites_ + rj;
				else
					return 3 * this->info_p_.nSitesSquared_ + ri * this->info_p_.nSites_ + rj;
			}
		}
		else	// !TODO: implement for 4 spin modes
			return 0;
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
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::setState(const Config_t& _st, bool _set)
	{
		NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::setState(_st, _set);
		if (_set)
		{
			// the state is already changed in the base class
			this->setX();							// set the X matrix for a given state
			this->setPfaffian();					// set Pfaffian value for newly set matrix given the 
		}
	}

	// ##########################################################################################################################################

	/**
	* @brief Sets the state and the corresponding PP state as well.
	* Updates the pfaffian matrix.
	* @param _st integer state to be set
	* @param _set set the matrices?
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::setState(u64 _st, bool _set)
	{
		NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::setState(_st, _set);
		if (_set)
		{
			// the state is already changed in the base class
			this->setX();							// set the X matrix for a given state
			this->setPfaffian();					// set Pfaffian value for newly set matrix
		}
	}

	// ##########################################################################################################################################

	/**
	* @brief Macro to instantiate a member function template for all combinations of template parameters.
	*
	* This macro is used to instantiate the member function template `setState` for all combinations of 
	* template parameters. The function `setState` is defined with the following signature:
	* 
	* @param Config_t& Configuration object reference.
	* @param bool A boolean flag.
	* 
	* The macro ensures that the function is instantiated for all possible combinations of the template 
	* parameters, allowing for flexibility and reusability of the code.
	*/
	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::setX(Config_cr_t _st)
	{
		this->pp_weights_.X_	= this->calculateX(_st);											// first it calculates the X matrix
		// !TODO make more stable - check if the matrix is invertible also it is skew-symmetric (faster)
	#ifdef NQS_USE_ARMA
		this->pp_weights_.X_inv = arma::inv(this->pp_weights_.X_);									// then it calculates the inverse of the X matrix
	#else
	#endif
	}

	// ##########################################################################################################################################

	/**
	* @brief Sets the Pfaffian and its logarithm for the given configuration state.
	* 
	* This function calculates the Pfaffian of the provided configuration state
	* and stores it in the `pp_weights_` member. It also computes the natural
	* logarithm of the Pfaffian and stores it in the `pp_weights_` member.
	* 
	* @tparam _spinModes The number of spin modes.
	* @tparam _Ht The Hamiltonian type.
	* @tparam _T The data type for numerical values.
	* @tparam _stateType The type representing the state.
	* @tparam _CorrState The type representing the correlated state.
	* 
	* @param _st The configuration state for which the Pfaffian is to be set.
	*/
	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::setPfaffian()
	{
		this->pp_weights_.pfaffian_ 	= this->getPfaffian(); 	// get the Pfaffian for the current state
		this->pp_weights_.pfaffianLog_ 	= std::log(this->pp_weights_.pfaffian_);
	}

	// ##########################################################################################################################################

	/**
	* @brief Macro to instantiate the setPfaffian function template for all combinations of template parameters.
	* 
	* This macro is used to generate all necessary instantiations of the setPfaffian function template
	* with different combinations of template parameters. It ensures that the function is available
	* for various types and configurations used in the codebase.
	* 
	* @param setPfaffian The name of the function template to instantiate.
	* @param inline void The return type of the function template.
	* @param () The parameter list of the function template.
	*/
	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::getPfaffian(Config_cr_t _st) const
	{
		return algebra::Pfaffian::pfaffian(this->calculateX(_st), this->info_p_.nParticles_);
	}

	// ##########################################################################################################################################

	template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline _T NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::getPfaffian() const 
	{
		return algebra::Pfaffian::pfaffian(this->pp_weights_.X_, this->info_p_.nParticles_);
	}

	// ##########################################################################################################################################

	/**
	* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
	* @param _path folder for the weights to be saved onto
	* @param _file name of the file to save the weights onto
	* @returns whether the load has been successful
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	bool NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::setWeights(std::string _path, std::string _file)
	{
		if(!NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::setWeights(_path, _file))
			return false;

		BEGIN_CATCH_HANDLER
		{
			const auto _prevsize = NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::size();
			this->pp_weights_.F_r1r2_s1s2_ = this->Weights_.subvec(_prevsize, this->PPsize_ - 1);
		}
		END_CATCH_HANDLER("Couldn't set the weights for the RBM PP NQS...", return false);
		return true;
	}

	// ##########################################################################################################################################

	/**
	* @brief After reading the weights from the path specified by the user, it sets the inner vectors from them.
	* @param _path folder for the weights to be saved onto
	* @param _file name of the file to save the weights onto
	* @link https://arxiv.org/pdf/1102.3440.pdf - for arxiv reference
	* @link https://math.stackexchange.com/questions/4426574/pfaffian-skew-symmetric-using-armadillo - for stackexchange
	* @returns whether the load has been successful
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	bool NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::saveWeights(std::string _path, std::string _file)
	{
		BEGIN_CATCH_HANDLER
		{
			const auto _prevsize = NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::size();
			this->Weights_.subvec(_prevsize, this->PPsize_ - 1) = this->pp_weights_.F_r1r2_s1s2_;
			if(!NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::saveWeights(_path, _file))
				return false;
		}
		END_CATCH_HANDLER("Couldn't save the weights for the RBM NQS...", return false);
		return true;
	}

	// ##########################################################################################################################################

	/**
	* @brief Updates the weights in the system according to a given gradient
	* @warning uses forces vector (member of NQS : dF_) to update the gradients - preallocation for optimization
	* @note the function is called after the gradient is calculated and inlined to the optimization process
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::updateWeights()
	{
		NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::updateWeights();
		this->setWeights();
	}

	// ##########################################################################################################################################

	/**
	* @brief Sets up the post-processing weights (F_r1r2_s1s2_) from the full weight vector.
	* 
	* This function extracts a subvector from the full weight vector (Weights_) starting from
	* the size of the base class weights up to PPsize_ - 1 and assigns it to the post-processing weights (F_r1r2_s1s2_).
	* The post-processing weights are used for additional transformations after the main NQS computation.
	*/
template <uint _spinModes, typename _Ht, typename _T, class _stateType, class _CorrState>
requires NQS_S_concept<_spinModes, _Ht, _T, _stateType, _CorrState>
	inline void NQS_PP<_spinModes, _Ht, _T, _stateType, _CorrState>::setWeights()
	{
		const auto _prevsize = NQS_ref<_spinModes, _Ht, _T, _stateType, _CorrState>::size();
		this->pp_weights_.F_r1r2_s1s2_ = this->Weights_.subvec(_prevsize, this->PPsize_ - 1);
	}

	// ##########################################################################################################################################

}; // namespace NQS_NS