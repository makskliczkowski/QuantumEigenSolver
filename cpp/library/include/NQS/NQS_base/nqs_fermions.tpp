#include "./nqs_spins.tpp"

//////////////////////////////////////////////////////////////////////////////////////////

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FERMIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

#ifndef NQS_4_H
#define NQS_4_H

namespace NQS_NS
{
	/*
	* @brief Template NQS for Fermions
	*/
	template <  typename _Ht, typename _T, class _stateType>
	class NQS_S<4, _Ht, _T, _stateType> : public NQS<4, _Ht, _T, _stateType>
	{
		// **********************************************************************************************************************************
		MCS_PUBLIC_TYPES(_T, _stateType, MCS_CONTAINER);				// type definitions for the Monte Carlo solver
		NQS_PUBLIC_TYPES(_T, _stateType);
		NQS_HAMIL_TYPES(_Ht, 4);										// type definitions for the Hamiltonian
		using NQSLS_p =	typename NQS<4, _Ht, _T, _stateType>::NQSLS_p;
		// **********************************************************************************************************************************
	public:
		virtual ~NQS_S() 	= default;
		NQS_S() 			= default;
		NQS_S(const NQS_Const_par_t<4, _Ht, _T, _stateType>& _p)
		{ 
			this->init();
		};
		NQS_S(const NQS_Const_par_t<4, _Ht, _T, _stateType>& _p, const NQSLS_p& _lower, const std::vector<double>& _beta)
		{ 
			this->init(); 
		};
		// **********************************************************************************************************************************
	protected:
		// --------------------------- F L I P S ---------------------------
		virtual void chooseRandomFlips() override;

		// apply flips to the temporary vector or the current vector according the template
		virtual void applyFlipsT() override								{ LOG_ERROR("NOT IMPLEMENTED FOR FERMIONS YET"); };
		virtual void applyFlipsC() override								{ LOG_ERROR("NOT IMPLEMENTED FOR FERMIONS YET"); };			
		virtual void setRandomFlipNum(uint _nFlips) override;

		////////////////////////////////////////////////////////////////////

		virtual auto clone() const -> MC_t_p override = 0;
		virtual auto clone(MC_t_p _n) -> void override { NQS<4, _Ht, _T, _stateType>::clone(_n); };
		////////////////////////////////////////////////////////////////////
	};

	// !!!!!!!!!!!!!!!!!! F L I P S !!!!!!!!!!!!!!!!!!

	/*
	* @brief Randomly flip the discrete variables at chosen flip places. Sets the random flips to the vector already saved.
	*/
	template<typename _Ht, typename _T, class _stateType>
	inline void NQS_S<4, _Ht, _T, _stateType>::chooseRandomFlips()
	{
		LOG_ERROR("NOT IMPLEMENTED FOR FERMIONS YET");
	}

	/*
	* @brief Set the number of random flips.
	* @param _nFlips number of flips to be used
	*/
	template<typename _Ht, typename _T, class _stateType>
	inline void NQS_S<4, _Ht, _T, _stateType>::setRandomFlipNum(uint _nFlips)
	{
		LOG_ERROR("NOT IMPLEMENTED FOR FERMIONS YET");
	}
};

#endif