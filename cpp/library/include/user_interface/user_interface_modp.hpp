/**
* @brief Structure representing various model parameters and configurations.
* 
* This structure contains parameters for different models such as Ising, XYZ, Kitaev, Heisenberg, QSM, 
* Rosenzweig-Porter, Ultrametric, Aubry-Andre, and Power Law Random Bandwidth. It also includes methods 
* for resizing vectors and setting default values.
* 
* Members:
* - modTyp: Model type.
* - modRanN: Number of random states.
* - modRanSeed: Seed for the random number generator.
* - modRanNIdx: Index of the random state.
* - eth: ETH related parameters.
* - modMidStates: States in the middle of the spectrum.
* - modEnDiff: Tolerance for the energy difference of the states in offdiagonal.
* - operators: Operators to be calculated for the model.
* 
* Ising Model Parameters:
* - J1: Spin exchange.
* - hz: Perpendicular field.
* - hx: Transverse field.
* 
* XYZ Model Parameters:
* - J2: Next nearest neighbors exchange.
* - eta1: Eta1 parameter.
* - eta2: Eta2 parameter.
* - dlt1: Delta1 parameter.
* - dlt2: Delta2 parameter.
* 
* Kitaev Model Parameters:
* - Kx_: Kx parameter.
* - Ky_: Ky parameter.
* - Kz_: Kz parameter.
* 
* Heisenberg Model Parameters:
* - heiJ_: J parameter.
* - heiDlt_: Delta parameter.
* - heiHx_: Hx parameter.
* - heiHz_: Hz parameter.
* 
* QSM Model Parameters:
* - qsm: QSM model parameters.
* 
* Rosenzweig-Porter Model Parameters:
* - rosenzweig_porter: Rosenzweig-Porter model parameters.
* 
* Ultrametric Model Parameters:
* - ultrametric: Ultrametric model parameters.
* 
* Quadratic Model Parameters:
* - q_gamma: Mixing quadratic states parameter.
* - q_manifold: Use the degenerate manifold flag.
* - q_manybody: Use the many body calculation flag.
* - q_randomCombNum: Number of random combinations for the average.
* - q_realizationNum: Number of realizations for the average.
* - q_shuffle: Shuffle the states flag.
* - q_broad: Broadening for spectral function.
* 
* Aubry-Andre Model Parameters:
* - aubry_andre: Aubry-Andre model parameters.
* 
* Power Law Random Bandwidth Model Parameters:
* - power_law_random_bandwidth: Power Law Random Bandwidth model parameters.
* 
* Methods:
* - resizeKitaev: Resizes the Kitaev model vectors.
* - resizeHeisenberg: Resizes the Heisenberg model vectors.
* - resizeQSM: Resizes the QSM model vectors.
* - resizeRP: Resizes the Rosenzweig-Porter model vectors.
* - resizeUM: Resizes the Ultrametric model vectors.
* - getRanReal(uint i): Returns the random realization at index i.
* - getRanReal(): Returns the current random realization.
* - setDefault(): Sets default values for the parameters.
* - checkComplex(): Checks whether the model itself is complex.
*/

#ifndef USER_INTERFACE_PARAMS_MOD_HPP
#define USER_INTERFACE_PARAMS_MOD_HPP

#include "./user_interface_include.hpp"

namespace UI_PARAMS
{
	// ----------------------------------------------------------------

	struct ModP 
	{
		// ################################## TYPE ##################################
		
		UI_PARAM_CREATE_DEFAULT(modTyp, MY_MODELS, MY_MODELS::ISING_M);
		UI_PARAM_CREATE_DEFAULTV(modRanN, uint);			// number of random states
		UI_PARAM_CREATE_DEFAULT(modRanSeed, u64, 0);		// seed for the random number generator
		UI_PARAM_CREATE_DEFAULT(modRanNIdx, uint, 0);		// index of the random state
		
		// eth related
		UI_PARAM_CREATE_DEFAULT(eth_entro, bool, false);
		UI_PARAM_CREATE_DEFAULT(eth_susc, bool, true);
		UI_PARAM_CREATE_DEFAULT(eth_ipr, bool, true);
		UI_PARAM_CREATE_DEFAULT(eth_offd, bool, false);
		UI_PARAM_CREATE_DEFAULTV(eth_end, double);

		UI_PARAM_CREATE_DEFAULTD(modMidStates, double, 1.0);// states in the middle of the spectrum
		UI_PARAM_CREATE_DEFAULTD(modEnDiff, double, 1.0);	// tolerance for the energy difference of the states in offdiagonal
		std::vector<std::string> operators;					// operators to be calculated for the model
		
		UI_PARAM_CREATE_DEFAULT(Ntimes, uint, 100000);		// number of time steps for the time evolution
		
		//! ########################## I N T E R A C T I N G #########################
		
		// ############## ISING ################
		
		UI_PARAM_STEP(double, J, 1.0);			// spin exchange
		UI_PARAM_STEP(double, hz, 1.0);			// perpendicular field
		UI_PARAM_STEP(double, hx, 1.0);			// transverse field

		// ############### XYZ #################		
		
		UI_PARAM_STEP(double, J2, 2.0);			// next nearest neighbors exchange
		UI_PARAM_STEP(double, eta1, 0.5);
		UI_PARAM_STEP(double, eta2, 0.5);
		UI_PARAM_STEP(double, dlt1, 0.3);
		UI_PARAM_STEP(double, dlt2, 0.3);

		// ############# KITAEV ################

		v_1d<double> Kx_;
		v_1d<double> Ky_;
		v_1d<double> Kz_;
		void resizeKitaev(const size_t Ns)
		{
			this->Kx_.resize(Ns);
			this->Ky_.resize(Ns);
			this->Kz_.resize(Ns);
		};

		// ########### HEISENBERG ##############

		v_1d<double> heiJ_;
		v_1d<double> heiDlt_;
		v_1d<double> heiHx_;
		v_1d<double> heiHz_;

		void resizeHeisenberg(const size_t Ns)
		{
			this->heiJ_.resize(Ns);
			this->heiDlt_.resize(Ns);
			this->heiHx_.resize(Ns);
			this->heiHz_.resize(Ns);
		};

		// ############### QSM #################

		struct qsm_t
		{
			UI_PARAM_CREATE_DEFAULTD(qsm_N, size_t, 1);
			UI_PARAM_CREATE_DEFAULTD(qsm_Ntot, size_t, 1);
			UI_PARAM_CREATE_DEFAULTD(qsm_gamma, double, 1.0);
			UI_PARAM_CREATE_DEFAULTD(qsm_g0, double, 1.0);

			UI_PARAM_CREATE_DEFAULTV(qsm_alpha, double);
			UI_PARAM_CREATE_DEFAULTV(qsm_xi, double);
			UI_PARAM_CREATE_DEFAULTV(qsm_h, double);
			void resizeQSM()
			{
				auto _N = this->qsm_Ntot_ - this->qsm_N_;
				if (_N < 0)
					return;
				this->qsm_alpha_r_ = 0;
				this->qsm_alpha_.resize(this->qsm_Ntot_ - this->qsm_N_);
				this->qsm_xi_r_ = 0;
				this->qsm_xi_.resize(this->qsm_Ntot_ - this->qsm_N_);
				this->qsm_h_r_ = 0;
				this->qsm_h_.resize(_N);
			};
		} qsm;

		// ######### ROSENZWEIG PORTER #########

		struct rosenzweig_porter_t
		{
			UI_PARAM_CREATE_DEFAULTV(rp_g, double);
			UI_PARAM_CREATE_DEFAULTD(rp_single_particle, bool, 0);
			UI_PARAM_CREATE_DEFAULTD(rp_be_real, bool, 1);
			UI_PARAM_CREATE_DEFAULTD(rp_g_sweep_n, int, 1);

			void resizeRP()
			{
				this->rp_g_.resize(this->rp_g_sweep_n_);
			};

		} rosenzweig_porter;

		// ############ ULTRAMETRIC ############

		struct ultrametric_t
		{
			UI_PARAM_CREATE_DEFAULTD(um_N, size_t, 1);
			UI_PARAM_CREATE_DEFAULTD(um_Ntot, size_t, 1);
			UI_PARAM_CREATE_DEFAULTV(um_alpha, double);
			UI_PARAM_CREATE_DEFAULTD(um_g, double, 1.0);

			void resizeUM()
			{
				auto _N = this->um_Ntot_ - this->um_N_;
				if (_N < 0)
					return;
				this->um_alpha_r_ = 0;
				this->um_alpha_.resize(_N);
			};
		} ultrametric;

		// #####################################
		// ######### Q U A D R A T I C #########
		// #####################################
		
		// for simulation
		UI_PARAM_CREATE_DEFAULT(q_gamma, uint, 1);					// mixing quadratic states
		UI_PARAM_CREATE_DEFAULT(q_manifold, bool, false);			// use the degenerate manifold?
		UI_PARAM_CREATE_DEFAULT(q_manybody, bool, true);			// use the many body calculation?
		UI_PARAM_CREATE_DEFAULT(q_randomCombNum, uint, 100);		// number of random combinations for the average (to choose from)
		UI_PARAM_CREATE_DEFAULT(q_realizationNum, uint, 100);		// number of realizations for the average
		UI_PARAM_CREATE_DEFAULT(q_shuffle, bool, true);				// shuffle the states?
		UI_PARAM_CREATE_DEFAULTD(q_broad, double, 0.1);				// broadening for spectral function

		// ########### AUBRY_ANDRE ############
		
		struct aubry_andre_t
		{
			UI_PARAM_STEP(double, aa_J, 1.0);						// hopping
			UI_PARAM_STEP(double, aa_lambda, 0.5);					// modulation strength
			UI_PARAM_STEP(double, aa_beta, (1 + std::sqrt(5)) / 2);	// phase multiplication
			UI_PARAM_STEP(double, aa_phi, 1.0);						// phase addition
		} aubry_andre;

		// ############ POWER LAW ##############

		struct power_law_random_bandwidth_t
		{
			UI_PARAM_CREATE_DEFAULTV(plrb_a, double);
			// UI_PARAM_CREATE_DEFAULTD(plrb_a, double, 1.0);
			UI_PARAM_CREATE_DEFAULTD(plrb_b, double, 1.0);
			UI_PARAM_CREATE_DEFAULTD(plrb_mb, bool, false);

		} power_law_random_bandwidth;

		// -------------------------------------

		// #####################################

		// -------------------------------------
		uint getRanReal(uint i) const { return i < this->modRanN_.size() ? this->modRanN_[i] : this->modRanN_[this->modRanN_.size()-1]; 	};
		uint getRanReal() 		const { return this->modRanNIdx_ < this->modRanN_.size() ? this->modRanN_[this->modRanNIdx_] : this->modRanN_[this->modRanN_.size()-1]; };

		void setDefault() 
		{
			UI_PARAM_SET_DEFAULT(modTyp);

			// -------------------------------------
			// default operators
			this->operators = {"sz/L", "sz/1"};
			this->modRanN_ = { 1 };

			// -------------------------------------
			// SPIN
			{
				// ising
				{
					UI_PARAM_SET_DEFAULT_STEP(J);
					UI_PARAM_SET_DEFAULT_STEP(hz);
					UI_PARAM_SET_DEFAULT_STEP(hx);
				}
				// xyz
				{
					UI_PARAM_SET_DEFAULT_STEP(J2);
					UI_PARAM_SET_DEFAULT_STEP(eta1);
					UI_PARAM_SET_DEFAULT_STEP(eta2);
					UI_PARAM_SET_DEFAULT_STEP(dlt1);
					UI_PARAM_SET_DEFAULT_STEP(dlt2);
				}
				// kitaev
				{
					this->Kx_		= v_1d<double>(1, 1.0);
					this->Ky_		= v_1d<double>(1, 1.0);
					this->Kz_		= v_1d<double>(1, 1.0);
					this->heiJ_		= v_1d<double>(1, 1.0);
					this->heiDlt_	= v_1d<double>(1, 1.0);
					this->heiHz_	= v_1d<double>(1, 1.0);
					this->heiHx_	= v_1d<double>(1, 1.0);
				}
				// QSM
				{
					UI_PARAM_SET_DEFAULT_STRUCT(qsm, qsm_gamma);
					UI_PARAM_SET_DEFAULT_STRUCT(qsm, qsm_g0);
					UI_PARAM_SET_DEFAULT_STRUCT(qsm, qsm_Ntot);
					UI_PARAM_SET_DEFAULT_STRUCT(qsm, qsm_N);
					this->qsm.qsm_alpha_	= v_1d<double>(1, 1.0);
					this->qsm.qsm_xi_		= v_1d<double>(1, 1.0);
					this->qsm.qsm_h_		= v_1d<double>(1, 1.0);
				}
				// Rosenzweig-Porter
				{
					this->rosenzweig_porter.rp_g_ = v_1d<double>(1, 1.0);
				}
			}

			// -------------------------------------
			
			// QUADRATIC
			{
				// aubry-andre
				{
					UI_PARAM_SET_DEFAULT_STRUCT(aubry_andre, aa_J);
					UI_PARAM_SET_DEFAULT_STRUCT(aubry_andre, aa_lambda);
					UI_PARAM_SET_DEFAULT_STRUCT(aubry_andre, aa_beta);
					UI_PARAM_SET_DEFAULT_STRUCT(aubry_andre, aa_phi);
				}
			}
		}

		// -------------------------------------

		/**
		* @brief Check whether the model itself is complex...
		*/
		bool checkComplex() const
		{
			if (this->modTyp_ == MY_MODELS::FREE_FERMIONS_M)
				return true;
			return false;
		}
	};
};

// ##########################################################################################################################################

namespace HamiltonianHelpers
{
	// ############################################################################

	/**
	* @brief Estimates the number of sites (ns) based on the input value nh and the local dimension nh_loc.
	*
	* This function computes an estimate of the number of sites by taking the logarithm (base nh_loc) of nh.
	* By default, nh_loc is set to 2, corresponding to a binary system. For nh_loc values of 2 or 4, optimized
	* calculations are performed. For other values, the logarithm is computed with respect to nh_loc.
	*
	* @param nh The total number of possible sites (must be greater than 0).
	* @param nh_loc The local dimension (default is 2). Common values are 2 (binary) or 4 (quaternary).
	* @return The estimated number of sites as a long integer.
	*/
	inline long get_ns_est(u64 nh, int nh_loc = 2)
	{
		int ns = 0;
		if (nh_loc == 2)
			ns = std::log2(nh);
		else if (nh_loc == 4)
			ns = std::log2(nh) / std::log2(4);
		else
			ns = std::log2(nh) / std::log2(nh_loc);
		return ns;
	}

	// ############################################################################
	
	/**
	* @brief Estimates the Heisenberg time for a given quantum model.
	*
	* This function computes an estimate of the Heisenberg time based on the model type,
	* a model parameter, the Hilbert space dimension, and optionally the local Hilbert space dimension.
	*
	* @param mod_typ   The type of quantum model (enumerated in MY_MODELS).
	* @param param     A model-specific parameter (e.g., disorder strength, coupling constant).
	* @param nh        The Hilbert space dimension.
	* @param nh_loc    The local Hilbert space dimension (default is 2).
	* @return          Estimated Heisenberg time as a long double.
	*/
	inline long double get_heisenberg_time_est(MY_MODELS mod_typ, double param, u64 nh, int nh_loc = 2)
	{
		int ns = get_ns_est(nh, nh_loc);
		switch (mod_typ)
		{
		// interacting models
		case MY_MODELS::ISING_M:
		case MY_MODELS::XYZ_M:
		case MY_MODELS::HEI_KIT_M:
		case MY_MODELS::QSM_M:
		case MY_MODELS::RP_M:
		// return RP_types::RP_default::getMeanLvlSpacing(param, ns);
		case MY_MODELS::POWER_LAW_RANDOM_BANDED_M:
		// return 1.0l / PRLB_types::PRLB_default::getMMeanLvlSpacing(param, ns);
			return (long double)nh;
		case MY_MODELS::ULTRAMETRIC_M:
            return 1.0l / Ultrametric_types::UM_default::getMeanLvlSpacing(param, ns);
		default:
			return (long double)nh;
		}
	}

	// ############################################################################

	/**
	* @brief Estimates the bandwidth for a given model type and parameters.
	*
	* This function computes an estimated bandwidth based on the specified model type (`mod_typ`),
	* a model parameter (`param`), the Hilbert space dimension (`nh`), and an optional local Hilbert space dimension (`nh_loc`).
	* For certain model types, the bandwidth is simply the Hilbert space dimension.
	* For others, it is computed using model-specific static methods.
	*
	* @param mod_typ   The model type (enumeration of MY_MODELS).
	* @param param     Model-specific parameter used in bandwidth estimation.
	* @param nh        The Hilbert space dimension.
	* @param nh_loc    The local Hilbert space dimension (default is 2).
	* @return          Estimated bandwidth as a long double.
	*/
	inline long double get_bandwidth_est(MY_MODELS mod_typ, double param, u64 nh, int nh_loc = 2)
	{
		int ns = get_ns_est(nh, nh_loc);
		switch (mod_typ)
		{
		case MY_MODELS::ISING_M:
		case MY_MODELS::XYZ_M:
		case MY_MODELS::HEI_KIT_M:
		case MY_MODELS::QSM_M:
			return (long double)nh;
		case MY_MODELS::RP_M:
			return RP_types::RP_default::getBandwidth(param, ns);
		case MY_MODELS::POWER_LAW_RANDOM_BANDED_M:
			return PRLB_types::PRLB_default::getBandwidth(param, ns);
		case MY_MODELS::ULTRAMETRIC_M:
			return Ultrametric_types::UM_default::getBandwidth(param, ns);
		default:
			return nh;
		}
	}

	// ############################################################################

	/**
	* @brief Estimates the energy width for a given model type and parameters.
	*
	* This function computes an estimate of the energy width based on the specified model type,
	* a model parameter, the Hilbert space dimension, and an optional local Hilbert space dimension.
	* The estimation method depends on the model type:
	*   - For ISING_M, XYZ_M, HEI_KIT_M, and QSM_M models, the function returns the Hilbert space dimension.
	*   - For RP_M, POWER_LAW_RANDOM_BANDED_M, and ULTRAMETRIC_M models, the function calls the corresponding
	*     static getVariance method from the appropriate type, passing the model parameter and estimated system size.
	*   - For unknown model types, the function returns the Hilbert space dimension.
	*
	* @param mod_typ   The model type (enumeration of MY_MODELS).
	* @param param     Model-specific parameter (e.g., disorder strength).
	* @param nh        Hilbert space dimension.
	* @param nh_loc    Local Hilbert space dimension (default is 2).
	* @return Estimated energy width as a long double.
	*/
	inline long double get_energy_width_est(MY_MODELS mod_typ, double param, u64 nh, int nh_loc = 2)
	{
		int ns = get_ns_est(nh, nh_loc);
		switch (mod_typ)
		{
		case MY_MODELS::ISING_M:
		case MY_MODELS::XYZ_M:
		case MY_MODELS::HEI_KIT_M:
		case MY_MODELS::QSM_M:
			return (long double)nh;
		case MY_MODELS::RP_M:
			return RP_types::RP_default::getVariance(param, ns);
		case MY_MODELS::POWER_LAW_RANDOM_BANDED_M:
			return PRLB_types::PRLB_default::getVariance(param, ns);
		case MY_MODELS::ULTRAMETRIC_M:
			return Ultrametric_types::UM_default::getVariance(param, ns);
		default:
			return nh;
		}
	}

	// ############################################################################

	inline long double get_thouless_est(MY_MODELS mod_typ, double param, u64 nh, int nh_loc = 2)
	{
		int ns = get_ns_est(nh, nh_loc);
		switch (mod_typ)
		{
		case MY_MODELS::ISING_M:
		case MY_MODELS::XYZ_M:
		case MY_MODELS::HEI_KIT_M:
		case MY_MODELS::QSM_M:
		case MY_MODELS::RP_M:
		case MY_MODELS::POWER_LAW_RANDOM_BANDED_M:
			return (long double)nh;
		case MY_MODELS::ULTRAMETRIC_M:
			return Ultrametric_types::UM_default::getThouless(param, ns);
		default:
			return (long double)nh;
		}
	}

	// ############################################################################
}

#endif