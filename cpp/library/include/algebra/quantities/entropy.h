#pragma once

/*********************************
* Contains the calculators for the
* entanglement entropy. Acts as an
*********************************/

#ifndef ENTROPY_H
#define ENTROPY_H

#ifndef DENSITY_MATRIX_H
#	include "density_matrix.h"
#endif
#include <cmath>

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# E N T R O P Y ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

namespace Entropy 
{
	namespace Entanglement 
	{
		namespace Bipartite 
		{
			using namespace DensityMatrix;
			// ##########################################################################################################################################

			/*
			* @brief Calculates the von Neuman entropy
			* @param _s state to construct the density matrix from
			* @param _sizeA subsystem size
			* @param _hilb used Hilbert space - contains the mapping
			* @param _ch method choice
			* @returns the bipartite entanglement entropy
			*/
			template <typename _T, typename _Ht>
			[[nodiscard]]
			double vonNeuman(const arma::Col<_T>& _s,
							 uint _sizeA,
							 Hilbert::HilbertSpace<_Ht>& _hilb,
							 DensityMatrix::RHO_METHODS _ch = DensityMatrix::RHO_METHODS::SCHMIDT)
			{
				// get the reduced density matrix
				auto rho = redDensMat<_T>(_s, _sizeA, _hilb, _ch);
				// get the values
				arma::vec vals;
				if (_ch == RHO_METHODS::SCHMIDT)
				{
					vals = arma::svd(rho);
					vals = arma::square(vals);
				}
				else
					arma::eig_sym(vals, rho);

				// calculate entropy
				double entropy = 0.0;
				
				// go through the values
				for (const auto& val: vals)
					if (val > 0.0)
						entropy += -val * std::log(val);
				return entropy;
			};

			/*
			* @brief Calculates the von Neuman entropy
			* @param _s			state to construct the density matrix from
			* @param _sizeA		subsystem size
			* @param _hilb		size of the Hilbert space in many body system
			* @param _ch		method choice
			* @returns the bipartite entanglement entropy
			*/
			template <typename _T>
			[[nodiscard]]
			double vonNeuman(const arma::Col<_T>& _s,
							uint _sizeA,
							uint _Ns,
							DensityMatrix::RHO_METHODS _ch	= DensityMatrix::RHO_METHODS::SCHMIDT,
							uint _localHilbert				= 2)
			{
				// get the reduced density matrix
				auto rho = redDensMat<_T>(_s, _sizeA, _Ns, _ch, _localHilbert);

				// get the values
				arma::vec vals;
				if (_ch == RHO_METHODS::SCHMIDT)
				{
					vals = arma::svd(rho);
					vals = arma::square(vals);
				}
				else
					arma::eig_sym(vals, rho);

				// calculate entropy
				double entropy = 0.0;

				// go through the values
				for (const auto& val: vals)
					if (val > 0.0)
						entropy += -val * std::log(val);
				return entropy;
			};

			// ##########################################################################################################################################

			template <typename _T, typename _TV>
			[[nodiscard]]
			inline typename std::enable_if<std::is_arithmetic<_TV>::value, double>::type
			vonNeuman(const arma::Col<_T>& _s,
					uint _sizeA,
					size_t _Ns,
					const _TV& _maskA,
					DensityMatrix::RHO_METHODS _ch = DensityMatrix::RHO_METHODS::SCHMIDT,
					uint _locHilbert = 2)
			{
				// get the reduced density matrix
				auto rho = redDensMat<_T>(_s, _sizeA, _Ns, _maskA, _ch, _locHilbert);
				// get the values
				arma::vec vals;
				if (_ch == RHO_METHODS::SCHMIDT)
				{
					vals = arma::svd(rho);
					vals = arma::square(vals);
				}
				else
					arma::eig_sym(vals, rho);

				// calculate entropy
				double entropy = 0.0;
				
				// go through the values
				for (const auto& val: vals)
					if (val > 0.0)
						entropy += -val * std::log(val);
				return entropy;
			};

			// ##########################################################################################################################################
			
			// ###################################################### S I N G L E   P A R T I C L E #####################################################
			
			// ##########################################################################################################################################
			
			namespace SingleParticle
			{
				/*
				* @brief For the single particle states, the von Neuman entropy is defined withing the 
				* single particle correlation matrix
				* @param _J single particle correlation matrix
				* @returns the von Neuman entropy
				*/
				template<typename _T>
				inline double vonNeuman(const arma::Mat<_T>& _J)
				{
					arma::Col<_T> eigV;
					// diagonalize
					arma::eig_gen(eigV, _J);

					double S = 0.0;
					for (const auto& eV : eigV)
					{
						const double eVin = std::real(eV);
						if(eVin > -1.0)
							S += (1.0 + eVin) * (std::log1p(eVin) - LOG_TWO);
						if(eVin < 1.0)
							S += (1.0 - eVin) * (std::log1p(-eVin) - LOG_TWO);
					}
					return -0.5 * S;
				}
			}
		};
	};
};


#endif // !ENTROPY_H