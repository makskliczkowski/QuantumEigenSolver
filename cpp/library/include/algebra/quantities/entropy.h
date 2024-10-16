#pragma once

/*********************************
* Contains the calculators for the
* entanglement entropy. Acts as an
* Operator subclass.
*********************************/

#ifndef HILBERT_H
#include "../hilbert.h"
#endif // !HILBERT_H

#ifndef ENTROPY_H
#define ENTROPY_H

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ SINGLE PARTICLE #############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

namespace SingleParticle 
{
	/*
	* @brief Create single particle correlation matrix for a given Fock state in orbital basis
	* @param _Ns number of lattice sites
	* @param _W_A transformation matrix to quasiparticle operators reduced to subsystem fraction A
	* @param _W_A_CT transformation matrix to quasiparticle operators reduced to subsystem fraction A - hermitian conjguate
	* @param _state a vector of occupations in quasiparticle operators
	* @param _rawRho shall return a raw rho 2*(c+c) correlation, or total correlation matrix ([c+,c])
	* @returns single particle correlation matrix for a single product state in quasiparticle basis
	*/
	template<typename _T1>
	inline arma::Mat<_T1> corrMatrixSingle(
								uint							_Ns,
								const arma::Mat<double>&		_W_A, 
								const arma::Mat<double>&		_W_A_CT,
								const arma::uvec&				_state, 
								bool							_rawRho = false	
		)
	{
		if (!_rawRho)
		{
			auto prefactors				= 2 * VEC::transformIdxToState(_Ns, _state) - 1;
			arma::Mat<_T1> W_A_CT_P		= _W_A_CT;
			for (auto _row = 0; _row < W_A_CT_P.n_rows; ++_row)
				W_A_CT_P.row(_row)		= W_A_CT_P.row(_row) * prefactors;
			return W_A_CT_P * _W_A;
		}
		// raw rho matrix (without delta_ij)
		arma::Mat<double> _J(_W_A_CT.n_rows, _W_A_CT.n_rows, arma::fill::zeros);
		arma::Mat<double> _left		=	_W_A_CT.cols(_state);
		arma::Mat<double> _right	=	_W_A.rows(_state);
		_J							=	(2.0 * _left * _right);
		return _J;
	};

	template<>
	inline arma::Mat<cpx> corrMatrixSingle(	
								uint							_Ns,
								const arma::Mat<double>&		_W_A,
								const arma::Mat<double>&		_W_A_CT,
								const arma::uvec&				_state, 
								bool							_rawRho	
		)
	{
		if (!_rawRho)
		{
			auto prefactors				= 2 * VEC::transformIdxToState(_Ns, _state) - 1;
			arma::Mat<cpx> W_A_CT_P;
			W_A_CT_P.set_real(_W_A_CT);

			for (auto _row = 0; _row < W_A_CT_P.n_rows; ++_row)
				W_A_CT_P.row(_row)		= W_A_CT_P.row(_row) * prefactors;
			return W_A_CT_P * _W_A;
		}
		// raw rho matrix (without delta_ij)
		arma::Mat<cpx> _J(_W_A_CT.n_rows, _W_A_CT.n_rows, arma::fill::zeros);
		arma::Mat<double> _left		= _W_A_CT.cols(_state);
		arma::Mat<double> _right	= _W_A.rows(_state);
		_J.set_real(2.0 * _left * _right);
		return _J;
	};

	template<typename _T1>
	inline arma::Mat<_T1> corrMatrixSingle(
								uint							_Ns,
								const arma::Mat<cpx>&			_W_A, 
								const arma::Mat<cpx>&			_W_A_CT,
								const arma::uvec&				_state, 
								bool							_rawRho = false	
		)
	{
		if (!_rawRho)
		{
			auto prefactors				= 2 * VEC::transformIdxToState(_Ns, _state) - 1;
			arma::Mat<_T1> W_A_CT_P		= _W_A_CT;
			for (auto _row = 0; _row < W_A_CT_P.n_rows; ++_row)
				W_A_CT_P.row(_row)		= W_A_CT_P.row(_row) * prefactors;
			return W_A_CT_P * _W_A;
		}
		// raw rho matrix (without delta_ij)
		arma::Mat<cpx> _left			= _W_A_CT.cols(_state);
		arma::Mat<cpx> _right			= _W_A.rows(_state);
		return 2.0 * _left * _right;
	};

	template<>
	inline arma::Mat<double> corrMatrixSingle(
								uint							_Ns,
								const arma::Mat<cpx>&			_W_A,
								const arma::Mat<cpx>&			_W_A_CT,
								const arma::uvec&				_state,
								bool							_rawRho
	)
	{
		if (!_rawRho)
		{
			auto prefactors					= 2 * VEC::transformIdxToState(_Ns, _state) - 1;
			arma::Mat<cpx> W_A_CT_P			= _W_A_CT;
			for (auto _row = 0; _row < W_A_CT_P.n_rows; ++_row)
				W_A_CT_P.row(_row)			= W_A_CT_P.row(_row) * prefactors;
			return arma::real(W_A_CT_P * _W_A);
		}
		// raw rho matrix (without delta_ij)
		auto _left							= _W_A_CT.cols(_state);
		auto _right							= _W_A.rows(_state);
		return arma::real(2.0 * _left * _right);
	};

	// #######################################################

	/*
	* @brief Create correlation matrix for multiple states
	*/
	template<typename _T1>
	inline arma::Mat<cpx> corrMatrix(
								uint							_Ns,
								const arma::Mat<_T1>&			_W_A, 
								const arma::Mat<_T1>&			_W_A_CT, 
								const v_1d<arma::uvec>&			_states,
								arma::Col<cpx>&					_coeff,
								randomGen&						_gen,
								bool							_rawRho = false
							){
		// get the number of states in the mixture ($\gamma$)
		auto _gamma		=		_states.size();

		if (_gamma == 0)
			throw std::runtime_error(std::string("Cannot create a correlation matrix out of no states, damnit..."));

		// if there is a single state only - go for it!
		if (_gamma == 1)
		{
			if(!_rawRho)
				return SingleParticle::corrMatrixSingle<cpx>(_Ns, _W_A, _W_A_CT, _states[0], true) - DIAG(arma::eye(_W_A.n_cols, _W_A.n_cols));
			else
				return SingleParticle::corrMatrixSingle<cpx>(_Ns, _W_A, _W_A_CT, _states[0], true);
		}

		// check the size of coefficients, otherwise create new if they are bad...
		if(_coeff.n_elem != _gamma)
			_coeff		=		_gen.createRanState(_gamma);
		// save the conjugate coefficients to be quicker
		auto _coeffC	=		arma::conj(_coeff);

		// correlation matrix
		uint La			=		_W_A.n_cols;
		arma::Mat<cpx> J=		_rawRho ? arma::Mat<cpx>(La, La, arma::fill::zeros) : -arma::Mat<cpx>(La, La, arma::fill::eye);

		// ### E Q U A L    P A R T ###
		for (int mi = 0; mi < _gamma; ++mi)
			J			+=		_coeff[mi] * _coeffC[mi] * SingleParticle::corrMatrixSingle<cpx>(_Ns, _W_A, _W_A_CT, _states[mi], true);

		// ### U E Q U L ###
		// go through states <m|
		for (int mi = 0; mi < _gamma; ++mi)
		{
			const auto& _m		=	_states[mi];
			const auto& _mb		=	VEC::transformIdxToBitset(_Ns, _m);
			// go through states |n> (higher than m)
			for (int ni = mi + 1; ni < _gamma; ++ni)
			{
				const auto& _n	=	_states[ni];
				const auto& _nb	=	VEC::transformIdxToBitset(_Ns, _n);

				// xor to check the difference
				auto x			=	_mb ^ _nb;
				auto _counter	=	x.count();

				if (_counter != 2)
					continue;

				v_1d<uint> qs;
				// add position orbitals that are occupied
				x.iterate_bits_on([&](uint _pos) { qs.push_back(_Ns - _pos - 1); });

				// go through occupied orbitals
				std::tuple<uint, uint> qs_nor = {qs[0], qs[1]};
				std::tuple<uint, uint> qs_rev = {qs[1], qs[0]};
				v_1d<std::tuple<uint, uint>> qs_get = { qs_nor, qs_rev };
				for (auto& [q1, q2] : qs_get)
				{
					// q1 has to be occupied in n or occupied in m (if occupied in n then for sure not occupied in m)
					if (!(_n[q1] || _m[q1]))
						continue;

					if (!(_n[q2] || _m[q2]))
						continue;

					auto COEFF			=	_n[q2] ? (2.0 * _coeffC[mi] * _coeff[ni]) : (2.0 * _coeff[mi] * _coeffC[ni]);
					arma::Mat<_T1> Mult	=	(_W_A_CT.col(q1) * _W_A.row(q2));
					J			+=	COEFF * Mult;
				}
			}
		}
		return J;
	};
	
	// #######################################################

};

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

			// ###############################################################

			enum class RHO_METHODS 
			{
				STANDARD,
				STANDARD_CAST,
				SCHMIDT
			};

			// ###############################################################

			/*
			* @brief Calculates the bipartite reduced density matrix of the system via the state mixing
			* @param _s state to construct the density matrix from
			* @param _sizeA subsystem size
			* @param _Ns number of lattice sites
			* @param _Nint number of local fermionic modes
			* @returns the bipartite reduced density matrix
			*/
			template <typename _T>
			inline arma::Mat<_T> redDensMatStandard(const arma::Col<_T>& _s, uint _Ns, uint _sizeA, uint _Nint) {
				// set subsystems size
				auto bitNum		= (uint)std::log2(_Nint);
				const u64 dimA	= ULLPOW(bitNum * _sizeA);
				const u64 dimB	= ULLPOW((_Ns - bitNum * _sizeA));
				const u64 Nh	= dimA * dimB;

				arma::Mat<_T> rho(dimA, dimA, arma::fill::zeros);

				//u64 _startingM	= 0;
				// loop over configurational basis
				for (u64 n = 0; n < Nh; n++) 
				{
					u64 counter = 0;
					// find index of state with same B-side (by dividing the last bits are discarded)
					u64 idx		= n / dimB;
					// pick out state with same B side (last L-A_size bits)
					for (u64 m = n % dimB; m < Nh; m += dimB) 
					//for (u64 m = _startingM; m < Nh; m += dimB)
					{
						rho(idx, counter) += algebra::conjugate(_s(n)) * _s(m);
						// increase counter to move along reduced basis
						counter++;
					}
					//if(++_startingM >= dimB) _startingM = 0;
				}
				return rho;
			};

			// ###############################################################

			/*
			* @brief Calculates the bipartite reduced density matrix of the system via the state mixing. Knowing the mapping with global symmetry.
			* @param _s state to construct the density matrix from
			* @param _sizeA subsystem size
			* @param _hilb used Hilbert space - contains the mapping
			* @returns the bipartite reduced density matrix
			*/
			template <typename _T>
			inline arma::Mat<_T> redDensMatStandard(const arma::Col<_T>& _s, uint _sizeA, const Hilbert::HilbertSpace<_T>& _hilb) {
				// set subsystems size
				uint Ns			= _hilb.getLatticeSize();
				uint Nint		= _hilb.getLocalHilbertSize() * _hilb.getNum();
				uint bitNum		= (uint)std::log2(Nint);
				const u64 dimA	= ULLPOW(bitNum * _sizeA);
				const u64 dimB	= ULLPOW((Ns - bitNum * _sizeA));
				const u64 Nh	= dimA * dimB;
				if (!_hilb.checkGSym())		return		Entropy::Entanglement::Bipartite::redDensMatStandard<_T>(_s, Ns, _sizeA, Nint);

				auto map		= _hilb.getFullMap();
				const u64 N		= map.size();
				// otherwise find in mapping
				auto find_index = [&](u64 _idx) 
					{ 
						return binarySearch(map, 0, Nh - 1, _idx); 
					};

				arma::Mat<_T> rho(dimA, dimA, arma::fill::zeros);
				for (u64 n = 0; n < N; n++) {
					// loop over configurational basis
					u64 ctr = 0;
					u64 true_n = map[n];
					// pick out state with same B side (last L-A_size bits)
					u64 idx = true_n / dimB;
					for (u64 m = true_n % dimB; m < Nh; m += dimB) {
						u64 j	= find_index(m);
						if (j >= 0)
							rho(idx, ctr) += algebra::conjugate(_s(n)) * _s(j);
						// increase counter to move along reduced basis
						ctr++;
					}
				}
				return rho;
			};

			// ###############################################################

			/*
			* @brief Using reshape method to calculate the reduced density matrix
			* @param _s state to construct the density matrix from
			* @param _sizeA subsystem size
			* @param _hilb used Hilbert space - contains the mapping
			* @returns the bipartite reduced density matrix
			*/
			template<typename _T>
			inline arma::Mat<_T> redDensMatSchmidt(const arma::Col<_T>& _s, uint _sizeA, const Hilbert::HilbertSpace<_T>& _hilb) {
				// set subsystems size
				uint Ns			= _hilb.getLatticeSize();
				uint bitNum		= (uint)std::log2(_hilb.getLocalHilbertSize());
				const u64 dimA	= ULLPOW(bitNum * _sizeA);
				const u64 dimB	= ULLPOW(bitNum * (Ns - _sizeA));
				return arma::reshape(_s, dimA, dimB);
			}
			
			// ###############################################################

			/*
			* @brief Calculates the reduced density matrix with one of the methods
			* @param _s state to construct the density matrix from
			* @param _sizeA subsystem size
			* @param _hilb used Hilbert space - contains the mapping
			* @param _ch method choice
			* @returns the bipartite reduced density matrix
			*/
			template <typename _T>
			inline arma::Mat<_T> redDensMat(const arma::Col<_T>& _s, uint _sizeA,
				Hilbert::HilbertSpace<_T>& _hilb,
				RHO_METHODS _ch = RHO_METHODS::SCHMIDT)
			{
				switch (_ch) {
				case RHO_METHODS::STANDARD:
					return redDensMatStandard<_T>(_s, _sizeA, _hilb);
					break;
				case RHO_METHODS::STANDARD_CAST:
					return redDensMatStandard<_T>(_hilb.castToFull(_s), _hilb.getLatticeSize(), _sizeA, _hilb.getNum());
					break;
				case RHO_METHODS::SCHMIDT:
					return redDensMatSchmidt<_T>(_hilb.castToFull(_s), _sizeA, _hilb);
					break;
				default:
					return redDensMatSchmidt<_T>(_hilb.castToFull(_s), _sizeA, _hilb);
					break;
				}
			}

			// ##########################################################################################################################################

			/*
			* @brief Calculates the von Neuman entropy
			* @param _s state to construct the density matrix from
			* @param _sizeA subsystem size
			* @param _hilb used Hilbert space - contains the mapping
			* @param _ch method choice
			* @returns the bipartite entanglement entropy
			*/
			template <typename _T>
			[[nodiscard]]
			double vonNeuman(const arma::Col<_T>& _s, uint _sizeA,
				Hilbert::HilbertSpace<_T>& _hilb,
				RHO_METHODS _ch = RHO_METHODS::SCHMIDT)
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
				// #pragma omp parallel for reduction(+: entropy)
				for (auto i = 0; i < vals.size(); i++)
					entropy += (std::abs(vals(i)) > 0) ? -vals(i) * std::log(vals(i)) : 0.0;
				return entropy;
			};

			// ##########################################################################################################################################
			
			namespace SingleParticle
			{
				template<typename _T>
				inline double vonNeuman(const arma::Mat<_T>& _J)
				{
					// auto Ns			=		_J.n_rows;
					// arma::Mat<_T> eigS;
					arma::Col<_T> eigV;
					// diagonalize
					arma::eig_gen(eigV, _J);
					auto S = 0.0;
					for (auto& eV : eigV)
					{
						auto eVin = std::real(eV);
						S += (eVin > -1.0) ? ((1.0 + eVin) * std::log((1.0 + eVin) / 2.0)) : 0.0;
						S += (eVin < 1.0)  ? ((1.0 - eVin) * std::log((1.0 - eVin) / 2.0)) : 0.0;
					}
					return -0.5 * algebra::real(S);
				}
			}
		};
	};
};


#endif // !ENTROPY_H