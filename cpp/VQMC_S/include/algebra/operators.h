#ifndef OPERATORS_H
#define OPERATORS_H

#ifndef SYMMETRIES_H
#include "symmetries.h"
#endif // !SYMMETRIES_H

namespace Operators 
{
	constexpr double _SPIN			=		0.5;
	constexpr double _SPIN_RBM		=		0.5;

	// ##########################################################################################################################################

	std::pair<u64, double> sigma_x(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<double> makeSigmaX(std::shared_ptr<Lattice>& lat, uint site); 

	std::pair<u64, cpx> sigma_y(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<cpx> makeSigmaY(std::shared_ptr<Lattice>& lat, uint site);

	std::pair<u64, double> sigma_z(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<double> makeSigmaZ(std::shared_ptr<Lattice>& lat, uint site);

	// ##########################################################################################################################################
};

//class avOperators {
//public:
//	std::string lat_type = "";
//	int Ns = 1;
//	int Lx = 1;
//	int Ly = 1;
//	int Lz = 1;
//
//
//	// sigma z
//	double s_z = 0.0;
//
//	// correlation with z neighbor
//	double s_z_nei = 0.0;
//	//v_3d<double> s_z_cor;
//	mat s_z_cor;
//	vec s_z_i;
//
//	// sigma x
//	cpx s_x = 0.0;
//	// correlation with x neighbor
//	double s_x_nei = 0.0;
//	//v_3d<double> s_x_cor;
//	mat s_x_cor;
//	cx_vec s_x_i;
//
//	// sigma y
//	cpx s_y = 0.0;
//	// correlation with x neighbor
//	cpx s_y_nei = 0.0;
//	//v_3d<double> s_x_cor;
//	//mat s_y_cor;
//	//cx_vec s_y_i;
//
//
//	// entropy
//	vec ent_entro;
//
//	// energy
//	cpx en = 0.0;
//
//	avOperators() = default;
//	avOperators(int Lx, int Ly, int Lz, int Ns, std::string lat_type)
//		: Lx(Lx), Ly(Ly), Lz(Lz), Ns(Ns), lat_type(lat_type)
//	{
//		//v_3d<double> corr_vec;
//		//if (lat_type == "square") {
//		//	corr_vec = SPACE_VEC_D(Lx, Ly, Lz);
//		//}
//		//else if (lat_type == "hexagonal") {
//		//	corr_vec = SPACE_VEC_D(Lx, 2 * Ly, Lz);
//		//}
//
//		this->s_z_cor = mat(Ns, Ns, arma::fill::zeros);
//		this->s_z_i = arma::vec(Ns, arma::fill::zeros);
//		this->s_x_cor = mat(Ns, Ns, arma::fill::zeros);
//		this->s_x_i = arma::cx_vec(Ns, arma::fill::zeros);
//		this->ent_entro = arma::vec(Ns - 1, arma::fill::zeros);
//	};
//
//	void reset() {
//		//v_3d<double> corr_vec;
//		//if (lat_type == "square") {
//		//	corr_vec = SPACE_VEC_D(Lx, Ly, Lz);
//		//}
//		//else if (lat_type == "hexagonal") {
//		//	corr_vec = SPACE_VEC_D(Lx, 2 * Ly, Lz);
//		//}
//		this->s_x = 0.0;
//		this->s_x_nei = 0.0;
//		this->s_y = 0.0;
//		this->s_y_nei = 0.0;
//		this->s_z = 0.0;
//		this->s_z_nei = 0.0;
//
//		this->s_z_cor = mat(Ns, Ns, arma::fill::zeros);
//		this->s_z_i = arma::vec(Ns, arma::fill::zeros);
//		this->s_x_cor = mat(Ns, Ns, arma::fill::zeros);
//		this->s_x_i = arma::cx_vec(Ns, arma::fill::zeros);
//		this->ent_entro = arma::vec(Ns - 1, arma::fill::zeros);
//	};
//
//	void normalise(u64 norm, const v_3d<int>& spatialNorm) {
//		this->s_z /= double(norm);
//		this->s_y /= double(norm);
//		this->s_x /= double(norm);
//		this->s_z_i /= double(norm);
//		this->s_x_i /= double(norm);
//		this->s_z_nei /= double(norm);
//		this->s_x_nei /= double(norm);
//		this->s_y_nei /= double(norm);
//
//		this->s_x_cor /= double(norm);
//		this->s_z_cor /= double(norm);
//
//		//for (int i = 0; i < this->s_x_cor.size(); i++) {
//		//	for (int j = 0; j < this->s_x_cor[i].size(); j++) {
//		//		for (int k = 0; k < this->s_x_cor[i][j].size(); k++) {
//		//			this->s_x_cor[i][j][k] /= spatialNorm[i][j][k] * norm;
//		//			this->s_z_cor[i][j][k] /= spatialNorm[i][j][k] * norm;
//		//		}
//		//	}
//		//}
//		this->en /= double(norm);
//	};
//};

namespace Entropy {
	namespace Entanglement {
		namespace Bipartite {
			
			enum RHO_METHODS {
				STANDARD,
				STANDARD_CAST,
				SCHMIDT
			};

			template <typename _T>
			arma::Mat<_T> redDensMat(const arma::Col<_T>& _s, uint _sizeA, 
				const Hilbert::HilbertSpace<_T>& _hilb, RHO_METHODS _ch = RHO_METHODS::SCHMIDT) 
			{
				switch (_ch) {
				case RHO_METHODS::STANDARD:
					return redDensMatStandard<_T>(_s, _sizeA, _hilb);
					break;
				case RHO_METHODS::STANDARD_CAST:
					if (!_hilb.checkGSym())
						return redDensMatStandard(_s, _sizeA, _hilb);
					else
						return redDensMatStandard(castToFullHilbert(_s, _hilb), _hilb.getLatticeSize(), _sizeA, _hilb.getNum());
					break;
				case RHO_METHODS::SCHMIDT:
					return redDensMatSchmidt(...)
					break;
				default:
					return redDensMatSchmidt(...)
					break;
				}
			}

			/*
			* @brief Calculates the bipartite reduced density matrix of the system via the state mixing
			* @param _s state to construct the density matrix from
			* @param _sizeA subsystem size
			* @param _Ns number of lattice sites
			* @param _Nint number of local fermionic modes
			* @returns the bipartite reduced density matrix
			*/
			template <typename _T>
			arma::Mat<_T> redDensMatStandard(const arma::Col<_T>& _s, uint _Ns, uint _sizeA, uint _Nint) {
				// set subsystems size
				int bitNum					=			std::log2(_Nint);
				const u64 dimA				=			ULLPOW(bitNum * _sizeA);
				const u64 dimB				=			ULLPOW((_Ns - bitNum * A_size));
				const u64 Nh				=			dimA * dimB;

				arma::Mat<_T> rho(dimA, dimA, arma::fill::zeros);
				// loop over configurational basis
				for (u64 n = 0; n < Nh; n++) {
					u64 counter				=			0;
					// pick out state with same B side (last L-A_size bits)
					for (u64 m = n % dimB; m < Nh; m += dimB) {
						// find index of state with same B-side (by dividing the last bits are discarded)
						u64 idx				=			n / dimB;
						rho(idx, counter)	+=			algebra::conjugate(state(n)) * state(m);
						// increase counter to move along reduced basis
						counter++;
					}
				}
				return rho;
			};

			/*
			* @brief Calculates the bipartite reduced density matrix of the system via the state mixing. Knowing the mapping with global symmetry.
			* @param _s state to construct the density matrix from
			* @param _sizeA subsystem size
			* @param _hilb used Hilbert space - contains the mapping
			* @returns the bipartite reduced density matrix
			*/
			template <typename _T>
			arma::Mat<_T> redDensMatStandard(const arma::Col<_T>& _s, uint _sizeA, const Hilbert::HilbertSpace<_T>& _hilb) {
				// set subsystems size
				uint Ns						=			_hilb.getLatticeSize();
				uint Nint					=			_hilb.getNum();
				uint bitNum					=			std::log2(Nint);
				const u64 dimA				=			ULLPOW(bitNum * _sizeA);
				const u64 dimB				=			ULLPOW((Ns - bitNum * A_size));
				const u64 Nh				=			dimA * dimB;
				if (!_hilb.checkGSym())		return		Entropy::Entanglement::Bipartite::redDensMatStandard<_T>(_s, Ns, _sizeA, Nint);

				const u64 N					=			_hilb.getFullHilbertSize();
				auto map					=			_hilb.getFullMap();
				// otherwise find in mapping
				auto find_index				=			[&](u64 _idx) { return binarySearch(map, 0, Nh - 1, _idx); };

				arma::Mat<_T> rho(dimA, dimA, arma::fill::zeros);
				for (u64 n = 0; n < N; n++) {
					// loop over configurational basis
					u64 counter				=			0;
					u64 true_n				=			map[n];
					for (u64 m = true_n % dimB; m < Nh; m += dimB) {
						// pick out state with same B side (last L-A_size bits)
						u64 idx				=			true_n / dimB;
						u64 j				=			find_index(m);
						if (j >= 0)			rho(idx, counter) += algebra::conjugate(state(n)) * state(j);
						// increase counter to move along reduced basis
						counter++;
					}
				}
				return rho;
			};
		
		};
		
		// ############## CALCULATOR VIA SCHMIDT DECOMPOSITION ##############
		template <typename _T>
		double schmidt_decomposition(const arma::Col<_T>& _s, uint _sizeA, const Hilbert::HilbertSpace<_T>& _hilb) {
			int num_of_bits = log2(config);
			const long long dimA = (ULLPOW((num_of_bits * A_size)));
			const long long dimB = (ULLPOW((num_of_bits * (Ns - A_size))));
			const long long full_dim = dimA * dimB;

			// reshape array to matrix
			arma::Mat<_type> rho = arma::reshape(cast_state_to_full(state, map, full_dim), dimA, dimB);

			// get schmidt coefficients from singular-value-decomposition
			arma::vec schmidt_coeff = arma::svd(rho);

			// calculate entropy
			double entropy = 0;
			// #pragma omp parallel for reduction(+: entropy)
			for (int i = 0; i < schmidt_coeff.size(); i++) {
				const auto value = schmidt_coeff(i) * schmidt_coeff(i);
				entropy += (abs(value) > 0) ? -value * std::log(value) : 0;
			}
			return entropy;
		}

	};
};



		// -----------------------------------------------  				   ENTROPY 				    ----------------------------------------------



		//double entanglement_entropy(const Col<_type>& state, int A_size, const v_1d<u64>& map = {}) const;					// entanglement entropy 
		//double entanglement_entropy(const Mat<_type>& red_dens_mat) const;
		//vec entanglement_entropy_sweep(const Col<_type>& state) const;														// entanglement entropy sweep over bonds


		//// helpers
		//void calculate_operators(const Col<_type>& alfa, avOperators& av_op, bool cal_entro = true);
		////void calculate_operators(const Col<_type>& alfa, const Col<_type>& beta, avOperators& av_op);

		//// -----------------------------------------------  				   HISTOGRAMS 				    ----------------------------------------------
		//void calculate_histogram(const Mat<_type>& eigstates);

		// -----------------------------------------------  				   STATE CASTING				    ----------------------------------------------

	//	arma::Col<_type> cast_state_to_full(const Col<_type>& state, const std::vector<u64>& map, size_t dim_max) const;	// use global symmetry mapping to cast state to full hilbert space


//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op)
//{
//	cpx value = 0;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		for (int j = 0; j < Ns; j++) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, j));
//			value += val * conj(alfa(new_idx)) * beta(k);
//		}
//	}
//	return value / double(this->Ns);
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, std::vector<int> sites)
//{
//	for (auto& site : sites)
//		if (site < 0 || site >= this->Ns) throw "Site index exceeds chain";
//	cpx value = 0;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		for (auto const& site : sites) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, site));
//			value += val * conj(alfa(new_idx)) * beta(k);
//		}
//	}
//	return value;
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, int site_a, int site_b)
//{
//	if (site_a < 0 || site_b < 0 || site_a >= this->Ns || site_b >= this->Ns) throw "Site index exceeds chain";
//	cpx value = 0;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		const auto& [new_idx, val] = op(k, Ns, v_1d<int>{site_a, site_b});
//		value += val * conj(alfa(new_idx)) * beta(k);
//	}
//	return value;
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, int corr_len)
//{
//	return cpx();
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op)
//{
//	cpx value = 0;
//	//stout << alfa << EL;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		for (int j = 0; j < Ns; j++) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, j));
//			value += val * conj(alfa(new_idx)) * alfa(k);
//			//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << EL;
//		}
//	}
//	return value / double(this->Ns);
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, std::vector<int> sites)
//{
//	for (auto& site : sites)
//		if (site < 0 || site >= this->Ns) throw "Site index exceeds chain";
//	cpx value = 0;
//	//stout << alfa << EL;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		for (auto const& site : sites) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, site));
//			value += val * conj(alfa(new_idx)) * alfa(k);
//			//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << "," << VEQ(site) << EL;
//		}
//	}
//	return value;
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, int site_a, int site_b)
//{
//	cpx value = 0;
//	//stout << alfa << EL;
////#pragma omp parallel for reduction (+: value)
//	for (int k = 0; k < alfa.n_elem; k++) {
//		if (!(site_a < 0 || site_b < 0 || site_a >= this->Ns || site_b >= this->Ns)) {
//			const auto& [new_idx, val] = op(k, Ns, v_1d<int>{site_a, site_b});
//			value += val * conj(alfa(new_idx)) * alfa(k);
//		}
//		//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << "," << VEQ(site_a) << "," << VEQ(site_b) << EL;
//	}
//	return value;
//}
//
//template<typename _type>
//inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, int corr_len)
//{
//	return cpx();
//}


// ----------------------------   				   STATE TRANSFORMATION  				    ----------------------------------


/*
* @brief For global symmetries in the system, cast the state back to the original Hilbert space via mapping
* @tparam _type input state type
* @param state input state in reduced Hilbert size
* @param map mapping to the full Hilbert space
* @param dim_max maximal dimension we can get
*/
//template <typename _type>
//inline arma::Col<_type> Operators<_type>::cast_state_to_full(const arma::Col<_type>& state, const std::vector<u64>& map, size_t dim_max) const {
//	if (map.empty())
//		return state;
//	else {
//		// cast full state to zeros
//		Col<_type> full_state(dim_max, arma::fill::zeros);
//		for (int i = 0; i < map.size(); i++)
//			full_state(map[i]) = state(i);
//		return full_state;
//	}
//};
//
//// ----------------------------   				   ENTROPY  				    ----------------------------------
//
///*
//* @brief Calculates the entropy using the Schmidt decomposition of a wavefunction
//* @param _type input state type
//* @param state input state in Hilbert space
//* @param A_size subsystem size
//* @param config on-site configuration number (local hilbert space)
//* @return entanglement entropy
//*/
//template<typename _type>
//inline double Operators<_type>::schmidt_decomposition(const Col<_type>& state, int A_size, const v_1d<u64>& map, int config) const

//}
//
//}
//
//// ---- ENTROPY ----
//
///*
//*  @brief Calculates the entropy of the system via the mixed density matrix
//*  @param state state to produce the density matrix
//*  @param A_size size of subsystem
//*  @returns entropy of considered systsem
//*/
//template<typename _type>
//inline double Operators<_type>::entanglement_entropy(const Col<_type>& state, int A_size, const v_1d<u64>& map) const {
//	Mat<_type> rho = !map.empty() ? this->red_dens_mat(state, map, A_size) : this->red_dens_mat(state, A_size);
//	vec probabilities;
//	// diagonalize to find probabilities and calculate trace in rho's eigenbasis
//	eig_sym(probabilities, rho);
//
//	double entropy = 0;
//	//#pragma omp parallel for reduction(+: entropy)
//	for (auto i = 0; i < probabilities.size(); i++) {
//		const auto value = probabilities(i);
//		entropy += (abs(value) < 1e-10) ? 0 : -value * log(abs(value));;
//	}
//	//double entropy = -real(trace(rho * real(logmat(rho))));
//	return entropy;
//}
//
//template<>
//inline double Operators<cpx>::entanglement_entropy(const Col<cpx>& state, int A_size, const v_1d<u64>& map) const {
//
//	Mat<cpx> rho = !map.empty() ? this->red_dens_mat(state, map, A_size) : this->red_dens_mat(state, A_size);
//	vec probabilities;
//	//// diagonalize to find probabilities and calculate trace in rho's eigenbasis
//	eig_sym(probabilities, rho);
//
//	double entropy = 0;
//	//#pragma omp parallel for reduction(+: entropy)
//	for (auto i = 0; i < probabilities.size(); i++) {
//		const auto value = probabilities(i);
//		entropy += (abs(value) < 1e-10) ? 0 : -value * log(abs(value));
//	}
//	return entropy;
//	// return -real(trace(rho * (logmat(rho))));
//}
//
///*
//* @brief entanglement entropy using the precalculated reduced density matrix
//*/
//template<typename _type>
//inline double Operators<_type>::entanglement_entropy(const Mat<_type>& rho) const
//{
//	vec probabilities;
//	eig_sym(probabilities, rho);
//
//	double entropy = 0;
//	//#pragma omp parallel for reduction(+: entropy)
//	for (auto i = 0; i < probabilities.size(); i++) {
//		const auto value = probabilities(i);
//		entropy += (abs(value) < 1e-17) ? 0 : -value * log(abs(value));
//	}
//	//double entropy = -real(trace(rho * real(logmat(rho))));
//	return entropy;
//}
//// ---- SWEEP ----
//
///*
//* @brief Calculates the entropy of the system via the mixed density matrix
//* @param state state vector to produce the density matrix
//* @returns entropy of considered systsem for different subsystem sizes
//*/
//template<typename _type>
//inline vec Operators<_type>::entanglement_entropy_sweep(const Col<_type>& state) const
//{
//	vec entropy(this->Ns - 1, arma::fill::zeros);
//#pragma omp parallel for
//	for (int i = 0; i < this->Ns - 1; i++)
//		entropy(i) = entanglement_entropy(state, i + 1);
//	//stout << EL << EL << entropy << EL << EL;
//	return entropy;
//}
//
//// -----------------   				   HELPERS  				    -------------------
//template<typename _type>
//inline void Operators<_type>::calculate_operators(const Col<_type>& eigvec, avOperators& av_op, bool cal_entro)
//{
//	av_op.reset();
//
//	// --------------------- compare sigma_z ---------------------
//
//	// S_z_vector extensive
//	av_op.s_z = std::real(this->av_operator(eigvec, this->sigma_z));
//
//	// S_z at each site
//	for (auto i = 0; i < Ns; i++)
//		av_op.s_z_i(i) = std::real(this->av_operator(eigvec, this->sigma_z, v_1d<int>(1, i)));
//	// stout << av_op.s_z_i << EL;
//	// S_z correlations
//	for (auto i = 0; i < Ns; i++) {
//		auto z_nei = this->lat->get_z_nn(i);
//		if (z_nei >= 0)
//			av_op.s_z_nei += std::real(this->av_operator(eigvec, this->sigma_z, i, z_nei));
//
//		for (auto j = 0; j < Ns; j++) {
//			av_op.s_z_cor(i, j) += std::real(this->av_operator(eigvec, this->sigma_z, i, j));
//		}
//	}
//	av_op.s_z_nei /= Ns;
//	// --------------------- compare sigma_u ---------------------
//
//	// S_y_vector extensive
//	//av_op.s_y = std::real(this->av_operator(eigvec, this->sigma_y));
//
//	// S_y at each site
//	//for (auto i = 0; i < Ns; i++)
//	//	av_op.s_y_i(i) = std::real(this->av_operator(eigvec, this->sigma_y, v_1d<int>(1, i)));
//
//	// S_y correlations
//	for (auto i = 0; i < Ns; i++) {
//		int y_nei = this->lat->get_y_nn(i);
//		if (y_nei >= 0)
//			av_op.s_y_nei += std::real(this->av_operator(eigvec, this->sigma_y, i, y_nei));
//	}
//	av_op.s_y_nei /= Ns;
//	// --------------------- compare sigma_x ---------------------
//
//	// S_x_vector extensive
//	av_op.s_x = std::real(this->av_operator(eigvec, this->sigma_x));
//
//	// S_x at each site
//	for (auto i = 0; i < Ns; i++)
//		av_op.s_x_i(i) = std::real(this->av_operator(eigvec, this->sigma_x, v_1d<int>(1, i)));
//
//	// S_x correlations
//	for (auto i = 0; i < Ns; i++) {
//		int x_nei = this->lat->get_x_nn(i);
//		if (x_nei >= 0)
//			av_op.s_x_nei += std::real(this->av_operator(eigvec, this->sigma_x, i, x_nei));
//
//		for (auto j = 0; j < Ns; j++) {
//			av_op.s_x_cor(i, j) += std::real(this->av_operator(eigvec, this->sigma_x, i, j));
//		}
//	}
//	av_op.s_x_nei /= Ns;
//
//	// --------------------- entropy ----------------------
//	if (cal_entro)
//		av_op.ent_entro = this->entanglement_entropy_sweep(eigvec);
//}

#endif