#ifndef OPERATORS_H
#define OPERATORS_H

#ifndef HILBERT_H
#include "hilbert.h"
#endif // !HILBERT_H

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
				int bitNum					=			std::log2(_Nint);
				const u64 dimA				=			ULLPOW(bitNum * _sizeA);
				const u64 dimB				=			ULLPOW((_Ns - bitNum * _sizeA));
				const u64 Nh				=			dimA * dimB;

				arma::Mat<_T> rho(dimA, dimA, arma::fill::zeros);
				// loop over configurational basis
				for (u64 n = 0; n < Nh; n++) {
					u64 counter				=			0;
					// pick out state with same B side (last L-A_size bits)
					for (u64 m = n % dimB; m < Nh; m += dimB) {
						// find index of state with same B-side (by dividing the last bits are discarded)
						u64 idx				=			n / dimB;
						rho(idx, counter)	+=			algebra::conjugate(_s(n)) * _s(m);
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
			inline arma::Mat<_T> redDensMatStandard(const arma::Col<_T>& _s, uint _sizeA, const Hilbert::HilbertSpace<_T>& _hilb) {
				// set subsystems size
				uint Ns						=			_hilb.getLatticeSize();
				uint Nint					=			_hilb.getNum();
				uint bitNum					=			std::log2(Nint);
				const u64 dimA				=			ULLPOW(bitNum * _sizeA);
				const u64 dimB				=			ULLPOW((Ns - bitNum * _sizeA));
				const u64 Nh				=			dimA * dimB;
				if (!_hilb.checkGSym())		return		Entropy::Entanglement::Bipartite::redDensMatStandard<_T>(_s, Ns, _sizeA, Nint);

				auto map					=			_hilb.getFullMap();
				const u64 N					=			map.size();
				// otherwise find in mapping
				auto find_index				=			[&](u64 _idx) { return binarySearch(map, 0, Nh - 1, _idx); };

				arma::Mat<_T> rho(dimA, dimA, arma::fill::zeros);
				for (u64 n = 0; n < N; n++) {
					// loop over configurational basis
					u64 ctr					=			0;
					u64 true_n				=			map[n];
					for (u64 m = true_n % dimB; m < Nh; m += dimB) {
						// pick out state with same B side (last L-A_size bits)
						u64 idx				=			true_n / dimB;
						u64 j				=			find_index(m);
						if (j >= 0)			
							rho(idx, ctr)	+=			algebra::conjugate(_s(n)) * _s(j);
						// increase counter to move along reduced basis
						ctr++;
					}
				}
				return rho;
			};
			
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
				uint Ns						=			_hilb.getLatticeSize();
				uint Nint					=			_hilb.getNum();
				uint bitNum					=			std::log2(Nint);
				const u64 dimA				=			ULLPOW(bitNum * _sizeA);
				const u64 dimB				=			ULLPOW((Ns - bitNum * _sizeA));
				return arma::reshape(_s, dimA, dimB);
			}
		
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
			double vonNeuman(const arma::Col<_T>& _s, uint _sizeA,
							Hilbert::HilbertSpace<_T>& _hilb,
							RHO_METHODS _ch = RHO_METHODS::SCHMIDT)
			{
				// get the reduced density matrix
				auto rho					=			redDensMat<_T>(_s, _sizeA, _hilb, _ch);
				// get the values
				arma::vec vals;
				if (_ch == RHO_METHODS::SCHMIDT)
				{
					vals					=			arma::svd(rho);
					vals					=			vals * vals;
				}
				else
					arma::eig_sym(vals, rho);

				// calculate entropy
				double entropy				=			0.0;
				// #pragma omp parallel for reduction(+: entropy)
				for (auto i = 0; i < vals.size(); i++)
					entropy					+=			(std::abs(vals(i)) > 0) ? -vals(i) * std::log(vals(i)) : 0.0;
				return entropy;
			};
		};
	};
};

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