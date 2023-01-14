#pragma once
#ifndef LATTICE_H
#include "../../source/src/lattices.h"
#endif
#ifndef BINARY_H
#include "../../source/src/binary.h"
#endif


#ifndef OPERATORS_H
#define OPERATORS_H
#include <queue>

namespace operators {
	constexpr double _SPIN = 0.5;
	constexpr double _SPIN_RBM = 0.5;
}


class avOperators {
public:
	std::string lat_type = "";
	int Ns = 1;
	int Lx = 1;
	int Ly = 1;
	int Lz = 1;


	// sigma z
	double s_z = 0.0;

	// correlation with z neighbor
	double s_z_nei = 0.0;
	//v_3d<double> s_z_cor;
	mat s_z_cor;
	vec s_z_i;

	// sigma x
	cpx s_x = 0.0;
	// correlation with x neighbor
	double s_x_nei = 0.0;
	//v_3d<double> s_x_cor;
	mat s_x_cor;
	cx_vec s_x_i;

	// sigma y
	cpx s_y = 0.0;
	// correlation with x neighbor
	cpx s_y_nei = 0.0;
	//v_3d<double> s_x_cor;
	//mat s_y_cor;
	//cx_vec s_y_i;


	// entropy
	vec ent_entro;

	// energy
	cpx en = 0.0;

	avOperators() = default;
	avOperators(int Lx, int Ly, int Lz, int Ns, std::string lat_type)
		: Lx(Lx), Ly(Ly), Lz(Lz), Ns(Ns), lat_type(lat_type)
	{
		//v_3d<double> corr_vec;
		//if (lat_type == "square") {
		//	corr_vec = SPACE_VEC_D(Lx, Ly, Lz);
		//}
		//else if (lat_type == "hexagonal") {
		//	corr_vec = SPACE_VEC_D(Lx, 2 * Ly, Lz);
		//}

		this->s_z_cor = mat(Ns, Ns, arma::fill::zeros);
		this->s_z_i = arma::vec(Ns, arma::fill::zeros);
		this->s_x_cor = mat(Ns, Ns, arma::fill::zeros);
		this->s_x_i = arma::cx_vec(Ns, arma::fill::zeros);
		this->ent_entro = arma::vec(Ns - 1, arma::fill::zeros);
	};

	void reset() {
		//v_3d<double> corr_vec;
		//if (lat_type == "square") {
		//	corr_vec = SPACE_VEC_D(Lx, Ly, Lz);
		//}
		//else if (lat_type == "hexagonal") {
		//	corr_vec = SPACE_VEC_D(Lx, 2 * Ly, Lz);
		//}
		this->s_x = 0.0;
		this->s_x_nei = 0.0;
		this->s_y = 0.0;
		this->s_y_nei = 0.0;
		this->s_z = 0.0;
		this->s_z_nei = 0.0;

		this->s_z_cor = mat(Ns, Ns, arma::fill::zeros);
		this->s_z_i = arma::vec(Ns, arma::fill::zeros);
		this->s_x_cor = mat(Ns, Ns, arma::fill::zeros);
		this->s_x_i = arma::cx_vec(Ns, arma::fill::zeros);
		this->ent_entro = arma::vec(Ns - 1, arma::fill::zeros);
	};

	void normalise(u64 norm, const v_3d<int>& spatialNorm) {
		this->s_z /= double(norm);
		this->s_y /= double(norm);
		this->s_x /= double(norm);
		this->s_z_i /= double(norm);
		this->s_x_i /= double(norm);
		this->s_z_nei /= double(norm);
		this->s_x_nei /= double(norm);
		this->s_y_nei /= double(norm);

		this->s_x_cor /= double(norm);
		this->s_z_cor /= double(norm);

		//for (int i = 0; i < this->s_x_cor.size(); i++) {
		//	for (int j = 0; j < this->s_x_cor[i].size(); j++) {
		//		for (int k = 0; k < this->s_x_cor[i][j].size(); k++) {
		//			this->s_x_cor[i][j][k] /= spatialNorm[i][j][k] * norm;
		//			this->s_z_cor[i][j][k] /= spatialNorm[i][j][k] * norm;
		//		}
		//	}
		//}
		this->en /= double(norm);
	};
};


using op_type = std::function<std::pair<u64, cpx>(u64, int, std::vector<int>)>;
/*
* @brief multiplies two operators together
* @param A a left operator
* @param B a right operator
*/
template <typename T1, typename T2>
inline function<T1(T1, T2)> multiply_operators(const function<T1(T1, T2)>& A, const std::function<T1(T1, T2)>& B) {
	//auto result = [A, B](T1 n, T2 L) { return A(B(n, L), L); };
	return [A, B](T1 n, T2 L) { return A(B(n, L), L); };
}

template<typename _type>
class Operators {
protected:
	shared_ptr<Lattice> lat;
	int Ns = 1;

public:
	~Operators() = default;

	Operators(std::shared_ptr<Lattice> lat)
	{
		this->lat = lat;
		this->Ns = this->lat->get_Ns();
	};

	/*
	* @brief multiplication of sigma_xi | state >
	* @param L lattice dimensionality (base vector length)
	* @param sites the sites to meassure correlation at
	*/
	static std::pair<u64, cpx> sigma_x(u64 base_vec, int L, const v_1d<int>& sites) {
		auto tmp = base_vec;
		for (auto const& site : sites)
			tmp = flip(tmp, L - 1 - site);
		return std::make_pair(tmp, operators::_SPIN);
	};

	/*
	* @brief multiplication of sigma_yi | state >
	* @param L lattice dimensionality (base vector length)
	* @param sites the sites to meassure correlation at
	*/
	static std::pair<u64, cpx> sigma_y(u64 base_vec, int L, const v_1d<int>& sites) {
		auto tmp = base_vec;
		cpx val = 1.0;
		for (auto const& site : sites) {
			val *= checkBit(tmp, L - 1 - site) ? imn * operators::_SPIN : -imn * operators::_SPIN;
			tmp = flip(tmp, L - 1 - site);
		}
		return std::make_pair(tmp, val);
	};

	/*
	* @brief multiplication of sigma_zi | state >
	* @param L lattice dimensionality (base vector length)
	* @param sites the sites to meassure correlation at
	*/
	static std::pair<u64, cpx> sigma_z(u64 base_vec, int L, const v_1d<int>& sites) {
		double val = 1.0;
		for (auto const& site : sites)
			val *= checkBit(base_vec, L - 1 - site) ? operators::_SPIN : -operators::_SPIN;
		return std::make_pair(base_vec, val);
	};

	/*
	*/
	static std::pair<u64, cpx> spin_flip(u64 base_vec, int L, v_1d<int> sites) {
		if (sites.size() > 2) throw "Not implemented such exotic operators, choose 1 or 2 sites\n";
		auto tmp = base_vec;
		cpx val = 0.0;
		auto it = sites.begin() + 1;
		auto it2 = sites.begin();
		if (!(checkBit(base_vec, L - 1 - *it))) {
			tmp = flip(tmp, L - 1 - *it);
			val = 2.0;
			if (sites.size() > 1) {
				if (checkBit(base_vec, L - 1 - *it2)) {
					tmp = flip(tmp, L - 1 - *it2);
					val *= 2.0;
				}
				else val = 0.0;
			}
		}
		else val = 0.0;
		return std::make_pair(tmp, val);
	};

	// -----------------------------------------------  				   AVERAGE OPERATOR 				    ----------------------------------------------

	// calculates the matrix element of operator at given sites (sum)
	cpx av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op);
	// calculates the matrix element of operator at given sites
	cpx av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, std::vector<int> sites);
	// calculates the matrix element of operator at given site in extensive form (a sum) with pair sites
	cpx av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, int site_a, int site_b);
	// calculates the matrix element of operator at given site in extensive form (a sum) with corr_len
	cpx av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, int corr_len);

	// -----------------------------------------------				 SINGLE (diagonal elements)

	// calculates the matrix element of operator at given sites (sum)
	cpx av_operator(const Col<_type>& alfa, op_type op);
	// calculates the matrix element of operator at given sites
	cpx av_operator(const Col<_type>& alfa, op_type op, std::vector<int> sites);
	// calculates the matrix element of operator at given site in extensive form (a sum) with pair sites
	cpx av_operator(const Col<_type>& alfa, op_type op, int site_a, int site_b);
	// calculates the matrix element of operator at given site in extensive form (a sum) with corr_len
	cpx av_operator(const Col<_type>& alfa, op_type op, int corr_len);

	// -----------------------------------------------  				   ENTROPY 				    ----------------------------------------------

	Mat<_type> red_dens_mat(const Col<_type>& state, int A_size) const;													// calculate the reduced density matrix
	Mat<_type> red_dens_mat(const Col<_type>& state, const v_1d<u64>& map, int A_size) const;

	double schmidt_decomposition(const Col<_type>& state, int A_size, const v_1d<u64>& map = {}, int config = 2) const;	// entanglement entropy via schmidt decomposition
	double entanglement_entropy(const Col<_type>& state, int A_size, const v_1d<u64>& map = {}) const;					// entanglement entropy 
	double entanglement_entropy(const Mat<_type>& red_dens_mat) const;
	vec entanglement_entropy_sweep(const Col<_type>& state) const;														// entanglement entropy sweep over bonds


	// helpers
	void calculate_operators(const Col<_type>& alfa, avOperators& av_op, bool cal_entro = true);
	//void calculate_operators(const Col<_type>& alfa, const Col<_type>& beta, avOperators& av_op);

	// -----------------------------------------------  				   HISTOGRAMS 				    ----------------------------------------------
	void calculate_histogram(const Mat<_type>& eigstates);

	// -----------------------------------------------  				   STATE CASTING				    ----------------------------------------------

	arma::Col<_type> cast_state_to_full(const Col<_type>& state, const std::vector<u64>& map, size_t dim_max) const;	// use global symmetry mapping to cast state to full hilbert space
};


template<typename _type>
inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op)
{
	cpx value = 0;
#pragma omp parallel for reduction (+: value)
	for (int k = 0; k < alfa.n_elem; k++) {
		for (int j = 0; j < Ns; j++) {
			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, j));
			value += val * conj(alfa(new_idx)) * beta(k);
		}
	}
	return value / double(this->Ns);
}

template<typename _type>
inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, std::vector<int> sites)
{
	for (auto& site : sites)
		if (site < 0 || site >= this->Ns) throw "Site index exceeds chain";
	cpx value = 0;
#pragma omp parallel for reduction (+: value)
	for (int k = 0; k < alfa.n_elem; k++) {
		for (auto const& site : sites) {
			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, site));
			value += val * conj(alfa(new_idx)) * beta(k);
		}
	}
	return value;
}

template<typename _type>
inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, int site_a, int site_b)
{
	if (site_a < 0 || site_b < 0 || site_a >= this->Ns || site_b >= this->Ns) throw "Site index exceeds chain";
	cpx value = 0;
#pragma omp parallel for reduction (+: value)
	for (int k = 0; k < alfa.n_elem; k++) {
		const auto& [new_idx, val] = op(k, Ns, v_1d<int>{site_a, site_b});
		value += val * conj(alfa(new_idx)) * beta(k);
	}
	return value;
}

template<typename _type>
inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, const Col<_type>& beta, op_type op, int corr_len)
{
	return cpx();
}

template<typename _type>
inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op)
{
	cpx value = 0;
	//stout << alfa << EL;
#pragma omp parallel for reduction (+: value)
	for (int k = 0; k < alfa.n_elem; k++) {
		for (int j = 0; j < Ns; j++) {
			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, j));
			value += val * conj(alfa(new_idx)) * alfa(k);
			//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << EL;
		}
	}
	return value / double(this->Ns);
}

template<typename _type>
inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, std::vector<int> sites)
{
	for (auto& site : sites)
		if (site < 0 || site >= this->Ns) throw "Site index exceeds chain";
	cpx value = 0;
	//stout << alfa << EL;
#pragma omp parallel for reduction (+: value)
	for (int k = 0; k < alfa.n_elem; k++) {
		for (auto const& site : sites) {
			const auto& [new_idx, val] = op(k, Ns, v_1d<int>(1, site));
			value += val * conj(alfa(new_idx)) * alfa(k);
			//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << "," << VEQ(site) << EL;
		}
	}
	return value;
}

template<typename _type>
inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, int site_a, int site_b)
{
	cpx value = 0;
	//stout << alfa << EL;
#pragma omp parallel for reduction (+: value)
	for (int k = 0; k < alfa.n_elem; k++) {
		if (!(site_a < 0 || site_b < 0 || site_a >= this->Ns || site_b >= this->Ns)) {
			const auto& [new_idx, val] = op(k, Ns, v_1d<int>{site_a, site_b});
			value += val * conj(alfa(new_idx)) * alfa(k);
		}
		//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << "," << VEQ(site_a) << "," << VEQ(site_b) << EL;
	}
	return value;
}

template<typename _type>
inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, int corr_len)
{
	return cpx();
}


// ----------------------------   				   STATE TRANSFORMATION  				    ----------------------------------


/*
* @brief For global symmetries in the system, cast the state back to the original Hilbert space via mapping
* @tparam _type input state type
* @param state input state in reduced Hilbert size
* @param map mapping to the full Hilbert space
* @param dim_max maximal dimension we can get
*/
template <typename _type>
inline arma::Col<_type> Operators<_type>::cast_state_to_full(const arma::Col<_type>& state, const std::vector<u64>& map, size_t dim_max) const {
	if (map.empty())
		return state;
	else {
		// cast full state to zeros
		Col<_type> full_state(dim_max, arma::fill::zeros);
		for (int i = 0; i < map.size(); i++)
			full_state(map[i]) = state(i);
		return full_state;
	}
};

// ----------------------------   				   ENTROPY  				    ----------------------------------

/*
* @brief Calculates the entropy using the Schmidt decomposition of a wavefunction
* @param _type input state type
* @param state input state in Hilbert space
* @param A_size subsystem size
* @param config on-site configuration number (local hilbert space)
* @return entanglement entropy
*/
template<typename _type>
inline double Operators<_type>::schmidt_decomposition(const Col<_type>& state, int A_size, const v_1d<u64>& map, int config) const
{
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

/*
* @brief Calculates the reduced density matrix of the system via the mixed density matrix
* @param state state to produce the density matrix
* @param A_size size of subsystem
* @returns reduced density matrix
*/
template<typename _type>
inline Mat<_type> Operators<_type>::red_dens_mat(const Col<_type>& state, int A_size) const
{
	// set subsytsems size
	const long long dimA = ULLPOW(A_size);
	const long long dimB = ULLPOW(Ns - A_size);
	const long long Nh = dimA * dimB;

	Mat<_type> rho(dimA, dimA, arma::fill::zeros);
	// loop over configurational basis
	for (long long n = 0; n < Nh; n++) {
		u64 counter = 0;
		// pick out state with same B side (last L-A_size bits)
		for (long long m = n % dimB; m < Nh; m += dimB) {
			// find index of state with same B-side (by dividing the last bits are discarded)
			u64 idx = n / dimB;
			rho(idx, counter) += std::conj(state(n)) * state(m);
			// increase counter to move along reduced basis
			counter++;
		}
	}
	return rho;
}

template<>
inline Mat<double> Operators<double>::red_dens_mat(const Col<double>& state, int A_size) const
{
	// set subsytsems size
	const long long dimA = ULLPOW(A_size);
	const long long dimB = ULLPOW(Ns - A_size);
	const long long Nh = dimA * dimB;

	Mat<double> rho(dimA, dimA, arma::fill::zeros);
	// loop over configurational basis
	for (long long n = 0; n < Nh; n++) {
		u64 counter = 0;
		// pick out state with same B side (last L-A_size bits)
		for (long long m = n % dimB; m < Nh; m += dimB) {
			// find index of state with same B-side (by dividing the last bits are discarded)
			u64 idx = n / dimB;
			rho(idx, counter) += (state(n)) * state(m);
			// increase counter to move along reduced basis
			counter++;
		}
	}
	return rho;
}

template<typename _type>
inline Mat<_type> Operators<_type>::red_dens_mat(const Col<_type>& state, const v_1d<u64>& map, int A_size) const
{
	const long long dimA = ULLPOW(A_size);
	const long long dimB = ULLPOW(Ns - A_size);
	const long long Nh = dimA * dimB;
	const long long N = map.size();

	auto find_index = [&](u64 index) { return binary_search(map, 0, N - 1, index); };

	Mat<_type> rho(dimA, dimA, arma::fill::zeros);
	for (long long n = 0; n < N; n++) {
		// loop over configurational basis
		u64 counter = 0;
		long long true_n = map[n];
		for (long long m = true_n % dimB; m < Nh; m += dimB) {
			// pick out state with same B side (last L-A_size bits)
			long idx = true_n / dimB;
			long long j = find_index(m);
			if (j >= 0)
				rho(idx, counter) += std::conj(state(n)) * state(j);
			counter++;
			// increase counter to move along reduced basis
		}
	}
	return rho;
}

template<>
inline Mat<double> Operators<double>::red_dens_mat(const Col<double>& state, const v_1d<u64>& map, int A_size) const
{
	const long long dimA = ULLPOW(A_size);
	const long long dimB = ULLPOW(Ns - A_size);
	const long long Nh = dimA * dimB;
	const long long N = map.size();


	auto find_index = [&](u64 index) { return binary_search(map, 0, N - 1, index); };

	Mat<double> rho(dimA, dimA, arma::fill::zeros);
	for (long long n = 0; n < N; n++) {
		// loop over configurational basis
		u64 counter = 0;
		long long true_n = map[n];
		for (long long m = true_n % dimB; m < Nh; m += dimB) {
			// pick out state with same B side (last L-A_size bits)
			long idx = true_n / dimB;
			long long j = find_index(m);
			if (j >= 0)
				rho(idx, counter) += (state(n)) * state(j);
			counter++;
			// increase counter to move along reduced basis
		}
	}
	return rho;
}

// ---- ENTROPY ----

/*
*  @brief Calculates the entropy of the system via the mixed density matrix
*  @param state state to produce the density matrix
*  @param A_size size of subsystem
*  @returns entropy of considered systsem
*/
template<typename _type>
inline double Operators<_type>::entanglement_entropy(const Col<_type>& state, int A_size, const v_1d<u64>& map) const {
	Mat<_type> rho = !map.empty() ? this->red_dens_mat(state, map, A_size) : this->red_dens_mat(state, A_size);
	vec probabilities;
	// diagonalize to find probabilities and calculate trace in rho's eigenbasis
	eig_sym(probabilities, rho);

	double entropy = 0;
	//#pragma omp parallel for reduction(+: entropy)
	for (auto i = 0; i < probabilities.size(); i++) {
		const auto value = probabilities(i);
		entropy += (abs(value) < 1e-10) ? 0 : -value * log(abs(value));;
	}
	//double entropy = -real(trace(rho * real(logmat(rho))));
	return entropy;
}

template<>
inline double Operators<cpx>::entanglement_entropy(const Col<cpx>& state, int A_size, const v_1d<u64>& map) const {

	Mat<cpx> rho = !map.empty() ? this->red_dens_mat(state, map, A_size) : this->red_dens_mat(state, A_size);
	vec probabilities;
	//// diagonalize to find probabilities and calculate trace in rho's eigenbasis
	eig_sym(probabilities, rho);

	double entropy = 0;
	//#pragma omp parallel for reduction(+: entropy)
	for (auto i = 0; i < probabilities.size(); i++) {
		const auto value = probabilities(i);
		entropy += (abs(value) < 1e-10) ? 0 : -value * log(abs(value));
	}
	return entropy;
	// return -real(trace(rho * (logmat(rho))));
}

/*
* @brief entanglement entropy using the precalculated reduced density matrix
*/
template<typename _type>
inline double Operators<_type>::entanglement_entropy(const Mat<_type>& rho) const
{
	vec probabilities;
	eig_sym(probabilities, rho);

	double entropy = 0;
	//#pragma omp parallel for reduction(+: entropy)
	for (auto i = 0; i < probabilities.size(); i++) {
		const auto value = probabilities(i);
		entropy += (abs(value) < 1e-17) ? 0 : -value * log(abs(value));
	}
	//double entropy = -real(trace(rho * real(logmat(rho))));
	return entropy;
}
// ---- SWEEP ----

/*
* @brief Calculates the entropy of the system via the mixed density matrix
* @param state state vector to produce the density matrix
* @returns entropy of considered systsem for different subsystem sizes
*/
template<typename _type>
inline vec Operators<_type>::entanglement_entropy_sweep(const Col<_type>& state) const
{
	vec entropy(this->Ns - 1, arma::fill::zeros);
#pragma omp parallel for
	for (int i = 0; i < this->Ns - 1; i++)
		entropy(i) = entanglement_entropy(state, i + 1);
	//stout << EL << EL << entropy << EL << EL;
	return entropy;
}

// -----------------   				   HELPERS  				    -------------------
template<typename _type>
inline void Operators<_type>::calculate_operators(const Col<_type>& eigvec, avOperators& av_op, bool cal_entro)
{
	av_op.reset();

	// --------------------- compare sigma_z ---------------------

	// S_z_vector extensive
	av_op.s_z = std::real(this->av_operator(eigvec, this->sigma_z));

	// S_z at each site
	for (auto i = 0; i < Ns; i++)
		av_op.s_z_i(i) = std::real(this->av_operator(eigvec, this->sigma_z, v_1d<int>(1, i)));
	// stout << av_op.s_z_i << EL;
	// S_z correlations
	for (auto i = 0; i < Ns; i++) {
		auto z_nei = this->lat->get_z_nn(i);
		if (z_nei >= 0)
			av_op.s_z_nei += std::real(this->av_operator(eigvec, this->sigma_z, i, z_nei));

		for (auto j = 0; j < Ns; j++) {
			av_op.s_z_cor(i, j) += std::real(this->av_operator(eigvec, this->sigma_z, i, j));
		}
	}
	av_op.s_z_nei /= Ns;
	// --------------------- compare sigma_u ---------------------

	// S_y_vector extensive
	//av_op.s_y = std::real(this->av_operator(eigvec, this->sigma_y));

	// S_y at each site
	//for (auto i = 0; i < Ns; i++)
	//	av_op.s_y_i(i) = std::real(this->av_operator(eigvec, this->sigma_y, v_1d<int>(1, i)));

	// S_y correlations
	for (auto i = 0; i < Ns; i++) {
		int y_nei = this->lat->get_y_nn(i);
		if (y_nei >= 0)
			av_op.s_y_nei += std::real(this->av_operator(eigvec, this->sigma_y, i, y_nei));
	}
	av_op.s_y_nei /= Ns;
	// --------------------- compare sigma_x ---------------------

	// S_x_vector extensive
	av_op.s_x = std::real(this->av_operator(eigvec, this->sigma_x));

	// S_x at each site
	for (auto i = 0; i < Ns; i++)
		av_op.s_x_i(i) = std::real(this->av_operator(eigvec, this->sigma_x, v_1d<int>(1, i)));

	// S_x correlations
	for (auto i = 0; i < Ns; i++) {
		int x_nei = this->lat->get_x_nn(i);
		if (x_nei >= 0)
			av_op.s_x_nei += std::real(this->av_operator(eigvec, this->sigma_x, i, x_nei));

		for (auto j = 0; j < Ns; j++) {
			av_op.s_x_cor(i, j) += std::real(this->av_operator(eigvec, this->sigma_x, i, j));
		}
	}
	av_op.s_x_nei /= Ns;

	// --------------------- entropy ----------------------
	if (cal_entro)
		av_op.ent_entro = this->entanglement_entropy_sweep(eigvec);
}


template<typename _type>
inline void Operators<_type>::calculate_histogram(const Mat<_type>& eigstates)
{

}
#endif // !OPERATORS_H
