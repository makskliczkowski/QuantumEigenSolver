#pragma once
#ifndef LATTICE_H
	#include "../lattice.h"
#endif


#ifndef OPERATORS_H
#define OPERATORS_H
#include <queue>


using op_type = std::function<std::pair<u64, cpx>(u64, int, std::vector<int>)>;

class avOperators {
public:
	std::string lat_type = "";
	int Ns = 1;
	int Lx = 1;
	int Ly = 1;
	int Lz = 1;

	// sigma z
	double s_z = 0.0;
	//v_3d<double> s_z_cor;
	mat s_z_cor;
	vec s_z_i;

	// sigma x
	cpx s_x = 0.0;
	//v_3d<double> s_x_cor;
	mat s_x_cor;
	cx_vec s_x_i;

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

		this->s_z_cor = mat(Ns, Ns, arma::fill::zeros);
		this->s_z_i = arma::vec(Ns, arma::fill::zeros);
		this->s_x_cor = mat(Ns, Ns, arma::fill::zeros);
		this->s_x_i = arma::cx_vec(Ns, arma::fill::zeros);
		this->ent_entro = arma::vec(Ns - 1, arma::fill::zeros);
	};

	void normalise(u64 norm, const v_3d<int>& spatialNorm) {
		this->s_z /= double(norm);
		this->s_x /= double(norm);
		this->s_z_i /= double(norm);
		this->s_x_i /= double(norm);

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
		return std::make_pair(tmp, 1.0);
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
			val *= checkBit(tmp, L - 1 - site) ? imn : -imn;
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
			val *= checkBit(base_vec, L - 1 - site) ? 1.0 : -1.0;
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
	//Mat<_type> red_dens_mat(const std::map<u64, _type>& state, int A_size) const;										// calculate the reduced density matrix with a map
	//Mat<_type> red_dens_mat(const std::priority_queue<u64, _type>& state, int A_size) const;							// calculate the reduced density matrix with a priority queue
	double entanglement_entropy(const Col<_type>& state, int A_size) const;												// entanglement entropy 
	//double entanglement_entropy(const std::map<u64, _type>& state, int A_size) const;									// entanglement entropy with a map
	//double entanglement_entropy(const std::priority_queue<u64, _type>& state, int A_size) const;						// entanglement entropy with a priority queue
	vec entanglement_entropy_sweep(const Col<_type>& state) const;														// entanglement entropy sweep over bonds
	//vec entanglement_entropy_sweep(const std::map<u64, _type>& state) const;											// entanglement entropy sweep over bonds with a map
	//vec entanglement_entropy_sweep(const std::priority_queue<u64, _type>& state) const;									// entanglement entropy sweep over bonds with a priority queue


	// helpers
	void calculate_operators(const Col<_type>& alfa, avOperators& av_op, bool cal_entro = true);
	//void calculate_operators(const Col<_type>& alfa, const Col<_type>& beta, avOperators& av_op);

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
	if (site_a < 0 || site_b < 0 || site_a >= this->Ns || site_b >= this->Ns) throw "Site index exceeds chain";
	cpx value = 0;
	//stout << alfa << EL;
#pragma omp parallel for reduction (+: value)
	for (int k = 0; k < alfa.n_elem; k++) {
		const auto& [new_idx, val] = op(k, Ns, v_1d<int>{site_a, site_b});
		value += val * conj(alfa(new_idx)) * alfa(k);
		//stout << VEQ(k) << "," << VEQ(new_idx) << "," << VEQ(val) << ", " << VEQ(value) << "," << VEQ(site_a) << "," << VEQ(site_b) << EL;
	}
	return value;
}

template<typename _type>
inline cpx Operators<_type>::av_operator(const Col<_type>& alfa, op_type op, int corr_len)
{
	return cpx();
}


// ----------------------------   				   ENTROPY  				    ----------------------------------
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
	const u64 dimA = ULLPOW(A_size);
	const u64 dimB = ULLPOW(Ns - A_size);
	const u64 Nh = state.n_elem;

	Mat<_type> rho(dimA, dimA, arma::fill::zeros);
	// loop over configurational basis
	for (auto n = 0; n < Nh; n++) {
		u64 counter = 0;
		// pick out state with same B side (last L-A_size bits)
		for (u64 m = n % dimB; m < Nh; m += dimB) {
			// find index of state with same B-side (by dividing the last bits are discarded)
			u64 idx = n / dimB;
			rho(idx, counter) += conj(state(n)) * state(m);
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
	const u64 dimA = ULLPOW(A_size);
	const u64 dimB = ULLPOW(Ns - A_size);
	const u64 Nh = state.n_elem;

	Mat<double> rho(dimA, dimA, arma::fill::zeros);
	// loop over configurational basis
	for (auto n = 0; n < Nh; n++) {
		u64 counter = 0;
		// pick out state with same B side (last L-A_size bits)
		for (u64 m = n % dimB; m < Nh; m += dimB) {
			// find index of state with same B-side (by dividing the last bits are discarded)
			u64 idx = n / dimB;
			rho(idx, counter) += (state(n)) * state(m);
			// increase counter to move along reduced basis
			counter++;
		}
	}
	return rho;
}

/*
*  @brief Calculates the entropy of the system via the mixed density matrix
*  @param state state to produce the density matrix
*  @param A_size size of subsystem
*  @returns entropy of considered systsem
*/
template<typename _type>
inline double Operators<_type>::entanglement_entropy(const Col<_type>& state, int A_size) const {
	Mat<_type> rho = red_dens_mat(state, A_size);
	vec probabilities;
	// diagonalize to find probabilities and calculate trace in rho's eigenbasis
	eig_sym(probabilities, rho);

	double entropy = 0;
	//#pragma omp parallel for reduction(+: entropy)
	for (auto i = 0; i < probabilities.size(); i++) {
		const auto value = probabilities(i);
		entropy += (abs(value) < 1e-10) ? 0 : -value * log(abs(value));
	}
	//double entropy = -real(trace(rho * real(logmat(rho))));
	return entropy;
}

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
	stout << EL << EL << entropy << EL << EL;
	return entropy;
}

// -----------------   				   HELPERS  				    -------------------
template<typename _type>
inline void Operators<_type>::calculate_operators(const Col<_type>& eigvec, avOperators& av_op, bool cal_entro)
{
	
	// --------------------- compare sigma_z ---------------------

	// S_z_vector extensive
	av_op.s_z = std::real(this->av_operator(eigvec, this->sigma_z));

	// S_z at each site
	for (auto i = 0; i < Ns; i++)
		av_op.s_z_i(i) = std::real(this->av_operator(eigvec, this->sigma_z, v_1d<int>(1, i)));
	// stout << av_op.s_z_i << EL;
	// S_z correlations
	for (auto i = 0; i < Ns; i++) {
		for (auto j = 0; j < Ns; j++) {
			//const auto [x, y, z] = this->lat->getSiteDifference(i, j);
			//av_op.s_z_cor[abs(x)][abs(y)][abs(z)] += std::real(this->av_operator(eigvec, this->sigma_z, i, j)) / this->lat->get_spatial_norm(abs(x), abs(y), abs(z));
			av_op.s_z_cor(i, j) += std::real(this->av_operator(eigvec, this->sigma_z, i, j));
			//stout << VEQ(av_op.s_z_cor[abs(x)][abs(y)][abs(z)]) << EL;
		}
	}
	//stout << av_op.s_z_cor << EL;
	// --------------------- compare sigma_x ---------------------
	
	// S_x_vector extensive
	av_op.s_x = std::real(this->av_operator(eigvec, this->sigma_x));

	// S_x at each site
	for (auto i = 0; i < Ns; i++)
		av_op.s_x_i(i) = std::real(this->av_operator(eigvec, this->sigma_x, v_1d<int>(1, i)));

	// S_x correlations
	for (auto i = 0; i < Ns; i++) {
		for (auto j = 0; j < Ns; j++) {
			//const auto [x, y, z] = this->lat->getSiteDifference(i, j);
			//av_op.s_x_cor[abs(x)][abs(y)][abs(z)] += std::real(this->av_operator(eigvec, this->sigma_x, i, j)) / this->lat->get_spatial_norm(abs(x), abs(y), abs(z));
			av_op.s_x_cor(i, j) += std::real(this->av_operator(eigvec, this->sigma_x, i, j));
		}
	}

	// --------------------- entropy ----------------------
	if(cal_entro)
		av_op.ent_entro = this->entanglement_entropy_sweep(eigvec);
}


#endif // !OPERATORS_H
