#pragma once
/***********************************
* Contains the most common operators.
* Is used for more general opeartors.
* Also defines various acting on a 
* Hilbert space.
* DECEMBER 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***********************************/
#ifndef OPERATORS_H
#define OPERATORS_H

#ifndef ENTROPY_H
#	include "quantities/entropy.h"
#endif // !ENTROPY_H

namespace Operators 
{
	constexpr double _SPIN			=		0.5;
	constexpr double _SPIN_RBM		=		_SPIN;

	// ##########################################################################################################################################

	std::pair<u64, double> sigma_x(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<double> makeSigmaX(std::shared_ptr<Lattice> lat, uint site); 

	std::pair<u64, cpx> sigma_y(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<cpx> makeSigmaY(std::shared_ptr<Lattice> lat, uint site);

	std::pair<u64, double> sigma_z(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<double> makeSigmaZ(std::shared_ptr<Lattice> lat, uint site);

	// ##########################################################################################################################################

	std::pair<u64, double> c_dag_up(u64 base_vec, uint L, v_1d<uint> _sites);
	Operators::Operator<double> makeCDagUp(std::shared_ptr<Lattice> _lat, uint _site);

	std::pair<u64, double> c_up(u64 base_vec, uint L, v_1d<uint> _sites);
	Operators::Operator<double> makeCUp(std::shared_ptr<Lattice> _lat, uint _site);

	std::pair<u64, double> c_dag_dn(u64 base_vec, uint L, v_1d<uint> _sites);
	Operators::Operator<double> makeCDagDn(std::shared_ptr<Lattice> _lat, uint _site);

	std::pair<u64, double> c_dn(u64 base_vec, uint L, v_1d<uint> _sites);
	Operators::Operator<double> makeCDn(std::shared_ptr<Lattice> _lat, uint _site);
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