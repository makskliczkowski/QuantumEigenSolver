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

constexpr auto OPERATOR_SEP			= "/";
constexpr auto OPERATOR_SEP_CORR	= "-";
constexpr auto OPERATOR_SEP_DIV		= "_";

namespace Operators 
{
	// ##########################################################################################################################################

	constexpr double _SPIN			=		0.5;
	constexpr double _SPIN_RBM		=		_SPIN;

	// ##########################################################################################################################################
	
	inline std::string createOperatorName(const std::string& _type, const std::string& _name)
	{
		return _type + std::string(OPERATOR_SEP) + _name;
	}

	inline std::string createOperatorName(const std::string& _type, const std::string& _name, const std::string& _site)
	{
		return _type + std::string(OPERATOR_SEP) + _name + OPERATOR_SEP + _site;
	}

	inline std::string createOperatorName(const std::string& _type, const std::string& _name, const std::string& _site, const std::string& _site2)
	{
		return _type + std::string(OPERATOR_SEP) + _name + OPERATOR_SEP + _site + OPERATOR_SEP_CORR + _site2;
	}
	
	// ##########################################################################################################################################

	/*
	* @brief Applies the many body matrix to a given state O|\Psi>
	* @param _C many body state
	* @param _M many body matrix
	*/
	template<typename _Ct, typename _M>
	inline _Ct apply(const _Ct& _C, const _M& _mat)
	{
		return _mat * _C;
	}

	/*
	* @brief Applies the many body matrix to a given state and saves the overlap <\Psi|O|\Psi>
	* @param _C many body state
	* @param _M many body matrix
	* @returns the overlap <\Psi|O|\Psi>
	*/
	template<typename _M, typename _Ct>
	inline inner_type_t<_Ct> applyOverlap(const _Ct& _C, const _M& _mat)
	{
		return arma::cdot(_C, _mat * _C);
	}

	/*
	* @brief Applies the overlap between all the states in the matrix.
	* @param _eigvecs the eigenvectors matrix
	* @param _mat the many body matrix
	* @returns the overlap matrix
	*/
	template<typename _T, typename _M2>
	inline arma::Mat<_T> applyOverlapMat(const arma::Mat<_T>& _eigvecs, const _M2& _mat)
	{
		return _eigvecs.t() * (_mat * _eigvecs);
	}

	template<typename _T, typename _T2>
	inline arma::Mat<_T> applyOverlapMat(const arma::Mat<_T>& _eigvecs, const GeneralizedMatrix<_T2>& _mat)
	{
		if (_mat.isSparse())
			return _eigvecs.t() * (arma::SpMat<_T2>(_mat.getSparse()) * _eigvecs);
		else
			return _eigvecs.t() * (_mat.getDense() * _eigvecs);
	}

	// _____________________________________________________________________________________________________________________________

	template<typename _M, typename _Ct>
	inline _Ct applyOverlap(const arma::subview_col<_Ct>& _C, const _M& _mat)
	{
		return arma::cdot(_C, _mat * _C);
	}

	/*
	* @brief Applies the many body matrix to a given state and saves the overlap <\Psi|O|\Psi>
	* @param _Cleft many body state
	* @param _Cright many body state
	* @param _M many body matrix
	* @returns the overlap <\Psi|O|\Psi>
	*/
	template <typename _Ct, typename _M>
	inline inner_type_t<_Ct> applyOverlap(const _Ct& _Cleft, const _Ct& _Cright, const _M& _mat)
	{
		return arma::cdot(_Cleft, _mat * _Cright);
	}

	template <typename _Ct, typename _M>
	inline _Ct applyOverlap(const arma::subview_col<_Ct>& _Cleft, const arma::subview_col<_Ct>& _Cright, const _M& _mat)
	{
		return arma::cdot(_Cleft, _mat * _Cright);
	}

	// ##########################################################################################################################################

	/*
	* @brief The spin operator namespace. Contains the most common spin operators.
	*/
	namespace SpinOperators
	{
		std::pair<u64, double> sig_x(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_x(size_t _Ns, size_t _part);
		Operators::Operator<double> sig_x(size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_x(size_t _Ns);

		std::pair<u64, double> sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_z(size_t _Ns, size_t _part);
		Operators::Operator<double> sig_z(size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_z(size_t _Ns);
	}

	// ##########################################################################################################################################

	/*
	* @brief For Quadratic Operators, we will treat the operators as acting on the integer index as it was not the configuration!
	*/
	namespace QuadraticOperators
	{
		// -------- n_i Operators --------

		Operators::Operator<double> site_occupation(size_t _Ns, const size_t _site);
		Operators::Operator<double> site_occupation_r(size_t _Ns, const v_1d<double>& _coeffs);
		Operators::Operator<double> site_occupation_r(size_t _Ns, const v_1d<size_t>& _sites, const v_1d<double>& _coeffs);

		// -------- n_q Operators --------
		
		Operators::Operator<double> site_nq(size_t _Ns, const size_t _momentum);

		// ------ n_i n_j Operators ------

		Operators::Operator<double> nn_correlation(size_t _Ns, const size_t _site_plus, const size_t _site_minus);

		// --- quasimomentum Operators ---

		Operators::Operator<std::complex<double>> quasimomentum_occupation(size_t _Ns, const size_t _momentum);
		Operators::Operator<double> quasimomentum_occupation(size_t _Ns);

		// ----- kinectic Operators ------

		Operators::Operator<double> kinetic_energy(size_t _Nx, size_t _Ny, size_t _Nz);
	}

	// ##########################################################################################################################################

	std::pair<u64, double> sigma_x(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<double> makeSigmaX(std::shared_ptr<Lattice> lat, uint site);

	std::pair<u64, cpx> sigma_y(u64 base_vec, int L, const v_1d<uint>& sites);
	Operators::Operator<cpx> makeSigmaY(std::shared_ptr<Lattice> lat, uint site);

	/*
	* @brief multiplication of sigma_zi | state >
	* @param L lattice dimensionality (base vector length)
	* @param sites the sites to meassure correlation at
	*/
	template <typename _T>
	std::pair<u64, _T> sigma_z(u64 base_vec, int L, const v_1d<uint>& sites)
	{
		_T val = 1.0;
		for (auto const& site : sites)
			val *= checkBit(base_vec, L - 1 - site) ? Operators::_SPIN : -Operators::_SPIN;
		return std::make_pair(base_vec, val);
	}
	template <typename _T>
	Operators::Operator<_T> makeSigmaZ(std::shared_ptr<Lattice> lat, uint site)
	{
		typename _OP<_T>::GLB fun_ = [&](u64 state) { return Operators::sigma_z<_T>(state, lat->get_Ns(), { site }); };
		return Operator<_T>(lat, 1.0, fun_, SymGenerators::SZ);
	}
	
	/*
	* @brief Creates local sigma z
	*/
	template <typename _T>
	Operators::Operator<_T, uint> sigmaZ_L(std::shared_ptr<Lattice> lat)
	{
		typename _OP<_T>::LOC fun_ = [&](u64 state, uint i) { return Operators::sigma_z<_T>(state, lat->get_Ns(), { i }); };
		return Operator<_T, uint>(lat, 1.0, fun_, SymGenerators::SZ);
	}

	/*
	* @brief Creates correlation sigma z
	*/
	template <typename _T>
	Operators::Operator<_T, uint, uint> sigmaZ_C(std::shared_ptr<Lattice> lat)
	{
		typename _OP<_T>::COR fun_ = [&](u64 state, uint i, uint j) { return Operators::sigma_z<_T>(state, lat->get_Ns(), { i, j }); };
		return Operator<_T, uint, uint>(lat, 1.0, fun_, SymGenerators::SZ);
	}

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

#endif

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
