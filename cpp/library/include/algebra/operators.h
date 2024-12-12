#ifndef OPERATORS_H
#define OPERATORS_H

#include "operators/operators_final.hpp"

// ##########################################################################################################################################

namespace Operators
{
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

	/**
	* @brief Applies the many body matrix to a given state O|\Psi>
	* @param _C many body state
	* @param _M many body matrix
	*/
	template<typename _Ct, typename _M>
	inline auto apply(const _Ct& _C, const _M& _mat)
	{
		return _mat * _C;
	}

	// ##########################################################################################################################################

	/**
	* @brief Applies the overlap operation on a given vector and matrix.
	*
	* This function computes the dot product of a vector `_C` and the product of a matrix `_mat` with the vector `_C`.
	* It is only enabled for non-complex element types.
	*
	* @tparam _M Type of the matrix.
	* @tparam _Ct Type of the column vector, defaults to `arma::Col` with the element type of `_M`.
	* @param _C The column vector.
	* @param _mat The matrix.
	* @return The result of the overlap operation as an inner type of `_Ct`.
	*/
	template<typename _M, typename _Ct = arma::Col<typename _M::elem_type>>
	inline typename std::enable_if<!is_complex<typename _Ct::elem_type>::value, inner_type_t<_Ct>>::type
	applyOverlap(const _Ct& _C, const _M& _mat)
	{
		return arma::dot(_C, _mat * _C);
	}

	template<typename _M, typename _Ct = arma::Col<typename _M::elem_type>>
	inline typename std::enable_if<is_complex<typename _Ct::elem_type>::value, std::complex<double>>::type
	applyOverlap(const _Ct& _C, const _M& _mat)
	{
		return arma::cdot(_C, _mat * _C);  // For complex types, use the complex dot product.
	}

	/**
	* @brief Applies the overlap operation on a given container and a generalized matrix.
	*
	* This function computes the overlap of a container with a generalized matrix.
	* If the matrix is sparse, it uses the sparse representation for the computation.
	* Otherwise, it uses the dense representation.
	*
	* @tparam _Ct The type of the container.
	* @tparam _T2 The type of the elements in the generalized matrix (default is the element type of _Ct).
	* @param _C The container on which the overlap operation is applied.
	* @param _mat The generalized matrix used in the overlap operation.
	* @return The result of the overlap operation as an inner type of the container.
	*/
	template<typename _Ct, typename _T2 = typename _Ct::elem_type, typename = std::enable_if_t<arma::is_arma_type<_Ct>::value>>
	inline inner_type_t<_Ct> applyOverlap(const _Ct& _C, const GeneralizedMatrix<_T2>& _mat)
	{	
		if (_mat.isSparse())
			return CAST<inner_type_t<_Ct>>(arma::cdot(_C, _mat.getSparse() * _C));
		else
			return CAST<inner_type_t<_Ct>>(arma::cdot(_C, _mat.getDense() * _C));
	}

	// ##########################################################################################################################################

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
	
	namespace GeneralOperators
	{
		// ##########################################################################################################################################

		/*
		* @brief For the states in state occupation representation, check if they
		* correspond to the given state, if yes, return 1.0, otherwise 0.0.
		* @note The function is used not in the full Hilbert space representation, but in the state occupation representation.
		* Therefore, although the states are not diagonal in terms of the dot product, they are in the full Hilbert space
		* Those are represented by the integer numbers - and 1 in the state in full Hilbert space corresponds to using exactly this state.
		* @param base_vec the base vector
		* @param _proj the projected vector
		* @returns the pair of the base vector and the value of the projector
		*/
		template<typename _T>
		std::pair<u64, _T> projector(u64 base_vec, u64 _proj)
		{
			return std::make_pair(base_vec, (base_vec == _proj) ? 1.0 : 0.0);
		}
		
		/*
		* @brief For the states in state occupation representation, check if they
		* correspond to the given state, if yes, return 1.0, otherwise 0.0. 
		* @note The function is used not in the full Hilbert space representation, but in the state occupation representation.
		* Therefore, although the states are not diagonal in terms of the dot product, they are in the full Hilbert space 
		* Those are represented by the integer numbers - and 1 in the state in full Hilbert space corresponds to using exactly this state.
		* @example Let's say we have hardcore bosons on the lattice. State in full Hilbert space |0, 1, 0, 0> corresponds to the state 1 = |01> in the state occupation representation.
		* but state |0, 0, 1, 0> corresponds to the state 2 = |10> in the state occupation representation and |0, 0, 0, 1> corresponds to the state 4 = |11> in the state occupation representation.
		* @param base_vec the base vector
		* @param _proj the projected vector
		* @returns the pair of the base vector and the value of the projector
		* @note Uses vector representation!
		*/
		template <typename _T>
		std::pair<_OP_V_T, _T> projector(_OP_V_T_CR base_vec, _OP_V_T_CR _proj)
		{
			if (base_vec.n_elem != _proj.n_elem) throw std::invalid_argument("The states have different dimensions.");
			return std::make_pair(base_vec, arma::approx_equal(base_vec, _proj, "absdiff", 0.0) ? 1.0 : 0.0);
		}

		template <typename _T>
		Operators::Operator<_T> projector(size_t _Ns, u64 _proj)
		{
			_OP_V_T _projv(_Ns);
			Binary::int2base<inner_type_t<_OP_V_T>, arma::Col<inner_type_t<_OP_V_T>>, false>(_proj, _projv);

			typename _OP<_T>::GLB fun_ 		= [_proj](u64 _state) 			{ return projector<_T>(_state, _proj); };
			typename _OP_V<_T>::GLB funV_ 	= [_projv](_OP_V_T_CR _state) 	{ return projector<_T>(_state, _projv); };
			return Operator<_T>(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
		}

		template <typename _T>
		Operators::Operator<_T> projector(size_t _Ns, _OP_V_T_CR _projv)
		{
			u64 _proj = Binary::base2int<inner_type_t<_OP_V_T>, _OP_V_T, false>(_projv);

			typename _OP<_T>::GLB fun_ 		= [_proj](u64 _state) 			{ return projector<_T>(_state, _proj); };
			typename _OP_V<_T>::GLB funV_ 	= [_projv](_OP_V_T_CR _state) 	{ return projector<_T>(_state, _projv); };
			return Operator<_T>(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
		}

		template <typename _T>
		Operators::OperatorComb<_T> projectorComb(size_t _Ns, u64 _proj)
		{
			return OperatorComb<_T>(projector<_T>(_Ns, _proj));
		}

		template <typename _T>
		Operators::OperatorComb<_T> projectorComb(size_t _Ns, _OP_V_T_CR _projv)
		{
			return OperatorComb<_T>(projector<_T>(_Ns, _projv));
		}

		// ##########################################################################################################################################

		template <typename _T>
		std::pair<u64, _T> projectorSum(u64 _projectTo, u64 _state, std::function<_T(u64)> _application)
		{
			auto _val = _application(_state);
			return std::make_pair(_projectTo, _val);
		}

		template <typename _T>
		std::pair<_OP_V_T, _T> projectorSum(_OP_V_T_CR _projectTo, _OP_V_T_CR _state, std::function<_T(_OP_V_T_CR)> _application)
		{
			auto _val = _application(_state);
			return std::make_pair(_projectTo, _val);	
		}

		template <typename _T> 
		Operators::Operator<_T> projectorSum(size_t _Ns, const u64 _projectTo, std::function<_T(u64)> _application, std::function<_T(_OP_V_T_CR)> _applicationV)
		{
			_OP_V_T _projv(_Ns);
			Binary::int2base<inner_type_t<_OP_V_T>, arma::Col<inner_type_t<_OP_V_T>>, false>(_projectTo, _projv);

			typename _OP<_T>::GLB fun_ 		= [_projectTo, _application](u64 _state) 		{ return projectorSum<_T>(_projectTo, _state, _application); 	};
			typename _OP_V<_T>::GLB funV_ 	= [_projv, _applicationV](_OP_V_T_CR _state) 	{ return projectorSum<_T>(_projv, _state, _applicationV); 		};
			return Operator<_T>(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
		}

		/*
		* @brief Creates the operator that projects the state to the given state and sums the values of the operator.
		* This is: \sum _{s'} |s><s'| * pRatio[excited state](s' -> s)) 
		* @param _Ns number of sites
		* @param _projectTo the state to project to - in a vector representation
		* @param _application the function that calculates the value of the operator for a given state (in the integer representation)
		* @param _applicationV the function that calculates the value of the operator for a given state (in the vector representation)
		* @param _projectInt if the projection is in the integer representation and shall be converted to the vector representation
		* @note For NQS, the function _applicationV is the probability ratio of changing the state 
		*/
		template <typename _T, bool _projectInt = false>
		typename std::enable_if<!_projectInt, typename Operators::Operator<_T>>::type 
		projectorSum(size_t _Ns, _OP_V_T_CR _projectTo, std::function<_T(u64)> _application, 
			std::function<_T(_OP_V_T_CR)> _applicationV)
		{			
			// not really used in the function, but we need to have it
			typename _OP_V<_T>::GLB funV_ 	= [_projectTo, _applicationV](_OP_V_T_CR _state) 	{ return projectorSum<_T>(_projectTo, _state, _applicationV); };
			return Operator<_T>(_Ns, 1.0, Operator<_T>::E, funV_, SymGenerators::OTHER);
		}

		template <typename _T, bool _projectInt>
		typename std::enable_if<_projectInt, typename Operators::Operator<_T>>::type
		projectorSum(size_t _Ns, _OP_V_T_CR _projectTo, std::function<_T(u64)> _application, 
			std::function<_T(_OP_V_T_CR)> _applicationV)
		{			
			const u64 _proj = Binary::base2int<inner_type_t<_OP_V_T>, _OP_V_T, false>(_projectTo);

			// not really used in the function, but we need to have it
			typename _OP<_T>::GLB fun_ 		= [_proj, _application](u64 _state) 				{ return projectorSum<_T>(_proj, _state, _application); };
			typename _OP_V<_T>::GLB funV_ 	= [_projectTo, _applicationV](_OP_V_T_CR _state) 	{ return projectorSum<_T>(_projectTo, _state, _applicationV); };
			return Operator<_T>(_Ns, 1.0, fun_, funV_, SymGenerators::OTHER);
		}

		// ##########################################################################################################################################

		template <typename _T>
		Operators::OperatorComb<_T> projectorSumComb(size_t _Ns, const u64 _projectTo, 
				std::function<_T(u64)> _application)
		{
			auto _proj = projectorSum<_T>(_Ns, _projectTo, _application, Operators::Operator<_T>::E_V_F);
			return OperatorComb<_T>(std::move(_proj));
		}

		template <typename _T>
		Operators::OperatorComb<_T> projectorSumComb(size_t _Ns, _OP_V_T_CR _projectTo, std::function<_T(_OP_V_T_CR)> _applicationV)
		{
			auto _proj = projectorSum<_T, false>(_Ns, _projectTo, Operators::Operator<_T>::E_F, _applicationV);
			return OperatorComb<_T>(std::move(_proj));
		}
	};

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
	std::pair <_OP_V_T, _T> sigma_z(_OP_V_T_CR base_vec, int L, const v_1d<uint>& sites)
	{
		_T val = 1.0;
		for (auto const& site : sites)
			val *= Binary::check(base_vec, site) ? Operators::_SPIN : -Operators::_SPIN;
		return std::make_pair(base_vec, val);
	}

	// ##########################################################################################################################################
	
	template <typename _T>
	inline Operators::Operator<_T> makeSigmaZ(std::shared_ptr<Lattice> lat, uint site)
	{
		typename _OP<_T>::GLB fun_ = [&](u64 state) { return Operators::sigma_z<_T>(state, lat->get_Ns(), { site }); };
		return Operator<_T>(lat, 1.0, fun_, SymGenerators::SZ);
	}
	
	/*
	* @brief Creates local sigma z
	*/
	template <typename _T>
	inline Operators::Operator<_T, uint> makeSigZ_l(std::shared_ptr<Lattice> lat)
	{
		typename _OP<_T>::LOC fun_ 		= [&](u64 state, uint i) { return Operators::sigma_z<_T>(state, lat->get_Ns(), { i }); };
		typename _OP_V<_T>::LOC funV_ 	= [&](_OP_V_T_CR state, uint i) { return Operators::sigma_z<_T>(state, lat->get_Ns(), { i }); };
		return Operator<_T, uint>(lat, 1.0, fun_, SymGenerators::SZ);
	}

	/*
	* @brief Creates correlation sigma z
	*/
	template <typename _T>
	inline Operators::Operator<_T, uint, uint> sigmaZ_C(std::shared_ptr<Lattice> lat)
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

	// ##########################################################################################################################################
};

#include "operators/operator_parser.hpp"

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
