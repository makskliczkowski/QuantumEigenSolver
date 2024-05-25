#pragma once
/***************************************
* Defines the Hamiltonian Matrix override
* for sparse and dense matrices.
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#ifndef SYMMETRIES_H
#	include "algebra/operators.h"
#endif // !SYMMETRIES_H

#ifndef SYSTEM_PROPERTIES_H
#	include "quantities/statistics.h"
#endif // !SYSTEM_PROPERTIES_H

// ############################################################################################################

/*
* @brief This class will allow to create Hamiltonian matrices for different models within the sparse and dense representation.
* @tparam _T: Type of the matrix elements.
*/
template <typename _T>
class HamiltonianMatrix
{
public:
	size_t n_rows	= 0;
	size_t n_cols	= 0;

	// sparsity flag
	bool isSparse_	= true;
protected:
	u64 Nh_ = 1;

	// matrices
	arma::SpMat<_T> H_sparse_;
	arma::Mat<_T> H_dense_;

public:

	// ##################################

	// Constructors

	// Destructor
	~HamiltonianMatrix()
	{
		DESTRUCTOR_CALL;
	}

	// Default constructor
	HamiltonianMatrix() = default;

	// Constructor for distinguishing between sparse and dense matrices
	HamiltonianMatrix(u64 _Nh, bool _isSparse = true)
		: isSparse_(_isSparse), Nh_(_Nh)
	{
		if (isSparse_)
			H_sparse_ = arma::SpMat<_T>(Nh_, Nh_);
		else
			H_dense_ = arma::Mat<_T>(Nh_, Nh_, arma::fill::zeros);
		this->n_cols	= _Nh;
		this->n_rows	= _Nh;
	}

	// Constructor for dense matrices
	HamiltonianMatrix(const arma::Mat<_T>& _H)
		: isSparse_(false), Nh_(_H.n_rows), H_dense_(_H)
	{
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		CONSTRUCTOR_CALL;
	}

	// Constructor for sparse matrices
	HamiltonianMatrix(const arma::SpMat<_T>& _H)
		: isSparse_(true), Nh_(_H.n_rows), H_sparse_(_H)
	{
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		CONSTRUCTOR_CALL;
	}

	// copy constructor
	HamiltonianMatrix(const HamiltonianMatrix& _H)
	{
		this->isSparse_ = _H.isSparse_;
		this->Nh_		= _H.Nh_;
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;

		// copy the matrix
		if (this->isSparse_)
		{
			this->H_dense_.clear();
			this->H_sparse_ = _H.H_sparse_;
		}
		else
		{
			this->H_sparse_.clear();
			this->H_dense_	= _H.H_dense_;
		}
	}

	// move constructor
	HamiltonianMatrix(HamiltonianMatrix&& _H) noexcept
	{
		this->isSparse_ = _H.isSparse_;
		this->Nh_		= _H.Nh_;
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;

		// move the matrix
		if (this->isSparse_)
		{
			this->H_dense_.clear();
			this->H_sparse_ = std::move(_H.H_sparse_);
		}
		else
		{
			this->H_sparse_.clear();
			this->H_dense_ = std::move(_H.H_dense_);
		}
	}

	// ##################################

	// Access operator
	_T operator()(u64 row, u64 col) const 
	{
		if(this->isSparse_)
			return this->H_sparse_(row, col);
		return this->H_dense_(row, col);
	}

	// Modification operator
	_T& operator()(u64 row, u64 col) 
	{
		if (this->isSparse_)
		{
			LOG_ERROR("Cannot modify sparse matrix directly.");
			return this->H_sparse_(row, col);
		}
		return this->H_dense_(row, col);
	}

	// ##################################

	// Getters
	auto t()				const -> HamiltonianMatrix<_T>		{ return this->isSparse_ ? HamiltonianMatrix<_T>(this->H_sparse_.t()) : HamiltonianMatrix<_T>(this->H_dense_.t()); }
	auto st()				const -> HamiltonianMatrix<_T>		{ return this->isSparse_ ? HamiltonianMatrix<_T>(this->H_sparse_.st()) : HamiltonianMatrix<_T>(this->H_dense_.st()); }
	auto meanLevelSpacing()	const -> double;
	auto diag()				const -> arma::Col<_T>;
	auto diag(size_t _k)	const -> arma::Col<_T>;
	auto diagD()			-> arma::diagview<_T>				{ return this->H_dense_.diag(); }
	auto diagD(size_t k)	-> arma::diagview<_T>				{ return this->H_dense_.diag(k); }
	auto diagSp()			-> arma::spdiagview<_T>				{ return this->H_sparse_.diag(); }
	auto diagSp(size_t k)	-> arma::spdiagview<_T>				{ return this->H_sparse_.diag(k); }
	auto getNh()			const -> u64						{ return this->Nh_; }
	auto isSparse()			const -> bool						{ return this->isSparse_; }
	auto getSparse()		-> arma::SpMat<_T>&					{ return this->H_sparse_; }
	auto getDense()			-> arma::Mat<_T>&					{ return this->H_dense_; }
	auto size()				const -> u64						{ return this->Nh_; }
	// Method to convert to dense matrix
	auto toDense()			const -> arma::Mat<_T>				{ return arma::Mat<_T>(H_sparse_); }

	// Method to convert to sparse matrix
	auto toSparse()			const -> arma::SpMat<_T>			{ return arma::SpMat<_T>(H_dense_); }
	auto symmetrize()		-> void;

	// Setters
	void setSparse(const arma::SpMat<_T>& _H)					{ this->isSparse_ = true; this->H_dense_.clear(); this->H_sparse_ = _H; }
	void setDense(const arma::Mat<_T>& _H)						{ this->isSparse_ = false; this->H_sparse_.clear(); this->H_dense_ = _H; }

	// ##################################
	
	template<typename _T2>
	void set(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) = _val;
		else
			this->H_dense_(_row, _col) = _val;
	}
	template<typename _T2>
	void add(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) += _val;
		else
			this->H_dense_(_row, _col) += _val;
	}
	template<typename _T2>
	void sub(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) -= _val;
		else
			this->H_dense_(_row, _col) -= _val;
	}
	template<typename _T2>
	void mul(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) *= _val;
		else
			this->H_dense_(_row, _col) *= _val;
	}
	template<typename _T2>
	void div(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) /= _val;
		else
			this->H_dense_(_row, _col) /= _val;
	}

	// ##################################

	// Overloaded operators
	
	// Copy assignment operator
	HamiltonianMatrix<_T>& operator=(const HamiltonianMatrix<_T>& other) 
	{
		// Check for self-assignment
		if (this != &other) 
		{  
			this->isSparse_		= other.isSparse_;
			this->Nh_			= other.Nh_;
			if (isSparse_)
			{
				this->H_dense_.clear();
				this->H_sparse_	= other.H_sparse_;
			}
			else
			{
				this->H_sparse_.clear();
				this->H_dense_	= other.H_dense_;
			}
		}
		return *this;
	}

	// Move assignment operator
	HamiltonianMatrix<_T>& operator=(HamiltonianMatrix<_T>&& other) noexcept 
	{
		// Check for self-assignment
		if (this != &other) 
		{  
			this->isSparse_		= other.isSparse_;
			this->Nh_			= other.Nh_;
			this->n_cols		= other.n_cols;
			this->n_rows		= other.n_rows;
			if (isSparse_)
			{
				this->H_dense_.clear();
				this->H_sparse_	= std::move(other.H_sparse_);
			}
			else
			{
				this->H_sparse_.clear();
				this->H_dense_	= std::move(other.H_dense_);
			}
		}
		return *this;
	}

	HamiltonianMatrix<_T>& operator=(const arma::Mat<_T>& _H)
	{
		this->isSparse_		= false;
		this->H_sparse_.clear();
		this->H_dense_		= _H;
		this->n_cols		= _H.n_cols;
		this->n_rows		= _H.n_rows;
		return *this;
	}

	HamiltonianMatrix<_T>& operator=(const arma::SpMat<_T>& _H)
	{
		this->isSparse_ = true;
		this->H_dense_.clear();
		this->H_sparse_ = _H;
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		return *this;
	}

	HamiltonianMatrix<_T>& operator=(const arma::Mat<_T>&& _H)
	{
		this->isSparse_ = false;
		this->H_sparse_.clear();
		this->H_dense_	= std::move(_H);
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		return *this;
	}

	HamiltonianMatrix<_T>& operator=(const arma::SpMat<_T>&& _H)
	{
		this->isSparse_ = true;
		this->H_dense_.clear();
		this->H_sparse_ = std::move(_H);
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		return *this;
	}
	// ##################################
	
	// Operator to multiply by a column vector (or anything else I guess)
	template<typename _OtherType>
	auto operator*(const _OtherType& other) const 
	{
		// check the same type of matrix
		if constexpr (std::is_same_v<_OtherType, arma::Mat<_T>> || std::is_same_v<_OtherType, arma::SpMat<_T>>) 
		{
			if (other.n_rows != this->Nh_) 
				throw std::invalid_argument("Matrix rows must match vector/matrix rows.");
			if (this->isSparse_)
				return this->H_sparse_* other;
			else
				return this->H_dense_* other;		
		} 
		// subviews and columns
		else if constexpr (std::is_same_v<_OtherType, arma::Col<_T>> || std::is_same_v<_OtherType, arma::subview_col<_T>>) 
		{
			if (other.n_rows != this->Nh_) 
				throw std::invalid_argument("Vector size must match matrix size.");
			if (this->isSparse_)
				return static_cast<arma::Col<_T>>(this->H_sparse_ * other);
			else
				return static_cast<arma::Col<_T>>(this->H_dense_ * other);
		} 
		// for the other subviews
		else if constexpr (std::is_same_v<_OtherType, arma::subview<_T>>) 
		{
			if (other.n_rows != this->Nh_)
				throw std::invalid_argument("Subview size must match matrix size.");
			if (this->isSparse_)
				return static_cast<arma::Col<_T>>(this->H_sparse_* other);
			else
				return static_cast<arma::Col<_T>>(this->H_dense_* other);		
		} 
		else 
		{
			static_assert(arma::is_arma_type<_OtherType>::value, "Invalid type for matrix multiplication.");
		}
		if (this->isSparse_)
			return static_cast<arma::Col<_T>>(this->H_sparse_* other);
		else
			return static_cast<arma::Col<_T>>(this->H_dense_* other);	
	}

	// Override addition operator for dense and sparse matrices
	template<typename _T2>
	HamiltonianMatrix<typename std::common_type<_T, _T2>> operator+(const HamiltonianMatrix<_T2>& other) const 
	{
		using _common = typename std::common_type<_T, _T2>;

		HamiltonianMatrix<_common> result(this->Nh_, this->isSparse_);

		if (this->isSparse_ && other.isSparse_)
			result.H_sparse_ = algebra::cast<_common>(this->H_sparse_) + algebra::cast<_common>(other.H_sparse_);
		else if (!this->isSparse_ && !other.isSparse_)
			result.H_dense_ = algebra::cast<_common>(this->H_dense_) + algebra::cast<_common>(other.H_dense_);
		else 
		{
			// Convert sparse to dense or dense to sparse for addition
			result.reset();
			result.H_dense_ = arma::Mat<_common>(algebra::cast<_common>(H_sparse_)) + algebra::cast<_common>(other.H_dense_);
		}
		return result;
	}

	// Addition operator overload to add HamiltonianMatrix with arma::Mat
	template<typename _T2>
	friend HamiltonianMatrix<typename std::common_type<_T, _T2>> operator+(const HamiltonianMatrix<_T2>& lhs, const arma::Mat<_T2>& rhs)
	{
		using _common = typename std::common_type<_T, _T2>;
		HamiltonianMatrix<_common> result(lhs);
		if (lhs.isSparse_)
			result.H_sparse_ += algebra::cast<_common>(rhs);
		else
			result.H_dense_ += algebra::cast<_common>(rhs);
		return result;
	}

	HamiltonianMatrix<_T>& operator+=(const arma::Mat<_T>& rhs)
	{
		if (this->isSparse_)
			this->H_sparse_ += algebra::cast<_T>(rhs);
		else
			this->H_dense_ += algebra::cast<_T>(rhs);
		return *this;
	}

	// Addition operator overload to add HamiltonianMatrix with arma::SpMat
	template<typename _T2>
	friend HamiltonianMatrix<typename std::common_type<_T, _T2>> operator+(const HamiltonianMatrix<_T2>& lhs, const arma::SpMat<_T2>& rhs)
	{
		using _common = typename std::common_type<_T, _T2>;
		HamiltonianMatrix<_common> result(lhs);
		if (lhs.isSparse_)
			result.H_sparse_ += algebra::cast<_common>(rhs);
		else
			result.H_dense_ += algebra::cast<_common>(rhs);
		return result;
	}

	HamiltonianMatrix<_T>& operator+=(const arma::SpMat<_T>& rhs)
	{
		if (this->isSparse_)
			this->H_sparse_ += algebra::cast<_T>(rhs);
		else
			this->H_dense_ += algebra::cast<_T>(rhs);
		return *this;
	}

	// ##################################

	void print() const 
	{
		if (isSparse_) 
			this->H_sparse_.print("Sparse Matrix:");
		else 
			this->H_dense_.print("Dense Matrix:");
	}

	void reset()
	{
		this->H_sparse_.clear();
		this->H_dense_.clear();
	}

	// ##################################

	// Override cast operator to convert to arma::Mat<_T>
	operator arma::Mat<_T>() const 
	{
		if(this->isSparse_)
			return arma::Mat<_T>(H_sparse_);
		return H_dense_;
	}

	// Override cast operator to convert to arma::SpMat<_T>
	operator arma::SpMat<_T>() const
	{
		if (this->isSparse_)
			return H_sparse_;
		return arma::SpMat<_T>(H_dense_);
	}
};

// ############################################################################################################

template<typename _T>
inline auto HamiltonianMatrix<_T>::meanLevelSpacing() const -> double
{
	if(this->isSparse_)
		return SystemProperties::mean_lvl_gamma(this->H_sparse_);
	return SystemProperties::mean_lvl_gamma(this->H_dense_);
}

// ############################################################################################################

template<typename _T>
inline arma::Col<_T> HamiltonianMatrix<_T>::diag() const
{
	if (this->isSparse_)
		return static_cast<arma::Col<_T>>(this->H_sparse_.diag());
	return this->H_dense_.diag();
}

// ############################################################################################################

template<typename _T>
inline arma::Col<_T> HamiltonianMatrix<_T>::diag(size_t _k) const
{
	if (this->isSparse_)
		return static_cast<arma::Col<_T>>(this->H_sparse_.diag(_k));
	return this->H_dense_.diag(_k);
}

// ############################################################################################################

template<typename _T>
inline auto HamiltonianMatrix<_T>::symmetrize() -> void
{
	if (this->isSparse_)
		this->H_sparse_ = 0.5 * (this->H_sparse_ + this->H_sparse_.t());
	else
		this->H_dense_ = 0.5 * (this->H_dense_ + this->H_dense_.t());
}

// ############################################################################################################