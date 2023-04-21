#pragma once

/*******************************
* Contains the possible methods
* for linear algebra usage.
*******************************/

#ifndef ALG_H
	#define ALG_H
using uint = unsigned int;
// #############################################################				   INCLUDE FROM ARMADILLO				   #############################################################

#define ARMA_WARN_LEVEL 3
#define ARMA_USE_LAPACK             
#define ARMA_PRINT_EXCEPTIONS
//#define ARMA_BLAS_LONG_LONG                                                                 // using long long inside LAPACK call
//#define ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS
//#define ARMA_DONT_USE_WRAPPER
//#define ARMA_USE_SUPERLU
//#define ARMA_USE_ARPACK 
#define ARMA_USE_MKL_ALLOC
#define ARMA_USE_MKL_TYPES
#define ARMA_DONT_USE_OPENMP
#define ARMA_USE_HDF5
////#define ARMA_USE_OPENMP
#define ARMA_ALLOW_FAKE_GCC
#define ARMA_DONT_PRINT_CXX11_WARNING
#define ARMA_DONT_PRINT_CXX03_WARNING
#include <armadillo>

#define DH5_USE_110_API
#define D_HDF5USEDLL_ 

// #############################################################				   DEFINITIONS FROM ARMADILLO				   #############################################################

#define DIAG(X)									arma::diagmat(X)
#define EYE(X)									arma::eye(X,X)
#define ZEROV(X)								arma::zeros(X)
#define ZEROM(X)								arma::zeros(X,X)
#define SUBV(X, fst, lst)						X.subvec(fst, lst)
#define SUBM(X, fstr, fstc, lstr, lstc)			X.submat(fstr, fstc, lstr, lstc)
#define UPDATEV(L, R, condition)				(condition) ? (L += R) : (L -= R)

namespace algebra {
	// #############################################################					CONJUGATE								#############################################################

	template <typename _T>
	inline auto conjugate(_T x)		-> _T		{ return std::conj(x); };
	template <>
	inline auto conjugate(double x)	-> double	{ return x; };
	template <>
	inline auto conjugate(float x)	-> float	{ return x; };
	template <>
	inline auto conjugate(int x)	-> int		{ return x; };

	// #############################################################				   MATRIX MULTIPLICATION				   #############################################################
	/*
	* @brief Allows to calculate the matrix consisting of COL vector times ROW vector
	* @param setMat matrix to set the elements onto
	* @param setVec column vector to set the elements from
	*/
	template <typename _type>
	inline void setKetBra(arma::Mat<_type>& setMat, const arma::Col<_type>& setVec) {
		setMat = arma::cdot(setVec, setVec.as_row());
	}
	
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief Allows to calculate the matrix consisting of COL vector times ROW vector
	* @param setMat matrix to set the elements onto
	* @param setVec column vector to set the elements from
	* @param plus if add or substract
	*/
	template <typename _type>
	inline void setKetBra(arma::Mat<_type>& setMat, const arma::Col<_type>& setVec, bool plus) {
		UPDATEV(setMat, arma::cdot(setVec, setVec.as_row()), plus);
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* Puts the given matrix mSet(smaller) to a specific place in the m2Set (bigger) matrix
	* @param m2Set (bigger) matrix to find the submatrix in and set it's elements
	* @param mSet (smaller) matrix to be put in the m2Set
	* @param row row of the left upper element (row,col) of M2Set
	* @param col col of the left upper element (row,col) of M2Set
	* @param update if we shall add or substract MSet elements from M2Set depending on minus parameter
	* @param minus substract?
	*/
	template <typename _type1, typename _type2>
	void setSubMFromM(arma::Mat<_type1>& m2Set, const arma::Mat<_type2>& mSet, uint row, uint col, uint nrow, uint ncol, bool update = true, bool minus = false)
	{
		if (update)
			UPDATEV(SUBM(m2Set, row, col, row + nrow, col + ncol), mSet, !minus);
		else
			SUBM(m2Set, row, col, row + nrow, col + ncol) = mSet;
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief Uses the given matrix MSet (bigger) to set the M2Set (smaller) matrix
	* @param M2Set (smaller) matrix to find the submatrix in and set it's elements
	* @param MSet (bigger) matrix to be put in the M2Set
	* @param row row of the left upper element (row,col) of MSet
	* @param col col of the left upper element (row,col) of MSet
	* @param update if we shall add or substract MSet elements from M2Set depending on minus parameter
	* @param minus substract?
	*/
	template <typename _type1, typename _type2>
	void setMFromSubM(arma::Mat<_type1>& m2Set, const arma::Mat<_type2>& mSet, uint row, uint col, uint nrow, uint ncol, bool update = true, bool minus = false)
	{
		if (update)
			UPDATEV(mSet, SUBM(m2Set, row, col, row + nrow, col + ncol), !minus);
		else
			mSet = SUBM(m2Set, row, col, row + nrow, col + ncol);
	}

	// #############################################################				   MATRIX DECOMPOSITIONS				   #############################################################

	template<typename _T>
	class UDT{
	public:
		arma::Mat<_T> U;
		arma::Col<_T> D;			// here we will put D vector - diagonal part of R
		arma::Col<_T> Di;			// here we will put D vector inverse
		arma::Mat<_T> T;
		arma::Col<_T> Db;			// Dmax
		arma::Col<_T> Ds;			// Dmin 

		~UDT() = default;
		UDT() = default;
		UDT(const UDT<_T>& o) : U(o.U), D(o.D), Di(o.Di), Db(o.Db), Ds(o.Ds), T(o.T) {};
		UDT(UDT<_T>&& o) : U(std::move(o.U)), D(std::move(o.D)), Di(std::move(o.Di)), Db(std::move(o.Db)), Ds(std::move(o.Ds)), T(std::move(o.T)) {};
		UDT(const arma::Mat<_T>& u, const arma::Col<_T>& d, const arma::Mat<_T>& t) : U(u), D(d), Di(1.0/d), Db(d), Ds(d), T(t) {};
		UDT(arma::Mat<_T>&& u, arma::Col<_T>&& d, arma::Mat<_T>&& t) : U(u), D(d), Di(1.0/d), Ds(d), Db(d), T(t) {};

		virtual void decompose() = 0;
		virtual void decompose(const arma::Mat<_T>& M) = 0;
		virtual arma::Mat<_T> eval() { return U * DIAG(D) * T;  };
	
		virtual void loh() = 0;
		virtual void loh_inv() = 0;
		virtual void loh_inplace() = 0;

		/*
		* @brief copy assignment
		*/
		UDT<_T>& operator=(const UDT<_T>& o) {
			U = o.U;
			D = o.D;
			Di = o.Di;
			Ds = o.Ds;
			Db = o.Db;
			T = o.T;
			return *this;
		};

		/*
		* @brief move assignment
		*/
		UDT<_T>& operator=(UDT<_T>&& o) {
			U = std::move(o.U);
			D = std::move(o.D);
			Di = std::move(o.Di);
			Ds = std::move(o.Ds);
			Db = std::move(o.Db);
			T = std::move(o.T);
			return *this;
		};
	
		// ----------------------------------- OPERATIONS -----------------------------------

		// ----------------------------------- INVERSE

		/*
		* @brief Calculates the inverse of the UDT decomposition of a matrix. With return.
		*/
		arma::Mat<_T> inv() { return arma::solve(T, Di) * U.t(); };

		/*
		* @brief Calculates the inverse of the UDT decomposition of a matrix. 
		* @param M matrix to set the inverse onto
		*/
		void inv(arma::Mat<_T>& M) { M = arma::solve(T, Di) * U.t(); };

		// ----------------------------------- MULTIPLICATION 

		/*
		* @brief Stabilized multiplication of two `UDT` decompositions.
		* @return UDT factorization object
		*/
		//static UDT<_T> factMult(const UDT<_T>& A, const UDT<_T>& B){
		//	UDT<_T> ret = A;
		//	ret.factMult(B);
		//	return ret;
		//}

		// -----------------------------------

		/*
		* @brief Stabilized multiplication of two `UDT` decompositions.
		*/
		void factMult(const UDT<_T>& B){
			arma::Mat<_T> mat = T * B.U; 		// Tl * Ur
			mat = D * mat;						// Rl * (*up)
			mat = mat * B.D;
			decompose(mat);
		} 
	
		virtual void factMult(const arma::Mat<_T> Ml) = 0;

		// -----------------------------------

		// ------ (1+A)^(-1) ------
		virtual arma::Mat<_T> inv1P() = 0;
		virtual void inv1P(arma::Mat<_T>& setMat) = 0;

		// ------ (A+B)^(-1) ------
		virtual arma::Mat<_T> invSum(UDT<_T>* right) = 0;
		virtual arma::Mat<_T> invSum(UDT<_T>* right, arma::Mat<_T>& setMat) = 0;
	};

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief UDT decomposition using QR decomposition
	* @cite doi:10.1016/j.laa.2010.06.023
	*/
	template<typename _T>
	class UDT_QR : public UDT<_T>{
		// arma::Mat<_T> U = Q;		// in this case the U matrix serves as Q	
		arma::Mat<_T> R;			// right triangular matrix
		arma::umat P;				// permutation matrix 

		~UDT_QR() = default;
		UDT_QR() = default;
		UDT_QR(const arma::Mat<_T>& M) : UDT<_T>()
		{
			decompose(M);
			this->Db = ZEROV(M.col(0).n_rows);
			this->Ds = ZEROV(M.col(0).n_rows);
		};
		UDT_QR(const arma::Mat<_T>& q, const arma::Mat<_T>& r, const arma::umat& p)
			: R(r), P(p), UDT<_T>(q, arma::ones(q.n_rows), ZEROM(q.n_rows))
		{
			decompose();
			this->Db = ZEROV(q.col(0).n_rows);
			this->Ds = ZEROV(q.col(0).n_rows);
		};
		UDT_QR(const UDT_QR<_T>& o): R(o.R), P(o.P), UDT<_T>(o) {};
		UDT_QR(UDT_QR<_T>&& o) noexcept : R(std::move(o.R)), P(std::move(o.P)), UDT<_T>(std::move(o)) {};

		/*
		* @brief copy assignment operator
		*/
		UDT_QR<_T>& operator=(const UDT_QR<_T>& o) { 
			UDT<_T>::operator=(o);
			R = o.R;
			P = o.P;
			this->Db = o.Db;
			this->Ds = o.Ds;
			return *this;
		}

		/*
		* @brief move assignment operator
		*/
		UDT_QR<_T>& operator=(UDT_QR<_T>&& o) { 
			UDT<_T>::operator=(std::move(o));
			R = std::move(o.R);
			P = std::move(o.P);
			this->Db = std::move(o.Db);
			this->Ds = std::move(o.Ds);
			return *this;
		}

		// ----------------------------------- DECOMPOSITIONS
	
		/*
		* @brief Create a decomposition using preset matrices
		*/
		void decompose() override {
			// inverse the vector D during setting
			this->D = R.diag();
			this->Di = 1.0 / R.diag();
			this->T = (DIAG(this->Di) * R) * P.t();
		}
	
		/*
		* @brief Create a decomposition
		* @param M Matrix to decompose
		*/
		void decompose(const arma::Mat<_T>& M) override {
			if (!arma::qr(this->U, R, P, M)) throw "Decomposition failed\n";
			decompose();
		}

		/*
		* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher.
		*/
		void loh(){	
			for (auto i = 0; i < R.n_rows; i++)
			{
				if (abs(this->D(i)) > 1.0) {
					this->Db(i) = this->D(i);	// max (R(i,i), 1)
					this->Ds(i) = 1.0;			// min (R(i,i), 1)
				}
				else {
					this->Ds(i) = this->D(i);
					this->Db(i) = 1.0;
				}
			}
		}
	
		/*
		* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher.
		* @warning Saves the inverse to Db and Ds!
		*/
		void loh_inv(){
			for (auto i = 0; i < R.n_rows; i++)
			{
				if (abs(this->D(i)) > 1.0) {
					this->Db(i) = this->Di(i);	// max (R(i,i), 1)
					this->Ds(i) = 1.0;	// min (R(i,i), 1)
				}
				else {
					this->Db(i) = 1.0;
					this->Ds(i) = this->D(i);
				}
			}
		}

		/*
		* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher.
		* @attention ->(1, R) -> we save that in R -> therefore change is only made to set R(i,i) to 1
		* @attention ->max(1, R) -> we set that as an inverse onto D already
		* @warning Changes D and R!
		*/
		void loh_inplace(){
			for (int i = 0; i < R.n_rows; i++){
				if (abs(R(i, i)) > 1)
					R(i, i) = 1;				// min(1,R(i,i))
				else 							// (abs(R(i, i)) <= 1)
					this->Di(i) = 1;			// inv of max(1,R(i,i)) because Di is already an inverse
			}
		}

		// ########################################## MULTIPLICATION ##########################################
	
		/*
		* @brief Multiply the UDT decomposition by a matrix from the left
		* @param Ml left matrix
		* @link https://github.com/carstenbauer/StableDQMC.jl/blob/master/src/qr_udt.jl
		*/
		void factMult(const arma::Mat<_T> Ml) override {
			if (!arma::qr(this->U, R, P, (Ml * this->U) * DIAG(this->D))) throw "decomposition failed\n";
			// inverse during setting
			this->D = R.diag();
			this->Di = 1.0 / this->D;
			// premultiply old T by new T from left
			this->T = ((DIAG(this->Di) * R) * P.t()) * this->T;
		}
	
		/*
		* @brief (UDT + 1)^(-1) with QR decomposition.
		*/
		arma::Mat<_T> inv1P() override {
			// decompose first
			loh_inv();
			return arma::solve(DIAG(this->Db) * this->U.t() + DIAG(this->Ds) * this->T, DIAG(this->Db) * this->U.t());
			// loh();
			// return arma::solve(arma::inv(DIAG(Db)) * U.t() + DIAG(Ds) * T, arma::inv(DIAG(Db)) * U.t());
			// without the decomposition
			// return = arma::inv(DIAG(Di) * U.t() + D * T) * DIAG(D_up) * Q_up.t();
		}
	
		void inv1P(arma::Mat<_T>& setMat) override { setMat = inv1P(); };

		// ########################################## (A+B)^(-1) ##########################################
	
		/*
		* @brief Stabilized calculation of [UaDaTa + UbDbTb]^{-1}
		*/
		arma::Mat<_T> invSum(UDT<_T>* right) override {
			// calculate loh decomposition
			loh();
			right->loh();

			// dimension
			const auto d = this->Ds.n_elem;

			// matL = D_min_a * Ta * Tb^{-1} / D_max_b
			arma::Mat<_T> matL = this->T * arma::inv(arma::trimatu(right->T));
			for(int i = 0; i < d; i++)
				for(int j = 0; j < d; j++)
					matL(i,j) *= this->Ds(i) / right->Db(j);

			// matR = 1/(D_max_a) * Ua^\dag * Ub * D_min_b
			arma::Mat<_T> matR = this->U.t() * right->U;
			for(int i = 0; i < d; i++)
				for(int j = 0; j < d; j++)
					matL(i,j) *= right->Ds(i) / this->Db(j);

			// add two matrices
			matL += matR;

			// create inner decomposition arma::solve(T, Di)
			UDT_QR<_T> inner(matL);
			matR = DIAG(inner.D) * inner.T;//arma::solve(inner.T, DIAG(inner.Di)); //arma::inv(arma::trimatu() * DIAG(inner.Di);
			matR = arma::solve(matR, this->U.t());
			for(int i = 0; i < d; i++)
				for(int j = 0; j < d; j++)
					matR(i,j) /= right->Db(i) * this->Db(j);
		
			// decompose again
			inner.decompose(matR);
			inner.U = arma::solve(arma::trimatu(right->T), inner.U);
			inner.T = inner.T * this->U.t();

			// return evaluated one
			return inner.eval();
		}
		arma::Mat<_T> invSum(UDT<_T>* right, arma::Mat<_T>& setMat) override
		{
			setMat = invSum(right);
		}

	};
};
#endif