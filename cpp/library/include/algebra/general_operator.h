/***************************************
* Defines a class for a general operator.
* It can be later expanded for some more
* complicated operators acting on Hiblert
* space.
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#pragma once

// =====================================
#ifndef GENERAL_OPERATOR_H
#define GENERAL_OPERATOR_H
#include <memory>

#ifndef LATTICE_H
#include "../../source/src/lattices.h"
#endif

#ifndef BINARY_H
#include "../../source/src/binary.h"
#endif

#ifndef OPALG_H
#include "operator_algebra.h"
#endif

// ################################################################ FORWARD DECLARATIONS ########################################################################

namespace Hilbert {
	template <typename _T, uint _spinModes = 2>
	class HilbertSpace;
};

// ################################################ D E F I N I T I O N S   O F   I M P L E M E N T A T I O N ###################################################

namespace Operators {

	// ##########################################################################################################################################

	/*
	* @brief Standarizes the operator so that it is traceless and has a unit norm 1. The norm is calculated as the Frobenius norm.
	* @param _mat the matrix to standarize 
	*/
	template<typename _MatT>
	inline void standarizeOperator(_MatT& _mat)
	{
		auto _nrows = _mat.n_rows;
		_mat.diag() -= arma::trace(_mat) / double(_nrows);
		auto _Hs	= arma::trace(arma::square(_mat)) / (double)_nrows;
		_mat		= _mat / std::sqrt(_Hs);	
	}

	// ##########################################################################################################################################

	/*
	* @brief Prints the operator information. This is only available in debug mode.
	* @param _mat the matrix to print the information about
	*/
	template<typename _T, template<class _T2 = _T> typename _MatT>
	inline void operatorInfo(const _MatT<_T>& _mat)
	{
#ifdef _DEBUG
		try {
			stout << "---------------"				<< EL;
			stout << "Operator Info: "				<< EL;
			stout << "Operator Trace: "				<< arma::trace(_mat) << EL;
			stout << "Operator Frobenius Norm: "	<< arma::trace(_mat * _mat) / (double)_mat.n_rows << EL;
			stout << "Operator Max: "				<< arma::max(_mat) << EL;
			stout << "Operator Min: "				<< arma::min(_mat) << EL;
			stout << "Operator Mean: "				<< arma::mean(_mat) << EL;
			stout << "---------------"				<< EL;
		}
		catch (const std::exception& e)
		{
			LOGINFO("Error in operatorInfo: " + std::string(e.what()), LOG_TYPES::ERROR, 0);
		}
#endif
	}

	template<typename _T>
	inline void operatorInfo(const GeneralizedMatrix<_T>& _mat)
	{
#ifdef _DEBUG
		try {
			stout << "---------------"				<< EL;
			stout << "Operator Info: "				<< EL;
			stout << "Operator Trace: "				<< _mat.trace() << EL;
			stout << "Operator Frobenius Norm: "	<< _mat.hsnorm() << EL;
			stout << "Operator Max: "				<< _mat.max() << EL;
			stout << "Operator Min: "				<< _mat.min() << EL;
			stout << "Operator Mean: "				<< _mat.mean() << EL;
			stout << "---------------"				<< EL;
		}
		catch (const std::exception& e)
		{
			LOGINFO("Error in operatorInfo: " + std::string(e.what()), LOG_TYPES::ERROR, 0);
		}
#endif
	}

	// ################################################################# G E N E R A L ##########################################################################


	/*
	* @brief Implemented symmetry types
	*/
	enum SymGenerators { E, T, Tr, R, PX, PY, PZ, OTHER, SX, SY, SZ };

	BEGIN_ENUM(SymGenerators)
	{
		DECL_ENUM_ELEMENT(E),
		DECL_ENUM_ELEMENT(T),
		DECL_ENUM_ELEMENT(Tr),
		DECL_ENUM_ELEMENT(R),
		DECL_ENUM_ELEMENT(PX),
		DECL_ENUM_ELEMENT(PY),
		DECL_ENUM_ELEMENT(PZ),
		// other
		DECL_ENUM_ELEMENT(OTHER),
		DECL_ENUM_ELEMENT(SX),
		DECL_ENUM_ELEMENT(SY),
		DECL_ENUM_ELEMENT(SZ)
	}
	END_ENUM(SymGenerators);

	/*
	* @brief Implemented fermionic operators
	*/
	enum FermionicOperators { C_UP_DAG, C_UP, C_DOWN_DAG, C_DOWN };

	BEGIN_ENUM(FermionicOperators)
	{
		DECL_ENUM_ELEMENT(C_UP_DAG),
		DECL_ENUM_ELEMENT(C_UP),
		DECL_ENUM_ELEMENT(C_DOWN_DAG),
		DECL_ENUM_ELEMENT(C_DOWN),
	}
	END_ENUM(FermionicOperators);
};

// ################################################################### G E N E R A L ############################################################################

namespace Operators {

	/*
	* @brief A class describing the general operator. It can be later expanded for some more complicated operators acting on Hiblert space or other spaces.
	* Can take several template arguments - the first one is the type of the operator, the rest are the potential non-local variables (like the index of the state (site) etc.)
	* The operator can be acting on the states or can be a matrix operator.
	*/
	template <typename _T, typename repType, typename repTypeV, typename ..._Ts>
	class GeneralOperator
	{
	public:
		/*
		* @brief Function for creating the matrix (overriden). If this is set, the matrix will be created using this function. Otherwise it is created 
		* via standard procedure of acting on the states. This is useful for the operators that are not acting on the states directly or are random operators.
		*/
		GeneralizedMatrixFunction<_T> overridenMatFun_;		

		// ====================================================================================================

	public:
		// -------- LATTICE OR GRAPH RELATED --------
		size_t Ns_											=			1;					// number of elements in the vector (for one to know how to act on it)
		std::shared_ptr<Lattice> lat_;														// lattice type to be used later on, !! the lattice can be empty if not needed !!

		// -------------- EIGENVALUE ----------------
		_T eigVal_											=			1.0;				// eigenvalue for symmetry generator (if there is an inner value)

		// --------------- VECTORS ------------------
		using _VT 											=			Operators::_OP_V_T;	// type of the vector to be used for the operator
		using _VT_CR 										=			const _VT&;			// type of the vector to be used for the operator - const reference
		bool hasVectorFun_									=			false;				// if the operator has a vector function - this is then set to true

		// -------------- QUADRATIC -----------------
		bool isQuadratic_									=			false;				// based on this, we will create the operator differently (we want act on a many body state per se)

		// -------------- SYMMETRY ------------------
		// used for checking on which states the operator acts when forgetting and using the matrix only
		u64 acton_											=			0;					// check on states the operator acts, this is stored as a number and the bitmask is applied! For many body
		SymGenerators name_									=			SymGenerators::E;   // name of the operator
		std::string nameS_									=			"E";				// name of the operator in the string

		// ====================================================================================================

	public:
				
		using repType_ 										= 			repType;
		using repTypeV_ 									= 			repTypeV;
		repType_ fun_;																		// function allowing to use the symmetry operation
		repTypeV_ funV_;																	// function allowing to use the symmetry operation - for vectors

		// ====================================================================================================
	
	public:
		
		// return type of the operator
	    using ReturnType 									= 			decltype(std::declval<repType>()(std::declval<u64>(), std::declval<_Ts>()...));
		using ReturnTypeV 									= 			decltype(std::declval<repTypeV>()(std::declval<_VT_CR>(), std::declval<_Ts>()...));

		// ====================================================================================================

	public:
		virtual ~GeneralOperator()							=			default;			// default destructor

		// ----------------------------------------------------------------------------------------------------

		// Default constructor
		GeneralOperator() 
		{
			this->init();
		}

		// Constructor with state size and optional name
		GeneralOperator(size_t Ns, const std::string& _nameS = "")
			: GeneralOperator()
		{
			this->Ns_ = Ns;
			this->nameS_ = _nameS;
		}

		// Constructor with eigenvalue and optional name
		GeneralOperator(size_t Ns, _T _eigVal, const std::string& _nameS = "")
			: GeneralOperator(Ns, _nameS) 
		{
			this->eigVal_ = _eigVal;
		}

		// Constructor with eigenvalue, name, and optional string
		GeneralOperator(size_t Ns, _T _eigVal, SymGenerators _name, const std::string& _nameS = "")
			: GeneralOperator(Ns, _eigVal, _nameS) 
		{
			this->name_ = _name;
		}

		// Constructor with lattice (shared_ptr) and optional name
		GeneralOperator(std::shared_ptr<Lattice> _lat, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS) 
		{
			this->lat_ = _lat;
		}

		// Constructor with lattice (raw pointer) and optional name
		GeneralOperator(Lattice* _lat, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ = std::make_shared<Lattice>(_lat);
		}

		// Constructor with lattice, eigenvalue, and optional name
		GeneralOperator(std::shared_ptr<Lattice> _lat, _T _eigVal, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _eigVal, _nameS) 
		{
			this->lat_ = _lat;
		}

		// Same as above, but with a raw pointer lattice
		GeneralOperator(Lattice* _lat, _T _eigVal, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _eigVal, _nameS) 
		{
			this->lat_ = std::make_shared<Lattice>(_lat);
		}

		// Constructor with lattice, eigenvalue, name, and optional string
		GeneralOperator(std::shared_ptr<Lattice> _lat, _T _eigVal, SymGenerators _name, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _eigVal, _name, _nameS) 
		{
			this->lat_ = _lat;
		}

		// Same as above, but with a raw pointer lattice
		GeneralOperator(Lattice* _lat, _T _eigVal, SymGenerators _name, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _eigVal, _name, _nameS) 
		{
			this->lat_ = std::make_shared<Lattice>(_lat);
		}
		// ----------------------------------------------------------------------------------------------------

		// Constructor with lattice, function, and optional name
		GeneralOperator(std::shared_ptr<Lattice> _lat, repType _fun, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ = _lat;
			this->fun_ = _fun;
			this->init();
		}

		GeneralOperator(Lattice* _lat, repType _fun, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ = std::make_shared<Lattice>(_lat);
			this->fun_ = _fun;
			this->init();
		}

		// ----------------------------------------------------------------------------------------------------

		// Constructor with lattice, function, vector function, and optional name
		GeneralOperator(std::shared_ptr<Lattice> _lat, repType _fun, repTypeV _funV, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_			= _lat;
			this->fun_			= _fun;
			this->funV_			= _funV;
			this->hasVectorFun_ = true;
			this->init();
		}

		GeneralOperator(Lattice* _lat, repType _fun, repTypeV _funV, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_			= std::make_shared<Lattice>(_lat);
			this->fun_			= _fun;
			this->funV_			= _funV;
			this->hasVectorFun_ = true;
			this->init();
		}

		// ----------------------------------------------------------------------------------------------------

		// Constructor with lattice, eigenvalue, function, and optional name
		GeneralOperator(std::shared_ptr<Lattice> _lat, _T _eigVal, repType _fun, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ 		= _lat;
			this->eigVal_ 	= _eigVal;
			this->fun_ 		= _fun;
			this->init();
		}

		// Constructor with lattice, eigenvalue, function, and optional name
		GeneralOperator(Lattice* _lat, _T _eigVal, repType _fun, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ 		= std::make_shared<Lattice>(_lat);
			this->eigVal_ 	= _eigVal;
			this->fun_ 		= _fun;
			this->init();
		}

		// ----------------------------------------------------------------------------------------------------

		// Constructor with lattice, eigenvalue, function, vector function, and optional name
		GeneralOperator(std::shared_ptr<Lattice> _lat, _T _eigVal, repType _fun, repTypeV _funV, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ 			= _lat;
			this->eigVal_ 		= _eigVal;
			this->fun_ 			= _fun;
			this->funV_ 		= _funV;
			this->hasVectorFun_ = true;
			this->init();
		}

		// Constructor with lattice, eigenvalue, function, vector function, and optional name
		GeneralOperator(Lattice* _lat, _T _eigVal, repType _fun, repTypeV _funV, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ 			= std::make_shared<Lattice>(_lat);
			this->eigVal_ 		= _eigVal;
			this->fun_ 			= _fun;
			this->funV_ 		= _funV;
			this->hasVectorFun_ = true;
			this->init();
		}

		// ----------------------------------------------------------------------------------------------------

		// Constructor with lattice, eigenvalue, function, name, and optional name
		GeneralOperator(std::shared_ptr<Lattice> _lat, _T _eigVal, repType _fun, SymGenerators _name, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ 			= _lat;
			this->eigVal_ 		= _eigVal;
			this->name_ 		= _name;
			this->fun_ 			= _fun;
			this->init();
		}

		// Constructor with lattice, eigenvalue, function, name, and optional name
		GeneralOperator(Lattice* _lat, _T _eigVal, repType _fun, SymGenerators _name, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ 			= std::make_shared<Lattice>(_lat);
			this->eigVal_ 		= _eigVal;
			this->name_ 		= _name;
			this->fun_ 			= _fun;
			this->init();
		}

		// ----------------------------------------------------------------------------------------------------

		// Constructor with lattice, eigenvalue, function, vector function, name, and optional name
		GeneralOperator(std::shared_ptr<Lattice> _lat, _T _eigVal, repType _fun, repTypeV _funV, SymGenerators _name, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ 			= _lat;
			this->eigVal_ 		= _eigVal;
			this->name_ 		= _name;
			this->fun_ 			= _fun;
			this->funV_ 		= _funV;
			this->hasVectorFun_ = true;
			this->init();
		}

		// Constructor with lattice, eigenvalue, function, vector function, name, and optional name
		GeneralOperator(Lattice* _lat, _T _eigVal, repType _fun, repTypeV _funV, SymGenerators _name, const std::string& _nameS = "")
			: GeneralOperator(_lat->get_Ns(), _nameS)
		{
			this->lat_ 			= std::make_shared<Lattice>(_lat);
			this->eigVal_ 		= _eigVal;
			this->name_ 		= _name;
			this->fun_ 			= _fun;
			this->funV_ 		= _funV;			
			this->hasVectorFun_ = true;
			this->init();
		}		

		// ----------------------------------------------------------------------------------------------------

		// Constructor with function
		GeneralOperator(size_t Ns, repType _fun, const std::string& _nameS = "")
			: GeneralOperator(Ns, _nameS) 
		{
			fun_ = _fun;
		}

		// Constructor with function and vector function
		GeneralOperator(size_t Ns, repType _fun, repTypeV _funV, const std::string& _nameS = "")
			: GeneralOperator(Ns, _fun, _nameS) 
		{
			funV_ = _funV;
			hasVectorFun_ = true;
		}

		// Constructor with function, eigenvalue, and optional name
		GeneralOperator(size_t Ns, _T _eigVal, repType _fun, const std::string& _nameS = "")
			: GeneralOperator(Ns, _eigVal, _nameS) 
		{
			fun_ = _fun;
		}

		// Constructor with function, eigenvalue, and vector function
		GeneralOperator(size_t Ns, _T _eigVal, repType _fun, repTypeV _funV, const std::string& _nameS = "")
			: GeneralOperator(Ns, _eigVal, _fun, _nameS) 
		{
			funV_ = _funV;
			hasVectorFun_ = true;
		}

		// Constructor with eigenvalue, function, name, and optional name
		GeneralOperator(size_t Ns, _T _eigVal, repType _fun, SymGenerators _name, const std::string& _nameS = "")
			: GeneralOperator(Ns, _nameS)
		{
			this->eigVal_ 		= _eigVal;
			this->name_ 		= _name;
			this->fun_		 	= _fun;
			this->init();
		}

		// Constructor with eigenvalue, function, vector function, name, and optional name
		GeneralOperator(size_t Ns, _T _eigVal, repType _fun, repTypeV _funV, SymGenerators _name, const std::string& _nameS = "")
			: GeneralOperator(Ns, _nameS)
		{
			this->eigVal_ 		= _eigVal;
			this->name_ 		= _name;
			this->fun_		 	= _fun;
			this->funV_ 		= _funV;
			this->hasVectorFun_ = true;
			this->init();
		}

		// Move constructor
		GeneralOperator(GeneralOperator<_T, repType, repTypeV, _Ts...>&& o)
			: overridenMatFun_(std::move(o.overridenMatFun_)),
			Ns_(std::move(o.Ns_)),
			lat_(std::move(o.lat_)),
			eigVal_(std::move(o.eigVal_)),
			hasVectorFun_(std::move(o.hasVectorFun_)),
			isQuadratic_(std::move(o.isQuadratic_)),
			acton_(std::move(o.acton_)),
			name_(std::move(o.name_)),
			nameS_(std::move(o.nameS_)) 
		{
			this->init();
		}

		// Copy constructor
		GeneralOperator(const GeneralOperator<_T, repType, repTypeV, _Ts...>& o)
			: overridenMatFun_(o.overridenMatFun_),
			Ns_(o.Ns_),
			lat_(o.lat_),
			eigVal_(o.eigVal_),
			hasVectorFun_(o.hasVectorFun_),
			isQuadratic_(o.isQuadratic_),
			acton_(o.acton_),
			name_(o.name_),
			nameS_(o.nameS_) 
		{
			this->init();
		}

		// ----------------------------------------------------------------------------------------------------
	public:
		// operators = and move operators

		GeneralOperator<_T, repType, repTypeV, _Ts...>& operator=(const GeneralOperator<_T, repType, repTypeV, _Ts...>& _other)
		{
			if (this != &_other)
			{
				this->Ns_			=		_other.Ns_;
				this->lat_			=		_other.lat_;
				this->eigVal_		=		_other.eigVal_;
				this->hasVectorFun_ =		_other.hasVectorFun_;
				this->isQuadratic_	=		_other.isQuadratic_;
				this->acton_		=		_other.acton_;
				this->name_			=		_other.name_;
				this->nameS_		=		_other.nameS_;
				this->overridenMatFun_ =	_other.overridenMatFun_;
				// copy the functions
				this->fun_			=		_other.fun_;
				this->funV_			=		_other.funV_;
			}
			return *this;
		}

		// ----------------------------------------------------------------------------------------------------

		GeneralOperator<_T, repType, repTypeV, _Ts...>& operator=(const GeneralOperator<_T, repType, repTypeV, _Ts...>&& _other)
		{
			if (this != &_other)
			{
				this->Ns_			=		std::move(_other.Ns_);
				this->lat_			=		std::move(_other.lat_);
				this->eigVal_		=		std::move(_other.eigVal_);
				this->hasVectorFun_ =		std::move(_other.hasVectorFun_);
				this->isQuadratic_	=		std::move(_other.isQuadratic_);
				this->acton_		=		std::move(_other.acton_);
				this->name_			=		std::move(_other.name_);
				this->nameS_		=		std::move(_other.nameS_);
				this->overridenMatFun_ =	std::move(_other.overridenMatFun_);

				// copy the functions
				this->fun_			=		std::move(_other.fun_);
				this->funV_			=		std::move(_other.funV_);
			}
			return *this;
		}

		// ----------------------------------------------------------------------------------------------------

	public:
		// ----------------- V I R T U A L ----------------
		virtual void init() {};

		// ----------------- S E T T E R S ----------------
				
		auto setIsQuadratic(bool _is)					-> void							{ this->isQuadratic_ = _is;								};	// set if the operator is quadratic
		auto setActOn(u64 _acton)						-> void							{ this->acton_ = _acton;								};	// which states the operator acts on, saved in a number form (binary bitmask)
		
		// names
		auto setName(SymGenerators _name)				-> void							{ this->name_ = _name;									}; 
		auto setNameS(const std::string& _name)			-> void							{ this->nameS_ = _name;									};

		auto setVal(_T _val)							-> void							{ this->eigVal_ = _val;									};
		auto setNs(size_t Ns)							-> void							{ this->Ns_ = Ns;										};

		// quadratic matrices and overriden functions creating the matrix
		auto setFun(GeneralizedMatrixFunction<_T>& _fun) -> void						{ this->overridenMatFun_ = _fun;						};
		auto setFun(GeneralizedMatrixFunction<_T>&& _fun) -> void						{ this->overridenMatFun_ = std::move(_fun);				};
		auto setFun(const GeneralizedMatrixFunction<_T>&& _fun) -> void					{ this->overridenMatFun_ = std::move(_fun);				};

		// functions (using integers)
		auto setFun(const repType& _fun)				-> void							{ this->fun_ = _fun;									}; // set the function
		auto setFun(repType&& _fun)						-> void							{ this->fun_ = std::move(_fun);							}; // set the function (move)
		
		// functions (using vectors)
		auto setFun(const repTypeV& _fun)				-> void							{ this->funV_ = _fun;									}; // set the function
		auto setFun(repTypeV&& _fun)					-> void							{ this->funV_ = std::move(_fun);						}; // set the function (move)

		// ----------------- G E T T E R S ----------------
		auto getIsQuadratic()							const -> bool					{ return this->isQuadratic_;							};
		auto getActOn()									const -> u64					{ return this->acton_;									};
		auto getNs()									const -> size_t					{ return this->Ns_;										};
		auto getVal()									const -> _T						{ return this->eigVal_;									};
		// names
		auto getName()									const -> SymGenerators			{ return this->name_;									};
		auto getNameG()									const -> std::string			{ return SSTR(getSTR_SymGenerators(this->name_));		};
		auto getNameS()									const -> std::string			{ return this->nameS_;									};
		// functions
		auto getFun()									const -> repType				{ return this->fun_;									};
		auto getFunV()									const -> repTypeV				{ return this->funV_;									};
		
		// ----------------------------------------------------------------------------------------------------

		public:

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F R I E N D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		/*
		* @brief representative eigenvalue calculator for the operator 
		* @param _op operator to calculate the eigenvalue for
		* @returns the eigenvalue of the operator
		*/
		friend _T chi(const GeneralOperator<_T, repType, repTypeV, _Ts...>& _op)		{ return _op.eigVal_; 									};

		// ----------------------------------------------------------------------------------------------------

		// -------------- O P E R A T O R ( ) -------------
		
		virtual auto operator()(u64 s, _Ts... a)		const -> ReturnType				= 0;		// operator acting on the states
		virtual auto operator()(u64 s, _Ts... a)		-> ReturnType					= 0;		// operator acting on the states			
		virtual auto operator()(_VT_CR s, _Ts... a)		const -> ReturnTypeV			= 0;		// operator acting on the vectors
		virtual auto operator()(_VT_CR s, _Ts... a)		-> ReturnTypeV					= 0;		// operator acting on the vectors

		// ----------------------------------------------------------------------------------------------------		

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T,  template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;
		
		// ----------------------------------------------------------------------------------------------------

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType>
		typename std::enable_if<HasMatrixType<_MatType<_TinMat>>, _MatType<_TinMat>>::type
			standaridizeMatrix(_MatType<_TinMat>& _mat) const;

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType>
		typename std::enable_if<!HasMatrixType<_MatType<_TinMat>>, GeneralizedMatrix<_TinMat>>::type
			standaridizeMatrix(GeneralizedMatrix<_TinMat>& _mat) const;

	};

	// ##########################################################################################################################################

	/*
	* @brief Returns the matrix representation of the operator. This is a default implementation for the operators that are acting on the states.
	* The matrix here is created ad-hoc by an overriden matrix function. If the function is not overriden, the function throws an error.
	* @param _dim dimension of the matrix
	* @param _arg additional arguments
	* @returns the matrix representation of the operator
	*/
	template <typename _T, typename repType, typename repTypeV, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _InT>
	typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
		Operators::GeneralOperator<_T, repType, repTypeV, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
	{
		_MatType<_TinMat> op(_dim, _dim);
#ifdef _DEBUG
		LOGINFO("Operator has a special, overriden matrix function!", LOG_TYPES::CHOICE, 1);
#endif
		if constexpr (std::is_same_v<_MatType<_TinMat>, arma::Mat<_TinMat>>)
			op = this->overridenMatFun_(_dim).getDense();
		else if constexpr (std::is_same_v<_MatType<_TinMat>, arma::SpMat<_TinMat>>)
			op = this->overridenMatFun_(_dim).getSparse();
		else
		{
			LOGINFO("Type is neither arma::Mat nor arma::SpMat", LOG_TYPES::ERROR, 0);
			throw std::logic_error("Unsupported matrix type in GeneralOperator class");
		}
		return op;
	}

	// ##########################################################################################################################################

	template <typename _T, typename repType, typename repTypeV, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _InT>
	typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
		Operators::GeneralOperator<_T, repType, repTypeV, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
	{
		GeneralizedMatrix<_TinMat> op(_dim);
		if (this->overridenMatFun_)
		{
#ifdef _DEBUG
			LOGINFO("Operator has a special, overriden matrix function!", LOG_TYPES::CHOICE, 1);
#endif
			op = this->overridenMatFun_(_dim);
		}
		else
		{
			LOGINFO("Operator does not have a special, overriden matrix function!", LOG_TYPES::CHOICE, 1);
			throw std::logic_error("Operator does not have a special, overriden matrix function!");
		}
		return op;
	}

	// ##########################################################################################################################################
	
	/*
	* @brief Standarizes the matrix representation of the operator. This is a default implementation for the operators that are acting on the states.
	* The matrix here is standarized by the standarizeOperator function. If the function is not overriden, the function throws an error.
	* @param _mat matrix to be standarized
	* @returns the standarized matrix representation of the operator
	*/
	template <typename _T, typename repType, typename repTypeV, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType>
	typename std::enable_if<HasMatrixType<_MatType<_TinMat>>, _MatType<_TinMat>>::type
		Operators::GeneralOperator<_T, repType, repTypeV, _Ts...>::standaridizeMatrix(_MatType<_TinMat>& _mat) const
	{
		// standarize the operator
		if (_standarize)
		{
			Operators::standarizeOperator(_mat);
			Operators::operatorInfo(_mat);
		}
		return _mat;
	}

	// ##########################################################################################################################################

	template <typename _T, typename repType, typename repTypeV, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType>
	typename std::enable_if<!HasMatrixType<_MatType<_TinMat>>, GeneralizedMatrix<_TinMat>>::type
		Operators::GeneralOperator<_T, repType, repTypeV, _Ts...>::standaridizeMatrix(GeneralizedMatrix<_TinMat>& _mat) const
	{
		// standarize the operator
		if (_standarize)
		{
			_mat.standarize();
			Operators::operatorInfo(_mat);
		}
		return _mat;
	}
};

// ################################################################ N O N - M I X I N G #########################################################################

namespace Operators {
	
	// forward declaration
	template <typename _T, typename ..._Ts>
	class OperatorComb;

	/*
	* @brief A class describing the local operator acting on specific states. It returns a value and changes a state.
	* Can take several template arguments
	*/
	template<typename _T, typename ..._Ts>
	class Operator : public GeneralOperator<_T, 
											typename _OP<_T>::template INP<_Ts...>,
											typename _OP_V<_T>::template INP<_Ts...>,
											_Ts...>
	{
	public:
		using baseType 	= 	GeneralOperator<_T, 
								typename _OP<_T>::template INP<_Ts...>, 
								typename _OP_V<_T>::template INP<_Ts...>, 
								_Ts...>;												// type of the operator - base type
		// ----------------------------------------------------------------------------------------------------

		// Inherit constructors from GeneralOperator
   	 	using GeneralOperator<_T, typename _OP<_T>::template INP<_Ts...>, typename _OP_V<_T>::template INP<_Ts...>, _Ts...>::GeneralOperator;  									
		// using GeneralOperator<_T, typename OperatorsCombination::_OP<_T>::template INP<_Ts...>, typename OperatorsCombination::_OP_V<_T>::template INP<_Ts...>, _Ts...>::operator=;
		// using GeneralOperator<_T, typename OperatorsCombination::_OP<_T>::template INP<_Ts...>, typename OperatorsCombination::_OP_V<_T>::template INP<_Ts...>, _Ts...>::operator();		
		
		// ----------------------------------------------------------------------------------------------------
		using _VT 			= baseType::_VT;											// type of the vector to be used for the operator
		using _VT_CR 		= baseType::_VT_CR;											// type of the vector to be used for the operator - const reference		
	public:
		using repType 		= typename baseType::repType_;								// type of the function to be used for the operator
		using repTypeV 		= typename baseType::repTypeV_;								// type of the function to be used for the operator - for vectors
		using ReturnType 	= typename baseType::ReturnType;							// return type of the operator
		using ReturnTypeV 	= typename baseType::ReturnTypeV;							// return type of the operator - for vectors

	public:

		// Default constructor
		Operator() : baseType() {};														// default constructor
		
		explicit operator OperatorComb<_T, _Ts...>() const {
			return OperatorComb<_T, _Ts...>(*this);
		}

		// ----------------------------------------------------------------------------------------------------
	public:
		// -------------- O P E R A T O R ( ) -------------
		
		virtual auto operator()(u64 s, _Ts... a)		const -> ReturnType override	{ auto [s2, _val] = this->fun_(s, a...); return std::make_pair(s2, this->eigVal_ * _val);  };
		virtual auto operator()(u64 s, _Ts... a)		-> ReturnType override			{ auto [s2, _val] = this->fun_(s, a...); return std::make_pair(s2, this->eigVal_ * _val);  };
		virtual auto operator()(_VT_CR s, _Ts... a)		const -> ReturnTypeV override	{ auto [s2, _val] = this->funV_(s, a...); return std::make_pair(s2, this->eigVal_ * _val); };
		virtual auto operator()(_VT_CR s, _Ts... a)		-> ReturnTypeV override			{ auto [s2, _val] = this->funV_(s, a...); return std::make_pair(s2, this->eigVal_ * _val); };
		//virtual std::function<std::pair<u64, _T>(_Ts...)> operator*(u64 s)	const	{ return std::bind(this->fun_, s, std::placeholders::_1); };
		//virtual std::function<std::pair<u64, _T>(_Ts...)> operator*(u64 s)			{ return std::bind(this->fun_, s, std::placeholders::_1); };

		// ----------------------------------------------------------------------------------------------------

		// -------------------- STATIC --------------------
		static auto E(u64 s, _Ts...)					-> ReturnType					{ return std::make_pair(s, _T(1.0));				};
		static inline std::function <_T(u64, _Ts...)> E_F								= [](u64 s, _Ts...) { return _T(1.0); };
		static auto E_V(_VT_CR s, _Ts...)				-> ReturnTypeV					{ return std::make_pair(s, _T(1.0));				};
		static inline std::function <_T(_VT_CR, _Ts...)> E_V_F							= [](_VT_CR s, _Ts...) { return _T(1.0); };

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   J O I N %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		/*
		* Joins the operators into one operator. This combines the operators acting on the same Hilbert space.
		*/
		template <typename T_ = _T, 
			typename std::enable_if<std::is_same<T_, cpx>::value>::type* = nullptr>
		Operator<T_, _Ts...> operator%(const Operator<double, _Ts...>& op) const
		{
			if (this->lat_)
				return Operators::Operator<cpx, _Ts...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_, this->funV_ % op.funV_, Operators::SymGenerators::OTHER);
			else
				return Operators::Operator<cpx, _Ts...>(this->Ns_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_, this->funV_ % op.funV_, Operators::SymGenerators::OTHER);
		}

		template <typename T_ = _T, 
			typename std::enable_if<!std::is_same<T_, cpx>::value>::type* = nullptr> 
		Operator<T_, _Ts...> operator%(const Operator<double, _Ts...>& op) const
		{
			if (this->lat_)
				return Operators::Operator<T_, _Ts...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_, this->funV_ % op.funV_, Operators::SymGenerators::OTHER);
			else
				return Operators::Operator<T_, _Ts...>(this->Ns_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_, this->funV_ % op.funV_, Operators::SymGenerators::OTHER);
		}

		Operator<cpx, _Ts...> operator%(const Operator<cpx, _Ts...>& op) const
		{
			if (this->lat_)
				return Operators::Operator<cpx, _Ts...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_, this->funV_ % op.funV_, Operators::SymGenerators::OTHER);
			else
				return Operators::Operator<cpx, _Ts...>(this->Ns_, this->eigVal_ * op.eigVal_, this->fun_ % op.fun_, this->funV_ % op.funV_, Operators::SymGenerators::OTHER);
		}

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   C O N C A T %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		template <typename T_ = _T, typename ..._T2s,
			typename std::enable_if<std::is_same<T_, cpx>::value>::type* = nullptr>
		Operator<T_, _Ts..., _T2s...> operator*(const Operator<double, _T2s...>& op) const
		{
			if (this->lat_)
				return Operators::Operator<cpx, _Ts..., _T2s...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_, this->funV_ * op.funV_, Operators::SymGenerators::OTHER);
			else
				return Operators::Operator<cpx, _Ts..., _T2s...>(this->Ns_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_, this->funV_ * op.funV_, Operators::SymGenerators::OTHER);
		}

		template <typename T_ = _T, typename ..._T2s,
			typename std::enable_if<!std::is_same<T_, cpx>::value>::type* = nullptr>
		Operator<T_, _Ts..., _T2s...> operator*(const Operator<double, _T2s...>& op) const
		{
			if (this->lat_)
				return Operators::Operator<T_, _Ts..., _T2s...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_, this->funV_ * op.funV_, Operators::SymGenerators::OTHER);
			else
				return Operators::Operator<T_, _Ts..., _T2s...>(this->Ns_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_, this->funV_ * op.funV_, Operators::SymGenerators::OTHER);
		}
		
		template <typename ..._T2s>
		Operator<cpx, _Ts..., _T2s...> operator*(const Operator<cpx, _T2s...>& op) const
		{
			if (this->lat_)
				return Operators::Operator<cpx, _Ts..., _T2s...>(this->lat_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_, this->funV_ * op.funV_, Operators::SymGenerators::OTHER);
			else
				return Operators::Operator<cpx, _Ts..., _T2s...>(this->Ns_, this->eigVal_ * op.eigVal_, this->fun_ * op.fun_, this->funV_ * op.funV_, Operators::SymGenerators::OTHER);
		}

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S   P O W E R %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		template <typename _T1, typename std::enable_if<std::is_integral<_T1>::value>::type* = nullptr>
		[[nodiscard]]
		Operator<_T, _Ts...> operator^(_T1 _n);
			
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F R I E N D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		/*
		* @brief calculate operator acting on state num eigenvalue
		*/
		friend _T chi(const Operator<_T, _Ts...>& _op, u64 _s, _Ts... _a)				{ auto [state, val] = _op(_s, std::forward<_Ts>(_a)...); return val * _op.eigVal_; };

		/*
		* @brief calculate operator acting on state num eigenvalue (vector version)
		*/
		friend _T chi(const Operator<_T, _Ts...>& _op, const _VT& _s, _Ts... _a)		{ auto [state, val] = _op(_s, std::forward<_Ts>(_a)...); return val * _op.eigVal_; };
	
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% H I L B E R T   S P A C E %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T,  template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, uint _spinModes = 2>
		_MatType<typename std::common_type<_TinMat, _T1>::type>
		generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, typename _T2, uint _spinModes = 2>
		_MatType<typename std::common_type<_TinMat, _T1, _T2>::type> 
		generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil1, const Hilbert::HilbertSpace<_T2, _spinModes>& _Hil2, _Ts... _arg);

		// calculates the matrix element of operator given a single state
		template <typename _T1, typename _T2>
		[[deprecated]]
		static _T avOp(const arma::Col<_T1>& _alfa, const arma::Col<_T2>& _beta, const Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace);
		template <typename _T1>
		[[deprecated]]
		static _T avOp(const arma::Col<_T1>& _alfa, const Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace);
	};

	// ##########################################################################################################################################
	
	// for the containers on the vectors
	//using OpVec_glb_t = v_1d<std::variant<Operators::Operator<cpx>, Operators::Operator<double>>>;
	//using OpVec_loc_t = v_1d<std::variant<Operators::Operator<cpx, uint>, Operators::Operator<double, uint>>>;
	//using OpVec_cor_t = v_1d<std::variant<Operators::Operator<cpx, uint, uint>, Operators::Operator<double, uint, uint>>>;

	// ##############################################################################################################################################################

	/*
	* @brief Raising the operator to the power of n (n is an integer) 
	* @param _n the power to which the operator is raised
	* @returns the operator raised to the power of n
	*/
	template<typename _T, typename ..._Ts>
	template<typename _T1, typename std::enable_if<std::is_integral<_T1>::value>::type*>
	inline Operators::Operator<_T,_Ts...> Operators::Operator<_T, _Ts...>::operator ^(_T1 _n)
	{
		if (_n == 0)
			return Operator<_T, _Ts...>(this->lat_, 1.0, Operators::Operator<_T, _Ts...>::E);
		else if (_n == 1)
			return *this;

		auto _f = [_n, this](u64 _s, _Ts... _args) {
			_T val = 1.0;
			do {
				auto [newS, newV]	= this->operator()(_s, _args...);
				_s					= newS;
				val					*= newV;
				--_n;
				} while (_n);
			return std::make_tuple(_s, val);
		};
		return Operator<_T, _Ts...>(this->lat_, std::pow(this->eigVal_, _n), _f);
	}

	// ##########################################################################################################################################

	//!TODO
	template<typename _T, typename ..._Ts>
	template<typename _T1, typename _T2>
	inline _T Operators::Operator<_T, _Ts...>::avOp(const arma::Col<_T1>& _alfa, const arma::Col<_T2>& _beta, const Operators::Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace)
	{
		return _T();
	}

	//!TODO
	template<typename _T, typename ..._Ts>
	template<typename _T1>
	inline _T Operators::Operator<_T, _Ts...>::avOp(const arma::Col<_T1>& _alfa, const Operators::Operator<_T, _Ts...>& _op, const Hilbert::HilbertSpace<_T>& _hSpace)
	{
		return _T();
	}

	// #################################################### M A T R I X   G E N E R A T I O N ####################################################

	/*
	* @brief Creates a most basic operator matrix knowing only the dimension of the Hilbert space. 
	* The operator is acting on the same Hilbert space as the one it is acting on.
	* For the total Hilbert space known to be without symmetries - not looking for representatives
	* @brief _dim A dimension of the Hilbert space
	* @returns A matrix representing the operator
	*/
	template<typename _T, typename ..._Ts> 
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _InT>
	typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
		Operators::Operator<_T, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
	{
	#ifdef _DEBUG
		LOGINFO("Creating operator matrix: " + VEQ(this->nameS_), LOG_TYPES::INFO, 3);
	#endif

		_MatType<_TinMat> op(_dim, _dim);

		// check whether the operator has an overriden matrix function
		if (this->overridenMatFun_)
			op = GeneralOperator<_T, repType, repTypeV, _Ts...>::template generateMat<_standarize, _TinMat, _MatType>(_dim, _arg...);
		else
		{
			// otherwise create the operator matrix
		#pragma omp parallel for
			for (u64 _base = 0; _base < _dim; ++_base) 
			{
				auto [_idx, _val]	=	this->operator()(_base, _arg...);
				op(_idx, _base)		+=	_val;
			}

		}
		return GeneralOperator<_T, repType, repTypeV, _Ts...>::template standaridizeMatrix<_standarize, _TinMat, _MatType>(op);
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates a most basic operator matrix knowing only the dimension of the Hilbert space. 
	* The operator is acting on the same Hilbert space as the one it is acting on.
	* For the total Hilbert space known to be without symmetries - not looking for representatives
	* @brief _dim A dimension of the Hilbert space
	* @returns A matrix representing the operator
	*/
	template<typename _T, typename ..._Ts> 
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _InT>
	typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
	Operators::Operator<_T, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
	{
	#ifdef _DEBUG
		LOGINFO("Creating operator matrix: " + VEQ(this->nameS_), LOG_TYPES::INFO, 3);
	#endif

		GeneralizedMatrix<_TinMat> op(_dim);

		// check whether the operator has an overriden matrix function
		if (this->overridenMatFun_ && this->isQuadratic_)
			op = GeneralOperator<_T, repType, repTypeV, _Ts...>::template generateMat<_standarize, _TinMat, _MatType>(_dim, _arg...);
		else
		{
			// otherwise create the operator matrix
	#pragma omp parallel for
			for (u64 _base = 0; _base < _dim; ++_base) 
			{
				auto [_idx, _val]	=	this->operator()(_base, _arg...);
				op.add(_idx, _base, _val);
			}

		}

		return GeneralOperator<_T, repType, repTypeV, _Ts...>::template standaridizeMatrix<_standarize, _TinMat, _MatType>(op);
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates an operator matrix whenever the operator is not transforming the state from a different symmetry sector.
	* Uses the Hilbert space that stores the state transformations from the representative base.
	* @param _Hil the Hilbert space in which we operate
	* @param _arg arguments for the operator
	* @returns A matrix representing the operator
	*/
	template<typename _T, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _T1, uint _spinModes>
	inline _MatType<typename std::common_type<_TinMat, _T1>::type> Operators::Operator<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil, _Ts ..._arg) const
	{
		using res_typ	=	typename std::common_type<_T1, _TinMat>::type;
		u64 Nh			=	_Hil.getHilbertSize();
		_MatType<res_typ> op(Nh, Nh);

		for (u64 _idx = 0; _idx < Nh; _idx++)
		{
			auto [_newState, _val]		=	this->operator()(_Hil.getMapping(_idx), _arg...);

			// why even bother?
			[[unlikely]] if (EQP(std::abs(_val), 0.0, 1e-14))
				continue;

			// looking for the representative
			auto [_newIdx, _eigval]		=	_Hil.findRep(_newState, _Hil.getNorm(_idx));

			// go to it manually
			[[likely]]
			if(_newIdx < Nh)
				op(_newIdx, _idx)		+=	_val * _eigval;
		}
		// standarize the operator
		//if(_standarize)
		//	Operators::standarizeOperator(op);
		return op;
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates an operator matrix whenever the operator is transforming the state to a different symmetry sector 
	* @param _Hil the Hilbert space in which we operate
	* @warning (takes into account that we are going to a different symmetry sector so matrix is not square)
	* @trace O = \sum _{i \in A} \sum _{j \in _B} |i>_A <j|_B O_{ij}
	*/
	template<typename _T, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _T1, typename _T2, uint _spinModes>
	inline _MatType<typename std::common_type<_TinMat, _T1, _T2>::type> Operators::Operator<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil1, const Hilbert::HilbertSpace<_T2, _spinModes>& _Hil2, _Ts ..._arg)
	{
		// using res_typ		=	typename std::common_type<_T1, _T, _T2>::type;
		u64 NhA				=	_Hil1.getHilbertSize();
		u64 NhB				=	_Hil2.getHilbertSize();
		arma::SpMat<_TinMat> op(NhA, NhB);

		for (u64 _idxB = 0; _idxB < NhB; _idxB++) 
		{
			// act with an operator on beta sector (right)
			auto [_newStateB, _valB]				=	this->operator()(_Hil2.getMapping(_idxB), _arg...);

			// why even bother?
			[[unlikely]] if (EQP(std::abs(_valB), 0.0, 1e-14))
				continue;

			// find the corresponding index and value in the A sector (left)
			auto [newIdxA, symValA]					=	_Hil1.findRep(_newStateB, _Hil2.getNorm(_idxB));

			// check if the state is there
			if (newIdxA < NhA)
				op(newIdxA, _idxB)					+=	_valB * algebra::conjugate(symValA);
		}
		// standarize the operator
		if(_standarize)
			return standarizeOperator(op);
		return op;
	}
};


// ##############################################################################################################################################################

// ################################################## O P E R A T O R   R E T U R N S   C O M B I N A T I O N ###################################################

// ##############################################################################################################################################################

namespace Operators {
	template <typename _T, typename ..._Ts>
	class OperatorComb : public GeneralOperator<_T, 
												typename OperatorsCombination::_OP<_T>::template INP<_Ts...>,
												typename OperatorsCombination::_OP_V<_T>::template INP<_Ts...>,
												_Ts...>
	{
	public:
		// type of the operator - base type
		using baseType 	= 	GeneralOperator<_T, typename OperatorsCombination::_OP<_T>::template INP<_Ts...>, 
								typename OperatorsCombination::_OP_V<_T>::template INP<_Ts...>, _Ts...>;	
		// ----------------------------------------------------------------------------------------------------

		// Inherit constructors from GeneralOperator
   	 	using GeneralOperator<_T, typename OperatorsCombination::_OP<_T>::template INP<_Ts...>, typename OperatorsCombination::_OP_V<_T>::template INP<_Ts...>, _Ts...>::GeneralOperator;  
		using GeneralOperator<_T, typename OperatorsCombination::_OP<_T>::template INP<_Ts...>, typename OperatorsCombination::_OP_V<_T>::template INP<_Ts...>, _Ts...>::operator=;
		// using GeneralOperator<_T, typename OperatorsCombination::_OP<_T>::template INP<_Ts...>, typename OperatorsCombination::_OP_V<_T>::template INP<_Ts...>, _Ts...>::operator();				
		
		// ----------------------------------------------------------------------------------------------------
		using _VT 			= baseType::_VT;											// type of the vector to be used for the operator
		using _VT_CR 		= baseType::_VT_CR;											// type of the vector to be used for the operator - const reference

	public:
		using repType 		= typename baseType::repType_;								// type of the function to be used for the operator
		using repTypeV 		= typename baseType::repTypeV_;								// type of the function to be used for the operator - for vectors
		using ReturnType 	= typename baseType::ReturnType;							// return type of the operator
		using ReturnTypeV 	= typename baseType::ReturnTypeV;							// return type of the operator - for vectors

		// ----------------------------------------------------------------------------------------------------
		typename _OP<_T>::template INP<_Ts...> s_fun_;									// function for the operator
		typename _OP_V<_T>::template INP<_Ts...> s_funV_;								// function for the operator - for vectors

	public:
		// ----------------------------------------------------------------------------------------------------

		// Default constructor
		OperatorComb() : baseType() {};													// default constructor
		
		OperatorComb(const Operator<_T, _Ts...>& other)
		{
			this->name_             = other.name_;
			this->nameS_            = other.nameS_;
			this->overridenMatFun_  = other.overridenMatFun_;
			this->isQuadratic_      = other.isQuadratic_;
			this->lat_              = other.lat_;
			this->Ns_               = other.Ns_;
			this->eigVal_           = other.eigVal_;
			this->hasVectorFun_     = other.hasVectorFun_;

			// Move scalar function
			this->s_fun_ 			= other.fun_;
			
			// Create a new lambda that captures a copy of `this->s_fun_`
			{
				if (this->s_fun_) 
				{
					this->fun_ = [s_fun_copy = this->s_fun_](u64 _s, _Ts... _args) {
						std::pair<u64, _T> _out = s_fun_copy(_s, _args...);
						return std::vector<std::pair<u64, _T>>({ _out });
					};
				} else {
					this->fun_ = nullptr;
				}
			}

			// Handle vector function if present
			if (this->hasVectorFun_) 
			{
				{
					this->s_funV_ = other.s_funV_;
					
					// Create a new lambda that captures a copy of `this->s_funV_`
					if (this->s_funV_) 
					{
						this->funV_ = [s_funV_copy = this->s_funV_](const _VT& _s, _Ts... _args) {
							std::pair<_VT, _T> _out = s_funV_copy(_s, _args...);
							return std::vector<std::pair<_VT, _T>>({ _out });
						};
					} else {
						this->funV_ = nullptr;
					}
				}
			}
		};

		OperatorComb(Operator<_T, _Ts...>&& other)
		{
			this->name_             = std::move(other.name_);
			this->nameS_            = std::move(other.nameS_);
			this->overridenMatFun_  = std::move(other.overridenMatFun_);
			this->isQuadratic_      = std::move(other.isQuadratic_);
			this->lat_              = std::move(other.lat_);
			this->Ns_               = std::move(other.Ns_);
			this->eigVal_           = std::move(other.eigVal_);
			this->hasVectorFun_     = other.hasVectorFun_;

			// Move scalar function
			this->s_fun_ 			= std::move(other.fun_);
			
			// Create a new lambda that captures a copy of `this->s_fun_`
			{
				if (this->s_fun_) 
				{
					this->fun_ = [s_fun_copy = this->s_fun_](u64 _s, _Ts... _args) {
						std::pair<u64, _T> _out = s_fun_copy(_s, _args...);
						return std::vector<std::pair<u64, _T>>({ _out });
					};
				} else {
					this->fun_ = nullptr;
				}
			}

			// Handle vector function if present
			if (this->hasVectorFun_) 
			{
				{
					this->s_funV_ = std::move(other.funV_);
					
					// Create a new lambda that captures a copy of `this->s_funV_`
					if (this->s_funV_) 
					{
						this->funV_ = [s_funV_copy = this->s_funV_](const _VT& _s, _Ts... _args) {
							std::pair<_VT, _T> _out = s_funV_copy(_s, _args...);
							return std::vector<std::pair<_VT, _T>>({ _out });
						};
					} else {
						this->funV_ = nullptr;
					}
				}
			}

			// Optional: Clear the moved-from `other` to prevent accidental use
			other.fun_ 		= nullptr;
			other.funV_ 	= nullptr;

		}

		// ---------------------------------------- O P E R A T O R ( ) ---------------------------------------
		
		virtual auto operator()(u64 s, _Ts... a)		const -> ReturnType override;
		virtual auto operator()(u64 s, _Ts... a)		-> ReturnType override;
		virtual auto operator()(_VT_CR s, _Ts... a)		const -> ReturnTypeV override;
		virtual auto operator()(_VT_CR s, _Ts... a)		-> ReturnTypeV override;		
			
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F R I E N D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		friend std::vector<_T> chi(const OperatorComb<_T, _Ts...>& _op, u64 _s, _Ts... _a)\
		{
			std::vector<_T> _res;
			for (const auto [s2, _val] : _op(_s, std::forward<_Ts>(_a)...))
				_res.push_back(_val);
			return _res;
		}

		friend std::vector<_T> chi(const OperatorComb<_T, _Ts...>& _op, const _VT& _s, _Ts... _a)
		{
			std::vector<_T> _res;
			for (const auto [s2, _val] : _op(_s, std::forward<_Ts>(_a)...))
				_res.push_back(_val);
			return _res;
		}
	
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% H I L B E R T   S P A C E %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T,  template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		// template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, uint _spinModes = 2>
		// _MatType<typename std::common_type<_TinMat, _T1>::type>
		// generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil, _Ts... _arg) const;

		// template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, typename _T2, uint _spinModes = 2>
		// _MatType<typename std::common_type<_TinMat, _T1, _T2>::type> 
		// generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil1, const Hilbert::HilbertSpace<_T2, _spinModes>& _Hil2, _Ts... _arg);
	};

	// ##########################################################################################################################################

	/*
	* @brief Calculating the application of the operator on the state and the values it returns
	* @param s the state on which the operator is acting
	* @param a the values of the operator
	* @returns the states and the values of the operator
	*/
	template <typename _T, typename ..._Ts>
	OperatorComb<_T, _Ts...>::ReturnType OperatorComb<_T, _Ts...>::operator()(u64 s, _Ts... a) const
	{
		// return value
		std::vector<std::pair<u64, _T>> _res;

		// calculate the operator
		for (const auto [s2, _val] : this->fun_(s, a...))
			_res.push_back(std::make_pair(s2, this->eigVal_ * _val));
		return _res;
	}

	template <typename _T, typename ..._Ts>
	OperatorComb<_T, _Ts...>::ReturnType OperatorComb<_T, _Ts...>::operator()(u64 s, _Ts... a)
	{
		// return value
		std::vector<std::pair<u64, _T>> _res;

		// calculate the operator
		for (const auto [s2, _val] : this->fun_(s, a...))
			_res.push_back(std::make_pair(s2, this->eigVal_ * _val));
		return _res;
	}
	
	// ------------------------------------------------------------------------------------------------------------------------------------------

	/*
	* @brief Calculating the application of the operator on the state and the values it returns (vector version)
	* @param s the state on which the operator is acting
	* @param a the values of the operator
	* @returns the states and the values of the operator
	* @note the operator is acting on the vector state and returns the vector of states and values
	*/
	template <typename _T, typename ..._Ts>
	OperatorComb<_T, _Ts...>::ReturnTypeV OperatorComb<_T, _Ts...>::operator()(_VT_CR s, _Ts... a) const
	{
		// return value
		std::vector<std::pair<_VT, _T>> _res;

		// calculate the operator
		for (const auto [s2, _val] : this->funV_(s, a...))
			_res.push_back(std::make_pair(s2, this->eigVal_ * _val));
		return _res;
	}
	
	template <typename _T, typename ..._Ts>
	OperatorComb<_T, _Ts...>::ReturnTypeV OperatorComb<_T, _Ts...>::operator()(_VT_CR s, _Ts... a)
	{
		// return value
		std::vector<std::pair<_VT, _T>> _res;

		// calculate the operator
		for (const auto [s2, _val] : this->funV_(s, a...))
			_res.push_back(std::make_pair(s2, this->eigVal_ * _val));
		return _res;
	}

	// ##########################################################################################################################################

	template <typename _T, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class _TM> class _MatType, typename _InT>
	typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
	OperatorComb<_T, _Ts...>::generateMat(_InT _dim, _Ts... _arg) const
	{
	#ifdef _DEBUG
		LOGINFO("Creating operator matrix: " + VEQ(this->nameS_), LOG_TYPES::INFO, 3);
	#endif
	
		_MatType<_TinMat> op(_dim, _dim);

		// check whether the operator has an overriden matrix function
		if (this->overridenMatFun_)
			op = GeneralOperator<_T, repType, repTypeV, _Ts...>::template generateMat<_standarize, _TinMat, _MatType>(_dim, _arg...);
		else
		{
			for (u64 _base = 0; _base < _dim; ++_base) 
			{
				for (const auto [_idx, _val] : this->operator()(_base, _arg...))
					op(_idx, _base)		+=	_val;
			}
		}

		return GeneralOperator<_T, repType, repTypeV, _Ts...>::template standaridizeMatrix<_standarize, _TinMat, _MatType>(op);
	}

	// ##########################################################################################################################################

	template <typename _T, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class _TM> class _MatType, typename _InT>
	typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
	OperatorComb<_T, _Ts...>::generateMat(_InT _dim, _Ts... _arg) const
	{
	#ifdef _DEBUG
		LOGINFO("Creating operator matrix: " + VEQ(this->nameS_), LOG_TYPES::INFO, 3);
	#endif
		
		GeneralizedMatrix<_TinMat> op(_dim);

		// check whether the operator has an overriden matrix function
		if (this->overridenMatFun_ && this->isQuadratic_)
			op = GeneralOperator<_T, repType, repTypeV, _Ts...>::template generateMat<_standarize, _TinMat, _MatType>(_dim, _arg...);
		else
		{
			for (u64 _base = 0; _base < _dim; ++_base) 
			{
				for (const auto [_idx, _val] : this->operator()(_base, _arg...))
					op.add(_idx, _base, _val);
			}
		}

		return GeneralOperator<_T, repType, repTypeV, _Ts...>::template standaridizeMatrix<_standarize, _TinMat, _MatType>(op);
	}
	
	// ##########################################################################################################################################

};

// ##############################################################################################################################################################

// ############################################################ O P E R A T O R   C O N T A I N E R #############################################################

// ##############################################################################################################################################################

namespace Operators {
	namespace Containers
	{
		#define OperatorContainerS this->sizeX_, this->sizeY_
		// stores elements in a matrix form
		template <typename _T>
		using OperatorContainer_t = arma::Mat<_T>;


		// ##############################################################################################################################################
		template <typename _T>
		class OperatorContainer
		{
		protected:
			_OP_V_T state_;															// store the column state vector						

		public:
			using OperatorContainer_typ = OperatorContainer_t<_T>;
			size_t sizeX_				= 1;										// size of the operator (in X direction)
			size_t sizeY_				= 1;										// size of the operator (in Y direction)
			size_t Ns_					= 1;										// number of states in the quantum state

			// ##########################################################################################################################################

			size_t sample_num_			= 0;										// number of samples taken - for a single addition of the operator (Monte Carlo etc.)
			OperatorContainer_typ currentValue_;									// current value of the operator (at each point in X and Y) - for a single state
			v_1d<OperatorContainer_typ> samples_;									// store the samples for the operator - samples taken from the quantum state

			// ##########################################################################################################################################
		
			// ####### MANY BODY #######
			// store the matrix for the matrices obtained directly from the many body matrix (for the many body operators) 
			size_t sample_num_mb_		= 0;										// number of samples taken for the matrix representation of the operator (Monte Carlo etc.)
			GeneralizedMatrix<_T> manyBodyMatrix_;
			OperatorContainer_typ manyBodyVal_;

			// ##########################################################################################################################

			// ######## INDICES ########
			v_1d<uint> indices_;													// for finding out the index in the variadic variable - for multiple locality operators
			uint currentIdx_			= 0;										// current index in the variadic variable (currently processed) - for multiple locality operators

		public:

			// ##########################################################################################################################

			~OperatorContainer() 		= default;

			OperatorContainer()
				: Ns_(1)
			{
				this->samples_ 	= {};
				this->state_ 	= _OP_V_T(1);
			}

			OperatorContainer(size_t _Ns)
				: Ns_(_Ns)
			{
				this->samples_ 	= {};
				this->state_ 	= _OP_V_T(_Ns);
			}

			// ##########################################################################################################################

			template <class _Tt = uint>
			typename std::enable_if<std::is_arithmetic<_Tt>::value, void>::type
			updCurrent(_T _val);

			template <class _Tt, typename ..._Tss>
			typename std::enable_if<std::is_arithmetic<_Tt>::value, void>::type
			updCurrent(_T _val, _Tt i, _Tss...a);

			template <typename ..._Tss>
			void updCurrent(_T _val, _Tss...a)										{ currentIdx_ = 0; this->updCurrent(_val, a...);	};

			// sample number - for a single realization
			void updSampleNum()														{ this->sample_num_++;								};
			void updSampleNumMB()													{ this->sample_num_mb_++;							};
			void resetSampleNum()													{ this->sample_num_ = 0;							};
			void resetSampleNumMB()													{ this->sample_num_mb_ = 0;							};

			// ##########################################################################################################################
			
			template <typename ..._Ts>
			auto decideSize()			-> void;									// decide about the size of the container

			// --------------------------------------------------------------------------------------------------------------------------
			
			// for the colected samples
			auto operator[](uint i) 	const -> OperatorContainer_typ 				{ return this->samples_[i]; 						};			
			auto operator[](uint i)		-> const OperatorContainer_typ&				{ return this->samples_[i]; 						};
			auto operator()() 			const	-> OperatorContainer_typ			{ return this->currentValue_;						};
			auto operator()()			-> const OperatorContainer_typ&				{ return this->currentValue_;						};

		public:
			// ######## S E T T E R S ########
			auto reset()				-> void { this->resetSamples(); this->resetValue(); this->resetMB();							};
			auto resetSamples()			-> void { this->samples_ = {};																	};
			auto resetValue()			-> void { this->currentValue_ = OperatorContainer_typ(OperatorContainerS, arma::fill::zeros);	};
			auto resetMB()				-> void { this->manyBodyVal_ = OperatorContainer_typ(OperatorContainerS, arma::fill::zeros);	};

			// many body value storage
			auto setManyBodyVal(_T _val, size_t i = 0, size_t j = 0) -> void 		{ this->manyBodyVal_(i, j) = _val;					};
			auto updManyBodyVal(_T _val, size_t i = 0, size_t j = 0) -> void 		{ this->manyBodyVal_(i, j) += _val;					};

			// ######## G E T T E R S ########
			auto mbmat()			   	const -> GeneralizedMatrix<_T>				{ return this->manyBodyMatrix_;						};
			auto mbmat()			   	-> const GeneralizedMatrix<_T>&				{ return this->manyBodyMatrix_;						};
			auto mbval()			   	const -> arma::Mat<_T>						{ return this->manyBodyVal_;						};
			auto mbval()			   	-> const arma::Mat<_T>&						{ return this->manyBodyVal_;						};
			template<typename _T1>
			auto var()				   	const -> OperatorContainer_t<_T1>			{ return CAST<_T1>(Vectors::var<_T>(this->samples_));	};
			template<typename _T1>
			auto mean()				   	const -> OperatorContainer_t<_T1>			{ return CAST<_T1>(Vectors::mean<_T>(this->samples_));};
			auto value()			   	const -> OperatorContainer_t<_T>			{ return this->currentValue_;						};
			auto value(uint i)		   	const -> OperatorContainer_t<_T>			{ return this->samples_[i];							};
			auto samples()		   		const -> v_1d<OperatorContainer_typ>		{ return this->samples_;							};
			auto samples()			  	-> const v_1d<OperatorContainer_typ>&		{ return this->samples_;							};

			// ##########################################################################################################################

			template <typename _T2, typename ..._Ts>
			auto setManyBodyMat(const Hilbert::HilbertSpace<_T2>& _hilb, Operator<_T, _Ts...>* _op, _Ts... a) -> void;

			template <typename _T2, typename ..._Ts>
			auto setManyBodyMat(size_t _Nh, Operator<_T, _Ts...>* _op, _Ts... a) -> void;

			template <typename _T2, typename ..._Ts>
			auto setManyBodyMat(const Hilbert::HilbertSpace<_T2>& _hilb, std::vector<Operator<_T, _Ts...>*>& _op, _Ts... a) -> void;

			template <typename _T2, typename ..._Ts>
			auto setManyBodyMat(size_t _Nh, std::vector<Operator<_T, _Ts...>*>& _op, _Ts... a) -> void;
			
			// combination

			template <typename _T2, typename ..._Ts>
			auto setManyBodyMat(const Hilbert::HilbertSpace<_T2>& _hilb, OperatorComb<_T, _Ts...>* _op, _Ts... a) -> void;

			template <typename _T2, typename ..._Ts>
			auto setManyBodyMat(size_t _Nh, OperatorComb<_T, _Ts...>* _op, _Ts... a) -> void;

			template <typename _T2, typename ..._Ts>
			auto setManyBodyMat(const Hilbert::HilbertSpace<_T2>& _hilb, std::vector<OperatorComb<_T, _Ts...>*>& _op, _Ts... a) -> void;

			template <typename _T2, typename ..._Ts>
			auto setManyBodyMat(size_t _Nh, std::vector<OperatorComb<_T, _Ts...>*>& _op, _Ts... a) -> void;

			// ##########################################################################################################################

			// ######## S A M P L I N G ########

			inline void normalize(bool reset = true);
			
			template <typename _T1>
			arma::Col<_T1> sample(const arma::Col<_T1>& _state, size_t i = 0, size_t j = 0);

		};

		// ##########################################################################################################################################

		template<typename _T>
		template<class _Tt>
		inline typename std::enable_if<std::is_arithmetic<_Tt>::value, void>::type
		Operators::Containers::OperatorContainer<_T>::updCurrent(_T _val)
		{
			// check the size of the indices
			if (this->indices_.size() == 0)
				this->currentValue_(0, 0) += _val;
			else if (this->indices_.size() == 1)
				this->currentValue_(this->indices_[0], 0) += _val;
			else if (this->indices_.size() == 2)
				this->currentValue_(this->indices_[0], this->indices_[1]) += _val;
			else
				throw std::runtime_error("Not implemented such exotic operators...");
			this->sample_num_++;
		}

		/*
		* @brief Updates the current value of the operator with the given value and indices
		* @param _val value to add
		* @param i index to add to the current value of the index
		* @param a additional indices - in case of the many body operator or the operator that needs correlation saving
		* @note The function is recursive and goes through all the indices to update the value. See above for the implementation.
		*/
		template<typename _T>
		template<class _Tt, typename ..._Tss>
		inline typename std::enable_if<std::is_arithmetic<_Tt>::value, void>::type
		Operators::Containers::OperatorContainer<_T>::updCurrent(_T _val, _Tt i, _Tss...a)
		{
			this->indices_[this->currentIdx_] = i;
			this->currentIdx_++;
			updCurrent(_val, a...);
		};


		// ##########################################################################################################################################

		/*
		* @brief Resize the current value so one can store only the necessary values in the matrix (sizeX, sizeY)
		* Global		has sizeX = sizeY = 1 		-	stores only one value (scalar)
		* Local			has sizeX = Ns, sizeY = 1,	- 	stores Ns values (vector)
		* Correlation	has sizeX = Ns, sizeY = Ns. - 	stores Ns x Ns values (matrix)
		* @template _T type of the operator			
		* @template _Ts types of the additional parameters
		* @throws runtime_error if the number of arguments is not 0, 1 or 2
		*/
		template <typename _T>
		template <typename ..._Ts>
		void Operators::Containers::OperatorContainer<_T>::decideSize()
		{
			
			// get the size of template operators to decide on the opeartor type
			constexpr size_t numArgs 		= sizeof...(_Ts);
			
			LOGINFO("Deciding the size of the operator container...", LOG_TYPES::DEBUG, 3);
			LOGINFO("Number of arguments: " + VEQ(numArgs), LOG_TYPES::DEBUG, 4);

			if (numArgs == 0)
			{
				this->sizeX_				= 1;
				this->sizeY_				= 1;
				this->indices_				= {};
			}
			else if (numArgs == 1)
			{
				this->sizeX_				= this->Ns_;
				this->sizeY_				= 1;
				this->indices_				= { 0 };
			}
			else if (numArgs == 2)
			{
				this->sizeX_				= this->Ns_;
				this->sizeY_				= this->Ns_;
				this->indices_				= { 0, 0 };
			}
			else
				throw std::runtime_error("Not implemented for more than two arguments!");

			this->reset();
		}

		// ##########################################################################################################################################
		
		/*
		* @brief Sets the many body matrix (Operator Matrix) to apply it later for the states in a many body representation. It can be used as a combination 
		* of the operators acting on the Hilbert space. This means that one uses the Hilbert space to generate the matrix for the operator.
		* @param _H Hilbert space - the Hilbert space in which the operator is acting
		* @param _op operator to be used for the many body matrix creation 
		* @param ...a additional parameters to the operators - if needed - from the general operator definition
		*/
		template<typename _T>
		template <typename _T2, typename ..._Ts>
		inline void Operators::Containers::OperatorContainer<_T>::setManyBodyMat(const Hilbert::HilbertSpace<_T2>& _hilb, Operator<_T, _Ts...>* _op, _Ts ...a)
		{
			using res_typ = typename std::common_type<_T, _T2>::type; 		// get the common type from the operators and the Hilbert space - the result type

			const size_t _Nh 		= _hilb.getHilbertSize();
			const size_t _Nhfull	= _hilb.getFullHilbertSize();

			// store all the measured values
			this->manyBodyMatrix_ 	= GeneralizedMatrix<_T>(_hilb.getHilbertSize(), true);
			const bool _isFull 		= _Nh == _Nhfull;

			// setup the matrix
			const bool _isQuadratic	= _op->getIsQuadratic();				// check if the operator is quadratic
			if (_isFull) {
				if (_isQuadratic)
					this->manyBodyMatrix_ = algebra::cast<_T>(_op->template generateMat<true, res_typ, GeneralizedMatrix>(_Nh, a...));
				else
					this->manyBodyMatrix_ = algebra::cast<_T>(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_Nh, a...));
			}
			else {
				this->manyBodyMatrix_.setSparse(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_hilb, a...));
			}
		}

		template <typename _T>
		template <typename _T2, typename ..._Ts>
		inline void Operators::Containers::OperatorContainer<_T>::setManyBodyMat(const Hilbert::HilbertSpace<_T2>& _hilb, OperatorComb<_T, _Ts...>* _op, _Ts ...a)
		{
			using res_typ = typename std::common_type<_T, _T2>::type; 		// get the common type from the operators and the Hilbert space - the result type

			const size_t _Nh 		= _hilb.getHilbertSize();
			const size_t _Nhfull	= _hilb.getFullHilbertSize();

			// store all the measured values
			this->manyBodyMatrix_ 	= GeneralizedMatrix<_T>(_hilb.getHilbertSize(), true);
			const bool _isFull 		= _Nh == _Nhfull;

			// setup the matrix
			const bool _isQuadratic	= _op->getIsQuadratic();				// check if the operator is quadratic
			if (_isFull) {
				if (_isQuadratic)
					this->manyBodyMatrix_ = algebra::cast<_T>(_op->template generateMat<true, res_typ, GeneralizedMatrix>(_Nh, a...));
				else
					this->manyBodyMatrix_ = algebra::cast<_T>(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_Nh, a...));
			}
			// else {
			// 	this->manyBodyMatrix_.setSparse(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_hilb, a...));
			// }
		}
		// ------------------------------------------------------------------------------------------------------------------------------------------

		template <typename _T>
		template <typename _T2, typename ..._Ts>
		inline void Operators::Containers::OperatorContainer<_T>::setManyBodyMat(size_t _Nh, Operator<_T, _Ts...>* _op, _Ts ...a)
		{
			using res_typ = typename std::common_type<_T, _T2>::type; 		// get the common type from the operators and the Hilbert space - the result type

			// store all the measured values
			this->manyBodyMatrix_ 	= GeneralizedMatrix<_T>(_Nh, true);

			// setup the matrix
			const bool _isQuadratic	= _op->getIsQuadratic();				// check if the operator is quadratic
			if (_isQuadratic)
				this->manyBodyMatrix_ = algebra::cast<_T>(_op->template generateMat<true, res_typ, GeneralizedMatrix>(_Nh, a...));
			else
				this->manyBodyMatrix_ = algebra::cast<_T>(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_Nh, a...));	
		}

		template <typename _T>
		template <typename _T2, typename ..._Ts>
		inline void Operators::Containers::OperatorContainer<_T>::setManyBodyMat(size_t _Nh, OperatorComb<_T, _Ts...>* _op, _Ts ...a)
		{
			using res_typ = typename std::common_type<_T, _T2>::type; 		// get the common type from the operators and the Hilbert space - the result type

			// store all the measured values
			this->manyBodyMatrix_ 	= GeneralizedMatrix<_T>(_Nh, true);

			// setup the matrix
			const bool _isQuadratic	= _op->getIsQuadratic();				// check if the operator is quadratic
			if (_isQuadratic)
				this->manyBodyMatrix_ = algebra::cast<_T>(_op->template generateMat<true, res_typ, GeneralizedMatrix>(_Nh, a...));
			else
				this->manyBodyMatrix_ = algebra::cast<_T>(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_Nh, a...));	
		}

		// ------------------------------------------------------------------------------------------------------------------------------------------

		/*
		* @brief Sets the many body matrix (Operator Matrix) to apply it later for the states in a many body representation. It can be used as a combination
		* of the operators acting on the Hilbert space. This means that one uses the Hilbert space to generate the matrix for the operator.
		* @param _H Hilbert space - the Hilbert space in which the operator is acting
		* @param _op operator to be used for the many body matrix creation - multiple operators in the vector - must act in the same Hilbert space
		* @param ...a additional parameters to the operators - if needed - from the general operator definition
		*/
		template<typename _T>
		template <typename _T2, typename ..._Ts>
		inline void Operators::Containers::OperatorContainer<_T>::setManyBodyMat(const Hilbert::HilbertSpace<_T2>& _hilb, std::vector<Operator<_T, _Ts...>*>& _op, _Ts ...a)
		{
			using res_typ = typename std::common_type<_T, _T2>::type; 		// get the common type from the operators and the Hilbert space - the result type

			const size_t _Nh 		= _hilb.getHilbertSize();
			const size_t _Nhfull	= _hilb.getFullHilbertSize();

			// store all the measured values
			this->manyBodyMatrix_ 	= GeneralizedMatrix<_T>(_hilb.getHilbertSize(), true);
			const bool _isFull 		= _Nh == _Nhfull;

			// setup the matrix
			for (const Operators::Operator<_T, _Ts...>& _op : this->op_)
			{		
				const bool _isquadratic = _op.getIsQuadratic();			// check if the operator is quadratic
				// if we don't need to apply the symmetries
				if (_isFull) {	
					GeneralizedMatrix<_T> _Min;
					if (_isquadratic)
						_Min = algebra::cast<_T>(_op->template generateMat<true, res_typ, GeneralizedMatrix>(_Nh, a...));
					else
						_Min = algebra::cast<_T>(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_Nh, a...));

					this->manyBodyMatrix_ += _Min;
				}
				else {	
					GeneralizedMatrix<_T> _Min;
					_Min.setSparse(_op.template generateMat<false, res_typ, typename arma::SpMat>(_hilb, a...));
					this->manyBodyMatrix_ += algebra::cast<_T>(_Min);
				}
			}
		}

		template<typename _T>
		template <typename _T2, typename ..._Ts>
		inline void Operators::Containers::OperatorContainer<_T>::setManyBodyMat(const Hilbert::HilbertSpace<_T2>& _hilb, std::vector<OperatorComb<_T, _Ts...>*>& _op, _Ts ...a)
		{
			using res_typ = typename std::common_type<_T, _T2>::type; 		// get the common type from the operators and the Hilbert space - the result type

			const size_t _Nh 		= _hilb.getHilbertSize();
			const size_t _Nhfull	= _hilb.getFullHilbertSize();

			// store all the measured values
			this->manyBodyMatrix_ 	= GeneralizedMatrix<_T>(_hilb.getHilbertSize(), true);
			const bool _isFull 		= _Nh == _Nhfull;

			// setup the matrix
			for (const Operators::Operator<_T, _Ts...>& _op : this->op_)
			{		
				const bool _isquadratic = _op.getIsQuadratic();			// check if the operator is quadratic
				// if we don't need to apply the symmetries
				if (_isFull) {	
					GeneralizedMatrix<_T> _Min;
					if (_isquadratic)
						_Min = algebra::cast<_T>(_op->template generateMat<true, res_typ, GeneralizedMatrix>(_Nh, a...));
					else
						_Min = algebra::cast<_T>(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_Nh, a...));

					this->manyBodyMatrix_ += _Min;
				}
				else {	
					GeneralizedMatrix<_T> _Min;
					_Min.setSparse(_op.template generateMat<false, res_typ, typename arma::SpMat>(_hilb, a...));
					this->manyBodyMatrix_ += algebra::cast<_T>(_Min);
				}
			}
		}

		// ------------------------------------------------------------------------------------------------------------------------------------------

		template <typename _T>
		template <typename _T2, typename ..._Ts>
		inline void Operators::Containers::OperatorContainer<_T>::setManyBodyMat(size_t _Nh, std::vector<Operator<_T, _Ts...>*>& _op, _Ts ...a)
		{
			using res_typ = typename std::common_type<_T, _T2>::type; 		// get the common type from the operators and the Hilbert space - the result type

			// store all the measured values
			this->manyBodyMatrix_ 	= GeneralizedMatrix<_T>(_Nh, true);

			// setup the matrix
			for (const Operators::Operator<_T, _Ts...>& _op : this->op_)
			{		
				const bool _isquadratic = _op.getIsQuadratic();			// check if the operator is quadratic
				// if we don't need to apply the symmetries
				GeneralizedMatrix<_T> _Min;
				if (_isquadratic)
					_Min = algebra::cast<_T>(_op->template generateMat<true, res_typ, GeneralizedMatrix>(_Nh, a...));
				else
					_Min = algebra::cast<_T>(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_Nh, a...));

				this->manyBodyMatrix_ += _Min;
			}
		}

		template <typename _T>
		template <typename _T2, typename ..._Ts>
		inline void Operators::Containers::OperatorContainer<_T>::setManyBodyMat(size_t _Nh, std::vector<OperatorComb<_T, _Ts...>*>& _op, _Ts ...a)
		{
			using res_typ = typename std::common_type<_T, _T2>::type; 		// get the common type from the operators and the Hilbert space - the result type

			// store all the measured values
			this->manyBodyMatrix_ 	= GeneralizedMatrix<_T>(_Nh, true);

			// setup the matrix
			for (const Operators::Operator<_T, _Ts...>& _op : this->op_)
			{		
				const bool _isquadratic = _op.getIsQuadratic();			// check if the operator is quadratic
				// if we don't need to apply the symmetries
				GeneralizedMatrix<_T> _Min;
				if (_isquadratic)
					_Min = algebra::cast<_T>(_op->template generateMat<true, res_typ, GeneralizedMatrix>(_Nh, a...));
				else
					_Min = algebra::cast<_T>(_op->template generateMat<false, res_typ, GeneralizedMatrix>(_Nh, a...));

				this->manyBodyMatrix_ += _Min;
			}
		}

		// ##########################################################################################################################################

		/*
		* @brief Samples the operator for the given state and stores the value in the samples
		* @param _state state on which the operator is acting
		* @param i index of the sample
		* @param j index of the sample
		* @returns the value of the operator for the given state
		*/
		template<typename _T>
		template<typename _T1>
		inline arma::Col<_T1> Operators::Containers::OperatorContainer<_T>::sample(const arma::Col<_T1>& _state, size_t i, size_t j)
		{
			if (this->manyBodyMatrix_.isSparse())
			{
				arma::Col<_T1> _Cout = this->manyBodyMatrix_.getSparse() * _state;
				this->manyBodyVal_(i, j) = arma::cdot(_state, _Cout);
				return _Cout;
			}
			auto _Cout 					= this->manyBodyMatrix_.getDense() * _state;
			this->manyBodyVal_(i, j) 	= arma::cdot(_state, _Cout);
			return _Cout;
		}

		// ##########################################################################################################################################

		/*
		* @brief Normalizes the operator value and stores it in the samples - for the Monte Carlo sampling etc. - this allows 
		* to store the value of the operator for the given state and then normalize it to the number of samples taken and obtain
		* all averages for multiple blocks of the samples. [nBlocks x nSamples (average)]
		* @param reset if true, the current value of the operator is reset to zero and the number of samples is reset to zero 
		*/
		template <typename _T>
		inline void Operators::Containers::OperatorContainer<_T>::normalize(bool reset)
		{
			this->samples_.push_back(this->currentValue_ / (long double)this->sample_num_);
			if (reset)
			{
				this->resetValue();
				this->resetSampleNum();
			}
		}

	};
};


// ##########################################################################################################################################

// ############################################### E X T E N S I V E   O P E R A T O R S ####################################################

// ##########################################################################################################################################

namespace Operators {
 	template <typename _T, typename ..._Ts>
 	class OperatorExt : std::vector<Operator<_T, _Ts...>>
 	{
 		using base_t = Operator<_T, _Ts...>;
 		//typedef typename _OP<_T>::template INP_EXT<_Ts...> repType;	 // type returned for representing, what it does with state and values!

		SymGenerators name_									=			SymGenerators::E;   // name of the operator
		std::string nameS_									=			"Extensive";		// name of the operator in string

		// normalization constant
		_T norm_											=			1.0;				// normalization constant

		// ----------------- V I R T U A L ----------------
		virtual void init()									{};

	public:
		// ----------------------------------------------------------------------------------------------------

		virtual ~OperatorExt()								=			default;			// default destructor

		// standard constructor
		OperatorExt() 
		{ 
			this->init(); 
			this->resize(0);
		};

		// with the usage of the elements in the state vector
		OperatorExt(const std::vector<base_t>& _operators, const std::string& _nameS, _T _norm = 0.0)
			: nameS_(_nameS), norm_(_norm)
		{ 
			this->assign(_operators.begin(), _operators.end());
			init(); 
		};

		// Copy constructor
		OperatorExt(const OperatorExt& other)
			: std::vector<Operator<_T, _Ts...>>(other), name_(other.name_), nameS_(other.nameS_), norm_(other.norm_)
		{
			this->init();
		}

		// Move constructor
		OperatorExt(OperatorExt&& other) noexcept
			: std::vector<Operator<_T, _Ts...>>(std::move(other)), name_(std::move(other.name_)), nameS_(std::move(other.nameS_)), norm_(std::move(other.norm_))
		{
			this->init();
		}

		// Copy assignment operator
		OperatorExt& operator=(const OperatorExt& other)
		{
			if (this != &other) 
			{
				std::vector<Operator<_T, _Ts...>>::operator=(other);
				name_	= other.name_;
				nameS_	= other.nameS_;
				norm_	= other.norm_;
			}
			return *this;
		}

		// Move assignment operator
		OperatorExt& operator=(OperatorExt&& other) noexcept
		{
			if (this != &other) {
				std::vector<Operator<_T, _Ts...>>::operator=(std::move(other));
				name_	= std::move(other.name_);
				nameS_	= std::move(other.nameS_);
				norm_	= std::move(other.norm_);
			}
			return *this;
		}

		// ----------------------------------------------------------------------------------------------------

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>& _other)
		{
			this->Ns_		=		_other.Ns_;
			this->lat_		=		_other.lat_;
			this->fun_		=		_other.fun_;
			this->eigVal_	=		_other.eigVal_;
			this->name_		=		_other.name_;
			return *this;
		}

		Operator<_T, _Ts...>& operator=(const Operator<_T, _Ts...>&& _other)
		{
			this->Ns_		=		std::move(_other.Ns_);
			this->lat_		=		std::move(_other.lat_);
			this->fun_		=		std::move(_other.fun_);
			this->eigVal_	=		std::move(_other.eigVal_);
			this->name_		=		std::move(_other.name_);
			return *this;
		}
 	
		// ----------------------------------------------------------------------------------------------------

		// -------------------- SETTERS -------------------

		auto setName(SymGenerators _name)					-> void							{ this->name_ = _name;											};
		auto setNameS(const std::string& _name)				-> void							{ this->nameS_ = _name;											};
		auto setNorm(_T _norm)								-> void							{ this->norm_ = _norm;											};
		auto setOperators(const std::vector<base_t>& _ops)	-> void							{ this->assign(_ops.begin(), _ops.end());						};
		auto setOperators(std::vector<base_t>&& _ops)		-> void							{ this->assign(std::move(_ops.begin()), std::move(_ops.end())); };
		auto setOperators(const OperatorExt& _ops)			-> void							{ this->assign(_ops.begin(), _ops.end());						};
		auto setOperators(OperatorExt&& _ops)				-> void							{ this->assign(std::move(_ops.begin()), std::move(_ops.end())); };
		auto setOperator(size_t _idx, const base_t& _op)	-> void							{ this->at(_idx) = _op;											};
		auto setOperator(size_t _idx, base_t&& _op)			-> void							{ this->at(_idx) = std::move(_op);								};

		// -------------------- ADDERS ---------------------

		auto addOperator(const base_t& _op)					-> void							{ this->push_back(_op);											};
		auto addOperator(base_t&& _op)						-> void							{ this->push_back(std::move(_op));								};

		// -------------------- GETTERS --------------------

		auto getName()										const -> SymGenerators			{ return this->name_;											};
		auto getNameS()										const -> std::string			{ return this->nameS_;											};
		auto getNameG()										const -> std::string			{ return SSTR(getSTR_SymGenerators(this->name_));				};
		auto getOperators()									const -> std::vector<base_t>	{ return *this;													};
		auto getOperator(size_t _idx)						const -> const base_t&			{ return this->at(_idx);										};
		auto getNorm()										const -> _T						{ return this->norm_;											};
		
		// ----------------------------------------------------------------------------------------------------

	public:

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% O P E R A T O R S

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T,  template <class _TM = _TinMat> class _MatType, typename _InT = u64>
		typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
			generateMat(_InT _dim, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, uint _spinModes = 2>
		_MatType<typename std::common_type<_TinMat, _T1>::type> 
			generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil, _Ts... _arg) const;

		template<bool _standarize = false, typename _TinMat = _T, template <class _TM = _TinMat> class _MatType, typename _T1, typename _T2, uint _spinModes = 2>
		_MatType<typename std::common_type<_TinMat, _T1, _T2>::type> 
			generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil1, const Hilbert::HilbertSpace<_T2, _spinModes>& _Hil2, _Ts... _arg);
	};

	// ##########################################################################################################################################

	// ##########################################################################################################################################

	// ##########################################################################################################################################
 
	/*
	* @brief Creates a most basic operator matrix knowing only the dimension of the Hilbert space. 
	* The operator is acting on the same Hilbert space as the one it is acting on.
	* For the total Hilbert space known to be without symmetries - not looking for representatives
	* @brief _dim A dimension of the Hilbert space
	* @returns A matrix representing the operator
	*/
	template<typename _T, typename ..._Ts> 
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _InT>
	typename std::enable_if<HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, _MatType<_TinMat>>::type
		Operators::OperatorExt<_T, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
	{

		_MatType<_TinMat> op(_dim, _dim);
		for (auto& _op : *this)
		{
			_MatType<_TinMat> _opMat = _op.template generateMat<_TinMat, _MatType, _InT, false>(_dim, _arg...);
			op += _opMat;
		}

		// standarize the operator
		if(_standarize)
			return standarizeOperator(op);
		return op;
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates a most basic operator matrix knowing only the dimension of the Hilbert space. 
	* The operator is acting on the same Hilbert space as the one it is acting on.
	* For the total Hilbert space known to be without symmetries - not looking for representatives
	* @brief _dim A dimension of the Hilbert space
	* @returns A matrix representing the operator
	*/
	template<typename _T, typename ..._Ts> 
	template<bool _standarize, typename _TinMat,  template <class> class _MatType, typename _InT>
	typename std::enable_if<!HasMatrixType<_MatType<_TinMat>> && std::is_integral<_InT>::value, GeneralizedMatrix<_TinMat>>::type
		Operators::OperatorExt<_T, _Ts...>::generateMat(_InT _dim, _Ts ..._arg) const
	{

		GeneralizedMatrix<_TinMat> op(_dim);

		for (auto& _op : *this)
		{
			auto _opMat = _op.template generateMat<_TinMat, _InT, false>(_dim, _arg...);
			op += _opMat;
		}

		// standarize the operator
		op.standarize();

		return op;
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates an operator matrix whenever the operator is not transforming the state from a different symmetry sector.
	* Uses the Hilbert space that stores the state transformations from the representative base.
	* @param _Hil the Hilbert space in which we operate
	* @param _arg arguments for the operator
	* @returns A matrix representing the operator
	*/
	template<typename _T, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _T1, uint _spinModes>
	inline _MatType<typename std::common_type<_TinMat, _T1>::type> Operators::OperatorExt<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil, _Ts ..._arg) const
	{
		using res_typ	=	typename std::common_type<_T1, _TinMat>::type;
		u64 Nh			=	_Hil.getHilbertSize();
		_MatType<res_typ> op(Nh, Nh);

		for(auto& _op : *this)
		{
			_MatType<res_typ> _opMat = _op.template generateMat<_TinMat, _MatType, _T1, _spinModes, false>(_Hil, _arg...);
			op += _opMat;
		}

		// standarize the operator
		if(_standarize)
			return standarizeOperator(op);
		return op;
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates an operator matrix whenever the operator is transforming the state to a different symmetry sector 
	* @param _Hil the Hilbert space in which we operate
	* @warning (takes into account that we are going to a different symmetry sector so matrix is not square)
	* @trace O = \sum _{i \in A} \sum _{j \in _B} |i>_A <j|_B O_{ij}
	*/
	template<typename _T, typename ..._Ts>
	template<bool _standarize, typename _TinMat, template <class> class _MatType, typename _T1, typename _T2, uint _spinModes>
	inline _MatType<typename std::common_type<_TinMat, _T1, _T2>::type> Operators::OperatorExt<_T, _Ts...>::generateMat(const Hilbert::HilbertSpace<_T1, _spinModes>& _Hil1, const Hilbert::HilbertSpace<_T2, _spinModes>& _Hil2, _Ts ..._arg)
	{
		// using res_typ		=	typename std::common_type<_T1, _T, _T2>::type;
		u64 NhA				=	_Hil1.getHilbertSize();
		u64 NhB				=	_Hil2.getHilbertSize();
		arma::SpMat<_TinMat> op(NhA, NhB);

		for(auto& _op : *this)
		{
			_MatType<_TinMat> _opMat = _op.template generateMat<_TinMat, _MatType, _T1, _T2, _spinModes, false>(_Hil1, _Hil2, _arg...);
			op += _opMat;
		}

		// standarize the operator
		if(_standarize)
			return standarizeOperator(op);
		return op;
	}

	// ##########################################################################################################################################

};

#endif // !GENERAL_OPERATOR_H
