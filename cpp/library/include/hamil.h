#pragma once
/***************************************
* Defines the generic lattice Hamiltonian
* class. Allows for later inhertiance
* for a fine model specialization.
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#ifndef SYMMETRIES_H
	#include "algebra/operators.h"
#endif // !SYMMETRIES_H

#ifndef HAMIL_H
#define HAMIL_H

// ############################ EXISTING MODELS ############################
enum MY_MODELS 															// #
{																		// #	
	ISING_M, XYZ_M, HEI_KIT_M, QSM_M, NONE 								// #
};																		// #
BEGIN_ENUM(MY_MODELS)													// #
{																		// #
	DECL_ENUM_ELEMENT(ISING_M),											// #
	DECL_ENUM_ELEMENT(XYZ_M),											// #
	DECL_ENUM_ELEMENT(HEI_KIT_M),										// #
	DECL_ENUM_ELEMENT(QSM_M),											// #
	DECL_ENUM_ELEMENT(NONE)												// #
}																		// #
END_ENUM(MY_MODELS)									 					// #	
// #########################################################################

// ########################### SAVING EXTENSIONS ###########################
enum HAM_SAVE_EXT {														// #
	dat, h5																// #
};																		// #
BEGIN_ENUM(HAM_SAVE_EXT)												// #
{																		// #
	DECL_ENUM_ELEMENT(dat), DECL_ENUM_ELEMENT(h5)						// #
}																		// #
END_ENUM(HAM_SAVE_EXT)													// #	
// #########################################################################
	
const std::string DEF_INFO_SEP		= std::string("_");												// defalut separator in info about the model
#define DISORDER_EQUIV(type, param) type param		= 1;	\
									type param##0	= 0;	\
									arma::Col<type> d##param

#define PARAM_W_DISORDER(param, s)	(this->param + this->d##param(s))								// gets the value moved by the disorder strength

#define PARAMS_S_DISORDER(p, toSet)	toSet += SSTR(",") + SSTR(#p) + SSTR("=")  + STRP(this->p, 3);	\
									toSet += ((this->p##0 == 0.0) ? "" : SSTR(",") + SSTR(#p) + SSTR("0=") + STRP(this->p##0, 3))
																									// gets the information about the disorder

// include statistics
#include "quantities/statistics.h"

// ##########################################################################################################################################

// ######################################################### H A M I L T O N I A N ##########################################################

// ##########################################################################################################################################

template <typename _T, uint _spinModes = 2>
class Hamiltonian 
{
public:
	// definitions
	using NQSFun										= std::function<cpx(std::initializer_list<int>, std::initializer_list<double>)>;
	const uint Nhl										= _spinModes;

	// Hilbert space
	Hilbert::HilbertSpace<_T, _spinModes> hilbertSpace;

protected:
	// lattice
	std::shared_ptr<Lattice> lat_;
	// ------------------------------------------- CLASS TYPES ----------------------------------------------
	MY_MODELS type_										= MY_MODELS::NONE;
	uint Ns_											= 1;
	uint Ns												= 1;
	u64 Nh_												= 1;
	u64 Nh												= 1;

	// ------------------------------------------- CLASS FIELDS ---------------------------------------------
	u64 avEnIdx											= -1;														
	double avEn											= 0.0;
	double minEn										= 0.0;
	double maxEn										= 0.0;
	
	// matrices
	arma::Mat<_T> eigVec_;								// matrix of the eigenvectors in increasing order
	arma::SpMat<_T> H_;									// the Hamiltonian
	arma::Mat<_T> K_;									// the Krylov Vectors (if needed)
	arma::vec eigVal_;									// eigenvalues vector
public:
	randomGen ran_;										// consistent quick random number generator
	std::string info_;									// information about the model
	
	// -------------------------------------------- CONSTRUCTORS --------------------------------------------
	
	virtual ~Hamiltonian();
	Hamiltonian();
	Hamiltonian(const size_t _Ns);
	Hamiltonian(const Hilbert::HilbertSpace<_T, _spinModes>& hilbert);
	Hamiltonian(Hilbert::HilbertSpace<_T, _spinModes>&& hilbert);

	// ------------------------------------------- PRINTERS ---------------------------------------------------

	static void printBaseState(	std::ostream& output,	u64 _s, _T val, v_1d<int>& _tmpVec,	double _tol = 5e-2);

	template <template <typename> class _V, typename _TV = _T>
	static void prettyPrint(	std::ostream& output,	const _V<_TV>& state, uint Ns, double _tol = 5e-2);	
	
	void print(u64 _id)									const										{ this->eigVec_.col(_id).print("|"+STR(_id)+">\n");								};
	void print()										const										{ this->H_.print("H=\n");														};
																																												
	// --------------------------------------------- INFO -----------------------------------------------------
	
	std::string info(std::string name = "", const v_1d<std::string>& skip = {}, std::string sep = "_")	const;
	virtual std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2)	const = 0;

	// --------------------------------------------- INITS ----------------------------------------------------
	virtual auto randomize(double,double,const strVec&)	-> void										{};
	auto init()											-> void;

	// -------------------------------------------- GETTERS ---------------------------------------------------
	auto getDegeneracies()								const -> v_2d<u64>;
	auto getEnArndAvIdx(long long _l, long long _r)		const -> std::pair<u64, u64>;
	auto getEnArndEIdx(long long _l, long long _r, u64)	const -> std::pair<u64, u64>;
	auto getEnArndEnEps(u64 _Eidx, double _eps)			const -> std::pair<u64, u64>;

	auto getEnPairsIdx(u64 _mn, u64 _mx, double _tol)	const -> v_1d<std::tuple<double, u64, u64>>;
	auto getEnAvIdx()									const -> u64								{ return this->avEnIdx;															};								
	auto getEnAv()										const -> double								{ return this->avEn;															};
	// hilbert
	auto getHilbertSize()								const -> u64								{ return this->Nh;																};			
	auto getHilbertSpace()								const -> const Hilbert::HilbertSpace<_T>&	{ return this->hilbertSpace;													};							
	// hamiltonian
	auto getHamiltonian()								const -> const arma::SpMat<_T>&				{ return this->H_;																};
	auto getHamiltonian(u64 i, u64 j)					const -> _T									{ return this->H_(i, j);														};
	auto getHamiltonianSize()							const -> double								{ return this->H_.size() * sizeof(this->H_(0, 0));								};								
	auto getHamiltonianSizeH()							const -> double								{ return std::pow(this->hilbertSpace.getHilbertSize(), 2) * sizeof(_T); };
	auto getSymRot()									const -> arma::SpMat<_T>					{ return this->hilbertSpace.getSymRot();										};
	// eigenvectors
	auto getEigVec()									const -> const arma::Mat<_T>&				{ return this->eigVec_;															};							
	auto getEigVec(u64 idx)								const -> arma::subview_col<_T>				{ return this->eigVec_.col(idx);												};			
	auto getEigVec(u64 idx, u64 elem)					const -> _T									{ return this->eigVal_(elem, idx);												};				
	auto getEigVec(std::string _dir, u64 _mid, 
		HAM_SAVE_EXT _typ, bool _app = false)			const -> void;
	// eigenvalues
	auto getEigVal()									const -> const arma::vec&					{ return this->eigVal_;															};						
	virtual auto getEigVal(std::string _dir,
		HAM_SAVE_EXT _typ, bool _app = false)			const -> void;
	auto getEigVal(u64 idx)								const -> double								{ return this->eigVal_(idx);													};	
	// info
	auto getInfo(const strVec& skip = {},
		std::string sep = DEF_INFO_SEP, int prec = 2)	const -> std::string						{ return this->info_;															};
	// types
	auto getType()										const -> std::string						{ return SSTR(getSTR_MY_MODELS(this->type_));									};
	auto getTypeI()										const -> uint								{ return this->type_;															};
	// lattice (if applicable)
	auto getLat()										const -> std::shared_ptr<Lattice>			{ return this->lat_;															};
	auto getNs()										const -> uint								{ return this->Ns_;																};
	auto getBC()										const -> BoundaryConditions					{ return this->lat_->get_BC();													};
	
	// ------------------------------------------- SETTERS -----------------------------------------------------
	
	auto setSeed(u64 seed)								-> void										{ this->ran_.newSeed(seed);														};

	// ----------------------------------------- HAMILTONIAN ---------------------------------------------------
protected:
	virtual void hamiltonian();
	auto setHElem(u64 k, _T val, u64 newIdx)			-> void;									// sets the Hamiltonian elements in a virtual way
	auto calcAvEn()										-> void;									// calculate the average energy
public:
	auto calcEnIdx(double _E)							-> u64;										// calculate the index of the energy closest to the given energy

public:
	auto buildHamiltonian()								-> void;
	auto diagH(bool woEigVec = false)					-> void;									// diagonalize the Hamiltonian
	auto diagHs(bool woEigVec = false)					-> void;									// diagonalize the Hamiltonian sparse
	auto diagH(bool woEigVec, 
			   uint k, 
			   uint subdim = 0, 
			   uint maxiter = 1000,
			   double tol = 0, 
			   std::string form = "sm")					-> void;									// diagonalize the Hamiltonian using Lanczos' method

public:
	// ------------------------------------------ LOCAL ENERGY -------------------------------------------------
	virtual void locEnergy(	u64 _elemId, 
							u64 _elem, 
							uint _site)					= 0; 
	virtual cpx locEnergy(u64 _id, uint s, NQSFun f1)	= 0;										// returns the local energy for VQMC purposes
	virtual cpx locEnergy(const arma::Col<double>& v, 
						  uint site,
						  NQSFun f1)					{ return 0; };								// returns the local energy for VQMC purposes
	
	// ----------------------------------------- FOR OTHER TYPES -----------------------------------------------
	virtual void updateInfo()							= 0;

public:
	void generateFullMap()								{ this->hilbertSpace.generateFullMap();		}; // generates the full Hilbert space map

	// --------------------------------------------- CLEAR -----------------------------------------------------
	void clearEigVec()									{ this->eigVec_.reset();					}; // resets the eigenvectors memory to 0
	void clearEigVal()									{ this->eigVal_.reset();					}; // resets the energy memory to 0
	void clearH()										{ this->H_.reset();							}; // resets the hamiltonian memory to 0
	void clear()										{ this->clearEigVec(); this->clearEigVal(); this->clearH();	}; 

	// --------------------------------------------- OTHER -----------------------------------------------------
};


// ##########################################################################################################################################

// ############################################################## C O N S T R ###############################################################

// ##########################################################################################################################################

template<typename _T, uint _spinModes>
inline Hamiltonian<_T, _spinModes>::Hamiltonian()
	: ran_(randomGen())
{
	CONSTRUCTOR_CALL;
}

/*
* @brief Constructor of the Hamiltonian class for the systems that don't require the lattice
* @param _Ns number of particles in the system
*/
template<typename _T, uint _spinModes>
inline Hamiltonian<_T, _spinModes>::Hamiltonian(const size_t _Ns)
	: Hamiltonian<_T, _spinModes>()
{
	this->Ns_	= _Ns;
	this->Ns	= _Ns;
	this->hilbertSpace = Hilbert::HilbertSpace<_T, _spinModes>(Ns_);
	this->Nh	=  ULLPOW(Ns_);
	this->Nh_	=  ULLPOW(Ns_);
}

/*
* @brief Constructor of the Hamiltonian with the Hilbert space
*/
template<typename _T, uint _spinModes>
inline Hamiltonian<_T, _spinModes>::Hamiltonian(const Hilbert::HilbertSpace<_T, _spinModes>& hilbert)
	: Hamiltonian<_T, _spinModes>()
{
	this->hilbertSpace = hilbert;
	this->lat_	=	this->hilbertSpace.getLattice();
	this->Ns	=	this->hilbertSpace.getNs();
	this->Ns_   =	this->Ns;
	this->Nh	=	this->hilbertSpace.getHilbertSize();
	this->Nh_	=	this->Nh;
};

/*
* @brief Constructor with move semantics for the Hilbert space
*/
template<typename _T, uint _spinModes>
inline Hamiltonian<_T, _spinModes>::Hamiltonian(Hilbert::HilbertSpace<_T, _spinModes>&& hilbert)
	: hilbertSpace(std::move(hilbert))
{
	this->ran_	=	randomGen();
	this->lat_	=	this->hilbertSpace.getLattice();
	this->Ns	=	this->hilbertSpace.getNs();
	this->Ns_	=	this->Ns;
	this->Nh	=	this->hilbertSpace.getHilbertSize();
	this->Nh_	=	this->Nh;
};			

// ##########################################################################################################################################

// ################################################################ I N F O #################################################################

// ##########################################################################################################################################

/*
* @brief Sets and gets the information about the model
* @param skip vector of elements to be skipped in the info showcase
* @returns trimmed information about the model
*/
template<typename _T, uint _spinModes>
std::string Hamiltonian<_T, _spinModes>::info(std::string name, const v_1d<std::string>& skip, std::string sep)	const
{
	auto tmp = (name == "") ? splitStr(this->info_, ",") : splitStr(name, ",");
	std::string tmp_str = "";
	for (int i = 0; i < tmp.size(); i++) {
		bool save = true;
		for (auto& skip_param : skip)
			// skip the element if we don't want it to be included in the info
			save = !(splitStr(tmp[i], "=")[0] == skip_param);
		if (save) tmp_str += tmp[i] + ",";
	}
	tmp_str.pop_back();
	return tmp_str;
};

// ##########################################################################################################################################

template<typename _T, uint _spinModes>
Hamiltonian<_T, _spinModes>::~Hamiltonian()
{
	DESTRUCTOR_CALL;
	LOGINFO("Base Hamiltonian destructor called.", LOG_TYPES::INFO, 3);
	this->H_.reset();
	this->eigVal_.reset();
	this->eigVec_.reset();
}

// ##########################################################################################################################################

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template<typename _T, uint _spinModes>
void Hamiltonian<_T, _spinModes>::hamiltonian()
{
	if (this->Nh == 0)
	{
		LOGINFOG("Empty Hilbert, not building anything.", LOG_TYPES::INFO, 1);
		return;
	}
	this->init();
	for (u64 k = 0; k < this->Nh; ++k)
	{
		u64 kMap = this->hilbertSpace.getMapping(k);
		for (uint site_ = 0; site_ <= this->Ns - 1; ++site_)
			this->locEnergy(k, kMap, site_);
	}
}

// ##########################################################################################################################################

/*
* @brief Initialize Hamiltonian matrix.
*/
template<typename _T, uint _spinModes>
void Hamiltonian<_T, _spinModes>::init()
{
	// hamiltonian memory reservation
	BEGIN_CATCH_HANDLER
	{
		this->H_ = arma::SpMat<_T>(this->Nh, this->Nh);
	}
	END_CATCH_HANDLER("Memory exceeded", std::runtime_error("Memory for the Hamiltonian setting exceeded"););
};

// ##########################################################################################################################################

/*
* @builds Hamiltonian and gets specific info! 
*/
template<typename _T, uint _spinModes>
inline void Hamiltonian<_T, _spinModes>::buildHamiltonian()
{
	auto _t = NOW;
	LOGINFO("Started buiding Hamiltonian" + this->getInfo(), LOG_TYPES::TRACE, 2);
	this->hamiltonian();
	LOGINFO("Finished buiding Hamiltonian" + this->getInfo(), LOG_TYPES::TRACE, 2);
	LOGINFO(_t, "Hamiltonian: " + this->getInfo(), 3);
}

// ##########################################################################################################################################

/*
* @brief Calculates the index closest to the average energy in the Hamiltonian. The index is stored in the avEnIdx variable.
*/
template<typename _T, uint _spinModes>
inline void Hamiltonian<_T, _spinModes>::calcAvEn()
{
	// calculates the middle spectrum element
	this->avEn	= arma::mean(this->eigVal_);
	this->minEn = this->eigVal_(0);
	this->maxEn = this->eigVal_(this->Nh - 1);
	u64 Nh		= this->getHilbertSize();

	// calculates the energy difference to find the closest element (the smallest)
	v_1d<double> tmp(Nh, 0.0);
	for (int i = 0; i < Nh; i++)
		tmp[i] = std::abs(this->getEigVal(i) - this->avEn);

	// get the iterator
	v_1d<double>::iterator _it	= std::min_element(std::begin(tmp), std::end(tmp));
	this->avEnIdx				= _it - std::begin(tmp);
}

/*
* @brief Calculates the index of the energy closest to the given energy
* @param _E energy to be compared with
* @returns index of the energy closest to the given energy
*/
template<typename _T, uint _spinModes>
inline u64 Hamiltonian<_T, _spinModes>::calcEnIdx(double _E)
{
	// calculates the energy difference to find the closest element (the smallest)
	v_1d<double> tmp(Nh, 0.0);
	for (int i = 0; i < Nh; i++)
		tmp[i] = std::abs(this->getEigVal(i) - _E);

	// get the iterator
	v_1d<double>::iterator _it	= std::min_element(std::begin(tmp), std::end(tmp));
	return _it - std::begin(tmp);
}

// ##########################################################################################################################################

/*
* @brief Calculates the index closest to the average energy in the Hamiltonian. The index is stored in the avEnIdx variable.
* @param _l number of elements to the left from the average energy
* @param _r number of elements to the right from the average energy
* @returns pair of indices of the energy spectrum around the average energy
*/
template<typename _T, uint _spinModes>
inline std::pair<u64, u64> Hamiltonian<_T, _spinModes>::getEnArndAvIdx(long long _l, long long _r) const
{
	return SystemProperties::hs_fraction_around_idx(_l, _r, this->avEnIdx, this->Nh);
}

// ##########################################################################################################################################

/*
* @brief Calculates the index closest to the given energy in the Hamiltonian.
* @param _l number of elements to the left from the given energy
* @param _r number of elements to the right from the given energy
* @param _idx index of the energy to be compared with
* @returns pair of indices of the energy spectrum around the given energy
*/
template<typename _T, uint _spinModes>
inline std::pair<u64, u64> Hamiltonian<_T, _spinModes>::getEnArndEIdx(long long _l, long long _r, u64 _idx) const
{
	return SystemProperties::hs_fraction_around_idx(_l, _r, _idx, this->Nh);
}

// ------------------------------------------------------------------------------------------------------------------------------------------

/*
* @brief Calculates the indices of the energy pairs in the spectrum that are close to each other within the given tolerance
* This is based on the energy density! E/V
* @param _Eidx index of the energy spectrum
* @param _eps tolerance of the energy difference
*/
template<typename _T, uint _spinModes>
inline std::pair<u64, u64> Hamiltonian<_T, _spinModes>::getEnArndEnEps(u64 _Eidx, double _eps) const
{
	u64 _imax = _Eidx;
	u64 _imin = _Eidx;
	// find the maximum index
	while (((this->eigVal_(_imax) - this->eigVal_(_Eidx)) / this->Ns) < _eps && _imax < this->Nh)
		_imax++;
	// find the minimum index
	while (((this->eigVal_(_Eidx) - this->eigVal_(_imin)) / this->Ns) < _eps && _imin >= 0)
		_imin--;
	return std::make_pair(_imin, _imax);
}

// ##########################################################################################################################################

/*
* @brief Calculates the indices of the energy pairs in the spectrum that are close to each other within the given tolerance
* @param _mn minimum index of the energy spectrum
* @param _mx maximum index of the energy spectrum
* @param _tol tolerance of the energy difference
* @returns vector of tuples of the energy difference and the indices of the energy spectrum
*/
template<typename _T, uint _spinModes>
inline v_1d<std::tuple<double, u64, u64>> Hamiltonian<_T, _spinModes>::getEnPairsIdx(u64 _mn, u64 _mx, double _tol) const
{
	return SystemProperties::hs_fraction_offdiag(_mn, _mx, this->Nh, this->eigVal_, this->avEn, _tol * this->Ns_, true);
}

// ##########################################################################################################################################

// ############################################################ P R I N T I N G #############################################################

// ##########################################################################################################################################

/*
* @brief prints the base state in the BRAKET notation
* @param _s base state to be printed in the integer form
* @param _val coefficient before the state
* @param _tmpVec to be used as reference vector
* @param _tol tolerance of the coefficient absolute value
*/
template<typename _T, uint _spinModes>
inline void Hamiltonian<_T, _spinModes>::printBaseState(std::ostream& output, u64 _s, _T val, v_1d<int>& _tmpVec, double _tol)
{
	INT_TO_BASE(_s, _tmpVec);
	if (!EQP(std::abs(val), 0.0, _tol)) 
	{
		auto pm				=			(algebra::real(val) >= 0) ? "+" : "";
		output				<< pm << val << "*|" << _tmpVec << +">";
	}
}

/*
* @brief goes through the whole state and prints it in numerical basis
* @param state the state to be printed
* @param Ns number of lattice sites
* @param tol tolerance of the coefficients absolute value
*/
template<typename _T, uint _spinModes>
template <template <typename> class _V, typename _TV>
inline void Hamiltonian<_T, _spinModes>::prettyPrint(std::ostream& output, const _V<_TV>& state, uint Ns, double tol)
{
	v_1d<int> tmpVec(Ns);
	for (u64 k = 0; k < state.size(); k++)
		printBaseState(output, k, state[k], tmpVec, tol);
	output << EL;
}

// ##########################################################################################################################################

// ######################################################### H A M I L T O N I A N ##########################################################

// ##########################################################################################################################################

/*
* @brief Sets the non-diagonal elements of the Hamimltonian matrix with symmetry sectors: therefore the matrix elements are summed over the SEC
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template<typename _T, uint _spinModes>
inline void Hamiltonian<_T, _spinModes>::setHElem(u64 k, _T val, u64 newIdx)
{
	u64 kMap = this->hilbertSpace.getMapping(k);
	BEGIN_CATCH_HANDLER
	{
		if (kMap != newIdx)
		{
			auto [idx, symEig] = this->hilbertSpace.findRep(newIdx, this->hilbertSpace.getNorm(k));
			// set Hamiltonian element. If map is empty, returns the same element as wanted - the symmetry is None
			this->H_(idx, k) += val * symEig;
		}
		else
			this->H_(k, k) += val;
	}
	END_CATCH_HANDLER("Exception in setting the Hamiltonian elements: " + VEQ(k) + "," + VEQ(kMap) + "," + VEQ(newIdx), exit(-1););
}

// ##########################################################################################################################################

// ######################################################### D I A G O N A L I Z E ##########################################################

// ##########################################################################################################################################

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* @param withoutEigenVec doesnot compute eigenvectors to save memory potentially
*/
template <typename _T, uint _spinModes>
inline void Hamiltonian<_T, _spinModes>::diagH(bool woEigVec)
{
	if (woEigVec)	Diagonalizer<_T>::diagS(this->eigVal_, this->H_);
	else				Diagonalizer<_T>::diagS(this->eigVal_, this->eigVec_, this->H_);
	this->calcAvEn();
}

// ##########################################################################################################################################

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* Modes (form):
*		From ARMA:
			- la - largest algebraic
			- sa - smallest algebraic
			- sg - smallest magnitude
			- lm - largest magnitude
*		Mine:
*			- lanczos - Lanczos method
* @param woEigVec does not compute eigenvectors to save memory potentially
* @param k number of eigenvalues to be computed
* @param subdim dimension of the subspace to be used in the Lanczos method
* @param maxiter maximum number of iterations in the Lanczos method
* @param tol tolerance of the Lanczos method
* @param form form of the diagonalization
*/
template <typename _T, uint _spinModes>
inline void Hamiltonian<_T, _spinModes>::diagH(bool woEigVec, uint k, uint subdim, uint maxiter, double tol, std::string form) 
{
	BEGIN_CATCH_HANDLER
	{
		arma::eigs_opts opts;
		opts.tol					= tol;
		opts.maxiter			= maxiter;
		opts.subdim				= (subdim == 0) ? (2 * int(k) + 1) : subdim;
		
		if (form == "sg")
		{
			LOGINFO("Diagonalizing Hamiltonian. Using: S&I", LOG_TYPES::INFO, 3);
			//if (woEigVec)		arma::eigs_sym(this->eigVal_, this->H_, arma::uword(k), 0.0, opts);
			//else					arma::eigs_sym(this->eigVal_, this->eigVec_, this->H_, arma::uword(k), 0.0, opts);
		}
		else if (form == "lanczos")
		{
			LOGINFO("Diagonalizing Hamiltonian. Using: Lanczos", LOG_TYPES::INFO, 3);
			LanczosMethod<_T>::diagS(this->eigVal_, this->eigVec_, this->H_, k, &this->ran_);
			//this->calcAvEn();
		}
		else
		{
			LOGINFO("Diagonalizing Hamiltonian. Using: Lanczos from Armadillo", LOG_TYPES::INFO, 3);
			//if (woEigVec)		arma::eigs_sym(this->eigVal_, this->H_, arma::uword(k), form.c_str());
			//else					arma::eigs_sym(this->eigVal_, this->eigVec_, this->H_, arma::uword(k), form.c_str());
		}
	}
	END_CATCH_HANDLER("Memory exceeded. DIM(H)=" + STR(this->H_.size() * sizeof(this->H_(0, 0))) + " bytes", ;);
}

// ##########################################################################################################################################

template<typename _T, uint _spinModes>
inline void Hamiltonian<_T, _spinModes>::diagHs(bool woEigVec)
{
	BEGIN_CATCH_HANDLER
	{
		if (woEigVec)			arma::eigs_sym(this->eigVal_, this->H_, this->getHilbertSize());
		else						arma::eigs_sym(this->eigVal_, this->eigVec_, this->H_, this->getHilbertSize());
	}
	END_CATCH_HANDLER("Memory exceeded. DIM(H)=" + STR(this->H_.size() * sizeof(this->H_(0, 0))) + " bytes", ;);
	this->calcAvEn();
}

// ##########################################################################################################################################

// ############################################################# G E T T E R S ##############################################################

// ##########################################################################################################################################

/*
* @brief Prints the eigenvalues into some file "energies" in some directory
* @param _dir directory to be saved onto
* @param _typ type of the file extension (dat, h5 or other).
* @param _app shall append?
*/
template<typename _T, uint _spinModes>
inline auto Hamiltonian<_T, _spinModes>::getEigVal(std::string _dir, HAM_SAVE_EXT _typ, bool _app) const -> void
{
	std::string extension = "." + SSTR(getSTR_HAM_SAVE_EXT(_typ));
	std::ofstream file;
	BEGIN_CATCH_HANDLER
	{
		switch (_typ)
		{
		case HAM_SAVE_EXT::dat:
			if(_app)
				openFile(file, _dir + "energy" + this->getInfo() + extension, std::ios::app);
			else
				openFile(file, _dir + "energy" + this->getInfo() + extension);
			for (auto i = 0; i < this->eigVal_.size(); i++)
				file << this->eigVal_(i) << EL;
			file.close();
			break;
		case HAM_SAVE_EXT::h5:
			if(_app)
				this->eigVal_.save(arma::hdf5_name(_dir + this->getInfo() + extension, "energy", arma::hdf5_opts::append));
			else
				this->eigVal_.save(arma::hdf5_name(_dir + this->getInfo() + extension, "energy"));
			break;
		default:
			LOGINFO("Wrong extension, not saving a file. Available are [.dat, .h5]", LOG_TYPES::WARNING, 1);
			break;
		}
	}
	END_CATCH_HANDLER("Exception in saving the Eigenvalues: ", getEigVal(_dir, HAM_SAVE_EXT::dat, _app););
}

// ##########################################################################################################################################

/*
* @brief Prints the eigenvectors into some file "energies" in some directory
* @param _dir directory to be saved onto
* @param _mid how many states to be saved
* @param _typ type of the file extension (dat, h5 or other).
* @param _app shall append?
*/
template<typename _T, uint _spinModes>
inline auto Hamiltonian<_T, _spinModes>::getEigVec(std::string _dir, u64 _mid, HAM_SAVE_EXT _typ, bool _app) const -> void
{
	const auto inLeft			= (this->avEnIdx - u64(_mid / 2)) >= 0 ? (this->avEnIdx - u64(_mid / 2)) : 0;
	const auto inRight			= (this->avEnIdx + u64(_mid / 2)) < this->Nh ? (this->avEnIdx + u64(_mid / 2)) : (this->Nh - 1);

	BEGIN_CATCH_HANDLER
	{
		const arma::Mat<_T> states	= (_mid == this->Nh) ? this->eigVec_ : this->eigVec_.submat(0, inLeft, this->Nh - 1, inRight);
		std::string extension		= "." + SSTR(getSTR_HAM_SAVE_EXT(_typ));
		switch (_typ)
		{
		case HAM_SAVE_EXT::dat:
			states.save(_dir + "states" + this->getInfo() + extension, arma::raw_ascii);
			break;
		case HAM_SAVE_EXT::h5:
			if (_app)
				states.save(arma::hdf5_name(_dir + this->getInfo() + extension, "states", arma::hdf5_opts::append));
			else
				states.save(arma::hdf5_name(_dir + this->getInfo() + extension, "states"));
			break;
		default:
			states.save(_dir + "states" + this->getInfo() + ".dat", arma::raw_ascii);
			break;
		}
	}
	END_CATCH_HANDLER("Exception in saving the Eigenvectors: ", getEigVec(_dir, _mid, HAM_SAVE_EXT::dat, _app););

}

// ##########################################################################################################################################

/*
* @brief Calculates the degeneracy histogram of the eigenvalues.
* @returns The degeneracy histogram of the degeneracies in the eigenspectrum.
*/
template<typename _T, uint _spinModes>
inline auto Hamiltonian<_T, _spinModes>::getDegeneracies() const -> v_2d<u64>
{
	// map of degeneracies (vector - V[degeneracy] = {indices in the manifold}
	v_2d<u64> degeneracyMap		=	v_1d<v_1d<u64>>(Ns * 10, v_1d<u64>(0));
	// placeholder for degeneracies
	v_1d<u64> degeneracyPlh		=	v_1d<u64>(0);
	u64 _iter					=	0;

	double _prevE				=	this->eigVal_[0];
	double _E					=	this->eigVal_[0];
	while (true)
	{
		int _counter			=	0;
		while (true)
		{
			// read current energy
			_E					= this->eigVal_[_iter];
			// if this energy not equals the previous energy, vreak the loop
			if (!EQP(_E, _prevE, 1e-14))
				break;
			// append the degeneracy placeholder
			degeneracyPlh.push_back(_iter);
			_counter++;
			_iter++;
		}
		// append states to current degeneracy counter in map
		for (auto& _item : degeneracyPlh)
			degeneracyMap[_counter].push_back(_item);
		degeneracyPlh.clear();

		_prevE					=	this->eigVal_[_iter];
		if (_iter >= Nh)
			break;
	}
	return degeneracyMap;
}

// ##########################################################################################################################################

#endif