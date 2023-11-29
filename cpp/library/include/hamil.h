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

// ######################### EXISTING MODELS ############################
enum MY_MODELS {													 // #
	ISING_M, XYZ_M, NONE 											 // #
};																	 // #
BEGIN_ENUM(MY_MODELS)												 // #
{																	 // #
	DECL_ENUM_ELEMENT(ISING_M), DECL_ENUM_ELEMENT(XYZ_M),			 // #
	DECL_ENUM_ELEMENT(NONE)											 // #
}																	 // #
END_ENUM(MY_MODELS)                                                  // #	
// ######################################################################

// ######################### SAVING EXTENSIONS ##########################
enum HAM_SAVE_EXT {													 // #
	dat, h5															 // #
};																	 // #
BEGIN_ENUM(HAM_SAVE_EXT)											 // #
{																	 // #
	DECL_ENUM_ELEMENT(dat), DECL_ENUM_ELEMENT(h5)					 // #
}																	 // #
END_ENUM(HAM_SAVE_EXT)                                               // #	
// ######################################################################

const std::string DEF_INFO_SEP		= std::string("_");										// defalut separator in info about the model
#define DISORDER_EQUIV(type, param) type param		= 1;	\
									type param##0	= 0;	\
									arma::Col<type> d##param
#define PARAM_W_DISORDER(param, s)	(this->param + this->d##param(s))						// gets the value moved by the disorder strength
#define PARAMS_S_DISORDER(p, toSet)	toSet += SSTR(",") + SSTR(#p) + SSTR("=")  + STRP(this->p, 3);	\
									toSet += ((this->p##0 == 0.0) ? "" : SSTR(",") + SSTR(#p) + SSTR("0=") + STRP(this->p##0, 3))
										// gets the information about the disorder

template <typename _T>
class Hamiltonian {
public:
	using NQSFun										=									std::function<cpx(std::initializer_list<int>, std::initializer_list<double>)>;
	Hilbert::HilbertSpace<_T> hilbertSpace;

protected:
	// ------------------------------------------- CLASS TYPES ----------------------------------------------
	MY_MODELS type_										=									MY_MODELS::NONE;
	uint Ns												=									1;
	u64 Nh												=									1;
	std::shared_ptr<Lattice> lat_;

	// ------------------------------------------- CLASS FIELDS ---------------------------------------------
	double avEn											=									0.0;
	u64 avEnIdx											=									-1;														
	
	arma::SpMat<_T> H_;																		// the Hamiltonian
	arma::Mat<_T> eigVec_;																	// matrix of the eigenvectors in increasing order
	arma::vec eigVal_;																		// eigenvalues vector
public:
	randomGen ran_;																			// consistent quick random number generator
	std::string info_;																		// information about the model

	//vec tmp_vec;																			// tmp vector for base states if the system is too big
	//vec tmp_vec2;
	//uint state_val_num;																	// basic number of state_values
	virtual ~Hamiltonian() {
		LOGINFO("Base Hamiltonian destructor called.", LOG_TYPES::INFO, 3);
		this->H_.reset();
		this->eigVal_.reset();
		this->eigVec_.reset();
	}
	Hamiltonian() : ran_(randomGen(time(0))) {};
	Hamiltonian(const Hilbert::HilbertSpace<_T>& hilbert)	
		: hilbertSpace(hilbert)
	{
		this->ran_	= randomGen(time(0));
		this->lat_	= this->hilbertSpace.getLattice();
		this->Ns	= this->lat_->get_Ns();
		this->Nh	= this->hilbertSpace.getHilbertSize();
	};
	Hamiltonian(Hilbert::HilbertSpace<_T>&& hilbert)		
		: hilbertSpace(std::move(hilbert))
	{
		this->ran_	=	randomGen(time(0));
		this->lat_	=	this->hilbertSpace.getLattice();
		this->Ns	=	this->lat_->get_Ns();
		this->Nh	=	this->hilbertSpace.getHilbertSize();
	};

	// virtual ~SpinHamiltonian() = 0;																								// pure virtual destructor

	// ------------------------------------------- PRINTERS ---------------------------------------------------
	static void printBaseState(	std::ostream& output,	u64 _s, _T val, v_1d<int>& _tmpVec,	double _tol = 5e-2);					// pretty prints the base state
	static void prettyPrint(	std::ostream& output,	const arma::Col<_T>& state, uint Ns,double _tol = 5e-2);					// pretty prints the state
	void print(u64 _id)									const								{ this->eigVec_.col(_id).print("|"+STR(_id)+">=\n"); };
	void print()										const								{ this->H_.print("H=\n"); };

	// --------------------------------------------- INFO -----------------------------------------------------

	virtual std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const = 0;

	/*
	* @brief sets and gets the information about the model
	* @param skip vector of elements to be skipped in the info showcase
	* @returns trimmed information about the model
	*/
	std::string info(std::string name = "", const v_1d<std::string>& skip = {}, std::string sep = "_") const {
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

	// --------------------------------------------- INITS ----------------------------------------------------
	
	/*
	* @brief Initialize Hamiltonian matrix
	*/
	void init() {
		// hamiltonian memory reservation
		BEGIN_CATCH_HANDLER
			this->H_ = arma::SpMat<_T>(this->Nh, this->Nh);
		END_CATCH_HANDLER("Memory exceeded", ;);
	};

	// -------------------------------------------- GETTERS ---------------------------------------------------
	
	auto getDegeneracies()								const -> v_2d<u64>;
	auto getEnAvIdx()									const -> u64						{ return this->avEnIdx;													};								
	auto getHilbertSize()								const -> u64						{ return this->Nh;														};			
	auto getHilbertSpace()								const -> Hilbert::HilbertSpace<_T>	{ return this->hilbertSpace;											};							
	auto getHamiltonian()								const -> arma::SpMat<_T>			{ return this->H_;														};								
	auto getHamiltonianSize()							const -> double						{ return this->H_.size() * sizeof(this->H_(0, 0));						};								
	auto getHamiltonianSizeH()							const -> double						{ return std::pow(this->hilbertSpace.getHilbertSize(), 2) * sizeof(_T); };
	auto getSymRot()									const -> arma::SpMat<_T>			{ return this->hilbertSpace.getSymRot();								};
	auto getEigVec()									const -> arma::Mat<_T>				{ return this->eigVec_;													};							
	auto getEigVec(u64 idx)								const -> arma::Col<_T>				{ return this->eigVec_.col(idx);										};			
	auto getEigVec(u64 idx, u64 elem)					const -> _T							{ return this->eigVal_(elem, idx);										};				
	auto getEigVec(std::string _dir, u64 _mid, 
		HAM_SAVE_EXT _typ, bool _app = false)			const -> void;
	auto getEigVal()									const -> arma::vec					{ return this->eigVal_;													};						
	auto getEigVal(std::string _dir,
				HAM_SAVE_EXT _typ, bool _app = false)	const -> void;
	auto getEigVal(u64 idx)								const -> double						{ return this->eigVal_(idx);											};	
	auto getInfo(const strVec& skip = {},
		std::string sep = DEF_INFO_SEP, int prec = 2)	const -> std::string				{ return this->info_;													};
	auto getType()										const -> std::string				{ return SSTR(getSTR_MY_MODELS(this->type_));							};
	auto getTypeI()										const -> uint						{ return this->type_;													};
	auto getLat()										const -> std::shared_ptr<Lattice>	{ return this->lat_;													};
	auto getNs()										const -> uint						{ return this->lat_->get_Ns();											};
	auto getBC()										const -> BoundaryConditions			{ return this->lat_->get_BC();											};
	// ------------------------------------------- SETTERS ----------------------------------------------------
	
	virtual void hamiltonian()							=									0;								

	void setHElem(u64 k, _T val, u64 newIdx);												// sets the Hamiltonian elements in a virtual way
	void diagH(bool woEigVec = false);														// diagonalize the Hamiltonian
	void diagHs(bool woEigVec = false);														// diagonalize the Hamiltonian sparse
	void diagH(bool woEigVec, uint k, uint subdim = 0, uint maxiter = 1000,
		double tol = 0, std::string form = "sm");											// diagonalize the Hamiltonian using Lanczos' method
	void calcAvEn();																		// calculate the average energy

public:
	// ------------------------------------------ LOCAL ENERGY -------------------------------------------------
	virtual void locEnergy(u64 _elemId, 
						   u64 _elem, uint _site)		=									0; 
	virtual cpx locEnergy(u64 _id, uint s, NQSFun f1)	= 0;								// returns the local energy for VQMC purposes
	virtual cpx locEnergy(const arma::Col<double>& v, 
						  uint site,
						  NQSFun f1,
						  arma::Col<double>& tmp)		= 0;								// returns the local energy for VQMC purposes
	
	// ----------------------------------------- FOR OTHER TYPES -----------------------------------------------
	virtual void updateInfo()							=									0;
public:
	void generateFullMap()								{ this->hilbertSpace.generateFullMap(); };

	// --------------------------------------------- CLEAR -----------------------------------------------------
	void clearEigVec()									{ this->eigVec_.reset();				}; // resets the eigenvectors memory to 0
	void clearEigVal()									{ this->eigVal_.reset();				}; // resets the energy memory to 0
	void clearH()										{ this->H_.reset();						}; // resets the hamiltonian memory to 0

	// --------------------------------------------- OTHER -----------------------------------------------------

	// --------------------------------------- T R A N S F O R M -----------------------------------------------

	// Heisenberg-dots
	//void set_angles() {};
	//void set_angles(const vec& phis, const vec& thetas) {};
	//void set_angles(const vec& sin_phis, const vec& sin_thetas, const vec& cos_phis, const vec& cos_thetas) {};
	//void set_angles(int position, double sin_phis, double sin_thetas, double cos_phis, double cos_thetas) {};
};

// ##########################################################################################################################################

/*
* @brief Calculates the index closest to the average energy in the Hamiltonian
*/
template<typename _T>
inline void Hamiltonian<_T>::calcAvEn()
{
	// calculates the middle spectrum element
	double Eav	= arma::mean(this->eigVal_);
	u64 Nh		= this->getHilbertSize();

	// calculates the energy difference to find the closest element (the smallest)
	v_1d<double> tmp(Nh, 0.0);
	for (int i = 0; i < Nh; i++)
		tmp[i] = std::abs(this->getEigVal(i) - Eav);

	// get the iterator
	v_1d<double>::iterator _it	= std::min_element(std::begin(tmp), std::end(tmp));
	this->avEnIdx				= _it - std::begin(tmp);
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################ P R I N T I N G #############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief prints the base state in the BRAKET notation
* @param _s base state to be printed in the integer form
* @param _val coefficient before the state
* @param _tmpVec to be used as reference vector
* @param _tol tolerance of the coefficient absolute value
*/
template<typename _T>
inline void Hamiltonian<_T>::printBaseState(std::ostream& output, u64 _s, _T val, v_1d<int>& _tmpVec, double _tol)
{
	INT_TO_BASE(_s, _tmpVec);
	if (!EQP(std::abs(val), 0.0, _tol)) 
	{
		auto pm				=			(val >= 0) ? "+" : "";
		output				<< pm << val << "*|" << _tmpVec << +">";
	}
}

/*
* @brief goes through the whole state and prints it in numerical basis
* @param state the state to be printed
* @param Ns number of lattice sites
* @param tol tolerance of the coefficients absolute value
*/
template<typename _T>
inline void Hamiltonian<_T>::prettyPrint(std::ostream& output, const arma::Col<_T>& state, uint Ns, double tol)
{
	v_1d<int> tmpVec(Ns);
	for (u64 k = 0; k < state.size(); k++)
		printBaseState(output, k, state(k), tmpVec, tol);
	output << EL;
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ######################################################### H A M I L T O N I A N ##########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Sets the non-diagonal elements of the Hamimltonian matrix with symmetry sectors: therefore the matrix elements are summed over the SEC
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template<typename _T>
inline void Hamiltonian<_T>::setHElem(u64 k, _T val, u64 newIdx)
{
	u64 kMap = this->hilbertSpace.getMapping(k);
	BEGIN_CATCH_HANDLER
		if (kMap != newIdx)
		{
			auto [idx, symEig] = this->hilbertSpace.findRep(newIdx, this->hilbertSpace.getNorm(k));
			// set Hamiltonian element. If map is empty, returns the same element as wanted - the symmetry is None
			this->H_(idx, k) += val * symEig;
		}
		else
			this->H_(k, k) += val;
	END_CATCH_HANDLER("Exception in setting the Hamiltonian elements: " + VEQ(k) + "," + VEQ(kMap) + "," + VEQ(newIdx), exit(-1););
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ######################################################### D I A G O N A L I Z E ##########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* @param withoutEigenVec doesnot compute eigenvectors to save memory potentially
*/
template <typename _T>
inline void Hamiltonian<_T>::diagH(bool woEigVec) 
{
	if (woEigVec)	Diagonalizer<_T>::diagS(this->eigVal_, this->H_);
	else			Diagonalizer<_T>::diagS(this->eigVal_, this->eigVec_, this->H_);
	this->calcAvEn();
}

// ##########################################################################################################################################

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* @param withoutEigenVec doesnot compute eigenvectors to save memory potentially
*/
template <typename _T>
inline void Hamiltonian<_T>::diagH(bool woEigVec, uint k, uint subdim, uint maxiter, double tol, std::string form) {
	BEGIN_CATCH_HANDLER
		arma::eigs_opts opts;
		opts.tol				= tol;
		opts.maxiter			= maxiter;
		opts.subdim				= (subdim == 0) ? (2 * int(k) + 1) : subdim;
		
		LOGINFO("Diagonalizing Hamiltonian. Using: " + SSTR((form == "la" || form == "sa" || form == "lm") ? "Lanczos" : "S&I"), 
				LOG_TYPES::INFO, 3);
		if (form == "sg")
		{
			if (woEigVec)			arma::eigs_sym(this->eigVal_, this->H_, arma::uword(k), 0.0, opts);
			else					arma::eigs_sym(this->eigVal_, this->eigVec_, this->H_, arma::uword(k), 0.0, opts);
		}
		else
		{
			if (woEigVec)			arma::eigs_sym(this->eigVal_, this->H_, arma::uword(k), form.c_str());
			else					arma::eigs_sym(this->eigVal_, this->eigVec_, this->H_, arma::uword(k), form.c_str());
		}
		END_CATCH_HANDLER("Memory exceeded. DIM(H)=" + STR(this->H_.size() * sizeof(this->H_(0, 0))) + " bytes", ;);
}

template <>
inline void Hamiltonian<cpx>::diagH(bool woEigVec, uint k, uint subdim, uint maxiter, double tol, std::string form)
{
	//try {
	//	eigs_opts opts;
	//	opts.tol = tol;
	//	opts.maxiter = maxiter;
	//	opts.subdim = (subdim == 0) ? (2 * int(k) + 1) : subdim;

	//	stout << "\t\t\t->Using " << ((form == "la" || form == "sa" || form == "lm") ? "Lanczos" : "S&I") << EL;

	//	//if (form == "sg")
	//	//{
	//	//	stout << "\t\t\t->Using sigma." << EL;
	//	//	if (withoutEigenVec) arma::eigs_sym(this->eigenvalues, this->H, uword(k), 0.0, opts);
	//	//	else				 arma::eigs_sym(this->eigenvalues, this->eigenvectors, this->H, uword(k), 0.0, opts);
	//	//}
	//	//else
	//	//{
	//	stout << "\t\t\t->Using standard." << EL;
	//	if (withoutEigenVec) arma::eigs_sym(this->eigenvalues, this->H, uword(k), form.c_str());
	//	else				 arma::eigs_sym(this->eigenvalues, this->eigenvectors, this->H, uword(k), form.c_str());
	//	//}
	//}
	//catch (const std::bad_alloc& e) {
	//	stout << "Memory exceeded" << e.what() << EL;
	//	stout << "dim(H) = " << H.size() * sizeof(H(0, 0)) << "bytes" << EL;
	//	assert(false);
	//}
	//E_av_idx = int(k / 2.0);
}

// ##########################################################################################################################################

template<typename _type>
inline void Hamiltonian<_type>::diagHs(bool woEigVec)
{
	BEGIN_CATCH_HANDLER
		if (woEigVec)			arma::eigs_sym(this->eigVal_, this->H_, this->getHilbertSize());
		else					arma::eigs_sym(this->eigVal_, this->eigVec_, this->H_, this->getHilbertSize());
	END_CATCH_HANDLER("Memory exceeded. DIM(H)=" + STR(this->H_.size() * sizeof(this->H_(0, 0))) + " bytes", ;);
	this->calcAvEn();
}

// ##########################################################################################################################################
// ##########################################################################################################################################
// ############################################################# G E T T E R S ##############################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Prints the eigenvalues into some file "energies" in some directory
* @param _dir directory to be saved onto
* @param _typ type of the file extension (dat, h5 or other).
* @param _app shall append?
*/
template<typename _T>
inline auto Hamiltonian<_T>::getEigVal(std::string _dir, HAM_SAVE_EXT _typ, bool _app) const -> void
{
	std::string extension = "." + SSTR(getSTR_HAM_SAVE_EXT(_typ));
	std::ofstream file;
	BEGIN_CATCH_HANDLER
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
template<typename _T>
inline auto Hamiltonian<_T>::getEigVec(std::string _dir, u64 _mid, HAM_SAVE_EXT _typ, bool _app) const -> void
{
	const auto inLeft			= (this->avEnIdx - u64(_mid / 2)) >= 0 ? (this->avEnIdx - u64(_mid / 2)) : 0;
	const auto inRight			= (this->avEnIdx + u64(_mid / 2)) < this->Nh ? (this->avEnIdx + u64(_mid / 2)) : (this->Nh - 1);

	BEGIN_CATCH_HANDLER
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
	END_CATCH_HANDLER("Exception in saving the Eigenvectors: ", getEigVec(_dir, _mid, HAM_SAVE_EXT::dat, _app););

}

// ##########################################################################################################################################

/*
* @brief Calculates the degeneracy histogram of the eigenvalues.
* @returns The degeneracy histogram of the degeneracies in the eigenspectrum.
*/
template<typename _T>
inline auto Hamiltonian<_T>::getDegeneracies() const -> v_2d<u64>
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

#endif