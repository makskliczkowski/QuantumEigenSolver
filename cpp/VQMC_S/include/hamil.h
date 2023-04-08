#pragma once
#ifndef SYMMETRIES_H
	#include "algebra/operators.h"
#endif // !SYMMETRIES_H

#ifndef HAMIL_H
#define HAMIL_H

// ######################### EXISTING MODELS ############################
enum MY_MODELS {													 // #
	ISING_M, XYZ_M													 // #
};																	 // #
BEGIN_ENUM(MY_MODELS)												 // #
{																	 // #
	DECL_ENUM_ELEMENT(ISING_M), DECL_ENUM_ELEMENT(XYZ_M)			 // #
}																	 // #
END_ENUM(MY_MODELS)                                                  // #	
// ######################################################################


const std::string DEF_INFO_SEP		= std::string("_");										// defalut separator in info about the model
#define DISORDER_EQUIV(type, param) type param		= 1;	\
									type param##0	= 0;	\
									arma::Col<type> d##param
#define PARAM_W_DISORDER(param, s)	(this->param + this->d##param(s))						// gets the value moved by the disorder strength
#define PARAMS_S_DISORDER(p, toSet)	toSet += SSTR(",") + SSTR(#p) + SSTR("=")  + STRP(this->##p, 2);	\
									toSet += SSTR(",") + SSTR(#p) + SSTR("0=") +						\
									((this->##p    != 0.0)   ? STRP(this->##p##0, 2) : "")	// gets the information about the disorder

template <typename _T>
class Hamiltonian {
protected:
	//double _SPIN = operators::_SPIN;														// spin value used in calculations (can be 1/2 for spin 1/2 but 1 is ok)
	int Ns												=									1;
	int Nh												=									1;
	Hilbert::HilbertSpace<_T> hilbertSpace;
	std::shared_ptr<Lattice> lat_;
	// ------------------------------------------- CLASS FIELDS -------------------------------------------
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

	Hamiltonian()											= default;
	Hamiltonian(const Hilbert::HilbertSpace<_T>& hilbert)	
		: hilbertSpace(hilbert), lat_(hilbert.getLattice()), Ns(lat_->get_Ns()), Nh(hilbert.getHilbertSize()) 
	{};
	Hamiltonian(Hilbert::HilbertSpace<_T>&& hilbert)		
		: hilbertSpace(std::move(hilbert)), lat_(hilbertSpace.getLattice()), Ns(lat_->get_Ns()), Nh(hilbertSpace.getHilbertSize()) 
	{};

	// virtual ~SpinHamiltonian() = 0;																								// pure virtual destructor

	// ------------------------------------------- PRINTERS -------------------------------------------
	//static Col<_type> map_to_state(std::map<u64, _type> mp, int N_hilbert);														// converts a map to arma column (VQMC)
	static void printBaseState(	std::ostream& output,	u64 _s, _T val, v_1d<int>& _tmpVec,	double _tol = 5e-2);					// pretty prints the base state
	static void prettyPrint(	std::ostream& output,	const arma::Col<_T>& state, uint Ns,double _tol = 5e-2);					// pretty prints the state
	void print(u64 _id)									const								{ this->eigVec_.col(_id).print("|"+STR(_id)+">=\n"); };
	void print()										const								{ this->H_.print("H=\n"); };

	// ------------------------------------------- INFO -------------------------------------------

	virtual std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const = 0;

	/*
	* @brief sets and gets the information about the model
	* @param skip vector of elements to be skipped in the info showcase
	* @returns trimmed information about the model
	*/
	virtual std::string info(std::string name = "", const v_1d<std::string>& skip = {}, std::string sep = "_") const {
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

	// -------------------------------------------	SETTERS	-------------------------------------------
	
	/*
	* @brief Initialize Hamiltonian matrix
	*/
	void init() {
		try {
			//  hamiltonian memory reservation
			this->H_ = arma::SpMat<_T>(this->getHilbertSize(), this->getHilbertSize());
		}
		catch (const std::bad_alloc& e) {
			stout << "Memory exceeded" << e.what() << "\n";
			exit(-1);
		}
	};

	// ------------------------------------------- GETTERS -------------------------------------------
	
	auto getEnAvIdx()									const -> u64						{ return this->avEnIdx; };								
	auto getHilbertSize()								const -> u64						{ return this->hilbertSpace.getHilbertSize(); };			
	auto getHilbertSpace()								const -> Hilbert::HilbertSpace<_T>	{ return this->hilbertSpace; };							
	auto getHamiltonian()								const -> arma::SpMat<_T>			{ return this->H_; };								
	auto getEigVec()									const -> arma::Mat<_T>				{ return this->eigVec_; };							
	auto getEigVec(u64 idx)								const -> arma::Col<_T>				{ return this->eigVec_.col(idx); };			
	auto getEigVec(u64 idx, u64 elem)					const -> _T							{ return this->eigVal_(elem, idx); };				
	auto getEigVal()									const -> arma::vec					{ return this->eigVal_; };						
	auto getEigVal(u64 idx)								const -> double						{ return this->eigVal_(idx); };	
	auto getInfo(const v_1d<std::string>& skip = {}, 
		std::string sep = DEF_INFO_SEP, int prec = 2)	const -> std::string				{ return this->info("", skip, sep); };

	// ------------------------------------------- 	SETTERS -------------------------------------------
	
	virtual void hamiltonian()							=									0;								

	void setHElem(u64 k, _T val, u64 newIdx);												// sets the Hamiltonian elements in a virtual way
	void diagH(bool woEigVec = false);														// diagonalize the Hamiltonian
	void diagHs(bool woEigVec = false);														// diagonalize the Hamiltonian sparse
	void diagH(bool woEigVec, uint k, uint subdim = 0, uint maxiter = 1000,
		double tol = 0, std::string form = "sm");											// diagonalize the Hamiltonian using Lanczos' method
	void calcAvEn();																		// calculate the average energy

	// ------------------------------------------- 	LOCAL ENERGY -------------------------------------------
	
	virtual void locEnergy(u64 _elemId, u64 _elem, uint _site)	= 0;
	virtual cpx locEnergy(u64 _id, uint site, Operators::_OP<cpx>::INP<double> f1,
											  std::function<cpx(const arma::vec&)> f2,
											  arma::vec& tmp)	= 0;						// returns the local energy for VQMC purposes
	virtual cpx locEnergy(const arma::vec& v, uint site, Operators::_OP<cpx>::INP<double> f1,
											  std::function<cpx(const arma::vec&)> f2,
											  arma::vec& tmp)	= 0;						// returns the local energy for VQMC purposes

	//void set_loc_en_elem(int i, u64 state, _type value) { this->locEnergies[i] = std::make_pair(state, value); };		// sets given element of local energies to state, value pair
	
	// ------------------------------------------- FOR OTHER TYPES --------------------------------------------
	virtual void updateInfo()							=									0;
public:
	// ------------------------------------------- CLEAR -------------------------------------------
	void clearEigVal()									{ this->eigVal_.reset(); };																			// resets the energy memory to 0
	void clearEigVec()									{ this->eigVec_.reset(); };
	void clearH()										{ this->H_.reset(); };																					// resets the hamiltonian memory to 0

	// -------------------------------------------					    OTHER 						  -------------------------------------------

	// Heisenberg-dots
	//void set_angles() {};
	//void set_angles(const vec& phis, const vec& thetas) {};
	//void set_angles(const vec& sin_phis, const vec& sin_thetas, const vec& cos_phis, const vec& cos_thetas) {};
	//void set_angles(int position, double sin_phis, double sin_thetas, double cos_phis, double cos_thetas) {};
};

// ################################################################################################################################################

template<typename _T>
inline void Hamiltonian<_T>::calcAvEn()
{
	// calculates the middle spectrum element
	double Eav = arma::mean(this->eigVal_);

	// calculates the middle spectrum element
	u64 Nh = this->getHilbertSize();
	v_1d<double> tmp(Nh, 0.0);
	for (int i = 0; i < Nh; i++)
		tmp[i] = std::abs(this->getEigVal(i) - Eav);

	// get the iterator
	v_1d<double>::iterator _it = std::min_element(std::begin(tmp), std::end(tmp));
	this->avEnIdx = _it - std::begin(tmp);
}

// ################################################################################################################################################

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

// ################################################################################################################################################

/*
* @brief Sets the non-diagonal elements of the Hamimltonian matrix with symmetry sectors: therefore the matrix elements are summed over the SEC
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template<typename _T>
inline void Hamiltonian<_T>::setHElem(u64 k, _T val, u64 newIdx)
{
	auto [idx, symEig] = this->hilbertSpace.findRep(newIdx, this->hilbertSpace.getSymNorm(k));
	// set Hamiltonian element. If map is empty, returns the same element as wanted - the symmetry is None
	try {
		this->H_(idx, k) += val * symEig;
	}
	catch (const std::exception& err) {
		stout << "EXCEPTION" << err.what() << "\n";
		printSeparated(stout, '\t', 15, true, VEQ(k), VEQ(val), VEQ(newIdx), VEQ(idx));
	}
}

// ################################################################################################################################################

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* @param withoutEigenVec doesnot compute eigenvectors to save memory potentially
*/
template <typename _T>
inline void Hamiltonian<_T>::diagH(bool woEigVec) {
	try {
		if (woEigVec)			arma::eig_sym(this->eigVal_, arma::Mat<_T>(this->H_));
		else					arma::eig_sym(this->eigVal_, this->eigVec_, arma::Mat<_T>(this->H_));
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H_.size() * sizeof(H_(0, 0)) << "bytes" << EL;
		exit(-2);
	}
	this->calcAvEn();
}

template <>
inline void Hamiltonian<cpx>::diagH(bool woEigVec) {
	try {
		if (woEigVec)			arma::eig_sym(this->eigVal_, arma::Mat<cpx>(this->H_));
		else					arma::eig_sym(this->eigVal_, this->eigVec_, arma::Mat<cpx>(this->H_));
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H_.size() * sizeof(H_(0, 0)) << "bytes" << EL;
		exit(-2);
	}
	this->calcAvEn();
}

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* @param withoutEigenVec doesnot compute eigenvectors to save memory potentially
*/
template <typename _T>
inline void Hamiltonian<_T>::diagH(bool woEigVec, uint k, uint subdim, uint maxiter, double tol, std::string form) {
	try {
		arma::eigs_opts opts;
		opts.tol				= tol;
		opts.maxiter			= maxiter;
		opts.subdim				= (subdim == 0) ? (2 * int(k) + 1) : subdim;

		stout << "\t\t\t->Using " << ((form == "la" || form == "sa" || form == "lm") ? "Lanczos" : "S&I") << EL;

		if (woEigVec)			arma::eig_sym(this->eigVal_, arma::Mat<_T>(this->H_));
		else					arma::eig_sym(this->eigVal_, this->eigVec_, arma::Mat<_T>(this->H_));
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H_.size() * sizeof(H_(0, 0)) << "bytes" << EL;
		exit(-2);
	}
	this->calcAvEn();
}

template <>
inline void Hamiltonian<double>::diagH(bool woEigVec, uint k, uint subdim, uint maxiter, double tol, std::string form)
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

template<typename _type>
inline void Hamiltonian<_type>::diagHs(bool woEigVec)
{
	try {
		if (woEigVec)			arma::eigs_sym(this->eigVal_, this->H_, this->getHilbertSize());
		else					arma::eigs_sym(this->eigVal_, this->eigVec_, this->H_, this->getHilbertSize());
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H_.size() * sizeof(H_(0, 0)) << "bytes" << EL;
		exit(-2);
	}
	this->calcAvEn();
}

// ################################################################################################################################################
#endif