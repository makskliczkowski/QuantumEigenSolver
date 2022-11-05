#pragma once
#ifndef  OPERATORS_H
#include "./operators/operators.h"
#endif // ! BINARY_H

#ifndef HAMIL_H
#define HAMIL_H


namespace ham_sym {
	struct global_sym {

		// particle number conservation
		bool su2 = false;
		double su2v = 0.0;
		bool check_su2(u64 state) const {
			auto value = __builtin_popcountll(state);
			return (this->su2 && double(value) != su2v);
		}
		void set_su2(double su2v, bool outter_condition, int Ns, double SPIN = 0.5) {
			this->su2v = su2v;
			this->su2 = (outter_condition && su2v >= 0 && su2v <= double(Ns)) ? true : false;
		}

	};
}


using namespace std;
template <typename _type>
class SpinHamiltonian {
protected:
	ham_sym::global_sym global;

	double _SPIN = operators::_SPIN;																								// spin value used in calculations (can be 1/2 for spin 1/2 but 1 is ok)
public:
	string info;																													// information about the model
	randomGen ran;																													// consistent quick random number generator

	SpMat<_type> H;																													// the Hamiltonian
	Mat<_type> eigenvectors;																										// matrix of the eigenvectors in increasing order
	vec eigenvalues;																												// eigenvalues vector
	u64 E_av_idx = -1;																												// average energy

	u64 N = 1;																														// the Hilbert space size
	u64 Ns = 1;																														// lattice sites number
	mutex my_mute_button;																											// thread mutex
	u32 thread_num = 1;																												// number of threads to be used
	shared_ptr<Lattice> lattice;																									// contains all the information about the lattice

	vec tmp_vec;																													// tmp vector for base states if the system is too big
	vec tmp_vec2;
	v_1d<u64> mapping;																												// mapping for the reduced Hilbert space
	v_1d<_type> normalisation;																										// used for normalization in the symmetry case
	uint state_val_num;																												// basic number of state_values

	virtual u64 map(u64 index) const = 0;																							// function returning either the mapping(symmetries) or the input index (no-symmetry: 1to1 correspondance)

	// virtual ~SpinHamiltonian() = 0;																								// pure virtual destructor

// ------------------------------------------- 				          PRINTERS 				          -------------------------------------------
	static Col<_type> map_to_state(std::map<u64, _type> mp, int N_hilbert);															// converts a map to arma column (VQMC)
	static void print_base_state(u64 state, _type val, v_1d<int>& base_vector, double tol);											// pretty prints the base state
	static void print_state_pretty(const Col<_type>& state, int Ns, double tol = 0.05);												// pretty prints the eigenstate at a given idx
	void print_state(u64 _id)					const { this->eigenvectors(_id).print(); };											// prints the eigenstate at a given idx

	// ------------------------------------------- 				        INFO 				          -------------------------------------------

	/*
	* @brief sets and gets the information about the model
	* @param skip vector of elements to be skipped in the info showcase
	* @returns trimmed information about the model
	*/
	virtual string inf(string name = "", const v_1d<string>& skip = {}, string sep = "_") const {
		auto tmp = (name == "") ? split_str(this->info, ",") : split_str(name, ",");
		string tmp_str = "";
		for (int i = 0; i < tmp.size(); i++) {
			bool save = true;
			for (auto& skip_param : skip)
				// skip the element if we don't want it to be included in the info
				save = !(split_str(tmp[i], "=")[0] == skip_param);
			if (save) tmp_str += tmp[i] + ",";
		}
		tmp_str.pop_back();
		return tmp_str;
	};
	virtual string inf(const v_1d<string>& skip = {}, string sep = "_", int prec = 2) const = 0;

	// -------------------------------------------				       SETTERS					      -------------------------------------------
	void init_ham_mat() {
		try {
			this->H = SpMat<_type>(this->N, this->N);										//  hamiltonian memory reservation
		}
		catch (const std::bad_alloc& e) {
			std::cout << "Memory exceeded" << e.what() << "\n";
			assert(false);
		}
	};

	// -------------------------------------------  				   GETTERS  					  -------------------------------------------

	auto get_en_av_idx()											const RETURNS(this->E_av_idx);									// return the index closest to the mean energy
	auto get_hilbert_size()											const RETURNS(this->N);											// get the Hilbert space size 2^N
	auto get_mapping()												const RETURNS(this->mapping);									// constant reference to the mapping
	auto get_global_sym()											const RETURNS(this->global);									// returns global symmetries
	auto get_hamiltonian()											const RETURNS(this->H);											// get the const reference to a Hamiltonian
	auto get_eigenvectors()											const RETURNS(this->eigenvectors);								// get the const reference to the eigenvectors
	auto get_eigenvalues()											const RETURNS(this->eigenvalues);								// get the const reference to eigenvalues
	auto get_eigenEnergy(u64 idx)									const RETURNS(this->eigenvalues(idx));							// get eigenenergy at a given idx
	auto get_eigenState(u64 idx)									const RETURNS(this->eigenvectors.col(idx));						// get an eigenstate at a given idx
	auto get_eigenStateValue(u64 idx, u64 elem)						const RETURNS(this->eigenvectors(elem, idx));					// get an eigenstate at a given idx
	virtual auto get_eigenStateFull(u64 idx)						const RETURNS(Col<_type>(eigenvectors.col(idx)));				// get an eigenstate at a given idx but in symmetries it changes
	virtual auto get_eigenStateFull(u64 idx, v_1d<u64> map)			const RETURNS(Col<_type>(eigenvectors.col(idx)));				// get an eigenstate at a given idx but in symmetries it changes
	virtual v_1d<u64> get_mapping_full()							const { return this->mapping; };								// returns the full mapping
	const Col<_type>& get_eigenStateRef(u64 idx)					const { this->eigenvectors.col(idx); };							// get the reference to the eigenstate
	auto get_info(const v_1d<string>& skip = {}, string sep = "_", int prec = 2)	const RETURNS(this->inf("", skip, sep));						// get the info about the model


	// ------------------------------------------- 				   GENERAL METHODS  				  -------------------------------------------
	virtual void hamiltonian() = 0;																									// pure virtual Hamiltonian creator
	virtual void setHamiltonianElem(u64 k, _type value, u64 new_idx) = 0;															// sets the Hamiltonian elements in a virtual way

	void diag_h(bool withoutEigenVec = false);																						// diagonalize the Hamiltonian
	void diag_hs(bool withoutEigenVec = false);																						// diagonalize the Hamiltonian sparse
	void diag_h(bool withoutEigenVec, uint k, uint subdim = 0, uint maxiter = 1000, \
		double tol = 0, std::string form = "lm");																					// diagonalize the Hamiltonian using Lanczos' method
	void diag_h(bool withoutEigenVec, int k, _type sigma);																			// diagonalize the Hamiltonian using shift and inverse

	// ------------------------------------------- 				        VQMC 				          -------------------------------------------
	virtual cpx locEnergy(u64 _id, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) = 0;		// returns the local energy for VQMC purposes
	virtual cpx locEnergy(const vec& v, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) = 0;	// returns the local energy for VQMC purposes

	void set_loc_en_elem(int i, u64 state, _type value) { this->locEnergies[i] = std::make_pair(state, value); };		// sets given element of local energies to state, value pair
	// -------------------------------------------				   FOR OTHER TYPES                    --------------------------------------------
	virtual void update_info() = 0;

	// -------------------------------------------                    OPERATORS						  --------------------------------------------

	// -------------------------------------------
public:
	// -------------------------------------------					    CLEAR 						  -------------------------------------------
	void clear_energies() { this->eigenvalues.reset(); };																			// resets the energy memory to 0
	void clear_hamiltonian() { this->H.reset(); };																					// resets the hamiltonian memory to 0

	// -------------------------------------------					    OTHER 						  -------------------------------------------

	// Heisenberg-dots
	void set_angles() {};
	void set_angles(const vec& phis, const vec& thetas) {};
	void set_angles(const vec& sin_phis, const vec& sin_thetas, const vec& cos_phis, const vec& cos_thetas) {};
	void set_angles(int position, double sin_phis, double sin_thetas, double cos_phis, double cos_thetas) {};

	virtual Mat<_type> symmetryRotationMat(const v_1d<u64>& full_map = {}) const { return Mat<_type>(); };
	virtual Mat<_type> reduced_dens_mat(u64 state, int A_size) { return Mat<_type>(); };
};

// ------------------------------------------------------------  				   PRINTERS 				    ------------------------------------------------------------

/*
* @brief converts a map to armadillo column
* @param mp map from state index to a given
*/
template<typename _type>
inline Col<_type> SpinHamiltonian<_type>::map_to_state(std::map<u64, _type> mp, int N_hilbert)
{
	Col<_type> tmp(N_hilbert, arma::fill::zeros);
	for (auto const& [state, val] : mp)
	{
		tmp(state) = val;
	}
	tmp = arma::normalise(tmp);
	return tmp;
}


/*
* @brief prints the base state in the braket notation
* @param state base state to be printed in the integer form
* @param val coefficient before the state
* @param base_vector to be used as reference vector
* @param tol tolerance of the coefficient absolute value
*/
template<typename _type>
inline void SpinHamiltonian<_type>::print_base_state(u64 state, _type val, v_1d<int>& base_vector, double tol)
{
	string tmp = "";
	intToBaseBit(state, base_vector);
	if (!valueEqualsPrec(std::abs(val), 0.0, tol)) {
		auto pm = val >= 0 ? "+" : "";
		stout << pm << str_p(val, 3) << "*|" << base_vector << +">";
	}
}

/*
* @brief prints the base state in the braket notation - overwriten for complex numbers only
* @param state base state to be printed in the integer form
* @param val coefficient before the state
* @param base_vector to be used as reference vector
* @param tol tolerance of the coefficient absolute value
*/
template<>
inline void SpinHamiltonian<cpx>::print_base_state(u64 state, cpx val, v_1d<int>& base_vector, double tol)
{
	string tmp = "";
	intToBase(state, base_vector, 2);
	if (!valueEqualsPrec(std::abs(val), 0.0, tol))
		stout << print_cpx(val, 3) << "*|" << base_vector << +">";
}

/*
* @brief goes through the whole state and prints it in numerical basis
* @param state the state to be printed
* @param Ns number of lattice sites
* @param tol tolerance of the coefficients absolute value
*/
template<typename _type>
inline void SpinHamiltonian<_type>::print_state_pretty(const Col<_type>& state, int Ns, double tol)
{
	v_1d<int> base_vector(Ns);
	for (auto k = 0; k < state.size(); k++)
		print_base_state(k, state(k), base_vector, tol);
	stout << EL;
}

// ------------------------------------------------------------  				    HAMILTONIAN  				    ------------------------------------------------------------

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* @param withoutEigenVec doesnot compute eigenvectors to save memory potentially
*/
template <typename T>
inline void SpinHamiltonian<T>::diag_h(bool withoutEigenVec) {
	try {
		if (withoutEigenVec) arma::eig_sym(this->eigenvalues, arma::Mat<T>(this->H));
		else				 arma::eig_sym(this->eigenvalues, this->eigenvectors, arma::Mat<T>(this->H));
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H.size() * sizeof(H(0, 0)) << "bytes" << EL;
		assert(false);
	}

	// calculates the middle spectrum element
	//double E_av = trace(eigenvalues) / double(N);
	double E_av = arma::mean(eigenvalues);
	// calculates the middle spectrum element
	v_1d<double> vect(this->N, 0.0);
	for (int i = 0; i < this->N; i++)
		vect[i] = abs(eigenvalues(i) - E_av);

	auto i = std::min_element(std::begin(vect), std::end(vect));
	this->E_av_idx = i - std::begin(vect);
}

template <>
inline void SpinHamiltonian<cpx>::diag_h(bool withoutEigenVec) {
	try {
		if (withoutEigenVec) arma::eig_sym(this->eigenvalues, arma::Mat<cpx>(this->H));
		else				 arma::eig_sym(this->eigenvalues, this->eigenvectors, arma::Mat<cpx>(this->H));
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H.size() * sizeof(H(0, 0)) << "bytes" << EL;
		assert(false);
	}

	// calculates the middle spectrum element
	//double E_av = trace(eigenvalues) / double(N);
	//auto i = std::min_element(std::begin(eigenvalues), std::end(eigenvalues), [=](int x, int y) {
	//	return abs(x - E_av) < abs(y - E_av);
	//	});
	double E_av = arma::mean(eigenvalues);
	//this->E_av_idx = i - std::begin(eigenvalues);
	v_1d<double> vect(this->N, 0.0);
	for (int i = 0; i < this->N; i++)
		vect[i] = abs(eigenvalues(i) - E_av);

	auto i = std::min_element(std::begin(vect), std::end(vect));
	this->E_av_idx = i - std::begin(vect);


}

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* @param withoutEigenVec doesnot compute eigenvectors to save memory potentially
*/
template <typename T>
inline void SpinHamiltonian<T>::diag_h(bool withoutEigenVec, uint k, uint subdim, uint maxiter, double tol, std::string form) {
	try {
		eigs_opts opts;
		opts.tol = tol;
		opts.maxiter = maxiter;
		opts.subdim = (subdim == 0) ? max(2 * int(k) + 1, 80) : subdim;
		stout << "\t\t\t->Using Lanczos" << EL;
		if (withoutEigenVec) arma::eigs_sym(this->eigenvalues, this->H, k, "sa", opts);
		else				 arma::eigs_sym(this->eigenvalues, this->eigenvectors, this->H, k, "sa", opts);
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H.size() * sizeof(H(0, 0)) << "bytes" << EL;
		assert(false);
	}
}

template <>
inline void SpinHamiltonian<cpx>::diag_h(bool withoutEigenVec, uint k, uint subdim, uint maxiter, double tol, std::string form) {
	try {
		eigs_opts opts;
		opts.tol = tol;
		opts.maxiter = maxiter;
		opts.subdim = (subdim == 0) ? max(2 * int(k) + 1, 80) : subdim;
		stout << "\t\t\t->Using Lanczos" << EL;
		cx_vec eigval(this->N);
		arma::eig_sym(this->eigenvalues, arma::Mat<cpx>(this->H));
		this->eigenvalues = arma::real(eigval);
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H.size() * sizeof(H(0, 0)) << "bytes" << EL;
		assert(false);
	}
}

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* @param withoutEigenVec doesnot compute eigenvectors to save memory potentially
*/
template <typename _type>
inline void SpinHamiltonian<_type>::diag_h(bool withoutEigenVec, int k, _type sigma) {
	try {
		if (withoutEigenVec) arma::eigs_sym(this->eigenvalues, this->H, sigma);
		else				 arma::eigs_sym(this->eigenvalues, this->eigenvectors, this->H, sigma);
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H.size() * sizeof(H(0, 0)) << "bytes" << EL;
		assert(false);
	}
}

template<typename _type>
inline void SpinHamiltonian<_type>::diag_hs(bool withoutEigenVec)
{
	try {
		if (withoutEigenVec) arma::eigs_sym(this->eigenvalues, this->H, this->N);
		else				 arma::eigs_sym(this->eigenvalues, this->eigenvectors, this->H, this->N);
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H.size() * sizeof(H(0, 0)) << "bytes" << EL;
		assert(false);
	}

	// calculates the middle spectrum element
	//double E_av = trace(eigenvalues) / double(N);
	//auto i = std::min_element(std::begin(eigenvalues), std::end(eigenvalues), [=](int x, int y) {
	//	return abs(x - E_av) < abs(y - E_av);
	//	});
	//this->E_av_idx = i - std::begin(eigenvalues);
}

// ------------------------------------------------------------  				     PHYSICAL QUANTITES   				    ------------------------------------------------------------

#endif