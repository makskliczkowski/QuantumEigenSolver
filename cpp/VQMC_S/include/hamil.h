#pragma once
#ifndef  OPERATORS_H
#include "./operators/operators.h"
#endif // ! BINARY_H

#ifndef LATTICE_H
#include "lattice.h"
#endif // !LATTICE_H

using namespace std;
template <typename _type>
class SpinHamiltonian {
public:
	string info;																					// information about the model
	randomGen ran;																					// consistent quick random number generator

	SpMat<_type> H;																					// the Hamiltonian
	Mat<_type> eigenvectors;																		// matrix of the eigenvectors in increasing order
	vec eigenvalues;																				// eigenvalues vector
	u64 E_av_idx = -1;																				// average energy

	u64 loc_states_num;																				// local energy states number
	u64 N;																							// the Hilbert space size
	u64 Ns;																							// lattice sites number
	mutex my_mute_button;																			// thread mutex
	shared_ptr<Lattice> lattice;																	// contains all the information about the lattice

	v_1d<u64> mapping;																				// mapping for the reduced Hilbert space
	v_1d<cpx> normalisation;																		// used for normalization in the symmetry case
	v_1d<tuple<u64, _type>> locEnergies;															// local energies map

	virtual u64 map(u64 index) = 0;																	// function returning either the mapping(symmetries) or the input index (no-symmetry: 1to1 correspondance)
	// virtual ~SpinHamiltonian() = 0;																	// pure virtual destructor
	
	// ---------------------------------- PRINTERS -------------------------------------
	static void print_base_state(u64 state, _type val, v_1d<int>& base_vector, double tol);			// pretty prints the base state
	static void print_state_pretty(const Col<_type>& state, int Ns, double tol = 0.05);				// pretty prints the eigenstate at a given idx
	void print_state(u64 _id)					const { this->eigenvectors(_id).print(); };			// prints the eigenstate at a given idx
	
	// ---------------------------------- INFO ----------------------------------------
	/*
	* @brief sets and gets the information about the model
	* @param skip vector of elements to be skipped in the info showcase
	* @returns trimmed information about the model
	*/
	virtual std::string inf(std::string name = "", const v_1d<std::string>& skip = {}, std::string sep = "_") const {
		auto tmp = (name == "") ? split_str(this->info, ",") : split_str(name, ",");
		std::string tmp_str = "";
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
	virtual std::string inf(const v_1d<std::string>& skip = {}, std::string sep = "_") const = 0;
	
	// ---------------------------------- CALCULATORS -----------------------------------------------------
	Mat<_type> red_dens_mat(const Col<_type>& state, int A_size) const;													// calculate the reduced density matrix
	Mat<_type> red_dens_mat(u64 state, int A_size) const;																// calculate the reduced density matrix based on eigenstate

	double entanglement_entropy(const Col<_type>& state, int A_size) const;												// entanglement entropy 
	double entanglement_entropy(u64 state, int A_size) const;															// entanglement entropy for eigenstate

	vec entaglement_entropy_sweep(const Col<_type>& state) const;														// entanglement entropy sweep over bonds
	vec entaglement_entropy_sweep(u64 state) const;																		// entanglement entropy sweep over bonds for eigenstate
	
	
	// ---------------------------------- GETTERS ----------------------------------
	const v_1d<std::tuple<u64, _type>>& get_localEnergyRef() const { return this->locEnergies; };						// returns the constant reference to local energy
	const v_1d<std::tuple<u64, _type>>& get_localEnergyRef(u64 _id)														// returns the constant reference to local energy
	{ 
		this->locEnergy(_id);
		return this->locEnergies; 
	};	
	auto get_hilbert_size()											const RETURNS(this->N);								// get the Hilbert space size 2^N
	auto get_mapping()												const RETURNS(this->mapping);						// constant reference to the mapping
	auto get_hamiltonian()											const RETURNS(this->H);								// get the const reference to a Hamiltonian
	auto get_eigenvectors()											const RETURNS(this->eigenvectors);					// get the const reference to the eigenvectors
	auto get_eigenvalues()											const RETURNS(this->eigenvalues);					// get the const reference to eigenvalues
	auto get_eigenEnergy(u64 idx)									const RETURNS(this->eigenvalues(idx));				// get eigenenergy at a given idx
	auto get_eigenState(u64 idx)									const RETURNS(this->eigenvectors.col(idx));			// get an eigenstate at a given idx
	auto get_eigenStateValue(u64 idx, u64 elem)						const RETURNS(this->eigenvectors(elem, idx));		// get an eigenstate at a given idx
	auto get_info(const v_1d<string>& skip = {}, string sep = "_")	const RETURNS(this->inf("", skip, sep));			// get the info about the model

	// ---------------------------------- GENERAL METHODS ----------------------------------
	virtual void hamiltonian() = 0;																						// pure virtual Hamiltonian creator
	virtual void locEnergy(u64 _id) = 0;																				// returns the local energy for VQMC purposes
	virtual void setHamiltonianElem(u64 k, _type value, u64 new_idx) = 0;												// sets the Hamiltonian elements in a virtual way
	void diag_h(bool withoutEigenVec = false);																			// diagonalize the Hamiltonian
};

// --------------------------------------------------------------------------------------- PRINTERS ---------------------------------------------------------------------------------------

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
	std::string tmp = "";
	intToBase(state, base_vector, 2);
	if (!valueEqualsPrec(std::abs(val), 0.0, tol))
		stout << str_p(val,3) << "*|" << base_vector << "> + ";
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
	std::string tmp = "";
	intToBase(state, base_vector, 2);
	if (!valueEqualsPrec(std::abs(val), 0.0, tol))
		stout << print_cpx(val, 3) << "*|" << base_vector << "> + ";
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

// --------------------------------------------------------------------------------------- HAMILTONIAN ---------------------------------------------------------------------------------------

/*
* @brief General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
* @param withoutEigenVec doesnot compute eigenvectors to save memory potentially
*/
template <typename T> 
void SpinHamiltonian<T>::diag_h(bool withoutEigenVec) {
	try {
		if (withoutEigenVec) arma::eig_sym(this->eigenvalues, arma::Mat<T>(this->H));
		else				 arma::eig_sym(this->eigenvalues, this->eigenvectors, arma::Mat<T>(this->H));
	}
	catch (const std::bad_alloc& e) {
		stout << "Memory exceeded" << e.what() << EL;
		stout << "dim(H) = " << H.size() * sizeof(H(0, 0)) << EL;
		assert(false);
	}

	// calculates the middle spectrum element
	double E_av = trace(eigenvalues) / double(N);
	auto i = std::ranges::min_element(std::begin(eigenvalues), std::end(eigenvalues), [=](int x, int y) {
		return abs(x - E_av) < abs(y - E_av);
		});
	this->E_av_idx = i - std::begin(eigenvalues);
}


// --------------------------------------------------------------------------------------- ENTROPY ---------------------------------------------------------------------------------------

/*
* @brief Calculates the reduced density matrix of the system via the mixed density matrix
* @param state state to produce the density matrix
* @param A_size size of subsystem
* @returns reduced density matrix 
*/
template<typename _type>
inline Mat<_type> SpinHamiltonian<_type>::red_dens_mat(const Col<_type>& state, int A_size) const
{
	const auto Ns = this->lattice->get_Ns();
	// set subsytsems size
	const u64 dimA = ULLPOW(A_size);
	const u64 dimB = ULLPOW(Ns - A_size);
	
	Mat<_type> rho(dimA, dimA, arma::fill::zeros);
	// loop over configurational basis
	for (auto n = 0; n < this->N; n++) {						
		u64 counter = 0;
		// pick out state with same B side (last L-A_size bits)
		for (u64 m = n % dimB; m < N; m += dimB) {	
			// find index of state with same B-side (by dividing the last bits are discarded)
			u64 idx = n / dimB;							
			rho(idx, counter) += conj(state(n)) * state(m);
			// increase counter to move along reduced basis
			counter++;										
		}
	}
	return rho;
}

/*
*  @brief Calculates the reduced density matrix of the system via the mixed density matrix
*  @param state state to produce the density matrix
*  @param A_size size of subsystem
*  @returns reduced density matrix
*/
template<typename _type>
inline Mat<_type> SpinHamiltonian<_type>::red_dens_mat(u64 state, int A_size) const
{
	const auto Ns = this->lattice->get_Ns();
	// set subsytsems size
	const u64 dimA = ULLPOW(A_size);
	const u64 dimB = ULLPOW(Ns - A_size);

	Mat<_type> rho(dimA, dimA, arma::fill::zeros);
	// loop over configurational basis
	for (auto n = 0; n < this->N; n++) {
		u64 counter = 0;
		// pick out state with same B side (last L-A_size bits)
		for (u64 m = n % dimB; m < N; m += dimB) {
			// find index of state with same B-side (by dividing the last bits are discarded)
			u64 idx = n / dimB;
			rho(idx, counter) += conj(this->get_eigenStateValue(state, n)) * this->get_eigenStateValue(state, m);
			// increase counter to move along reduced basis
			counter++;
		}
	}
	return rho;
}

/*
*  @brief Calculates the entropy of the system via the mixed density matrix
*  @param state state to produce the density matrix
*  @param A_size size of subsystem
*  @returns entropy of considered systsem
*/
template<typename _type>
inline double SpinHamiltonian<_type>::entanglement_entropy(const Col<_type>& state, int A_size) const {
	auto rho = reduced_density_matrix(state, A_size);
	vec probabilities;
	// diagonalize to find probabilities and calculate trace in rho's eigenbasis
	eig_sym(probabilities, rho); 

	double entropy = 0;
	//#pragma omp parallel for reduction(+: entropy)
	for (auto i = 0; i < probabilities.size(); i++) {
		const auto value = probabilities(i);
		entropy += (abs(value) < 1e-10) ? 0 : -value * log(abs(value));
	}
	//double entropy = -real(trace(rho * real(logmat(rho))));
	return entropy;
}

/*
*  @brief Calculates the entropy of the system via the mixed density matrix
*  @param state state index to produce the density matrix
*  @param A_size size of subsystem
*  @returns entropy of considered systsem
*/
template<typename _type>
inline double SpinHamiltonian<_type>::entanglement_entropy(u64 state, int A_size) const {
	auto rho = reduced_density_matrix(state, A_size);
	vec probabilities;
	// diagonalize to find probabilities and calculate trace in rho's eigenbasis
	eig_sym(probabilities, rho);

	double entropy = 0;
	//#pragma omp parallel for reduction(+: entropy)
	for (auto i = 0; i < probabilities.size(); i++) {
		const auto value = probabilities(i);
		entropy += (abs(value) < 1e-10) ? 0 : -value * log(abs(value));
	}
	//double entropy = -real(trace(rho * real(logmat(rho))));
	return entropy;
}

/*
* @brief Calculates the entropy of the system via the mixed density matrix
* @param state state vector to produce the density matrix
* @returns entropy of considered systsem for different subsystem sizes
*/
template<typename _type>
inline vec SpinHamiltonian<_type>::entaglement_entropy_sweep(const Col<_type>& state) const
{
	vec entropy(this->L - 1, arma::fill::zeros);
#pragma omp parallel for
	for (int i = 0; i < this->L - 1; i++)
		entropy(i) = entanglement_entropy(state, i + 1);
	return entropy;
}

/*
* @brief Calculates the entropy of the system via the mixed density matrix
* @param state state index to produce the density matrix
* @returns entropy of considered systsem for different subsystem sizes
*/
template<typename _type>
inline vec SpinHamiltonian<_type>::entaglement_entropy_sweep(u64 state) const
{
	vec entropy(this->L - 1, arma::fill::zeros);
#pragma omp parallel for
	for (int i = 0; i < this->L - 1; i++)
		entropy(i) = entanglement_entropy(state, i + 1);
	return entropy;
}