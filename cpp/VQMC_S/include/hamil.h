#pragma once
#ifndef  BINARY_H
#include "../src/binary.h"
#endif // ! BINARY_H

#ifndef LATTICE_H
#include "lattice.h"
#endif // !LATTICE_H




template <typename _type>
class SpinHamiltonian {
public:
	std::string info;																				// information about the model
	randomGen ran;																					// consistent quick random number generator

	SpMat<_type> H;																					// the Hamiltonian
	Mat<_type> eigenvectors;																		// matrix of the eigenvectors in increasing order
	vec eigenvalues;																				// eigenvalues vector
	u64 E_av_idx = -1;																				// average energy

	u64 loc_states_num;																				// local energy states number
	u64 N;																							// the Hilbert space size
	std::mutex my_mute_button;																		// thread mutex
	std::shared_ptr<Lattice> lattice;																// contains all the information about the lattice

	v_1d<u64> mapping;																				// mapping for the reduced Hilbert space
	v_1d<cpx> normalisation;																		// used for normalization in the symmetry case
	v_1d<std::tuple<u64, _type>> locEnergies;														// local energies map

	virtual u64 map(u64 index) = 0;																	// function returning either the mapping(symmetries) or the input index (no-symmetry: 1to1 correspondance)
	//virtual ~SpinHamiltonian() = 0;																	// pure virtual destructor
	// ---------------------------------- GETTERS ----------------------------------
	
	/*
	* gets the information about the model
	* @param skip vector of elements to be skipped in the info showcase
	* @returns trimmed information about the model
	*/
	auto get_info(v_1d<std::string> skip = {}) const->std::string {
		auto tmp = split_str(this->info, ",");
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

	// ---------------------------------- PRINTERS -------------------------------------
	static void print_base_state(u64 state, _type val, v_1d<int>& base_vector, double tol);			// pretty prints the base state
	static void print_state_pretty(const Col<_type>& state, int Ns);								// pretty prints the eigenstate at a given idx
	const v_1d<std::tuple<u64, _type>>& get_localEnergyRef() const { return this->locEnergies; };	// returns the constant reference to local energy
	const v_1d<std::tuple<u64, _type>>& get_localEnergyRef(u64 _id)									// returns the constant reference to local energy
	{ 
		this->locEnergy(_id);
		return this->locEnergies; 
	};	

	void print_state(u64 _id)					const { this->eigenvectors(_id).print(); };			// prints the eigenstate at a given idx
	auto get_hilbert_size()						const RETURNS(this->N);								// get the Hilbert space size 2^N
	auto get_mapping()							const RETURNS(this->mapping);						// constant reference to the mapping
	auto get_hamiltonian()						const RETURNS(this->H);								// get the const reference to a Hamiltonian
	auto get_eigenvectors()						const RETURNS(this->eigenvectors);					// get the const reference to the eigenvectors
	auto get_eigenvalues()						const RETURNS(this->eigenvalues);					// get the const reference to eigenvalues
	auto get_eigenEnergy(u64 idx)				const RETURNS(this->eigenvalues(idx));				// get eigenenergy at a given idx
	auto get_eigenState(u64 idx)				const RETURNS(this->eigenvectors.col(idx));			// get an eigenstate at a given idx
	auto get_eigenStateValue(u64 idx, u64 elem) const RETURNS(this->eigenvectors(elem, idx));		// get an eigenstate at a given idx

	// ---------------------------------- GENERAL METHODS ----------------------------------
	virtual void hamiltonian() = 0;																	// pure virtual Hamiltonian creator
	virtual void locEnergy(u64 _id) = 0;															// returns the local energy for VQMC purposes
	virtual void setHamiltonianElem(u64 k, double value, u64 new_idx) = 0;							// sets the Hamiltonian elements in a virtual way
	void diag_h(bool withoutEigenVec = false);// diagonalize the Hamiltonian
	
	static std::string set_info(std::string name, const v_1d<std::string>& skip = {}, std::string sep = "_") {
		auto tmp = split_str(name, ",");
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
};

template<typename _type>
inline void SpinHamiltonian<_type>::print_base_state(u64 state, _type val, v_1d<int>& base_vector, double tol)
{
	std::string tmp = "";
	intToBase(state, base_vector, 2);
	if (!valueEqualsPrec(std::abs(val), 0.0, tol))
		stout << str_p(val,3) << "*|" << base_vector << "> + ";
}

template<>
inline void SpinHamiltonian<cpx>::print_base_state(u64 state, cpx val, v_1d<int>& base_vector, double tol)
{
	std::string tmp = "";
	intToBase(state, base_vector, 2);
	if (!valueEqualsPrec(std::abs(val), 0.0, tol))
		stout << print_cpx(val, 3) << "*|" << base_vector << "> + ";
}

/*
* 
*/
template<typename _type>
inline void SpinHamiltonian<_type>::print_state_pretty(const Col<_type>& state, int Ns)
{
	auto tmp = state;
	tmp.clean(0.05);
	std::vector<int> base_vector(Ns);
	for (auto k = 0; k < tmp.size(); k++)
		print_base_state(k, tmp(k), base_vector, 1e-3);
	stout << EL;
}

/*
* General procedure to diagonalize the Hamiltonian using eig_sym from the Armadillo library
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

	double E_av = trace(eigenvalues) / double(N);
	///auto i = std::ranges::min_element(std::begin(eigenvalues), std::end(eigenvalues), [=](int x, int y) {
	//	return abs(x - E_av) < abs(y - E_av);
	//	});
	//this->E_av_idx = i - std::begin(eigenvalues);
}