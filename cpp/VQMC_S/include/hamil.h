#pragma once
#include "../src/binary.h"
#include "lattice.h"



template <typename _type>
class SpinHamiltonian {
public:
	std::string info;																				// information about the model
	randomGen ran;																					// consistent quick random number generator

	SpMat<_type> H;																					// the Hamiltonian
	Mat<_type> eigenvectors;																		// matrix of the eigenvectors in increasing order
	vec eigenvalues;																				// eigenvalues vector
	u64 E_av_idx = -1;																				// average energy

	u64 N;																							// the Hilbert space size
	std::mutex my_mute_button;																		// thread mutex
	std::shared_ptr<Lattice> lattice;																// contains all the information about the lattice

	v_1d<u64> mapping;																				// mapping for the reduced Hilbert space
	v_1d<cpx> normalisation;																		// used for normalization in the symmetry case

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
	static void print_state_pretty(const Col<_type>& state, int Ns);								// pretty prints the eigenstate at a given idx
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
	virtual v_1d<std::tuple<u64, _type>> locEnergy(u64 _id) = 0;									// returns the local energy for VQMC purposes
	virtual void setHamiltonianElem(u64 k, double value, u64 new_idx) = 0;							// sets the Hamiltonian elements in a virtual way
	void diag_h(bool withoutEigenVec = false);														// diagonalize the Hamiltonian
};

/*
* 
*/
template<typename _type>
inline void SpinHamiltonian<_type>::print_state_pretty(const Col<_type>& state, int Ns)
{
	auto tmp = state;
	tmp.clean(0.05);
	std::vector<int> base_vector(Ns);
	for (auto k = 0; k < tmp.size(); k++) {
		intToBase(k, base_vector, 2);
		if (std::abs(tmp(k)) != 0) {
			stout << tmp(k) << " * |" << base_vector << "> + ";
		}
	}
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