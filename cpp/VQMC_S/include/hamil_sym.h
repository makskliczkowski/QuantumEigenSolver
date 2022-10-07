#pragma once

#include "hamil.h"
#ifndef HAMILSYM_H
#define HAMILSYM_H



template <typename _type>
class SpinHamiltonianSym : public SpinHamiltonian<_type> {
public:
	// -------------------------------- SYMMETRY ELEMENTS --------------------------------

	v_1d<function<u64(u64, int)>> symmetry_group;																		// precalculate the symmetry exponents
	v_1d<_type>	symmetry_eigval;																						// save all of the symmetry representatives eigenvalues

	virtual void createSymmetryGroup() = 0;																				// create symmetry group elements and their corresponding eigenvalues
	pair<u64, _type> find_SEC_repr(u64 base_idx) const;																	// for a given base vector finds its corresponding symmetry representative

	// -------------------------------- GETTERS --------------------------------

	_type get_symmetry_norm(u64 base_idx) const;																		// returns the normalization of a symmetrized state

	// -------------------------------- MULTITHREADED MAPPING --------------------------------

	void generate_mapping();																							// utilizes the mapping kernel
	void mapping_kernel(u64 start, u64 stop, v_1d<u64>& map_thr, v_1d<_type>& norm_thr, int t);							// multithreaded mapping
	v_1d<u64> generate_full_map() const;																				// generates a full map without symmetries

	u64 map(u64 index) const override
	{
		if (index >= this->N) throw "OOR\n";
		return this->mapping[index];
	};

	const v_1d<pair<u64, _type>>& locEnergy(u64 _id, uint site) override { return v_1d<pair<u64, _type>>(); };			// returns the local energy for VQMC purposes
	const v_1d<pair<u64, _type>>& locEnergy(const vec& _id, uint site) override { return v_1d<pair<u64, _type>>(); };	// returns the local energy for VQMC purposes
	void setHamiltonianElem(u64 k, _type value, u64 new_idx) override;													// sets the Hamiltonian elements
public:
	Col<_type> symmetryRotation(u64 state) const;
	Mat<_type> symmetryRotationMat() const override;
	// -------------------------------- GETTERS --------------------------------

	auto get_normalization()					const RETURNS(this->normalisation);										// returns the normalization
	auto get_symmetry_group()					const RETURNS(this->symmetry_group);									// returns the symmetry group generators
	auto get_symmetry_eigval()					const RETURNS(this->symmetry_eigval);									// returns the symmetry group generators eigenvalues
	auto get_eigenStateFull(u64 idx)			const RETURNS(this->symmetryRotation(idx)) override;					// rotates the state with symmetry group

	// -------------------------------- CALCULATORS 
	pair<u64, _type> find_rep_and_sym_eigval(u64 base_idx, _type normalisation_beta);

};

template<typename _type>
inline pair<u64, _type> SpinHamiltonianSym<_type>::find_rep_and_sym_eigval(u64 base_idx, _type normalisation_beta)
{
	// found representative already
	u64 idx = binary_search(this->mapping, 0, this->N - 1, base_idx);
	if (idx < this->N)
		return std::make_pair(idx, this->normalisation[idx] / normalisation_beta);

	// need to find the representative
	auto [min, sym_eig] = this->find_SEC_repr(base_idx);
	idx = binary_search(this->mapping, 0, this->N - 1, min);
	if (idx < this->N)
		return std::make_pair(idx, this->normalisation[idx] / normalisation_beta * conj(sym_eig));
	// haven't found the representative - differen block sector
	else
		return std::make_pair(0, 0);
}

template<>
inline pair<u64, double> SpinHamiltonianSym<double>::find_rep_and_sym_eigval(u64 base_idx, double normalisation_beta)
{
	// found representative already
	u64 idx = binary_search(this->mapping, 0, this->N - 1, base_idx);
	if (idx < this->N)
		return std::make_pair(idx, this->normalisation[idx] / normalisation_beta);

	// need to find the representative
	auto [min, sym_eig] = this->find_SEC_repr(base_idx);
	idx = binary_search(this->mapping, 0, this->N - 1, min);
	if (idx < this->N)
		return std::make_pair(idx, this->normalisation[idx] / normalisation_beta * sym_eig);
	// haven't found the representative - differen block sector
	else
		return std::make_pair(0, 0);
}


/*
* finds the representative for a given base_idx in sector_alfa normalisation potentailly from other symmetry sector beta(the same is used creating the Hamiltonian with beta = alfa)
* @returns representative binary number and eigenvalue from symmetries to return to that state from base_idx
*/
//template<typename _type2>
//inline pair<u64, _type2> find_rep_and_sym_eigval(u64 base_idx, const SpinHamiltonianSym<_type2>& sector_alfa, _type2 normalisation_beta)
//{
//	// found representative already
//	u64 idx = binary_search(sector_alfa.mapping, 0, sector_alfa.N - 1, base_idx);
//	if (idx < sector_alfa.N)	return std::make_pair(idx, sector_alfa.normalisation[idx] / normalisation_beta);
//
//	// need to find the representative
//	auto [min, sym_eig] = sector_alfa.find_SEC_representative(base_idx);
//	idx = binary_search(sector_alfa.mapping, 0, sector_alfa.N - 1, min);
//	if (idx < sector_alfa.N)	return std::make_pair(idx, sector_alfa.normalisation[idx] / normalisation_beta * conj(sym_eig));
//	// haven't found the representative - differen block sector
//	else						return std::make_pair(0, 0);
//}

// ---------------------------------------------------------------- SYMMETRY ELEMENTS ----------------------------------------------------------------

/*
* @brief Find representatives of other EC generated by reflection, spin-flip and (reflection x spin-flip) symmetry
* @param @base_idx current base vector index to act on with symmetries
*/
template<typename _type>
inline pair<u64, _type> SpinHamiltonianSym<_type>::find_SEC_repr(u64 base_idx) const
{
	u64 SEC = INT64_MAX;
	int _min = INT_MAX;
	for (int l = 0; l < this->symmetry_group.size(); l++) {
		if (u64 new_idx = this->symmetry_group[l](base_idx, this->Ns); new_idx < SEC) {
			SEC = new_idx;
			_min = l;
		}
	}
	return std::make_pair(SEC, this->symmetry_eigval[_min]);
}

/*
* @brief From applying symmetry operators the function finds the normalisation for a given state
* @param base_idx current base vector index to act on with symmetries
*/
template<typename _type>
inline _type SpinHamiltonianSym<_type>::get_symmetry_norm(u64 base_idx) const
{
	_type normalisation = 0.0;
	for (int l = 0; l < this->symmetry_group.size(); l++) {
		// if we return to the same state by acting with symmetry group operators
		if (this->symmetry_group[l](base_idx, this->Ns) == base_idx)
			normalisation += this->symmetry_eigval[l];
	}
	return sqrt(normalisation);
}

template<typename _type>
inline Mat<_type> SpinHamiltonianSym<_type>::symmetryRotationMat() const
{

	u64 max_dim = ULLPOW(this->Ns);
	Mat<cpx> U(max_dim, this->N, arma::fill::zeros);

	for (long int k = 0; k < this->N; k++) {
		for (int i = 0; i < this->symmetry_group.size(); i++) {
			auto idx = this->symmetry_group[i](this->mapping[k], this->Ns);
			if (idx < max_dim) // only if exists in sector
				U(idx, k) += conj(this->symmetry_eigval[i] / (this->normalisation[k] * sqrt(double(this->symmetry_group.size()))));
		}
	}
	return U;
}

template<>
inline Mat<double> SpinHamiltonianSym<double>::symmetryRotationMat() const
{

	u64 max_dim = ULLPOW(this->Ns);
	Mat<double> U(max_dim, this->N, arma::fill::zeros);

	for (long int k = 0; k < this->N; k++) {
		for (int i = 0; i < this->symmetry_group.size(); i++) {
			auto idx = this->symmetry_group[i](this->mapping[k], this->Ns);
			if (idx < max_dim) // only if exists in sector
				U(idx, k) += (this->symmetry_eigval[i] / (this->normalisation[k] * sqrt(this->symmetry_group.size())));
		}
	}
	return U;
}

template<typename _type>
inline Col<_type> SpinHamiltonianSym<_type>::symmetryRotation(u64 state) const
{
	u64 dim = ULLPOW(this->Ns);

	Col<_type> output(dim, arma::fill::zeros);
	for (long int k = 0; k < this->N; k++) {
		for (int i = 0; i < this->symmetry_group.size(); i++) {
			auto idx = this->symmetry_group[i](this->mapping[k], this->Ns);
			if (idx < dim) // only if exists in sector
				output(idx) += conj(this->symmetry_eigval[i] / (this->normalisation[k] * sqrt(this->symmetry_group.size()))) * this->eigenvectors(k, state);
		}
	}
	return output;
}

template<>
inline Col<double> SpinHamiltonianSym<double>::symmetryRotation(u64 state) const
{
	u64 dim = ULLPOW(this->Ns);

	Col<double> output(dim, arma::fill::zeros);
	for (long int k = 0; k < this->N; k++) {
		for (int i = 0; i < this->symmetry_group.size(); i++) {
			auto idx = this->symmetry_group[i](this->mapping[k], this->Ns);
			if (idx < dim) // only if exists in sector
				output(idx) += this->symmetry_eigval[i] / (this->normalisation[k] * sqrt(this->symmetry_group.size())) * this->eigenvectors(k, state);
		}
	}
	return output;
}

// ----------------------------------------------------------------- MAPPING ----------------------------------------------------------

/*
* @brief Generates the mapping to the reduced Hilbert space
* The procedure hase been successfully optimized using multithreading:
* - each thread functions in the range [start, stop)
* @param start first index for a given thread from the original Hilbert space
* @param stop last index for a given thread from the original Hilbert space
* @param map_thr vector containing the mapping from the reduced basis to the original Hilbert space
* for a given thread, the whole mapping will be merged in the generate_mapping() procedure
* @param norm_thr vector containing the norms from the reduced basis to the original Hilbert space
*/
template<typename _type>
inline void SpinHamiltonianSym<_type>::mapping_kernel(u64 start, u64 stop, v_1d<u64>& map_thr, v_1d<_type>& norm_thr, int t)
{
	for (u64 j = start; j < stop; j++) {
		if (const auto [SEC, some_value] = find_SEC_repr(j); SEC == j) {
			// normalisation condition -- check if state in basis
			if (_type N = get_symmetry_norm(j); std::abs(N) > 1e-6) {
				map_thr.push_back(j);
				norm_thr.push_back(N);
			}
		}
	}
}

/*
* @brief  Splits the mapping onto threads, where each finds basis states in the reduced Hilbert space within a given range.
* The mapping is retrieved by concatenating the resulting maps from each thread
*/
template<typename _type>
inline void SpinHamiltonianSym<_type>::generate_mapping()
{
	u64 start = 0;
	auto stop = static_cast<u64>(pow(2, this->Ns));
	u64 two_powL = BinaryPowers[this->Ns];
#ifndef DEBUG
	int num_of_threads = this->thread_num;
#else
	int num_of_threads = 1;
#endif // !DEBUG

	if (num_of_threads == 1) {
		mapping_kernel(start, stop, this->mapping, this->normalisation, 0);
	}
	else {
		//Threaded
		v_2d<u64> map_threaded(num_of_threads);
		v_2d<_type> norm_threaded(num_of_threads);
		v_1d<std::thread> threads;

		// reserve threads
		threads.reserve(num_of_threads);
		for (int t = 0; t < num_of_threads; t++) {
			start = (u64)(two_powL / (double)num_of_threads * t);
			stop = ((t + 1) == num_of_threads ? two_powL : u64(two_powL / (double)num_of_threads * (double)(t + 1)));

			map_threaded[t] = v_1d<u64>();
			norm_threaded[t] = v_1d<_type>();
			threads.emplace_back(&SpinHamiltonianSym::mapping_kernel, this, start, stop, ref(map_threaded[t]), ref(norm_threaded[t]), t);
		}

		// join the threads together
		for (auto& t : threads)
			t.join();

		for (auto& t : map_threaded)
			this->mapping.insert(this->mapping.end(), std::make_move_iterator(t.begin()), std::make_move_iterator(t.end()));

		for (auto& t : norm_threaded)
			this->normalisation.insert(this->normalisation.end(), std::make_move_iterator(t.begin()), std::make_move_iterator(t.end()));
	}
	this->N = this->mapping.size();
}

/*
* @brief generates full mapping of the Hamiltonian
*/
template<typename _type>
inline v_1d<u64> SpinHamiltonianSym<_type>::generate_full_map() const
{
	//v_1d<u64> full_map;
	//for (u64 j = 0; j < (ULLPOW(this->Ns)); j++) {
	//	if (__builtin_popcountll(j) == this->Ns / 2.)
	//		full_map.push_back(j);
	//}
	return v_1d<u64>();
}

// ---------------------------------------------------------------- SYMMETRY ELEMENTS ----------------------------------------------------------------

/*
* @brief Sets the non-diagonal elements of the Hamimltonian matrix with symmetry sectors: therefore the matrix elements are summed over the SEC
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template<typename _type>
inline void SpinHamiltonianSym<_type>::setHamiltonianElem(u64 k, _type value, u64 new_idx)
{
	auto [idx, sym_eig] = this->find_rep_and_sym_eigval(new_idx, this->normalisation[k]);
	this->H(idx, k) += value * sym_eig;
}

#endif // !HAMILSYM_H