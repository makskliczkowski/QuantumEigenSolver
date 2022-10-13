#pragma once
#ifndef HAMIL_H
#include "../hamil.h"
#endif // !HAMIL_H


#ifndef TESTER_H
#define TESTER_H

/*
* Model with disorder thus with no symmetries Nuclear PhysicsB966(2021)115373
*/
template <typename _type>
class TesterModel : public SpinHamiltonian<_type> {
public:
	// ------------------------------------------- 				 Constructors				  -------------------------------------------
	~TesterModel() = default;
	TesterModel() = default;
	TesterModel(std::shared_ptr<Lattice> lat);

	double J = 1.0;
	double hx = 0.2;
	double hz = 0.8;
	double Delta = 0.9;
	double eta = 0.5;


private:
	u64 map(u64 index) const override;

public:
	// METHODS
	void hamiltonian() override;
	void setHamiltonianElem(u64 k, _type value, u64 new_idx) override;											// sets the Hamiltonian elements
	const v_1d<pair<u64, _type>>& locEnergy(u64 _id, uint site) override;										// returns the local energy for VQMC purposes
	const v_1d<pair<u64, _type>>& locEnergy(const vec& _id, uint site) override;								// returns the local energy for VQMC purposes

	// ------------------------------------------- 				 Info				  -------------------------------------------

	string inf(const v_1d<string>& skip = {}, string sep = "_") const override
	{
		auto Ns = this->lattice->get_Ns();
		string name = sep + \
			"TesterModel,Ns=" + STR(Ns) + \
			",bc=" + STR(this->lattice->get_BC());

		return this->SpinHamiltonian<_type>::inf(name, skip, sep);
	}
	void update_info() override { this->info = this->inf(); };
};

// ----------------------------------------------------------------------------- CONSTRUCTORS -----------------------------------------------------------------------------

/*
* @brief Ising disorder constructor
* @param lat general lattice class that informs about the topology of the system lattice
*/
template <typename _type>
TesterModel<_type>::TesterModel(std::shared_ptr<Lattice> lat)
{
	this->lattice = lat;
	this->ran = randomGen();
	this->Ns = this->lattice->get_Ns();
	this->N = ULLPOW(this->Ns);															// Hilber space size
	this->state_val_num = 2;
	this->state_val = v_1d<std::pair<u64, _type>>(this->state_val_num);

	//change info
	this->info = this->inf();

}

// ----------------------------------------------------------------------------- BASE GENERATION AND RAPPING -----------------------------------------------------------------------------

/*
* Return the index in the case of no mapping in disorder
* index index to take
* @returns index
*/
template <typename _type>
u64 TesterModel<_type>::map(u64 index) const {
	if (index < 0 || index >= std::pow(2, this->lattice->get_Ns())) throw "Element out of range\n No such index in map\n";
	return index;
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
const v_1d<pair<u64, _type>>& TesterModel<_type>::locEnergy(u64 _id, uint site) {

	return this->state_val;
}

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
const v_1d<pair<u64, _type>>& TesterModel<_type>::locEnergy(const vec& v, uint site) {
	return this->state_val;
}

// ----------------------------------------------------------------------------- BUILDING HAMILTONIAN -----------------------------------------------------------------------------

/*
* Sets the non-diagonal elements of the Hamimltonian matrix, by acting with the operator on the k-th state
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template <typename _type>
void TesterModel<_type>::setHamiltonianElem(u64 k, _type value, u64 new_idx) {
	this->H(new_idx, k) += value;
}

/*
* Sets the non-diagonal elements of the Hamimltonian matrix, by acting with the operator on the k-th state
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template <>
inline void TesterModel<cpx>::setHamiltonianElem(u64 k, cpx value, u64 new_idx) {
	this->H(new_idx, k) += value;
}

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void TesterModel<_type>::hamiltonian() {
	this->init_ham_mat();
	this->_SPIN = 1.0;
	//this->_SPIN = 0.5;
	for (long int k = 0; k < this->N; k++) {
		for (int j = 0; j <= this->Ns - 1; j++) {

			// true - spin up, false - spin down
			double s_i = checkBit(k, this->Ns - 1 - j) ? this->_SPIN : -this->_SPIN;

			// flip with S^x_i with the transverse field
			const u64 new_idx = flip(k, this->Ns - 1 - j);
			this->H(new_idx, k) += this->hx * (1.0 - 0.5 * ((j == 0) ? 1.0 : 0.0));
			//this->H(new_idx, k) += 4.0;

			// diagonal elements setting the perpendicular field
			this->H(k, k) += this->hz * (1.0 - 0.5 * ((j == this->Ns - 1) ? 1.0 : 0.0)) * s_i;
			//this->H(k, k) += 16.0 * s_i;

			// check the Siz Si+1z
			int nei = this->lattice->get_BC() == 0 ? ((j != this->Ns - 1) ? j + 1 : 0) : (j + 1);
			if (nei < this->Ns) {
				// Ising-like spin correlation
				double s_j = checkBit(k, this->Ns - 1 - nei) ? this->_SPIN : -this->_SPIN;
				// setting the neighbors elements
				this->H(k, k) += this->Delta * s_i * s_j;
				//this->H(k, k) += 9.0 * s_i * s_j;

				const u64 flip_idx_nn = flip(new_idx, this->Ns - 1 - nei);
				// sigma x
				this->H(flip_idx_nn, k) += this->J * (1.0 - this->eta);
				//this->H(flip_idx_nn, k) += 5.0;
				// sigma y
				this->H(flip_idx_nn, k) -= this->J * (1.0 + this->eta) * (s_i) * (s_j);
				//this->H(flip_idx_nn, k) -= 15.0 * (s_i) * (s_j);
			}
			// next nearest
			int nnn = this->lattice->get_BC() == 0 ? ((j + 2) % this->Ns) : (j + 2);
			if (nnn < this->Ns) {
				double s_j = checkBit(k, this->Ns - 1 - nnn) ? this->_SPIN : -this->_SPIN;
				// setting the neighbors elements
				this->H(k, k) += this->Delta * s_i * s_j;
				//this->H(k, k) += 9.0 * s_i * s_j;

				const u64 flip_idx_nn = flip(new_idx, this->Ns - 1 - nnn);
				// sigma x
				this->H(flip_idx_nn, k) += this->J * (1.0 - this->eta);
				//this->H(flip_idx_nn, k) += 5.0;
				// sigma y
				this->H(flip_idx_nn, k) -= this->J * (1.0 + this->eta) * (s_i) * (s_j);
				//this->H(flip_idx_nn, k) -= 15.0 * (s_i) * (s_j);
			}
		}
	}
}


#endif // !ISING_H