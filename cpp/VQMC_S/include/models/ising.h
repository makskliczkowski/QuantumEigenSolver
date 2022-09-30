#pragma once
#ifndef HAMIL_H
#include "../hamil.h"
#endif // !HAMIL_H


#ifndef ISINGMODEL
#define ISINGMODEL

/*
* Model with disorder thus with no symmetries
*/
template <typename _type>
class IsingModel : public SpinHamiltonian<_type> {
private:
	// MODEL BASED PARAMETERS 
	double J = 1;																								// spin exchange
	double g = 1;																								// transverse magnetic field
	double h = 1;																								// perpendicular magnetic field

	vec tmp_vec;
	vec tmp_vec2;

	vec dh;																										// disorder in the system - deviation from a constant h value
	double w = 0;																								// the distorder strength to set dh in (-disorder_strength, disorder_strength)
	vec dJ;																										// disorder in the system - deviation from a constant J0 value
	double J0 = 0;																								// spin exchange coefficient
	vec dg;																										// disorder in the system - deviation from a constant g0 value
	double g0 = 0;																								// transverse magnetic field
public:
	// ------------------------------------------- 				 Constructors				  -------------------------------------------
	~IsingModel() = default;
	IsingModel() = default;
	IsingModel(double J, double J0, double g, double g0, double h, double w, std::shared_ptr<Lattice> lat);

private:
	u64 map(u64 index) const override;

public:
	// METHODS
	void hamiltonian() override;
	void setHamiltonianElem(u64 k, _type value, u64 new_idx) override;											// sets the Hamiltonian elements
	v_1d<pair<u64, _type>> locEnergy(u64 _id, uint site) override;												// returns the local energy for VQMC purposes
	v_1d<pair<u64, _type>> locEnergy(const vec& _id, uint site) override;										// returns the local energy for VQMC purposes
	
	// ------------------------------------------- 				 Info				  -------------------------------------------

	string inf(const v_1d<string>& skip = {}, string sep = "_") const override
	{
		auto Ns = this->lattice->get_Ns();
		string name = sep + \
			"ising,Ns=" + STR(Ns) + \
			",J=" + STRP(J, 2) + \
			",J0=" + STRP(J0, 2) + \
			",g=" + STRP(g, 2) + \
			",g0=" + STRP(g0, 2) + \
			",h=" + STRP(h, 2) + \
			",w=" + STRP(w, 2) + \
			",bc=" + STR(this->lattice->get_BC());
		
		return this->SpinHamiltonian<_type>::inf(name, skip, sep);
	}
	void update_info() override { this->info = this->inf(); };
};

// ----------------------------------------------------------------------------- CONSTRUCTORS -----------------------------------------------------------------------------

/*
* @brief Ising disorder constructor
* @param J interaction between Sz's on the nearest neighbors
* @param J0 disorder at J interaction from (-J0,J0) added to J
* @param g transverse magnetic field
* @param g0 disorder at g field from (-g0, g0) added to g
* @param h perpendicular magnetic field 
* @param w disorder at h field from (-w, w) added to h
* @param lat general lattice class that informs about the topology of the system lattice
*/
template <typename _type>
IsingModel<_type>::IsingModel(double J, double J0, double g, double g0, double h, double w, std::shared_ptr<Lattice> lat)
	: J(J), g(g), h(h), w(w), J0(J0), g0(g0)
{
	this->lattice = lat;
	this->ran = randomGen();
	this->Ns = this->lattice->get_Ns();
	this->N = ULLPOW(this->Ns);															// Hilber space size
	this->dh = create_random_vec(this->Ns, this->ran, this->w);							// creates random disorder vector
	this->dJ = create_random_vec(this->Ns, this->ran, this->J0);						// creates random exchange vector
	this->dg = create_random_vec(this->Ns, this->ran, this->g0);						// creates random transverse field vector
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
u64 IsingModel<_type>::map(u64 index) const {
	if (index < 0 || index >= std::pow(2, this->lattice->get_Ns())) throw "Element out of range\n No such index in map\n";
	return index;
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
v_1d<pair<u64, _type>> IsingModel<_type>::locEnergy(u64 _id, uint site) {
	// sumup the value of non-changed state
	double localVal = 0;

	uint nn_number = this->lattice->get_nn_forward_num(site);

	// true - spin up, false - spin down
	double si = checkBit(_id, this->Ns - site - 1) ? 1.0 : -1.0;								
		
	// diagonal elements setting the perpendicular field
	localVal += (this->h + dh(site)) * si;												
		
	// check the Siz Si+1z
	for (auto nn = 0; nn < nn_number; nn++) {
		auto n_num = this->lattice->get_nn_forward_num(site, nn);
		if (auto nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			double sj = checkBit(_id, this->Ns - 1 - nei) ? 1.0 : -1.0;
			localVal += (this->J + this->dJ(site)) * si * sj;
		}
	}
	
	// flip with S^x_i with the transverse field
	u64 new_idx = flip(_id, this->Ns - 1 - site);
	this->state_val[1] = std::pair{ new_idx, this->g + this->dg(site) };
	// append unchanged at the very end
	this->state_val[0] = std::pair{ _id, static_cast<_type>(localVal) };

	return this->state_val;
}

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
v_1d<pair<u64, _type>> IsingModel<_type>::locEnergy(const vec& v, uint site) {
	double localVal = 0;

	uint nn_number = this->lattice->get_nn_forward_num(site);


	// check Sz 
	double si = checkBitV(v, site) > 0 ? 1.0 : -1.0;

	// diagonal elements setting the perpendicular field
	localVal += (this->h + dh(site)) * si;

	// check the Siz Si+1z
	for (auto nn = 0; nn < nn_number; nn++) {
		// double checking neighbors
		auto n_num = this->lattice->get_nn_forward_num(site, nn);
		if (auto nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			double sj = checkBitV(v, nei) > 0 ? 1.0 : -1.0;
			localVal += (this->J + this->dJ(site)) * si * sj;
		}
	}
	// flip with S^x_i with the transverse field
	this->tmp_vec = v;
	flipV(this->tmp_vec, site);
	const u64 new_idx = baseToInt(this->tmp_vec);
	this->state_val[1] = std::pair{ new_idx, this->g + this->dg(site) };

	// append unchanged at the very end
	this->state_val[0] = std::pair{ baseToInt(v), static_cast<_type>(localVal) };

	return state_val;
}

// ----------------------------------------------------------------------------- BUILDING HAMILTONIAN -----------------------------------------------------------------------------

/*
* Sets the non-diagonal elements of the Hamimltonian matrix, by acting with the operator on the k-th state
* @param k index of the basis state acted upon with the Hamiltonian 
* @param value value of the given matrix element to be set 
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state 
*/
template <typename _type>
void IsingModel<_type>::setHamiltonianElem(u64 k, _type value, u64 new_idx) {
	this->H(new_idx, k) += value;
}

/*
* Sets the non-diagonal elements of the Hamimltonian matrix, by acting with the operator on the k-th state
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template <>
inline void IsingModel<cpx>::setHamiltonianElem(u64 k, cpx value, u64 new_idx) {
	this->H(new_idx, k) += value;
}

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void IsingModel<_type>::hamiltonian() {
	this->init_ham_mat();

	for (long int k = 0; k < this->N; k++) {
		for (int j = 0; j <= this->Ns - 1; j++) {
			uint nn_number = this->lattice->get_nn_forward_num(j);
			// true - spin up, false - spin down
			double s_i = checkBit(k, this->Ns - 1 - j) ? 1.0 : -1.0;							
			
			// flip with S^x_i with the transverse field
			u64 new_idx = flip(k, this->Ns - 1 - j);
			setHamiltonianElem(k, this->g + this->dg(j), new_idx);

			// diagonal elements setting the perpendicular field
			this->H(k, k) += (this->h + dh(j)) * s_i;

			// check the Siz Si+1z
			for (auto nn = 0; nn < nn_number; nn++) {
				auto n_num = this->lattice->get_nn_forward_num(j, nn);
				if (auto nei = this->lattice->get_nn(j, n_num); nei >= 0) {
					// Ising-like spin correlation
					double s_j = checkBit(k, this->Ns - 1 - nei) ? 1.0 : -1.0;
					// setting the neighbors elements
					this->H(k, k) += (this->J + this->dJ(j)) * s_i * s_j;
				}
			}
		}
	}
}


#endif // !ISING_H