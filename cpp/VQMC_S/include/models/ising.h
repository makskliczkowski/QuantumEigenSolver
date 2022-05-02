#pragma once
#ifndef HAMIL_H
#include "../hamil.h"
#endif // !HAMIL_H


#ifndef ISINGMODEL
#define ISINGMODEL

/*
/// Model with disorder thus with no symmetries
*/
template <typename _type>
class IsingModel : public SpinHamiltonian<_type> {
private:
	// MODEL BASED PARAMETERS 
	double J = 1;																								// spin exchange
	double g = 1;																								// transverse magnetic field
	double h = 1;																								// perpendicular magnetic field

	vec dh;																										// disorder in the system - deviation from a constant h value
	double w = 0;																								// the distorder strength to set dh in (-disorder_strength, disorder_strength)
	vec dJ;																										// disorder in the system - deviation from a constant J0 value
	double J0 = 0;																								// spin exchange coefficient
	vec dg;																										// disorder in the system - deviation from a constant g0 value
	double g0 = 0;																								// transverse magnetic field
public:
	/* Constructors */
	~IsingModel() = default;
	IsingModel() = default;
	IsingModel(double J, double J0, double g, double g0, double h, double w, std::shared_ptr<Lattice> lat);

private:
	u64 map(u64 index) override;

public:
	// METHODS
	void hamiltonian() override;
	void locEnergy(u64 _id) override;																			// returns the local energy for VQMC purposes
	void setHamiltonianElem(u64 k, _type value, u64 new_idx) override;											// sets the Hamiltonian elements

	std::string inf(const v_1d<std::string>& skip = {}, std::string sep = "_") const
	{
		std::string name = sep + \
			"ising,Ns=" + STR(Ns) + \
			",J=" + STRP(J, 2) + \
			",J0=" + STRP(J0, 2) + \
			",g=" + STRP(g, 2) + \
			",g0=" + STRP(g0, 2) + \
			",h=" + STRP(h, 2) + \
			",w=" + STRP(w, 2);
		return SpinHamiltonian::inf(name, skip, sep);
	}
};

// ----------------------------------------------------------------------------- CONSTRUCTORS -----------------------------------------------------------------------------

/*
* Ising disorder constructor
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
	this->loc_states_num = this->Ns + 1;												// number of states after local energy work
	this->locEnergies = v_1d<std::tuple<u64, _type>>(this->loc_states_num);				// set local energies vector
	this->N = ULLPOW(Ns);																// Hilber space size
	this->dh = create_random_vec(Ns, this->ran, this->w);								// creates random disorder vector
	this->dJ = create_random_vec(Ns, this->ran, this->J0);								// creates random exchange vector
	this->dg = create_random_vec(Ns, this->ran, this->g0);								// creates random transverse field vector

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
u64 IsingModel<_type>::map(u64 index) {
	if (index < 0 || index >= std::pow(2, this->lattice->get_Ns())) throw "Element out of range\n No such index in map\n";
	return index;
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
void IsingModel<_type>::locEnergy(u64 _id) {
	// sumup the value of a non-changed state
	double localVal = 0;
#pragma omp parallel for reduction(+ : localVal)
	for (auto i = 0; i < Ns; i++) {
		// true - spin up, false - spin down
		double si = checkBit(_id, this->Ns - i - 1) ? 1.0 : -1.0;								
		
		// diagonal elements setting the perpendicular field
		localVal += (this->h + dh(i)) * si;												
		
		// diagonal elements setting the interaction field
		if (auto nei = this->lattice->get_nn(i, 0);  nei >= 0) {
			double sj = checkBit(_id, Ns - 1 - nei) ? 1.0 : -1.0;
			localVal += (this->J + this->dJ(i)) * si * sj;								
		}

		// flip with S^x_i with the transverse field
		u64 new_idx = flip(_id, BinaryPowers[this->Ns - 1 - i], this->Ns - 1 - i);
		this->locEnergies[i] = std::make_tuple(new_idx,
			this->g + this->dg(i));
	}
	// append unchanged at the very end
	locEnergies[this->Ns] = std::make_tuple(_id, static_cast<_type>(localVal));				
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
void IsingModel<cpx>::setHamiltonianElem(u64 k, cpx value, u64 new_idx) {
	this->H(new_idx, k) += value;
}

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void IsingModel<_type>::hamiltonian() {
	try {
		this->H = SpMat<_type>(this->N, this->N);										//  hamiltonian memory reservation
	}
	catch (const std::bad_alloc& e) {
		std::cout << "Memory exceeded" << e.what() << "\n";
		assert(false);
	}

	for (long int k = 0; k < this->N; k++) {
		for (int j = 0; j <= this->Ns - 1; j++) {
			// true - spin up, false - spin down
			double s_i = checkBit(k, this->Ns - 1 - j) ? 1.0 : -1.0;							
			
			// flip with S^x_i with the transverse field
			u64 new_idx = flip(k, BinaryPowers[Ns - 1 - j], Ns - 1 - j);
			setHamiltonianElem(k, this->g + this->dg(j), new_idx);

			// diagonal elements setting the perpendicular field
			H(k, k) += (this->h + dh(j)) * s_i;											

			if (auto nn = this->lattice->get_nn(j, 0); nn >= 0) {
				// Ising-like spin correlation
				double s_j = checkBit(k, Ns - 1 - nn) ? 1.0 : -1.0;
				// setting the neighbors elements
				this->H(k, k) += (this->J + this->dJ(j)) * s_i * s_j;		
			}
		}
	}
}


#endif // !ISING_H