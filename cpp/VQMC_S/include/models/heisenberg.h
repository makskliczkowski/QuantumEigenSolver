#pragma once
#ifndef HAMIL_H
#include "../hamil.h"
#endif // !HAMIL_H


#ifndef HEISENBERGMODEL
#define HEISENBERGMODEL

template <typename _type>
class Heisenberg : public SpinHamiltonian<_type> {
protected:
	// MODEL BASED PARAMETERS 
	double J = 1;																								// spin exchange
	double g = 1;																								// transverse magnetic field
	double h = 1;																								// perpendicular magnetic field
	double delta = 0;																							// delta with Sz_i * Sz_ip1

	vec dh;																										// disorder in the system - deviation from a constant h value
	double w;																									// the distorder strength to set dh in (-disorder_strength, disorder_strength)
	vec dJ;																										// disorder in the system - deviation from a constant J0 value
	double J0;																									// spin exchange coefficient
	vec dg;																										// disorder in the system - deviation from a constant g0 value
	double g0;																									// transverse magnetic field

	u64 map(u64 index) override;
public:
	// Constructors 
	~Heisenberg() = default;
	Heisenberg() = default;
	Heisenberg(double J, double J0, double g, double g0, double h, double w, double delta, std::shared_ptr<Lattice> lat);

	// METHODS
	void hamiltonian() override;
	void locEnergy(u64 _id) override;																			// returns the local energy for VQMC purposes
	void setHamiltonianElem(u64 k, _type value, u64 new_idx) override;

	virtual string inf(const v_1d<string>& skip = {}, string sep = "_") const override
	{
		string name = sep + \
			"heisenberg,Ns=" + STR(Ns) + \
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
* @brief Heisenberg disorder constructor
* @param J interaction between Sz's on the nearest neighbors
* @param J0 disorder at J interaction from (-J0,J0) added to J
* @param g transverse magnetic field
* @param g0 disorder at g field from (-g0, g0) added to g
* @param h perpendicular magnetic field
* @param w disorder at h field from (-w, w) added to h
* @param delta J*delta stands next to Sz_iSz_ip1
* @param lat general lattice class that informs about the topology of the system lattice
*/
template <typename _type>
Heisenberg<_type>::Heisenberg(double J, double J0, double g, double g0, double h, double w, double delta, std::shared_ptr<Lattice> lat)
	: J(J), g(g), h(h), w(w), J0(J0), g0(g0), delta(delta)
{
	this->lattice = lat;
	this->ran = randomGen();
	this->Ns = this->lattice->get_Ns();																// number of lattice sites
	this->loc_states_num = 2 * this->Ns + 1;														// number of states after local energy work
	this->locEnergies = v_1d<std::tuple<u64, _type>>(this->loc_states_num, std::make_tuple(0,0));	// set local energies vector
	this->N = ULLPOW(this->Ns);																		// Hilber space size
	this->dh = create_random_vec(Ns, this->ran, this->w);											// creates random disorder vector
	this->dJ = create_random_vec(Ns, this->ran, this->J0);											// creates random exchange vector
	this->dg = create_random_vec(Ns, this->ran, this->g0);											// creates random transverse field vector

	// change info
	this->info = this->inf();

}

// ----------------------------------------------------------------------------- BASE GENERATION AND RAPPING -----------------------------------------------------------------------------

/*
* Return the index in the case of no mapping in disorder
* index index to take
* @returns index
*/
template <typename _type>
u64 Heisenberg<_type>::map(u64 index) {
	if (index < 0 || index >= std::pow(2, this->lattice->get_Ns())) throw "Element out of range\n No such index in map\n";
	return index;
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
void Heisenberg<_type>::locEnergy(u64 _id) {
	// sumup the value of non-changed state
	double localVal = 0;
#ifndef DEBUG
#pragma omp parallel for reduction(+ : localVal)
#endif // !DEBUG
	for (auto i = 0; i < this->Ns; i++) {
		// check all the neighbors
		auto nn_number = this->lattice->get_nn_number(i);

		// true - spin up, false - spin down
		double si = checkBit(_id, this->Ns - i - 1) ? 1.0 : -1.0;								

		// perpendicular field (SZ)
		localVal += (this->h + dh(i)) * si;

		// transverse field (SX)
		u64 new_idx = flip(_id, this->Ns - 1 - i);
		this->locEnergies[i] = std::make_tuple(new_idx, this->g + this->dg(i));

		for (auto n_num = 0; n_num < nn_number; n_num++) {
			if (const auto nn = this->lattice->get_nn(i, n_num); nn >= 0) { //&& nn >= j
				double sj = checkBit(_id, this->Ns - 1 - nn) ? 1.0 : -1.0;

				auto interaction = (this->J + this->dJ(i));
				// diagonal elements setting  interaction field
				localVal += interaction * this->delta * si * sj;

				// S+S- + S-S+
				if (si * sj < 0) {
					auto new_new_idx = flip(new_idx, this->Ns - 1 - nn);
					this->locEnergies[this->Ns + i] = std::make_tuple(new_new_idx, 0.5 * interaction);
				}
				// change if we don't hit the energy
				else
					this->locEnergies[this->Ns + i] = std::make_tuple(LONG_MAX, 0);
			}
		}
	}
	// append unchanged at the very end
	locEnergies[2*this->Ns] = std::make_tuple(_id, static_cast<_type>(localVal));				
}

// ----------------------------------------------------------------------------- BUILDING HAMILTONIAN -----------------------------------------------------------------------------

/*
* @brief Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void Heisenberg<_type>::hamiltonian() {
	//  hamiltonian memory reservation
	try {
		this->H = SpMat<_type>(this->N, this->N);										
	}
	catch (const std::bad_alloc& e) {
		std::cout << "Memory exceeded" << e.what() << "\n";
		assert(false);
	}

	for (auto k = 0; k < this->N; k++) {
		for (auto i = 0; i < this->Ns; i++) {
			// check all the neighbors
			auto nn_number = this->lattice->get_nn_number(i);

			// true - spin up, false - spin down
			double si = checkBit(k, this->Ns - 1 - i) ? 1.0 : -1.0;
				
			// disorder // perpendicular magnetic field
			this->H(k, k) += (this->h + dh(i)) * si;									

			// transverse field
			u64 new_idx = flip(k, this->Ns - 1 - i);			
			setHamiltonianElem(k, this->g + this->dg(i), new_idx);	

			// check if nn exists
			for (auto n_num = 0; n_num < nn_number; n_num++) {
				if (const auto nn = this->lattice->get_nn(i, n_num); nn >= 0) { //  && nn >= i
					// Ising-like spin correlation - check the bit on the nn
					double sj = checkBit(k, this->Ns - 1 - nn) ? 1.0 : -1.0;
					auto interaction = (this->J + this->dJ(i));

					// setting the neighbors elements
					this->H(k, k) += interaction * this->delta * si * sj;

					// S+S- + S-S+ hopping
					if (si * sj < 0) {
						auto new_new_idx = flip(new_idx, this->Ns - 1 - nn);
						setHamiltonianElem(k, 0.5 * interaction, new_new_idx);
					}
				}
			}
		}
	}
}

/*
* @brief Sets the non-diagonal elements of the Hamimltonian matrix, by acting with the operator on the k-th state
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template <typename _type>
void Heisenberg<_type>::setHamiltonianElem(u64 k, _type value, u64 new_idx) {
	this->H(new_idx, k) += value;
}

/*
* @brief Sets the non-diagonal elements of the Hamimltonian matrix, by acting with the operator on the k-th state - complex override
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template <>
inline void Heisenberg<cpx>::setHamiltonianElem(u64 k, cpx value, u64 new_idx) {
	this->H(new_idx, k) += value;
}


#endif // !HEISENBERG_H
