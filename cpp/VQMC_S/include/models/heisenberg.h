#pragma once

#include "../hamil.h"


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
	vec tmp_vec;
	vec tmp_vec2;

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
	void locEnergy(const vec& _id) override;																			// returns the local energy for VQMC purposes
	void setHamiltonianElem(u64 k, _type value, u64 new_idx) override;

	virtual string inf(const v_1d<string>& skip = {}, string sep = "_") const override
	{
		string name = sep + \
			"heisenberg,Ns=" + STR(this->Ns) + \
			",J=" + STRP(this->J, 2) + \
			",J0=" + STRP(this->J0, 2) + \
			",dlt=" + STRP(this->delta, 2) + \
			",g=" + STRP(this->g, 2) + \
			",g0=" + STRP(this->g0, 2) + \
			",h=" + STRP(this->h, 2) + \
			",w=" + STRP(this->w, 2);
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
	this->Ns = this->lattice->get_Ns();																		// number of lattice sites
	this->loc_states_num = 2 * this->Ns + 1;																// number of states after local energy work
	this->locEnergies = v_1d<std::pair<u64, _type>>(this->loc_states_num, std::pair(LLONG_MAX, 0.0));		// set local energies vector
	this->N = ULLPOW(this->Ns);																				// Hilber space size
	this->dh = create_random_vec(this->Ns, this->ran, this->w);												// creates random disorder vector
	this->dJ = create_random_vec(this->Ns, this->ran, this->J0);											// creates random exchange vector
	this->dg = create_random_vec(this->Ns, this->ran, this->g0);											// creates random transverse field vector

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
		this->locEnergies[i] = std::pair{ new_idx, this->g + this->dg(i) };

		for (auto n_num = 0; n_num < nn_number; n_num++) {
			if (const auto nn = this->lattice->get_nn(i, n_num); nn >= 0) { //&& nn >= j
				double sj = checkBit(_id, this->Ns - 1 - nn) ? 1.0 : -1.0;

				auto interaction = (this->J + this->dJ(i));
				// diagonal elements setting  interaction field
				localVal += interaction * this->delta * si * sj;

				// S+S- + S-S+
				if (si * sj < 0) {
					auto new_new_idx = flip(new_idx, this->Ns - 1 - nn);
					this->locEnergies[this->Ns + i] = std::pair{ new_new_idx, 0.5 * interaction };
				}
				// change if we don't hit the energy
				else
					this->locEnergies[this->Ns + i] = std::pair{ LONG_MAX, 0 };
			}
		}
	}
	// append unchanged at the very end
	this->locEnergies[2 * this->Ns] = std::pair{ _id, static_cast<_type>(localVal) };
}

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param v base state vector
*/
template <typename _type>
void Heisenberg<_type>::locEnergy(const vec& v) {

	// sumup the value of non-changed state
	double localVal = 0;
	for (auto i = 0; i < this->Ns; i++) {
		// check all the neighbors

		auto nn_number = this->lattice->get_nn_number(i);

		// true - spin up, false - spin down
		double si = checkBitV(v, i) > 0 ? 1.0 : -1.0;

		// perpendicular field (SZ) - HEISENBERG
		localVal += (this->h + this->dh(i)) * si;

		// transverse field (SX) - HEISENBERG
		this->tmp_vec = v;
		flipV(tmp_vec, i);
		const u64 new_idx = baseToInt(tmp_vec);
		this->locEnergies[i] = std::pair{ new_idx, this->g + this->dg(i) };

		// check the correlations
		for (auto n_num = 0; n_num < nn_number; n_num++) {
			this->tmp_vec2 = this->tmp_vec;
			if (auto nn = this->lattice->get_nn(i, n_num); nn >= 0) {//&& nn >= i
				stout << VEQ(i) << ", nei=" << VEQ(nn) << EL;
				// check Sz 
				double sj = checkBitV(v, nn) > 0 ? 1.0 : -1.0;

				// --------------------- HEISENBERG 

				// diagonal elements setting  interaction field
				auto interaction = this->J + this->dJ(i);
				auto sisj = si * sj;
				localVal += interaction * this->delta * sisj;


				// S+S- + S-S+
				if (sisj < 0) {
					flipV(tmp_vec2, nn);
					auto flip_idx_nn = baseToInt(tmp_vec2);
					this->locEnergies[this->Ns + i] = std::pair{ flip_idx_nn, 0.5 * interaction };
				}
				else
					this->locEnergies[this->Ns + i] = std::pair{ LONG_MAX, 0 };
			}
		}
	}
	// append unchanged at the very end
	locEnergies[2 * this->Ns] = std::pair{ baseToInt(v), static_cast<_type>(localVal) };
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
			this->setHamiltonianElem(k, this->g + this->dg(i), new_idx);

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
						this->setHamiltonianElem(k, 0.5 * interaction, new_new_idx);
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
