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

	u64 map(u64 index) const override;
public:
	// Constructors 
	~Heisenberg() = default;
	Heisenberg() = default;
	Heisenberg(double J, double J0, double g, double g0, double h, double w, double delta, std::shared_ptr<Lattice> lat);

	// METHODS
	virtual void hamiltonian() override;
	virtual cpx locEnergy(u64 _id, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) override;								// returns the local energy for VQMC purposes
	virtual cpx locEnergy(const vec& _id, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) override;						// returns the local energy for VQMC purposes
	void setHamiltonianElem(u64 k, _type value, u64 new_idx) override;

	virtual string inf(const v_1d<string>& skip = {}, string sep = "_", int prec = 2) const override
	{
		string name = sep + \
			"heisenberg,Ns=" + STR(this->Ns) + \
			",J=" + STRP(this->J, prec) + \
			",J0=" + STRP(this->J0, prec) + \
			",dlt=" + STRP(this->delta, prec) + \
			",g=" + STRP(this->g, prec) + \
			",g0=" + STRP(this->g0, prec) + \
			",h=" + STRP(this->h, prec) + \
			",w=" + STRP(this->w, prec);
		return this->SpinHamiltonian<_type>::inf(name, skip, sep);
	}
	virtual void update_info() override { this->info = this->inf(); };
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
u64 Heisenberg<_type>::map(u64 index) const {
	if (index < 0 || index >= this->get_hilbert_size()) throw "Element out of range\n No such index in map\n";
	return index;
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
inline cpx Heisenberg<_type>::locEnergy(u64 _id, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) {
	// sumup the value of non-changed state
	double localVal = 0;
	cpx changedVal = 0.0;

	const uint nn_number = this->lattice->get_nn_forward_num(site);

	// true - spin up, false - spin down
	const double si = checkBit(_id, this->Ns - site - 1) ? this->_SPIN : -this->_SPIN;

	// perpendicular field (SZ)
	localVal += (this->h + dh(site)) * si;

	// transverse field (SX)
	changedVal += f1(site, si) * this->_SPIN * (this->g + this->dg(site));


	// check the Siz Si+1z
	for (auto nn = 0; nn < nn_number; nn++) {
		// double checking neighbors
		const uint n_num = this->lattice->get_nn_forward_num(site, nn);
		if (int nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			// check Sz 
			const double sj = checkBit(_id, this->Ns - 1 - nei) ? this->_SPIN : -this->_SPIN;

			const double interaction = this->J + this->dJ(site);
			// diagonal elements setting  interaction field
			localVal += interaction * this->delta * si * sj;

			// S+S- + S-S+
			if (si * sj < 0) {
				const u64 flip_idx_nn = flip(flip(_id, this->Ns - 1 - nei), this->Ns - 1 - site);
				INT_TO_BASE_BIT(flip_idx_nn, tmp);
				changedVal += 0.5 * interaction * f2(tmp);
			}
		}
	}
	return changedVal + localVal;
}

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param v base state vector
*/
template <typename _type>
inline cpx Heisenberg<_type>::locEnergy(const vec& v, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) {
	double localVal = 0;
	cpx changedVal = 0.0;

	const uint nn_number = this->lattice->get_nn_forward_num(site);

	// true - spin up, false - spin down
	const double si = checkBitV(v, site) > 0 ? this->_SPIN : -this->_SPIN;

	// perpendicular field (SZ) - HEISENBERG
	localVal += (this->h + this->dh(site)) * si;

	// transverse field (SX) - HEISENBERG
	changedVal += f1(site, si) * this->_SPIN * (this->g + this->dg(site));
	tmp = v;
	flipV(tmp, site);

	// check the Siz Si+1z
	for (auto nn = 0; nn < nn_number; nn++) {
		// double checking neighbors
		const uint n_num = this->lattice->get_nn_forward_num(site, nn);
		if (int nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			// check Sz 
			const double sj = checkBitV(v, nei) > 0 ? this->_SPIN : -this->_SPIN;

			// --------------------- HEISENBERG 

			// diagonal elements setting  interaction field
			const auto interaction = this->J + this->dJ(site);
			const auto sisj = si * sj;
			localVal += interaction * this->delta * sisj;

			// S+S- + S-S+
			if (sisj < 0) {
				flipV(tmp, nei);
				changedVal += 0.5 * interaction * f2(tmp);
				flipV(tmp, nei);
			}
		}
	}
	return changedVal + localVal;
}
// ----------------------------------------------------------------------------- BUILDING HAMILTONIAN -----------------------------------------------------------------------------

/*
* @brief Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void Heisenberg<_type>::hamiltonian() {
	//  hamiltonian memory reservation
	this->init_ham_mat();

	for (auto k = 0; k < this->N; k++) {
		for (auto i = 0; i < this->Ns; i++) {
			// check all the neighbors
			v_1d<uint> nn_number = this->lattice->get_nn_forward_number(i);

			// true - spin up, false - spin down
			double si = checkBit(k, this->Ns - 1 - i) ? this->_SPIN : -this->_SPIN;

			// disorder // perpendicular magnetic field
			this->H(k, k) += (this->h + dh(i)) * si;

			// transverse field
			u64 new_idx = flip(k, this->Ns - 1 - i);
			this->setHamiltonianElem(k, this->_SPIN * (this->g + this->dg(i)), new_idx);

			// check if nn exists
			for (auto n_num : nn_number) {
				if (const auto nn = this->lattice->get_nn(i, n_num); nn >= 0) {
					// Ising-like spin correlation - check the bit on the nn
					double sj = checkBit(k, this->Ns - 1 - nn) ? this->_SPIN : -this->_SPIN;
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
