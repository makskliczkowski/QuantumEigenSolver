#pragma once
#ifndef HAMIL_H
#include "../hamil.h"
#endif // !HAMIL_H


#ifndef HEISENBERGMODEL
#define HEISENBERGMODEL

/*
/// Model with disorder thus with no symmetries
*/
template <typename _type>
class Heisenberg : public SpinHamiltonian<_type> {
private:
	// MODEL BASED PARAMETERS 
	double J = 1;																								// spin exchange
	double g = 1;																								// transverse magnetic field
	double h = 1;																								// perpendicular magnetic field

	vec dh;																										// disorder in the system - deviation from a constant h value
	double w;																									// the distorder strength to set dh in (-disorder_strength, disorder_strength)
	vec dJ;																										// disorder in the system - deviation from a constant J0 value
	double J0;																									// spin exchange coefficient
	vec dg;																										// disorder in the system - deviation from a constant g0 value
	double g0;																									// transverse magnetic field
public:
	// Constructors 
	~Heisenberg() = default;
	Heisenberg() = default;
	Heisenberg(double J, double J0, double g, double g0, double h, double w, std::shared_ptr<Lattice> lat);

private:
	u64 map(u64 index) override;

public:
	// METHODS
	void hamiltonian() override;
	void locEnergy(u64 _id) override;																			// returns the local energy for VQMC purposes
	void setHamiltonianElem(u64 k, double value, u64 new_idx) override;

	static std::string set_info(int Ns, double J, double J0, double g, double g0, double h, double w,
		const v_1d<std::string>& skip = {}, std::string sep = "_")
	{
		std::string name = sep + \
			"heisenberg,Ns=" + STR(Ns) + \
			",J=" + STRP(J, 2) + \
			",J0=" + STRP(J0, 2) + \
			",g=" + STRP(g, 2) + \
			",g0=" + STRP(g0, 2) + \
			",h=" + STRP(h, 2) + \
			",w=" + STRP(w, 2);
		return SpinHamiltonian::set_info(name, skip, sep);
	}
};

// ----------------------------------------------------------------------------- CONSTRUCTORS -----------------------------------------------------------------------------

/*
* Ising disorder constructor
*/
template <typename _type>
Heisenberg<_type>::Heisenberg(double J, double J0, double g, double g0, double h, double w, std::shared_ptr<Lattice> lat)
	: J(J), g(g), h(h), w(w), J0(J0), g0(g0)

{
	this->lattice = lat;
	auto Ns = this->lattice->get_Ns();

	this->ran = randomGen();
	//this->ran.seed(7);
	this->locEnergies = v_1d<std::tuple<u64, _type>>(2*Ns + 1);							// set local energies vector
	this->N = ULLPOW(Ns);
	this->dh = create_random_vec(Ns, this->ran, this->w);								// creates random disorder vector
	this->dJ = create_random_vec(Ns, this->ran, this->J0);								// creates random exchange vector
	this->dg = create_random_vec(Ns, this->ran, this->g0);								// creates random transverse field vector

	//change info
	this->info = this->set_info(Ns, J, J0, g, g0, h, w);

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
*/
template <typename _type>
void Heisenberg<_type>::locEnergy(u64 _id) {
	auto Ns = this->lattice->get_Ns();

	// sumup the value of non-changed state
	double localVal = 0;
#pragma omp parallel for reduction(+ : localVal)
	for (auto i = 0; i < Ns; i++) {
		double si = checkBit(_id, Ns - i - 1) ? 1.0 : -1.0;								// true - spin up, false - spin down

		// perpendicular field
		localVal += (this->h + dh(i)) * si;

		// transverse field
		u64 new_idx = flip(_id, BinaryPowers[Ns - 1 - i], Ns - 1 - i);
		this->locEnergies[i] = std::make_tuple(new_idx, this->g + this->dg(i));

		// check the Siz Si+1z
		if (auto nei = this->lattice->get_nn(i, 0); nei >= 0) {
			double sj = checkBit(_id, Ns - 1 - nei) ? 1.0 : -1.0;

			// diagonal elements setting  interaction field
			localVal += (this->J + this->dJ(i)) * si * sj;		

			// S+S- + S-S+
			if (si * sj < 0)
				this->locEnergies[Ns + i] = std::make_tuple(flip(new_idx, BinaryPowers[Ns - 1 - nei], Ns - 1 - nei),
					0.5 * (this->J + this->dJ(i)));
		}
	}
	locEnergies[2*Ns] = std::make_tuple(_id, static_cast<_type>(localVal));				// append unchanged at the very end
}

// ----------------------------------------------------------------------------- BUILDING HAMILTONIAN -----------------------------------------------------------------------------

/*
* Sets the non-diagonal elements of the Hamimltonian matrix, by acting with the operator on the k-th state
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param temp resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template <typename _type>
void Heisenberg<_type>::setHamiltonianElem(u64 k, double value, u64 new_idx) {
	this->H(new_idx, k) += value;
}

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void Heisenberg<_type>::hamiltonian() {
	auto Ns = this->lattice->get_Ns();
	try {
		this->H = SpMat<_type>(this->N, this->N);										//  hamiltonian memory reservation
	}
	catch (const std::bad_alloc& e) {
		std::cout << "Memory exceeded" << e.what() << "\n";
		assert(false);
	}

	for (auto k = 0; k < this->N; k++) {
		double s_i = 0;
		for (int j = 0; j <= Ns - 1; j++) {
			s_i = checkBit(k, Ns - 1 - j) ? 1.0 : -1.0;									// true - spin up, false - spin down
				
			// disorder
			this->H(k, k) += (this->h + dh(j)) * s_i;									// perpendicular magnetic field

			// transverse field
			u64 new_idx = flip(k, BinaryPowers[Ns - 1 - j], Ns - 1 - j);			
			setHamiltonianElem(k, this->g + this->dg(j), new_idx);	

			// interaction
			const auto nn = this->lattice->get_nn(j, 0);
			if (nn >= 0) {						// check if nn exists
				// Ising-like spin correlation 
				double s_j = checkBit(k, Ns - 1 - nn) ? 1.0 : -1.0;						// check the bit on the nn
				auto interaction = (this->J + this->dJ(j));
				this->H(k, k) += interaction * s_i * s_j;				// setting the neighbors elements
		
				// S+S- + S-S+ hopping
				if (s_i * s_j < 0)
					setHamiltonianElem(k, 0.5 * interaction, flip(new_idx, BinaryPowers[Ns - 1 - nn], Ns - 1 - nn));

			}
		}
	}
}


#endif // !HEISENBERG_H


#ifndef HEISENBERG_CL_END
#define HEISENBERG_CL_END
template <typename _type>
class Heisenberg_cl_end : public Heisenberg<_type> {
private:
	double J_dot = 1;
	~Heisenberg_cl_end() = default;

public:
	void locEnergy(u64 _id) override;
	void hamiltonian() override;

};
#endif