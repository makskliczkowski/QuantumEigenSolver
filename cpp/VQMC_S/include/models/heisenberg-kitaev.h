#pragma once
#ifndef HEISENBERG_H
#include "heisenberg.h"
#endif // !HEISENBERG_H

// --------------------------------------------------------------------------- HEISENBERG INTERACTING WITH KITAEV SPINS ---------------------------------------------------------------------------
#ifndef HEISENBERG_KITAEV
#define HEISENBERG_KITAEV
template<typename _type>
class Heisenberg_kitaev : public Heisenberg<_type> {
private:
	double Kx = 1.0;									// kitaev model exchange 
	double Ky = 1.0;									// kitaev model exchange 
	double Kz = 1.0;									// kitaev model exchange 
	vec dKx;											// kitaev model exchange vector
	vec dKy;											// kitaev model exchange vector
	vec dKz;											// kitaev model exchange vector
	double K0;											// disorder with Kitaev exchange

public:
	~Heisenberg_kitaev() = default;
	Heisenberg_kitaev(double J, double J0, double g, double g0, double h, double w, double delta, std::tuple<double, double, double> K, double K0, std::shared_ptr<Lattice> lat)
		: Heisenberg<_type>(J, J0, g, g0, h, w, delta, lat)
	{
		this->Kx = std::get<0>(K);
		this->Ky = std::get<1>(K);
		this->Kz = std::get<2>(K);
		this->K0 = K0;
		// creates random disorder vector
		this->dKx = create_random_vec(this->Ns, this->ran, this->K0);						
		this->dKy = create_random_vec(this->Ns, this->ran, this->K0);
		this->dKz = create_random_vec(this->Ns, this->ran, this->K0);
		this->loc_states_num = 4 * this->Ns + 1;																// number of states after local energy work
		this->locEnergies = v_1d<std::tuple<u64, _type>>(this->loc_states_num, std::make_tuple(LONG_MAX, 0));	// set local energies vector
		// change info
		this->info = this->inf();
	};
	// ----------------------------------- SETTERS ---------------------------------

	// ----------------------------------- GETTERS ---------------------------------
	void locEnergy(u64 _id) override;
	void hamiltonian() override;

	string inf(const v_1d<string>& skip = {}, string sep = "_") const override
	{
		string name = sep + \
			"heisenberg_kitaev," + VEQ(Ns) + \
			",J=" + STRP(J, 2) + \
			",J0=" + STRP(J0, 2) + \
			",d=" + STRP(delta, 2) + \
			",g=" + STRP(g, 2) + \
			",g0=" + STRP(g0, 2) + \
			",h=" + STRP(h, 2) + \
			",w=" + STRP(w, 2) + \
			",Kx=" + STRP(Kx, 2) + \
			",Ky=" + STRP(Ky, 2) + \
			",Kz=" + STRP(Kz, 2) + \
			",K0=" + STRP(K0, 2);
		return SpinHamiltonian::inf(name, skip, sep);
	}
};

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* @brief Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
void Heisenberg_kitaev<_type>::locEnergy(u64 _id) {


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

		// perpendicular field (SZ) - HEISENBERG
		localVal += (this->h + this->dh(i)) * si;

		// transverse field (SX) - HEISENBERG
		const u64 new_idx = flip(_id, this->Ns - 1 - i);
		this->locEnergies[i] = std::make_tuple(new_idx, this->g + this->dg(i));

		// check the correlations
		for (auto n_num = 0; n_num < nn_number; n_num++) {
			if (auto nn = this->lattice->get_nn(i, n_num); nn >= 0 && nn >= i) {//&& nn >= i
				// stout << VEQ(i) << ", nei=" << VEQ(nn) << EL;
				// check Sz 
				double sj = checkBit(_id, this->Ns - 1 - nn) ? 1.0 : -1.0;
				
				// --------------------- HEISENBERG 
				
				// diagonal elements setting  interaction field
				auto interaction = this->J + this->dJ(i);
				auto sisj = si * sj;
				localVal += interaction * this->delta * sisj;
				
				auto flip_idx_nn = flip(new_idx, this->Ns - 1 - nn);

				// S+S- + S-S+
				if (sisj < 0)
					this->locEnergies[this->Ns + i] = std::make_tuple(flip_idx_nn, 0.5 * interaction);
				else
					this->locEnergies[this->Ns + i] = std::make_tuple(LONG_MAX, 0);
				
				// --------------------- KITAEV
				if (n_num == 0)
					localVal += (this->Kz + this->dKz(i)) * sisj;
				else if (n_num == 1)
					this->locEnergies[2 * this->Ns + i] = std::make_tuple(flip_idx_nn, -(this->Ky + this->dKy(i)) * sisj);
				else if (n_num == 2)
					this->locEnergies[3 * this->Ns + i] = std::make_tuple(flip_idx_nn, this->Kx + this->dKx(i));
				else
				{
					this->locEnergies[2 * this->Ns + i] = std::make_tuple(LONG_MAX, 0);
					this->locEnergies[3 * this->Ns + i] = std::make_tuple(LONG_MAX, 0);
				}
			}
		}
	}
	// append unchanged at the very end
	this->locEnergies[4 * this->Ns] = std::make_tuple(_id, static_cast<_type>(localVal));				
}

// ----------------------------------------------------------------------------- BUILDING HAMILTONIAN -----------------------------------------------------------------------------

/*
* @brief Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void Heisenberg_kitaev<_type>::hamiltonian() {
	try {
		this->H = SpMat<_type>(this->N, this->N);										//  hamiltonian memory reservation
	}
	catch (const std::bad_alloc& e) {
		std::cout << "Memory exceeded" << e.what() << "\n";
		assert(false);
	}


	for (auto k = 0; k < this->N; k++) {
		for (int j = 0; j < this->Ns; j++) {
			// check all the neighbors
			auto nn_number = this->lattice->get_nn_number(j);
			
			// true - spin up, false - spin down
			double si = checkBit(k, this->Ns - j - 1) ? 1.0 : -1.0;

			// perpendicular field (SZ) - HEISENBERG
			this->H(k, k) += (this->h + dh(j)) * si;

			// HEISENBERG
			const u64 new_idx = flip(k, this->Ns - 1 - j);
			setHamiltonianElem(k, this->g + this->dg(j), new_idx);

			// check the correlations
			for (auto n_num = 0; n_num < nn_number; n_num++) {
				if (auto nn = this->lattice->get_nn(j, n_num); nn >= 0 && nn > j) { //  
					// check Sz 
					double sj = checkBit(k, this->Ns - 1 - nn) ? 1.0 : -1.0;

					// --------------------- HEISENBERG 
					
					// diagonal elements setting  interaction field
					auto interaction = (this->J + this->dJ(j));
					auto sisj = si * sj;

					// setting the neighbors elements
					this->H(k, k) += interaction * this->delta * sisj;

					const u64 flip_idx_nn = flip(new_idx, this->Ns - 1 - nn);

					// S+S- + S-S+ hopping
					if (si * sj < 0)
						setHamiltonianElem(k, 0.5 * interaction, flip_idx_nn);
					
					// --------------------- KITAEV
					if (n_num == 0)
						setHamiltonianElem(k, (this->Kz + this->dKz(j)) * sisj, k);
					else if (n_num == 1)
						setHamiltonianElem(k, -(this->Ky + this->dKy(j)) * sisj, flip_idx_nn);
					else if (n_num == 2)
						setHamiltonianElem(k, this->Kx + this->dKx(j), flip_idx_nn);
				}
			}
		}
	}
}



#endif
