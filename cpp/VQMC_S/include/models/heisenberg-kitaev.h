#pragma once

#include "heisenberg.h"


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
	vec tmp_vec;
	vec tmp_vec2;
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

		// change info
		this->info = this->inf();
	};
	// ----------------------------------- SETTERS ---------------------------------

	// ----------------------------------- GETTERS ---------------------------------
	v_1d<pair<u64, _type>> locEnergy(u64 _id, uint site) override;
	v_1d<pair<u64, _type>> locEnergy(const vec& v, uint site) override;
	void hamiltonian() override;

	string inf(const v_1d<string>& skip = {}, string sep = "_") const override
	{
		string name = sep + \
			"hei_kitv,Ns=" + STR(this->Ns) + \
			",J=" + STRP(this->J, 2) + \
			",J0=" + STRP(this->J0, 2) + \
			",d=" + STRP(this->delta, 2) + \
			",g=" + STRP(this->g, 2) + \
			",g0=" + STRP(this->g0, 2) + \
			",h=" + STRP(this->h, 2) + \
			",w=" + STRP(this->w, 2) + \
			",K=(" + STRP(this->Kx, 2) + "," + STRP(this->Ky,2) + "," + STRP(this->Ky, 2) + ")" \
			",K0=" + STRP(this->K0, 2);
		return this->SpinHamiltonian<_type>::inf(name, skip, sep);
	}
};

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* @brief Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
inline v_1d<pair<u64, _type>> Heisenberg_kitaev<_type>::locEnergy(u64 _id, uint site) {

	// sumup the value of non-changed state
	double localVal = 0;

	uint iter = 1;
	v_1d<uint> nn_number = this->lattice->get_nn_forward_number(site);
	v_1d<std::pair<u64, _type>> state_val(2 + nn_number.size(), std::pair(LLONG_MAX, 0.0));


	// true - spin up, false - spin down
	double si = checkBit(_id, this->Ns - site - 1) ? 1.0 : -1.0;

	// perpendicular field (SZ) - HEISENBERG
	localVal += (this->h + this->dh(site)) * si;

	// transverse field (SX) - HEISENBERG
	const u64 new_idx = flip(_id, this->Ns - 1 - site);
	state_val[iter++] = std::make_pair(new_idx, this->g + this->dg(site));

	// check the Siz Si+1z
	for (auto n_num : nn_number) {
		// double checking neighbors
		if (auto nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			// check Sz 
			double sj = checkBit(_id, this->Ns - 1 - nei) ? 1.0 : -1.0;

			// --------------------- HEISENBERG 

			// diagonal elements setting  interaction field
			const double interaction = this->J + this->dJ(nei);
			const double sisj = si * sj;
			localVal += interaction * this->delta * sisj;

			const u64 flip_idx_nn = flip(new_idx, this->Ns - 1 - nei);
			double flip_val = 0.0;

			// S+S- + S-S+
			if (sisj < 0)
				flip_val += 0.5 * interaction;

			// --------------------- KITAEV
			if (n_num == 0)
				localVal += (this->Kz + this->dKz(site)) * sisj;
			else if (n_num == 1)
				flip_val -= (this->Ky + this->dKy(site)) * sisj;

			else if (n_num == 2)
				flip_val += this->Kx + this->dKx(site);

			state_val[iter++] = std::make_pair(flip_idx_nn, flip_val);
		}
	}
	// append unchanged at the very end
	state_val[0] = std::make_pair(_id, static_cast<_type>(localVal));
	return state_val;
}

/*
* @brief Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
v_1d<pair<u64, _type>> Heisenberg_kitaev<_type>::locEnergy(const vec& v, uint site) {
	double localVal = 0;

	uint iter = 1;
	v_1d<uint> nn_number = this->lattice->get_nn_forward_number(site);
	v_1d<std::pair<u64, _type>> state_val(2 + nn_number.size(), std::pair(LLONG_MAX, 0.0));

	// true - spin up, false - spin down
	double si = checkBitV(v, site) > 0 ? 1.0 : -1.0;

	// perpendicular field (SZ) - HEISENBERG
	localVal += (this->h + this->dh(site)) * si;

	// transverse field (SX) - HEISENBERG
	this->tmp_vec = v;
	flipV(tmp_vec, site);
	const u64 new_idx = baseToInt(tmp_vec);
	state_val[iter++] = std::pair{ new_idx, this->g + this->dg(site) };

	// check the Siz Si+1z
	for (auto n_num : nn_number) {
		this->tmp_vec2 = this->tmp_vec;
		if (auto nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			// check Sz 
			double sj = checkBitV(v, nei) > 0 ? 1.0 : -1.0;

			// --------------------- HEISENBERG 

			// diagonal elements setting  interaction field
			const auto interaction = this->J + this->dJ(site);
			const auto sisj = si * sj;
			localVal += interaction * this->delta * sisj;

			flipV(tmp_vec2, nei);
			auto flip_idx_nn = baseToInt(tmp_vec2);
			double flip_val = 0.0;

			// S+S- + S-S+
			if (sisj < 0)
				flip_val += 0.5 * interaction;

			// --------------------- KITAEV
			if (n_num == 0)
				localVal += (this->Kz + this->dKz(site)) * sisj;
			else if (n_num == 1)
				flip_val -= (this->Ky + this->dKy(site)) * sisj;

			else if (n_num == 2)
				flip_val += this->Kx + this->dKx(site);
			state_val[iter++] = std::make_pair(flip_idx_nn, flip_val);
		}
	}
	// append unchanged at the very end
	state_val[0] = std::make_pair(baseToInt(v), static_cast<_type>(localVal));
	return state_val;
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
		for (int i = 0; i < this->Ns; i++) {
			// check all the neighbors
			v_1d<uint> nn_number = this->lattice->get_nn_forward_number(i);
			
			// true - spin up, false - spin down
			double si = checkBit(k, this->Ns - i - 1) ? 1.0 : -1.0;

			// perpendicular field (SZ) - HEISENBERG
			this->H(k, k) += (this->h + this->dh(i)) * si;

			// HEISENBERG
			const u64 new_idx = flip(k, this->Ns - 1 - i);
			this->setHamiltonianElem(k, this->g + this->dg(i), new_idx);

			// check if nn exists
			for (auto n_num : nn_number) {
				if (const auto nn = this->lattice->get_nn(i, n_num); nn >= 0) {
					// check Sz 
					double sj = checkBit(k, this->Ns - 1 - nn) ? 1.0 : -1.0;

					// --------------------- HEISENBERG 
					
					// diagonal elements setting  interaction field
					double interaction = (this->J + this->dJ(i));
					double sisj = si * sj;

					// setting the neighbors elements
					this->H(k, k) += interaction * this->delta * sisj;

					const u64 flip_idx_nn = flip(new_idx, this->Ns - 1 - nn);

					// S+S- + S-S+ hopping
					if (si * sj < 0)
						this->setHamiltonianElem(k, 0.5 * interaction, flip_idx_nn);
					
					// --------------------- KITAEV
					if (n_num == 0)
						this->setHamiltonianElem(k, (this->Kz + this->dKz(i)) * sisj, k);
					else if (n_num == 1)
						this->setHamiltonianElem(k, -(this->Ky + this->dKy(i)) * sisj, flip_idx_nn);
					else if (n_num == 2)
						this->setHamiltonianElem(k, this->Kx + this->dKx(i), flip_idx_nn);
				}
			}
		}
	}
}



#endif
