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

		// state values number in local energy without the number of nearest neighbors
		this->state_val_num = 2;
		this->state_val = v_1d<std::pair<u64, _type>>(this->state_val_num + this->lattice->get_nn_number(0), std::pair(LLONG_MAX, 0.0));
		// change info
		this->info = this->inf();
	};
	// ----------------------------------- SETTERS ---------------------------------

	// ----------------------------------- GETTERS ---------------------------------
	const v_1d<pair<u64, _type>>& locEnergy(u64 _id, uint site) override;
	const v_1d<pair<u64, _type>>& locEnergy(const vec& v, uint site) override;
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
			",K=(" + STRP(this->Kx, 2) + "," + STRP(this->Ky, 2) + "," + STRP(this->Ky, 2) + ")" \
			",K0=" + STRP(this->K0, 2);
		return this->SpinHamiltonian<_type>::inf(name, skip, sep);
	};
	void update_info() override { this->info = this->inf(); };
};

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* @brief Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
inline const v_1d<pair<u64, _type>>& Heisenberg_kitaev<_type>::locEnergy(u64 _id, uint site) {

	// sumup the value of non-changed state
	double localVal = 0;

	uint iter = 1;
	uint nn_number = this->lattice->get_nn_forward_num(site);

	// true - spin up, false - spin down
	double si = checkBit(_id, this->Ns - site - 1) ? this->_SPIN : -this->_SPIN;

	// perpendicular field (SZ) - HEISENBERG
	localVal += (this->h + this->dh(site)) * si;

	// transverse field (SX) - HEISENBERG
	const u64 new_idx = flip(_id, this->Ns - 1 - site);
	this->state_val[iter++] = std::make_pair(new_idx, this->_SPIN * (this->g + this->dg(site)));

	// check the Siz Si+1z
	for (auto nn = 0; nn < nn_number; nn++) {
		// double checking neighbors
		auto n_num = this->lattice->get_nn_forward_num(site, nn);
		if (auto nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			// check Sz 
			const double sj = checkBit(_id, this->Ns - 1 - nei) ? this->_SPIN : -this->_SPIN;
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
				flip_val += this->_SPIN * this->_SPIN * (this->Kx + this->dKx(site));

			this->state_val[iter++] = std::make_pair(flip_idx_nn, flip_val);
		}
		else
			this->state_val[iter++] = std::make_pair(INT64_MAX, 0.0);
	}
	// append unchanged at the very end
	this->state_val[0] = std::make_pair(_id, static_cast<_type>(localVal));
	return this->state_val;
}

/*
* @brief Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
const v_1d<pair<u64, _type>>& Heisenberg_kitaev<_type>::locEnergy(const vec& v, uint site) {
	double localVal = 0;

	uint iter = 1;
	uint nn_number = this->lattice->get_nn_forward_num(site);

	// true - spin up, false - spin down
	double si = checkBitV(v, site) > 0 ? this->_SPIN : -this->_SPIN;

	// perpendicular field (SZ) - HEISENBERG
	localVal += (this->h + this->dh(site)) * si;

	// transverse field (SX) - HEISENBERG
	this->tmp_vec = v;
	flipV(tmp_vec, site);
	const u64 new_idx = baseToInt(tmp_vec);
	this->state_val[iter++] = std::pair{ new_idx, this->_SPIN * (this->g + this->dg(site)) };

	// check the Siz Si+1z
	for (auto nn = 0; nn < nn_number; nn++) {
		// double checking neighbors
		auto n_num = this->lattice->get_nn_forward_num(site, nn);
		this->tmp_vec2 = this->tmp_vec;
		if (auto nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			// check Sz 
			double sj = checkBitV(v, nei) > 0 ? this->_SPIN : -this->_SPIN;

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
				flip_val += this->_SPIN * this->_SPIN * (this->Kx + this->dKx(site));
			this->state_val[iter++] = std::make_pair(flip_idx_nn, flip_val);
		}
		else
			this->state_val[iter++] = std::make_pair(INT64_MAX, 0.0);
	}
	// append unchanged at the very end
	this->state_val[0] = std::make_pair(baseToInt(v), static_cast<_type>(localVal));
	return this->state_val;
}


// ----------------------------------------------------------------------------- BUILDING HAMILTONIAN -----------------------------------------------------------------------------

/*
* @brief Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void Heisenberg_kitaev<_type>::hamiltonian() {

	this->init_ham_mat();

	for (auto k = 0; k < this->N; k++) {
		for (int i = 0; i < this->Ns; i++) {
			// check all the neighbors
			uint nn_number = this->lattice->get_nn_forward_num(i);

			// true - spin up, false - spin down
			double si = checkBit(k, this->Ns - i - 1) ? this->_SPIN : -this->_SPIN;

			// perpendicular field (SZ) - HEISENBERG
			this->H(k, k) += (this->h + this->dh(i)) * si;

			// HEISENBERG
			const u64 new_idx = flip(k, this->Ns - 1 - i);
			this->setHamiltonianElem(k, this->_SPIN * (this->g + this->dg(i)), new_idx);

			// check the Siz Si+1z
			for (auto nn = 0; nn < nn_number; nn++) {
				// double checking neighbors
				auto n_num = this->lattice->get_nn_forward_num(i, nn);
				if (auto nei = this->lattice->get_nn(i, n_num); nei >= 0) {
					// check Sz 
					double sj = checkBit(k, this->Ns - 1 - nn) ? this->_SPIN : -this->_SPIN;

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
						this->setHamiltonianElem(k, this->_SPIN * this->_SPIN * (this->Kx + this->dKx(i)), flip_idx_nn);
				}
			}
		}
	}
}



#endif
