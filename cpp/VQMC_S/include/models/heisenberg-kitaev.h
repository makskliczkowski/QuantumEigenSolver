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
		// change info
		this->info = this->inf();
	};
	// ----------------------------------- SETTERS ---------------------------------

	// ----------------------------------- GETTERS ---------------------------------
	cpx locEnergy(u64 _id, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) override;
	cpx locEnergy(const vec& _id, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) override;
	void hamiltonian() override;

	string inf(const v_1d<string>& skip = {}, string sep = "_", int prec = 2) const override
	{
		string name = sep + \
			"hei_kitv,Ns=" + STR(this->Ns) + \
			",J=" + STRP(this->J, prec) + \
			",J0=" + STRP(this->J0, prec) + \
			",d=" + STRP(this->delta, prec) + \
			",g=" + STRP(this->g, prec) + \
			",g0=" + STRP(this->g0, prec) + \
			",h=" + STRP(this->h, prec) + \
			",w=" + STRP(this->w, prec) + \
			",K=(" + STRP(this->Kx, prec) + "," + STRP(this->Ky, prec) + "," + STRP(this->Ky, prec) + ")" \
			",K0=" + STRP(this->K0, prec);
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
inline cpx Heisenberg_kitaev<_type>::locEnergy(u64 _id, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) {

	// sumup the value of non-changed state
	double localVal = 0;
	cpx changedVal = 0.0;

	const uint nn_number = this->lattice->get_nn_forward_num(site);

	// true - spin up, false - spin down
	const double si = checkBit(_id, this->Ns - site - 1) ? this->_SPIN : -this->_SPIN;

	// perpendicular field (SZ) - HEISENBERG
	localVal += (this->h + this->dh(site)) * si;

	// transverse field (SX) - HEISENBERG
	changedVal += f1(site, si) * this->_SPIN * (this->g + this->dg(site));


	// check the Siz Si+1z
	for (auto nn = 0; nn < nn_number; nn++) {
		// double checking neighbors
		const uint n_num = this->lattice->get_nn_forward_num(site, nn);
		if (int nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			// check Sz 
			const double sj = checkBit(_id, this->Ns - 1 - nei) ? this->_SPIN : -this->_SPIN;
			// --------------------- HEISENBERG 

			// diagonal elements setting  interaction field
			const double interaction = this->J + this->dJ(nei);
			const double sisj = si * sj;
			localVal += interaction * this->delta * sisj;

			const u64 flip_idx_nn = flip(flip(_id, this->Ns - 1 - nei), this->Ns - 1 - site);
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

			INT_TO_BASE_BIT(flip_idx_nn, tmp);
			changedVal += flip_val * f2(tmp);
		}
	}
	return changedVal + localVal;
}

/*
* @brief Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
inline cpx Heisenberg_kitaev<_type>::locEnergy(const vec& v, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) {
	// sumup the value of non-changed state
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

			flipV(tmp, nei);
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

			flipV(tmp, nei);
			changedVal += flip_val * f2(tmp);
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
