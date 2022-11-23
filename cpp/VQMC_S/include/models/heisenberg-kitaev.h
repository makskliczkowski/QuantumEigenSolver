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
			"hei_ktv,Ns=" + STR(this->Ns) + \
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

	//const uint nn_number = this->lattice->get_nn_forward_num(site);
	auto nns = this->lattice->get_nn_forward_number(site);

	// true - spin up, false - spin down
	const double si = checkBit(_id, this->Ns - site - 1) ? operators::_SPIN_RBM : -operators::_SPIN_RBM;

	// perpendicular field (SZ) - HEISENBERG
	localVal += (this->h + this->dh(site)) * si;

	// transverse field (SX) - HEISENBERG
	changedVal += f1(site, si) * operators::_SPIN_RBM * (this->g + this->dg(site));


	// check the Siz Si+1z
	for (auto n_num: nns) {
		if (int nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			// check Sz 
			const double sj = checkBit(_id, this->Ns - 1 - nei) ? operators::_SPIN_RBM : -operators::_SPIN_RBM;
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
				flip_val += operators::_SPIN_RBM * operators::_SPIN_RBM * (this->Kx + this->dKx(site));

			INT_TO_BASE_BIT(flip_idx_nn, tmp);
			changedVal += flip_val * f2(tmp * operators::_SPIN_RBM);
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

	//const uint nn_number = this->lattice->get_nn_forward_num(site);
	auto nns = this->lattice->get_nn_forward_number(site);

	// true - spin up, false - spin down
	const double si = checkBitV(v, site) > 0 ? operators::_SPIN_RBM : -operators::_SPIN_RBM;

	// perpendicular field (SZ) - HEISENBERG
	localVal += (this->h + this->dh(site)) * si;

	// transverse field (SX) - HEISENBERG
	changedVal += f1(site, si) * operators::_SPIN_RBM * (this->g + this->dg(site));

	tmp = v;
	flipV(tmp, site);

	// check the Siz Si+1z
	for (auto n_num : nns) {
		if (int nei = this->lattice->get_nn(site, n_num); nei >= 0) {
			// check Sz 
			const double sj = checkBitV(v, nei) > 0 ? operators::_SPIN_RBM : -operators::_SPIN_RBM;

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
				flip_val += operators::_SPIN_RBM * operators::_SPIN_RBM * (this->Kx + this->dKx(site));

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
	auto Ns = this->lattice->get_Ns();

	for (u64 k = 0; k < this->N; k++) {
		// loop over all sites
		u64 idx = 0;
		cpx val = 0.0;
		for (int i = 0; i < this->Ns; i++) {
			// check all the neighbors
			auto nns = this->lattice->get_nn_forward_number(i);

			// perpendicular field (SZ) - HEISENBERG
			std::tie(idx, val) = Operators<cpx>::sigma_z(k, Ns, { i });
			this->H(idx, k) += (this->h + this->dh(i)) * real(val);

			// flip with S^x_i with the transverse field
			std::tie(idx, val) = Operators<cpx>::sigma_x(k, Ns, { i });
			this->setHamiltonianElem(k, (this->g + this->dg(i)) * real(val), idx);

			// check the Siz Si+1z
			for (auto n_num : nns) {
				if (auto nei = this->lattice->get_nn(i, n_num); nei >= 0) {
					// --------------------- HEISENBERG ---------------------

					// diagonal elements setting  interaction field
					double interaction = (this->J + this->dJ(i));
					auto [idx_z, val_z] = Operators<cpx>::sigma_z(k, Ns, { i });
					auto [idx_z2, val_z2] = Operators<cpx>::sigma_z(idx_z, Ns, { nei });

					// setting the neighbors elements
					this->H(idx_z2, k) += interaction * this->delta * real(val_z * val_z2);

					// setting the neighbors elements
					auto [idx_x, val_x] = Operators<cpx>::sigma_x(k, Ns, { i });
					auto [idx_x2, val_x2] = Operators<cpx>::sigma_x(idx_x, Ns, { nei });

					// S+S- + S-S+ hopping
					if (real(val_z * val_z2) < 0)
						this->setHamiltonianElem(k, 0.5 * interaction, idx_x2);

					// --------------------- KITAEV ---------------------
					
					// z_bond
					if (n_num == 0)
						this->setHamiltonianElem(k, (this->Kz + this->dKz(i)) * real(val_z * val_z2), idx_z2);
					// y_bond
					else if (n_num == 1) {
						auto [idx_y, val_y] = Operators<cpx>::sigma_y(k, Ns, { i });
						auto [idx_y2, val_y2] = Operators<cpx>::sigma_y(idx_y, Ns, { nei });
						this->setHamiltonianElem(k, (this->Ky + this->dKy(i)) * real(val_y * val_y2), idx_y2);
					}
					// x_bond
					else if (n_num == 2)
						this->setHamiltonianElem(k, (this->Kx + this->dKx(i)) * real(val_x * val_x2), idx_x2);
				}
			}
		}
	}
}



#endif
