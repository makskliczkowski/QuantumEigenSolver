#pragma once
#ifndef HAMIL_H
#include "../hamil.h"
#endif // !HAMIL_H


#ifndef XYZ_H
#define XYZ_H

/*
* Model with disorder thus with no symmetries Nuclear PhysicsB966(2021)115373
*/
template <typename _type>
class XYZ : public SpinHamiltonian<_type> {
public:
	// ------------------------------------------- 				 Constructors				  -------------------------------------------
	~XYZ() = default;
	XYZ() = default;
	XYZ(std::shared_ptr<Lattice> lat, bool parity_break = true);
	XYZ(std::shared_ptr<Lattice> lat, double Ja, double Jb, double hx, double hz, double Delta_a, double Delta_b, double eta_a, double eta_b, bool parity_break = true)
		: Ja(Ja), Jb(Jb), hx(hx), hz(hz), Delta_a(Delta_a), Delta_b(Delta_b), eta_a(eta_a), eta_b(eta_b), XYZ(lat, parity_break) {};

	double Ja = 1.0;																									// nearest neighbors J
	double Jb = 1.0;																									// next nearest neighbors J	
	double hx = 0.2;																									// sigma x field
	double hz = 0.8;																									// sigma z field
	double Delta_a = 0.9;																								// sigma_z*sigma_z nearest neighbors
	double Delta_b = 0.9;																								// sigma_z*sigma_z next nearest neighbors
	double eta_a = 0.5;
	double eta_b = 0.5;

	bool parity_break = true;

private:
	u64 map(u64 index) const override;

public:
	// METHODS
	void hamiltonian() override;
	void setHamiltonianElem(u64 k, _type value, u64 new_idx) override;													// sets the Hamiltonian elements
	cpx locEnergy(u64, uint, std::function<cpx(int, double)>, std::function<cpx(const vec&)>, vec&) override;					// returns the local energy for VQMC purposes
	cpx locEnergy(const vec&, uint, std::function<cpx(int, double)>, std::function<cpx(const vec&)>, vec&) override;			// returns the local energy for VQMC purposes

	// ------------------------------------------- 				 Info				  -------------------------------------------

	string inf(const v_1d<std::string>& skip = {}, std::string sep = "_") const override
	{
		std::string name = sep + \
			"xyz,Ns=" + STR(this->Ns) + \
			",Ja=" + STRP(this->Ja, 2) + \
			",Jb=" + STRP(this->Jb, 2) + \
			",hx=" + STRP(this->hx, 2) + \
			",hz=" + STRP(this->hz, 2) + \
			",da=" + STRP(this->Delta_a, 2) + \
			",db=" + STRP(this->Delta_b, 2) + \
			",ea=" + STRP(this->eta_a, 2) + \
			",eb=" + STRP(this->eta_b, 2) + \
			",pb=" + STR(this->parity_break) + \
			",bc=" + STR(this->lattice->get_BC());

		return this->SpinHamiltonian<_type>::inf(name, skip, sep);
	}
	void update_info() override { this->info = this->inf(); };
};

// ----------------------------------------------------------------------------- CONSTRUCTORS -----------------------------------------------------------------------------

/*
* @brief Ising disorder constructor
* @param lat general lattice class that informs about the topology of the system lattice
*/
template <typename _type>
XYZ<_type>::XYZ(std::shared_ptr<Lattice> lat, bool parity_break)
{
	this->lattice = lat;
	this->ran = randomGen();
	this->parity_break = parity_break;
	this->Ns = this->lattice->get_Ns();
	this->N = ULLPOW(this->Ns);															// Hilber space size
	this->state_val_num = 2;

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
u64 XYZ<_type>::map(u64 index) const {
	if (index < 0 || index >= std::pow(2, this->lattice->get_Ns())) throw "Element out of range\n No such index in map\n";
	return index;
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
cpx XYZ<_type>::locEnergy(u64, uint, std::function<cpx(int, double)>, std::function<cpx(const vec&)>, vec& v) {

	return 0.0;
}

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
cpx XYZ<_type>::locEnergy(const vec&, uint, std::function<cpx(int, double)>, std::function<cpx(const vec&)>, vec& v) {
	return 0.0;
}

// ----------------------------------------------------------------------------- BUILDING HAMILTONIAN -----------------------------------------------------------------------------

/*
* Sets the non-diagonal elements of the Hamimltonian matrix, by acting with the operator on the k-th state
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template <typename _type>
void XYZ<_type>::setHamiltonianElem(u64 k, _type value, u64 new_idx) {
	this->H(new_idx, k) += value;
}

/*
* Sets the non-diagonal elements of the Hamimltonian matrix, by acting with the operator on the k-th state
* @param k index of the basis state acted upon with the Hamiltonian
* @param value value of the given matrix element to be set
* @param new_idx resulting vector form acting with the Hamiltonian operator on the k-th basis state
*/
template <>
inline void XYZ<cpx>::setHamiltonianElem(u64 k, cpx value, u64 new_idx) {
	this->H(new_idx, k) += value;
}

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void XYZ<_type>::hamiltonian() {
	this->init_ham_mat();
	auto Ns = this->lattice->get_Ns();

	// loop over all states
	for (u64 k = 0; k < this->N; k++) {
		// loop over all sites
		u64 idx = 0;
		cpx val = 0.0;
		for (int j = 0; j <= Ns - 1; j++) {
			const uint nn_number = this->lattice->get_nn_forward_num(j);
			const uint nnn_number = this->lattice->get_nnn_forward_num(j);

			// diagonal elements setting the perpendicular field

			const double perpendicular_val = ((j == Ns - 1) && this->parity_break) ? 0.5 * this->hz : this->hz;
			std::tie(idx, val) = Operators<cpx>::sigma_z(k, Ns, { j });
			this->H(idx, k) += perpendicular_val * real(val);

			if (this->parity_break && Ns < 10)
				stout << VEQ(k) << "\t" << VEQ(j) << "\t" << VEQ(perpendicular_val) << EL;
			//this->H(k, k) += 16.0 * s_i;

			// flip with S^x_i with the transverse field
			const double transverse_val = ((j == 0) && this->parity_break) ? 0.5 * hx : hx;
			std::tie(idx, val) = Operators<cpx>::sigma_x(k, Ns, { j });
			this->setHamiltonianElem(k, transverse_val * real(val), idx);

			if (this->parity_break && Ns < 10)
				stout << VEQ(k) << "\t" << VEQ(j) << "\t" << VEQ(transverse_val) << EL;
			//this->H(new_idx, k) += 4.0;

			// -------------- CHECK NN ---------------
			for (auto nn = 0; nn < nn_number; nn++) {
				auto n_num = this->lattice->get_nn_forward_num(j, nn);
				if (auto nei = this->lattice->get_nn(j, n_num); nei >= 0) {
					if (this->parity_break && Ns < 10)
						stout << VEQ(k) << "\t" << VEQ(j) << "\t" << VEQ(nn_number) << "\t" << VEQ(n_num) << "\t" << VEQ(nei) << EL;

					// setting the neighbors elements
					auto [idx_x, val_x] = Operators<cpx>::sigma_x(k, Ns, { j });
					auto [idx_x2, val_x2] = Operators<cpx>::sigma_x(idx_x, Ns, { nei });
					this->H(idx_x2, k) += this->Ja * (1.0 - this->eta_a) * real(val_x * val_x2);

					auto [idx_y, val_y] = Operators<cpx>::sigma_y(k, Ns, { j });
					auto [idx_y2, val_y2] = Operators<cpx>::sigma_y(idx_y, Ns, { nei });
					this->H(idx_y2, k) += this->Ja * (1.0 + this->eta_a) * real(val_y * val_y2);

					auto [idx_z, val_z] = Operators<cpx>::sigma_z(k, Ns, { j });
					auto [idx_z2, val_z2] = Operators<cpx>::sigma_z(idx_z, Ns, { nei });
					this->H(idx_z2, k) += this->Ja * this->Delta_a * real(val_z * val_z2);

					if (this->parity_break && Ns < 10)
						stout << VEQ(k) << "\t" << VEQ(j) << "\t" << VEQ(real(val_x * val_x2)) << "\t" << VEQ(real(val_y * val_y2)) << EL;

					//this->H(flip_idx_nn, k) -= 15.0 * (s_i) * (s_j);
				}
			}

			// -------------- CHECK NNN ---------------
			for (auto nnn = 0; nnn < nnn_number; nnn++) {
				auto n_num = this->lattice->get_nnn_forward_num(j, nnn);
				if (auto nei = this->lattice->get_nnn(j, n_num); nei >= 0 && (j > 0 || !this->parity_break)) {
					if (this->parity_break && Ns < 10)
						stout << VEQ(k) << "\t" << VEQ(j) << "\t" << VEQ(nnn_number) << "\t" << VEQ(n_num) << "\t" << VEQ(nei) << EL;

					// setting the neighbors elements
					auto [idx_x, val_x] = Operators<cpx>::sigma_x(k, Ns, { j });
					auto [idx_x2, val_x2] = Operators<cpx>::sigma_x(idx_x, Ns, { nei });
					this->H(idx_x2, k) += this->Jb * (1.0 - this->eta_b) * real(val_x * val_x2);

					auto [idx_y, val_y] = Operators<cpx>::sigma_y(k, Ns, { j });
					auto [idx_y2, val_y2] = Operators<cpx>::sigma_y(idx_y, Ns, { nei });
					this->H(idx_y2, k) += this->Jb * (1.0 + this->eta_b) * real(val_y * val_y2);

					auto [idx_z, val_z] = Operators<cpx>::sigma_z(k, Ns, { j });
					auto [idx_z2, val_z2] = Operators<cpx>::sigma_z(idx_z, Ns, { nei });
					this->H(idx_z2, k) += this->Jb * this->Delta_b * real(val_z * val_z2);

					if (this->parity_break && Ns < 10)
						stout << VEQ(k) << "\t" << VEQ(j) << "\t" << VEQ(real(val_x * val_x2)) << "\t" << VEQ(real(val_y * val_y2)) << EL;
				}
			}
			if (this->parity_break && Ns < 10) stout << EL;
		}
	}
}


#endif // !ISING_H