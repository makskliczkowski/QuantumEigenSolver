#pragma once
#ifndef HEISENBERG_H
#include "heisenberg.h"
#endif // !HEISENBERG_H

// --------------------------------------------------------------------------- HEISENBERG INTERACTING WITH CLASSICAL SPINS ---------------------------------------------------------------------------
#ifndef HEISENBERG_DOTS
#define HEISENBERG_DOTS
template <typename _type>
class Heisenberg_dots : public Heisenberg<_type> {
private:
	size_t dot_num;										// number of the dots
	vec J_dot;											// interaction of a single classical spin
	vec J_dots;											// interactions with the classical spins
	double J_dot0 = 0;									// disorder at the classical spin interaction

	v_1d<int> positions;								// positions of the dots on the lattice
	vec cos_thetas;										// parametrized angles of the classical spins [0,pi] - z plane - cosinuses
	vec sin_thetas;										// parametrized angles of the classical spins [0,pi] - z plane - sinuses
	vec cos_phis;										// parametrized angles of the classical spins [0,2pi] - xy plane - cosinuses
	vec sin_phis;										// parametrized angles of the classical spins [0,2pi] - xy plane - sinuses

	tuple<u64, double> sx_int;							// tuple to store the value of sx local interaction (after flip)
	tuple<u64, _type> sy_int;							// tuple to store the value of sy local interaction (after flip)
	tuple<u64, double> sz_int;							// tuple to store the value of sz local interaction (no flip)

public:
	~Heisenberg_dots() = default;
	Heisenberg_dots(double J, double J0, double g, double g0, double h, double w, double delta, std::shared_ptr<Lattice> lat,
		const v_1d<int>& positions, const vec& J_dot = { 0,0,1 }, double J_dot0 = 0);
	// ----------------------------------- 				 SETTERS 				 ---------------------------------
	void set_angles();
	void set_angles(const vec& phis, const vec& thetas);

	// -----------------------------------				 GETTERS 				 ---------------------------------
	void get_dot_interaction(u64 state, int position_elem);
	tuple<double, _type, double> get_dot_int_return(double si, int position_elem);

	// ----------------------------------- 				 OTHER STUFF 				 ---------------------------------
	void locEnergy(u64 _id) override;
	void hamiltonian() override;

	string inf(const v_1d<string>& skip = {}, string sep = "_") const override
	{
		string name = sep + \
			"heisenberg_with_dots," + VEQ(dot_num) + "," + VEQ(Ns) + \
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
template<typename _type>
inline Heisenberg_dots<_type>::Heisenberg_dots(double J, double J0, double g, double g0, double h, double w, double delta, std::shared_ptr<Lattice> lat, const v_1d<int>& positions, const vec& J_dot, double J_dot0)
	: Heisenberg<_type>(J, J0, g, g0, h, w, delta, lat)
{
	this->positions = positions;
	// sort the postitions vector for building block convinience
	std::ranges::sort(this->positions.begin(), this->positions.end());

	this->dot_num = positions.size();
	this->J_dot = J_dot;
	this->J_dot0 = J_dot0;
	// creates random disorder vector
	this->J_dots = create_random_vec(dot_num, this->ran, this->J_dot0);

	// reserve memory
	this->set_angles();

	// set info
	this->info = this->inf();
}


// ----------------------------------------------------------------------------- INITIAL CONDITIONS -----------------------------------------------------------------------------
/*
* @brief sets the angles to random
*/
template<typename _type>
inline void Heisenberg_dots<_type>::set_angles()
{

	vec phis = vec(this->dot_num, arma::fill::randu);
	vec thetas = vec(this->dot_num, arma::fill::randu);
	phis = phis * TWOPI;
	thetas = thetas * PI;
	this->cos_thetas = cos(thetas);
	this->sin_thetas = sin(thetas);
	this->cos_phis = cos(phis);
	this->sin_phis = sin(phis);
}
/*
* @brief sets the angles
* @param phis classical spins [0,1] - xy plane - you give this in PI units
* @param thetas classical spins [0,1] - z plane - you give this in PI units
*/
template<typename _type>
inline void Heisenberg_dots<_type>::set_angles(const vec& phis, const vec& thetas)
{
	vec a_phis(this->dot_num);
	vec a_thetas(this->dot_num);

	// check if phis is not full if so make it random
	if (phis.n_elem < (this->dot_num))
		a_phis = vec(this->dot_num, arma::fill::randu) * TWOPI;
	else
		a_phis = phis;

	// check if thetas is not full if so make it random
	if (thetas.n_elem < (this->dot_num))
		a_thetas = vec(this->dot_num, arma::fill::randu) * PI;
	else
		a_thetas = thetas;

	// set the angles now
	this->cos_thetas = cos(a_thetas);
	this->sin_thetas = sin(a_thetas);
	this->cos_phis = cos(a_phis);
	this->sin_phis = sin(a_phis);
}

// ----------------------------------------------------------------------------- DOT INTERACTION -----------------------------------------------------------------------------

/*
* @brief Gets the interaction with the dots at the given position_elem from positions vector and returns it
* @param si spin at i'th position
* @param position_elem - element of possitions in dots
*/
template<typename _type>
inline tuple<double, _type, double> Heisenberg_dots<_type>::get_dot_int_return(double si, int position_elem)
{
	const auto position = this->positions[position_elem];
	double s_z_int = 0.0;
	_type s_y_int = 0.0;
	double s_x_int = 0.0;
	
	// check the position at elem site as if the dot would be on top of the site (left nei)
	if (position >= 0 && position < this->Ns) {
		// set the s_z element
		const auto Jz = this->J_dots(position_elem) + this->J_dot(2);
		s_z_int = Jz * si * this->cos_thetas[position_elem];

		const auto Jy = this->J_dots(position_elem) + this->J_dot(1);
		// set the s_y element 
		s_y_int = Jy * imn * si * this->sin_thetas[position_elem] * this->sin_phis[position_elem];

		const auto Jx = this->J_dots(position_elem) + this->J_dot(0);
		// set the s_x element 
		s_x_int = Jx * this->sin_thetas[position_elem] * this->cos_phis[position_elem];
	}
	return std::make_tuple(s_x_int, s_y_int, s_z_int);
}

/*
* @brief Gets the interaction with the dots at the given position_elem from positions vector and returns it -
* but it is without y'th component of spin
* @param si spin at i'th position
* @param position_elem element of possitions in dots
*/
template<>
inline tuple<double, double, double> Heisenberg_dots<double>::get_dot_int_return(double si, int position_elem)
{
	const auto position = this->positions[position_elem];
	double s_z_int = 0.0;
	double s_y_int = 0.0;
	double s_x_int = 0.0;
	// check the position at elem site as if the dot would be on top of the site (left nei)
	if (position >= 0 && position < this->Ns) {
		// set the s_z element
		const auto Jz = this->J_dots(position_elem) + this->J_dot(2);
		s_z_int = this->cos_thetas[position_elem] > 0 ? Jz * si : -Jz * si;

		// set the s_y element 
		s_y_int = 0;

		const auto Jx = this->J_dots(position_elem) + this->J_dot(0);
		// set the s_x element 
		s_x_int = Jx * this->sin_thetas[position_elem] * this->cos_phis[position_elem];
	}
	return std::make_tuple(s_x_int, s_y_int, s_z_int);
}

/*
* @brief Gets the interaction with the dots at the given position_elem from positions vector
* @param state current state on which the flip is done
* @param position_elem element of possitions in dots
*/
template<typename _type>
inline void Heisenberg_dots<_type>::get_dot_interaction(u64 state, int position_elem)
{
	const auto position = this->positions[position_elem];
	// check the position at elem site as if the dot would be on top of the site (left nei)
	if (position >= 0 && position < this->Ns) {
		double si = checkBit(state, this->Ns - 1 - position) ? 1.0 : -1.0;

		// set the s_z element
		auto Jz = (this->J_dots(2) + this->J_dot(2));
		sz_int = make_tuple(state, Jz * si * this->cos_thetas[position_elem]);

		// flip the state 
		u64 new_state = flip(state, this->Ns - 1 - position);

		auto Jy = (this->J_dots(1) + this->J_dot(1));
		// set the s_y element 
		sy_int = make_tuple(new_state, Jy * imn * si * this->sin_thetas[position_elem] * this->sin_phis[position_elem]);

		auto Jx = (this->J_dots(0) + this->J_dot(0));
		// set the s_x element 
		sx_int = make_tuple(new_state, Jx * this->sin_thetas[position_elem] * this->cos_phis[position_elem]);
	}
}

// ----------------------------------------------------------- 				 BUILDING HAMILTONIAN 				 -----------------------------------------------------------

/*
* @brief Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _type>
void Heisenberg_dots<_type>::hamiltonian() {
	try {
		this->H = SpMat<_type>(this->N, this->N);										//  hamiltonian memory reservation
	}
	catch (const std::bad_alloc& e) {
		std::cout << "Memory exceeded" << e.what() << "\n";
		assert(false);
	}
	// build the Hamiltonian
	for (auto k = 0; k < this->N; k++) {
		// interaction with the dot at left site will be held with single variable as position is sorted
		int dot_iter = 0;
		for (int j = 0; j < this->Ns; j++) {
			// true - spin up, false - spin down
			double si = checkBit(k, this->Ns - 1 - j) ? 1.0 : -1.0;

			// perpendicular magnetic field
			this->H(k, k) += (this->h + dh(j)) * si;

			// transverse field
			const u64 new_idx = flip(k, this->Ns - 1 - j);
			setHamiltonianElem(k, this->g + this->dg(j), new_idx);

			// interaction - check if nn exists
			if (const auto nn = this->lattice->get_nn(j, 0); nn >= 0) {						
				// Ising-like spin correlation // check the bit on the nn
				double sj = checkBit(k, this->Ns - 1 - nn) ? 1.0 : -1.0;
				auto interaction = (this->J + this->dJ(j));
				
				// setting the neighbors elements
				this->H(k, k) += this->delta * interaction * si * sj;				

				// S+S- + S-S+ hopping
				if (si * sj < 0)
					setHamiltonianElem(k, 0.5 * interaction, flip(new_idx, this->Ns - 1 - nn));
			}
			// handle the dot
			if (positions[dot_iter] == j) {
				const auto [s_x_i, s_y_i, s_z_i] = this->get_dot_int_return(si, dot_iter);
				// set sz_int
				this->H(k, k) += s_z_i;
				// set sy_int
				this->setHamiltonianElem(k, s_y_i, new_idx);
				// set sx_int 
				this->setHamiltonianElem(k, s_x_i, new_idx);
				dot_iter++;
			}
		}
	}
}

// -----------------------------------------------------------  				 LOCAL ENERGY 				  -----------------------------------------------------------

/*
* @brief Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _type>
void Heisenberg_dots<_type>::locEnergy(u64 _id) {
	// sumup the value of non-changed state
	double localVal = 0;
	
	// cannot use omp because of dot_iter
	int dot_iter = 0;

	for (auto i = 0; i < this->Ns; i++) {
		// true - spin up, false - spin down
		double si = checkBit(_id, this->Ns - i - 1) ? 1.0 : -1.0;								

		// perpendicular field
		localVal += (this->h + this->dh(i)) * si;

		// transverse field
		const u64 new_idx = flip(_id, this->Ns - 1 - i);
		_type s_flipped_en = this->g + this->dg(i);

		// check the Siz Si+1z
		if (auto nei = this->lattice->get_nn(i, 0); nei >= 0) {
			double sj = checkBit(_id, this->Ns - 1 - nei) ? 1.0 : -1.0;
			auto interaction = (this->J + this->dJ(i));
			// diagonal elements setting  interaction field
			localVal += this->delta * interaction * si * sj;

			// S+S- + S-S+
			if (si * sj < 0)
				this->locEnergies[this->Ns + i] = std::make_tuple(flip(new_idx, this->Ns - 1 - nei), 0.5 * interaction);
			else
				this->locEnergies[this->Ns + i] = std::make_tuple(LONG_MAX, 0);
		}

		// handle the dot
		if (positions[dot_iter] == i) {
			const auto [s_x_i, s_y_i, s_z_i] = this->get_dot_int_return(si, dot_iter);
			// set sz_int
			localVal += s_z_i;
			// set sy_int and sx_int
			s_flipped_en += s_y_i + s_z_i;
			// next position!
			dot_iter++;
		}
		// set the flipped state
		this->locEnergies[i] = std::make_tuple(new_idx, s_flipped_en);
	}
	// append unchanged at the very end
	locEnergies[2 * this->Ns] = std::make_tuple(_id, static_cast<_type>(localVal));				
}
#endif