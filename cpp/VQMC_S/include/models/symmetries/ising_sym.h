#pragma once
#ifndef HAMILSYM_H
#include "../../hamil_sym.h"
#endif // !HAMIL_H

#include "../ising.h"

namespace ising_sym {
	// structure that contains the possible symmetries
	struct sym {
		double k_sym;																	// translational symmetry generator
		int p_sym;																		// parity symmetry generator
		int x_sym;																		// spin-flip symmetry generator
		bool operator==(sym other) {
			return
				this->k_sym == other.k_sym
				&& this->p_sym == other.p_sym
				&& this->x_sym == other.x_sym;
		};
	};


	template <typename _type>
	class IsingModelSym : public SpinHamiltonianSym<_type> {
	private:
		// MODEL BASED PARAMETERS 
		double J = 1;																	// spin exchange
		double g = 1;																	// transverse magnetic field
		double h = 1;																	// perpendicular magnetic field
		
		ising_sym::sym symmetries;
		
		// ------------------- TRANSLATION -------------------
		bool k_sector;																	// if the k-sector allows parity symmetry
		v_1d<_type> k_exponents;														// precalculate the symmetry exponents for current k vector
		
		// -------------------------------- SYMMETRY ELEMENTS --------------------------------

	public:
		~IsingModelSym() = default;
		IsingModelSym() = default;
		IsingModelSym(double J, double g, double h, std::shared_ptr<Lattice> lat, int k_sym = 0, bool p_sym = true, bool x_sym = true, u32 thread_num = 1);


		// -------------------------------- GETTERS --------------------------------
		auto get_k_sector()								const RETURNS(this->k_sector);
		auto get_symmetries()							const RETURNS(this->symmetries);
		auto get_k_exp()								const RETURNS(this->k_exponents);

		// -------------------------------- OVERRIDING --------------------------------
		void createSymmetryGroup() override;											// create symmetry group elements and their corresponding eigenvalues
		void hamiltonian() override;													// creates the Hamiltonian itself
		
		string inf(const v_1d<string>& skip = {}, string sep = "_") const override
		{
			auto Ns = this->lattice->get_Ns();
			string name = sep + \
				"ising_sym,Ns=" + STR(Ns) + \
				",J=" + STRP(J, 2) + \
				",g=" + STRP(g, 2) + \
				",h=" + STRP(h, 2) + \
				",k=" + STRP(symmetries.k_sym, 2) + \
				",p=" + STRP(symmetries.p_sym, 2) + \
				",x=" + STRP(symmetries.x_sym, 2) + \
				",bc=" + STR(this->lattice->get_BC());

			return this->SpinHamiltonian<_type>::inf(name, skip, sep);
		}
		void update_info() override { this->info = this->inf(); };
	};

	/*
	* @brief standard constructor 
	*/
	template<typename _type>
	inline ising_sym::IsingModelSym<_type>::IsingModelSym(double J, double g, double h, std::shared_ptr<Lattice> lat, int k_sym, bool p_sym, bool x_sym, u32 thread_num)
		: J(J), g(g), h(h)
	{
		this->lattice = lat;
		this->thread_num = thread_num;
		this->Ns = lat->get_Ns();
		symmetries.p_sym = (p_sym) ? 1 : -1;
		symmetries.x_sym = (x_sym) ? 1 : -1;
		symmetries.k_sym = k_sym;
		//change info
		this->info = this->inf();
		symmetries.k_sym = k_sym * TWOPI / double(this->Ns);
		k_sector = valueEqualsPrec(symmetries.k_sym, 0.0, 1e-4) || valueEqualsPrec(symmetries.k_sym, double(PI), 1e-4);
		stout << "\t->Making complex\n";
		// printSeparatedP(stout, '\t', 6, true, 4, symmetries.k_sym, k_sym, VEQ(k_sector));

		// precalculate the exponents
		this->k_exponents = v_1d<_type>(this->Ns, 0.0);
		for (int l = 0; l < this->Ns; l++) 
			this->k_exponents[l] = std::exp(-imn * this->symmetries.k_sym * double(l));

		// calculate symmetry group
		this->createSymmetryGroup();

		this->mapping = v_1d<u64>();
		this->normalisation = v_1d<cpx>();
		this->generate_mapping();
		if (this->N <= 0) {
			stout << "No states in Hilbert space" << EL;
			return;
		}
		this->hamiltonian();
	}

	template<>
	inline ising_sym::IsingModelSym<double>::IsingModelSym(double J, double g, double h, std::shared_ptr<Lattice> lat, int k_sym, bool p_sym, bool x_sym, u32 thread_num)
		: J(J), g(g), h(h)
	{
		this->lattice = lat;
		this->thread_num = thread_num;
		this->Ns = lat->get_Ns();
		symmetries.p_sym = (p_sym) ? 1 : -1;
		symmetries.x_sym = (x_sym) ? 1 : -1;
		symmetries.k_sym = k_sym;
		//change info
		this->info = this->inf();
		symmetries.k_sym = k_sym * TWOPI / double(this->Ns);
		k_sector = valueEqualsPrec(symmetries.k_sym, 0.0, 1e-4) || valueEqualsPrec(symmetries.k_sym, double(PI), 1e-4);
		stout << "\t->Making double\n";
		// printSeparatedP(stout, '\t', 6, true, 4, symmetries.k_sym, k_sym, VEQ(k_sector));

		// precalculate the exponents
		this->k_exponents = v_1d<double>(this->Ns, 0.0);

		for (int l = 0; l < this->Ns; l++) {
			auto val = std::exp(-imn * this->symmetries.k_sym * double(l));
			this->k_exponents[l] = std::real(val);
		}

		// calculate symmetry group
		this->createSymmetryGroup();

		this->mapping = v_1d<u64>();
		this->normalisation = v_1d<double>();
		this->generate_mapping();
		if (this->N <= 0) {
			stout << "No states in Hilbert space" << EL;
			return;
		}
		this->hamiltonian();
	}

	// ---------------------------------------------------------------- SYMMETRY ELEMENTS ----------------------------------------------------------------

	/*
	* @brief creates the symmetry group operators
	*/
	template<typename _type>
	inline void ising_sym::IsingModelSym<_type>::createSymmetryGroup()
	{
		this->symmetry_group = v_1d<function<u64(u64, int)>>();
		this->symmetry_eigval = v_1d<_type>();

		function<u64(u64, int)> e = [](u64 n, int L) {return n; };						// neutral element
		function<u64(u64, int)> T = e;													// in 1st iteration is neutral element
		function<u64(u64, int)> Z = static_cast<u64(*)(u64, int)>(&flip);				// spin flip operator (all spins)
		function<u64(u64, int)> P = reverseBits;										// parity operator


		// loop through all the possible states
		if (this->lattice->get_BC() == 0) {
			for (int k = 0; k < this->Ns; k++) {
				this->symmetry_group.push_back(T);
				this->symmetry_eigval.push_back(this->k_exponents[k]);
				if (this->h == 0) {
					this->symmetry_group.push_back(multiply_operators(Z, T));
					this->symmetry_eigval.push_back(this->k_exponents[k] * double(this->symmetries.x_sym));
				}
				// if parity can be applied
				if (this->k_sector) {
					this->symmetry_group.push_back(multiply_operators(P, T));
					this->symmetry_eigval.push_back(this->k_exponents[k] * double(this->symmetries.p_sym));
					if (this->h == 0) {
						this->symmetry_group.push_back(multiply_operators(multiply_operators(P, Z), T));
						NO_OVERFLOW(this->symmetry_eigval.push_back(this->k_exponents[k] * double(this->symmetries.p_sym * (long)this->symmetries.x_sym));)
					}
				}
				T = multiply_operators(function<u64(u64, int)>(rotateLeft), T);
			}
		}
		else if(this->lattice->get_BC() == 1) {
			// neutral element
			this->symmetry_group.push_back(T);
			this->symmetry_eigval.push_back(1.0);

			// check if spin flip is eligable
			if (this->h == 0) {
				this->symmetry_group.push_back(multiply_operators(Z, T));
				this->symmetry_eigval.push_back(double(this->symmetries.x_sym));
			}

			// if parity can be applied
			this->symmetry_group.push_back(multiply_operators(P, T));
			this->symmetry_eigval.push_back(double(this->symmetries.p_sym));
			// furthermore add the spin flip
			if (this->h == 0) {
				this->symmetry_group.push_back(multiply_operators(multiply_operators(P, Z), T));
				this->symmetry_eigval.push_back(double(this->symmetries.p_sym * (double)this->symmetries.x_sym));
			}
		}
	}

	// ----------------------------------------------------------- 				 BUILDING HAMILTONIAN 				 -----------------------------------------------------------
	
	/*
	* @brief Generates the total Hamiltonian of the system. The diagonal part is straightforward,
	* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
	*/
	template<typename _type>
	inline void ising_sym::IsingModelSym<_type>::hamiltonian()
	{
		this->init_ham_mat();
#pragma omp parallel for num_threads(this->thread_num)
		for (long int k = 0; k < this->N; k++) {
			for (int j = 0; j <= this->Ns - 1; j++) {
				uint nn_number = this->lattice->get_nn_forward_num(j);
				// true - spin up, false - spin down
				double s_i = checkBit(this->mapping[k], this->Ns - 1 - j) ? this->_SPIN : -this->_SPIN;

				// flip with S^x_i with the transverse field
				if (this->g != 0) {
					u64 new_idx = flip(this->mapping[k], this->Ns - 1 - j);
					this->setHamiltonianElem(k, this->g, new_idx);
				}

				// diagonal elements setting the perpendicular field
				this->H(k, k) += this->h * s_i;

				for (auto nn = 0; nn < nn_number; nn++) {
					auto n_num = this->lattice->get_nn_forward_num(j, nn);
					if (auto nei = this->lattice->get_nn(j, n_num); nei >= 0) {
						// Ising-like spin correlation
						double s_j = checkBit(this->mapping[k], this->Ns - 1 - nei) ? this->_SPIN : -this->_SPIN;
						// setting the neighbors elements
						this->H(k, k) += this->J * s_i * s_j;
					}
				}
			}
		}
	}
};