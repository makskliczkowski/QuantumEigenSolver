#pragma once
#ifndef HAMILSYM_H
#include "../../hamil_sym.h"
#endif // !HAMIL_H

#include "../XYZ.h"

namespace xyz_sym {
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
	class XYZSym : public SpinHamiltonianSym<_type> {
	private:
		// MODEL BASED PARAMETERS 
		double Ja = 1.0;																						// nearest neighbors J
		double Jb = 1.0;																						// next nearest neighbors J	
		double hx = 0.2;																						// sigma x field
		double hz = 0.8;																						// sigma z field
		double Delta_a = 0.9;																					// sigma_z*sigma_z nearest neighbors
		double Delta_b = 0.9;																					// sigma_z*sigma_z next nearest neighbors
		double eta_a = 0.5;
		double eta_b = 0.5;

		ising_sym::sym symmetries;

		// ------------------- TRANSLATION -------------------
		bool k_sector;																							// if the k-sector allows parity symmetry
		v_1d<_type> k_exponents;																				// precalculate the symmetry exponents for current k vector

	public:
		~XYZSym() = default;
		XYZSym() = default;
		XYZSym(std::shared_ptr<Lattice> lat, u32 thread_num = 1, int k_sym = 0, bool p_sym = true, bool x_sym = true);
		XYZSym(std::shared_ptr<Lattice> lat, double Ja, double Jb, double hx, double hz, double Delta_a, double Delta_b, double eta_a, double eta_b, int k_sym = 0, bool p_sym = true, bool x_sym = true, u32 thread_num = 1)
			: Ja(Ja), Jb(Jb), hx(hx), hz(hz), Delta_a(Delta_a), Delta_b(Delta_b), eta_a(eta_a), eta_b(eta_b), XYZSym(lat, thread_num, k_sym, p_sym, x_sym) {};


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
				"xyz_sym,Ns=" + STR(Ns) + \
				",Ja=" + STRP(this->Ja, 2) + \
				",Jb=" + STRP(this->Jb, 2) + \
				",hx=" + STRP(this->hx, 2) + \
				",hz=" + STRP(this->hz, 2) + \
				",da=" + STRP(this->Delta_a, 2) + \
				",db=" + STRP(this->Delta_b, 2) + \
				",ea=" + STRP(this->eta_a, 2) + \
				",eb=" + STRP(this->eta_b, 2) + \
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
	inline xyz_sym::XYZSym<_type>::XYZSym(std::shared_ptr<Lattice> lat, u32 thread_num, int k_sym, bool p_sym, bool x_sym)
	{
		this->lattice = lat;
		this->thread_num = thread_num;
		this->Ns = lat->get_Ns();

		this->symmetries.p_sym = (p_sym) ? 1 : -1;
		this->symmetries.x_sym = (x_sym) ? 1 : -1;
		this->symmetries.k_sym = k_sym;

		//change info
		this->info = this->inf();

		// calculate the ks
		this->symmetries.k_sym = k_sym * TWOPI / double(this->Ns);
		this->k_sector = valueEqualsPrec(this->symmetries.k_sym, 0.0, 1e-6) || valueEqualsPrec(this->symmetries.k_sym, double(PI), 1e-6);
		stout << "\t->Making complex\n";

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
	inline xyz_sym::XYZSym<double>::XYZSym(std::shared_ptr<Lattice> lat, u32 thread_num, int k_sym, bool p_sym, bool x_sym)
	{
		this->lattice = lat;
		this->thread_num = thread_num;
		this->Ns = lat->get_Ns();
		this->symmetries.p_sym = (p_sym) ? 1 : -1;
		this->symmetries.x_sym = (x_sym) ? 1 : -1;
		this->symmetries.k_sym = k_sym;

		//change info
		this->info = this->inf();
		this->symmetries.k_sym = double(k_sym * TWOPI / double(this->Ns));
		this->k_sector = valueEqualsPrec(this->symmetries.k_sym, 0.0, 1e-4) || valueEqualsPrec(this->symmetries.k_sym, double(PI), 1e-4);
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
	inline void xyz_sym::XYZSym<_type>::createSymmetryGroup()
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
				if (valueEqualsPrec(this->hz, 0.0, 1e-6)) {
					this->symmetry_group.push_back(multiply_operators(Z, T));
					this->symmetry_eigval.push_back(this->k_exponents[k] * double(this->symmetries.x_sym));
				}
				// if parity can be applied
				if (this->k_sector) {
					this->symmetry_group.push_back(multiply_operators(P, T));
					this->symmetry_eigval.push_back(this->k_exponents[k] * double(this->symmetries.p_sym));
					if (valueEqualsPrec(this->hz, 0.0, 1e-6)) {
						this->symmetry_group.push_back(multiply_operators(multiply_operators(P, Z), T));
						NO_OVERFLOW(this->symmetry_eigval.push_back(this->k_exponents[k] * double(this->symmetries.p_sym * (long)this->symmetries.x_sym));)
					}
				}
				T = multiply_operators(function<u64(u64, int)>(rotateLeft), T);
			}
		}
		else if (this->lattice->get_BC() == 1) {
			// neutral element
			this->symmetry_group.push_back(T);
			this->symmetry_eigval.push_back(1.0);

			// check if spin flip is eligable
			if (valueEqualsPrec(this->hz, 0.0, 1e-6)) {
				this->symmetry_group.push_back(multiply_operators(Z, T));
				this->symmetry_eigval.push_back(double(this->symmetries.x_sym));
			}

			// if parity can be applied
			this->symmetry_group.push_back(multiply_operators(P, T));
			this->symmetry_eigval.push_back(double(this->symmetries.p_sym));
			// furthermore add the spin flip
			if (valueEqualsPrec(this->hz, 0.0, 1e-6)) {
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
	inline void xyz_sym::XYZSym<_type>::hamiltonian()
	{
		this->init_ham_mat();
		auto Ns = this->lattice->get_Ns();
		this->_SPIN = 1.0;
		//#pragma omp parallel for num_threads(this->thread_num)
		for (u64 k = 0; k < this->N; k++) {
			for (int j = 0; j <= Ns - 1; j++) {
				const uint nn_number = this->lattice->get_nn_forward_num(j);
				const uint nnn_number = this->lattice->get_nnn_forward_num(j);
				// true - spin up, false - spin down
				double s_i = checkBit(this->mapping[k], Ns - 1 - j) ? this->_SPIN : -this->_SPIN;

				// diagonal elements setting the perpendicular field
				const double perpendicular_val = this->hz * s_i;
				this->H(k, k) += perpendicular_val;

				// flip with S^x_i with the transverse field
				u64 new_idx = flip(this->mapping[k], Ns - 1 - j);
				if (!valueEqualsPrec(this->hx, 0.0, 1e-6)) {
					this->setHamiltonianElem(k, this->hx * this->_SPIN, new_idx);
				}

				// -------------- CHECK NN ---------------
				for (auto nn = 0; nn < nn_number; nn++) {
					auto n_num = this->lattice->get_nn_forward_num(j, nn);
					if (auto nei = this->lattice->get_nn(j, n_num); nei >= 0) {
						//stout << VEQ(j) << "\t" << VEQ(nn_number) << "\t" << VEQ(n_num) << "\t" << VEQ(nei) << EL;

						// Ising-like spin correlation
						const double s_j = checkBit(this->mapping[k], Ns - 1 - nei) ? this->_SPIN : -this->_SPIN;
						// setting the neighbors elements
						this->H(k, k) += this->Ja * this->Delta_a * s_i * s_j;
						//this->H(k, k) += 9.0 * s_i * s_j;

						u64 flip_idx_nn = flip(new_idx, Ns - 1 - nei);

						// sigma x
						this->setHamiltonianElem(k, this->_SPIN * this->_SPIN * this->Ja * (1.0 - this->eta_a), flip_idx_nn);
						//this->H(flip_idx_nn, k) += 5.0;
						// sigma y
						this->setHamiltonianElem(k, -this->Ja * (1.0 + this->eta_a) * s_i * s_j, flip_idx_nn);
						//this->H(flip_idx_nn, k) -= 15.0 * (s_i) * (s_j);
					}
				}

				// -------------- CHECK NNN ---------------
				for (auto nnn = 0; nnn < nnn_number; nnn++) {
					auto n_num = this->lattice->get_nnn_forward_num(j, nnn);
					if (auto nei = this->lattice->get_nnn(j, n_num); nei >= 0) {
						//stout << VEQ(j) << "\t" << VEQ(nnn_number) << "\t" << VEQ(n_num) << "\t" << VEQ(nei) << EL;

						// Ising-like spin correlation
						const double s_j = checkBit(this->mapping[k], Ns - 1 - nei) ? this->_SPIN : -this->_SPIN;
						// setting the neighbors elements
						this->H(k, k) += this->Jb * this->Delta_b * s_i * s_j;
						//this->H(k, k) += 9.0 * s_i * s_j;

						u64 flip_idx_nn = flip(new_idx, Ns - 1 - nei);

						// sigma x
						this->setHamiltonianElem(k, this->_SPIN * this->_SPIN * this->Jb * (1.0 - this->eta_b), flip_idx_nn);
						//this->H(flip_idx_nn, k) += 5.0;
						// sigma y
						this->setHamiltonianElem(k, -this->Jb * (1.0 + this->eta_b) * s_i * s_j, flip_idx_nn);
						//this->H(flip_idx_nn, k) -= 15.0 * (s_i) * (s_j);
					}
				}
			}
		}
	}
};