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
		double Delta_a = 0.8;																					// sigma_z*sigma_z nearest neighbors
		double Delta_b = 0.8;																					// sigma_z*sigma_z next nearest neighbors
		double eta_a = 0.5;
		double eta_b = 0.5;

		xyz_sym::sym symmetries;

		// ------------------- TRANSLATION -------------------
		bool k_sector;																							// if the k-sector allows parity symmetry
		v_1d<_type> k_exponents;																				// precalculate the symmetry exponents for current k vector

	public:
		~XYZSym() = default;
		XYZSym() = default;
		XYZSym(std::shared_ptr<Lattice> lat, double Ja, double Jb, double hx, double hz, double Delta_a, double Delta_b, double eta_a, double eta_b, int k_sym = 0, bool p_sym = true, bool x_sym = true, int su_val = -INT_MAX, u32 thread_num = 1);
		XYZSym(std::shared_ptr<Lattice> lat, u32 thread_num = 1, int k_sym = 0, bool p_sym = true, bool x_sym = true, int su_val = -INT_MAX)
			: XYZSym(lat, 1.0, 1.0, 0.2, 0.8, 0.9, 0.9, 0.5, 0.5, k_sym, p_sym, x_sym, su_val, thread_num) {};


		// -------------------------------- GETTERS --------------------------------
		auto get_k_sector()								const RETURNS(this->k_sector);
		auto get_symmetries()							const RETURNS(this->symmetries);
		auto get_k_exp()								const RETURNS(this->k_exponents);

		// -------------------------------- OVERRIDING --------------------------------
		void createSymmetryGroup() override;											// create symmetry group elements and their corresponding eigenvalues
		void hamiltonian() override;													// creates the Hamiltonian itself

		string inf(const v_1d<string>& skip = {}, string sep = "_", int prec = 2) const override
		{
			auto Ns = this->lattice->get_Ns();
			string name = sep + \
				"xyz_sym,Ns=" + STR(Ns) + \
				",Ja=" + STRP(this->Ja, prec) + \
				",Jb=" + STRP(this->Jb, prec) + \
				",hx=" + STRP(this->hx, prec) + \
				",hz=" + STRP(this->hz, prec) + \
				",da=" + STRP(this->Delta_a, prec) + \
				",db=" + STRP(this->Delta_b, prec) + \
				",ea=" + STRP(this->eta_a, prec) + \
				",eb=" + STRP(this->eta_b, prec) + \
				",k=" + STRP(this->symmetries.k_sym, 2) + \
				",p=" + STRP(this->symmetries.p_sym, 2) + \
				",x=" + STRP(this->symmetries.x_sym, 2) + \
				(this->global.su ? (",su=" + STR(this->global.su_val)) : "") + \
				",bc=" + STR(this->lattice->get_BC());
			return this->SpinHamiltonian<_type>::inf(name, skip, sep);
		}
		void update_info() override { this->info = this->inf(); };

	private:
		void init(double Ja, double Jb, double hx, double hz, double Delta_a, double Delta_b, double eta_a, double eta_b) {
			this->Ja = Ja;
			this->Jb = Jb;
			this->hx = hx;
			this->hz = hz;
			this->Delta_a = Delta_a;
			this->Delta_b = Delta_b;
			this->eta_a = eta_a;
			this->eta_b = eta_b;
		}
	};

	/*
	* @brief standard constructor
	*/
	template<typename _type>
	inline xyz_sym::XYZSym<_type>::XYZSym(std::shared_ptr<Lattice> lat, double Ja, double Jb, double hx, double hz, double Delta_a, double Delta_b, double eta_a, double eta_b, int k_sym, bool p_sym, bool x_sym, int su_val , u32 thread_num)
	{
		this->init(Ja, Jb, hx, hz, Delta_a, Delta_b, eta_a, eta_b);
		this->lattice = lat;
		this->thread_num = thread_num;
		this->Ns = lat->get_Ns();

		this->global.set_su(su_val, (this->eta_a == 0.0 && this->eta_b == 0.0), this->Ns);

		this->symmetries.p_sym = (p_sym) ? 1 : -1;
		this->symmetries.x_sym = (x_sym) ? 1 : -1;
		this->symmetries.k_sym = k_sym;

		//change info
		this->info = this->inf();

		// calculate the ks
		//this->symmetries.k_sym = k_sym * TWOPI / double(this->Ns);
		stout << "\t->Making complex\n";

		// precalculate the exponents
		this->k_exponents = v_1d<_type>(this->Ns, 0.0);
		double momentum = k_sym * TWOPI / double(this->Ns);
		this->k_sector = valueEqualsPrec(momentum, 0.0, 1e-6) || valueEqualsPrec(momentum, double(PI), 1e-6);
		for (int l = 0; l < this->Ns; l++)
			this->k_exponents[l] = std::exp(-imn * momentum * double(l));

		// calculate symmetry group
		this->createSymmetryGroup();

		this->mapping = v_1d<u64>();
		this->normalisation = v_1d<cpx>();
		this->generate_mapping();
		if (this->N <= 0) {
			stout << "No states in Hilbert space" << EL;
			return;
		}
		//this->hamiltonian();
	}

	template<>
	inline xyz_sym::XYZSym<double>::XYZSym(std::shared_ptr<Lattice> lat, double Ja, double Jb, double hx, double hz, double Delta_a, double Delta_b, double eta_a, double eta_b, int k_sym, bool p_sym, bool x_sym, int su_val, u32 thread_num)
	{
		this->init(Ja, Jb, hx, hz, Delta_a, Delta_b, eta_a, eta_b);
		this->lattice = lat;
		this->thread_num = thread_num;
		this->Ns = lat->get_Ns();

		this->global.set_su(su_val, (this->eta_a == 0.0 && this->eta_b == 0.0), this->Ns);

		this->symmetries.p_sym = (p_sym) ? 1 : -1;
		this->symmetries.x_sym = (x_sym) ? 1 : -1;
		this->symmetries.k_sym = k_sym;

		//change info
		this->info = this->inf();
		//this->symmetries.k_sym = double(k_sym * TWOPI / double(this->Ns));
		stout << "\t->Making double\n";

		// printSeparatedP(stout, '\t', 6, true, 4, symmetries.k_sym, k_sym, VEQ(k_sector));

		// precalculate the exponents
		this->k_exponents = v_1d<double>(this->Ns, 0.0);
		double momentum = k_sym * TWOPI / double(this->Ns);
		this->k_sector = valueEqualsPrec(momentum, 0.0, 1e-6) || valueEqualsPrec(momentum, double(PI), 1e-6);
		for (int l = 0; l < this->Ns; l++) {
			auto val = std::exp(-imn * momentum * double(l));
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
		//this->hamiltonian();
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
		function<u64(u64, int)> Z = static_cast<u64(*)(u64, int)>(&flipAll);				// spin flip operator (all spins)
		function<u64(u64, int)> P = reverseBits;										// parity operator

		bool su_0 = !this->global.su || (this->global.su && this->global.su_val == this->Ns / 2);
		const bool include_sz_flip = su_0 && valueEqualsPrec(this->hz, 0.0, 1e-9) && valueEqualsPrec(this->hx, 0.0, 1e-9);

		// loop through all the possible states
		if (this->lattice->get_BC() == 0) {
			for (int k = 0; k < this->Ns; k++) {
				this->symmetry_group.push_back(T);
				this->symmetry_eigval.push_back(this->k_exponents[k]);
				if (include_sz_flip) {
					this->symmetry_group.push_back(multiply_operators(Z, T));
					this->symmetry_eigval.push_back(this->k_exponents[k] * double(this->symmetries.x_sym));
				}
				// if parity can be applied
				if (this->k_sector) {
					this->symmetry_group.push_back(multiply_operators(P, T));
					this->symmetry_eigval.push_back(this->k_exponents[k] * double(this->symmetries.p_sym));
					if (include_sz_flip) {
						this->symmetry_group.push_back(multiply_operators(multiply_operators(P, Z), T));
						this->symmetry_eigval.push_back(this->k_exponents[k] * double(this->symmetries.p_sym * this->symmetries.x_sym));
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
			if (include_sz_flip) {
				this->symmetry_group.push_back(multiply_operators(Z, T));
				this->symmetry_eigval.push_back(double(this->symmetries.x_sym));
			}

			// if parity can be applied
			this->symmetry_group.push_back(multiply_operators(P, T));
			this->symmetry_eigval.push_back(double(this->symmetries.p_sym));
			// furthermore add the spin flip
			if (include_sz_flip) {
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

		//#pragma omp parallel for num_threads(this->thread_num)
		for (u64 k = 0; k < this->N; k++) {
			u64 idx = 0;
			cpx val = 0.0;
			for (int j = 0; j <= Ns - 1; j++) {
				const uint nn_number = this->lattice->get_nn_forward_num(j);
				const uint nnn_number = this->lattice->get_nnn_forward_num(j);

				// diagonal elements setting the perpendicular field
				const double perpendicular_val = this->hz;
				if (!valueEqualsPrec(perpendicular_val, 0.0, 1e-9)) {
					std::tie(idx, val) = Operators<cpx>::sigma_z(this->mapping[k], Ns, { j });
					this->setHamiltonianElem(k, perpendicular_val * real(val), idx);
				}
				// flip with S^x_i with the transverse field
				const double transverse_val = this->hx;
				if (!valueEqualsPrec(transverse_val, 0.0, 1e-9)) {
					std::tie(idx, val) = Operators<cpx>::sigma_x(this->mapping[k], Ns, { j });
					this->setHamiltonianElem(k, transverse_val * real(val), idx);
				}

				// -------------- CHECK NN ---------------
				for (auto nn = 0; nn < nn_number; nn++) {
					auto n_num = this->lattice->get_nn_forward_num(j, nn);
					if (auto nei = this->lattice->get_nn(j, n_num); nei >= 0) {
						//stout << VEQ(j) << "\t" << VEQ(nn_number) << "\t" << VEQ(n_num) << "\t" << VEQ(nei) << EL;

						// Ising-like spin correlation
						auto [idx_z, val_z] = Operators<cpx>::sigma_z(this->mapping[k], Ns, { j });
						auto [idx_z2, val_z2] = Operators<cpx>::sigma_z(idx_z, Ns, { nei });
						this->setHamiltonianElem(k, this->Ja * this->Delta_a * real(val_z * val_z2), idx_z2);

						// sigma x
						auto [idx_x, val_x] = Operators<cpx>::sigma_x(this->mapping[k], Ns, { j });
						auto [idx_x2, val_x2] = Operators<cpx>::sigma_x(idx_x, Ns, { nei });
						this->setHamiltonianElem(k, this->Ja * (1.0 - this->eta_a) * real(val_x * val_x2), idx_x2);
						//this->H(idx_x2, k) += this->Ja * (1.0 - this->eta_a) * real(val_x * val_x2);
						//this->setHamiltonianElem(k, this->_SPIN * this->_SPIN * this->Ja * (1.0 - this->eta_a), flip_idx_nn);

						// sigma y
						auto [idx_y, val_y] = Operators<cpx>::sigma_y(this->mapping[k], Ns, { j });
						auto [idx_y2, val_y2] = Operators<cpx>::sigma_y(idx_y, Ns, { nei });
						this->setHamiltonianElem(k, this->Ja * (1.0 + this->eta_a) * real(val_y * val_y2), idx_y2);
						//this->H(idx_y2, k) += this->Ja * (1.0 + this->eta_a) * real(val_y * val_y2);
						//this->setHamiltonianElem(k, -this->Ja * (1.0 + this->eta_a) * s_i * s_j, flip_idx_nn);
					}
				}

				// -------------- CHECK NNN ---------------
				for (auto nnn = 0; nnn < nnn_number; nnn++) {
					auto n_num = this->lattice->get_nnn_forward_num(j, nnn);
					if (auto nei = this->lattice->get_nnn(j, n_num); nei >= 0) {
						//stout << VEQ(j) << "\t" << VEQ(nnn_number) << "\t" << VEQ(n_num) << "\t" << VEQ(nei) << EL;
						// Ising-like spin correlation
						auto [idx_z, val_z] = Operators<cpx>::sigma_z(this->mapping[k], Ns, { j });
						auto [idx_z2, val_z2] = Operators<cpx>::sigma_z(idx_z, Ns, { nei });
						this->setHamiltonianElem(k, this->Jb * this->Delta_b * real(val_z * val_z2), idx_z2);

						// sigma x
						auto [idx_x, val_x] = Operators<cpx>::sigma_x(this->mapping[k], Ns, { j });
						auto [idx_x2, val_x2] = Operators<cpx>::sigma_x(idx_x, Ns, { nei });
						this->setHamiltonianElem(k, this->Jb * (1.0 - this->eta_b) * real(val_x * val_x2), idx_x2);
						//this->H(idx_x2, k) += this->Ja * (1.0 - this->eta_a) * real(val_x * val_x2);
						//this->setHamiltonianElem(k, this->_SPIN * this->_SPIN * this->Ja * (1.0 - this->eta_a), flip_idx_nn);

						// sigma y
						auto [idx_y, val_y] = Operators<cpx>::sigma_y(this->mapping[k], Ns, { j });
						auto [idx_y2, val_y2] = Operators<cpx>::sigma_y(idx_y, Ns, { nei });
						this->setHamiltonianElem(k, this->Jb * (1.0 + this->eta_b) * real(val_y * val_y2), idx_y2);
						//this->H(idx_y2, k) += this->Ja * (1.0 + this->eta_a) * real(val_y * val_y2);
						//this->setHamiltonianElem(k, -this->Ja * (1.0 + this->eta_a) * s_i * s_j, flip_idx_nn);
					}
				}
			}
		}
	}
};