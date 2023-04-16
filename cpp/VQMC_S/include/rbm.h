#pragma once
#ifndef RBM_H
#define RBM_H

#define USE_SR
//#define USE_ADAM
//#define USE_RMS

#define RBM_ANGLES_UPD
//#define PLOT

//#define DEBUG
#ifdef USE_SR
#define PINV
#define S_REGULAR
#endif

//#ifndef XYZ_H
//#include "models/XYZ.h"
//#endif // !TESTER_H
//#ifndef HEISENBERG_DOTS
//#include "models/heisenberg_dots.h"
//#endif
//#ifndef HEISENBERG_KITAEV
//#include "models/heisenberg-kitaev.h"
//#endif
//#ifndef ISINGMODEL
//#include "models/ising.h"
//#endif








template <typename _type, typename _hamtype>
class rbmState {
	// ------------------------------------------- 				 GETTERS				  ------------------------------------------
	//auto get_op_av()                                                    const RETURNS(this->op);


	// ------------------------------------------- 				 INITIALIZERS				  ------------------------------------------

	// initialize all
	//void initAv();

	// ------------------------------------------- 				 SAMPLING				  -------------------------------------------

	// average collection
	//void collectAv(_type loc_en);
	//map<u64, _type> avSampling(uint n_samples, uint n_blocks, uint n_therm, uint b_size, uint n_flips = 1);

};
//
//

//// ------------------------------------------------- 				 PRINTERS				  -------------------------------------------------
///*
//* @brief Pretty prints the state given the sample_states map
//* @param sample_states the map from u64 state integer to value at the state
//* @param tol tolerance on the absolute value
//*/
//template<typename _type, typename _hamtype>
//inline void rbmState<_type, _hamtype>::pretty_print(std::map<u64, _type>& sample_states, double tol) const {
//	v_1d<int> tmp(this->n_visible);
//	double norm = 0;
//	double phase = 0;
//	// normalise
//	for (const auto& [state, value] : sample_states) {
//		norm += std::pow(abs(value), 2.0);
//		phase = std::arg(value);
//	}
//
//	for (const auto& [state, value] : sample_states) {
//		auto val = value / sqrt(norm) * std::exp(-imn * phase);
//		sample_states[state] = val;
//		SpinHamiltonian<_type>::print_base_state(state, val, tmp, tol);
//	}
//	stout << EL;
//
//}
//

///*
//* @brief initialize operators average to be saved
//*/
//template<typename _type, typename _hamtype>
//inline void rbmState<_type, _hamtype>::initAv()
//{
//	auto Lx = this->hamil->lattice->get_Lx();
//	auto Ly = this->hamil->lattice->get_Ly();
//	auto Lz = this->hamil->lattice->get_Lz();
//	auto Ns = this->hamil->lattice->get_Ns();
//	this->op = avOperators(Lx, Ly, Lz, Ns, this->hamil->lattice->get_type());
//}
//

//// ------------------------------------------------- SAMPLING -------------------------------------------------
//

//
///*
//* @brief sample the vectors and find the ground state and the operators
//* @param n_samples number of samples to be used for training
//* @param n_therm number of steps to leave for thermalization
//* @param b_size size of correlation-reducers blocks
//* @param n_flips number of flips during a single step
//* @returns map of base state to the value of coefficient next to it
//*/
//template<typename _type, typename _hamtype>
//inline std::map<u64, _type> rbmState<_type, _hamtype>::avSampling(uint n_samples, uint n_blocks, uint n_therm, uint b_size, uint n_flips)
//{
//	stout << "\n\n\n->Looking for the ground state for " + this->get_info() << "," + VEQ(n_samples) + "," + VEQ(n_blocks) + "," + VEQ(b_size) << EL;
//	// start the timer!
//	auto start = std::chrono::high_resolution_clock::now();
//
//	// initialize averages
//	this->initAv();
//
//	// make the pbar!
//	this->pbar = pBar(50, n_samples);
//
//	// states to be returned
//	std::map<u64, _type> states;
//
//	auto Ns = this->hamil->lattice->get_Ns();
//
//	this->op.reset();
//	for (auto r = 0; r < n_samples; r++) {
//		// set the random state at each Monte Carlo iteration
//		// this->set_rand_state();
//
//		// thermalize system
//		this->blockSampling(n_therm * b_size, this->current_state, n_flips);
//		for (int i = 0; i < n_blocks; i++) {
//
//			// block sample the stuff
//			this->blockSampling(b_size, this->current_state, n_flips);
//
//			// look at the states coefficient (not found)
//
//			//if (auto coefficient = this->coeff(this->current_vector); !valueEqualsPrec(std::abs(coefficient), 0.0, 1e-3)) {
//			//	states[this->current_state] = coefficient;
//			//}
//
//			// append local energies
//			this->collectAv(this->locEn());
//		}
//		// update the progress bar
//		if (r % pbar.percentageSteps == 0)
//			pbar.printWithTime("-> PROGRESS");
//	}
//
//	this->op.normalise(n_samples * n_blocks, this->hamil->lattice->get_spatial_norm());
//
//	stouts("->Finished Monte Carlo state search after finding weights ", start);
//	stout << "\n------------------------------------------------------------------------" << EL;
//	stout << "GROUND STATE RBM ENERGY: " << VEQP(op.en, 4) << EL;
//	stout << "GROUND STATE RBM SIGMA_X EXTENSIVE: " << VEQP(op.s_x, 4) << EL;
//	stout << "GROUND STATE RBM SIGMA_Z EXTENSIVE: " << VEQP(op.s_z, 4) << EL;
//	stout << "\n------------------------------------------------------------------------\n|Psi>=" << EL;
//	this->pretty_print(states, 0.08);
//	stout << "\n------------------------------------------------------------------------" << EL;
//
//	return states;
//}
//
///*
//*
//*/
//template<typename _type, typename _hamtype>
//inline void rbmState<_type, _hamtype>::collectAv(_type loc_en)
//{
//	auto Ns = this->hamil->lattice->get_Ns();
//
//	// calculate sigma_z 
//	double s_z = 0.0;
//	double s_z_nei = 0.0;
////#pragma omp parallel for reduction(+ : s_z, s_z_nei) num_threads(this->thread_num)
//	for (int i = 0; i < Ns; i++) {
//		const auto& [state, val] = Operators<double>::sigma_z(this->current_state, Ns, v_1d<int>({ i }));
//		this->op.s_z_i(i) += real(val);
//		s_z += real(val);
//		for (int j = 0; j < Ns; j++) {
//			const auto& [state, val] = Operators<double>::sigma_z(this->current_state, Ns, v_1d<int>({ i, j }));
//			this->op.s_z_cor(i, j) += std::real(val);
//		}
//		auto nei = this->hamil->lattice->get_z_nn(i);
//		if (nei >= 0) {
//			const auto& [state_n, val_n] = Operators<double>::sigma_z(this->current_state, Ns, v_1d<int>({ i, nei }));
//			s_z_nei += real(val_n);
//		}
//	}
//	this->op.s_z += real(s_z / double(Ns));
//	this->op.s_z_nei += real(s_z_nei / double(Ns));
//
//	// calculate sigma_y
//	cpx s_y_nei = 0.0;
////#pragma omp parallel for reduction(+ : s_y_nei) num_threads(this->thread_num)
//	for (int i = 0; i < Ns; i++) {
//		const auto& [state, val] = Operators<double>::sigma_y(this->current_state, Ns, v_1d<int>({ i, this->hamil->lattice->get_y_nn(i) }));
//		const int vid = this->get_vec_id();
//
//		auto nei = this->hamil->lattice->get_y_nn(i);
//		if(nei >= 0)
//			s_y_nei += this->pRatioValChange(val, state, vid);
//	}
//	this->op.s_y_nei += real(s_y_nei / double(Ns));
//
//	// calculate sigma_x
//	cpx s_x = 0.0;
//	cpx s_x_nei = 0.0;
////#pragma omp parallel for reduction(+ : s_x, s_x_nei) num_threads(this->thread_num)
//	for (int i = 0; i < Ns; i++) {
//		const auto& [state, val] = Operators<double>::sigma_x(this->current_state, Ns, v_1d<int>({ i }));
//		const int vid = this->get_vec_id();
//
//		s_x += this->pRatioValChange(val, state, vid);
//		for (int j = 0; j < Ns; j++) {
//			const auto& [state, val] = Operators<double>::sigma_x(this->current_state, Ns, v_1d<int>({ i, j }));
//			this->op.s_x_cor(i, j) += std::real(this->pRatioValChange(val, state, vid));
//		}
//		auto nei = this->hamil->lattice->get_x_nn(i);
//		if (nei >= 0) {
//			const auto& [state_n, val_n] = Operators<double>::sigma_x(this->current_state, Ns, v_1d<int>({ i, this->hamil->lattice->get_x_nn(i) }));
//			s_x_nei += this->pRatioValChange(val_n, state_n, vid);
//		}
//	}
//	this->op.s_x_nei += real(s_x_nei / double(Ns));
//	this->op.s_x += real(s_x / double(Ns));
//	// local energy
//	this->op.en += loc_en;
//}
//
//
#endif // !RBM_H