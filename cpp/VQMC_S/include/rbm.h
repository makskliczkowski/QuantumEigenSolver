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

private:

	avOperators op;

	// optimizer
	std::unique_ptr<Adam<_type>> adam;                          // use the Adam optimizer for GD
	std::unique_ptr<RMSprop_mod<_type>> rms;                    // use the RMS optimizer for GD

	// saved training parameters
	Col<double> tmp_vector;                                     // tmp state vector during the simulation
	v_1d<Col<double>> tmp_vectors;                              // tmp vectors for omp 
	map<u64, _type> mostCommonStates;                           // save most common states energy to save the time


	// ------------------------------------------- 				 GETTERS				  ------------------------------------------
	auto get_op_av()                                                    const RETURNS(this->op);
	int get_vec_id() {
#ifndef DEBUG
		if (this->thread_num == 1)
			return 0;
		else
			return omp_get_thread_num() % this->thread_num;
#else
		return 0;
#endif
	}

	// ------------------------------------------- 				 INITIALIZERS				  ------------------------------------------

	// initialize all
	void initAv();
	// ------------------------------------------- 				 AMPLITUDES AND ANSTATZ REPRESENTATION				  -------------------------------------------

	// get probability ratio for a reference state v1 and v2 state
	_type pRatio(int tn = 1) const;
	_type pRatio(int flip_place, double flipped_spin) const;


	// get local energies
	void locEnKernel(uint start, uint stop, Col<_type>& energies, uint tid);
	_type locEnKernelAll();
	_type locEn();
	_type pRatioValChange(_type v, u64 state, uint vid);

	// ------------------------------------------- 				 SAMPLING				  -------------------------------------------

	// sample block
	void blockSampling(uint b_size, u64 start_stae, uint n_flips = 1, bool thermalize = true);

	// sample the probabilistic space
	Col<_type> mcSampling(uint n_samples, uint n_blocks, uint n_therm, uint b_size, uint n_flips = 1, std::string dir = "", bool save_weights = false);


	// average collection
	void collectAv(_type loc_en);
	map<u64, _type> avSampling(uint n_samples, uint n_blocks, uint n_therm, uint b_size, uint n_flips = 1);

};
//
//
//// ------------------------------------------------- 				 pValues				  -------------------------------------------------
//
/*
* @brief computes Log Psi'/Psi, where Psi' is the state with certain flipped spins - only one working for now
* Look-up tables are used for speed; the vector flips tells us which are flipped
*/
template<typename _type, typename _hamtype>
inline _type rbmState<_type, _hamtype>::pRatio(int flip_place, double flipped_spin) const
{

}
//
///*
//* @brief computes Log Psi'/Psi, where Psi' is the state with certain flipped spins - only one working for now
//*/
//template<typename _type, typename _hamtype>
//inline _type rbmState<_type, _hamtype>::pRatio(int tn) const
//{
//	return exp(dotm(this->b_v, Col<double>(this->tmp_vector - this->current_vector), tn) + sum(log(Fs(this->tmp_vector) / Fs(this->current_vector))));
//};
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

//// ------------------------------------------------- LOCAL ENERGY AND OPERATORS ------------------------------------------------------
//
//
///*
//* @brief probability ratio change due to the state change
//* @param v value of the ratio
//* @param state the state that we change onto
//*/
//template<typename _type, typename _hamtype>
//inline _type rbmState<_type, _hamtype>::pRatioValChange(_type v, u64 state, uint vid)
//{
//	INT_TO_BASE_BIT(state, this->tmp_vectors[vid]);
//	this->tmp_vectors[vid] *= operators::_SPIN_RBM;
//#ifndef RBM_ANGLES_UPD
//	return v * this->pRatio(this->current_vector, this->tmp_vectors[vid]);
//#else
//	return v * this->pRatio(this->tmp_vectors[vid]);
//#endif
//}
//
///*
//* @brief Calculate the local energy depending on the given Hamiltonian - kernel with OpenMP
//*/
//template<typename _type, typename _hamtype>
//inline _type rbmState<_type, _hamtype>::locEnKernelAll()
//{
//	const auto hilb = this->hamil->get_hilbert_size();
//	_type energy = 0;
//
//	// define the lambda function
//	std::function<cpx(int, double)> f1 = [&](int flip_place, double flip_spin) {return static_cast<cpx>(this->pRatio(flip_place, flip_spin)); };
//	std::function<cpx(const vec&)> f2 = [&](const vec& v)
//	{
//#ifndef RBM_ANGLES_UPD
//		return static_cast<cpx>(this->pRatio(this->current_vector, v));
//#else
//		return static_cast<cpx>(this->pRatio(v));
//#endif
//	};
//
//	// loop over all lattice sites
//#ifndef DEBUG
////#pragma omp parallel for reduction(+ : energy) num_threads(this->thread_num)
//#endif
//	for (uint i = 0; i < this->n_visible; i++) {
//		int vid = this->get_vec_id();
//		energy += this->hamil->locEnergy(this->current_state, i, f1, f2, this->tmp_vectors[vid]);
//	}
//
//	return energy;
//}
//
///*
//* @brief Calculate the local energy depending on the given Hamiltonian - kernel of the function for multithreading purposes
//* @param start start of the subrange
//* @param end end of the subrange
//* @param energies energies column to save onto
//* @param tid thread id
//*/
//template<typename _type, typename _hamtype>
//inline void rbmState<_type, _hamtype>::locEnKernel(uint start, uint stop, Col<_type>& energies, uint tid)
//{
//	_type energy = 0;
//	std::function<cpx(int, double)> f1 = [&](int flip_place, double flip_spin) {return static_cast<cpx>(this->pRatio(flip_place, flip_spin)); };
//	std::function<cpx(const vec&)> f2 = [&](vec v)
//	{
//#ifndef RBM_ANGLES_UPD
//		return static_cast<cpx>(this->pRatio(this->current_vector, v));
//#else
//		return static_cast<cpx>(this->pRatio(v));
//#endif
//	};
//	// loop over all lattice sites
//	for (uint i = start; i < stop; i++)
//		energy += this->hamil->locEnergy(this->current_state, i, f1, f2, this->tmp_vectors[this->get_vec_id()]);
//
//	energies[tid] = energy;
//}
//
///*
//* @brief Calculate the local energy depending on the given Hamiltonian
//*/
//template<typename _type, typename _hamtype>
//inline _type rbmState<_type, _hamtype>::locEn() {
//	auto loc_en_time = std::chrono::high_resolution_clock::now();
//	uint Ns = this->hamil->lattice->get_Ns();
//	// get thread id
//#ifndef DEBUG
//	int num_of_threads = (Ns < 50) ? 1 : std::min(this->thread_num, Ns);
//#else
//	int num_of_threads = 1;
//#endif
//
//	if (num_of_threads == 1)
//		// omp version of local energy
//		return this->locEnKernelAll();
//	else {
//		// Threaded
//		v_1d<std::thread> threads;
//		// reserve threads
//		threads.reserve(num_of_threads);
//		Col<_type> energies(num_of_threads, arma::fill::zeros);
//
//		for (int t = 0; t < num_of_threads; t++) {
//			auto start = (Ns / (double)num_of_threads * t);
//			auto stop = ((t + 1) == num_of_threads ? Ns : (Ns / (double)num_of_threads * (double)(t + 1)));
//			threads.emplace_back(&rbmState::locEnKernel, this, start, stop, ref(energies), t);
//		}
//
//		// join the threads together
//		for (auto& t : threads)
//			t.join();
//
//		PRT(loc_en_time, this->dbg_lcen);
//		return arma::mean(energies);
//	}
//}
//
//// ------------------------------------------------- SAMPLING -------------------------------------------------
//
///*
//* @brief block updates the current state according to Metropolis-Hastings algorithm
//* @param b_size the size of the correlation block
//* @param start_state the state to start from
//* @param n_flips number of flips at the single step
//*/
//template<typename _type, typename _hamtype>
//void rbmState<_type, _hamtype>::blockSampling(uint b_size, u64 start_state, uint n_flips, bool thermalize) {
//	if (start_state != this->current_state)
//		this->set_state(start_state, thermalize);
//
//	// set the tmp_vector to current state
//	this->tmp_vector = this->current_vector;
//
//	for (auto i = 0; i < b_size; i++) {
//
//		const int flip_place = this->hamil->ran.randomInt_uni(0, this->n_visible);
//		const double flip_spin = this->tmp_vector(flip_place);
//
//		flipV(this->tmp_vector, flip_place);
//
//#ifndef RBM_ANGLES_UPD
//		double proba = abs(this->pRatio(this->current_vector, this->tmp_vector, this->thread_num));
//#else
//		double proba = abs(this->pRatio(flip_place, flip_spin));
//
//#endif
//
//		if (this->hamil->ran.randomReal_uni() <= proba * proba) {
//			// update current state and vector
//			this->current_vector(flip_place) = this->tmp_vector(flip_place);
//			// update angles if needed
//#ifdef RBM_ANGLES_UPD
//			this->update_angles(flip_place, flip_spin);
//#endif
//		}
//		else {
//			// set the vector back to normal
//			this->tmp_vector(flip_place) = flip_spin;
//		}
//	}
//	this->current_state = BASE_TO_INT(vec(this->current_vector / operators::_SPIN_RBM));
//	// calculate the effective angles
//}
//
///*
//* @brief sample the vectors and converge model with gradients to find ground state
//* @param n_samples number of samples to be used for training
//* @param n_blocks number of correlation-reducers blocks
//* @param n_therm number of steps to leave for thermalization
//* @param b_size size of correlation-reducers blocks
//* @param n_flips number of flips during a single step
//* @returns energies obtained during each Monte Carlo step
//*/
//template<typename _type, typename _hamtype>
//Col<_type> rbmState<_type, _hamtype>::mcSampling(uint n_samples, uint n_blocks, uint n_therm, uint b_size, uint n_flips, std::string dir, bool save_weights) {
//#ifdef S_REGULAR
//	this->current_b_reg = this->b_reg_mult;
//#endif
//
//	// start the timer!
//	auto start = std::chrono::high_resolution_clock::now();
//	// make the pbar!
//	this->pbar = pBar(50, n_samples);
//
//	// check if the batch is not bigger than the blocks number
//	const auto norm = n_blocks - n_therm;
//
//	// save all average weights for covariance matrix
//	Col<_type> meanEnergies(n_samples, arma::fill::zeros);
//	Col<_type> energies(norm, arma::fill::zeros);
//	Mat<_type> derivatives(norm, this->full_size, arma::fill::zeros);
//
//	this->set_rand_state();
//	for (auto i = 0; i < n_samples; i++) {
//		// set the random state at each Monte Carlo iteration
//		//this->set_rand_state();
//
//		this->set_angles(this->current_vector);
//
//		// thermalize system
//		auto therm_time = std::chrono::high_resolution_clock::now();
//		this->blockSampling(n_therm * b_size, this->current_state, n_flips);
//		PRT(therm_time, this->dbg_thrm);
//
//		// to check whether the batch is ready already  
//		auto blocks_time = std::chrono::high_resolution_clock::now();
//		for (auto took = 0; took < norm; took++) {
//			// block sample the stuff
//			auto sample_time = std::chrono::high_resolution_clock::now();
//			this->blockSampling(b_size, this->current_state, n_flips);
//			PRT(sample_time, this->dbg_samp);
//
//			this->calcVarDeriv(this->current_vector);
//
//			// append local energies
//			energies(took) = this->locEn();
//#ifdef USE_SR
//			derivatives.row(took) = this->O_flat.st();
//#endif          
//		}
//		PRT(blocks_time, this->dbg_blck);
//
//		auto meanLocEn = arma::mean(energies);
//		this->F = arma::cov(conj(derivatives), energies);
//#ifdef USE_SR
//		this->S = arma::cov(conj(derivatives), derivatives);
//		// update model
//		this->updVarDerivSR(i);
//
//#ifdef S_REGULAR
//		this->current_b_reg = this->current_b_reg * b_reg;
//#endif // S_REGULAR
//#elif defined USE_ADAM
//		this->adam->update(this->F);
//		this->F = this->adam->get_grad();
//#elif defined USE_RMS
//		this->rms->update(this->F, averageWeights);
//		this->F = this->rms->get_grad();
//#else
//		// standard updater
//		this->F = this->lr * this->F;
//#endif
//		this->set_weights();
//		// add energy
//		meanEnergies(i) = meanLocEn;
//
//		// update the progress bar
//		if (i % pbar.percentageSteps == 0) {
//			pbar.printWithTime("-> PROGRESS");
//			// saving the weights, thus the progress
//			if (save_weights) {
//				this->W.save(arma::hdf5_name(dir + "W.h5", "interaction"));
//				this->b_h.save(arma::hdf5_name(dir + "bh.h5", "hidden"));
//				this->b_v.save(arma::hdf5_name(dir + "bv.h5", "visible"));
//			}
//		}
//	}
//	return meanEnergies;
//}
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