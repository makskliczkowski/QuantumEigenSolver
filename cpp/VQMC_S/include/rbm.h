#pragma once
#ifndef RBM_H
#define RBM_H

#include "../src/progress.h"

#ifndef HAMIL_H
#include "hamil.h"
#endif

#ifndef ADAM_H
#include "../include/ml.h"
#endif // !ADAM_H




#ifndef USE_ADAM
#define USE_ADAM
#endif // !USE_ADAM

#ifdef DONT_USE_ADAM
#undef USE_ADAM
#endif // !DONT_USE_ADAM


template <typename _type>
class rbmState{

private:
    // general parameters
    std::string info;                                           // info about the model
    size_t batch;                                               // batch size for stochastic
    size_t n_visible;                                           // visible neurons
    size_t n_hidden;                                            // hidden neurons
    size_t full_size;                                           // full size of the parameters
    size_t hilbert_size;                                        // hilbert space size
    double lr;                                                  // learning rate
    pBar pbar;                                                  // progress bar
    
    // network weights
    Mat<_type> W;                                               // weight matrix
    arma::Col<_type> b_v;                                       // visible bias
    arma::Col<_type> b_h;                                       // hidden bias

    // variational derivatives                                  
    arma::Col<_type> O_b_v;                                     // gradient of the visible bias
    arma::Col<_type> O_b_h;                                     // gradient of the hidden bias
    arma::Mat<_type> O_w;                                       // gradient of the weight matrices
    arma::Col<_type> O_flat;                                    // flattened output for easier calculation of the covariance
    arma::Mat<_type> S;                                         // positive semi-definite covariance matrix
    arma::Col<_type> F;                                         // forces
    
    // the Hamiltonian
    std::shared_ptr<SpinHamiltonian<_type>> hamil;              // unique ptr to a general Hamiltonian of the spin system

    // optimizer
    std::unique_ptr<Adam<_type>> adam;                          // use the Adam optimizer for GD

    // saved training parameters
    u64 current_state;                                          // current state during the simulation
    arma::Col<double> current_vector;                           // current state vector during the simulation
    arma::Col<double> tmp_vector;                               // tmp state vector during the simulation

public:
    ~rbmState() = default;
    rbmState() = default;
    rbmState(size_t nH, size_t nV, std::shared_ptr<SpinHamiltonian<_type>> const & hamiltonian,
            double lr, size_t batch
            ) 
            : n_hidden(nH), n_visible(nV)
            , lr(lr)
            , batch(batch)
            {
                //omp_set_num_threads(16);                        // Use threads for all consecutive parallel regions
                this->hamil = hamiltonian;
                this->hilbert_size = hamil->get_hilbert_size();
                stout << VEQ(this->hilbert_size) << EL;
                this->full_size = this->n_hidden + this->n_visible + this->n_hidden * this->n_visible;
                this->adam = std::make_unique<Adam<_type>>(this->lr, this->full_size);
                this->set_info();
                this->allocate();
                this->init();
            };
    // ------------------------------------------- HELPERS -------------------------------------------
    // pretty print the state sampled
    void pretty_print(const std::map<u64, _type>& sample_states, double tol = 5e-2) const;

    // ------------------------------------------- SETTTERS
    // sets info
    void set_info()                                                     { this->info = VEQ(n_visible) + "," + VEQ(n_hidden) + "," + VEQ(batch); };
    // sets the current state
    void set_state(u64 state) {
        this->current_state = state;
        //intToBase(state, this->current_vector, BinaryPowers);
        intToBase(state, this->current_vector, 2);
    }
    // set the current state to random
    void set_rand_state()                                               { this->set_state(this->hamil->ran.randomInt_uni(0, this->hilbert_size - 1)); };
    
    // set weights
    void set_weights(const arma::Col<_type>& gradient);

    // ------------------------------------------- GETTERS ------------------------------------------
    auto get_info()                                                     const RETURNS(this->info);

    // ------------------------------------------- INITIALIZERS
    // allocate the memory for the biases and weights
    void allocate();
    // initialize all
    void init();
    // ------------------------------------------- AMPLITUDES AND ANSTATZ REPRESENTATION -------------------------------------------

    // the hiperbolic cosine of the parameters
    arma::Col<_type> Fs(const Col<double>& v)                           const { return arma::cosh(this->b_h + this->W * v); };

    // get the current amplitude given vector
    auto coeff(const Col<double>& v)                                    const { return (exp(cdotm(this->b_v, v)) * arma::prod(Fs(v)) * std::pow(2.0, this->n_hidden)); };

    // get probability ratio for a reference state v1 and v2 state
    _type pRatio()                                                      const { return exp(cdotm(this->b_v, arma::Col<double>(this->tmp_vector - this->current_vector)) + sum(log(Fs(this->tmp_vector) / Fs(this->current_vector)))); };
    _type pRatio(const Col<double>& v1, const Col<double>& v2)          const { return exp(cdotm(this->b_v, arma::Col<double>(v2 - v1)) + arma::sum(arma::log(Fs(v2) / Fs(v1)))); };

    // get local energies
    _type locEn();

    // variational derivative calculation
    void calcVarDeriv(const Col<double>& v);

    // update weights after gradient descent
    void updVarDeriv();

    // sample block
    void blockSampling(size_t b_size, u64 start_stae, size_t n_flips = 1);

    // sample the probabilistic space
    v_1d<_type> mcSampling(size_t n_samples, size_t n_blocks, size_t n_therm, size_t b_size, size_t n_flips = 1);
    
};

/*
* Pretty prints the state given the sample_states map
* @param sample_states the map from u64 state in nteger
*/
template<typename _type>
inline void rbmState<_type>::pretty_print(const std::map<u64, _type>& sample_states, double tol) const {
    v_1d<int> tmp(this->n_visible);
    double norm = 0;
    double phase = 0;
    // normalise
    for (const auto& [state, value] : sample_states) {
        norm += std::pow(abs(value), 2.0);
        phase = std::arg(value);
    }

    for (const auto& [state, value] : sample_states) {
        auto val = value / sqrt(norm) * std::exp(-imn * phase);
        SpinHamiltonian<_type>::print_base_state(state, val, tmp, tol);
    }
    stout << EL;
}

/*
* 
*/
template<typename _type>
void rbmState<_type>::allocate() {
    // initialize biases
    this->b_v = arma::Col<_type>(this->n_visible, arma::fill::zeros);
    this->O_b_v = arma::Col<_type>(this->n_visible, arma::fill::zeros);
    this->b_h = arma::Col<_type>(this->n_hidden, arma::fill::zeros);
    this->O_b_h = arma::Col<_type>(this->n_hidden, arma::fill::zeros);
    this->W = arma::Mat<_type>(this->n_hidden, this->n_visible, arma::fill::zeros);
    this->O_w = arma::Mat<_type>(this->n_hidden, this->n_visible, arma::fill::zeros);
    this->O_flat = arma::Col<_type>(this->full_size, arma::fill::zeros);
    // allocate covariance and forces
    this->F = arma::Col<_type>(this->full_size, arma::fill::zeros);
    this->S = arma::Mat<_type>(this->full_size, this->full_size, arma::fill::zeros);
    // allocate vectors
    this->current_vector = Col<double>(this->n_visible, arma::fill::zeros);
    this->tmp_vector = Col<double>(this->n_visible, arma::fill::zeros);
}

/*
* Initialize the weights 
*/
template<typename T>
void rbmState<T>::init() {
    // initialize random state
    this->set_rand_state();

    // initialize biases visible
    for (int i = 0; i < this->n_visible; i++)
        this->b_v(i) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const);
    // hidden
    for (int i = 0; i < this->n_hidden; i++)
        this->b_h(i) = this->hamil->ran.xavier_uni(this->n_hidden, 1, this->xavier_const);
    // matrix
    for (int i = 0; i < this->W.n_rows; i++)
        for(int j = 0; j < this->W.n_cols; j++)
            this->W(i,j) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const);
}

template<>
inline void rbmState<cpx>::init() {
    // initialize random state
    this->set_rand_state();

    // initialize biases visible
    for (int i = 0; i < this->n_visible; i++)
        //this->b_v(i) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const) + imn * this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const);
        this->b_v(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.random_real_normal()) / 10.0;
    // hidden
    for (int i = 0; i < this->n_hidden; i++)
        //this->b_h(i) = this->hamil->ran.xavier_uni(this->n_hidden, 1, this->xavier_const) + imn * this->hamil->ran.xavier_uni(this->n_hidden, 1, this->xavier_const);
        this->b_h(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.randomReal_uni()) / 10.0;
    // matrix
    for (int i = 0; i < this->W.n_rows; i++)
        for (int j = 0; j < this->W.n_cols; j++)
            //this->W(i, j) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const) + imn * this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const);
            this->W(i, j) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.random_real_normal()) / 10.0;

    // calculate the flattened part
    // this->O_flat.subvec(0, this->n_visible - 1) = this->O_b_v;
    // this->O_b_v.print("\n");
    // this->O_flat.print("\n");
    // this->O_flat.subvec(this->n_visible, this->n_visible + this->n_hidden - 1) = this->O_b_h;
    // this->O_flat.subvec(this->n_visible + this->n_hidden,
    //     this->full_size - 1) = this->O_w.as_col();
}

/*
* sets the weights according to the gradient descent
*/
template<typename T>
void rbmState<T>::set_weights(const arma::Col<T>& gradient) {
    // update weights accordingly
#pragma omp parallel for
    for (auto i = 0; i < this->n_visible; i++)
        this->b_v(i) -= gradient(i);
        //this->b_v -= gradient.subvec(0, this->n_visible - 1);
#pragma omp parallel for
    for (auto i = 0; i < this->n_hidden; i++) {
        const auto elem = i + this->n_visible;
        this->b_h(i) -= gradient(elem);
    }
#pragma omp parallel for
    for (auto i = 0; i < this->n_hidden; i++) {
        for (auto j = 0; j < this->n_visible; j++) {
            const auto elem = (this->n_visible + this->n_hidden) + i + j * this->n_hidden;
            this->W(i, j) -= gradient(elem);
        }
    }

    //this->b_h -= gradient.subvec(this->n_visible, this->n_visible + this->n_hidden - 1);
    //this->W -= arma::reshape(
    //    gradient.subvec(this->n_visible + this->n_hidden,
    //    this->full_size - 1),
    //    this->W.n_rows,
    //    this->W.n_cols
    //);
}


/*
* @param v the base vector we want to calculate derivatives from
*/
template<typename _type>
void rbmState<_type>::calcVarDeriv(const Col<double>& v){
    // change of the first bias
    // change of the hidden bias
    this->O_b_h = arma::tanh(this->b_h + this->W * v);
    //this->O_b_v = v;
    // change of the weights matrix
    this->O_w = this->O_b_h * v.t();
    // calculate the flattened part
#pragma omp parallel for
    for (auto i = 0; i < this->n_visible; i++)
        this->O_flat(i) = v(i);
#pragma omp parallel for
    for (auto i = 0; i < this->n_hidden; i++) {
        const auto elem = i + this->n_visible;
        this->O_flat(elem) = this->O_b_h(i);
    }
#pragma omp parallel for
    for (auto i = 0; i < this->n_hidden; i++) {
        for (auto j = 0; j < this->n_visible; j++) {
            const auto elem = (this->n_visible + this->n_hidden) + i + j * this->n_hidden;
            this->O_flat(elem) = this->O_w(i, j);
        }
    }
    //this->O_flat.subvec(0, this->n_visible - 1) = arma::conv_to<arma::Col<_type>>::from(v);
    //this->O_flat.subvec(this->n_visible, n_visible + this->n_hidden - 1) = this->O_b_h;
    //this->O_flat.subvec(this->n_visible + this->n_hidden,
    //                    this->full_size -1) = arma::vectorise(this->O_b_h * v.t());
}

/*
* updates the weights using stochastic gradient descent
*/
template<typename T>
void rbmState<T>::updVarDeriv(){
    // greadient descent and psudo inversion
    double tol = 1e-6;
    auto Sinv = arma::pinv(this->S, tol);

    // update flat vector
#ifdef USE_ADAM
    this->set_weights(this->adam->get_grad());
    this->adam->update(Sinv * this->F);
#else 
    this->set_weights(this->lr * Sinv * this->F);
#endif // USE_ADAM

}

/*
* Calculate the local energy depending on the given Hamiltonian
*/
template<typename T>
inline T rbmState<T>::locEn(){
    v_1d<std::tuple<u64, T>> energies = this->hamil->locEnergy(this->current_state);
    T energy = 0;
//#pragma omp parallel for reduction(+ : energy)
    for (auto it = std::begin(energies); it != std::end(energies); ++it) {
        u64 state = std::get<0>(*it);
        T value = std::get<1>(*it);
        auto v = value;
        if (state != this->current_state) {
            //intToBase(state, this->tmp_vector, BinaryPowers);
            intToBase(state, this->tmp_vector, 2);
            v = v * this->pRatio(); // this->current_vector, this->tmp_vector
        }
        energy += v;
    }
    return energy;
}

/*
* Calculate the local energy depending on the given Hamiltonian
*/
template<>
inline cpx rbmState<cpx>::locEn() {
    v_1d<std::tuple<u64, cpx>> energies = this->hamil->locEnergy(this->current_state);
    double energy_real = 0;
    double energy_imag = 0;
//#pragma omp parallel for reduction(+ : energy_real, energy_imag)
    for (auto it = 0; it < energies.size(); it++)
    //for (auto it = std::begin(energies); it != std::end(energies); ++it)
    {
        //arma::Col<int> tmp(this->n_visible);
        u64 state = std::get<0>(energies[it]);
        cpx value = std::get<1>(energies[it]);
        //stout << VEQ(state) << "\t" << VEQ(value) << EL;
        auto v = value;
        if (state != this->current_state) {
            //intToBase(state, this->tmp_vector, BinaryPowers);
            intToBase(state, this->tmp_vector, 2);
            v = v * this->pRatio(); // this->current_vector, this->tmp_vector
        }
        energy_real += std::real(v);
        energy_imag += std::imag(v);
    }
    return cpx(energy_real, energy_imag);
}

/*
* @param v the base vector we want to calculate derivatives from
*/
template<typename T>
void rbmState<T>::blockSampling(size_t b_size, u64 start_state, size_t n_flips){
    this->set_state(start_state);

    u64 tmp_state = this->current_state;
    for(auto i = 0; i < b_size; i++){
        for (auto j = 0; j < n_flips; j++) {
            auto flip_place = this->hamil->ran.randomInt_uni(0, this->n_visible - 1);
            tmp_state = flip(this->current_state, BinaryPowers[flip_place], flip_place);
        }
        // create the vector
        //intToBase(tmp_state, this->tmp_vector, BinaryPowers);
        intToBase(tmp_state, this->tmp_vector, 2);
        T val = this->pRatio();//this->current_vector, this->tmp_vector
        double proba = std::abs(val);
        if (this->hamil->ran.randomReal_uni() <= proba * proba)
            this->set_state(tmp_state);
    }
}

/*
* @param v the base vector we want to calculate derivatives from
*/
template<typename T>
v_1d<T> rbmState<T>::mcSampling(size_t n_samples, size_t n_blocks, size_t n_therm, size_t b_size, size_t n_flips){
    auto start = std::chrono::high_resolution_clock::now();
    this->pbar = pBar(34, n_samples);
    auto mcsteps = n_samples - n_therm;
    auto batch_proba = (this->batch / double(n_blocks));
    auto norm = this->batch;
    if (batch_proba > 1) norm = n_blocks;


    std::map<u64, T> states;
    arma::Col<T> energies(n_blocks);
    arma::Col<T> averageWeights(this->full_size);
    v_1d<T> meanEnergies(mcsteps, 0);

    for(auto i = 0; i < mcsteps; i++){
        
        // thermalize system
        this->set_rand_state();
        this->blockSampling(n_therm * b_size, this->current_state, n_flips);

        // start the simulation
        this->S.zeros();                                                                // Fisher info
        this->F.zeros();                                                                // Gradient force
        averageWeights.zeros();

        size_t took = 0;                                                                // to check whether the batch is ready
        for(auto k = 0; k < n_blocks; k++){
            if (took == norm) break;
            this->blockSampling(b_size, this->current_state, n_flips);
            // look at the states
            if (i >= mcsteps - 3) {
                states[this->current_state] = this->coeff(this->current_vector);
            }
            if (this->hamil->ran.randomReal_uni() <= batch_proba) {
                this->calcVarDeriv(this->current_vector);
                // append local energies
                energies(k) = this->locEn();
                // append gradient forces with the first part of covariance <E_kO_k*>
                this->F += energies(k) * arma::conj(this->O_flat);
                // append covariance matrices with the first part of covariance <O_k*O_k'>
                this->S += arma::conj(this->O_flat) * this->O_flat.t();
                // average weight gradients
                averageWeights += this->O_flat;
                took += 1;
            }
        }
        // normalize
        averageWeights /= double(norm);
        this->F /= double(norm);
        this->S /= double(norm);

        // update the covariance
        const auto meanLocEn = arma::mean(energies);
        // append gradient forces with the first part of covariance <E_k><O_k*>
        this->F -= meanLocEn * arma::conj(averageWeights);
        // append covariance matrices with the first part of covariance <O_k*><O_k'>
        this->S -= arma::conj(averageWeights) * averageWeights.t();
        // update model
        this->updVarDeriv();
        // add energy
        meanEnergies[i] = meanLocEn;
        if (i % pbar.percentageSteps == 0)
            pbar.printWithTime("-> PROGRESS");
    }
    stout << "->\n\t\t\tTime taken: " << tim_s(start) << EL << EL << EL;
    stout << "------------------------------------------------------------------------" << EL;
    stout << "GROUND STATE RBM:" << EL;
    this->pretty_print(states, 0.01);
    stout << "------------------------------------------------------------------------" << EL;

    return meanEnergies;
}

#endif // !RBM_H