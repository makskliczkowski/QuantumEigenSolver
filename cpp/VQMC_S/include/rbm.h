#pragma once
#include "../src/progress.h"
#include "hamil.h"

template <typename _type>
class rbmState{

private:
    // general parameters
    std::string info;                                           // info about the model
    size_t n_visible;                                           // visible neurons
    size_t n_hidden;                                            // hidden neurons
    size_t full_size;                                           // full size of the parameters
    size_t hilbert_size;                                        // hilbert space size
    double lr, momentum, xavier_const;                          // learning rate, momentum and xavier initializer constant
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

    // saved training parameters
    u64 current_state;                                          // current state during the simulation
    vec current_vector;                                         // current state vector during the simulation
    vec tmp_vector;                                             // tmp state vector during the simulation

public:
    ~rbmState() = default;
    rbmState() = default;
    rbmState(size_t nH, size_t nV, std::shared_ptr<SpinHamiltonian<_type>> const & hamiltonian,
            double lr, double mom, double xav
            ) 
            : n_hidden(nH), n_visible(nV)
            , lr(lr), momentum(mom)
            , xavier_const(xav)
            {
                this->hamil = hamiltonian;
                this->hilbert_size = hamil->get_hilbert_size();
                stout << VEQ(this->hilbert_size) << EL;
                this->full_size = this->n_hidden + this->n_visible + this->n_hidden * this->n_visible;
                this->allocate();
                this->init();
            };
    
    // set the current state to random
    void set_rand_state() {
        this->current_state = this->hamil->ran.randomInt_uni(0, this->hilbert_size - 1);
        intToBase(this->current_state, this->current_vector, 2);
    };

    // pretty print the state sampled
    void pretty_print(const std::map<u64, _type>& sample_states, double tol = 5e-2) const;


    // allocate the memory for the biases and weights
    void allocate();
    // initialize all
    void init();
    // set weights
    void set_weights(const arma::Col<_type>& gradient);
    // ------------------------------------------- AMPLITUDES AND ANSTATZ REPRESENTATION -------------------------------------------

    // the hiperbolic cosine of the parameters
    arma::Col<_type> Fs(const vec& v) const{return arma::cosh(this->b_h + this->W * v); };

    // get the current amplitude
    _type coeff(u64 state){
        intToBase(state, this->tmp_vector);
        _type cd = 0;
        for (int i = 0; i < b_v.n_elem; i++)
            cd += (this->b_v(i)) * this->tmp_vector(i);
        return (exp(cd) * arma::prod(Fs(this->current_vector))) * std::pow(2.0, this->n_hidden);
    };
    // get the current amplitude given vector
    _type coeff(const vec& v) const {
        _type cd = 0;
        for (int i = 0; i < b_v.n_elem; i++)
            cd += (this->b_v(i)) * v(i);
        return (exp(cd) * arma::prod(Fs(v))) * std::pow(2.0, this->n_hidden);
    };

    // get probability ratio for a reference state v1 and v2 state
    _type pRatio(const vec& v1, const vec& v2) const {
        auto f1 = Fs(v1);
        auto f2 = Fs(v2);

        _type cd = 0;
        arma::vec difference = v2 - v1;
        for (int i = 0; i < difference.n_elem; i++)
            cd += (b_v(i)) * difference(i);
        return std::exp(cd + arma::sum(arma::log(Fs(v2) / Fs(v1))));
    };
    
    _type pRatio() const {
        return exp(arma::cdot(this->b_v, (this->tmp_vector - this->current_vector)) + arma::sum(arma::log(Fs(this->tmp_vector) / Fs(this->current_vector))));
    };


    // get local energies
    _type locEn();

    _type locEn(const vec& v);

    // variational derivative calculation
    void calcVarDeriv(const vec& v1);

    // update weights after gradient descent
    void updVarDeriv();

    // sample block
    void blockSampling(size_t b_size, u64 start_stae, size_t n_flips = 1);

    // sample the probabilistic space
    v_1d<_type> mcSampling(size_t n_samples, size_t n_blocks, size_t n_therm, size_t b_size, size_t n_flips = 1);
    
};

template<typename _type>
inline void rbmState<_type>::pretty_print(const std::map<u64, _type>& sample_states, double tol) const {
    v_1d<int> tmp(this->n_visible);
    double norm = 0;
    for (const auto& [state, value] : sample_states) {
        norm += abs(value);
    }

    for (const auto& [state, value] : sample_states) {
        intToBase(state, tmp, 2);
        if (std::abs(value / norm)  >= tol)
            stout << value / norm << " * |" << tmp << "> + ";
    }
    stout << EL;
}

/*

*/
template<typename T>
void rbmState<T>::allocate() {
    // initialize biases
    this->b_v = arma::Col<T>(this->n_visible, arma::fill::zeros);
    this->O_b_v = arma::Col<T>(this->n_visible, arma::fill::zeros);
    this->b_h = arma::Col<T>(this->n_hidden, arma::fill::zeros);
    this->O_b_h = arma::Col<T>(this->n_hidden, arma::fill::zeros);
    this->W = arma::Mat<T>(this->n_hidden, this->n_visible, arma::fill::zeros);
    this->O_w = arma::Mat<T>(this->n_hidden, this->n_visible, arma::fill::zeros);
    this->O_flat = arma::Col<T>(this->full_size, arma::fill::zeros);
    // allocate covariance and forces
    this->F = arma::Col<T>(this->full_size, arma::fill::zeros);
    this->S = arma::Mat<T>(this->full_size, this->full_size, arma::fill::zeros);
    // allocate vectors
    this->current_vector = vec(this->n_visible, arma::fill::zeros);
    this->tmp_vector = vec(this->n_visible, arma::fill::zeros);
}

/*
* Initialize the weights 
*/
template<typename T>
void rbmState<T>::init() {
    // initialize random state
    this->set_rand_state();
    intToBase(this->current_state, this->current_vector, 2);

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
void rbmState<cpx>::init() {
    // initialize random state
    this->set_rand_state();
    this->current_state = this->hamil->ran.randomInt_uni(0, this->n_visible);
    intToBase(this->current_state, this->current_vector, 2);

    // initialize biases visible
    for (int i = 0; i < this->n_visible; i++)
        this->b_v(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.random_real_normal()) / 10.0;
    // hidden
    for (int i = 0; i < this->n_hidden; i++)
        this->b_h(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.randomReal_uni()) / 10.0;
    // matrix
    for (int i = 0; i < this->W.n_rows; i++)
        for (int j = 0; j < this->W.n_cols; j++)
            this->W(i, j) = (this->hamil->ran.random_real_normal() + imn*this->hamil->ran.random_real_normal()) / 10.0;

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
    this->b_v -= gradient.subvec(0, this->n_visible - 1);
    this->b_h -= gradient.subvec(this->n_visible, this->n_visible + this->n_hidden - 1);
    this->W -= arma::reshape(
        gradient.subvec(this->n_visible + this->n_hidden,
        this->full_size - 1),
        this->W.n_rows,
        this->W.n_cols
    );
}

/*
* @param v the base vector we want to calculate derivatives from
*/
template<typename T>
void rbmState<T>::calcVarDeriv(const vec& v){
    const auto theta = this->b_h + this->W * v;
    // change of the first bias
    for (auto i = 0; i < this->O_b_v.n_elem; i++)
        this->O_b_v(i) = v(i);
    //this->O_b_v = v;
    // change of the hidden bias
    this->O_b_h = arma::tanh(theta);
    // change of the weights matrix
    this->O_w = this->O_b_h * this->O_b_v.t();
    // calculate the flattened part
    this->O_flat.subvec(0, this->n_visible - 1) = this->O_b_v;
    this->O_flat.subvec(this->n_visible, n_visible + this->n_hidden - 1) = this->O_b_h;
    this->O_flat.subvec(this->n_visible + this->n_hidden,
                        this->full_size -1) = arma::vectorise(this->O_w);
}

/*
* updates the weights using stochastic gradient descent
*/
template<typename T>
void rbmState<T>::updVarDeriv(){
    // greadient descent and psudo inversion
    double tol = 1e-7;
    auto Sinv = arma::pinv(this->S, tol);

    // update flat vector
    this->O_flat = this->lr * Sinv * this->F;
    this->set_weights(this->O_flat);
}

/*
* Calculate the local energy depending on the given Hamiltonian
*/
template<typename T>
T rbmState<T>::locEn(){
    v_1d<std::tuple<u64, T>> energies = this->hamil->locEnergy(this->current_state);
    T energy = 0;
    for (auto it = std::begin(energies); it != std::end(energies); ++it) {
        u64 state = std::get<0>(*it);
        T value = std::get<1>(*it);
        //stout << VEQ(state) << "\t" << VEQ(value) << EL;
        auto v = value;
        if (state != this->current_state) {
            intToBase(state, this->tmp_vector, 2);
            v = v * this->pRatio(this->current_vector, this->tmp_vector);
        }
        energy += v;
    }
    return energy;
}

/*
* Calculate the local energy depending on the given Hamiltonian
*/
template<typename T>
T rbmState<T>::locEn(const vec& v) {
    v_1d<std::tuple<u64, T>> energies = this->hamil->locEnergy(v);
    T energy = 0;
    for (auto it = std::begin(energies); it != std::end(energies); ++it) {
        u64 state = std::get<0>(*it);
        T value = std::get<1>(*it);
        auto v = value;
        if (state != this->current_state) {
            intToBase(state, this->tmp_vector, 2);
            v = v * this->pRatio(this->current_vector, this->tmp_vector);
        }
        energy += v;
    }
    return energy;
}

/*
* @param v the base vector we want to calculate derivatives from
*/
template<typename T>
void rbmState<T>::blockSampling(size_t b_size, u64 start_state, size_t n_flips){
    this->current_state = start_state;
    intToBase(this->current_state, this->current_vector, 2);

    u64 tmp_state = this->current_state;
    for(auto i = 0; i < b_size; i++){
        for (auto j = 0; j < n_flips; j++) {
            auto flip_place = this->hamil->ran.randomInt_uni(0, this->n_visible - 1);
            tmp_state = flip(this->current_state, BinaryPowers[flip_place], flip_place);
        }
        // create the vector
        intToBase(tmp_state, this->tmp_vector, 2);
        const T val = this->pRatio(this->current_vector, this->tmp_vector);
        double proba = std::pow(std::abs(val), 2);
        //stout << VEQ(proba) << EL;
        if (this->hamil->ran.randomReal_uni() <= proba) {
            this->current_state = tmp_state;
            this->current_vector = tmp_vector;
        }
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

    std::map<u64, T> states;
    arma::Col<T> energies(n_blocks);
    arma::Col<T> averageWeights(this->full_size);
    v_1d<T> meanEnergies(mcsteps, 0);

    for(auto i = 0; i < mcsteps; i++){
        
        // thermalize system
        this->set_rand_state();
        this->blockSampling(n_therm * b_size, this->current_state, n_flips);
        //stout << "->\n\t\t\tRelaxation time taken: " << tim_s(start) << EL;
        // start the simulation
        this->S.zeros();                                                                // Fisher info
        this->F.zeros();                                                                // Gradient force
        averageWeights.zeros();
        for(auto k = 0; k < n_blocks; k++){
            this->blockSampling(b_size, this->current_state, n_flips);
            this->calcVarDeriv(this->current_vector);

            // look at the states
            if (i == mcsteps - 1)
                states[k] = this->coeff(this->current_state);

            // append local energies
            energies(k) = this->locEn();
            // append gradient forces with the first part of covariance <E_kO_k*>
            this->F += energies(k) * arma::conj(this->O_flat);
            // append covariance matrices with the first part of covariance <O_k*O_k'>
            this->S += arma::conj(this->O_flat) * this->O_flat.t();
           
            // average weight gradients
            averageWeights += this->O_flat;
        }
        averageWeights /= n_blocks;
        this->F /= n_blocks;
        this->S /= n_blocks;

        // update the covariance
        auto meanLocEn = arma::mean(energies);
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

    this->pretty_print(states, 0.05);
    stout << "->\n\t\t\t\Time taken: " << tim_s(start) << EL << EL << EL;
    return meanEnergies;
}

