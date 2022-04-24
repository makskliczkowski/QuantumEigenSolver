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


template <typename _type, typename _hamtype>
class rbmState{

private:
    // debug bools
    bool dbg_lcen = false;
    bool dbg_grad = false;
    bool dbg_samp = false;
    bool dbg_drvt = false;
    bool dbg_updt = false;
    bool dbg_thrm = false;
    bool dbg_blck = false;


    // general parameters
    std::string info;                                           // info about the model
    size_t batch;                                               // batch size for stochastic
    size_t n_visible;                                           // visible neurons
    size_t n_hidden;                                            // hidden neurons
    size_t full_size;                                           // full size of the parameters
    size_t hilbert_size;                                        // hilbert space size
    size_t thread_num;                                          // thread number
    double lr;                                                  // learning rate
    pBar pbar;                                                  // progress bar
    
    // network weights
    Mat<_type> W;                                               // weight matrix
    Col<_type> b_v;                                             // visible bias
    Col<_type> b_h;                                             // hidden bias

    // variational derivatives                                  
    Col<_type> O_b_h;                                           // gradient of the hidden bias
    Col<_type> O_flat;                                          // flattened output for easier calculation of the covariance
    Mat<_type> S;                                               // positive semi-definite covariance matrix
    Col<_type> F;                                               // forces
    
    // the Hamiltonian
    std::shared_ptr<SpinHamiltonian<_hamtype>> hamil;           // unique ptr to a general Hamiltonian of the spin system

    // optimizer
    std::unique_ptr<Adam<_type>> adam;                          // use the Adam optimizer for GD

    // saved training parameters
    u64 current_state;                                          // current state during the simulation
    Col<double> current_vector;                                 // current state vector during the simulation
    Col<double> tmp_vector;                                     // tmp state vector during the simulation
    v_1d<Col<double>> tmp_vectors;                              // tmp vectors for omp 
public:
    ~rbmState() = default;
    rbmState() = default;
    rbmState(size_t nH, size_t nV, std::shared_ptr<SpinHamiltonian<_hamtype>> const & hamiltonian,
            double lr, size_t batch, size_t thread_num
            ) 
            : n_hidden(nH), n_visible(nV)
            , lr(lr)
            , batch(batch)
            {
                this->thread_num = thread_num;
                // checks for the debug info
                this->debug_check();          
                // creates the hamiltonian class
                this->hamil = hamiltonian;
                this->hilbert_size = hamil->get_hilbert_size();
                this->full_size = n_hidden + n_visible + n_hidden * n_visible;
#ifdef USE_ADAM
                this->adam = std::make_unique<Adam<_type>>(lr, full_size);
#endif
                this->set_info();
                // allocate memory
                this->allocate();
                // initialize random state
                this->set_rand_state();
                this->init();
            };
    // ------------------------------------------- HELPERS -------------------------------------------
    
    // debug checker
    void debug_check() {
#ifndef DEBUG
        omp_set_num_threads(this->thread_num);                        // Use threads for all consecutive parallel regions
#else
#ifdef DEBUG_RBM_SAMP
        this->dbg_samp = true;
#endif // DEBUG_RBM_SAMP
#ifdef DEBUG_RBM_LCEN
        this->dbg_lcen = true;
#endif // DEBUG_RBM_LCEN
#ifdef DEBUG_RBM_GRAD
        this->dbg_grad = true;
#endif // DEBUG_RBM_GRAD
#ifdef DEBUG_RBM_DRVT
        this->dbg_drvt = true;
#endif // DEBUG_RBM_DRVT
#ifdef DEBUG_RBM_UPDT
        this->dbg_updt = true;
#endif // DEBUG_RBM_UPDT
#ifdef DEBUG_RBM_THRM
        this->dbg_thrm = true;
#endif // DEBUG_RBM_THRM
#ifdef DEBUG_RBM_BLCK
        this->dbg_blck = true;
#endif // DEBUG_RBM_THRM
#endif // !DEBUG
    };

    // ------------------------------------------- PRINTERS -------------------------------------------
    // pretty print the state sampled
    void pretty_print(const std::map<u64, _type>& sample_states, double tol = 5e-2) const;

    // ------------------------------------------- SETTTERS -------------------------------------------
    // sets info
    void set_info()                                                     { this->info = VEQ(n_visible) + "," + VEQ(n_hidden) + "," + VEQ(batch) + "," + VEQ(hilbert_size); };

    // sets the current state
    void set_state(u64 state) {
        this->current_state = state;
        intToBaseBit(state, this->current_vector);
    }
    // set the current state to random
    void set_rand_state()                                               { this->set_state(this->hamil->ran.randomInt_uni(0, this->hilbert_size - 1)); };
    
    // set weights
    void set_weights(const Col<_type>& gradient);

    // ------------------------------------------- GETTERS ------------------------------------------
    auto get_info()                                                     const RETURNS(this->info);

    // ------------------------------------------- INITIALIZERS ------------------------------------------
    // allocate the memory for the biases and weights
    void allocate();
    // initialize all
    void init();
    // ------------------------------------------- AMPLITUDES AND ANSTATZ REPRESENTATION -------------------------------------------

    // the hiperbolic cosine of the parameters
    Col<_type> Fs(const Col<double>& v)                                 const { return arma::cosh(this->b_h + this->W * v); };

    // get the current amplitude given vector
    auto coeff(const Col<double>& v)                                    const { return (exp(cdotm(this->b_v, v)) * arma::prod(Fs(v)) * std::pow(2.0, this->n_hidden)); };

    // get probability ratio for a reference state v1 and v2 state
    _type pRatio()                                                      const { return exp(cdotm(this->b_v, Col<double>(this->tmp_vector - this->current_vector)) + sum(log(Fs(this->tmp_vector) / Fs(this->current_vector)))); };
    _type pRatio(const Col<double>& v1, const Col<double>& v2)          const { return exp(cdotm(this->b_v, Col<double>(v2 - v1)) + sum(log(Fs(v2) / Fs(v1)))); };

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

// ------------------------------------------------- PRINTERS -------------------------------------------------
/*
* Pretty prints the state given the sample_states map
* @param sample_states the map from u64 state integer to value at the state
* @param tol tolerance on the absolute value
*/
template<typename _type, typename _hamtype>
inline void rbmState<typename _type, typename _hamtype>::pretty_print(const std::map<u64, _type>& sample_states, double tol) const {
    v_1d<int> tmp(this->n_visible);
    double norm = 0;
    double phase = 0;
    // normalise
    for (const auto& [state, value] : sample_states) {
        norm += std::pow(abs(value), 2.0);
        phase = std::arg(value);
    }

    for (const auto& [state, value] : sample_states) {
        auto val = value / sqrt(norm);// *std::exp(-imn * phase);
        SpinHamiltonian<_type>::print_base_state(state, val, tmp, tol);
    }
    stout << EL;
}

// ------------------------------------------------- INITIALIZERS -------------------------------------------------
/*
* allocates memory for arma objects
*/
template<typename _type, typename _hamtype>
void rbmState<typename _type, typename _hamtype>::allocate() {
    auto Ns = this->hamil->lattice->get_Ns();
    // initialize biases
    this->b_v = Col<_type>(this->n_visible, arma::fill::randn) / double(Ns);
    this->b_h = Col<_type>(this->n_hidden, arma::fill::randn) / double(Ns);
    this->W = Mat<_type>(this->n_hidden, this->n_visible, arma::fill::randn) / double(Ns);
    // allocate gradients
    this->O_b_h = Col<_type>(this->n_hidden, arma::fill::zeros);
    this->O_flat = Col<_type>(this->full_size, arma::fill::zeros);
    // allocate covariance and forces
    this->F = Col<_type>(this->full_size, arma::fill::zeros);
    this->S = Mat<_type>(this->full_size, this->full_size, arma::fill::zeros);
    // allocate vectors
    this->current_vector = Col<double>(this->n_visible, arma::fill::zeros);
    this->tmp_vector = Col<double>(this->n_visible, arma::fill::zeros);
    this->tmp_vectors = v_1d<Col<double>>(this->thread_num, Col<double>(this->n_visible, arma::fill::zeros));
}

/*
* Initialize the weights 
*/
template<typename _type, typename _hamtype>
void rbmState<typename _type, typename _hamtype>::init() {
    // initialize random state
    this->set_rand_state();

    // initialize biases visible
    for (int i = 0; i < this->n_visible; i++)
        this->b_v(i) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, 6);
    // hidden
    for (int i = 0; i < this->n_hidden; i++)
        this->b_h(i) = this->hamil->ran.xavier_uni(this->n_hidden, 1, 6);
    // matrix
    for (int i = 0; i < this->W.n_rows; i++)
        for(int j = 0; j < this->W.n_cols; j++)
            this->W(i,j) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, 6);
}

/*
* Intialize the weights, overwritten for complex weights
*/
template<>
inline void rbmState<cpx, double>::init() {
    // initialize random state
    this->set_rand_state();
    auto Ns = this->hamil->lattice->get_Ns();
    // initialize biases visible
    for (int i = 0; i < this->n_visible; i++)
        //this->b_v(i) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const) + imn * this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const);
        this->b_v(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.random_real_normal()) / double(Ns);
    // hidden
    for (int i = 0; i < this->n_hidden; i++)
        //this->b_h(i) = this->hamil->ran.xavier_uni(this->n_hidden, 1, this->xavier_const) + imn * this->hamil->ran.xavier_uni(this->n_hidden, 1, this->xavier_const);
        this->b_h(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.randomReal_uni()) / double(Ns);
    // matrix
    for (int i = 0; i < this->W.n_rows; i++)
        for (int j = 0; j < this->W.n_cols; j++)
            //this->W(i, j) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const) + imn * this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const);
            this->W(i, j) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.random_real_normal()) / double(Ns);
}

/*
* Intialize the weights, overwritten for complex weights and complex Hamiltonian
*/
template<>
inline void rbmState<cpx, cpx>::init() {
    // initialize random state
    this->set_rand_state();
    auto Ns = this->hamil->lattice->get_Ns();
    // initialize biases visible
    for (int i = 0; i < this->n_visible; i++)
        //this->b_v(i) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const) + imn * this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const);
        this->b_v(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.random_real_normal()) / double(Ns);
    // hidden
    for (int i = 0; i < this->n_hidden; i++)
        //this->b_h(i) = this->hamil->ran.xavier_uni(this->n_hidden, 1, this->xavier_const) + imn * this->hamil->ran.xavier_uni(this->n_hidden, 1, this->xavier_const);
        this->b_h(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.randomReal_uni()) / double(Ns);
    // matrix
    for (int i = 0; i < this->W.n_rows; i++)
        for (int j = 0; j < this->W.n_cols; j++)
            //this->W(i, j) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const) + imn * this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, this->xavier_const);
            this->W(i, j) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.random_real_normal()) / double(Ns);
}


// ------------------------------------------------- SETTERS -------------------------------------------------
/*
* sets the weights according to the gradient descent
* @param gradient gradient calculated from the simulation
*/
template<typename _type, typename _hamtype>
void rbmState<typename _type, typename _hamtype>::set_weights(const Col<_type>& gradient) {
    // update weights accordingly
#pragma omp parallel for
    for (auto i = 0; i < this->n_visible; i++)
        this->b_v(i) -= gradient(i);
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
}
// ------------------------------------------------- CALCULATORS -------------------------------------------------
/*
* calculates the variational derivative analytically
* @param v the base vector we want to calculate derivatives from
*/
template<typename _type, typename _hamtype>
void rbmState<_type, typename _hamtype>::calcVarDeriv(const Col<double>& v){
    auto var_deriv_time = std::chrono::high_resolution_clock::now();

    // change of the hidden bias
    this->O_b_h = arma::tanh(this->b_h + this->W * v);

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
            this->O_flat(elem) = this->O_b_h(i) * v(j);
        }
    }
    PRT(var_deriv_time, this->dbg_drvt)
}

/*
* updates the weights using stochastic gradient descent
*/
template<typename _type, typename _hamtype>
void rbmState<typename _type, typename _hamtype>::updVarDeriv(){
    auto var_deriv_time_upd = std::chrono::high_resolution_clock::now();
   
    // greadient descent and psudo inversion
    const double tol = 1e-6;
    const Mat<_type> Sinv = arma::pinv(this->S, tol);

    // update flat vector
#ifdef USE_ADAM
    this->adam->update(Sinv * this->F);
    this->set_weights(this->adam->get_grad());
#else
    this->set_weights(this->lr * Sinv * this->F);
#endif  
    PRT(var_deriv_time_upd, this->dbg_updt)
}

/*
* Calculate the local energy depending on the given Hamiltonian
*/
template<typename _type, typename _hamtype>
inline _type rbmState<typename _type, typename _hamtype>::locEn(){
    auto loc_en_time = std::chrono::high_resolution_clock::now();
    // get the reference to all local energies and changed states from the model Hamiltonian
    auto energies = this->hamil->get_localEnergyRef(this->current_state);
    _type energy = 0;
#ifndef DEBUG
#pragma omp parallel for reduction(+ : energy) shared(energies)
#endif
    for (auto i = 0; i < energies.size(); i++) 
    //for (const auto& [state, value] : energies)
    {
        const auto& [state, value] = energies[i];
        _type v = value;
        // changes accordingly not to create data race
        if (state != this->current_state) {
#ifndef DEBUG
            const int tid = omp_get_thread_num();
            const int vid = tid % this->thread_num;
            intToBaseBit(state, this->tmp_vectors[vid]);
            v = v * this->pRatio(this->current_vector, this->tmp_vectors[vid]);
#else
            intToBaseBit(state, this->tmp_vector);
            v = v * this->pRatio();
#endif
        }
        energy += v;
    }

    PRT(loc_en_time, this->dbg_lcen)
    return energy;
}

// ------------------------------------------------- SAMPLING -------------------------------------------------
/*
* block updates the current state according to Metropolis-Hastings algorithm
* @param b_size the size of the correlation block
* @param start_state the state to start from
* @param n_flips number of flips at the single step
*/
template<typename _type, typename _hamtype>
void rbmState<typename _type, typename _hamtype>::blockSampling(size_t b_size, u64 start_state, size_t n_flips){
    this->set_state(start_state);

    u64 tmp_state = this->current_state;
    for(auto i = 0; i < b_size; i++){
        for (auto j = 0; j < n_flips; j++) {
            const auto flip_place = this->hamil->ran.randomInt_uni(0, this->n_visible - 1);
            tmp_state = flip(this->current_state, BinaryPowers[flip_place], flip_place);
        }
        // create the vector
        intToBase(tmp_state, this->tmp_vector, 2);
        double proba = std::abs(this->pRatio());
        if (this->hamil->ran.randomReal_uni() <= proba * proba) {
            this->current_state = tmp_state;
            this->current_vector = this->tmp_vector;
        }
    }
}

/*
* @param v the base vector we want to calculate derivatives from
*/
template<typename _type, typename _hamtype>
v_1d<_type> rbmState<typename _type, typename _hamtype>::mcSampling(size_t n_samples, size_t n_blocks, size_t n_therm, size_t b_size, size_t n_flips){
    // start the timer!
    auto start = std::chrono::high_resolution_clock::now();
    // make the pbar!
    this->pbar = pBar(34, n_samples);
    // take out thermalization steps
    const auto mcsteps = n_samples - n_therm;
    // calculate the probability that we include the element in the batch
    const auto batch_proba = (this->batch / double(n_blocks));
    // check if the batch is not bigger than the blocks number
    const auto norm = (batch_proba > 1) ? n_blocks : this->batch;

    // states to be returned
    std::map<u64, _type> states;
    Col<_type> energies(norm);
    Col<_type> averageWeights(this->full_size);
    v_1d<_type> meanEnergies(mcsteps, 0.0);
    

    for(auto i = 0; i < mcsteps; i++){
        // set the random state at each Monte Carlo iteration
        this->set_rand_state();

        // start the simulation
        this->S.zeros();                                                                // Fisher info
        this->F.zeros();                                                                // Gradient force
        averageWeights.zeros();                                                         // Weights gradients average

        // thermalize system
        auto therm_time = std::chrono::high_resolution_clock::now();
        this->blockSampling(n_therm * b_size, this->current_state, n_flips);
        PRT(therm_time, this->dbg_thrm)

        // to check whether the batch is ready already
        size_t took = 0;                                                               
        auto blocks_time = std::chrono::high_resolution_clock::now();
        //for(auto k = 0; k < n_blocks; k++){
        while(took < norm){
            // break if we got batch already
            //if (took == norm) break;

            // block sample the stuff
            auto sample_time = std::chrono::high_resolution_clock::now();
            this->blockSampling(b_size, this->current_state, n_flips);
            PRT(sample_time, this->dbg_samp)

            // look at the states coefficient
            if (i >= mcsteps - 3)
                states[this->current_state] = this->coeff(this->current_vector);
            

            if (norm == n_blocks || this->hamil->ran.randomReal_uni() <= batch_proba) {
                auto gradients_time = std::chrono::high_resolution_clock::now();
                this->calcVarDeriv(this->current_vector);
                // append local energies
                energies(took) = this->locEn();
                
                // append gradient forces with the first part of covariance <E_kO_k*>
                this->F += energies(took) * arma::conj(this->O_flat);
                
                // append covariance matrices with the first part of covariance <O_k*O_k'>
                this->S += arma::conj(this->O_flat) * this->O_flat.t();
                
                // average weight gradients
                averageWeights += this->O_flat;
                
                // append number of elements taken
                took += 1;
                PRT(gradients_time, this->dbg_grad)
            }
        }
        PRT(blocks_time, this->dbg_blck)
        
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
        
        // update the progress bar
        if (i % pbar.percentageSteps == 0)
            pbar.printWithTime("-> PROGRESS");
    }
    stouts("->\t\t\tMonte Carlo energy search ", start);
    stout << "\n------------------------------------------------------------------------" << EL;
    stout << "GROUND STATE RBM:" << EL;
    this->pretty_print(states, 0.01);
    stout << "\n------------------------------------------------------------------------" << EL;

    return meanEnergies;
}

#endif // !RBM_H