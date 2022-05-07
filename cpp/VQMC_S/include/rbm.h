#pragma once
#ifndef RBM_H
#define RBM_H

#include "../src/progress.h"

#ifndef HAMIL_H
#include "hamil.h"
#endif

#ifndef ADAM_H
    #ifndef USE_ADAM
        #define USE_ADAM
    #endif // !USE_ADAM
    #include "../include/ml.h"
#endif // !ADAM_H

#ifdef DONT_USE_ADAM
#undef USE_ADAM
#endif // !DONT_USE_ADAM

#define RBM_ANGLES_UPD

//#define PINV
#define S_REGULAR

#ifdef PINV
    constexpr auto pinv_tol = 1e-7;
    #ifdef S_REGULAR
        #undef S_REGULAR
    #endif
#elif defined S_REGULAR 
    constexpr auto lambda_0_reg = 100;
    constexpr auto b_reg = 0.9;
    constexpr auto lambda_min_reg = 1e-4;
    #ifdef PINV
        #undef PINV
    #endif
#endif



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
    string info;                                           // info about the model
    size_t batch;                                               // batch size for stochastic
    size_t n_visible;                                           // visible neurons
    size_t n_hidden;                                            // hidden neurons
    size_t full_size;                                           // full size of the parameters
    size_t hilbert_size;                                        // hilbert space size
    size_t thread_num;                                          // thread number
    double lr;                                                  // learning rate
#ifdef S_REGULAR
    double current_b_reg = b_reg;                               // parameter for regularisation, changes with Monte Carlo steps
#endif // S_REGULAR

    pBar pbar;                                                  // progress bar
    
    // network weights
    Mat<_type> W;                                               // weight matrix
    Col<_type> b_v;                                             // visible bias
    Col<_type> b_h;                                             // hidden bias

    // variational derivatives                                  
    Col<_type> thetas;                                          // effective angles
    Col<_type> O_flat;                                          // flattened output for easier calculation of the covariance
    Col<_type> grad;                                            // change of the weights
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
                this->init();
                this->set_rand_state();
            };
    // -------------------------------------------				 HELPERS				 -------------------------------------------
    
    // debug checker
    void debug_check() {
#ifndef DEBUG
        omp_set_num_threads(this->thread_num);                  // Use threads for all consecutive parallel regions
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

    // ------------------------------------------- 				 PRINTERS				  -------------------------------------------
    // pretty print the state sampled
    void pretty_print(std::map<u64, _type>& sample_states, double tol = 5e-2) const;

    // ------------------------------------------- 				 SETTTERS				  -------------------------------------------
    // sets info
    void set_info()                                                     { this->info = VEQ(n_visible) + "," + VEQ(n_hidden) + "," + VEQ(batch) + "," + VEQ(hilbert_size); };

    // sets the current state
    void set_state(u64 state, bool set = false) {
        this->current_state = state;

        INT_TO_BASE_BIT(state, this->current_vector);

#ifdef RBM_ANGLES_UPD
    if (set)
        this->set_angles(this->current_vector);
#endif
    }
    
    // set the current state to random
    void set_rand_state();
    
    // set weights
    void set_weights();

    // set effective angles
    void set_angles();
    void set_angles(const Col<double>& v);
    // ------------------------------------------- 				 UPDATERS				  -----------------------------------------
    void update_angles(int flip_place, double flipped_spin);
    void update_angles(const Col<double>& v, int flip_place);

    // ------------------------------------------- 				 GETTERS				  ------------------------------------------
    auto get_info()                                                     const RETURNS(this->info);

    // ------------------------------------------- 				 INITIALIZERS				  ------------------------------------------
    // allocate the memory for the biases and weights
    void allocate();
    // initialize all
    void init();
    // ------------------------------------------- 				 AMPLITUDES AND ANSTATZ REPRESENTATION				  -------------------------------------------

    // the hiperbolic cosine of the parameters
    Col<_type> Fs(const Col<double>& v)                                 const { return arma::cosh(this->b_h + this->W * v); };

    // get the current amplitude given vector
    auto coeff(const Col<double>& v)                                    const { return (exp(dotm(this->b_v, v)) * arma::prod(Fs(v)) * std::pow(2.0, this->n_hidden)); };

    // get probability ratio for a reference state v1 and v2 state
    _type pRatio()                                                      const { return exp(dotm(this->b_v, Col<double>(this->tmp_vector - this->current_vector)) + sum(log(Fs(this->tmp_vector) / Fs(this->current_vector)))); };
    _type pRatio(const Col<double>& v)                                  const { return exp(dotm(this->b_v, Col<double>(v - this->current_vector)) + sum(log(Fs(v) / arma::cosh(this->thetas)))); };
    _type pRatio(const Col<double>& v1, const Col<double>& v2)          const { return exp(dotm(this->b_v, Col<double>(v2 - v1)) + sum(log(Fs(v2) / Fs(v1)))); };

    // get local energies
    _type locEn();

    // variational derivative calculation
    void calcVarDeriv(const Col<double>& v);

    // update weights after gradient descent
    void updVarDeriv(int current_step);
    // ------------------------------------------- 				 SAMPLING				  -------------------------------------------
    // sample block
    void blockSampling(size_t b_size, u64 start_stae, size_t n_flips = 1, bool thermalize = true);

    // sample the probabilistic space
    Col<_type> mcSampling(size_t n_samples, size_t n_blocks, size_t n_therm, size_t b_size, size_t n_flips = 1);
    map<u64, _type> avSampling(size_t n_samples, size_t n_therm, size_t b_size, size_t n_flips = 1);

};

// ------------------------------------------------- 				 PRINTERS				  -------------------------------------------------
/*
* @brief Pretty prints the state given the sample_states map
* @param sample_states the map from u64 state integer to value at the state
* @param tol tolerance on the absolute value
*/
template<typename _type, typename _hamtype>
inline void rbmState<typename _type, typename _hamtype>::pretty_print(std::map<u64, _type>& sample_states, double tol) const {
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
        sample_states[state] = val;
        SpinHamiltonian<_type>::print_base_state(state, val, tmp, tol);
    }
    stout << EL;

}

// ------------------------------------------------- 				 INITIALIZERS 				 -------------------------------------------------
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
    this->O_flat = Col<_type>(this->full_size, arma::fill::zeros);
    this->thetas = Col<_type>(this->n_hidden, arma::fill::zeros);
    this->grad = Col<_type>(this->full_size, arma::fill::zeros);
    // allocate covariance and forces
    this->F = Col<_type>(this->full_size, arma::fill::zeros);
    this->S = Mat<_type>(this->full_size, this->full_size, arma::fill::zeros);
    // allocate vectors
    this->current_vector = Col<double>(this->n_visible, arma::fill::ones);
    this->tmp_vector = Col<double>(this->n_visible, arma::fill::ones);
    this->tmp_vectors = v_1d<Col<double>>(this->thread_num, Col<double>(this->n_visible, arma::fill::ones));
}

/*
* @brief Initialize the weights 
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
* @brief Intialize the weights, overwritten for complex weights
*/
template<>
inline void rbmState<cpx, double>::init() {
    // initialize random state
    this->set_rand_state();
    auto Ns = this->hamil->lattice->get_Ns();
    // initialize biases visible
    for (int i = 0; i < this->n_visible; i++)
        //this->b_v(i) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, 6) + imn * this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, 6);
        this->b_v(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.random_real_normal()) / double(Ns);
    // hidden
    for (int i = 0; i < this->n_hidden; i++)
        //this->b_h(i) = this->hamil->ran.xavier_uni(this->n_hidden, 1, 6) + imn * this->hamil->ran.xavier_uni(this->n_hidden, 1, 6);
        this->b_h(i) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.randomReal_uni()) / double(Ns);
    // matrix
    for (int i = 0; i < this->W.n_rows; i++)
        for (int j = 0; j < this->W.n_cols; j++)
            //this->W(i, j) = this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, 6) + imn * this->hamil->ran.xavier_uni(this->n_visible, this->n_hidden, 6);
            this->W(i, j) = (this->hamil->ran.random_real_normal() + imn * this->hamil->ran.random_real_normal()) / double(Ns);
}

/*
* @brief Intialize the weights, overwritten for complex weights and complex Hamiltonian
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
// ------------------------------------------------- 				 UPDATERS				  ------------------------------------------------

/*
* @brief update angles with the flipped spin (before the flip - hence -)
* @param flip_place place of the flip
* @param flipped_spin spin that would be flipped
*/
template<typename _type, typename _hamtype>
inline void rbmState<_type, _hamtype>::update_angles(int flip_place, double flipped_spin)
{
    this->thetas -= (2.0 * flipped_spin) * this->W.col(flip_place);
} 

/*
* @brief update angles with the vector (after the flip - hence +)
* @param v vector after flip
* @param flip_place place of the flip
*/
template<typename _type, typename _hamtype>
inline void rbmState<_type, _hamtype>::update_angles(const Col<double>& v, int flip_place)
{
    this->thetas += (2.0 * v(flip_place)) * this->W.col(flip_place);
}

// -------------------------------------------------				  SETTERS				  -------------------------------------------------

/*
* @brief sets the current state to uniform random
*/
template<typename _type, typename _hamtype>
inline void rbmState<_type, _hamtype>::set_rand_state()
{ 
    this->set_state(this->hamil->ran.randomInt_uni(0, this->hilbert_size - 1)); 
}

/*
* @brief sets the current angles vector according to arXiv:1606.02318v1
*/
template<typename _type, typename _hamtype>
inline void rbmState<_type, _hamtype>::set_angles()
{
    this->thetas = this->b_h + this->W * this->current_vector;
}

/*
* @brief sets the current angles vector according to arXiv:1606.02318v1
* @param v replaces current vector
*/
template<typename _type, typename _hamtype>
inline void rbmState<_type, _hamtype>::set_angles(const Col<double>& v)
{
    this->thetas = this->b_h + this->W * v;
}

/*
* @brief sets the weights according to the gradient descent - uses this->grad vector to update them
*/
template<typename _type, typename _hamtype>
void rbmState<typename _type, typename _hamtype>::set_weights() {
    // update weights accordingly
#pragma omp parallel for
    for (auto i = 0; i < this->n_visible; i++)
        this->b_v(i) += this->grad(i);
#pragma omp parallel for
    for (auto i = 0; i < this->n_hidden; i++) {
        const auto elem = i + this->n_visible;
        this->b_h(i) += this->grad(elem);
    }
#pragma omp parallel for
    for (auto i = 0; i < this->n_hidden; i++) {
        for (auto j = 0; j < this->n_visible; j++) {
            const auto elem = (this->n_visible + this->n_hidden) + i + j * this->n_hidden;
            this->W(i, j) += this->grad(elem);
        }
    }
}
// ------------------------------------------------- 				 CALCULATORS				  -------------------------------------------------

/*
* @brief calculates the variational derivative analytically
* @param v the base vector we want to calculate derivatives from
*/
template<typename _type, typename _hamtype>
void rbmState<_type, typename _hamtype>::calcVarDeriv(const Col<double>& v){
    auto var_deriv_time = std::chrono::high_resolution_clock::now();

#ifndef RBM_ANGLES_UPD
    this->set_angles(v);
#endif
    // calculate the flattened part
#pragma omp parallel for
    for (auto i = 0; i < this->n_visible; i++)
        this->O_flat(i) = v(i);
#pragma omp parallel for
    for (auto i = 0; i < this->n_hidden; i++) {
        const auto elem = i + this->n_visible;
        this->O_flat(elem) = std::tanh(this->thetas(i));
    }
#pragma omp parallel for
    for (auto i = 0; i < this->n_hidden; i++) {
        for (auto j = 0; j < this->n_visible; j++) {
            const auto elem = (this->n_visible + this->n_hidden) + i + j * this->n_hidden;
            const auto elem_hidden = i + this->n_visible;
            this->O_flat(elem) = this->O_flat(elem_hidden) * v(j);
        }
    }
    PRT(var_deriv_time, this->dbg_drvt)
}

/*
* @brief updates the weights using stochastic gradient descent
* @param current_step if we would like to optimize according to current mcstep
*/
template<typename _type, typename _hamtype>
void rbmState<typename _type, typename _hamtype>::updVarDeriv(int current_step){
    auto var_deriv_time_upd = std::chrono::high_resolution_clock::now();
   
    // update flat vector
#ifdef PINV
    this->grad = ((-this->lr) * (arma::pinv(this->S, pinv_tol)) * this->F);
#elif defined S_REGULAR 
    auto lambda_p = lambda_0_reg * this->current_b_reg; 
    if (lambda_p < lambda_min_reg) lambda_p = lambda_min_reg;
    this->S.diag() += lambda_p * S.diag();
    this->grad = (-this->lr) * arma::solve(this->S, this->F);
#else 
    this->grad = (-this->lr) * arma::solve(this->S, this->F);
#endif

    //this->grad = this->grad % arma::randi(this->full_size, distr_param(0, 1));
    //this->grad += this->lr * (exp(-current_step))  * arma::Col<_type>(this->full_size, arma::fill::randu);
#ifdef USE_ADAM
    this->adam->update(this->grad);
    this->grad = this->adam->get_grad();
#endif  
    this->set_weights();
    PRT(var_deriv_time_upd, this->dbg_updt)
}

/*
* @brief Calculate the local energy depending on the given Hamiltonian
*/
template<typename _type, typename _hamtype>
_type rbmState<typename _type, typename _hamtype>::locEn(){
    auto loc_en_time = std::chrono::high_resolution_clock::now();
    const auto hilb = this->hamil->get_hilbert_size();
    // get the reference to all local energies and changed states from the model Hamiltonian
    auto energies = this->hamil->get_localEnergyRef(this->current_state);
    _type energy = 0;
#ifndef DEBUG
#pragma omp parallel for reduction(+ : energy) shared(energies)
#endif
    for (auto i = 0; i < energies.size(); i++) 
    {
        const auto& [state, value] = energies[i];
        // if the state is not set
        if (state >= hilb)
            continue;

        _type v = value;
        // changes accordingly not to create data race
        if (state != this->current_state) {
        #ifndef DEBUG
            const int tid = omp_get_thread_num();
            const int vid = tid % this->thread_num;
        #else
            const int vid = 0;
        #endif

        INT_TO_BASE_BIT(state, this->tmp_vectors[vid]);

        #ifndef RBM_ANGLES_UPD
            v = v * this->pRatio(this->current_vector, this->tmp_vectors[vid]);
        #else
            v = v * this->pRatio(this->tmp_vectors[vid]);
        #endif
        }
        energy += v;
    }

    PRT(loc_en_time, this->dbg_lcen)
    return energy;
}

// ------------------------------------------------- SAMPLING -------------------------------------------------
/*
* @brief block updates the current state according to Metropolis-Hastings algorithm
* @param b_size the size of the correlation block
* @param start_state the state to start from
* @param n_flips number of flips at the single step
*/
template<typename _type, typename _hamtype>
void rbmState<typename _type, typename _hamtype>::blockSampling(size_t b_size, u64 start_state, size_t n_flips, bool thermalize){
    this->set_state(start_state, thermalize);

    // set the tmp_vector to current state
    this->tmp_vector = this->current_vector;

    for(auto i = 0; i < b_size; i++){

        const int flip_place = this->hamil->ran.randomInt_uni(0, this->n_visible - 1);
        const auto flip_spin = this->tmp_vector(flip_place);

        flipV(this->tmp_vector, flip_place);

        #ifndef RBM_ANGLES_UPD
        _type proba = std::abs(this->pRatio(this->current_vector, this->tmp_vector));
        #else
        _type proba = this->pRatio(this->tmp_vector);
        #endif
        if (this->hamil->ran.randomReal_uni() <= std::real(conj(proba) * proba)) {
            // update current state and vector
            this->current_state = flip(this->current_state, this->n_visible - 1 - flip_place);
            flipV(this->current_vector, flip_place);

            // update angles if needed
            #ifdef RBM_ANGLES_UPD
            this->update_angles(flip_place, flip_spin);
            #endif
        }
        else {
            // set the vector back to normal
            this->tmp_vector(flip_place) = flip_spin;
        }
    }
    // calculate the effective angles
}

/*
* @brief sample the vectors and converge model with gradients to find ground state
* @param n_samples number of samples to be used for training
* @param n_blocks number of correlation-reducers blocks
* @param n_therm number of steps to leave for thermalization
* @param b_size size of correlation-reducers blocks
* @param n_flips number of flips during a single step
* @returns energies obtained during each Monte Carlo step
*/
template<typename _type, typename _hamtype>
Col<_type> rbmState<typename _type, typename _hamtype>::mcSampling(size_t n_samples, size_t n_blocks, size_t n_therm, size_t b_size, size_t n_flips){
#ifdef S_REGULAR
    this->current_b_reg = b_reg;
#endif
    
    // start the timer!
    auto start = std::chrono::high_resolution_clock::now();
    // make the pbar!
    this->pbar = pBar(25, n_samples);
    // take out thermalization steps
    const auto mcsteps = n_samples - n_therm;
    // calculate the probability that we include the element in the batch
    const auto batch_proba = (this->batch / double(n_blocks));
    // check if the batch is not bigger than the blocks number
    const auto norm = (batch_proba > 1) ? n_blocks : this->batch;

    // save all average weights for covariance matrix
    Col<_type> averageWeights(this->full_size);
    Col<_type> energies(norm, arma::fill::zeros);

    Col<_type> meanEnergies(mcsteps, arma::fill::zeros);
    

    for(auto i = 0; i < mcsteps; i++){
        // set the random state at each Monte Carlo iteration
        this->set_rand_state();

        // start the simulation
        this->S.zeros();                                                                // Fisher info
        this->F.zeros();                                                                // Gradient force
        averageWeights.zeros();                                                         // Weights gradients average

        // thermalize system
        auto therm_time = std::chrono::high_resolution_clock::now();
        this->blockSampling(n_therm * b_size, this->current_state, n_flips, true);
        PRT(therm_time, this->dbg_thrm)

        // to check whether the batch is ready already
        size_t took = 1;                                                               
        auto blocks_time = std::chrono::high_resolution_clock::now();
        while(took <= norm){
            // block sample the stuff
            auto sample_time = std::chrono::high_resolution_clock::now();
            this->blockSampling(b_size, this->current_state, n_flips, false);
            PRT(sample_time, this->dbg_samp)

            if (norm == n_blocks || this->hamil->ran.randomReal_uni() <= batch_proba) {
                auto gradients_time = std::chrono::high_resolution_clock::now();
                this->calcVarDeriv(this->current_vector);
                // append local energies
                energies(took - 1) = this->locEn();

                // save conjugate first
                this->O_flat = arma::conj(this->O_flat);
                
                // append gradient forces with the first part of covariance <E_kO_k*>
                this->F += energies(took - 1) * this->O_flat;
                
                // append covariance matrices with the first part of covariance <O_k*O_k'>
                this->S += this->O_flat * this->O_flat.t();
                
                // average weight gradients
                averageWeights += this->O_flat;
                
                // append number of elements taken
                took += 1;
                PRT(gradients_time, this->dbg_grad)
            }
        }
        PRT(blocks_time, this->dbg_blck)

        // normalize
        averageWeights /= double(took);
        this->F /= double(took);
        this->S /= double(took);

        // update the covariance
        _type meanLocEn = arma::mean(energies);

        // append gradient forces with the first part of covariance <E_k><O_k*>
        this->F -= meanLocEn * averageWeights;
        
        // append covariance matrices with the first part of covariance <O_k*><O_k'>
        this->S -= averageWeights * averageWeights.t();

        // update model
        this->updVarDeriv(i);
        
        // add energy
        meanEnergies(i) = meanLocEn;
        
        #ifdef S_REGULAR
            this->current_b_reg = this->current_b_reg * b_reg;
        #endif // S_REGULAR

        // update the progress bar
        if (i % pbar.percentageSteps == 0)
            pbar.printWithTime("-> PROGRESS");
    }
    stouts("->\t\t\tMonte Carlo energy search ", start);
    return meanEnergies;
}

/*
* @brief sample the vectors and find the ground state and the operators
* @param n_samples number of samples to be used for training
* @param n_therm number of steps to leave for thermalization
* @param b_size size of correlation-reducers blocks
* @param n_flips number of flips during a single step
* @returns map of base state to the value of coefficient next to it
*/
template<typename _type, typename _hamtype>
inline std::map<u64, _type> rbmState<_type, _hamtype>::avSampling(size_t n_samples, size_t n_therm, size_t b_size, size_t n_flips)
{
    stout << "\t->Looking for the ground state for " + this->get_info();
    // start the timer!
    auto start = std::chrono::high_resolution_clock::now();

    // make the pbar!
    this->pbar = pBar(25, b_size);

    // states to be returned
    std::map<u64, _type> states;
    
    auto Ns = this->hamil->lattice->get_Ns();
    v_1d<int> sites(1, 0);



    _type en = 0;
    double s_z_rbm = 0;
    for (auto r = 0; r < b_size; r++) {
        // set the random state at each Monte Carlo iteration
        this->set_rand_state();

        // thermalize system
        auto therm_time = std::chrono::high_resolution_clock::now();
        this->blockSampling(n_therm * b_size, this->current_state, n_flips, true);
        PRT(therm_time, this->dbg_thrm)
        
        // go through samples
        for (auto i = 0; i < n_samples; i++) {

            // block sample the stuff
            auto sample_time = std::chrono::high_resolution_clock::now();
            this->blockSampling(b_size, this->current_state, n_flips, false);
            PRT(sample_time, this->dbg_samp)

            // look at the states coefficient (not found)
            if (!states.contains(this->current_state))
                states[this->current_state] = this->coeff(this->current_vector);

            // append local energies
            en += this->locEn();
            s_z_rbm += std::get<1>(sigma_z(this->current_state, Ns, sites));
        }
        // update the progress bar
        if (r % pbar.percentageSteps == 0)
            pbar.printWithTime("-> PROGRESS");
    }
    s_z_rbm /= double(n_samples * b_size);
    en /= double(n_samples * b_size);

    stouts("->\t\t\tMonte Carlo state search after finding weights ", start);
    stout << "\n------------------------------------------------------------------------" << EL;
    stout << "GROUND STATE RBM:" << EL;
    this->pretty_print(states, 0.05);
    stout << VEQP(s_z_rbm,5) << EL;
    stout << "\n------------------------------------------------------------------------" << EL;

    return states;

}

#endif // !RBM_H