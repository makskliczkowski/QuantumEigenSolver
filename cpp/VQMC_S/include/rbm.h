#include "../src/binary.h"
#include "../src/progress.h"

template <typename _type>
class rbmState{

private:
    // general parameters
    std::string info;                                           // info about the model
    size_t n_visible;                                           // visible neurons
    size_t n_hidden;                                            // hidden neurons
    size_t full_size;                                           // full size of the parameters
    double lr, momentum, xavier_const;                          // learning rate, momentum and xavier initializer constant
    size_t epo, nBins, nBlocks;                                 // number of epochs, number of bins for correlation removal, number of blocks of bins
    randomGen ran;										        // consistent quick random number generator
    pBar pbar;                                                  // progress bar
    
    // network weights
    Mat<_type> W;                                               // weight matrix
    arma::Col<_type> b_v;                                       // visible bias
    arma::Col<_type> b_h;                                       // hidden bias

    // variational derivatives                                  
    arma::Col<_type> O_b_v;                                     // gradient of the visible bias
    arma::Col<_type> O_b_h;                                     // gradient of the hidden bias
    arma::Col<_type> O_w;                                       // gradient of the weight matrices
    arma::Col<_type> O_flat;                                    // flattened output for easier calculation of the covariance
    arma::Mat<_type> S;                                         // positive semi-definite covariance matrix
    arma::Col<_type> F;                                         // forces
    
    // the Hamiltonian
    std::unique_ptr<SpinHamiltonian> hamil;                     // unique ptr to a general Hamiltonian of the spin system

    // saved training parameters
    u64 current_state;                                          // current state during the simulation
    arma::Col<int> current_vector;                              // current state vector during the simulation
    arma::Col<int> tmp_vector;                                  // tmp state vector during the simulation

public:
    rbmState(size_t nH, size_t nV,
            double lr, double mom, double xav,
            size_t epo, size_t nB, size_t nBlock
            ) 
            : n_hidden(nH), n_visible(nV)
            , lr(lr), momentum(mom)
            , xavier_const(xav)
            , epo(epo)
            , nBins(nB), nBlocks(nBlock)
            {
                this->allocate();
                this->init();

            };
    
    // allocate the memory for the biases and weights
    void allocate();
    // initialize all
    void init();

    // ------------------------------------------- AMPLITUDES AND ANSTATZ REPRESENTATION -------------------------------------------

    // the hiperbolic cosine of the parameters
    inline arma::Col<_type> Fs(const arma::Col<int>& v) const{return arma::cosh(this->b_h + this->W * v);};

    // get the current amplitude
    inline _type coeff(u64 state){
        intToBase(state, this->current_vector);
        return exp(cdot(this->b_v, this->current_vector)) * arma::prod(Fs(this->current_vector));
        };
    // get the current amplitude given vector
    inline _type coeff(const arma::Col<int>& v) const {return exp(cdot(this->b_v, this->current_vector)) * arma::prod(Fs(v));};

    // get probability ratio for a reference state v1 and v2 state
    inline _type pRatio(const arma::Col<int>& v1, const arma::Col<int>& v2) const {return exp(cdot(this->b_v, (v2-v1)) + arma::sum(arma::log(Fs(v2)/Fs(v1))));};

    // variational derivative calculation
    void calcVarDeriv(const arma::Col<int>& v1);

    // sample block
    void blockSampling(size_t b_size, u64 start_stae, size_t n_flips = 1);

    // sample the probabilistic space
    void mcSampling(size_t n_samples, size_t n_blocks, size_t b_size, u64 start_state, size_t n_flips = 1);
    

        


};


/*
* @param v the base vector we want to calculate derivatives from
*/
template<typename T>
void rbmState<T>::calcVarDeriv(const arma::Col<int>& v){
    const auto theta = this->b_h + arma::dot(this->W, v;
    // change of the first bias
    this->O_b_v = v;
    // change of the hidden bias
    this->O_b_h = arma::tanh(theta);
    // change of the weights matrix
    this->O_w = arma::dot(v, this->O_b_h.t());
    // calculate the flattened part
    this->O_flat.subvec(0, this->n_visible - 1) = this->O_b_v;
    this->O_flat.subvec(this->n_visible, this->n_hidden-1) = this->O_b_h;
    this->O_flat.subvec(this->n_visible + this->n_hidden,
                        this->full_size -1) = this->O_w.as_col();
}

/*
* @param v the base vector we want to calculate derivatives from
*/
template<typename T>
void rbmState<T>::blockSampling(size_t b_size, u64 start_state, size_t n_flips){
    this->current_state = start_state;
    intToBase(this->current_state, this->current_vector, 2);
    for(auto i = 0; i < b_size; i++){
        u64 tmp_state = this->current_state;
        for(auto j = 0; j < n_flips; j++)
            tmp_state = flip(tmp_state, this->n_visible, this->ran.randomInt_uni(0, this->n_visible));
        // create the vector
        intToBase(tmp_state, tmp_vector, 2);
        if(this->ran.randomReal_uni() <= abs(this->pRatio(this->current_state, tmp_state))^2)
            this->current_state = tmp_state;
            this->current_vector = tmp_vector;
    }
}

/*
* @param v the base vector we want to calculate derivatives from
*/
template<typename T>
void rbmState<T>::mcSampling(size_t n_samples, size_t n_blocks, size_t b_size, u64 start_state, size_t n_flips){
    auto start = std::chrono::high_resolution_clock::now();
    this->pbar(34, n_samples);
    
    arma::Col<T> energies(n_blocks, arma::fill::zeros);
    arma::Col<T> averageWeights(this->full_size, arma::fill::zeros);

    for(auto i = 0; i < n_samples; i++){
        this->S.zeros();                        // Fisher info
        this->F.zeros();                        // Gradient force
        energies.zeros();
        averageWeights.zeros();
        for(auto k = 0; k < n_blocks; k++){
            this->blockSampling(b_size, this->current_state, n_flips);
            this->calcVarDeriv(this->current_vector)
            energies(k) = this->local_energy();
            
        }


    }


//#pragma omp critical
    //stout << "For: " << this->get_info() << "->\n\t\t\t\tRelax time taken: " << tim_s(start) << " seconds. With sign: " << (pos_num - neg_num) / (1.0 * (pos_num + neg_num)) << "\n";

}

