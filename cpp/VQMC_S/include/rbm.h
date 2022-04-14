#include "../src/binary.h"

template <typename _type>
class rbmState{

private:
    std::string info;                                           // info about the model
    size_t n_visible;                                           // visible neurons
    size_t n_hidden;                                            // hidden neurons
    double lr, momentum, xavier_const;                          // learning rate, momentum and xavier initializer constant
    size_t epo, nBins, nBlocks;                                 // number of epochs, number of bins for correlation removal, number of blocks of bins

    // network weights
    Mat<_type> W;                                               // weight matrix
    arma::Col<_type> b_v;                                       // visible bias
    arma::Col<_type> b_h;                                       // hidden bias

    // variational derivatives                                  
    arma::Col<_type> O_b_v;                                     // gradient of the visible bias
    arma::Col<_type> O_b_h;                                     // gradient of the hidden bias
    arma::Col<_type> O_w;                                       // gradient of the weight matrices

    // the Hamiltonian
    std::unique_ptr<SpinHamiltonian> hamil;                     // unique ptr to a general Hamiltonian of the spin system

    // saved training parameters
    u64 current_state;                                          // current state during the simulation
    arma::Col<int> current_vector;                              // current state vector during the simulation

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

    // sample the probabilistic space
    void mcSampling(size_t n_samples, size_t n_flips = 1);
    

        


};