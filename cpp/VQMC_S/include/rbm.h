#include "../src/binary.h"

template <typename _type>
class rbmState{

private:
    std::string info;                                           // info about the model
    int n_visible;                                              // visible neurons
    int n_hidden;                                               // hidden neurons
    double lr, momentum, xavier_const;                          // learning rate, momentum and xavier initializer constant
    int epo, nBins, nBlocks;                                    // number of epochs, number of bins for correlation removal, number of blocks of bins
};