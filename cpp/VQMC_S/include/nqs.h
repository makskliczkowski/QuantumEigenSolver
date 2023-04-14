#pragma once

#ifndef NQS_H
#define NQS_H

// ##############################
// include the hamiltonian	 // #
#ifndef HAMIL_H				 // #
	#include "hamil.h"       // #
#endif // !HAMIL_H           // #
// ##############################

// include Machine Learning Stuff
// ##############################
#ifndef ML_H				 // #
	#include "ml.h"			 // #
#endif // !ML_H				 // #
// ##############################

#include "omp.h"

#ifdef NQS_SAVE_WEIGHTS
	#define NQS_SAVE_DIR "DATA" + kPS + "NQS" + kPS + "WEIGHTS" + kPS
#endif

template <typename _T, typename _Ht>
class NQS {
public:
	using NQSS							=		arma::Col<double>;
	using NQSB							=		arma::Col<_T>;

protected:
	std::string info_;
	pBar pBar_;																// for printing out the progress

	uint threadNum_						=		1;							// number of threads that works on this
	uint batch_							=		1;							// size of the batch
	uint nVis_							=		1;							// number of visible neurons
	uint fullSize_						=		1;							// full number of the parameters
	double lr_							=		0.0;						// specific learnign rate

	// Hamiltonian
	std::shared_ptr<Hamiltonian<_Ht>> H_;									// pointer to the Hamiltonian

	// Hilbert space info
	u64 Nh								=		1;

	// Random number generator
	randomGen ran_;														// consistent quick random number generator


	// --------------------- O P T I M I Z E R S --------------------


	// --------------------- T R A I N I N G ---------------------
	v_1d<uint> flipPlaces_;
	v_1d<double> flipVals_;
	u64 curState;
	NQSS curVec;
	NQSS tmpVec;

public:
	NQS()							=			default;
	
	/*
	* @brief interface constructor for Neural Network Quantum State
	*/
	NQS(std::shared_ptr<Hamiltonian<_Ht>>& _H, uint _batch, uint _threadNum, double _lr)
		: batch_(_batch), lr_(_lr)
	{
		this->H_					=			_H;
		this->nVis_					=			_H->getNs();
		this->ran_					=			_H->ran_;
#ifdef DEBUG
		this->threadNum_			=			1;
#else
		this->threadNum_			=			_threadNum;
#endif // DEBUG
		// Use threads for all consecutive parallel regions
		omp_set_num_threads(this->threadNum_);                  

		// set corresponding things
		this->allocate();
		this->setInfo();
		this->init();
		this->setRandomState();
	};

protected:
	// --------------------- S T A R T E R S -------------------
	virtual void setInfo()							=						0;
	virtual void allocate()							=						0;

	// --------------------- S E T T E R S ---------------------
	virtual void setState(NQSS _st, bool _set)		=						0;
	virtual void setState(NQSS _st)					=						0;
	virtual void setState(u64 _st)					=						0;
	virtual void setState(u64 _st, bool _set)		=						0;
	
	// --------------------- W E I G H T S ---------------------
	virtual void updateWeights()					=						0;

	// --------------------- T R A I N   E T C -----------------
	virtual void grad(const NQSS& _v)				=						0;
	
	// ---------------------------------------------------------
public:
	// --------------------- S E T T E R S ---------------------
	virtual void init()								=						0;
	virtual void init(const std::string& _p)		=						0;
	virtual void setRandomState()					=						0;
	
	// --------------------- G E T T E R S ---------------------
	auto getInfo()								    const -> std::string	{ return this->info_; };
	auto getNvis()									const -> uint			{ return this->nVis_; };

	// --------------------- S A M P L I N G ---------------------
	arma::Col<_T> train(uint nSam, uint nThrm, uint nBlck, uint bSize, uint nFlip = 1)		{};
	arma::Col<_T> collect(uint nSam, uint nThrm, uint nBlck, uint bSize, uint nFlip = 1)	{};

	// --------------------- F I N A L E -------------------------
	virtual _T ansatz(const NQSS& _in) const		=						0;
};

#endif