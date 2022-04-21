#pragma once
#ifndef COMMON_H
#include "../src/common.h"
#endif

#ifndef ML_H
#define ML_H

template<typename _type>
class Adam {
protected:
	size_t size;								// size of the gradient
	size_t current_time = 0;					// current iteration - starts from zero 
	double beta1 = 0.9;							// 1st order exponential decay
	double beta2 = 0.99;						// 2nd order exponential decay
	double eps = 1e-8;							// prevent zero-division
	double lr = 1e-2;							// learning step rate
	cpx alpha = 0;								// true learning step
	arma::Col<_type> m;							// moment vector
	arma::Col<_type> v;							// norm
public:
	// ---------------------------
	~Adam() = default;
	Adam() = default;
	Adam(double lr, size_t size)
		: lr(lr), size(size)
	{
		this->alpha = lr;
		// initialize to zeros
		this->m = arma::Col<_type>(size, arma::fill::zeros);
		this->v = arma::Col<_type>(size, arma::fill::zeros);
	};
	Adam(double beta1, double beta2, double lr, double eps, size_t size)
		: beta1(beta1), beta2(beta2), lr(lr), eps(eps), size(size)
	{
		this->alpha = lr;
		// initialize to zeros
		this->m = arma::Col<_type>(size, arma::fill::zeros);
		this->v = arma::Col<_type>(size, arma::fill::zeros);
	};
	/*
	* resets Adam
	*/
	void reset() {
		this->current_time = 0;
		this->m.zeros();
		this->v.zeros();
	}
	/*
	* updates Adam
	*/
	void update(const arma::Col<_type>& grad) {
		this->current_time += 1;
		this->m = beta1 * this->m + (1.0 - beta1) * grad;
		this->v = beta2 * this->v + (1.0 - beta2) * arma::square(grad);
		// update decays
		this->beta1 *= this->beta1;
		this->beta2 *= this->beta2;
		this->alpha = this->lr * sqrt(1.0 - this->beta2) / (1.0 - this->beta1);
	};
	/*
	* get the gradient :)
	*/
	arma::Col<_type> get_grad()				const { return this->alpha * this->m / (arma::sqrt(this->v) + this->eps); };

};

#endif