#ifndef ML_H
#define ML_H
    struct ML_PARAMS
    {
        double lr_          = 1e-2;         // learning rate
    };

#endif

//#ifndef COMMON_H
//#include "../src/common.h"
//#endif
//
//#ifndef ML_H
//#define ML_H
//
//template<typename _type>
//class Adam {
//private:
//	uint size;								// size of the gradient
//	uint current_time = 0;					// current iteration - starts from zero
//	double beta1_0 = 0.9;						// 1st order exponential decay starting parameter
//	double beta1 = 0.9;							// 1st order exponential decay
//	double beta2_0 = 0.99;						// 2nd order exponential decay starting parameter
//	double beta2 = 0.99;						// 2nd order exponential decay
//	double eps = 1e-8;							// prevent zero-division
//	double lr;									// learning step rate
//	cpx alpha = 0;								// true learning step
//	arma::Col<_type> m;							// moment vector
//	arma::Col<_type> v;							// norm
//	arma::Col<_type> gradient;					// gradient
//public:
//	// ---------------------------
//	~Adam() = default;
//	Adam() = default;
//	Adam(double lr, uint size)
//		: lr(lr), size(size)
//	{
//		this->beta1 = this->beta1_0;
//		this->beta2 = this->beta2_0;
//		this->alpha = lr;
//		this->initialize();
//	};
//	Adam(double beta1, double beta2, double lr, double eps, uint size)
//		: beta1(beta1), beta2(beta2), lr(lr), eps(eps), size(size)
//	{
//		this->beta1_0 = beta1;
//		this->beta2_0 = beta2;
//		this->alpha = lr;
//		this->initialize();
//	};
//	/*
//	* resets Adam
//	*/
//	void reset() {
//		this->current_time = 0;
//		this->beta1 = this->beta1_0;
//		this->beta2 = this->beta2_0;
//		this->m.zeros();
//		this->v.zeros();
//		this->gradient.zeros();
//	}
//	/*
//	* initialize Adam
//	*/
//	void initialize() {
//		// initialize to zeros
//		this->m = arma::Col<_type>(size, arma::fill::zeros);
//		this->v = arma::Col<_type>(size, arma::fill::zeros);
//		this->gradient = arma::Col<_type>(size, arma::fill::zeros);
//	}
//
//	/*
//	* updates Adam
//	*/
//	void update(const arma::Col<_type>& grad) {
//		this->current_time += 1;
//		this->m = this->beta1_0 * this->m + (1.0 - this->beta1_0) * grad;
//		this->v = this->beta2_0 * this->v + (1.0 - this->beta2_0) * arma::square(grad);
//		// update decays
//		this->beta1 *= this->beta1_0;
//		this->beta2 *= this->beta2_0;
//
//		this->alpha = this->lr * (1.0 - this->beta2) / (1.0 - this->beta1);
//		// calculate the new gradient according to Adam
//		this->gradient = this->alpha * this->m / (arma::sqrt(this->v) + this->eps);
//	};
//	/*
//	* get the gradient :)
//	*/
//	const arma::Col<_type>& get_grad()				const { return this->gradient; };
//	arma::Col<_type> get_grad_cpy()					const { return this->gradient; };
//};
//
//
//
//template<typename _type>
//class RMSprop_mod {
//private:
//	uint size;								// size of the gradient
//	uint current_time = 0;					// current iteration - starts from zero
//	double beta_0 = 0.9;						// exponential decay starting parameter
//	double beta = 0.9;							// exponential decay
//	double eps = 1e-8;							// prevent zero-division
//	double lr;									// learning step rate
//
//	arma::Col<_type> v;							// norm
//	arma::Col<_type> gradient;					// gradient
//public:
//	// ---------------------------
//	~RMSprop_mod() = default;
//	RMSprop_mod() = default;
//	RMSprop_mod(double lr, uint size)
//		: lr(lr), size(size)
//	{
//		this->beta_0 = 0.9;
//		this->beta = 0.9;
//		this->initialize();
//	};
//	RMSprop_mod(double beta, double lr, double eps, uint size)
//		: beta(beta), lr(lr), eps(eps), size(size)
//	{
//		this->beta_0 = beta;
//		this->initialize();
//	};
//	/*
//	* resets Adam
//	*/
//	void reset() {
//		this->current_time = 0;
//		this->beta = this->beta_0;
//		this->v.zeros();
//		this->gradient.zeros();
//	}
//	/*
//	* initialize Adam
//	*/
//	void initialize() {
//		// initialize to zeros
//		this->v = arma::Col<_type>(size, arma::fill::zeros);
//		this->gradient = arma::Col<_type>(size, arma::fill::zeros);
//	}
//
//	/*
//	* updates Adam
//	*/
//	void update(const arma::Col<_type>& grad, const arma::Col<_type>& O) {
//		this->current_time += 1;
//		this->v = this->beta * this->v + (1.0 - this->beta) * O % arma::conj(O);
//
//		//this->v.print("v");
//		// calculate the new gradient according to RMSProp arXiv:1910.11163v2
//		this->gradient = this->lr * grad / (arma::sqrt(this->v) + this->eps);
//		//this->gradient.print("grad");
//	};
//	/*
//	* get the gradient :)
//	*/
//	const arma::Col<_type>& get_grad()				const { return this->gradient; };
//	arma::Col<_type> get_grad_cpy()					const { return this->gradient; };
//};
//
//
//
//
//
//
//#endif#pragma once
//#ifndef COMMON_H
//#include "../src/common.h"
//#endif
//
//#ifndef ML_H
//#define ML_H
//
//template<typename _type>
//class Adam {
//private:
//	uint size;								// size of the gradient
//	uint current_time = 0;					// current iteration - starts from zero
//	double beta1_0 = 0.9;						// 1st order exponential decay starting parameter
//	double beta1 = 0.9;							// 1st order exponential decay
//	double beta2_0 = 0.99;						// 2nd order exponential decay starting parameter
//	double beta2 = 0.99;						// 2nd order exponential decay
//	double eps = 1e-8;							// prevent zero-division
//	double lr;									// learning step rate
//	cpx alpha = 0;								// true learning step
//	arma::Col<_type> m;							// moment vector
//	arma::Col<_type> v;							// norm
//	arma::Col<_type> gradient;					// gradient
//public:
//	// ---------------------------
//	~Adam() = default;
//	Adam() = default;
//	Adam(double lr, uint size)
//		: lr(lr), size(size)
//	{
//		this->beta1 = this->beta1_0;
//		this->beta2 = this->beta2_0;
//		this->alpha = lr;
//		this->initialize();
//	};
//	Adam(double beta1, double beta2, double lr, double eps, uint size)
//		: beta1(beta1), beta2(beta2), lr(lr), eps(eps), size(size)
//	{
//		this->beta1_0 = beta1;
//		this->beta2_0 = beta2;
//		this->alpha = lr;
//		this->initialize();
//	};
//	/*
//	* resets Adam
//	*/
//	void reset() {
//		this->current_time = 0;
//		this->beta1 = this->beta1_0;
//		this->beta2 = this->beta2_0;
//		this->m.zeros();
//		this->v.zeros();
//		this->gradient.zeros();
//	}
//	/*
//	* initialize Adam
//	*/
//	void initialize() {
//		// initialize to zeros
//		this->m = arma::Col<_type>(size, arma::fill::zeros);
//		this->v = arma::Col<_type>(size, arma::fill::zeros);
//		this->gradient = arma::Col<_type>(size, arma::fill::zeros);
//	}
//
//	/*
//	* updates Adam
//	*/
//	void update(const arma::Col<_type>& grad) {
//		this->current_time += 1;
//		this->m = this->beta1_0 * this->m + (1.0 - this->beta1_0) * grad;
//		this->v = this->beta2_0 * this->v + (1.0 - this->beta2_0) * arma::square(grad);
//		// update decays
//		this->beta1 *= this->beta1_0;
//		this->beta2 *= this->beta2_0;
//
//		this->alpha = this->lr * (1.0 - this->beta2) / (1.0 - this->beta1);
//		// calculate the new gradient according to Adam
//		this->gradient = this->alpha * this->m / (arma::sqrt(this->v) + this->eps);
//	};
//	/*
//	* get the gradient :)
//	*/
//	const arma::Col<_type>& get_grad()				const { return this->gradient; };
//	arma::Col<_type> get_grad_cpy()					const { return this->gradient; };
//};
//
//
//
//template<typename _type>
//class RMSprop_mod {
//private:
//	uint size;								// size of the gradient
//	uint current_time = 0;					// current iteration - starts from zero
//	double beta_0 = 0.9;						// exponential decay starting parameter
//	double beta = 0.9;							// exponential decay
//	double eps = 1e-8;							// prevent zero-division
//	double lr;									// learning step rate
//
//	arma::Col<_type> v;							// norm
//	arma::Col<_type> gradient;					// gradient
//public:
//	// ---------------------------
//	~RMSprop_mod() = default;
//	RMSprop_mod() = default;
//	RMSprop_mod(double lr, uint size)
//		: lr(lr), size(size)
//	{
//		this->beta_0 = 0.9;
//		this->beta = 0.9;
//		this->initialize();
//	};
//	RMSprop_mod(double beta, double lr, double eps, uint size)
//		: beta(beta), lr(lr), eps(eps), size(size)
//	{
//		this->beta_0 = beta;
//		this->initialize();
//	};
//	/*
//	* resets Adam
//	*/
//	void reset() {
//		this->current_time = 0;
//		this->beta = this->beta_0;
//		this->v.zeros();
//		this->gradient.zeros();
//	}
//	/*
//	* initialize Adam
//	*/
//	void initialize() {
//		// initialize to zeros
//		this->v = arma::Col<_type>(size, arma::fill::zeros);
//		this->gradient = arma::Col<_type>(size, arma::fill::zeros);
//	}
//
//	/*
//	* updates Adam
//	*/
//	void update(const arma::Col<_type>& grad, const arma::Col<_type>& O) {
//		this->current_time += 1;
//		this->v = this->beta * this->v + (1.0 - this->beta) * O % arma::conj(O);
//
//		//this->v.print("v");
//		// calculate the new gradient according to RMSProp arXiv:1910.11163v2
//		this->gradient = this->lr * grad / (arma::sqrt(this->v) + this->eps);
//		//this->gradient.print("grad");
//	};
//	/*
//	* get the gradient :)
//	*/
//	const arma::Col<_type>& get_grad()				const { return this->gradient; };
//	arma::Col<_type> get_grad_cpy()					const { return this->gradient; };
//};
//
//
//
//
//
//
//#endif