#pragma once

#ifndef HAMIL_QUADRATIC_H
#include "../../hamilQ.h"
#endif

#ifndef POWER_LAW_RANDOM_BANDED_H
#define POWER_LAW_RANDOM_BANDED_H

/*
* @brief: This class is an instance of the Power Law Random Banded model.
*/
template<typename _T>
class PowerLawRandomBanded : public QuadraticHamiltonian<_T>
{
	double a_ = 1.0;
	double b_ = 1.0;
protected:
	void checkQuadratic() override;
public:
	~PowerLawRandomBanded()
	{
		DESTRUCTOR_CALL;
	}
	PowerLawRandomBanded(size_t _Ns, double _a = 1.0, double _b = 1.0, bool _mb = false, double _constant = 0.0)
		: QuadraticHamiltonian<_T>(_Ns, _constant, true, false), a_(_a), b_(_b)
	{
		this->Ns_	= _Ns;
		this->Ns	= this->Ns_;
		this->Nh_	= (_mb && _Ns < UI_LIMITS_MAXFULLED) ? ULLPOW(_Ns) : _Ns;
		this->Nh	= this->Nh_;
		this->type_ = MY_MODELS::POWER_LAW_RANDOM_BANDED_M;
		this->info_ = this->info();
		// check if the model is quadratic
		this->checkQuadratic();
		LOGINFO("I am Power Law Random Banded model: ", LOG_TYPES::CHOICE, 2);
	};
	PowerLawRandomBanded(std::shared_ptr<Lattice> _lat, double _a = 1.0, double _b = 1.0, bool _mb = false, double _constant = 0.0)
		: QuadraticHamiltonian<_T>(_lat, _constant, true, false), a_(_a), b_(_b)
	{
		auto _Ns	= _lat->get_Ns();
		this->Ns_	= _Ns;
		this->Ns	= this->Ns_;
		this->Nh_	= (_mb && _Ns < UI_LIMITS_MAXFULLED) ? ULLPOW(_Ns) : _Ns;
		this->Nh	= this->Nh_;
		this->type_ = MY_MODELS::POWER_LAW_RANDOM_BANDED_M;
		this->info_ = this->info();
		// check if the model is quadratic
		this->checkQuadratic();
		LOGINFO("I am Power Law Random Banded model: ", LOG_TYPES::CHOICE, 2);
	};
	// Hilbert space constructor
	PowerLawRandomBanded(const Hilbert::HilbertSpace<_T>& _hil, double _a = 1.0, double _b = 1.0, bool _mb = false, double _constant = 0.0)
		: QuadraticHamiltonian<_T>(_hil, _constant, true, false), a_(_a), b_(_b)
	{
		auto _Ns	= _hil.get_Ns();
		this->Ns_	= _Ns;
		this->Ns	= this->Ns_;
		this->Nh_	= (_mb && _Ns < UI_LIMITS_MAXFULLED) ? ULLPOW(_Ns) : _Ns;
		this->Nh	= this->Nh_;
		this->type_ = MY_MODELS::POWER_LAW_RANDOM_BANDED_M;
		this->info_ = this->info();
		// check if the model is quadratic
		this->checkQuadratic();
		LOGINFO("I am Power Law Random Banded model: ", LOG_TYPES::CHOICE, 2);
	};
	// Hilbert space constructor move
	PowerLawRandomBanded(Hilbert::HilbertSpace<_T>&& _hil, double _a = 1.0, double _b = 1.0, bool _mb = false, double _constant = 0.0)
		: QuadraticHamiltonian<_T>(std::move(_hil), _constant, true, false), a_(_a), b_(_b)
	{
		auto _Ns	= this->Ns_;
		this->Ns_	= _Ns;
		this->Ns	= this->Ns_;
		this->Nh_	= (_mb && _Ns < UI_LIMITS_MAXFULLED) ? ULLPOW(_Ns) : _Ns;
		this->Nh	= this->Nh_;
		this->type_ = MY_MODELS::POWER_LAW_RANDOM_BANDED_M;
		this->info_ = this->info();
		// check if the model is quadratic
		this->checkQuadratic();
		LOGINFO("I am Power Law Random Banded model: ", LOG_TYPES::CHOICE, 2);
	};


	// ### H A M I L T O N I A N ###

	void hamiltonian() override;

	// ------------------------------------------- 				 Info				  -------------------------------------------

	std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const override
	{
		std::string name = sep + "plrb,Ns=" + STR(this->Ns);
		name += ",a=" + STRP(a_, 3);
		name += ",b=" + STRP(b_, 3);
		return this->QuadraticHamiltonian<_T>::info(name, skip, sep);
	}
	void updateInfo()									override final { this->info_ = this->info(); };
};

template<typename _T>
inline void PowerLawRandomBanded<_T>::checkQuadratic()
{
	if (this->Ns > UI_LIMITS_MAXFULLED) 
	{
		LOGINFOG("The number of particles is too big for it being quadratic. Be reasonable!", LOG_TYPES::WARNING, 1);
		this->isManyBody_	= false;
		this->isQuadratic_	= true;
		this->Nh_			= this->Ns_;
		this->Nh			= this->Ns_;
	}
	else
	{
		this->isManyBody_	= true;
		this->isQuadratic_	= true;
	}
}

// ### H A M I L T O N I A N ###

/*
* @brief: This function builds the Hamiltonian for the Power Law Random Banded model.
* @note: The Hamiltonian is built in the following way:
*		- The Hamiltonian is a banded matrix with the bandwidth of 1.
*		- The elements of the Hamiltonian are random numbers between -1 and 1.
*		- The elements are divided by the square root of the distance between the particles.
*		- The distance is calculated as the difference between the indices of the particles in the Hilbert space (either single or many body).
*/
template<typename _T>
inline void PowerLawRandomBanded<_T>::hamiltonian()
{
	if (this->Nh == 0)
	{
		LOGINFOG("Empty Hilbert, not building anything.", LOG_TYPES::INFO, 1);
		return;
	}
	this->init();

	const double _power = 2.0 * this->a_; 
	const double _binv  = 1.0 / this->b_;
	// go through the Hamiltonian and set the elements
	// notice that the Hamiltonian is symmetric and it is systematically
	// shrinked to the offdiagonal. Additionally, the elements are going through 
	// a power law decay (in single particle) but this changes to many body (exponential)
	// when the distance between the particles is calculated in many body Hilbert space.
	// (according to this, the Ns = log2(Nh) and the distance is calculated as the difference
	// between the indices of the particles in this big Hilbert space)
	for (size_t i = 0; i < this->Nh_; i++)
	{
		for (size_t j = i; j < this->Nh_; j++)
		{
			double _distance	= 0.0;
			if (i != j)
			{
				_distance = (long double)std::abs(int(i - j)) * _binv;
				_distance = std::pow(_distance, _power);
			}

			// set Hamiltonian element
			auto _val = this->ran_.random(-1.0, 1.0) / std::sqrt(1.0 + _distance);
			this->H_.set(i, j, _val);
			// do I need to set the other side of the matrix?
			this->H_.set(j, i, _val);
		}
	}
}

#endif // !POWER_LAW_RANDOM_BANDED_H

