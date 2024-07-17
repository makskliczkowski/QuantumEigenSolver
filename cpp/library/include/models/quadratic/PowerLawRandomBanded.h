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
		LOGINFO("I am Power Law Random Banded model: ", LOG_TYPES::CHOICE, 2);
	};


	// ### H A M I L T O N I A N ###

	void hamiltonian() override										
	{ 
		if (this->Nh == 0)
		{
			LOGINFOG("Empty Hilbert, not building anything.", LOG_TYPES::INFO, 1);
			return;
		}
		this->init();

		// go through the Hamiltonian and set the elements
		for (size_t i = 0; i < this->Nh_; i++)
		{
			for (size_t j = i; j < this->Nh_; j++)
			{
				double _ranval		= this->ran_.random(-1.0, 1.0);
				double _distance	= 0.0;
				if (i != j)
				{
					_distance = std::abs(int(i - j)) / this->b_;
					_distance = std::pow(_distance, 2 * this->a_);
				}

				// set Hamiltonian element
				auto _val = _ranval / std::sqrt(1.0 + _distance);
				this->H_.set(i, j, _val);
				this->H_.set(j, i, _val);
			}
		}
	}

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

#endif // !POWER_LAW_RANDOM_BANDED_H

