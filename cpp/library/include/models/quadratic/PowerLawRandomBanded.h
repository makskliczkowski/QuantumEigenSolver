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
public:
	~PowerLawRandomBanded()		= default;
	PowerLawRandomBanded(size_t _Ns, double _a = 1.0, double _b = 1.0, double _constant = 0.0)
		: QuadraticHamiltonian<_T>(_Ns, _constant, true), a_(_a), b_(_b)
	{
		this->type_ = MY_MODELS::POWER_LAW_RANDOM_BANDED_M;
		this->info_ = this->info();
		LOGINFO("I am Power Law Random Banded model: ", LOG_TYPES::CHOICE, 2);
	};
	PowerLawRandomBanded(std::shared_ptr<Lattice> _lat, double _a = 1.0, double _b = 1.0, double _constant = 0.0)
		: QuadraticHamiltonian<_T>(_lat, _constant, true), a_(_a), b_(_b)
	{
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
		for (size_t i = 0; i < this->Ns; i++)
		{
			for (size_t j = i; j < this->Ns; j++)
			{
				double _ranval = this->ran_.random(-1.0, 1.0);
				double _distance = 0.0;
				if (i != j)
				{
					_distance = std::abs(int(i - j)) / this->b_;
					_distance = std::pow(_distance, 2 * this->a_);
				}

				// set Hamiltonian element
				this->H_(i, j) = _ranval / std::sqrt(1.0 + _distance);
			}
		}

		this->H_ = this->ran_.template GOE<_T>(this->Nh, this->Nh) + algebra::cast<_T>(I) * arma::zeros(this->Nh, this->Nh); 
	}

	// ------------------------------------------- 				 Info				  -------------------------------------------

	std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const override
	{
		std::string name = sep + "PLRB,Ns=" + STR(this->Ns);
		name += ",a=" + STRP(a_, 3);
		name += ",b=" + STRP(b_, 3);
		return this->QuadraticHamiltonian<_T>::info(name, skip, sep);
	}
	void updateInfo()									override final { this->info_ = this->info(); };
};


#endif // !SYK2_M_H
