#pragma once

#ifndef HAMIL_QUADRATIC_H
	#include "../../hamilQ.h"
#endif

#ifndef SYK2_M_H
#define SYK2_M_H

template<typename _T>
class SYK2 : public QuadraticHamiltonian<_T>
{
public:
	~SYK2()		= default;
	SYK2(std::shared_ptr<Lattice> _lat, double _constant = 0.0)
		: QuadraticHamiltonian<_T>(_lat, _constant, true)
	{
		this->type_ = MY_MODELS_Q::SYK2_M;
		this->info_ = this->info();
		LOGINFO("I am SYK2 model: ", LOG_TYPES::CHOICE, 2);
	};

	// ### H A M I L T O N I A N ###

	void hamiltonian() override										
	{ 
		this->init();
		this->H_ = this->ran_.GOE(this->Nh, this->Nh) + algebra::cast<_T>(I) * arma::zeros(this->Nh, this->Nh); 
	}

	// ------------------------------------------- 				 Info				  -------------------------------------------

	std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const override
	{
		auto Ns = this->lat_->get_Ns();
		auto BC = this->lat_->get_BC();
		std::string name = sep + "SYK2,Ns=" + STR(Ns);
		name += ",BC=" + SSTR(getSTR_BoundaryConditions(BC));
		return this->QuadraticHamiltonian<_T>::info(name, skip, sep);
	}
	void updateInfo()									override final { this->info_ = this->info(); };
};


#endif // !SYK2_M_H
