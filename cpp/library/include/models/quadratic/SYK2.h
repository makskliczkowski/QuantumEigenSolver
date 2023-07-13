#pragma once

#ifndef HAMIL_QUADRATIC_H
	#include "../../hamilQ.h"
#endif

#ifndef SYK2_M_H
#define SYK2_M_H

class SYK2 : public QuadraticHamiltonian<double>
{
public:
	SYK2() = default;
	SYK2(std::shared_ptr<Lattice> _lat, double _constant, bool _partCons = true)
		: QuadraticHamiltonian<double>(_lat, _constant, _partCons)	{};

	void hamiltonian() override										{ this->H_ = this->ran_.GOE(this->Nh, this->Nh); }
};


#endif // !SYK2_M_H
