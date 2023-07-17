#pragma once

#ifndef HAMIL_QUADRATIC_H
#include "../../hamilQ.h"
#endif

#ifndef FF_M_H
#define FF_M_H

template<typename _T>
class FreeFermions : public QuadraticHamiltonian<_T>
{
protected:
	DISORDER_EQUIV(double, t_);
public:
	~FreeFermions()		= default;
	FreeFermions(std::shared_ptr<Lattice> _lat, double _t = 1.0, double _t0 = 0.0, double _constant = 0.0)
		: QuadraticHamiltonian<_T>(_lat, _constant, true), t_(_t), t_0(_t0)
	{
		this->info_			=		this->info();
		this->dt_			=		this->ran_.createRanVec(this->Ns, this->t_0);
		LOGINFO("I am Free Fermions model: ", LOG_TYPES::CHOICE, 2);
	};
	
	// ### H A M I L T O N I A N ###

	void hamiltonian() override
	{
		this->init();
		for (int i = 0; i < this->Ns; i++)
		{
			uint NUM_OF_NN	=	(uint)this->lat_->get_nn_ForwardNum(i);
			// -------------- CHECK NN ---------------
			for (uint nn = 0; nn < NUM_OF_NN; nn++) {
				uint N_NUMBER = this->lat_->get_nn_ForwardNum(i, nn);
				if (int nei = this->lat_->get_nn(i, N_NUMBER); nei >= 0)
					this->setHElem(i, PARAM_W_DISORDER(t_, i), nei);
			}
		}
	}

	// ------------------------------------------- 				 Info				  -------------------------------------------
	
	std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const override
	{
		auto Ns = this->hilbertSpace.getLatticeSize();
		auto BC = this->hilbertSpace.getBC();
		std::string name = sep + "FF,Ns=" + STR(Ns);
		PARAMS_S_DISORDER(t_, name);
		name += ",BC=" + SSTR(getSTR_BoundaryConditions(BC));
		return this->QuadraticHamiltonian<_T>::info(name, skip, sep);
	}
	void updateInfo() override final { this->info_ = this->info(); };

	// override 
	auto getTransMat()	-> arma::Mat<_T>							override;
	auto getSPEnMat()	-> arma::Col<double>						override;
};

// ###################################################################################

/*
* @brief Returns the transformation matrix for free fermions, as we know it by hand...
*/
template<typename _T>
inline arma::Mat<_T> FreeFermions<_T>::getTransMat()
{
	if (!this->eigVec_.empty())
		return this->eigVec_;
	this->eigVec_.zeros(this->Ns, this->Ns);
	for (int k = 0; k < this->Ns; k++)
		for (int i = 0; i < this->Ns; i++)
			this->eigVec_(k, i) = algebra::cast<_T>(std::exp(I * double(k * i * TWOPI / this->Ns)) / std::sqrt(this->Ns));
	return this->eigVec_;
}

// ###################################################################################

/*
* @brief Returns the transformation matrix for free fermions, as we know it by hand...
*/
template<typename _T>
inline arma::Col<double> FreeFermions<_T>::getSPEnMat()
{
	if (!this->eigVal_.empty())
		return this->eigVal_;
	this->eigVal_.zeros(this->Ns, this->Ns);
	for (int k = 0; k < this->Ns; k++)
		this->eigVal_(k) = 2.0 * PARAM_W_DISORDER(t_, k) * std::cos(double(k * TWOPI / this->Ns));
}

// ###################################################################################

#endif // !SYK2_M_H
