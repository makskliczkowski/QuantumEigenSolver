#pragma once
/***********************************
* Defines the Aubry Andre model.
***********************************/

#ifndef HAMIL_QUADRATIC_H
#include "../../hamilQ.h"
#endif

#ifndef AUBRY_ANDRE_M_H
#define AUBRY_ANDRE_M_H

template<typename _T>
class AubryAndre : public QuadraticHamiltonian<_T>
{
protected:
	DISORDER_EQUIV(double, J);
	DISORDER_EQUIV(double, B);
	DISORDER_EQUIV(double, P);
	DISORDER_EQUIV(double, Ld);
public:
	~AubryAndre() = default;
	AubryAndre(std::shared_ptr<Lattice> _lat,	double _J, 
												double _disorder, 
												double _beta		= (1.0 + std::sqrt(5)) / 2.0,
												double _phi			= 0.0,
												double _J0			= 0.0,
												double _l0			= 0.0,
												double _beta0		= 0.0,
												double _phi0		= 0.0,
												double _constant	= 0.0)
		: QuadraticHamiltonian<_T>(_lat, _constant, true), J(_J), J0(_J0),
		B(_beta), B0(_beta0), P(_phi), P0(_phi0), Ld(_disorder), Ld0(_l0)
	{
		this->type_		= MY_MODELS::AUBRY_ANDRE_M;
		this->info_		= this->info();
		this->dJ		= this->ran_.template createRanVec<double>(this->Ns, this->J0);
		this->dB		= this->ran_.template createRanVec<double>(this->Ns, this->B0);
		this->dP		= this->ran_.template createRanVec<double>(this->Ns, this->P0);
		this->dLd		= this->ran_.template createRanVec<double>(this->Ns, this->Ld0);
		LOGINFO("I am Aubry-Andre model (1): ", LOG_TYPES::CHOICE, 2);
	};
	// Lattice size constructor
	AubryAndre(uint _Ns,double _J, 
						double _disorder, 
						double _beta		= (1.0 + std::sqrt(5)) / 2.0,
						double _phi			= 0.0,
						double _J0			= 0.0,
						double _l0			= 0.0,
						double _beta0		= 0.0,
						double _phi0		= 0.0,
						double _constant	= 0.0)
		: QuadraticHamiltonian<_T>(_Ns, _constant, true), J(_J), J0(_J0),
		B(_beta), B0(_beta0), P(_phi), P0(_phi0), Ld(_disorder), Ld0(_l0)
	{
		this->type_		= MY_MODELS::AUBRY_ANDRE_M;
		this->info_		= this->info();
		this->dJ		= this->ran_.template createRanVec<double>(this->Ns, this->J0);
		this->dB		= this->ran_.template createRanVec<double>(this->Ns, this->B0);
		this->dP		= this->ran_.template createRanVec<double>(this->Ns, this->P0);
		this->dLd		= this->ran_.template createRanVec<double>(this->Ns, this->Ld0);
		LOGINFO("I am Aubry-Andre model (1): ", LOG_TYPES::CHOICE, 2);
	};
	// Hilbert space constructor
	AubryAndre(Hilbert::HilbertSpace<_T>& _hilb, 
						double _J,
						double _disorder,
						double _beta		= (1.0 + std::sqrt(5)) / 2.0,
						double _phi			= 0.0,
						double _J0			= 0.0,
						double _l0			= 0.0,
						double _beta0		= 0.0,
						double _phi0		= 0.0,
						double _constant	= 0.0)
		: QuadraticHamiltonian<_T>(_hilb, _constant, true), J(_J), J0(_J0), 
		B(_beta), B0(_beta0), P(_phi), P0(_phi0), Ld(_disorder), Ld0(_l0)
	{
		this->type_		= MY_MODELS::AUBRY_ANDRE_M;
		this->info_		= this->info();
		this->dJ		= this->ran_.template createRanVec<double>(this->Ns, this->J0);
		this->dB		= this->ran_.template createRanVec<double>(this->Ns, this->B0);
		this->dP		= this->ran_.template createRanVec<double>(this->Ns, this->P0);
		this->dLd		= this->ran_.template createRanVec<double>(this->Ns, this->Ld0);
		LOGINFO("I am Aubry-Andre model (1): ", LOG_TYPES::CHOICE, 2);
	}

	// Hilbert space move constructor
	AubryAndre(Hilbert::HilbertSpace<_T>&& _hilb,
						double _J,
						double _disorder,
						double _beta		= (1.0 + std::sqrt(5)) / 2.0,
						double _phi			= 0.0,
						double _J0			= 0.0,
						double _l0			= 0.0,
						double _beta0		= 0.0,
						double _phi0		= 0.0,
						double _constant	= 0.0)
		: QuadraticHamiltonian<_T>(std::move(_hilb), _constant, true), J(_J), J0(_J0),
		B(_beta), B0(_beta0), P(_phi), P0(_phi0), Ld(_disorder), Ld0(_l0)
	{
		this->type_		= MY_MODELS::AUBRY_ANDRE_M;
		this->info_		= this->info();
		this->dJ		= this->ran_.template createRanVec<double>(this->Ns, this->J0);
		this->dB		= this->ran_.template createRanVec<double>(this->Ns, this->B0);
		this->dP		= this->ran_.template createRanVec<double>(this->Ns, this->P0);
		this->dLd		= this->ran_.template createRanVec<double>(this->Ns, this->Ld0);
		LOGINFO("I am Aubry-Andre model (1): ", LOG_TYPES::CHOICE, 2);
	}

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// ### H A M I L T O N I A N ###

	void hamiltonian() override
	{
		this->init();
		auto _Lx = this->lat_->get_Lx();
		auto _Ly = this->lat_->get_Ly();
		
		for (int m = 0; m < _Lx; ++m)
		{
			for (int n = 0; n < _Ly; ++n)
			{
				const auto _idx	= _Lx * n + m;
				const auto _w	= PARAM_W_DISORDER(Ld, _idx) * PARAM_W_DISORDER(J, _idx);
				const auto Vs	= _w * (std::cos(TWOPI * PARAM_W_DISORDER(B, _idx) * (m + n) + PARAM_W_DISORDER(P, _idx)) +
										std::cos(TWOPI * PARAM_W_DISORDER(B, _idx) * (m - n) + PARAM_W_DISORDER(P, _idx)));

				// get the number of NN to include
				uint NUM_OF_NN	= (uint)this->lat_->get_nn_ForwardNum(_idx);

				// diagonal (modulation)
				this->setHElem(_idx, Vs, _idx);

				// -------------- CHECK NN ---------------
				for (uint nn = 0; nn < NUM_OF_NN; nn++) 
				{
					uint N_NUMBER = this->lat_->get_nn_ForwardNum(_idx, nn);
					if (int nei = this->lat_->get_nn(_idx, N_NUMBER); nei >= 0)
						this->setHElem(_idx, -PARAM_W_DISORDER(J, _idx), nei);
				}
			}
		}
		this->H_.symmetrize();
		if(this->isSparse_)
			this->H_.getSparse().diag() /= 4.0;
		else
			this->H_.getDense().diag() /= 4.0;
	}

	// ------------------------------------------- 				 Info				  -------------------------------------------

	std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const override
	{
		auto Ns = this->lat_->get_Ns();
		auto BC = this->lat_->get_BC();
		std::string name = sep + "AUBRYANDRE,Ns=" + STR(Ns);
		PARAMS_S_DISORDER(J, name);
		PARAMS_S_DISORDER(B, name);
		PARAMS_S_DISORDER(P, name);
		PARAMS_S_DISORDER(Ld, name);
		name += ",BC=" + SSTR(getSTR_BoundaryConditions(BC));
		return this->QuadraticHamiltonian<_T>::info(name, skip, sep);
	}
	void updateInfo()									override final { this->info_ = this->info(); };
};

#endif // !SYK2_M_H
