#pragma once

/***********************************
* Is an instance of the Transverse
* Field Ising model. 
* Derives from a general Hamiltonian.
************************************/

#ifndef HAMIL_H
#include "../hamil.h"
#endif // !HAMIL_H

#ifndef ISING_H
#define ISING_H

template <typename _T>
class IsingModel : public Hamiltonian<_T, 2> {
protected:
	// ------------------------------------------- MODEL BASED PARAMETERS -------------------------------------------
	DISORDER_EQUIV(double, J);
	DISORDER_EQUIV(double, hx);
	DISORDER_EQUIV(double, hz);

	//arma::vec tmp_vec;
	//vec tmp_vec2;

public:
	using NQSFun										= typename Hamiltonian<_T>::NQSFun;
	// ------------------------------------------- 				 Constructors				  -------------------------------------------
	~IsingModel()										
	{
		LOGINFO(this->info() + " - destructor called.", LOG_TYPES::INFO, 4);
	};
	IsingModel()										= default;
	IsingModel(const Hilbert::HilbertSpace<_T>& hilbert,double J, double hx, double hz,
														double J0 = 0, double hx0 = 0, double hz0 = 0);
	IsingModel(Hilbert::HilbertSpace<_T>&& hilbert,		double J, double hx, double hz,
														double J0 = 0, double hx0 = 0, double hz0 = 0);

	// -------------------------------------------				METHODS				-------------------------------------------
	void hamiltonian()									override final;
	void locEnergy(u64 _elemId, u64 _elem, uint _site)	override final;
	cpx locEnergy(u64 _id, uint site, NQSFun f1)		override final;
	cpx locEnergy(const arma::Col<double>& v,
				  uint site,
				  NQSFun f1)							override final;

	// ------------------------------------------- 				 Info				  -------------------------------------------

	std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const override
	{
		auto Ns = this->hilbertSpace.getLatticeSize();
		auto BC = this->hilbertSpace.getBC();
		std::string name = sep + "ising,Ns=" + STR(Ns);
		PARAMS_S_DISORDER(J, name);
		PARAMS_S_DISORDER(hx, name);
		PARAMS_S_DISORDER(hz, name);
		name += this->hilbertSpace.getSymInfo();
		name += ",BC=" + SSTR(getSTR_BoundaryConditions(BC));
		return this->Hamiltonian<_T>::info(name, skip, sep);
	}
	void updateInfo()									override final { this->info_ = this->info(); 			};
	void quenchHamiltonian()							override final { this->hz *= -1.0; this->updateInfo(); 	};
};

// ----------------------------------------------------------------------------- CONSTRUCTORS -----------------------------------------------------------------------------

/**
* @brief Quantum Ising Model constructor
* @param hilbert hilbert space handler
* @param J interaction between Sz's on the nearest neighbors
* @param J0 disorder at J interaction from (-J0,J0) added to J
* @param hx transverse magnetic field
* @param hx0 disorder at hx field from (-hx0, hx0) added to hx
* @param hz perpendicular magnetic field
* @param hz0 disorder at hz field from (-hz0, hz0) added to hz
*/
template <typename _T>
IsingModel<_T>::IsingModel(const Hilbert::HilbertSpace<_T>& hilbert, double J, double hx, double hz, double J0, double hx0, double hz0)
	: Hamiltonian<_T>(hilbert), J(J), J0(J0), hx(hx), hx0(hx0), hz(hz), hz0(hz0)
{
	this->ran_			=			randomGen();
	this->Ns			=			this->hilbertSpace.getLatticeSize();
	this->dhz			=			this->ran_.template createRanVec<double>(this->Ns, this->hz0);		// creates random disorder vector
	this->dJ			=			this->ran_.template createRanVec<double>(this->Ns, this->J0);		// creates random exchange vector
	this->dhx			=			this->ran_.template createRanVec<double>(this->Ns, this->hx0);		// creates random transverse field vector
	this->type_			=			MY_MODELS::ISING_M;

	//change info
	this->info_			=			this->info();
	this->updateInfo();
	LOGINFOG("I am Transverse Field Ising: " + this->info_, LOG_TYPES::CHOICE, 2);
}

template <typename _T>
IsingModel<_T>::IsingModel(Hilbert::HilbertSpace<_T>&&hilbert, double J, double hx, double hz, double J0, double hx0, double hz0)
	: Hamiltonian<_T>(std::move(hilbert)), J(J), J0(J0), hx(hx), hx0(hx0), hz(hz), hz0(hz0)
{
	this->ran_			=			randomGen();
	this->Ns			=			this->hilbertSpace.getLatticeSize();
	this->dhz			=			this->ran_.template createRanVec<double>(this->Ns, this->hz0);		// creates random disorder vector
	this->dJ			=			this->ran_.template createRanVec<double>(this->Ns, this->J0);		// creates random exchange vector
	this->dhx			=			this->ran_.template createRanVec<double>(this->Ns, this->hx0);		// creates random transverse field vector
	this->type_			=			MY_MODELS::ISING_M;

	//change info
	this->info_			=			this->info();
	this->updateInfo();
	LOGINFOG("I am Transverse Field Ising: " + this->info_, LOG_TYPES::CHOICE, 2);
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/**
* @brief body of setting up of the Hamiltonian
*/
template<typename _T>
inline void IsingModel<_T>::locEnergy(u64 _elemId, u64 _elem, uint _site)
{
	// get number of forward nn
	uint NUM_OF_NN	= (uint)this->lat_->get_nn_ForwardNum(_site);
	u64 newIdx		= 0;
	_T newVal		= 0;

	// -------------- perpendicular field --------------
	std::tie(newIdx, newVal) = Operators::sigma_z<_T>(_elem, this->Ns, { _site });
	// Python: -hz * Sz
	this->setHElem(_elemId, -PARAM_W_DISORDER(hz, _site) * newVal, newIdx);

	// -------------- transverse field --------------
	if (!EQP(this->hx, 0.0, 1e-9)) {
		std::tie(newIdx, newVal) = Operators::sigma_x(_elem, this->Ns, { _site });
		// Python: -hx * Sx
		this->setHElem(_elemId, -PARAM_W_DISORDER(hx, _site) * newVal, newIdx);
	}

	// -------------- CHECK NN ---------------
	for (uint nn = 0; nn < NUM_OF_NN; nn++) 
	{
		const uint N_NUMBER = nn;
		if (int nei = this->lat_->get_nnf(_site, N_NUMBER); nei >= 0) 
		{
			// Ising-like spin correlation
			auto [idx_z, val_z]			=		Operators::sigma_z<_T>(_elem, this->Ns, { _site });
			auto [idx_z2, val_z2]		=		Operators::sigma_z<_T>(idx_z, this->Ns, { (uint)nei });
			// Python: -J * Sz * Sz
			this->setHElem(_elemId, -PARAM_W_DISORDER(J, _site) * (val_z * val_z2), idx_z2);
		}
	}
	//stout << "____________________________" << EL << EL;
}

// -----------------------------------------------------------------------------

/*
* Calculate the local energy for the NQS purpose.
*/
template<typename _T>
inline cpx IsingModel<_T>::locEnergy(u64 _id, uint site, NQSFun f1)
{
	double _locVal		=	0.0;			// unchanged state value
	cpx _changedVal		=	0.0;			// changed state value			

	// get number of forward nn
	uint NUM_OF_NN		=	(uint)this->lat_->get_nn_ForwardNum(site);

	// check spin at a given site
	double _Si			=	checkBit(_id, this->Ns - site - 1) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;

	// add to a local value (Perpendicular field)
	_locVal				+=	-PARAM_W_DISORDER(hz, site) * _Si;

	// check the S_i^z * S_{i+1}^z
	for (uint nn = 0; nn < NUM_OF_NN; nn++) {
		auto N_NUMBER = this->lat_->get_nn_ForwardNum(site, nn);
		if (auto nei = this->lat_->get_nn(site, N_NUMBER); nei >= 0) {
			double _Sj	=	checkBit(_id, this->Ns - nei - 1) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
			_locVal		+=	-PARAM_W_DISORDER(J, site) * _Si * _Sj;
		}
	}
	// -----------------------------------------------------------
	// Transverse field
	_changedVal			+=	f1(std::initializer_list<int>({ (int)site }),
							   std::initializer_list<double>({ _Si })) * -PARAM_W_DISORDER(hx, site) * Operators::_SPIN_RBM;

	// -----------------------------------------------------------
	return _changedVal + _locVal;
}

template<typename _T>
inline cpx IsingModel<_T>::locEnergy(const arma::Col<double>& v, uint _site, NQSFun f1)
{
	double _locVal		=	0.0;			// unchanged state value
	cpx _changedVal		=	0.0;			// changed state value			

	// check spin at a given site
	double _Si			=	Binary::check(v, _site) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;

	// add to a local value
	_locVal				+=	-PARAM_W_DISORDER(hz, _site) * _Si;

	// check the S_i^z * S_{i+1}^z
	for (uint nn = 0; nn < (uint)this->lat_->get_nn_ForwardNum(_site); ++nn) 
	{
		if (int nei = this->lat_->get_nnf(_site, nn); nei >= 0) 
		{
			double _Sj	=	Binary::check(v, nei) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
			_locVal		+=	-PARAM_W_DISORDER(J, _site) * _Si * _Sj;
		}
	}
	// -----------------------------------------------------------
	if (!EQP(this->hx, 0.0, 1e-9)) {
		_changedVal += f1(std::initializer_list<int>({ (int)_site }), std::initializer_list<double>({ _Si })) * -PARAM_W_DISORDER(hx, _site) * Operators::_SPIN_RBM;
	}
	// -----------------------------------------------------------
	return _changedVal + _locVal;
}

// ----------------------------------------------------------------------------- HAMILTONIAN -------------------------------------------------------------------------------------

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _T>
void IsingModel<_T>::hamiltonian() {
	if (this->Nh == 0)
	{
		LOGINFOG("Empty Hilbert, not building anything.", LOG_TYPES::INFO, 1);
		return;
	}
	this->init();
	for (u64 k = 0; k < this->Nh; k++) {
		u64 kMap = this->hilbertSpace.getMapping(k);
		for (uint site_ = 0; site_ <= this->Ns - 1; site_++)
			this->locEnergy(k, kMap, site_);
	}
}


#endif // !ISING_H
