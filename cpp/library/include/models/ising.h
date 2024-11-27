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
	DISORDER_EQUIV(double, g);
	DISORDER_EQUIV(double, h);

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
	IsingModel(const Hilbert::HilbertSpace<_T>& hilbert,double J, double g, double h, 
														double J0 = 0, double g0 = 0, double h0 = 0);
	IsingModel(Hilbert::HilbertSpace<_T>&& hilbert,		double J, double g, double h,
														double J0 = 0, double g0 = 0, double h0 = 0);

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
		PARAMS_S_DISORDER(g, name);
		PARAMS_S_DISORDER(h, name);
		name += this->hilbertSpace.getSymInfo();
		name += ",BC=" + SSTR(getSTR_BoundaryConditions(BC));
		return this->Hamiltonian<_T>::info(name, skip, sep);
	}
	void updateInfo()									override final { this->info_ = this->info(); };
};

// ----------------------------------------------------------------------------- CONSTRUCTORS -----------------------------------------------------------------------------

/*
* @brief Quantum Ising Model constructor
* @param hilbert hilbert space handler
* @param J interaction between Sz's on the nearest neighbors
* @param J0 disorder at J interaction from (-J0,J0) added to J
* @param g transverse magnetic field
* @param g0 disorder at g field from (-g0, g0) added to g
* @param h perpendicular magnetic field
* @param w disorder at h field from (-w, w) added to h
*/
template <typename _T>
IsingModel<_T>::IsingModel(const Hilbert::HilbertSpace<_T>& hilbert, double J, double g, double h, double J0, double g0, double h0)
	: Hamiltonian<_T>(hilbert), J(J), J0(J0), g(g), g0(g0), h(h), h0(h0)
{
	this->ran_			=			randomGen();
	this->Ns			=			this->hilbertSpace.getLatticeSize();
	this->dh			=			this->ran_.template createRanVec<double>(this->Ns, this->h0);		// creates random disorder vector
	this->dJ			=			this->ran_.template createRanVec<double>(this->Ns, this->J0);		// creates random exchange vector
	this->dg			=			this->ran_.template createRanVec<double>(this->Ns, this->g0);		// creates random transverse field vector
	this->type_			=			MY_MODELS::ISING_M;

	//change info
	this->info_			=			this->info();
	this->updateInfo();
	LOGINFOG("I am Transverse Field Ising: " + this->info_, LOG_TYPES::CHOICE, 2);
}

template <typename _T>
IsingModel<_T>::IsingModel(Hilbert::HilbertSpace<_T>&&hilbert, double J, double g, double h, double J0, double g0, double h0)
	: Hamiltonian<_T>(std::move(hilbert)), J(J), J0(J0), g(g), g0(g0), h(h), h0(h0)
{
	this->ran_			=			randomGen();
	this->Ns			=			this->hilbertSpace.getLatticeSize();
	this->dh			=			this->ran_.template createRanVec<double>(this->Ns, this->h0);		// creates random disorder vector
	this->dJ			=			this->ran_.template createRanVec<double>(this->Ns, this->J0);		// creates random exchange vector
	this->dg			=			this->ran_.template createRanVec<double>(this->Ns, this->g0);		// creates random transverse field vector
	this->type_			=			MY_MODELS::ISING_M;

	//change info
	this->info_			=			this->info();
	this->updateInfo();
	LOGINFOG("I am Transverse Field Ising: " + this->info_, LOG_TYPES::CHOICE, 2);
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
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
	//stout << "Z:" << newIdx << ":" << newVal << EL;
	//intToBase(newIdx, tmp);
	//stout << tmp.t() << EL << EL;
	this->setHElem(_elemId, PARAM_W_DISORDER(h, _site) * newVal, newIdx);

	// -------------- transverse field --------------
	if (!EQP(this->g, 0.0, 1e-9)) {
		std::tie(newIdx, newVal) = Operators::sigma_x(_elem, this->Ns, { _site });
		//stout << "X:" << newIdx << ":" << newVal << EL;
		//intToBase(newIdx, tmp);
		//stout << tmp.t() << EL << EL;
		this->setHElem(_elemId, PARAM_W_DISORDER(g, _site) * newVal, newIdx);
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
			//stout << "NEI:" << idx_z2 << ":" << val_z2 * val_z << EL;
			//intToBase(idx_z2, tmp);
			//stout << tmp.t() << EL << EL;
			this->setHElem(_elemId, PARAM_W_DISORDER(J, _site) * (val_z * val_z2), idx_z2);
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

	// add to a local value
	_locVal				+=	PARAM_W_DISORDER(h, site) * _Si;

	// check the S_i^z * S_{i+1}^z
	for (uint nn = 0; nn < NUM_OF_NN; nn++) {
		auto N_NUMBER = this->lat_->get_nn_ForwardNum(site, nn);
		if (auto nei = this->lat_->get_nn(site, N_NUMBER); nei >= 0) {
			double _Sj	=	checkBit(_id, this->Ns - nei - 1) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
			_locVal		+=	PARAM_W_DISORDER(J, site) * _Si * _Sj;
		}
	}
	// -----------------------------------------------------------
	_changedVal			+=	f1(std::initializer_list<int>({ (int)site }),
							   std::initializer_list<double>({ _Si })) * PARAM_W_DISORDER(g, site) * Operators::_SPIN_RBM;

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
	_locVal				+=	PARAM_W_DISORDER(h, _site) * _Si;

	// check the S_i^z * S_{i+1}^z
	for (uint nn = 0; nn < (uint)this->lat_->get_nn_ForwardNum(_site); ++nn) 
	{
		if (int nei = this->lat_->get_nnf(_site, nn); nei >= 0) 
		{
			double _Sj	=	Binary::check(v, nei) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
			_locVal		+=	PARAM_W_DISORDER(J, _site) * _Si * _Sj;
		}
	}
	// -----------------------------------------------------------
	if (!EQP(this->g, 0.0, 1e-9)) {
		_changedVal += f1(std::initializer_list<int>({ (int)_site }), std::initializer_list<double>({ _Si })) * PARAM_W_DISORDER(g, _site) * Operators::_SPIN_RBM;
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