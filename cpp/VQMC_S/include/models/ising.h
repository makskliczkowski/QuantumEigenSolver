#pragma once
#ifndef HAMIL_H
#include "../hamil.h"
#endif // !HAMIL_H

#ifndef ISING_H
#define ISING_H

template <typename _T>
class IsingModel : public Hamiltonian<_T> {
protected:
	// ------------------------------------------- MODEL BASED PARAMETERS -------------------------------------------
	DISORDER_EQUIV(double, J);
	DISORDER_EQUIV(double, g);
	DISORDER_EQUIV(double, h);

	//arma::vec tmp_vec;
	//vec tmp_vec2;

public:
	// ------------------------------------------- 				 Constructors				  -------------------------------------------
	~IsingModel()										= default;
	IsingModel()										= default;
	IsingModel(const Hilbert::HilbertSpace<_T>& hilbert,double J, double g, double h, 
														double J0 = 0, double g0 = 0, double h0 = 0);
	IsingModel(Hilbert::HilbertSpace<_T>&& hilbert,		double J, double g, double h,
														double J0 = 0, double g0 = 0, double h0 = 0);

	// -------------------------------------------				METHODS				-------------------------------------------
	void hamiltonian()									override final;
	void locEnergy(u64 _elemId, u64 _elem, uint _site)	override final;
	cpx locEnergy(u64 _id, uint site, Operators::_OP<cpx>::INP<double> f1,
		std::function<cpx(const arma::vec&)> f2,
		arma::vec& tmp)									override final { return 0; };
	cpx locEnergy(const arma::vec& v, uint site, Operators::_OP<cpx>::INP<double> f1,
		std::function<cpx(const arma::vec&)> f2,
		arma::vec& tmp)									override final { return 0; };

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
		name +=	VEQ(BC);
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
	: Hamiltonian<_T>(hilbert), J(J), g(g), h(h), h0(h0), J0(J0), g0(g0)
{
	this->ran_			=			randomGen();
	this->Ns			=			this->hilbertSpace.getLatticeSize();
	this->Nh			=			ULLPOW(this->Ns);									
	this->dh			=			this->ran_.createRanVec(this->Ns, this->h0);			// creates random disorder vector
	this->dJ			=			this->ran_.createRanVec(this->Ns, this->J0);		// creates random exchange vector
	this->dg			=			this->ran_.createRanVec(this->Ns, this->g0);		// creates random transverse field vector

	//change info
	this->info_			=			this->info();
	this->updateInfo();
}

template <typename _T>
IsingModel<_T>::IsingModel(Hilbert::HilbertSpace<_T>&&hilbert, double J, double g, double h, double J0, double g0, double h0)
	: Hamiltonian<_T>(std::move(hilbert)), J(J), g(g), h(h), h0(h0), J0(J0), g0(g0)
{
	this->ran_			=			randomGen();
	this->Ns			=			this->hilbertSpace.getLatticeSize();
	this->Nh			=			ULLPOW(this->Ns);
	this->dh			=			this->ran_.createRanVec(this->Ns, this->h0);			// creates random disorder vector
	this->dJ			=			this->ran_.createRanVec(this->Ns, this->J0);		// creates random exchange vector
	this->dg			=			this->ran_.createRanVec(this->Ns, this->g0);		// creates random transverse field vector

	//change info
	this->info_			=			this->info();
	this->updateInfo();
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

/*
* @brief body of setting up of the Hamiltonian
*/
template<typename _T>
inline void IsingModel<_T>::locEnergy(u64 _elemId, u64 _elem, uint _site)
{
	// get number of forward nn
	uint NUM_OF_NN = (uint)this->lat_->get_nn_ForwardNum(_site);
	u64 newIdx	= 0;
	_T newVal	= 0;

	// -------------- perpendicular field --------------
	std::tie(newIdx, newVal) = Operators::sigma_z(_elem, this->Ns, { _site });
	this->setHElem(_elemId, PARAM_W_DISORDER(h, _site) * newVal, newIdx);

	// -------------- transverse field --------------
	if (!EQP(this->g, 0.0, 1e-9)) {
		std::tie(newIdx, newVal) = Operators::sigma_x(_elem, this->Ns, { _site });
		this->setHElem(_elemId, PARAM_W_DISORDER(g, _site) * newVal, newIdx);
	}

	// -------------- CHECK NN ---------------
	for (uint nn = 0; nn < NUM_OF_NN; nn++) {
		auto N_NUMBER = this->lat_->get_nn_ForwardNum(_site, nn);
		if (auto nei = this->lat_->get_nn(_site, N_NUMBER); nei >= 0) {
			// Ising-like spin correlation
			auto [idx_z, val_z]			=		Operators::sigma_z(_elem, this->Ns, { _site });
			auto [idx_z2, val_z2]		=		Operators::sigma_z(idx_z, this->Ns, { (uint)nei });
			this->setHElem(_elemId, 
								PARAM_W_DISORDER(J, _site) * (val_z * val_z2), 
								idx_z2);
		}
	}
}

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/

//template <typename _type>
//cpx IsingModel<_type>::locEnergy(u64 _id, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) {
//	// sumup the value of non-changed state
//	double localVal = 0.0;
//	cpx changedVal = 0.0;
//	const uint nn_number = this->lattice->get_nn_forward_num(site);
//
//	// true - spin up, false - spin down
//	double si = checkBit(_id, this->Ns - site - 1) ? this->_SPIN : -this->_SPIN;
//
//	// diagonal elements setting the perpendicular field
//	localVal += (this->h + dh(site)) * si;
//
//	// check the Siz Si+1z
//	for (auto nn = 0; nn < nn_number; nn++) {
//		const auto n_num = this->lattice->get_nn_forward_num(site, nn);
//		if (auto nei = this->lattice->get_nn(site, n_num); nei >= 0) {
//			double sj = checkBit(_id, this->Ns - 1 - nei) ? this->_SPIN : -this->_SPIN;
//			localVal += (this->J + this->dJ(site)) * si * sj;
//		}
//	}
//
//	// flip with S^x_i with the transverse field
//	changedVal += f1(site, si) * this->_SPIN * (this->g + this->dg(site));
//
//	return changedVal + localVal;
//}
//
///*
//* Calculate the local energy end return the corresponding vectors with the value
//* @param _id base state index
//*/
//template <typename _type>
//cpx IsingModel<_type>::locEnergy(const vec& v, uint site, std::function<cpx(int, double)> f1, std::function<cpx(const vec&)> f2, vec& tmp) {
//	double localVal = 0;
//	cpx changedVal = 0.0;
//
//	const uint nn_number = this->lattice->get_nn_forward_num(site);
//
//	// check Sz 
//	const double si = checkBitV(v, site) > 0 ? this->_SPIN : -this->_SPIN;
//
//	// diagonal elements setting the perpendicular field
//	localVal += (this->h + dh(site)) * si;
//
//	// check the Siz Si+1z
//	for (auto nn = 0; nn < nn_number; nn++) {
//		// double checking neighbors
//		auto n_num = this->lattice->get_nn_forward_num(site, nn);
//		if (auto nei = this->lattice->get_nn(site, n_num); nei >= 0) {
//			double sj = checkBitV(v, nei) > 0 ? this->_SPIN : -this->_SPIN;
//			localVal += (this->J + this->dJ(site)) * si * sj;
//		}
//	}
//	// flip with S^x_i with the transverse field
//
//	changedVal += f1(site, si) * this->_SPIN * (this->g + this->dg(site));
//
//	return changedVal + localVal;
//}

// ----------------------------------------------------------------------------- HAMILTONIAN -------------------------------------------------------------------------------------

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _T>
void IsingModel<_T>::hamiltonian() {
	this->init();
	for (u64 k = 0; k < this->Nh; k++)
		for (int site_ = 0; site_ <= this->Ns - 1; site_++)
			this->locEnergy(k, this->hilbertSpace.getMapping(k), site_);
}


#endif // !ISING_H