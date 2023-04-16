#pragma once
#ifndef HAMIL_H
#include "../hamil.h"
#endif // !HAMIL_H

#ifndef XYZ_H
#define XYZ_H

template <typename _T>
class XYZ : public Hamiltonian<_T> {
public:
	using NQSFunSingle									= Hamiltonian<_T>::NQSFunSingle;
	using NQSFunMultiple								= Hamiltonian<_T>::NQSFunMultiple;
protected:
	// ------------------------------------------- MODEL BASED PARAMETERS -------------------------------------------
	DISORDER_EQUIV(double, Ja);
	DISORDER_EQUIV(double, Jb);
	DISORDER_EQUIV(double, hx);
	DISORDER_EQUIV(double, hz);
	DISORDER_EQUIV(double, dA);
	DISORDER_EQUIV(double, dB);
	DISORDER_EQUIV(double, eA);
	DISORDER_EQUIV(double, eB);
	bool parityBreak_									= false;

public:
	// ------------------------------------------- 				 Constructors				  -------------------------------------------
	~XYZ()												= default;
	XYZ()												= default;

	XYZ(const Hilbert::HilbertSpace<_T>& hilbert,		double _Ja, double _Jb, 
														double _hx, double _hz, 
														double _dA, double _dB, 
														double _eA, double _eB, 
														bool _parityBreak = false);

	XYZ(Hilbert::HilbertSpace<_T>&& hilbert,			double _Ja, double _Jb,
														double _hx, double _hz,
														double _dA, double _dB,
														double _eA, double _eB,
														bool _parityBreak = false);

	XYZ(const Hilbert::HilbertSpace<_T>& hilbert,		double _Ja, double _Jb,
														double _hx, double _hz,
														double _dA, double _dB,
														double _eA, double _eB,
														double _Ja0, double _Jb0,
														double _hx0, double _hz0,
														double _dA0, double _dB0,
														double _eA0, double _eB0,
														bool _parityBreak = false)
		: XYZ<_T>(hilbert, _Ja, _Jb, _hx, _hz, _dA, _dB, _eA, _eB, _parityBreak)
	{
		this->Ja0 = _Ja0; this->Jb0 = _Jb0; this->hx0 = _hx0;
		this->hz0 = _hz0; this->dA0 = _dA0; this->dB0 = _dB0;
		this->eA0 = _eA0; this->eB0 = _eB0;
		this->dJa = this->ran_.createRanVec(this->Ns, this->Ja0);
		this->dJb = this->ran_.createRanVec(this->Ns, this->Jb0);
		this->dhx = this->ran_.createRanVec(this->Ns, this->hx0);
		this->dhz = this->ran_.createRanVec(this->Ns, this->hz0);
		this->ddA = this->ran_.createRanVec(this->Ns, this->dA0);
		this->ddB = this->ran_.createRanVec(this->Ns, this->dB0);
		this->deA = this->ran_.createRanVec(this->Ns, this->eA0);
		this->deB = this->ran_.createRanVec(this->Ns, this->eB0);
	};

	XYZ(Hilbert::HilbertSpace<_T>&& hilbert,			double _Ja, double _Jb,
														double _hx, double _hz,
														double _dA, double _dB,
														double _eA, double _eB,
														double _Ja0, double _Jb0,
														double _hx0, double _hz0,
														double _dA0, double _dB0,
														double _eA0, double _eB0,
														bool _parityBreak = false)
		: XYZ<_T>(std::move(hilbert), _Ja, _Jb, _hx, _hz, _dA, _dB, _eA, _eB, _parityBreak)
	{
		this->Ja0 = _Ja0; this->Jb0 = _Jb0; this->hx0 = _hx0;
		this->hz0 = _hz0; this->dA0 = _dA0; this->dB0 = _dB0;
		this->eA0 = _eA0; this->eB0 = _eB0;
		this->dJa = this->ran_.createRanVec(this->Ns, this->Ja0);
		this->dJb = this->ran_.createRanVec(this->Ns, this->Jb0);
		this->dhx = this->ran_.createRanVec(this->Ns, this->hx0);
		this->dhz = this->ran_.createRanVec(this->Ns, this->hz0);
		this->ddA = this->ran_.createRanVec(this->Ns, this->dA0);
		this->ddB = this->ran_.createRanVec(this->Ns, this->dB0);
		this->deA = this->ran_.createRanVec(this->Ns, this->eA0);
		this->deB = this->ran_.createRanVec(this->Ns, this->eB0);
	};

	// -------------------------------------------				METHODS				-------------------------------------------
	void hamiltonian()									override final;
	void locEnergy(u64 _elemId, u64 _elem, uint _site)	override final;
	cpx locEnergy(u64 _id, uint site, const NQSFunSingle& f1,
		const NQSFunMultiple& f2,
		arma::Col<double>& tmp)							override final { return 0; };
	cpx locEnergy(const arma::Col<double>& v, uint site,
		const NQSFunSingle& f1,
		const NQSFunMultiple& f2,
		arma::Col<double>& tmp)							override final { return 0; };

	// ------------------------------------------- 				 Info				  -------------------------------------------

	std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const override
	{
		auto Ns = this->hilbertSpace.getLatticeSize();
		auto BC = this->hilbertSpace.getBC();
		std::string name = sep + "xyz,Ns=" + STR(this->Ns);
		PARAMS_S_DISORDER(Ja, name);
		PARAMS_S_DISORDER(Jb, name);
		PARAMS_S_DISORDER(hx, name);
		PARAMS_S_DISORDER(hz, name);
		PARAMS_S_DISORDER(dA, name);
		PARAMS_S_DISORDER(dB, name);
		PARAMS_S_DISORDER(eA, name);
		PARAMS_S_DISORDER(eB, name);
		name += ",pb=" + STR(this->parityBreak_);
		name += this->hilbertSpace.getSymInfo();
		name += "," + VEQ(BC);
		return	this->Hamiltonian<_T>::info(name, skip, sep);
	}
	void updateInfo()									override final { this->info_ = this->info(); };
};

// ----------------------------------------------------------------------------- CONSTRUCTORS -----------------------------------------------------------------------------
template<typename _T>
inline XYZ<_T>::XYZ(const Hilbert::HilbertSpace<_T>& hilbert, double _Ja, double _Jb, double _hx, double _hz, double _dA, double _dB, double _eA, double _eB, bool _parityBreak)
	: Hamiltonian<_T>(hilbert), Ja(_Ja), Jb(_Jb), hx(_hx), hz(_hz), dA(_dA), dB(_dB), eA(_eA), eB(_eB), parityBreak_(_parityBreak)
{
	this->ran_			= randomGen();
	this->Ns			= this->hilbertSpace.getLatticeSize();
	this->Nh			= ULLPOW(this->Ns);
	this->dJa			= ZEROV(this->Ns);
	this->dJb			= ZEROV(this->Ns);
	this->dhx			= ZEROV(this->Ns);
	this->dhz			= ZEROV(this->Ns);
	this->ddA			= ZEROV(this->Ns);
	this->ddB			= ZEROV(this->Ns);
	this->deA			= ZEROV(this->Ns);
	this->deB			= ZEROV(this->Ns);

	//change info
	this->info_ = this->info();
	this->updateInfo();
}

template<typename _T>
inline XYZ<_T>::XYZ(Hilbert::HilbertSpace<_T>&& hilbert, double _Ja, double _Jb, double _hx, double _hz, double _dA, double _dB, double _eA, double _eB, bool _parityBreak)
	: Hamiltonian<_T>(std::move(hilbert)), Ja(_Ja), Jb(_Jb), hx(_hx), hz(_hz), dA(_dA), dB(_dB), eA(_eA), eB(_eB), parityBreak_(_parityBreak)
{
	this->ran_ = randomGen();
	this->Ns = this->hilbertSpace.getLatticeSize();
	this->Nh = ULLPOW(this->Ns);
	this->dJa = ZEROV(this->Ns);
	this->dJb = ZEROV(this->Ns);
	this->dhx = ZEROV(this->Ns);
	this->dhz = ZEROV(this->Ns);
	this->ddA = ZEROV(this->Ns);
	this->ddB = ZEROV(this->Ns);
	this->deA = ZEROV(this->Ns);
	this->deB = ZEROV(this->Ns);

	//change info
	this->info_ = this->info();
	this->updateInfo();
}

// ----------------------------------------------------------------------------- LOCAL ENERGY -------------------------------------------------------------------------------------

///*
//* Calculate the local energy end return the corresponding vectors with the value
//* @param _id base state index
//*/
//template <typename _type>
//cpx XYZ<_type>::locEnergy(u64, uint, std::function<cpx(int, double)>, std::function<cpx(const vec&)>, vec& v) {
//
//	return 0.0;
//}
//
///*
//* Calculate the local energy end return the corresponding vectors with the value
//* @param _id base state index
//*/
//template <typename _type>
//cpx XYZ<_type>::locEnergy(const vec&, uint, std::function<cpx(int, double)>, std::function<cpx(const vec&)>, vec& v) {
//	return 0.0;
//}

/*
* @brief body of setting up of the Hamiltonian
*/
template<typename _T>
inline void XYZ<_T>::locEnergy(u64 _elemId, u64 _elem, uint _site)
{
	// get number of forward nn
	uint NUM_OF_NN	= (uint)this->lat_->get_nn_ForwardNum(_site);
	uint NUM_OF_NNN = (uint)this->lat_->get_nnn_ForwardNum(_site);
	u64 newIdx = 0;
	_T newVal = 0;

	// -------------- perpendicular field --------------
	std::tie(newIdx, newVal) = Operators::sigma_z(_elem, this->Ns, { _site });
	this->setHElem(_elemId, PARAM_W_DISORDER(hz, _site) * newVal, newIdx);

	// -------------- transverse field --------------
	if (!EQP(this->hx, 0.0, 1e-9)) {
		std::tie(newIdx, newVal) = Operators::sigma_x(_elem, this->Ns, { _site });
		this->setHElem(_elemId, PARAM_W_DISORDER(hx, _site) * newVal, newIdx);
	}
	if (this->parityBreak_ && (_site == 0 || _site == this->Ns - 1))
		this->setHElem(_elemId, 1.0, newIdx);

	// -------------------------------------------------------- CHECK NN ---------------------------------------------------------
	for (uint nn = 0; nn < NUM_OF_NN; nn++) {
		auto N_NUMBER = this->lat_->get_nn_ForwardNum(_site, nn);
		if (auto nei = this->lat_->get_nn(_site, N_NUMBER); nei >= 0) {
			// SZiSZj
			auto [idx_z, val_z]		= Operators::sigma_z(_elem, this->Ns,	{ _site });
			auto [idx_z2, val_z2]	= Operators::sigma_z(idx_z, this->Ns,	{ (uint)nei });
			this->setHElem(_elemId, 
							PARAM_W_DISORDER(dA, _site) * PARAM_W_DISORDER(Ja, _site) * (val_z * val_z2),
							idx_z2);
			// SYiSYj
			auto [idx_y, val_y]		= Operators::sigma_y(_elem, this->Ns,	{ _site });
			auto [idx_y2, val_y2]	= Operators::sigma_y(idx_y, this->Ns,	{ (uint)nei });
			this->setHElem(_elemId, 
							PARAM_W_DISORDER(Ja, _site) * (1.0 + PARAM_W_DISORDER(eA, _site)) * std::real(val_y * val_y2),
							idx_y2);
			// SXiSXj
			auto [idx_x, val_x]		= Operators::sigma_x(_elem, this->Ns,	{ _site });
			auto [idx_x2, val_x2]	= Operators::sigma_x(idx_x, this->Ns,	{ (uint)nei });
			this->setHElem(_elemId,
							PARAM_W_DISORDER(Ja, _site) * (1.0 + PARAM_W_DISORDER(eA, _site)) * val_x * val_x2,
							idx_x2);
		}
	}

	// -------------------------------------------------------- CHECK NNN ---------------------------------------------------------
	for (uint nnn = 0; nnn < NUM_OF_NNN; nnn++) {
		auto N_NUMBER = this->lat_->get_nnn_ForwardNum(_site, nnn);
		if (auto nei = this->lat_->get_nnn(_site, N_NUMBER); nei >= 0) {
			// SZiSZj
			auto [idx_z, val_z]		= Operators::sigma_z(_elem, this->Ns,	{ _site });
			auto [idx_z2, val_z2]	= Operators::sigma_z(idx_z, this->Ns,	{ (uint)nei });
			this->setHElem(_elemId,
							PARAM_W_DISORDER(dB, _site) * PARAM_W_DISORDER(Jb, _site) * (val_z * val_z2),
							idx_z2);
			// SYiSYj
			auto [idx_y, val_y]		= Operators::sigma_y(_elem, this->Ns,	{ _site });
			auto [idx_y2, val_y2]	= Operators::sigma_y(idx_y, this->Ns,	{ (uint)nei });
			this->setHElem(_elemId,
							PARAM_W_DISORDER(Jb, _site) * (1.0 + PARAM_W_DISORDER(eB, _site)) * std::real(val_y * val_y2),
							idx_y2);
			// SXiSXj
			auto [idx_x, val_x]		= Operators::sigma_x(_elem, this->Ns,	{ _site });
			auto [idx_x2, val_x2]	= Operators::sigma_x(idx_x, this->Ns,	{ (uint)nei });
			this->setHElem(_elemId,
							PARAM_W_DISORDER(Jb, _site) * (1.0 + PARAM_W_DISORDER(eB, _site)) * val_x * val_x2,
							idx_x2);
		}
	}
}

// ----------------------------------------------------------------------------- BUILDING HAMILTONIAN -----------------------------------------------------------------------------

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
//template <typename _type>
//void XYZ<_type>::hamiltonian() {
//	this->init_ham_mat();
//	auto Ns = this->lattice->get_Ns();
//
//	// loop over all states
//	for (u64 k = 0; k < this->N; k++) {
//		// loop over all sites
//		u64 idx = 0;
//		cpx val = 0.0;
//		for (int j = 0; j <= Ns - 1; j++) {
//			const uint nn_number = this->lattice->get_nn_forward_num(j);
//			const uint nnn_number = this->lattice->get_nnn_forward_num(j);
//
//			// diagonal elements setting the perpendicular field
//
//			const double perpendicular_val = ((j == Ns - 1) && this->parity_break) ? 0.5 * this->hz : this->hz;
//			if (!valueEqualsPrec(perpendicular_val, 0.0, 1e-9)) {
//				std::tie(idx, val) = Operators<cpx>::sigma_z(k, Ns, { j });
//				this->H(idx, k) += perpendicular_val * real(val);
//			}
//			// flip with S^x_i with the transverse field -> just one place to break
//			const double transverse_val = ((j == 0) && this->parity_break) ? 0.5 * this->hx : this->hx;
//			if (!valueEqualsPrec(transverse_val, 0.0, 1e-9)) {
//				std::tie(idx, val) = Operators<cpx>::sigma_x(k, Ns, { j });
//				this->setHamiltonianElem(k, transverse_val * real(val), idx);
//			}
//			// -------------- CHECK NN ---------------
//			for (auto nn = 0; nn < nn_number; nn++) {
//				auto n_num = this->lattice->get_nn_forward_num(j, nn);
//				if (auto nei = this->lattice->get_nn(j, n_num); nei >= 0) {
//
//					// setting the neighbors elements
//					auto [idx_x, val_x] = Operators<cpx>::sigma_x(k, Ns, { j });
//					auto [idx_x2, val_x2] = Operators<cpx>::sigma_x(idx_x, Ns, { nei });
//					this->H(idx_x2, k) += this->Ja * (1.0 - this->eta_a) * real(val_x * val_x2);
//
//					auto [idx_y, val_y] = Operators<cpx>::sigma_y(k, Ns, { j });
//					auto [idx_y2, val_y2] = Operators<cpx>::sigma_y(idx_y, Ns, { nei });
//					this->H(idx_y2, k) += this->Ja * (1.0 + this->eta_a) * real(val_y * val_y2);
//
//					auto [idx_z, val_z] = Operators<cpx>::sigma_z(k, Ns, { j });
//					auto [idx_z2, val_z2] = Operators<cpx>::sigma_z(idx_z, Ns, { nei });
//					this->H(idx_z2, k) += this->Ja * this->Delta_a * real(val_z * val_z2);
//
//				}
//			}
//
//			// -------------- CHECK NNN ---------------
//			for (auto nnn = 0; nnn < nnn_number; nnn++) {
//				auto n_num = this->lattice->get_nnn_forward_num(j, nnn);
//				//if (auto nei = this->lattice->get_nnn(j, n_num); nei >= 0 && (j > 0 || !this->parity_break)) {
//				if (auto nei = this->lattice->get_nnn(j, n_num); nei >= 0) {
//
//					// setting the neighbors elements
//					auto [idx_x, val_x] = Operators<cpx>::sigma_x(k, Ns, { j });
//					auto [idx_x2, val_x2] = Operators<cpx>::sigma_x(idx_x, Ns, { nei });
//					this->H(idx_x2, k) += this->Jb * (1.0 - this->eta_b) * real(val_x * val_x2);
//
//					auto [idx_y, val_y] = Operators<cpx>::sigma_y(k, Ns, { j });
//					auto [idx_y2, val_y2] = Operators<cpx>::sigma_y(idx_y, Ns, { nei });
//					this->H(idx_y2, k) += this->Jb * (1.0 + this->eta_b) * real(val_y * val_y2);
//
//					auto [idx_z, val_z] = Operators<cpx>::sigma_z(k, Ns, { j });
//					auto [idx_z2, val_z2] = Operators<cpx>::sigma_z(idx_z, Ns, { nei });
//					this->H(idx_z2, k) += this->Jb * this->Delta_b * real(val_z * val_z2);
//
//				}
//			}
//		}
//	}
//}

// ----------------------------------------------------------------------------- HAMILTONIAN -------------------------------------------------------------------------------------

/*
* Generates the total Hamiltonian of the system. The diagonal part is straightforward,
* while the non-diagonal terms need the specialized setHamiltonainElem(...) function
*/
template <typename _T>
void XYZ<_T>::hamiltonian() {
	this->init();
	for (u64 k = 0; k < this->Nh; k++)
		for (int site_ = 0; site_ <= this->Ns - 1; site_++)
			this->locEnergy(k, this->hilbertSpace.getMapping(k), site_);
}

#endif // !XYZ_H