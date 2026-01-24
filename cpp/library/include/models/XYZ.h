#pragma once
/***********************************
* Is an instance of the XYZ model.
* Derives from a general Hamiltonian.
************************************/

#ifndef HAMIL_H
#	include "../hamil.h"
#endif // !HAMIL_H

#ifndef XYZ_H
#define XYZ_H

/**
* @brief XYZ Hamiltonian with next nearest neighbors.
*/
template <typename _T>
class XYZ : public Hamiltonian<_T, 2> 
{
public:
	using NQSFun												= typename Hamiltonian<_T>::NQSFun;
protected:
	// ######################################### Parameters ########################################
	DISORDER_EQUIV(double, Ja);
	DISORDER_EQUIV(double, Jb);
	DISORDER_EQUIV(double, hx);
	DISORDER_EQUIV(double, hz);
	DISORDER_EQUIV(double, dA);
	DISORDER_EQUIV(double, dB);
	DISORDER_EQUIV(double, eA);
	DISORDER_EQUIV(double, eB);
	bool parityBreak_											= false;

public:
	// ######################################## Constructors ########################################
	~XYZ()														{ LOGINFO(this->info() + " - destructor called.", LOG_TYPES::INFO, 3); };
	XYZ()															= default;
	XYZ(const Hilbert::HilbertSpace<_T>& hilbert, double _Ja, double _Jb, 
												  double _hx, double _hz, 
												  double _dA, double _dB, 
												  double _eA, double _eB, 
												  bool _parityBreak = false);

	XYZ(Hilbert::HilbertSpace<_T>&& hilbert, double _Ja, double _Jb,
											 double _hx, double _hz,
											 double _dA, double _dB,
											 double _eA, double _eB,
											 bool _parityBreak = false);

	XYZ(const Hilbert::HilbertSpace<_T>& hilbert, double _Ja, double _Jb,
												  double _hx, double _hz,
												  double _dA, double _dB,
												  double _eA, double _eB,
												  double _Ja0, double _Jb0,
												  double _hx0, double _hz0,
												  double _dA0, double _dB0,
												  double _eA0, double _eB0,
												  bool _parityBreak = false);

	XYZ(Hilbert::HilbertSpace<_T>&& hilbert, double _Ja, double _Jb,
											 double _hx, double _hz,
											 double _dA, double _dB,
											 double _eA, double _eB,
											 double _Ja0, double _Jb0,
											 double _hx0, double _hz0,
											 double _dA0, double _dB0,
											 double _eA0, double _eB0,
											 bool _parityBreak = false);

	// ########################################### Methods ###########################################

	void locEnergy(u64 _elemId,
					u64 _elem,
					uint _site)					override final;
	cpx locEnergy(u64 _id,
				uint site,
				NQSFun f1)						override final;
	cpx locEnergy(const DCOL& v,
				uint site,
				NQSFun f1)						override final;

	// ############################################ Info #############################################

	std::string info(const strVec& skip = {}, 
					std::string sep	= "_", 
					int prec = 2)				const override final;
	void updateInfo()							override final		{ this->info_ = this->info({}, ",", 3); };
};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ######################################################## C O N S T R U C T O R S #########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/**
* @brief Set the info about the XYZ model.
* @param skip which parameters to skip
* @param sep separator
* @param prec precision of showing the parameters
*/
template<typename _T>
inline std::string XYZ<_T>::info(const strVec& skip, std::string sep, int prec) const
{
	auto BC				= this->hilbertSpace.getBC();
	std::string name	= sep + "xyz,Ns=" + STR(this->Ns);
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
	return this->Hamiltonian<_T>::info(name, skip, sep);
}

// ##########################################################################################################################################

template<typename _T>
inline XYZ<_T>::XYZ(const Hilbert::HilbertSpace<_T>& hilbert, double _Ja, double _Jb, double _hx, double _hz, double _dA, double _dB, double _eA, double _eB, bool _parityBreak)
	: Hamiltonian<_T>(hilbert), Ja(_Ja), Jb(_Jb), hx(_hx), hz(_hz), dA(_dA), dB(_dB), eA(_eA), eB(_eB), parityBreak_(_parityBreak)
{
	this->ran_	= randomGen();
	this->Ns		= this->hilbertSpace.getLatticeSize();
	this->dJa	= ZEROV(this->Ns);
	this->dJb	= ZEROV(this->Ns);
	this->dhx	= ZEROV(this->Ns);
	this->dhz	= ZEROV(this->Ns);
	this->ddA	= ZEROV(this->Ns);
	this->ddB	= ZEROV(this->Ns);
	this->deA	= ZEROV(this->Ns);
	this->deB	= ZEROV(this->Ns);
	this->type_	= MY_MODELS::XYZ_M;

	//change info
	this->info_ = this->info();
	this->updateInfo();
}

// ##########################################################################################################################################

template<typename _T>
inline XYZ<_T>::XYZ(Hilbert::HilbertSpace<_T>&& hilbert, double _Ja, double _Jb, double _hx, double _hz, double _dA, double _dB, double _eA, double _eB, bool _parityBreak)
	: Hamiltonian<_T>(std::move(hilbert)), Ja(_Ja), Jb(_Jb), hx(_hx), hz(_hz), dA(_dA), dB(_dB), eA(_eA), eB(_eB), parityBreak_(_parityBreak)
{
	this->ran_	= randomGen();
	this->Ns		= this->hilbertSpace.getLatticeSize();
	this->dJa	= ZEROV(this->Ns);
	this->dJb	= ZEROV(this->Ns);
	this->dhx	= ZEROV(this->Ns);
	this->dhz	= ZEROV(this->Ns);
	this->ddA	= ZEROV(this->Ns);
	this->ddB	= ZEROV(this->Ns);
	this->deA	= ZEROV(this->Ns);
	this->deB	= ZEROV(this->Ns);
	this->type_ = MY_MODELS::XYZ_M;

	//change info
	this->info_ = this->info();
	this->updateInfo();
}

// ##########################################################################################################################################

template<typename _T>
inline XYZ<_T>::XYZ(const Hilbert::HilbertSpace<_T>& hilbert,	double _Ja, double _Jb,
																					double _hx, double _hz,
																					double _dA, double _dB,
																					double _eA, double _eB,
																					double _Ja0, double _Jb0,
																					double _hx0, double _hz0,
																					double _dA0, double _dB0,
																					double _eA0, double _eB0,
																					bool _parityBreak)
	: XYZ<_T>(hilbert, _Ja, _Jb, _hx, _hz, _dA, _dB, _eA, _eB, _parityBreak)
{
	this->Ja0	= _Ja0; this->Jb0 = _Jb0; this->hx0 = _hx0;
	this->hz0	= _hz0; this->dA0 = _dA0; this->dB0 = _dB0;
	this->eA0	= _eA0; this->eB0 = _eB0;
	this->dJa	= this->ran_.template createRanVec<double>(this->Ns, this->Ja0);
	this->dJb	= this->ran_.template createRanVec<double>(this->Ns, this->Jb0);
	this->dhx	= this->ran_.template createRanVec<double>(this->Ns, this->hx0);
	this->dhz	= this->ran_.template createRanVec<double>(this->Ns, this->hz0);
	this->ddA	= this->ran_.template createRanVec<double>(this->Ns, this->dA0);
	this->ddB	= this->ran_.template createRanVec<double>(this->Ns, this->dB0);
	this->deA	= this->ran_.template createRanVec<double>(this->Ns, this->eA0);
	this->deB	= this->ran_.template createRanVec<double>(this->Ns, this->eB0);
	LOGINFOG("I am XYZ model: " + this->info_, LOG_TYPES::CHOICE, 1);
	double Jx	= this->Ja * (1 - this->eA);
	double Jy	= this->Ja * (1 + this->eA);
	double Jz	= this->Ja * this->dA;
	LOGINFOG(VEQ(Jx) + "," + VEQ(Jy) + "," + VEQ(Jz), LOG_TYPES::CHOICE, 1);
	auto SUSY	= Jx * Jy + Jy * Jz + Jx * Jz;
	LOGINFOG(VEQ(SUSY), LOG_TYPES::CHOICE, 1);
};

// ##########################################################################################################################################

template<typename _T>
inline XYZ<_T>::XYZ(Hilbert::HilbertSpace<_T>&& hilbert,			double _Ja, double _Jb,
																					double _hx, double _hz,
																					double _dA, double _dB,
																					double _eA, double _eB,
																					double _Ja0, double _Jb0,
																					double _hx0, double _hz0,
																					double _dA0, double _dB0,
																					double _eA0, double _eB0,
																					bool _parityBreak)
	: XYZ<_T>(std::move(hilbert), _Ja, _Jb, _hx, _hz, _dA, _dB, _eA, _eB, _parityBreak)
{
	this->Ja0	= _Ja0; this->Jb0 = _Jb0; this->hx0 = _hx0;
	this->hz0	= _hz0; this->dA0 = _dA0; this->dB0 = _dB0;
	this->eA0	= _eA0; this->eB0 = _eB0;
	this->dJa	= this->ran_.template createRanVec<double>(this->Ns, this->Ja0);
	this->dJb	= this->ran_.template createRanVec<double>(this->Ns, this->Jb0);
	this->dhx	= this->ran_.template createRanVec<double>(this->Ns, this->hx0);
	this->dhz	= this->ran_.template createRanVec<double>(this->Ns, this->hz0);
	this->ddA	= this->ran_.template createRanVec<double>(this->Ns, this->dA0);
	this->ddB	= this->ran_.template createRanVec<double>(this->Ns, this->dB0);
	this->deA	= this->ran_.template createRanVec<double>(this->Ns, this->eA0);
	this->deB	= this->ran_.template createRanVec<double>(this->Ns, this->eB0);
	LOGINFOG("I am XYZ model: " + this->info_, LOG_TYPES::CHOICE, 1);
	double Jx	= this->Ja * (1 - this->eA);
	double Jy	= this->Ja * (1 + this->eA);
	double Jz	= this->Ja * this->dA;
	LOGINFOG(VEQ(Jx) + "," + VEQ(Jy) + "," + VEQ(Jz), LOG_TYPES::CHOICE, 1);
	auto SUSY	= Jx * Jy + Jy * Jz + Jx * Jz;
	LOGINFOG(VEQ(SUSY), LOG_TYPES::CHOICE, 1);
};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ######################################################## L O C A L   E N E R G Y #########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _id base state index
*/
template <typename _T>
cpx XYZ<_T>::locEnergy(u64 _cur, uint _site, XYZ<_T>::NQSFun _fun)
{
	// value that does not change
	double localVal	= 0.0;
	cpx changedVal	= 0.0;

	// get number of forward nn
	uint NUM_OF_NN	=	(uint)this->lat_->get_nn_ForwardNum(_site);
	uint NUM_OF_NNN	=	(uint)this->lat_->get_nnn_ForwardNum(_site);

	// -------------- perpendicular field --------------
	const double si	=	checkBit(_cur, this->Ns - _site - 1) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
	localVal		+=	PARAM_W_DISORDER(hz, _site) * si;

	// ---------------- transverse field ---------------
	if (!EQP(this->hx, 0.0, 1e-9)) 
		changedVal		+= _fun({ (int)_site }, { si }) * Operators::_SPIN_RBM * PARAM_W_DISORDER(hx, _site);

	// ------------------- CHECK NN --------------------
	for (uint nn = 0; nn < NUM_OF_NN; nn++) 
	{
		uint N_NUMBER = this->lat_->get_nn_ForwardNum(_site, nn);
		if (int nei = this->lat_->get_nn(_site, N_NUMBER); nei >= 0) 
		{
			// SZiSZj
			const double sj		=	checkBit(_cur, this->Ns - nei - 1) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
			localVal			+=	PARAM_W_DISORDER(dA, _site) * PARAM_W_DISORDER(Ja, _site) * si * sj;

			// SYiSYj
			auto siY			=	si > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
			auto sjY			=	sj > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
			auto changedIn		=	siY * sjY * PARAM_W_DISORDER(Ja, _site) * (1.0 + PARAM_W_DISORDER(eA, _site));

			// SXiSXj
			changedIn			+=	Operators::_SPIN_RBM * Operators::_SPIN_RBM * PARAM_W_DISORDER(Ja, _site) * (1.0 - PARAM_W_DISORDER(eA, _site));
			
			// apply change
			changedVal			+=	_fun({ (int)_site, nei }, { si, sj }) * changedIn; 
		}
	}

	// ------------------- CHECK NNN --------------------
	if (!EQP(this->Jb, 0.0, 1e-9))
	{
		for (uint nnn = 0; nnn < NUM_OF_NNN; nnn++) 
		{
			uint N_NUMBER = this->lat_->get_nnn_ForwardNum(_site, nnn);
			if (int nei = this->lat_->get_nnn(_site, N_NUMBER); nei >= 0) 
			{
				// SZiSZj
				const double sj	=	checkBit(_cur, this->Ns - nei - 1) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
				localVal				+= PARAM_W_DISORDER(dB, _site) * PARAM_W_DISORDER(Jb, _site) * si * sj;

				// SYiSYj
				auto siY				=	si > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
				auto sjY				=	sj > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
				auto changedIn		=  siY * sjY * PARAM_W_DISORDER(Jb, _site) * (1.0 + PARAM_W_DISORDER(eB, _site));

				// SXiSXj
				changedIn			+= Operators::_SPIN_RBM * Operators::_SPIN_RBM * PARAM_W_DISORDER(Jb, _site) * (1.0 - PARAM_W_DISORDER(eB, _site));
			
				// apply change
				changedVal			+= _fun({ (int)_site, nei }, { si, sj }) * changedIn; 
			}
		}
	}
	
	// return
	return changedVal + localVal;
}

// ##########################################################################################################################################

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _v base state
* @param _site lattice site
* @param _fun function for Neural Network Quantum State whenever the state changes (after potential flip) - nondiagonal
*/
template <typename _T>
cpx XYZ<_T>::locEnergy(const DCOL& _cur, uint _site, XYZ<_T>::NQSFun _fun)
{
	// value that does not change
	double localVal	= 0.0;
	cpx changedVal	= 0.0;

	// get number of forward nn
	uint NUM_OF_NN	=	(uint)this->lat_->get_nn_ForwardNum(_site);
	uint NUM_OF_NNN	=	(uint)this->lat_->get_nnn_ForwardNum(_site);

	// -------------- perpendicular field --------------
	const double si	=	checkBit(_cur, _site) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
	localVal		+=	PARAM_W_DISORDER(hz, _site) * si;

	// ---------------- transverse field ---------------
	if (!EQP(this->hx, 0.0, 1e-9)) 
		changedVal		+= _fun({ (int)_site }, { si }) * Operators::_SPIN_RBM * PARAM_W_DISORDER(hx, _site);

	// ------------------- CHECK NN --------------------
	for (uint nn = 0; nn < NUM_OF_NN; nn++) 
	{
		uint N_NUMBER = this->lat_->get_nn_ForwardNum(_site, nn);
		if (int nei = this->lat_->get_nn(_site, N_NUMBER); nei >= 0) 
		{
			// SZiSZj
			const double sj		=	checkBit(_cur, nei) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
			localVal			+=	PARAM_W_DISORDER(dA, _site) * PARAM_W_DISORDER(Ja, _site) * si * sj;

			// SYiSYj
			auto siY			=	si > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
			auto sjY			=	sj > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
			auto changedIn		=	siY * sjY * PARAM_W_DISORDER(Ja, _site) * (1.0 + PARAM_W_DISORDER(eA, _site));

			// SXiSXj
			changedIn			+=	Operators::_SPIN_RBM * Operators::_SPIN_RBM * PARAM_W_DISORDER(Ja, _site) * (1.0 - PARAM_W_DISORDER(eA, _site));
			
			// apply change
			changedVal			+=	_fun({ (int)_site, nei }, { si, sj }) * changedIn; 
		}
	}

	// ------------------- CHECK NNN --------------------
	if (!EQP(this->Jb, 0.0, 1e-9))
	{
		for (uint nnn = 0; nnn < NUM_OF_NNN; nnn++) 
		{
			uint N_NUMBER = this->lat_->get_nnn_ForwardNum(_site, nnn);
			if (int nei = this->lat_->get_nnn(_site, N_NUMBER); nei >= 0) 
			{
				// SZiSZj
				const double sj			=	checkBit(_cur, nei) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
				localVal				+=	PARAM_W_DISORDER(dB, _site) * PARAM_W_DISORDER(Jb, _site) * si * sj;

				// SYiSYj
				auto siY				=	si > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
				auto sjY				=	sj > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
				auto changedIn			=	siY * sjY * PARAM_W_DISORDER(Jb, _site) * (1.0 + PARAM_W_DISORDER(eB, _site));

				// SXiSXj
				changedIn				+=	Operators::_SPIN_RBM * Operators::_SPIN_RBM * PARAM_W_DISORDER(Jb, _site) * (1.0 - PARAM_W_DISORDER(eB, _site));
			
				// apply change
				changedVal				+=	_fun({ (int)_site, nei }, { si, sj }) * changedIn; 
			}
		}
	}
	
	// return
	return changedVal + localVal;
}

// ##########################################################################################################################################

/**
* @brief body of setting up of the Hamiltonian
*/
template<typename _T>
inline void XYZ<_T>::locEnergy(u64 _elemId, u64 _elem, uint _site)
{
	// get number of forward nn
	uint NUM_OF_NN		= (uint)this->lat_->get_nn_ForwardNum(_site);
	uint NUM_OF_NNN		= (uint)this->lat_->get_nnn_ForwardNum(_site);
	u64 newIdx			= 0;
	_T newVal			= 0;

	// -------------- perpendicular field --------------
	std::tie(newIdx, newVal) = Operators::sigma_z<_T>(_elem, this->Ns, { _site });
	this->setHElem(_elemId, PARAM_W_DISORDER(hz, _site) * newVal, newIdx);

	if (this->parityBreak_ && (_site == 0 || _site == this->Ns - 1))
		this->setHElem(_elemId, 1.0, newIdx);

	// -------------- transverse field --------------
	if (!EQP(this->hx, 0.0, 1e-9)) {
		std::tie(newIdx, newVal) = Operators::sigma_x(_elem, this->Ns, { _site });
		this->setHElem(_elemId, PARAM_W_DISORDER(hx, _site) * newVal, newIdx);
	}

	// -------------------------------------------------------- CHECK NN ---------------------------------------------------------
	for (uint nn = 0; nn < NUM_OF_NN; nn++) {
		uint N_NUMBER = this->lat_->get_nn_ForwardNum(_site, nn);
		if (int nei = this->lat_->get_nn(_site, N_NUMBER); nei >= 0) {
			// SZiSZj
			auto [idx_z, val_z]		= Operators::sigma_z<_T>(_elem, this->Ns,	{ _site });
			auto [idx_z2, val_z2]	= Operators::sigma_z<_T>(idx_z, this->Ns,	{ (uint)nei });
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
							PARAM_W_DISORDER(Ja, _site) * (1.0 - PARAM_W_DISORDER(eA, _site)) * val_x * val_x2,
							idx_x2);
		}
	}

	// -------------------------------------------------------- CHECK NNN ---------------------------------------------------------
	for (uint nnn = 0; nnn < NUM_OF_NNN; nnn++) {
		uint N_NUMBER = this->lat_->get_nnn_ForwardNum(_site, nnn);
		if (int nei = this->lat_->get_nnn(_site, N_NUMBER); nei >= 0) {
			// SZiSZj
			auto [idx_z, val_z]		= Operators::sigma_z<_T>(_elem, this->Ns,	{ _site });
			auto [idx_z2, val_z2]	= Operators::sigma_z<_T>(idx_z, this->Ns,	{ (uint)nei });
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
							PARAM_W_DISORDER(Jb, _site) * (1.0 - PARAM_W_DISORDER(eB, _site)) * val_x * val_x2,
							idx_x2);
		}
	}
}

// ##########################################################################################################################################

#endif // !XYZ_H