#pragma once
/***********************************
* Is an instance of the Heisenberg-Kitaev
* model. Derives from a general Hamiltonian.
************************************/

#ifndef HAMIL_H
#	include "../hamil.h"
#endif // !HAMIL_H

#ifndef HEISENBERG_KITAEV_H
#define HEISENBERG_KITAEV_H

/*
* @brief HeisenbergKitaev Hamiltonian.
*/
template <typename _T>
class HeisenbergKitaev : public Hamiltonian<_T, 2>
{
public:
	using NQSFun								= typename Hamiltonian<_T>::NQSFun;
protected:
	// ######################################### Parameters ########################################
	std::vector<double> Kx;
	std::vector<double> Ky;
	std::vector<double> Kz;
	std::vector<double> J;
	std::vector<double> delta;
	std::vector<double> hz;
	std::vector<double> hx;

public:
	// ######################################## Constructors ########################################
	~HeisenbergKitaev()						{ LOGINFO(this->info() + " - destructor called.", LOG_TYPES::INFO, 3); };
	HeisenbergKitaev()						= default;
	HeisenbergKitaev( const Hilbert::HilbertSpace<_T>& hilbert,
							const std::vector<double>& _Kx,
							const std::vector<double>& _Ky,
							const std::vector<double>& _Kz,
							const std::vector<double>& _J,
							const std::vector<double>& _delta,
							const std::vector<double>& _hz = {},
							const std::vector<double>& _hx = {});
	HeisenbergKitaev(Hilbert::HilbertSpace<_T>&& hilbert,
							const std::vector<double>& _Kx,
							const std::vector<double>& _Ky,
							const std::vector<double>& _Kz,
							const std::vector<double>& _J,
							const std::vector<double>& _delta,
							const std::vector<double>& _hz = {},
							const std::vector<double>& _hx = {});

	// ########################################### Methods ###########################################
	void locEnergy(u64 _elemId,
						u64 _elem,
						uint _site)				override final;
	cpx locEnergy(u64 _id,
						uint site,
						NQSFun f1)				override final;

	cpx locEnergy(const arma::Col<double>& _id,
						uint site,
						NQSFun f1)				override final;

	// ############################################ Info #############################################

	std::string info(	const strVec& skip	= {},
							std::string sep		= "_",
							int prec = 2)		const override final;
	void updateInfo()							override final { this->info_ = this->info({}, ",", 3); };
};

// ##########################################################################################################################################
// ##########################################################################################################################################
// ######################################################## C O N S T R U C T O R S #########################################################
// ##########################################################################################################################################
// ##########################################################################################################################################

/*
* @brief Set the info about the Heisenberg-Kitaev model.
* @param skip which parameters to skip
* @param sep separator
* @param prec precision of showing the parameters
*/
template<typename _T>
inline std::string HeisenbergKitaev<_T>::info(const strVec& skip, std::string sep, int prec) const
{
	bool _different_J	= !std::equal(this->J.begin() + 1, this->J.end(), this->J.begin());
	bool _different_Kx	= !std::equal(this->Kx.begin() + 1, this->Kx.end(), this->Kx.begin());
	bool _different_Ky	= !std::equal(this->Ky.begin() + 1, this->Ky.end(), this->Ky.begin());
	bool _different_Kz	= !std::equal(this->Kz.begin() + 1, this->Kz.end(), this->Kz.begin());
	bool _different_dlt	= !std::equal(this->delta.begin() + 1, this->delta.end(), this->delta.begin());
	bool _different_hz	= !std::equal(this->hz.begin() + 1, this->hz.end(), this->hz.begin());
	bool _different_hx	= !std::equal(this->hx.begin() + 1, this->hx.end(), this->hx.begin());
	auto BC					= this->hilbertSpace.getBC();

	std::string name		= sep + "heikit,Ns=" + STR(this->Ns);
	name += "," + (_different_J  ? "J=r"		: VEQVP(J,		J[0],	3));
	name += "," + (_different_Kx ? "Kx=r"		: VEQVP(Kx,		Kx[0],	3));
	name += "," + (_different_Ky ? "Ky=r"		: VEQVP(Ky,		Ky[0],	3));
	name += "," + (_different_Kz ? "Kz=r"		: VEQVP(Kz,		Kz[0],	3));
	name += "," + (_different_dlt? "dlt=r"		: VEQVP(delta, delta[0],3));
	name += "," + (_different_hz ? "hz=r"		: VEQVP(hz,		hz[0],	3));
	name += "," + (_different_hx ? "hx=r"		: VEQVP(hx,		hx[0],	3));
	name += this->hilbertSpace.getSymInfo();
	name += "," + VEQ(BC);
	return this->Hamiltonian<_T>::info(name, skip, sep);
}

// ##########################################################################################################################################

template<typename _T>
inline HeisenbergKitaev<_T>::HeisenbergKitaev(const Hilbert::HilbertSpace<_T>& hilbert, 
															 const std::vector<double>& _Kx,
															 const std::vector<double>& _Ky,
															 const std::vector<double>& _Kz,
															 const std::vector<double>& _J,
															 const std::vector<double>& _delta,
															 const std::vector<double>& _hz,
															 const std::vector<double>& _hx)


	: Hamiltonian<_T>(hilbert), Kx(_Kx), Ky(_Ky), Kz(_Kz), J(_J), delta(_delta)
{
	// handle perpendicular field
	if (_hz.size() == 0)
		this->hz = std::vector<double>(J.size(), 0.0);
	else
		this->hz = _hz;
	// handle transverse field
	if (_hx.size() == 0)
		this->hx = std::vector<double>(J.size(), 0.0);
	else
		this->hx = _hx;

	this->ran_	= randomGen();
	this->Ns	= this->hilbertSpace.getLatticeSize();
	this->type_ = MY_MODELS::HEI_KIT_M;

	//change info
	this->info_ = this->info();
	this->updateInfo();
}

template<typename _T>
inline HeisenbergKitaev<_T>::HeisenbergKitaev(Hilbert::HilbertSpace<_T>&& hilbert, 
															 const std::vector<double>& _Kx,
															 const std::vector<double>& _Ky,
															 const std::vector<double>& _Kz,
															 const std::vector<double>& _J,
															 const std::vector<double>& _delta,
															 const std::vector<double>& _hz,
															 const std::vector<double>& _hx)
	: Hamiltonian<_T>(std::move(hilbert)), Kx(_Kx), Ky(_Ky), Kz(_Kz), J(_J), delta(_delta)
{
	// handle perpendicular field
	if (_hz.size() == 0)
		this->hz = std::vector<double>(J.size(), 0.0);
	else
		this->hz = _hz;
	// handle transverse field
	if (_hx.size() == 0)
		this->hx = std::vector<double>(J.size(), 0.0);
	else
		this->hx = _hx;

	this->ran_	= randomGen();
	this->Ns		= this->hilbertSpace.getLatticeSize();
	this->type_ = MY_MODELS::HEI_KIT_M;

	//change info
	this->info_ = this->info();
	this->updateInfo();
}

// ##########################################################################################################################################

// ######################################################## L O C A L   E N E R G Y #########################################################

// ##########################################################################################################################################

/*
* Calculate the local energy end return the corresponding vectors with the value
* @param _cur base state index
* @param _site lattice site
* @param _fun function for Neural Network Quantum State whenever the state changes (after potential flip) - nondiagonal
*/
template <typename _T>
cpx HeisenbergKitaev<_T>::locEnergy(u64 _cur, uint _site, HeisenbergKitaev<_T>::NQSFun _fun)
{
	// value that does not change
	double localVal		= 0.0;
	cpx changedVal		= 0.0;

	// get number of forward nn
	uint NUM_OF_NN		= (uint)this->lat_->get_nn_ForwardNum(_site);

	// -------------- perpendicular field --------------
	const double si		=	checkBit(_cur, this->Ns - _site - 1) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
	localVal			+= hz[_site] * si;

	// ---------------- transverse field ---------------
	if (!EQP(this->hx[_site], 0.0, 1e-9))
		changedVal += _fun({ (int)_site }, { si }) * Operators::_SPIN_RBM * hx[_site];

	// ------------------- CHECK NN --------------------
	for (uint nn = 0; nn < NUM_OF_NN; nn++)
	{
		const uint N_NUMBER = nn;//this->lat_->get_nn(_site);

		// get the nearest neighbor
		if (int nei = this->lat_->get_nnf(_site, N_NUMBER); nei >= 0) 
		{
			// --------------------- HEISENBERG ---------------------
			// SZiSZj
			const double sj		=	checkBit(_cur, this->Ns - nei - 1) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
			localVal			+= J[_site] * delta[_site] * si * sj;

			// SYiSYj
			auto siY			=	si > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
			auto sjY			=	sj > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
			auto changedIn		=	siY * sjY * J[_site];

			// SXiSXj
			changedIn			+=	Operators::_SPIN_RBM * Operators::_SPIN_RBM * J[_site];

			// ----------------------- KITAEV -----------------------
			// z_bond
			if (N_NUMBER == 0)
				localVal		+= this->Kz[_site] * si * sj;
			// y_bond
			else if (N_NUMBER == 1)
				changedIn		+= siY * sjY * Ky[_site];
			// x_bond
			else if (N_NUMBER == 2)
				changedIn		+= Operators::_SPIN_RBM * Operators::_SPIN_RBM * Kx[_site];

			// apply change
			changedVal			+= _fun({ (int)_site, nei }, { si, sj }) * changedIn;
		}
	}
	// return all
	return changedVal + localVal;
}

// ##########################################################################################################################################

template<typename _T>
inline cpx HeisenbergKitaev<_T>::locEnergy(const arma::Col<double>& _cur, uint _site, NQSFun _fun)
{
	// value that does not change
	double localVal		= 	0.0;
	cpx changedVal		= 	0.0;

	// get number of forward nn
	const uint NUM_OF_NN= (uint)this->lat_->get_nn_ForwardNum(_site);

	// -------------- perpendicular field --------------
	const double si		=	Binary::check(_cur, _site) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
	if (!EQP(this->hz[_site], 0.0, 1e-9))
		localVal		+= 	this->hz[_site] * si;

	// ---------------- transverse field ---------------
	if (!EQP(this->hx[_site], 0.0, 1e-9))
		changedVal 		+= 	_fun({ (int)_site }, { si }) * Operators::_SPIN_RBM * hx[_site];

	// ------------------- CHECK NN --------------------
	for (uint nn = 0; nn < NUM_OF_NN; nn++)
	{
		const uint N_NUMBER = nn;//this->lat_->get_nn(_site);

		// get the nearest neighbor
		if (int nei = this->lat_->get_nnf(_site, N_NUMBER); nei >= 0) 
		{
			// --------------------- HEISENBERG ---------------------
			// SZiSZj
			const double sj		= 	Binary::check(_cur, nei) ? Operators::_SPIN_RBM : -Operators::_SPIN_RBM;
			localVal			+= 	this->J[_site] * this->delta[_site] * si * sj;

			// SYiSYj
			auto siY			=	si > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
			auto sjY			=	sj > 0 ? I * Operators::_SPIN_RBM : -I * Operators::_SPIN_RBM;
			auto changedIn		=	siY * sjY * this->J[_site];

			// SXiSXj
			changedIn			+=	Operators::_SPIN_RBM * Operators::_SPIN_RBM * this->J[_site];

			// ----------------------- KITAEV -----------------------
			if (N_NUMBER == 0) 		// z_bond
				localVal		+= this->Kz[_site] * si * sj;
			else if (N_NUMBER == 1) // y_bond
				changedIn		+= siY * sjY * Ky[_site];
			else if (N_NUMBER == 2) // x_bond
				changedIn		+= Operators::_SPIN_RBM * Operators::_SPIN_RBM * Kx[_site];

			// apply change
			changedVal			+= _fun({ (int)_site, nei }, { si, sj }) * changedIn;
		}
	}
	return changedVal + localVal;
}

// ##########################################################################################################################################

/*
* @brief body of setting up of the Hamiltonian
*/
template<typename _T>
inline void HeisenbergKitaev<_T>::locEnergy(u64 _elemId, u64 _elem, uint _site)
{
	uint NUM_OF_NN		= (uint)this->lat_->get_nn_ForwardNum(_site);			// get number of forward nn at a given site _site 
	u64 newIdx			= 0;													// new index (for the operators)		
	_T newVal			= 0;													// new value (for the operators)

	// -------------- perpendicular field --------------
	if (!EQP(this->hz[_site], 0.0, 1e-9)) {
		std::tie(newIdx, newVal) = Operators::SpinOperators::sig_z<_T>(_elem, this->Ns_, { _site });
		this->setHElem(_elemId, hz[_site] * newVal, newIdx);
	}

	// -------------- transverse field --------------
	if (!EQP(this->hx[_site], 0.0, 1e-9)) {
		std::tie(newIdx, newVal) = Operators::SpinOperators::sig_x<_T>(_elem, this->Ns_, { _site });
		this->setHElem(_elemId, hx[_site] * newVal, newIdx);
	}

	// ------------------- CHECK NN --------------------

#ifdef _DEBUG 
	if (_elemId == 0)
		std::cout << "Site: " << _site << " - ";
#endif

	for (uint nn = 0; nn < NUM_OF_NN; nn++)
	{
		const uint N_NUMBER = nn;//this->lat_->get_nn(_site);

		// get the nearest neighbor
		if (int nei = this->lat_->get_nnf(_site, N_NUMBER); nei >= 0) 
		{
			#ifdef _DEBUG 
				if (_elemId == 0)
					std::cout << nei << "[" << (N_NUMBER == 0 ? "Z" : (N_NUMBER == 1 ? "Y" : "X")) << "]" << " - ";
			#endif
			// --------------------- HEISENBERG ---------------------
			// SZiSZj (diagonal elements)
			auto [idx_z, val_z]		= Operators::SpinOperators::sig_z<_T>(_elem, this->Ns_, { _site });
			auto [idx_z2, val_z2]	= Operators::SpinOperators::sig_z<_T>(idx_z, this->Ns_, { (uint)nei });
			this->setHElem(_elemId,	J[_site] * delta[_site] * (val_z * val_z2), idx_z2);

			// SYiSYj
			auto [idx_y, val_y]		= Operators::sigma_y(_elem, this->Ns, { _site });
			auto [idx_y2, val_y2]	= Operators::sigma_y(idx_y, this->Ns, { (uint)nei });
			this->setHElem(_elemId, J[_site] * algebra::real(val_y * val_y2), idx_y2);

			// SXiSXj
			auto [idx_x, val_x]		= Operators::SpinOperators::sig_x<_T>(_elem, this->Ns, { _site });
			auto [idx_x2, val_x2]	= Operators::SpinOperators::sig_x<_T>(idx_x, this->Ns, { (uint)nei });
			this->setHElem(_elemId,	J[_site] * val_x * val_x2,	idx_x2);

			// ----------------------- KITAEV -----------------------
			if (N_NUMBER == 0) 		// z_bond
				this->setHElem(_elemId, this->Kz[_site] * algebra::real(val_z * val_z2), idx_z2);
			else if (N_NUMBER == 1) // y_bond
				this->setHElem(_elemId, this->Ky[_site] * algebra::real(val_y * val_y2), idx_y2);
			else if (N_NUMBER == 2) // x_bond
				this->setHElem(_elemId, this->Kx[_site] * algebra::real(val_x * val_x2), idx_x2);
		}
	}
#ifdef _DEBUG
	if (_elemId == 0)
	{
		std::cout << std::endl;
		std::flush(std::cout);
	}	
#endif	
}



#endif