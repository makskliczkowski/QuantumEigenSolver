#pragma once
/***********************************
* Is an instance of the QSM model.
* Derives from a general Hamiltonian.
* For more information see the QSM model.
* https://arxiv.org/pdf/2308.07431.pdf
************************************/

#ifndef HAMIL_H
#	include "../hamil.h"
#endif

#ifndef ROSENZWEIG_PORTER_H
#	define ROSENZWEIG_PORTER_H
#include "../quantities/statistics.h"

template<typename _T>
class RosenzweigPorter : public Hamiltonian<_T, 2>
{
public:
	using NQSFun = typename Hamiltonian<_T>::NQSFun;
protected:
	double gamma_ = 1.0;		// Hilbert-Schmidt norm of the coupling operator normalizer
	double gammaP_ = 1.0;
	double gammaP_inv_ = 1.0;
	arma::Col<double> diag_;
public:
	void randomize(double _a, double _s, const strVec& _which)	override final;
public:
	~RosenzweigPorter() override;

	RosenzweigPorter(const size_t _N, double _gamma = 0.0);
	RosenzweigPorter(const Hilbert::HilbertSpace<_T>& _hil, double _gamma = 0.0);
	RosenzweigPorter(Hilbert::HilbertSpace<_T>&& _hil, double _gamma = 0.0);

	// ############################################ Meth #############################################

	void initgamma(double _gamma)
	{ 
		this->gamma_ = _gamma; 
		this->gammaP_ = std::pow(this->Nh, _gamma);
		this->gammaP_inv_ = std::pow(this->Nh, -0.5 * _gamma); 
	};
	void hamiltonian()							override final;

	// ########################################### Methods ###########################################

	void locEnergy(u64 _elemId,
		u64 _elem,
		uint _site)								override final;
	cpx locEnergy(u64 _id,
		uint site,
		NQSFun f1)								override final		{ return 0; }
	cpx locEnergy(const DCOL& v,
		uint site,
		NQSFun f1)								override final		{ return 0; }

	// ############################################ Info #############################################

	std::string info(const strVec& skip = {}, 
		std::string sep	= "_", 
		int prec = 2)							const override final;
	void updateInfo()							override final		{ this->info_ = this->info({}, ",", 3); };

	// ###############################################################################################

};

// ##########################################################################################################################################

// ############################################################## C O N S T R ###############################################################

// ##########################################################################################################################################

/*
* @brief Destructor of the QSM model.
*/
template<typename _T>
inline RosenzweigPorter<_T>::~RosenzweigPorter()
{
	DESTRUCTOR_CALL;
}

// ##########################################################################################################################################

/*
* @brief Constructor of the QSM model. It takes the number of particles in the system and initializes the Hamiltonian.
* @param _N: the number of particles in the system.
* @param _gamma: multiplier of the offdiagonal couplings in the Hamiltonian.
*/
template<typename _T>
inline RosenzweigPorter<_T>::RosenzweigPorter(const size_t _N, double _gamma)
	: Hamiltonian<_T, 2>(_N), gamma_(_gamma)
{
	CONSTRUCTOR_CALL;
	//change info
	this->info_ = this->info();
	this->updateInfo();

	this->initgamma(_gamma);
}

template<typename _T>
inline RosenzweigPorter<_T>::RosenzweigPorter(const Hilbert::HilbertSpace<_T>& _hil, double _gamma)
	: Hamiltonian<_T, 2>(_hil), gamma_(_gamma)
{
	CONSTRUCTOR_CALL;
	//change info
	this->info_ = this->info();
	this->updateInfo();

	this->initgamma(_gamma);

}

template<typename _T>
inline RosenzweigPorter<_T>::RosenzweigPorter(Hilbert::HilbertSpace<_T>&& _hil, double _gamma)
	: Hamiltonian<_T, 2>(_hil), gamma_(_gamma)
{
	CONSTRUCTOR_CALL;
	//change info
	this->info_ = this->info();
	this->updateInfo();

	this->initgamma(_gamma);
}

// ##########################################################################################################################################

// ############################################################### I N I T S ################################################################

// ##########################################################################################################################################

/*
* @brief Randomizes the parameters of the QSM model. The parameters that can be randomized are: 
* alpha [alpha], magnetic field [h], distance random limits [xi] and the random dot GOE distribution [dot].
* The latter is done automatically and does not need to be specified.
* @param _around: the mean value of the random distribution.
* @param _str: the strength of the random distribution.
* @param _which: the parameters to randomize.
*/
template<typename _T>
inline void RosenzweigPorter<_T>::randomize(double _around, double _str, const strVec& _which)
{
	if (_which.empty())
	{
		LOGINFOG("Empty randomization list.", LOG_TYPES::INFO, 1);
		return;
	}
	this->diag_	= this->ran_.template randomNormal<double, double, arma::Col>(_around, _str, this->Nh);
}

// ##########################################################################################################################################

// ############################################################### I N F O S ################################################################

// ##########################################################################################################################################

/*
* @brief Returns the information about the QSM model.
* @param skip: the parameters to skip.
* @param sep: the separator between the parameters.
* @param prec: the precision of the output.
* @returns the information about the QSM model.
*/
template<typename _T>
inline std::string RosenzweigPorter<_T>::info(const strVec & skip, std::string sep, int prec) const
{
	std::string name		= sep + "rp,Ns=" + STR(this->Ns);
	if (std::is_same<_T, std::complex<double>>::value) 
		name += sep + "gue";
	else
		name += sep + "goe";
	name +=	sep +	VEQV(gamm, gamma_);
	name += this->hilbertSpace.getSymInfo();
	return this->Hamiltonian<_T>::info(name, skip, sep);
}

// ##########################################################################################################################################

// ############################################################### H A M I L ################################################################

// ##########################################################################################################################################

/*
* @brief Builds the Hamiltonian of the QSM model.
* @note The Hamiltonian is built in the following way:
* 1. The first term is the Hamiltonian of the dot particles. This is a random matrix.
* 2. The second term is the coupling between the dot and the outside world. This is a spin-flip interaction.
* 3. The third term is the magnetic field of the dot particles.
*/
template<typename _T>
inline void RosenzweigPorter<_T>::hamiltonian()
{
	if (this->Nh == 0)
	{
		LOGINFOG("Empty Hilbert, not building anything.", LOG_TYPES::INFO, 1);
		return;
	}
	this->init();

	this->randomize(0.0, 1.0, {"g"});

	this->H_.diag() = algebra::cast<_T>(this->diag_);

	// build the Hamiltonian (offdiagonal)
	this->H_ += this->gammaP_inv_ * this->ran_.template GUE<_T>(this->Nh_);
}

// ##########################################################################################################################################

/*
* @brief Calculates the local energy of the QSM model. The local energy is calculated for a specific particle at a specific site.
* @param _elemId: the index of the element in the Hilbert space.
* @param _elem: the element in the Hilbert space (considered when there are symmetries)
* @param _site: the site of the particle (or the position in the vector)
* @note The local energy is calculated in the following way:
*/
template<typename _T>
inline void RosenzweigPorter<_T>::locEnergy(u64 _elemId, u64 _elem, uint _site)
{

}

// ##########################################################################################################################################

#endif