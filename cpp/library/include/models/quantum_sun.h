#pragma once
/***********************************
* Is an instance of the QSM model.
* Derives from a general Hamiltonian.
* https://arxiv.org/pdf/2308.07431.pdf
************************************/

#ifndef HAMIL_H
#	include "../hamil.h"
#endif

#ifndef QSM_H
#	define QSM_H

template<typename _T>
class QSM : public Hamiltonian<_T, 2>
{
public:
	using NQSFun = typename Hamiltonian<_T>::NQSFun; 
protected:
	// particle in a dot
	arma::Mat<_T> Hdot_;			// Hamiltonian of the dot particles (random matrix)
	size_t Nin_		=	1;			// number of particle within a dot
	size_t Nout_	=	1;			// number of particles outside the dot
	u64 dimIn_		=	1;			// dimension of the dot Hilbert space
	u64 dimOut_		=	1;			// dimension of the dot Hilbert space
	v_1d<double> u_;				// "distances" of particles within a dot
	v_1d<double> au_;				// "distances" of particles within a dot
	v_1d<size_t> n_;				// indices of the particles within a dot randomly choosen

	double gamma_	=	1.0;		// Hilbert-Schmidt norm of the coupling operator normalizer
	double g0_ 		=	1.0;		// coupling strength between the dot and outside world
	v_1d<double> a_;				// coupling between the dot and outside world
	v_1d<double> h_;				// internal dot magnetic field
	v_1d<double> xi_;				// random box distribution couplings

	void checkSizes();
	void initializeParticles();

public:
	~QSM() override;
	QSM(const size_t _Nall);
	QSM(const size_t _Nall,
		const size_t _N,
		const double _gamma,
		const double _g0,
		const v_1d<double>& _a,
		const v_1d<double>& _h,
		const v_1d<double>& _xi);
	QSM(const Hilbert::HilbertSpace<_T>& _hil,
		const size_t _N,
		const double _gamma,
		const double _g0,
		const v_1d<double>& _a,
		const v_1d<double>& _h,
		const v_1d<double>& _xi);
	QSM(const Hilbert::HilbertSpace<_T>&& _hil,
		const size_t _N,
		const double _gamma,
		const double _g0,
		const v_1d<double>& _a,
		const v_1d<double>& _h,
		const v_1d<double>& _xi);

	// ############################################ Meth #############################################

	void hamiltonian()							override final;

	// ########################################### Methods ###########################################

	void locEnergy(u64 _elemId,
		u64 _elem,
		uint _site)								override final;
	cpx locEnergy(u64 _id,
		uint site,
		NQSFun f1)								override final { return 0; }
	cpx locEnergy(const DCOL& v,
		uint site,
		NQSFun f1)								override final { return 0; }

	// ############################################ Info #############################################

	std::string info(const strVec& skip = {}, 
		std::string sep	= "_", 
		int prec = 2)							const override final;
	void updateInfo()							override final		{ this->info_ = this->info({}, ",", 3); };

};

// ##########################################################################################################################################

// ############################################################## C O N S T R ###############################################################

// ##########################################################################################################################################

template<typename _T>
inline QSM<_T>::~QSM()
{
	DESTRUCTOR_CALL;
}

template<typename _T>
inline void QSM<_T>::checkSizes()
{
	if(this->Nin_ >= this->Ns_)
		throw(std::invalid_argument("QSM: the number of particles in the dot is larger than the number of particles in the system."));

	this->Nout_ = this->Ns_ - this->Nin_;

	if(this->a_.size() != this->Nout_)	
		throw(std::invalid_argument("QSM: the size of the coupling vector 'a' is not equal to the number of particles outside the dot."));

	if (this->xi_.size() != this->Nout_)
		throw(std::invalid_argument("QSM: the size of the random box distribution vector 'xi' is not equal to the number of particles outside the dot."));

	if(this->h_.size() != this->Nout_)
		throw(std::invalid_argument("QSM: the size of the magnetic field vector 'h' is not equal to the number of particles outside the dot."));

}

template<typename _T>
inline QSM<_T>::QSM(const size_t _Nall)
	: Hamiltonian<_T, 2>(_Nall)
{
	CONSTRUCTOR_CALL;
}

template<typename _T>
inline QSM<_T>::QSM(const size_t _Nall, const size_t _N, 
	const double _gamma, const double _g0, 
	const v_1d<double>& _a, const v_1d<double>& _h, const v_1d<double>& _xi)
	: QSM(_Nall), Nin_(_N), gamma_(_gamma), g0_(_g0), a_(_a), h_(_h), xi_(_xi)
{	
	// we will keep the particles from the dot at the beginning of the vectors
	// remember that this Hamiltonian is zero-dimensional, so we don't need to worry about the order of the particles

	// check the validity of the input
	this->checkSizes();

	this->initializeParticles();
}

// ##########################################################################################################################################

template<typename _T>
inline QSM<_T>::QSM(const Hilbert::HilbertSpace<_T>& _hil, const size_t _N, 
	const double _gamma, const double _g0, 
	const v_1d<double>& _a, const v_1d<double>& _h, const v_1d<double>& _xi)
	: Hamiltonian<_T, 2>(_hil), Nin_(_N), gamma_(_gamma), g0_(_g0), a_(_a), h_(_h), xi_(_xi)
{
	// we will keep the particles from the dot at the beginning of the vectors
	// remember that this Hamiltonian is zero-dimensional, so we don't need to worry about the order of the particles
	
	// check the validity of the input
	this->checkSizes();

	this->initializeParticles();
}

template<typename _T>
inline QSM<_T>::QSM(const Hilbert::HilbertSpace<_T>&& _hil, const size_t _N, 
	const double _gamma, const double _g0,
	const v_1d<double>& _a, const v_1d<double>& _h, const v_1d<double>& _xi)
	: Hamiltonian<_T, 2>(std::move(_hil)), Nin_(_N), gamma_(_gamma), g0_(_g0), a_(_a), h_(_h), xi_(_xi)
{
	// we will keep the particles from the dot at the beginning of the vectors
	// remember that this Hamiltonian is zero-dimensional, so we don't need to worry about the order of the particles

	// check the validity of the input
	this->checkSizes();

	this->initializeParticles();
}
// ##########################################################################################################################################

// ############################################################### I N I T S ################################################################

// ##########################################################################################################################################

template<typename _T>
inline void QSM<_T>::initializeParticles()
{
	// how many particles we have left
	this->Nout_		= this->Ns_ - this->Nin_;
	this->dimIn_	= ULLPOW(this->Nin_);
	this->dimOut_	= ULLPOW(this->Nout_);

	// initialize the random neighbors for the 'free' particles
	this->n_        = this->ran_.template randomInt<size_t, size_t, v_1d>((size_t)0, (size_t)this->Nin_, (size_t)this->Nout_);

	// initialize the random distances for the 'free' particles
	this->u_        = v_1d<double>(this->Nout_, 1);
	this->au_       = v_1d<double>(this->Nout_, 1);
	for (size_t i = 1; i < this->Nout_; ++i)
	{
		this->u_[i]	= this->ran_.template random<double>(i - this->xi_[i], i + this->xi_[i]);;
		this->au_[i]= std::pow(this->a_[i], this->u_[i]);
	}

	// generate the random Hamiltonian for the dot
	if(typeid(_T) == typeid(double))
		this->Hdot_ = this->ran_.template GOE<_T>(this->dimIn_);
	else
		this->Hdot_ = this->ran_.template CUE<_T>(this->dimIn_);
	// normalize
	this->Hdot_		= this->gamma_ / sqrt(this->dimIn_ + 1) * this->Hdot_;
}

// ##########################################################################################################################################

// ############################################################### I N F O S ################################################################

// ##########################################################################################################################################

template<typename _T>
inline std::string QSM<_T>::info(const strVec & skip, std::string sep, int prec) const
{
	bool _different_alpha	= !std::equal(this->a_.begin() + 1, this->a_.end(), this->a_.begin());
	bool _different_h		= !std::equal(this->h_.begin() + 1, this->h_.end(), this->h_.begin());
	bool _different_xi		= !std::equal(this->xi_.begin() + 1, this->xi_.end(), this->xi_.begin());

	std::string name		= sep + "qsm,Ns=" + STR(this->Ns);
	name += "," +	VEQV(N, Nin_);
	name +=	"," +	VEQV(gamm, gamma_);
	name += "," +	VEQV(g0, g0_);
	name += "," +	(_different_alpha	? "alpha=r"	:	VEQVP(alpha, a_[0], 3));
	name += "," +	(_different_h		? "h=r"		:	VEQVP(h, h_[0], 3));
	name += "," +	(_different_xi		? "xi=r"	:	VEQVP(xi, xi_[0], 3));

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
inline void QSM<_T>::hamiltonian()
{
	if (this->Nh == 0)
	{
		LOGINFOG("Empty Hilbert, not building anything.", LOG_TYPES::INFO, 1);
		return;
	}
	this->init();

	// go through all the elements of the Hilbert space
	for (u64 k = 0; k < this->Nh; ++k) 
	{
		u64 _elem	= this->hilbertSpace.getMapping(k);

		// go through all the particles outside the dot
		for (size_t _part = this->Nin_; _part <= (this->Ns_ - 1); ++_part)
			this->locEnergy(k, _elem, _part);
	}

	// add the random Hamiltonian of the dot. This is treated as an operator acting only on the left 
	// side of the tensor product and the identity on the right side (A^A \otimes I^B)
	// (THIRD TERM)
	this->H_ += arma::kron(this->Hdot_, EYE(this->dimOut_));
}

template<typename _T>
inline void QSM<_T>::locEnergy(u64 _elemId, u64 _elem, uint _site)
{
	// the particle is in the dot
	if(_site < this->Nin_)
		return;

	size_t _partIdx					= _site - this->Nin_;

	// check the spin of the particle
	auto [_idx, _val]				= Operators::sigma_x(_elem, this->Ns_, {_site});

	// apply magnetic field to the particle (THIRD TERM)
	this->setHElem(_idx, _elemId, this->h_[_partIdx] * _val);

	// apply the spin-flip interaction with the dot
	int _n							= this->n_[_partIdx];
	auto [_idx1, Sx_n]				= Operators::sigma_x(_elem, this->Ns_, {_n});
	auto [_idx2, Sx_j]				= Operators::sigma_x(_idx1, this->Ns_, {_site});
	// apply the coupling to the particle (SECOND TERM)
	this->setHElem(_idx2, _elemId, this->g0_ * this->au_[_partIdx] * Sx_j * Sx_n);
}

// ##########################################################################################################################################

#endif