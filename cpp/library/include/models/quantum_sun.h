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

#ifndef QSM_H
#	define QSM_H

// definitions
#define QSM_CHECK_HS_NORM 1

#include "../quantities/statistics.h"

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

	// ############################################ Getters ############################################

	double getMagnetic(int _idx)								const	{ return this->h_[_idx]; };

	// ############################################ Setters ############################################
	void setMagnetic(const v_1d<double>& _h)					{ this->h_ = _h; };
	void setMagnetic(int _idx, double _h)						{ this->h_[_idx] = _h; };
	void setRandomXi(double _around, double _strength)			{ this->xi_ = this->ran_.template rvector<v_1d<double>>(this->Nout_, _strength, _around); };
	void setRandomAlpha(double _around, double _strength)		{ this->a_ = this->ran_.template rvector<v_1d<double>>(this->Nout_, _strength, _around); };
	void setRandomMagnetic(double _around, double _strength)	{ this->h_ = this->ran_.template rvector<v_1d<double>>(this->Nout_, _strength, _around); };
	void setRandomHDot()										{ this->Hdot_ = this->ran_.template GOE<_T>(this->dimIn_); this->Hdot_ = this->gamma_ / sqrt(this->dimIn_ + 1) * this->Hdot_; };
	void randomize(double _a, double _s, const strVec& _which)	override final;
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
	QSM(Hilbert::HilbertSpace<_T>&& _hil,
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

	static double get_thouless_freq_est(double _alpha, double g_0, double _L)	
	{ 
		double _freq = 0.0;
		
		if (g_0 == 0.5)
		{
			if (_alpha > 0.95)
			{
				_freq = 0.39 * std::exp(-4.05 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else if (_alpha > 0.88)
			{
				_freq = 0.83 * std::exp(-2.05 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else if (_alpha > 0.8)
			{
				_freq = 2.2 * std::exp(-1.65 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else if (_alpha > 0.77)
			{
				_freq = 2.55 * std::exp(-1.13 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else if (_alpha > 0.74)
			{
				_freq = 4.5 * std::exp(-1.2 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else
			{
				_freq = 15 * std::exp(-1.0 * std::log(1.0 / _alpha / _alpha) * _L);
			}	 
		}
		else if (g_0 == 1.0)
		{
			if (_alpha > 0.95)
			{
				_freq = 0.39 * std::exp(-2.05 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else if (_alpha > 0.88)
			{
				_freq = 0.48 * std::exp(-1.18 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else if (_alpha > 0.8)
			{
				_freq = 2.55 * std::exp(-1.14 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else if (_alpha > 0.77)
			{
				_freq = 4.55 * std::exp(-1.06 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else if (_alpha > 0.74)
			{
				_freq = 7.5 * std::exp(-1.2 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else if (_alpha > 0.7)
			{
				_freq = 15 * std::exp(-1.0 * std::log(1.0 / _alpha / _alpha) * _L);
			}
			else
			{
				_freq = 20 * std::exp(-1.0 * std::log(1.0 / _alpha / _alpha) * _L);
			}
		}
		else
		{
			_freq = 2 * std::exp(-1.0 * std::log(1.0 / _alpha / _alpha) * _L);
		}

		return _freq;
	};
	double get_alpha()							const				{ return this->a_[0];	};
	double get_g0()								const				{ return this->g0_;		};
	double get_mobility_edge(double _E);
};

// ##########################################################################################################################################

// ############################################################## C O N S T R ###############################################################

// ##########################################################################################################################################

/*
* @brief Destructor of the QSM model.
*/
template<typename _T>
inline QSM<_T>::~QSM()
{
	DESTRUCTOR_CALL;
}

// ##########################################################################################################################################

/*
* @brief Checks the sizes of the vectors of the QSM model. It checks if the number of particles in the dot is larger than the number of particles in the system.
* It also checks if the sizes of the coupling vector 'a', the random box distribution vector 'xi' and the magnetic field vector 'h' are equal to the number of particles outside the dot.
* @throws std::invalid_argument if the number of particles in the dot is larger than the number of particles in the system.
* @throws std::invalid_argument if the size of the coupling vector 'a' is not equal to the number of particles outside the dot.
* @throws std::invalid_argument if the size of the random box distribution vector 'xi' is not equal to the number of particles outside the dot.
* @throws std::invalid_argument if the size of the magnetic field vector 'h' is not equal to the number of particles outside the dot.
*/
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

// ##########################################################################################################################################

/*
* @brief Constructor of the QSM model. It takes the number of particles in the system and initializes the Hamiltonian.
*/
template<typename _T>
inline QSM<_T>::QSM(const size_t _Nall)
	: Hamiltonian<_T, 2>(_Nall)
{
	CONSTRUCTOR_CALL;
	//change info
	this->info_ = this->info();
	this->updateInfo();
}

// ##########################################################################################################################################

/*
* @brief Constructor of the QSM model. It takes the number of particles in the system and initializes the Hamiltonian.
*/
template<typename _T>
inline QSM<_T>::QSM(const size_t _Nall, const size_t _N, 
	const double _gamma, const double _g0, 
	const v_1d<double>& _a, const v_1d<double>& _h, const v_1d<double>& _xi)
	: QSM(_Nall)
{	
	this->Nin_		= _N;
	this->gamma_	= _gamma;
	this->g0_		= _g0;
	this->a_		= _a;
	this->h_		= _h;
	this->xi_		= _xi;

	// we will keep the particles from the dot at the beginning of the vectors
	// remember that this Hamiltonian is zero-dimensional, so we don't need to worry about the order of the particles

	// check the validity of the input
	this->checkSizes();

	this->initializeParticles();
}

// ##########################################################################################################################################

/*
* @brief Constructor of the QSM model. It takes the Hilbert space and the parameters of the QSM model.
* @param _hil: the Hilbert space of the QSM model.
* @param _N: the number of particles within the dot.
* @param _gamma: the Hilbert-Schmidt norm of the coupling operator normalizer.
* @param _g0: the coupling strength between the dot and outside world.
* @param _a: the coupling between the dot and outside world.
* @param _h: the internal dot magnetic field.
* @param _xi: the random box distribution couplings.
* @note The Hilbert space is used to initialize the Hamiltonian.
* @note The number of particles within the dot is used to initialize the particles outside the dot.
* @note The parameters of the QSM model are used to initialize the Hamiltonian.
* @note The random Hamiltonian for the dot is initialized automatically.
* @note The random neighbors and distances for the particles outside the dot are initialized automatically
*/
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

	//change info
	this->info_ = this->info();
	this->updateInfo();
}

// ##########################################################################################################################################

/*
* @brief Constructor of the QSM model. It takes the Hilbert space and moves it along with
* the parameters of the QSM model.
* @param _hil: the Hilbert space of the QSM model.
* @param _N: the number of particles within the dot.
* @param _gamma: the Hilbert-Schmidt norm of the coupling operator normalizer.
* @param _g0: the coupling strength between the dot and outside world.
* @param _a: the coupling between the dot and outside world.
* @param _h: the internal dot magnetic field.
* @param _xi: the random box distribution couplings.
* @note The Hilbert space is used to initialize the Hamiltonian.
* @note The number of particles within the dot is used to initialize the particles outside the dot.
* @note The parameters of the QSM model are used to initialize the Hamiltonian.
* @note The random Hamiltonian for the dot is initialized automatically.
* @note The random neighbors and distances for the particles outside the dot are initialized automatically
*/
template<typename _T>
inline QSM<_T>::QSM(Hilbert::HilbertSpace<_T>&& _hil, const size_t _N, 
	const double _gamma, const double _g0,
	const v_1d<double>& _a, const v_1d<double>& _h, const v_1d<double>& _xi)
	: Hamiltonian<_T, 2>(std::move(_hil)), Nin_(_N), gamma_(_gamma), g0_(_g0), a_(_a), h_(_h), xi_(_xi)
{
	// we will keep the particles from the dot at the beginning of the vectors
	// remember that this Hamiltonian is zero-dimensional, so we don't need to worry about the order of the particles

	// check the validity of the input
	this->checkSizes();

	this->initializeParticles();

	//change info
	this->info_ = this->info();
	this->updateInfo();
}

// ##########################################################################################################################################

// ############################################################### M E T H S ################################################################

// ##########################################################################################################################################

/*
* @brief Returns the mobility edge of the QSM model. The mobility edge is calculated using the standard deviation of the eigenvalues of the Hamiltonian.
* It is calculated as in: Konrad Pawlik, Piotr Sierant, Lev Vidmar, Jakub Zakrzewski (2023)
*/
template<typename _T>
inline double QSM<_T>::get_mobility_edge(double _E)
{
	if (this->stdEn == 0.0)
		this->stdEn = arma::stddev(this->eigVal_);
	double _std = this->stdEn / std::sqrt(this->Ns_);
	double _eps = (_E - this->eigVal_(0)) / (this->eigVal_(this->Nh - 1) - this->eigVal_(0));
	double _bwd = (this->eigVal_(this->Nh_ - 1) - this->eigVal_(0));
	_bwd		= _bwd / (double)this->Ns_;
	return std::exp(_bwd * _bwd * (_eps - 0.5) * (_eps - 0.5) / _std / _std / 4.0) / std::sqrt(2.0);
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
inline void QSM<_T>::randomize(double _around, double _str, const strVec& _which)
{
	if(std::find(_which.begin(), _which.end(), "alpha") != _which.end())
		this->setRandomAlpha(_around, _str);
	if(std::find(_which.begin(), _which.end(), "h") != _which.end())
		this->setRandomMagnetic(_around, _str);
	if(std::find(_which.begin(), _which.end(), "xi") != _which.end())
		this->setRandomXi(_around, _str);
	// initialize the random Hamiltonian for the dot
	// this includes initializing the random neighbors and distances for the particles outside the dot
	this->initializeParticles();
}

// ##########################################################################################################################################

/*
* @brief Initializes the particles outside the dot. It also initializes the random neighbors and distances for the particles outside the dot.
* @note The random neighbors are used to create the spin-flip interaction between the dot and the outside world.
*/
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
	LOG_DEBUG(this->a_[0], "alpha=");
	LOG_DEBUG(this->u_, "u=");
	LOG_DEBUG(this->au_, "alpha^u=");

	// generate the random Hamiltonian for the dot
	if(typeid(_T) == typeid(double))
		this->Hdot_ = this->ran_.template GOE<_T>(this->dimIn_);
	else
		this->Hdot_ = this->ran_.template CUE<_T>(this->dimIn_);
	// normalize
	this->Hdot_		= this->gamma_ / std::sqrt(this->dimIn_ + 1) * this->Hdot_;
#if QSM_CHECK_HS_NORM
	// check the Hilbert-Schmidt norm
	_T _norm = SystemProperties::hilber_schmidt_norm(this->Hdot_);
	LOGINFO("QSM_DOT_NORM: " + VEQP(_norm, 5), LOG_TYPES::CHOICE, 3);
	this->Hdot_		/= std::sqrt(_norm);
	_norm = SystemProperties::hilber_schmidt_norm(this->Hdot_);
	LOGINFO("QSM_DOT_NORM_AFTER: " + VEQP(_norm, 5), LOG_TYPES::CHOICE, 4);
#endif
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
inline std::string QSM<_T>::info(const strVec & skip, std::string sep, int prec) const
{
	bool _different_alpha	= !std::equal(this->a_.begin() + 1, this->a_.end(), this->a_.begin());
	bool _different_h		= !std::equal(this->h_.begin() + 1, this->h_.end(), this->h_.begin());
	bool _different_xi		= !std::equal(this->xi_.begin() + 1, this->xi_.end(), this->xi_.begin());

	std::string name		= sep + "qsm,Ns=" + STR(this->Ns);
	name += sep +	VEQV(N, Nin_);
	name +=	sep +	VEQV(gamm, gamma_);
	name += sep +	VEQV(g0, g0_);
	if(std::find(skip.begin(), skip.end(), "alpha") == skip.end())
		name += sep +	(_different_alpha	? "alpha=r"	:	VEQVP(alpha, a_[0], 3));
	if(std::find(skip.begin(), skip.end(), "h") == skip.end())
		name += sep +	(_different_h		? "h=r"		:	VEQVP(h, h_[0], 3));
	if(std::find(skip.begin(), skip.end(), "xi") == skip.end())
		name += sep +	(_different_xi		? "xi=r"	:	VEQVP(xi, xi_[0], 3));

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

	// print disorder 
	for(int i = 0; i < this->Nout_; ++i)
	{
		LOGINFO("QSM: i=" + STR(i) + " -> h=" + STRP(this->h_[i], 5) + ", a^u=" + STRP(this->au_[i], 5), LOG_TYPES::INFO, 2);
	}

#ifdef _DEBUG
	std::cout << this->H_.getSparse() << std::endl;
#endif

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

//#ifdef _DEBUG
//	std::cout << this->H_.getSparse() << std::endl;
//#endif

#ifdef _DEBUG
	{
		Operators::Operator<double> sz	= Operators::SpinOperators::sig_z(this->Ns_, { this->Ns_ - 1 });
		arma::SpMat<double> _Min		= sz.template generateMat<double, typename arma::SpMat>(this->Nh_);
		auto _traceout					= arma::trace(_Min * this->H_.getSparse());
		auto _lastH						= this->h_[this->Nout_ - 1];

		std::ofstream _file;
		openFile(_file, "QSM_TRACE.txt", std::ios::app);
		_file << this->Ns_ << "\t" << STRP(_lastH, 5) << "\t" << STRP(_traceout / double(this->Nh_), 5) << std::endl;
		_file.close();
	};
#endif 
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
inline void QSM<_T>::locEnergy(u64 _elemId, u64 _elem, uint _site)
{
	// the particle is in the dot
	if(_site < this->Nin_)
		return;

	size_t _partIdx					= _site - this->Nin_;

	// check the spin of the particle
	auto [_idx, _val]				= Operators::SpinOperators::sig_z(_elem, this->Ns_, { _site });

	// apply magnetic field to the particle (THIRD TERM)
	this->setHElem(_elemId, this->h_[_partIdx] * _val, _idx);

	// apply the spin-flip interaction with the dot
	uint _n							= this->n_[_partIdx];
	auto [_idx1, Sx_n]				= Operators::SpinOperators::sig_x(_elem, this->Ns_, {_n});
	auto [_idx2, Sx_j]				= Operators::SpinOperators::sig_x(_idx1, this->Ns_, {_site});

	// apply the coupling to the particle (SECOND TERM)
	this->setHElem(_elemId, this->g0_ * this->au_[_partIdx] * Sx_j * Sx_n, _idx2);
}

// ##########################################################################################################################################

#endif