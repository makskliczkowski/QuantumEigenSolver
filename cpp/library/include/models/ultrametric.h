#pragma once
/***********************************
* Is an instance of the UM model.
* Derives from a general Hamiltonian.
* For more information see the UM model.
************************************/

#ifndef HAMIL_H
#	include "../hamil.h"
#endif

#ifndef ULTRAMETRIC_H
#	define ULTRAMETRIC_H
#	define ULTRAMETRIC_USE_DIFFERENT_BLOCKS

template<typename _T>
class Ultrametric : public Hamiltonian<_T, 2>
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
	v_1d<double> au_;				// "distances" of particles within a dot

	double g0_ 		=	1.0;		// coupling strength between the dot and outside world (J)
	v_1d<double> a_;				// coupling between the dot and outside world

	void checkSizes();
	void initializeParticles();

public:
	void setRandomHDot()										{ this->Hdot_ = this->ran_.template GOE<_T>(this->dimIn_); this->Hdot_ = 1.0 / sqrt(this->dimIn_ + 1) * this->Hdot_; };
	void randomize(double _a, double _s, const strVec& _which)	override final;
public:
	~Ultrametric() override;
	Ultrametric(const size_t _Nall);
	Ultrametric(const size_t _Nall,
		const size_t _N,
		const double _g0,
		const v_1d<double>& _a);
	Ultrametric(const Hilbert::HilbertSpace<_T>& _hil,
		const size_t _N,
		const double _g0,
		const v_1d<double>& _a);
	Ultrametric(Hilbert::HilbertSpace<_T>&& _hil,
		const size_t _N,
		const double _g0,
		const v_1d<double>& _a);

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

	double get_alpha()							const				{ return this->a_[0];	};
	double get_g0()								const				{ return this->g0_;		};
};

// ##########################################################################################################################################

// ############################################################## C O N S T R ###############################################################

// ##########################################################################################################################################

/*
* @brief Destructor of the QSM model.
*/
template<typename _T>
inline Ultrametric<_T>::~Ultrametric()
{
	DESTRUCTOR_CALL;
}

// ##########################################################################################################################################

/*
* @brief Checks the sizes of the vectors of the Ultrametric model. It checks if the number of particles in the dot is larger than the number of particles in the system.
* It also checks if the sizes of the coupling vector 'a', the random box distribution vector 'xi' and the magnetic field vector 'h' are equal to the number of particles outside the dot.
* @throws std::invalid_argument if the number of particles in the dot is larger than the number of particles in the system.
* @throws std::invalid_argument if the size of the coupling vector 'a' is not equal to the number of particles outside the dot.
*/
template<typename _T>
inline void Ultrametric<_T>::checkSizes()
{
	if(this->Nin_ >= this->Ns_)
		throw(std::invalid_argument("Ultrametric: the number of particles in the dot is larger than the number of particles in the system."));

	this->Nout_ = this->Ns_ - this->Nin_;

	if(this->a_.size() != this->Nout_)	
		throw(std::invalid_argument("Ultrametric: the size of the coupling vector 'a' is not equal to the number of particles outside the dot."));
}

// ##########################################################################################################################################

/*
* @brief Constructor of the QSM model. It takes the number of particles in the system and initializes the Hamiltonian.
*/
template<typename _T>
inline Ultrametric<_T>::Ultrametric(const size_t _Nall)
	: Hamiltonian<_T, 2>(_Nall, false)
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
inline Ultrametric<_T>::Ultrametric(const size_t _Nall, const size_t _N, const double _g0, const v_1d<double>& _a)
	: Ultrametric(_Nall)
{	
	this->Nin_		= _N;
	this->g0_		= _g0;
	this->a_		= _a;

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
* @param _g0: the coupling strength between the dot and outside world.
* @param _a: the coupling between the dot and outside world.
* @note The Hilbert space is used to initialize the Hamiltonian.
* @note The number of particles within the dot is used to initialize the particles outside the dot.
* @note The parameters of the QSM model are used to initialize the Hamiltonian.
* @note The random Hamiltonian for the dot is initialized automatically.
* @note The random neighbors and distances for the particles outside the dot are initialized automatically
*/
template<typename _T>
inline Ultrametric<_T>::Ultrametric(const Hilbert::HilbertSpace<_T>& _hil, const size_t _N, const double _g0, 
	const v_1d<double>& _a)
	: Hamiltonian<_T, 2>(_hil, false), Nin_(_N), g0_(_g0), a_(_a)
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
* @param _g0: the coupling strength between the dot and outside world.
* @param _a: the coupling between the dot and outside world.
* @note The Hilbert space is used to initialize the Hamiltonian.
* @note The number of particles within the dot is used to initialize the particles outside the dot.
* @note The parameters of the QSM model are used to initialize the Hamiltonian.
* @note The random Hamiltonian for the dot is initialized automatically.
* @note The random neighbors and distances for the particles outside the dot are initialized automatically
*/
template<typename _T>
inline Ultrametric<_T>::Ultrametric(Hilbert::HilbertSpace<_T>&& _hil, const size_t _N, const double _g0,
	const v_1d<double>& _a)
	: Hamiltonian<_T, 2>(std::move(_hil), false), Nin_(_N), g0_(_g0), a_(_a)
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
inline void Ultrametric<_T>::randomize(double _around, double _str, const strVec& _which)
{
	this->setRandomHDot();
	//this->hamiltonian();
}

// ##########################################################################################################################################

/*
* @brief Initializes the particles outside the dot. It also initializes the random neighbors and distances for the particles outside the dot.
* @note The random neighbors are used to create the spin-flip interaction between the dot and the outside world.
*/
template<typename _T>
inline void Ultrametric<_T>::initializeParticles()
{
	// how many particles we have left
	this->Nout_		= this->Ns_ - this->Nin_;
	this->dimIn_	= ULLPOW(this->Nin_);
	this->dimOut_	= ULLPOW(this->Nout_);

	// initialize the random distances for the 'free' particles
	this->au_       = v_1d<double>(this->Nout_, 1);
	for (size_t i = 1; i <= this->Nout_; ++i)
		this->au_[i - 1]= std::pow(this->a_[i - 1], i);

	LOG_DEBUG(this->a_[0], "alpha=");
	LOG_DEBUG(this->au_, "alpha^u=");

	// generate the random Hamiltonian for the dot
	if(typeid(_T) == typeid(double))
		this->Hdot_ = this->ran_.template GOE<_T>(this->dimIn_);
	else
		this->Hdot_ = this->ran_.template CUE<_T>(this->dimIn_);
	// normalize
	this->Hdot_		= 1.0 / std::sqrt(this->dimIn_ + 1) * this->Hdot_;
}

// ##########################################################################################################################################

// ############################################################### I N F O S ################################################################

// ##########################################################################################################################################

/*
* @brief Returns the information about the Ultrametric model.
* @param skip: the parameters to skip.
* @param sep: the separator between the parameters.
* @param prec: the precision of the output.
* @returns the information about the Ultrametric model.
*/
template<typename _T>
inline std::string Ultrametric<_T>::info(const strVec & skip, std::string sep, int prec) const
{
	bool _different_alpha	= !std::equal(this->a_.begin() + 1, this->a_.end(), this->a_.begin());

	std::string name		= sep + "ultrametric,Ns=" + STR(this->Ns);
	name += sep +	VEQV(N, Nin_);
	name += sep +	VEQV(g0, g0_);
	if(std::find(skip.begin(), skip.end(), "alpha") == skip.end())
		name += sep +	(_different_alpha	? "alpha=r"	:	VEQVP(alpha, a_[0], 3));
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
inline void Ultrametric<_T>::hamiltonian()
{
	if (this->Nh == 0)
	{
		LOGINFOG("Empty Hilbert, not building anything.", LOG_TYPES::INFO, 1);
		return;
	}
	this->init();

	// go through all the elements of the Hilbert space
#ifdef ULTRAMETRIC_USE_DIFFERENT_BLOCKS
	for (size_t k = 0; k <= this->Nout_; ++k) {
#else
	for (size_t k = 1; k <= this->Nout_; ++k) {
#endif
		// set the dimensions
		auto _dimrest	=	ULLPOW((this->Nout_ - k));	// L - k -> number of diagonal blocks of size 2^{N+k}
		auto _dim		=	ULLPOW((this->Nin_ + k));	// N + k -> size of the diagonal blocks
		auto _mult		=	k == 0 ? 1.0 : (this->g0_ * this->au_[k - 1]  / std::sqrt(_dim + 1));

#ifdef ULTRAMETRIC_USE_DIFFERENT_BLOCKS
		arma::Mat<_T> Hk(this->Nh, this->Nh, arma::fill::zeros);
		// create various blocks of the Hamiltonian
		for (int i = 0; i < _dimrest; ++i)
		{
			Hk.submat(i * _dim, i * _dim, (i + 1) * _dim - 1, (i + 1) * _dim - 1) += this->ran_.template GOE<_T>(_dim);
			//this->H_.print("H=" + VEQ(k) + ":" + VEQ(i));
		}
		this->H_ += _mult * Hk;
#else
		// repeat the blocks multiple times (sample the diagonal blocks independently)
		this->H_ += _mult * arma::kron(this->ran_.template GOE<_T>(_dim), EYE(_dimrest));
#endif
	}
	// add the random Hamiltonian of the dot. This is treated as an operator acting only on the left 
	// side of the tensor product and the identity on the right side (A^A \otimes I^B)
	// (THIRD TERM)
#ifndef ULTRAMETRIC_USE_DIFFERENT_BLOCKS
	this->H_ += arma::kron(this->Hdot_, EYE(this->dimOut_));
#endif
	//saveAlgebraic("C:/University/PHD/CODES/VQMC/QSolver/cpp/library/", "H.h5", arma::Mat<_T>(H_), "H", false);
}

// ##########################################################################################################################################

/*
*/
template<typename _T>
inline void Ultrametric<_T>::locEnergy(u64 _elemId, u64 _elem, uint _site)
{

}

// ##########################################################################################################################################

#endif