#pragma once
/***************************************
* Defines the quadratic lattice Hamiltonian
* for the translationally invariant FF.
* JULY 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#ifndef HAMIL_QUADRATIC_H
#include "../../hamilQ.h"
#endif

#ifndef FF_M_H
#define FF_M_H

// //////////////////////////////////////////////////////////////////////////

/*
* @brief The choice of momentum for the translationally invariant FF lays in the regions of -\Pi, -\Pi/2 and \Pi/2, \Pi.
* We want to move the momentum from the rest of the space to this region.
* @param k momentum
* @param L size
* @returns moved momentum
*/
template <typename _T>
inline _T move_momentum_cosine(_T k, uint32_t L)
{
	_T out = k;
	if (k > -PIHALF && k <= 0)
		out = -PI + std::abs(k);
	else if (k > 0 && k <= PIHALF)
		out = PI - std::abs(k);
	return out;
}

template <>
inline uint move_momentum_cosine(uint k, uint32_t L)
{	
	double out = k * TWOPI / L;
	if (out >= 0 && out < PIHALF)
		out = PI - std::abs(out);
	else if(out > 3 * PIHALF)
		out = PI + (TWOPI - std::abs(out));
	return std::round(out * L / TWOPI);
}

// //////////////////////////////////////////////////////////////////////////

template<typename _T>
class FreeFermions : public QuadraticHamiltonian<_T>
{
protected:
	DISORDER_EQUIV(double, t_);
public:
	~FreeFermions()		= default;
	FreeFermions(std::shared_ptr<Lattice> _lat, double _t = 1.0, double _t0 = 0.0, double _constant = 0.0)
		: QuadraticHamiltonian<_T>(_lat, _constant, true), t_(_t), t_0(_t0)
	{
		this->type_			=		MY_MODELS_Q::FREE_FERMIONS_M;
		this->info_			=		this->info();
		this->dt_			=		this->ran_.template createRanVec<double>(this->Ns, this->t_0);
		this->getSPEnMat();
		LOGINFO("I am Free Fermions model: ", LOG_TYPES::CHOICE, 2);
	};
	
	// --------------- H A M I L T O N I A N ---------------

	void hamiltonian() override
	{
		this->init();
		for (int i = 0; i < this->Ns; i++)
		{
			uint NUM_OF_NN	=	(uint)this->lat_->get_nn_ForwardNum(i);
			// -------------- CHECK NN ---------------
			for (uint nn = 0; nn < NUM_OF_NN; nn++) {
				uint N_NUMBER = this->lat_->get_nn_ForwardNum(i, nn);
				if (int nei = this->lat_->get_nn(i, N_NUMBER); nei >= 0)
					this->setHElem(i, PARAM_W_DISORDER(t_, i), nei);
			}
		}
	}
	// ------------------ M A N Y   B O D Y ------------------

	void getManyBodyEnergiesZero(uint N, 
								 v_1d<double>& manyBodySpectrum, 
								 v_1d<arma::uvec>& manyBodyOrbitals, 
								 int _num = -1)						override;

	// ----------------------- I N F O -----------------------
	
	std::string info(const v_1d<std::string>& skip = {}, std::string sep = "_", int prec = 2) const override
	{
		auto Ns = this->lat_->get_Ns();
		auto BC = this->lat_->get_BC();
		std::string name = sep + "FF,Ns=" + STR(Ns);
		PARAMS_S_DISORDER(t_, name);
		name += ",BC=" + SSTR(getSTR_BoundaryConditions(BC));
		return this->QuadraticHamiltonian<_T>::info(name, skip, sep);
	}
	void updateInfo() override final { this->info_ = this->info(); };

	// override 
	auto getTransMat()	-> arma::Mat<_T>							override;
	auto getSPEnMat()	-> arma::Col<double>						override;
};

// ###################################################################################

/*
* @brief Returns the transformation matrix for free fermions, as we know it by hand...
*/
template<typename _T>
inline arma::Mat<_T> FreeFermions<_T>::getTransMat()
{
	if (!this->eigVec_.empty())
		return this->eigVec_;
	this->eigVec_.zeros(this->Ns, this->Ns);
	for (int k = 0; k < this->Ns; k++)
		for (int i = 0; i < this->Ns; i++)
			this->eigVec_(k, i) = algebra::cast<_T>(std::exp(I * double(k * i * TWOPI / this->Ns)) / std::sqrt(this->Ns));
	return this->eigVec_;
}

// ###################################################################################

/*
* @brief Returns the transformation matrix for free fermions, as we know it by hand...
*/
template<typename _T>
inline arma::Col<double> FreeFermions<_T>::getSPEnMat()
{
	if (!this->eigVal_.empty())
		return this->eigVal_;
	this->eigVal_.zeros(this->Ns);
	for (int k = 0; k < this->Ns; k++)
		this->eigVal_(k) = 2.0 * PARAM_W_DISORDER(t_, k) * std::cos(double(k * TWOPI / this->Ns));
	return this->eigVal_;
}

// ###################################################################################

/*
* @brief Create combination of quasiparticle orbitals to obtain the many body product states...
* @warning Using zero energy (based on the knowledge of model). This is defined on the degenerate manifold of zero energies.
* @param N number of particles
* @param _num number of combinations
* !TODO implement more than 1D
*/
template<typename _T>
inline void FreeFermions<_T>::getManyBodyEnergiesZero(uint N, v_1d<double>& manyBodySpectrum, v_1d<arma::uvec>& manyBodyOrbitals, int _num)
{
	LOGINFO("Looking for degenerate zero energy many body states only!", LOG_TYPES::CHOICE, 2);
	if (this->Ns % 2 != 0)
		throw std::runtime_error(std::string("Method is not implemented for uneven system sizes..."));

	// find the highest divider to check how many zero energy states we have 
	// the divider will stand for the number of particle number to take out at the beginning
	// we will take them out by the usage of de Movre's theorem. Taking into account the periodicity (which is the divider), 
	// we immidietely can take out the zero energy states. The rest of them will be handled by choosing the momenta that take 
	// into account the symmetry of cosine function.
	uint divider = 1;
	// if the number of particles is even, the number of options is huge, otherwise we need to look after them
	if (N % 2 != 0)
	{
		for (uint i = 3; i < uint(this->Ns / 2) + 1; i += 2)
		{
			if (this->Ns % i == 0)
			{
				divider = i;
				break;
			}
		}
	}
	else
		divider = 0;
	LOGINFO(VEQ(divider), LOG_TYPES::CHOICE, 2);

	// create orbitals (take in the regions {-\Pi, -\Pi/2}, {0, \Pi/2} or equivalently {\pi/2, 3/2\pi})
	v_1d<uint> orbitals;
	for (int i = 0; i < this->Ns; ++i)
	{
		double momentum = i * 2 * PI / this->Ns;
		// check which indices we can take
		if((momentum > PIHALF && momentum < 3 * PI / 2))
			orbitals.push_back(i);
	}
	v_2d<uint> orbitals_cut;
	v_2d<uint> orbitals_taken;
	// resize, those will save the eigenspectrum of many body states
	manyBodySpectrum.clear();
	manyBodySpectrum.resize(_num);
	manyBodyOrbitals.clear();
	manyBodyOrbitals.resize(_num);

	// take out the states if we need to take them out (number of particles is uneven)
	if (divider != 0)
	{
		// choose the step
		uint _step = this->Ns / divider;

		// starting from this position, we will take the periodic states
		for (uint i = 0; i < _step; ++i)
		{
			LOGINFO("Start: " + STR(i) + ", " + VEQ(_step), LOG_TYPES::CHOICE, 3);
			v_1d<uint> momenta_movre;
			// append movre momenta
			for (uint j = 0; j < divider; ++j)
				momenta_movre.push_back(i + _step * j);
			// append the taken orbitals
			orbitals_taken.push_back(momenta_movre);
			// transform the orbitals to see which one is able to take
			v_1d<uint> orbitals_to_take_from;
			for (const auto& momentum_outer : orbitals)
			{
				bool isInMovre = false;
				// check if a given index is inside the movre momenta vector
				for (const auto& momentum : momenta_movre)
				{
					if (momentum_outer == move_momentum_cosine((uint)momentum, this->Ns))
					{
						isInMovre = true;
						break;
					}
				}

				// if we can still take the orbitals
				if (!isInMovre)
					orbitals_to_take_from.push_back(momentum_outer);
			}
			orbitals_cut.push_back(orbitals_to_take_from);
		}
	}
	else
	{
		orbitals_cut.push_back(orbitals);
		// orbitals_taken will be empty in this case
	}

	// get through combinations!
#pragma omp parallel for
	for (int i = 0; i < _num; ++i)
	{
		// which orbitals_cut to take
		uint startIdx			= this->ran_.template randomInt<uint>(0, orbitals_cut.size());
		// how many particles to choose from are left
		uint Nin				= orbitals_taken.empty() ? uint(N / 2) : uint((N - orbitals_taken[startIdx].size()) / 2);
		// create combination out of the particles that are left
		auto _combinationTmp	= this->ran_.choice(orbitals_cut[startIdx], Nin);
		auto _combination		= _combinationTmp;

		// push the rest...(the ones that will remove the energy that we included)
		for (const auto& _comb : _combinationTmp)
			_combination.push_back((_comb + int(this->Ns / 2)) % this->Ns);

		// append the starting momenta
		if(!orbitals_taken.empty())
			for (auto j = 0; j < orbitals_taken[startIdx].size(); ++j)
				_combination.push_back(orbitals_taken[startIdx][j]);

		// transform to uvec
		arma::uvec _combinationV(N);
		for (int j = 0; j < _combination.size(); j++)
			_combinationV(j) = _combination[j];

		// append
		manyBodyOrbitals[i] = _combinationV;

		// get energy
		double _manyBodyEn = this->getManyBodyEnergy(_combination);
		manyBodySpectrum[i] = _manyBodyEn;
	}
}

#endif // !SYK2_M_H
